import xarray as xr
import tensorflow as tf
import glob
import os
import numpy as np
from tqdm import tqdm

import warnings

class Gridded:
    ''' Clase básica introductoria que nos permite manejar y setear las primeras variables relacionadas a la ruta de los archivos que estamos leyendo. 
        Además, nos permite chequear la existencia de atributos propios de la data y atributos propios de la clase instanciada.
    '''
    
    def __init__(self, fName='*', path='./'):
        self.fName = fName #Nombres de los archivos a importar [str]
        self.path = path #ruta (path) de estos archivos. [str]

    def __value__(self, val, vals):
        '''Método que chequea la existencia de un atributo relacionado a "Data Variable"
        inp:
            val: [str] Atributo de "Data Variable" que se quiere chequear
            vals: [List] Lista con todos los atributos de "Data Variable"
        '''

        if val in vals:
            return val
        else:
            warnings.warn(val + " is not a possible value, using: " + vals[0])
            return vals[0]

    def getMyAttributes(self):
        '''Retorna los atributos propios de la clase instanciada dado las siguientes condiciones:
        no comienzan con '__', no son calleables ( '__call__()') y no son propios de una clase (genérica) como tal. 
        '''    
        return [a for a in dir(self) if
                not a.startswith('__') and not callable(getattr(self, a)) and not a in dir(Gridded)]

    def __str__(self):
        '''Método especial para printear cada vez que instanciamos nuestra clase.'''

        txt = "Class: gridded\n"
        for att in self.getMyAttributes():
            val = getattr(self, att)
            if type(val) is list:
                txt = txt + "  " + att + ":\n"
                for v in val:
                    txt = txt + "    " + str( v ) + "\n"
            else:
                txt = txt + "  " + att + ": " + str(getattr(self, att)) + "\n"
        return (txt)

    def __repr__(self):
        '''Método que convierte el objeto en una cadena'''       
        return self.__str__()

class GFSVar:
    '''Segunda clase básica que nos permite setear atributos importantes para extraer los DataArrays correctos del Dataset original. Su principal función será guardar la triada:
    (cfVarName: atributo de "Data Variable" para extraer el DataArray 
    typeOfLevel:  atributo typeOfLevel propio del DataArray
     long_name: nombre extendido del atributo de "Data Variable" definido anteriormente)
    '''

    #Atributos de clase propios de los archivos .nc
    typeOfLevels = [ 'surface','heightAboveGround']

    def __init__(self, cfVarName, typeOfLevel, long_name=''):
        self.cfVarName = cfVarName  #[str] Atributo de "Data Variable" a utilizar. (Ej: 't2m' o 'Band1')
        self.typeOfLevel = typeOfLevel #[str] Atributo typeOfLevel de nuestro DataArray
        self.value = None
        if long_name != '': 
            self.long_name = long_name #[str] Atributo de nuestro DataArray, representa el nombre extendido de cfVarName

    def __str__(self):
        '''método especial de printeo para cuando instanciemos nuestra clase.
        '''
        return self.cfVarName + ", " + self.typeOfLevel + ", " + self.long_name

    def __repr__(self):
        ''' método especial que retorna el objeto como string
        '''
        return self.__str__()

class NCFile(Gridded):
    ''' Clase principal para el preprocesamiento de los inputs. Está encargada de la importación, estandarización de variables, creación de parches,
        normalización y división de la data.
        Hereda los atributos de instancia de Gridded.
    '''

    #Atributos de clase por default:
    max_lat = -17.
    min_lat = -57.
    min_lon = 360. - 76. #80.
    max_lon = 360. - 66. #64.
    dx = 32 # >0
    dt = 3 # >0
    shift = 1 # >0 & < dx
    method = 'linear' #{"linear", "nearest", "zero", "slinear", "quadratic", "cubic", "polynomial", "barycentric", "krog", "pchip", "spline", "akima"}

    def __init__(self, fName='*', path='./'):
        super().__init__(fName, path) #Atributos heredados de Gridded
        
        self.currentVar = None #[str] atributo de "Data Variable" que nos interesa.
        self.vars = None  #None #[List] Contiene la información seteada por la clase GFSVar.
        self.values = None
        self.__getVars__()

        # predef values for methods
        self.max_lat = NCFile.max_lat
        self.min_lat = NCFile.min_lat
        self.min_lon = NCFile.min_lon
        self.max_lon = NCFile.max_lon
        self.dx = NCFile.dx
        self.dt = NCFile.dt
        self.shift=NCFile.shift
        self.method = NCFile.method

       ##### Bloque 1
    ### Bloque 1.1
    def __getFirstFile__(self):
        ''' Recibe el path y el fName (desde Gridded) y retorna su ruta como str en caso de que exista.
        '''
        try:
            ff = sorted(glob.glob(self.path + self.fName))[0]
            if os.path.isfile(ff):
                return ff
            else:
                raise Exception('NCFile:__getFirstFile__:File not found')
        except:
            raise Exception('NCFile:__getFirstFile__:File not found')

    def __getVars__(self):
        ''' Accedemos a la data y guardamos su información: Data Variables, typeOfLevel y data[data_variable].attrs['long_name']
        '''
        self.vars = []
        thisFile = self.__getFirstFile__()
        tmp = xr.open_dataset(thisFile)
        if "era5" or "dem" in thisFile:
            for var in list(tmp.keys()):
                thisVar = GFSVar(var, 'none', tmp[var].attrs['long_name']) 
                self.vars.append(thisVar)
        else:
            var=list(tmp.keys())[0]
            thisVar = GFSVar(var, tmp[var].GRIB_typeOfLevel , tmp[var].attrs['long_name'])   
            self.vars.append(thisVar)
    ##  --- ##

   ###Bloque 1.2:
    def getVarMetaData(self, var='', tol='*'):
        if var == '':
            return [a for a in self.vars]
        elif tol not in GFSVar.typeOfLevels:
            return [a for a in self.vars if a.cfVarName == var]
        else:
            return [a for a in self.vars if ((a.cfVarName == var) and (a.typeOfLevel == tol))]

    def setVar(self, var, tol='*'):
        ''' Definiremos al atributo de instancia currentVar como la triada obtenida al instanciar GFSVar (mismos valores)
        '''

        #triada list var/tol/var_long
        varData = self.getVarMetaData(var=var, tol=tol)
        
        if len(varData) > 1:
            for tvar in varData:
                print(tvar)
            raise Exception('Not single variable for var=' + var + ' and typeOfLevel=' + tol + '.\nTry specifying tol.')
        else:
            varData = varData[0]
        self.currentVar = varData
        return self.currentVar
    ## --- ##

    #Bloque 1.3: 
    def loadVar(self):
        '''Método que nos permite importar los distintos tipos de archivos (era5, dem, etc), estandarizar sus atributos internos y 
            retornar el DataArray definido por cfVarName (atributo self.values).

        Estandarización realizada:
            Para era5: cambiamos el nombre de la variable "time" y tomamos cada 3 tiempos. (total 40 tiempos)
            Para dem: cambiamos el nombre de lon y lat.
            Para GFS0.5: estandarizamos los valores de longitud y descartamos el último registro de tiempo (total 40 tiempos.)
        '''
        
        if "era5" in self.fName:
            self.values = xr.open_mfdataset(self.path + self.fName, concat_dim='time', combine='nested')
            self.values = self.values.rename( {'time':'valid_time'})
            self.values= self.values[self.currentVar.cfVarName][::3,:,:]

        elif "dem"in self.fName:
            self.values = xr.open_mfdataset(self.path + self.fName, concat_dim='time', combine='nested')
            self.values = self.values.rename( {'lon':'longitude', 'lat': 'latitude'})
            self.values= self.values[self.currentVar.cfVarName]

        else:
            self.values = xr.open_mfdataset(self.path + self.fName,
                                        concat_dim='valid_time', combine='nested')
            data_import= self.values[self.currentVar.cfVarName]
            self.values=data_import.assign_coords({'longitude':data_import.longitude.values-360})
            self.values=self.values[:-1,:,:]

        return self.values.copy()
    ##  --- ##

    #Bloque 1.4:   
    def upscaleVar(self, other, method=method):
        '''Método que interpola linealmente (default) las dimensiones del DataArray definido (values) de tal manera que tenga las mismas dimensiones
            que el DataArray dado
        inputs:
            other: [DataArray] DataArray sobre el cual se quiere compatibilizar las dimensiones
            method: [str] Método de interpolación, default='linear'
        return:
            self.values.copy(): [DataArray] Copia del DataArray sobre el cual se aplicó la interpolación y cuyas dimensiones son las mismas que 'other'.
        '''
        self.method = method
        self.values = self.values.interp_like(other, method=self.method)
        return self.values.copy()  
    ##  --- ## 

    #####Bloque 2: 
    ### Bloque 2.1:   
    def extraccion_parches(self,Var, dx=dx):

        '''Extrae parches de tamaño (3,32,32) y retorna el stack con todos los parches
    input:
        Var: [DataArray] Datarray retornado por getVar() de tamaño (40,401,101)
    return:
        stack: [tf tensor] Stack con los parches, stack.shape=(1368, 3, 32, 32)
        '''
    
        stack = tf.stack([],axis=0)

        #recorremos en el tiempo
        for dt in tqdm(range(0,38)):
            #recorremos en la latitude
            for dlat in range(0,401-dx,dx):
                #recorremos en la longitude
                for dlong in range(0,101-dx,dx):  
                    #Cortamos parches de 32x32 en 3 tiempos             
                    parche=Var[ dt:(dt+3), dlat:(dlat+dx), dlong:(dlong+dx)]

                    #si el stack no está vacío, concatenamos los parches
                    if tf.equal(tf.size(stack), 0) == False:
                        parche = tf.expand_dims(parche, axis=0)
                        stack= tf.concat((stack, parche), axis=0)
                        #y pasamos a la siguiente iteración
                        continue
                
                    #si el stack está vacío, lo inicializamos
                    stack=tf.stack([parche],axis=0)
        patches = tf.expand_dims(stack, 4) 
        print("Stack patches shape: ",patches.shape)               
        return patches

    def getVar(self):
        ''' Método que retorna una copia del DataArray. Se puede reemplazar por la llamada directa al atributo.
        '''
        return self.values.copy()             
    ## --- ##

    ### Bloque 2.2: Latitude
    def sampleLat(self):
        ''' Compatibiliza y adapta las dimensiones de los valores de la latitud del DataArray (guardado en values) a las dimensiones
            del propio DataArray.
        return:
            self.extraccion_parches(Var=lat): [tensor] Retorna la salida de la función extracion_parches( Var=lan)
        '''
        ll = tf.constant(self.values.latitude.values)
        vv = self.values.shape
        lat = tf.transpose(tf.broadcast_to(ll, [vv[0], vv[2], vv[1]]), perm=(0, 2, 1))
        return self.extraccion_parches(Var=lat)
    ## --- ##

    #### Bloque 2.3: Longitude
    def sampleLon(self):
        ''' Compatibiliza y adapta las dimensiones de los valores de la longitud del DataArray (guardado en values) a las dimensiones
            del propio DataArray.
        return:
            self.extraccion_parches(Var=lon): [tensor] Retorna la salida de la función extracion_parches( Var=lon)
        '''
        ll = tf.constant(self.values.longitude.values)
        vv = self.values.shape         
        lon = tf.broadcast_to(ll, vv)
        return self.extraccion_parches(Var=lon)
    ## --- ##

    #### Bloque 2.4: orog
    def sampleBand1(self):
        ''' Preprocesa los archivos relacionados a la orografía instanciándolos en una nueva clase y aplicando los mismos métodos que se emplearon para el input y el target 
            para posteriormente extraer sus parches.
            Parte del preprocesamiento conlleva reemplazar los datos Nan por ceros y concatenar los valores del tensor completo una cantidad de veces suficiente para que sus 
            dimensiones coincidan con las del DataArray.
        return:
            self.extraccion_parches(Var=stack_time): [tensor] Retorna la salida de la función extracion_parches( ) aplicada a los valores de stack_time.
        '''
        band1 = NCFile(fName= 'dem.nc', path='')
        band1.setVar('Band1')
        band1.loadVar()
        band1.upscaleVar(self.values)
        band1.values=tf.where(tf.math.is_nan(band1.values),tf.zeros_like(band1.values),band1.values)
        stack_time=tf.repeat(band1.values, 40, axis=0)
        return band1.extraccion_parches(Var=stack_time)
        ## --- ##
    
    ### Bloque 2.5:
    def sampleDayOfYear(self):
        ''' Método encargado de la adaptabilidad y compatibilidad de dimensiones de la variable 'valid_time' con el DataArray. Utiliza el tensor que contiene los nanosegundos transcurridos
            desde inicio de año hasta la fecha registrada en self.values.valid_time como base.
        return:
            time1 = self.extraccion_parches(Var=time) Retorna la salida de la función extracion_parches( ) aplicada a los valores de time.
            time2 = self.extraccion_parches(Var=time) Retorna la salida de la función extracion_parches( ) aplicada a los valores de time pero con un stride de 4.
        '''
        d = self.values.valid_time
        dd = (d-d.astype('M8[Y]')).astype(float) 
        ll = tf.constant(dd) 
        vv = self.values.shape 
        time = tf.transpose(tf.broadcast_to(ll, [vv[1], vv[2], vv[0]]), perm=(2,0,1))
        time1 = self.extraccion_parches(Var=time)
        time2 = self.extraccion_parches(Var=time, dx=4)

        return time1, time2 
    ## --- ##

    ##### Bloque 3: 
    def cantidad_nan(self,inputs=[]):
        '''Recorre cada variable que conforma el input de la red, y cuenta los valores nan de cada uno.
        inputs:
            inputs: [List] Lista con los inputs de la red.
        '''
        inputs_name=["target","inp","lat","lot","orog","time1","time2"]
        cantidad=[];cont=0
        for inp in inputs:
            check_nan= tf.math.is_nan(inp[:,0,:,:,0])
            cont=np.sum(check_nan)
            cantidad.append(cont)

        for nombre,num in zip(inputs_name,cantidad):
            print(f"{nombre}- num valores nan: {num}")

    def parches_nan_target(self,patch_target):
        ''' Recorremos cada parche dentro de target y contamos sus valores Nan, si estos resultan igual a la cantidad de elementos de dicho parche
        (32*32=1024) guardamos su índice en una lista.
        inp:
            patch_target: [tensor] tensor que contiene los parches extraídos de target

        return:
            idx: [list] posición de los parches que no están completamente compuestos con Nan's
        '''
        idx_nulos=[]

        for i in range(patch_target.shape[0]):
            check_nan= tf.math.is_nan(patch_target[i,0,:,:,0])
            if np.sum(check_nan) == patch_target.shape[2]* patch_target.shape[3]: #32*32
                idx_nulos.append(i)
        print("Cantidad de parches nulos: ",len(idx_nulos))

        #filtramos las posiciones de los parches completos de Nan's
        idx= [x for x in range(0,patch_target.shape[0]) if x not in idx_nulos]
        print("Cantidad de parches no nulos: ",len(idx))
        return idx

    def filtracion_nan(self,target,inp,lat,lon,orog,time1,idx):
        ''' A través del método tf.gather(), descarta los parches completos de Nan y retorna los nuevos parches que conformarán el input de la red.
        input:
            -target,inp,lat,lon,orog,time1: [tensor] tensor que contiene los parches de cada variable que conforma al input
            idx: [list] Poisición de los parches que no están completados únicamente con Nan
        return:
         Retorna los nuevos parches limpios para cada variable.
        '''
        target_clean=tf.gather( target, idx, axis=0)
        inp_clean=tf.gather( inp, idx, axis=0)
        lat_clean=tf.gather( lat, idx, axis=0)
        lon_clean= tf.gather( lon, idx, axis=0)
        orog_clean = tf.gather( orog, idx, axis=0)
        time1_clean = tf.gather( time1, idx, axis=0)

        return target_clean, inp_clean, lat_clean, lon_clean, orog_clean, time1_clean
    
    def normalization(self,data_in=input,inverse=False, for_target=False, scale_factor=[1,1]):
        '''Método que normaliza los valores entregados.
        input:
            data_in: [tensor] Tensor con los parches de la variable que forma parte del input de la Red
            inverse: [Boolean, default=False] True en caso de devolver la transformación
            for_target: [Boolean, default=False] True en caso que el data_in sea el target. 
        return: 
        variable normalizada
        '''
        if not inverse:
            if not for_target:
                scale_factor_ = np.max(np.abs(data_in))
                scale_factor_2 = np.min(np.abs(data_in))
                data_out = (data_in-scale_factor_2)/(scale_factor_ - scale_factor_2)
                scale_factor_ = [scale_factor_,scale_factor_2]
            else:
                target_nan_equal_zero= tf.where(tf.math.is_nan(data_in), tf.zeros_like(data_in), data_in)
                scale_factor_=np.max(np.abs(target_nan_equal_zero))
                target_nan_equal_max= tf.where(tf.math.is_nan(data_in), scale_factor_, data_in)
                scale_factor_2=np.min( (np.abs(target_nan_equal_max)))
                data_out = (data_in-scale_factor_2)/(scale_factor_ - scale_factor_2)
                scale_factor_ = [scale_factor_,scale_factor_2]                   
        else:
            data_out=(data_in * (scale_factor[0]-scale_factor[1]))+scale_factor[1]
            scale_factor_ = scale_factor
        return data_out, scale_factor_
    
    def mysplit(self, inp, time, orog, lat, lon, target):
        ''' Método encargado de la división  de todas las variables que conforman el input de la red en: entrenamiento (60%), validación (20%) y testeo.
        return:
            división de cada variable en testeo, validación y testeo.
        '''
        data = tf.concat([tf.cast(inp, dtype=tf.float32),
                      tf.cast(time, dtype=tf.float32),
                      tf.cast(orog, dtype=tf.float32),
                      tf.cast(lat, dtype=tf.float32),
                      tf.cast(lon, dtype=tf.float32),
                      tf.cast(target, dtype=tf.float32)], axis=4)
        data = tf.random.shuffle(data, seed=10)
        n = data.shape[0]
        n_train = int(n * .6)
        n_valid = int(n * .2)
        n_test = n - n_train - n_valid
        data = tf.split(data, [n_train, n_valid, n_test])
        return (tf.expand_dims(data[0][:, :, :, :, 0],4),
            tf.expand_dims(data[0][:, :, :, :, 1],4),
            tf.expand_dims(data[0][:, :, :, :, 2],4),
            tf.expand_dims(data[0][:, :, :, :, 3],4),
            tf.expand_dims(data[0][:, :, :, :, 4],4),
            tf.expand_dims(tf.expand_dims(data[0][:,1,:,:,5],1),4),
            tf.expand_dims(data[1][:, :, :, :, 0],4),
            tf.expand_dims(data[1][:, :, :, :, 1],4),
            tf.expand_dims(data[1][:, :, :, :, 2],4),
            tf.expand_dims(data[1][:, :, :, :, 3],4),
            tf.expand_dims(data[1][:, :, :, :, 4],4),
            tf.expand_dims(tf.expand_dims(data[1][:,1,:,:,5],1),4),
            tf.expand_dims(data[2][:, :, :, :, 0],4),
            tf.expand_dims(data[2][:, :, :, :, 1],4),
            tf.expand_dims(data[2][:, :, :, :, 2],4),
            tf.expand_dims(data[2][:, :, :, :, 3],4),
            tf.expand_dims(data[2][:, :, :, :, 4],4),
            tf.expand_dims(tf.expand_dims(data[2][:,1,:,:,5],1),4))