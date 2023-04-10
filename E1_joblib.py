import tensorflow as tf
import os
import numpy as np
import multiprocessing
import time
import readNCdata as nc
from joblib import delayed, Parallel

print("Number of cpu: ", multiprocessing.cpu_count())
#target
era5 = nc.NCFile('era5-land_*.nc', path='./Data/observed/2006/')
era5.setVar('t2m')
era5.loadVar()
print("Clase era5 inicializada")

#input
g05 = nc.NCFile('GFS0.5_t2m_heightAboveGround_instant_200610*.nc', path='./Data/forecasted/2006/2006101000/')
g05.setVar('t2m')
g05.loadVar()

print("Clase g05 inicializada")

g05.upscaleVar(era5.values)

def expand_dims(patch): 
    print(f"Process {os.getpid()} working record")
    return tf.expand_dims(patch, axis=0)

dx=32
parches = [g05.getVar()[ dt:(dt+3), dlat:(dlat+32), dlong:(dlong+32)]  for dt in range(0,38) for dlat in range(0,401-dx,dx) for dlong in range(0,101-dx,dx)]
print("Parches creados. \n Cantidad de parches: ", len(parches))


print("Inicio del proceso de paralelización")
tiempos=open("times_joblib.txt","w")
def expand_dims(patch): return tf.expand_dims(patch, axis=0)

start= time.time()
result = Parallel(n_jobs=-1)(delayed(expand_dims)(patch) for patch in parches)
end=time.time()
tiempos.write(str(end-start))
tiempos.close()
print("Fin del proceso.")
print(f"\nTime to complete: {end - start:.2f}s\n")
print("Tipo de variable: ", type(result))
print("Tipo de variable parche: ",type(result[0]))
print("Dimensiones: ", result[0].shape)