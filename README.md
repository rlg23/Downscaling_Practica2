# Práctica de Estudios II: Downscaling metereológico.

**Tema:** Adaptación de un sistema de downscaling para datos metereológicos basado en redes neuronales convolucionales (CNN: U-net).
Descripción: El trabajo incluye leer y escribir grandes volúmenes de datos en formatos científicos (NetCDF) utilizando python, pre-procesar dichos datos para ser ingresados a la red U-net a través de parches y adaptar aspectos internos de ésta misma.

**Nota:** Se estudia la posibilidad de utilizar el servidor Khipu como sistema de super cómputo para el entrenamiento y ejecución de la red.

Supervisor: Dr. Jorge Arévalo B. <br>
Rubro: Departamento de metereología, Universidad de Valparaíso, Chile. <br>
Correo:  [jab@meteo.uv.cl](jab@meteo.uv.cl)

Estudiante: Rubén Esteban Lagos Godoy <br>
Rubro: Estudiante Ing. Civil Matemática, Universidad de la Frontera, Chile. <br>
Correo: [r.lagos08@ufromail.cl](r.lagos08@ufromail.cl)

_________________________________________________________________________________

**Descripción de los archivos:**

[Data](Data): Carpeta con los archivos .nc necesarios. Contiene archivos .rar que se deben descomprimir en su respectivo lugar. Para más detalle de la disposición de los archivos visitar el fichero a continuación.

[A1_guiaNC.ipynb](A1_guiaNC.ipynb): Sirve como guía para la reproducibilidad del repositorio y contiene la descripción detallada de las clases y sus respectivos métodos que conforman el preprocesamiento de la data.

[A2_test3NC.ipynb](A2_test3NC.ipynb): Contiene el proceso completo (no detallado) del preprocesamiento de la data.

[B1_readNCdata.py](B1_readNCdata.py): Guarda todas las clases empleadas en el preprocesamiento.

[B2_Model.py](B2_Model.py): Contiene a la U-net

[C1_parches.rar](C1_parches.rar): *Descrompimir su contenido en el mismo lugar*. Contiene parches pre-procesados. 

[D1_orden_for.ipynb](D1_orden_for.ipynb): Contiene el testeo del orden de los ciclos for en la extracción de los parches

[D2_monos.ipynb](D2_monos.ipynb): Contiene el resumen de los tiempos para las distintas técnicas de optimización empleadas.

[E1_joblib.py](E1_joblib.py): Contiene el proceso de extracción de parches paralelizado con la librería *joblib*

[E2_multiprocessing.py](E2_multiprocessing.py): Contiene el proceso de extracción de parches paralelizado con la librería *multiprocessing*

[Extra_target_nan.ipynb](Extra_target_nan.ipynb): Expone el problema de los Nan's en el target.

[requirements.txt](requirements.txt): Contiene las dependencias necesarias para ejecutar el repositorio.

[registros_tiempos2023-03-21.pkl](registros_tiempos2023-03-21.pkl) Contiene los tiempos de cómputo de las distintas combinaciones de ciclos for

[times_joblib.txt](times_joblib.txt) Contiene los tiempos de cómputo para la extracción de parches paralelizada con el método *joblib*

[times_multiprocessing.txt](times_multiprocessing.txt) Contiene los tiempos de cómputo para la extracción de parches paralelizada con el método *multiprocessing*

**Extras:**

```
conda create -n <nombre_env> python=3.10.4
```  
```
pip install -r requirements.txt
```


