# pet_ct
Repositorio del estudio de detección de tumores en pacientes. Se implementan modelos multimodales de *deep learning* para realizar predicciones de tumores malignos en los datos. El repositorio también cuenta con la implementación de dos fuentes de datos relevantes para el estudio en *tensorflow*:

1. Cohorte local de pacientes de Clínica Santa María con diagnóstico de cancer de pulmon confirmado por histologia.
2. Fuente de datos pública de pacientes con cáncer de pulmón de tipo células no pequeñas correspondiente al Dataset NSCLC Radiogenomics. Este set de datos corresponde a 211 pacientes adultos con cáncer de pulmón, imágenes PET-CT, entre otros.

## Instalacion de Python y dependencias
Para manejar las dependencias necesarias para correr los códigos, en primer lugar es necesario instalar anaconda (o miniconda) e inicializar un nuevo ambiente virtual con **Python 3.8**:

```
conda create -n "myenv" python=3.8.12
```

A continuacion instalar *requirements.txt* que se refiere a las dependencias necesarias:

```
conda activate myenv
conda install --file requirements.txt
```

Los ambientes de conda se activan y desactivan, respectivamente, de la siguiente forma:

```
conda activate myenv
conda deactivate
```

Verificar que este instalada las siguientes versiones de las libreras:

```
tensorflow==2.12.0
tensorflow_datasets==4.9.2
```
**IMPORTANTE**: Puede que no funcione el sistema con otra versión de las librerías anteriores.

## Configuracion de los datos
Pedir acceso a los datos *'Data Clinica Santa Maria'* y *'Data NSCLC Radiogenomics'* a Hector Henriquez. 


### Configuración de datos de Santa María
Para el conjunto de datos de *Santa Maria*, crear una carpeta *santa_maria_data* dentro de la carpeta *santa_maria_ds* para descargar el contenido del la carpeta 'DATA NRRD'. Es decir, *santa_maria_data* debe tener:
- Carpeta 'EGFR-' 
- Carpeta 'EGFR+'
- El archivo *dataLungRadiomicsSantaMaria_ANON.csv*, renombrado a *santamaria_data.csv*.

Con los datos configurados, ejecutar el siguiente comando en la carpeta donde se encuentra *santa_maria_ds_dataset_builder.py* para generar el conjunto de datos:

```
tfds build
```

### Configuración de datos de Stanford
Para el conjunto de datos de *Stanford*, crear una carpeta *stanford_data* dentro de *stanford_ds* que contenga la carpeta *data* con los datos de los pacientes, y el archivo *DATA_LABELS_NSCLC_RADIOGENOMICS.csv* que se encuentra dentro de DATA NSCLC COMPLETO. Así, los archivos se ven de la siguiente forma:
- stanford_ds > stanford_data > contiene:
1. data
2. DATA_LABELS_NSCLC_RADIOGENOMICS.csv renombrado a *stanford_data_info.csv*-

A continuación, ejecutar el siguiente comando en la carpeta donde se encuentra el *builder* para generar el conjunto de datos:

```
tfds build
```

## Utilizar el conjunto de datos PET, CT y TORAX


### Utilizar los datos de la Clínica Santa María

Desde cualquier directorio, y después de haber construido el conjunto de datos, se pueden utilizar los tres conjuntos de datos de forma independiente PET, CT y TORAX de la siguiente forma:


```
ds = tfds.load('santa_maria_dataset/<tipo_examen>') # donde <tipo_examen> puede ser 'pet_1', 'body_1' o 'torax3d_1'.
dataset = ds['sm_001']                            # obtiene todos los datos del paciente 'sm_001'. Esto se puede hacer para todos los pacientes del conjunto de datos.
```

El archivo *load_santamaria_ds_examples.ipynb* realiza un ejemplo de como construir *k-fold* y divisiones de conjuntos de datos en *train* y *test* para entrenar modelos de aprendizaje. Revisar el archivo.


### Utilizar los datos Stanford
