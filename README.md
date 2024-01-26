# pet_ct
Repositorio del estudio de detección de tumores en pacientes. Se implementan modelos multimodales de *deep learning* para realizar predicciones de tumores malignos en los datos. El repositorio también cuenta con la implementación de dos fuentes de datos relevantes para el estudio en *tensorflow*:

1. Cohorte local de pacientes de Clínica Santa María con diagnóstico de cancer de pulmon confirmado por histologia.
2. Fuente de datos pública de pacientes con cáncer de pulmón de tipo células no pequeñas correspondiente al Dataset NSCLC Radiogenomics. Este set de datos corresponde a 211 pacientes adultos con cáncer de pulmón, imágenes PET-CT, entre otros.

## Instalacion de Python y dependencias
Para manejar las dependencias necesarias para correr los códigos, en primer lugar es necesario instalar anaconda (o miniconda) e inicializar un nuevo ambiente virtual con **Python 3.8**:

```
conda create -n "myenv" python=3.8.12
```

A continuacion crear ambiente virtual con el archivo de *environment.yml*.

```
conda env create -f environment.yml
```



Los ambientes de conda se activan y desactivan, respectivamente, de la siguiente forma:

```
conda activate lung_radiomics
conda deactivate
```

Verificar que estén instaladas las siguientes versiones de las libreras:

```
tensorflow==2.12.0
tensorflow_datasets==4.9.2
```
**IMPORTANTE**: Puede que no funcione el sistema con otra versión de las librerías anteriores.

## Uso de datos
Para usar el conjunto de datos PET, CT y TORAX, pedir los conjuntos de datos de *tensorflow* a Prof. José Saavedra y descomprimir sus contenidos en el directorio en 'home/tensorflow_datasets/'. Los conjuntos de datos son ser:
- santa_maria_dataset
- stanford_dataset

Habiendo configurados los datos, se pueden utilizar. Para aprender mas de *tensorflow_datasets* se puede revisar el siguiente [link](https://www.tensorflow.org/datasets/add_dataset?hl=es).

### Utilizar los datos de la Clínica Santa María

Desde cualquier directorio, se pueden utilizar los tres conjuntos de datos de forma independiente PET, CT y TORAX de la siguiente forma:


```
import tensoflow_datasets as tfds

ds = tfds.load('santa_maria_dataset/<tipo_examen>') # donde <tipo_examen> puede ser 'pet', 'body' o 'torax3d'.
dataset = ds['sm_001']                            # obtiene todos los datos del paciente 'sm_001'. Esto se puede hacer para todos los pacientes del conjunto de datos.
```


### Utilizar los datos Stanford

Desde cualquier directorio, se pueden utilizar los tres conjuntos de datos de forma independiente PET, CT y TORAX de la siguiente forma:


```
import tensoflow_datasets as tfds

ds = tfds.load('stanford_dataset/<tipo_examen>') # donde <tipo_examen> puede ser 'pet', 'ct' o 'chest_ct'.
dataset = ds['R01-001']                            # obtiene todos los datos del paciente 'R01-001'. Esto se puede hacer para todos los pacientes del conjunto de datos.
```

El archivo *load_datasets_example.ipynb* realiza un ejemplo de como construir *k-fold* y divisiones de conjuntos de datos en *train* y *test* para entrenar modelos de aprendizaje. Se puede también especificar la ubicación de los conjuntos de datos con el argumento *data_dir* en la función *tfds.load*.



## Entrenar modelos base

Se ofrecen dos *pipelines* de entrenamiento, validación y testeo para los conjuntos de datos de Santa María y de Stanford. Estos corresponden a los scripts */js/train_model.py* y */js/train_model_kfold.py*.

*train_model_kfold.py* evalúa modelos visuales con KFold para los conjuntos de Stanford y Santa Maria.

*train_model.py* evalúa modelos visuales de forma que son entrenados en el conjunto de Stanford y evaluados en el conjunto de Santa María.

Reciben los siguientes argumentos:

- **-d, --dataset**: Especifica el conjunto de datos a utilizar, que puede ser "santa_maria_dataset" o "stanford_dataset". Este parámetro determina la fuente de datos para el entrenamiento del modelo. Solo para *train_model_kfold*.

- **-p, --particion**: Define el tipo de partición en el conjunto de datos, siendo opciones válidas "torax3d", "body", "pet" para "santa_maria_dataset" y "pet", "ct", "chest_ct" para "stanford_dataset".

- **-b, --batch**: Controla el tamaño del lote utilizado durante el entrenamiento. Este parámetro influye en la cantidad de muestras que se utilizan en cada iteración de entrenamiento.

- **-s, --size**: Especifica el tamaño de la imagen para la extracción de la Región de Interés (ROI). Este parámetro determina las dimensiones de las imágenes que serán utilizadas en el modelo.

- **-e, --epochs**: Indica el número de épocas de entrenamiento que el modelo realizará. Una época corresponde a una iteración completa a través de todo el conjunto de datos de entrenamiento.

- **-n, --splits**: Define el número de divisiones para el método de validación cruzada KFold. Este parámetro es relevante para evaluar el rendimiento del modelo.

- **--seed**: Permite establecer una semilla aleatoria para asegurar la reproducibilidad de los experimentos.

### Validación de argumentos
El script incluye validaciones para garantizar que los argumentos proporcionados sean coherentes y válidos:

- Se verifica que el conjunto de datos especificado sea válido, limitándolo a "santa_maria_dataset" o "stanford_dataset".

- Dependiendo del conjunto de datos, se asegura que el tipo de partición sea apropiado, siendo "body", "pet", "torax3d" para "santa_maria_dataset", y "pet", "ct", "chest_ct" para "stanford_dataset".

En caso de que los argumentos no cumplan con estas validaciones, se lanzan excepciones con mensajes descriptivos indicando el problema. Este enfoque garantiza que el usuario proporcione configuraciones coherentes y adecuadas para el proceso de entrenamiento del modelo.

**Importante**: Para la ejecución de los scripts, ejecutar el siguiente comando:

```
export TF_CPP_MIN_LOG_LEVEL=2
```


## Modificación de los datos
Para modificar el conjunto de datos de *tensorflow_datasets*, en primer lugar es necesario pedir el acceso a los datos *'Data Clinica Santa Maria'* y *'Data NSCLC Radiogenomics'* a Hector Henriquez. 


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
