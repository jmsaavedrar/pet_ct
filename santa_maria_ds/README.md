# pet_ct
Fuente de datos de de cohorte local de pacientes de Clinica Santa Maria con diagnostico de cancer de pulmon confirmado por histologia, que tengan imagenes de PET CT y estudio mutacional. Solo tiene las imagenes del examen PET.

## Instalacion de Python y dependencias
Instalar anaconda (o miniconda), e inicializar un nuevo ambiente virtual con **Python 3.8**:

```
conda create -n "myenv" python=3.8.12
```

A continuacion instalar los requirements con las dependencias de librerias:

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

## Configuracion de los datos
Pedir acceso a los datos 'Data Clinica Santa Maria' a Hector Henriquez. Dentro del repositorio agregar una carpeta *'santa_maria_data'* con el contenido del la carpeta 'DATA NRRD'. Es decir, *santa_maria_data* debe tener las carpetas 'EGFR-' y 'EGFR+', y el archivo *dataLungRadiomicsSantaMaria_ANON.csv*.

Con los datos configurados, ejecutar el siguiente comando en la carpeta donde se encuentra *santa_maria_ds_dataset_builder* para generar el conjunto de datos:

```
tfds build
```

## Utilizar el conjunto de datos PET, CT y TORAX

Desde cualquier directorio, se pueden utilizar los tres conjuntos de datos de forma independiente PET, CT y TORAX:


```
ds = tfds.load('santa_maria_dataset/<tipo_examen>') # donde <tipo_examen> puede ser 'pet_1', 'body_1' o 'torax3d_1'.
dataset = ds['all_data']                            # obtiene todos los datos
```


Cambiar nombre de un dato de pet:

sm_027_pet_images.nrrd     a     sm_027_pet_image.nrrd 
