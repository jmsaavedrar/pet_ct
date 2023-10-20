"""santa_maria_ds_dataset."""

# se separan los pacientes en entrenamiento y testeo dados los id de los pacientes (de forma balanceada)
# separar entrenamiento y testeo
# hacer clases balanceadas
# que todas las clases de una misma persona se encuentren en el conjunto de entrenamiento, o en el de testeo


import tensorflow_datasets as tfds
import os
import nrrd
import csv
import random
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import cv2
from tqdm import trange
from pathlib import Path
from dataclasses import dataclass


def extractImages(data_img, data_mask):

    """
  Función que extrae sólo las imágenes de los niveles que contienen segmentación.
  INPUT: volumen de imágenes y máscaras numpy array 3D (numero de imagen, alto, ancho), directo de la
  carga del archivo .nrrd.
  Busca las máscaras con segmentaciones y extrae los cortes de estos niveles.
  OUTPUT: devuelve np.array 3D con las imágenes y máscaras sólo de los niveles del tumor.
    """

    images = []
    masks = []
    positive_slices = []

    for i in trange(data_img.shape[2]):
        segmentation = data_mask[:,:,i]
        if (np.sum(segmentation)>0):
            positive_slices.append(i)

        ## Extrae las imágenes sólo con segmentación - tumor
    for axial in positive_slices:
        segment = data_mask[:,:,axial]
        ct = data_img[:,:,axial]
        images.append(ct)
        masks.append(segment)

    images = np.stack(images, axis=2)
    masks = np.stack(masks, axis=2)
    return(images, masks)



@dataclass
class ExamConfig(tfds.core.BuilderConfig):
  img_type: str = 'pet'
  win_size: int = 1


class SantaMariaDataset(tfds.core.GeneratorBasedBuilder):
  """DatasetBuilder for santa_maria_ds dataset."""

  
  VERSION = tfds.core.Version('1.0.0')
  RELEASE_NOTES = {
      '1.0.0': 'Initial release.',
  }

  BUILDER_CONFIGS = [
      ExamConfig(name=f'{type}_{i}', description=f'Resultados de tomografia {type.upper()}', img_type=type, win_size=i)
      for type in ['torax3d', 'body', 'pet']
      for i in range(1, 2)
  ]


  MANUAL_DOWNLOAD_INSTRUCTIONS = """Para ejecutar en Google Colab. Pedir acceso al conjunto de datos a Hector
  Henriquez. Estos deben contener las carpetas Data Clínica Santa Maria/DATA NRRD/ y dentro de esta las carpetas:
  EGFR+ y EGFR- con los resultados positivos y negativos respectivamente. Dejar el contenido de la carpeta /DATA NRRD dentro
  de una carpeta santa_maria_ds"""

  def _info(self) -> tfds.core.DatasetInfo:
    """Returns the dataset metadata."""

    return self.dataset_info_from_configs(
        features=tfds.features.FeaturesDict({
            # Features of the dataset
            'patient_id': tfds.features.Text(doc='Id of patients of Santa Maria'),
            'img_exam': tfds.features.Tensor(shape=(None, None, self.builder_config.win_size),
                                            dtype=np.uint16,
                                            encoding='zlib',
                                            doc = 'Exam Images'),
            'mask_exam': tfds.features.Tensor(shape=(None, None, self.builder_config.win_size),
                                            dtype=np.uint16,
                                            encoding='zlib',
                                            doc = 'Tumor Mask'),
            'label': tfds.features.ClassLabel(num_classes=2, 
                                              doc='Results on the EGFR Mutation test.'),
        }),
        supervised_keys=None,  # Set to `None` to disable
        disable_shuffling=False,
    )

  def _split_generators(self, dl_manager):
    
    # Lists of patients
    egfr_positive_patients = [f'sm_{str(i).zfill(3)}' for i in range(1, 13)]
    egfr_negative_patients = [f'sm_{str(i).zfill(3)}' for i in range(13, 36)]

    # Shuffle the data randomly
    random.shuffle(egfr_positive_patients)
    random.shuffle(egfr_negative_patients)

    split_var = 0.8

    # Determine the split points for each class
    split_point_positive = int(split_var * len(egfr_positive_patients))
    split_point_negative = int(split_var * len(egfr_negative_patients))

    # Split the data into training and testing sets for each class
    training_positive = egfr_positive_patients[:split_point_positive]
    testing_positive = egfr_positive_patients[split_point_positive:]
    training_negative = egfr_negative_patients[:split_point_negative]
    testing_negative = egfr_negative_patients[split_point_negative:]

    # Combine the training and testing sets for both classes
    training_data = training_positive + training_negative
    testing_data = testing_positive + testing_negative

    print(training_data)
    print()
    print(testing_data)
    # Carpeta donde la data se encuentra
    archive_path = 'santa_maria_data/'

    # Retorna un Dict[all_data, Iterator[Key, Example]]
    return {
      'train': self._generate_examples(archive_path, training_data),
      'test': self._generate_examples(archive_path, testing_data)
    }

  def _generate_examples(self, path, patient_list):
    
    # Lee el archivo csv y retorna los ejemplos de un examen (pet, ct o torax) para una ventana (numero entero).

    data_file = Path(os.path.join(path, 'santamaria_data.csv'))
    with data_file.open() as f:
      for row in csv.DictReader(f):
        patient_id = row['PATIENT_ID']
        print(patient_id) 

        if patient_id in patient_list:
          results_folder = "EGFR+" if row['EGFR'] == '1' else "EGFR-"
          results_folder = os.path.join(path, results_folder, self.builder_config.img_type) # pytype: disable=attribute-error

          image_file_path = os.path.join(results_folder, "image", f'{patient_id}_{self.builder_config.img_type}_image.nrrd')
          label_file_path = os.path.join(results_folder, "label", f'{patient_id}_{self.builder_config.img_type}_segmentation.nrrd')

          # Revisa si la imagen y el label existen antes de retornarlos
          if os.path.exists(image_file_path) and os.path.exists(label_file_path):
            data_exam, _ = nrrd.read(image_file_path)
            mask_exam, _ = nrrd.read(label_file_path)
            print('shape of the exam images', data_exam.shape, mask_exam.shape)

            # Extrae solo los las imagenes  los niveles que contienen segmentacion
            cut_data_exam, cut_mask_exam = extractImages(data_exam, mask_exam)
            
            # Convierte el dtype de las imagenes a uint8
            cut_data_exam = cut_data_exam.astype(np.uint16)
            cut_mask_exam = cut_mask_exam.astype(np.uint16)

            print('shape of the exam cut images', cut_data_exam.shape, cut_mask_exam.shape)
            window_size = self.builder_config.win_size
            for i in range(cut_data_exam.shape[2] - window_size + 1):
              data_exam_i = cut_data_exam[:, :, i:i+window_size]
              mask_exam_i = cut_mask_exam[:, :, i:i+window_size]

              # Crea una llave unica usando el patient_id y el indice del loop
              example_key = f'{patient_id}_{i}'

              yield example_key, {
                  'patient_id': patient_id,
                  'img_exam': data_exam_i,
                  'mask_exam': mask_exam_i,
                  'label': row['EGFR'],
              }


