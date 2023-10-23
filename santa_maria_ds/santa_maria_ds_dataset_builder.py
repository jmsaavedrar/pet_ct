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


class SantaMariaDataset(tfds.core.GeneratorBasedBuilder):
  """DatasetBuilder for santa_maria_ds dataset."""

  
  VERSION = tfds.core.Version('1.0.0')
  RELEASE_NOTES = {
      '1.0.0': 'Initial release.',
  }

  BUILDER_CONFIGS = [
      ExamConfig(name=f'{type}', description=f'Resultados de tomografia {type.upper()}',
      img_type=type)
      for type in ['pet', 'torax3d', 'body']
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
            'img_exam': tfds.features.Tensor(shape=(None, None),
                                            dtype=np.uint16,
                                            encoding='zlib',
                                            doc = 'Exam Images'),
            'mask_exam': tfds.features.Tensor(shape=(None, None),
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

    exam_name = None
    if self.builder_config.img_type == 'torax3d': exam_name = '3D_TORAX_SEG'
    if self.builder_config.img_type == 'body': exam_name = 'BODY_CT_SEG'
    if self.builder_config.img_type == 'pet': exam_name = 'PET_SEG'

     # Carpeta donde la data se encuentra
    archive_path = 'santa_maria_data/'

    final_patients = []
    data_file = Path(os.path.join(archive_path, 'santamaria_data.csv'))
    with data_file.open() as f:
      for row in csv.DictReader(f):
        if row[exam_name] == '1': final_patients.append(row['PATIENT_ID'])

    # Create dictionaries of patients with their associated data
    return {patient: self._generate_examples(archive_path, patient) for patient in final_patients} 

  def _generate_examples(self, path, patient_list):
    
    # Lee el archivo csv y retorna los ejemplos de un examen (pet, ct o torax) para una ventana (numero entero).
    data_file = Path(os.path.join(path, 'santamaria_data.csv'))
    with data_file.open() as f:
      for row in csv.DictReader(f):
        patient_id = row['PATIENT_ID']
        if patient_id == patient_list:
          results_folder = "EGFR+" if row['EGFR'] == '1' else "EGFR-"
          results_folder = os.path.join(path, results_folder, self.builder_config.img_type) # pytype: disable=attribute-error

          image_file_path = os.path.join(results_folder, "image", f'{patient_id}_{self.builder_config.img_type}_image.nrrd')
          label_file_path = os.path.join(results_folder, "label", f'{patient_id}_{self.builder_config.img_type}_segmentation.nrrd')
          
          # Revisa si la imagen y el label existen antes de retornarlos
          if os.path.exists(image_file_path) and os.path.exists(label_file_path):
            data_exam, _ = nrrd.read(image_file_path)
            mask_exam, _ = nrrd.read(label_file_path)
            
            # Extrae solo los las imagenes  los niveles que contienen segmentacion
            cut_data_exam, cut_mask_exam = extractImages(data_exam, mask_exam)
            print(cut_data_exam.shape, cut_mask_exam.shape)
            # Convierte el dtype de las imagenes a uint8
            #cut_data_roi = cut_data_roi.astype(np.uint16)
            for i in range(cut_data_exam.shape[2]):
              data_exam_i = cut_data_exam[:,:,i].astype(np.uint16)
              mask_exam_i = cut_mask_exam[:,:,i].astype(np.uint16)

              # Create a unique key using the patient_id and the index of the loop
              example_key = f'{patient_id}_{i}'

              yield example_key, {
                  'patient_id': patient_id,
                  'img_exam': data_exam_i,
                  'mask_exam': mask_exam_i,
                  'label': row['EGFR'],
              }
