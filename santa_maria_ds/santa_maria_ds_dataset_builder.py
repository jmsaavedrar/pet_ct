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

    images = np.array(images)
    masks = np.array(masks)
    return(images, masks)


def roiExtraction(img, mask, total_size):
    """
    Function to extract ROIs from images while ensuring a consistent total size for all ROIs.

    INPUT:
    img: Numpy array of images.
    mask: Numpy array of masks.
    total_size: The desired total size (width and height) of the extracted ROIs.

    OUTPUT: Numpy array containing the ROIs.
    """
    roi_extract = []

    for i in range(mask.shape[0]):
        img_instance = img[i].copy()
        mask_instance = mask[i].copy()

        # Calculate the center of the mask.
        center_row = int(np.mean(index[0]))
        center_col = int(np.mean(index[1]))

        # Calculate the size of the ROI based on the total size.
        half_size = total_size // 2

        # Determine ROI boundaries with the margin.
        min_row = max(0, center_row - half_size)
        max_row = min(mask_instance.shape[0], center_row + half_size)
        min_col = max(0, center_col - half_size)
        max_col = min(mask_instance.shape[1], center_col + half_size)

        # Calculate the width and height of the ROI.
        roi_height = max_row - min_row
        roi_width = max_col - min_col

        # Case 1: If the ROI is smaller than the total_size, add a margin to make it total_size.
        if roi_height < total_size:
            margin = (total_size - roi_height) // 2
            min_row -= margin
            max_row += margin

        if roi_width < total_size:
            margin = (total_size - roi_width) // 2
            min_col -= margin
            max_col += margin

        # Case 2: If the ROI is larger than total_size, resize it.
        if roi_height > total_size or roi_width > total_size:
            scale_factor = total_size / max(roi_height, roi_width)
            new_height = int(roi_height * scale_factor)
            new_width = int(roi_width * scale_factor)
            min_row = max(center_row - new_height // 2, 0)
            max_row = min(min_row + new_height, mask_instance.shape[0])
            min_col = max(center_col - new_width // 2, 0)
            max_col = min(min_col + new_width, mask_instance.shape[1])

        # Extract the ROI with the desired size.
        roi = img_instance[min_row:max_row, min_col:max_col]
        # Add to the results.
        roi_extract.append(roi)

    roi_extract = np.array(roi_extract, dtype=object)
    return roi_extract




@dataclass
class ExamConfig(tfds.core.BuilderConfig):
  img_type: str = 'pet'
  img_size: int = 32


class SantaMariaDataset(tfds.core.GeneratorBasedBuilder):
  """DatasetBuilder for santa_maria_ds dataset."""

  
  VERSION = tfds.core.Version('1.0.0')
  RELEASE_NOTES = {
      '1.0.0': 'Initial release.',
  }

  BUILDER_CONFIGS = [
      ExamConfig(name=f'{type}_{tamano}', description=f'Resultados de tomografia {type.upper()} de tamaño {tamano}',
      img_type=type, img_size=tamano)
      for type in ['pet', 'torax3d', 'body']
      for tamano in [32, 64, 128]
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
            'img_exam': tfds.features.Tensor(shape=(self.builder_config.img_size, self.builder_config.img_size),
                                            dtype=np.uint16,
                                            encoding='zlib',
                                            doc = 'Exam Images'),
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
            cut_data_roi = roiExtraction(cut_data_exam, cut_mask_exam, self.builder_config.img_size)
            
            # Convierte el dtype de las imagenes a uint8
            #cut_data_roi = cut_data_roi.astype(np.uint16)
            for i in range(cut_data_roi.shape[0]):
              data_exam_i = cut_data_roi[i].astype(np.uint16)

              # Create a unique key using the patient_id and the index of the loop
              example_key = f'{patient_id}_{i}'

              yield example_key, {
                  'patient_id': patient_id,
                  'img_exam': data_exam_i,
                  'label': row['EGFR'],
              }
