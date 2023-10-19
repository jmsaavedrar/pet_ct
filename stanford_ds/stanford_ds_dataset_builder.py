"""stanford_ds dataset."""

import tensorflow_datasets as tfds
import os
import nrrd
import csv
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import cv2
from tqdm import trange
from pathlib import Path
from dataclasses import dataclass


margin = 4

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

# margen 4
def roiExtraction (img,mask,margin):
    """
  Función para extraer el ROI donde se encuentra el tumor en una imagen.
  INPUT: imagen y máscaras numpy array 2D.
  margin: corresponde al número de pixeles como margen por fuera de los pixeles de la máscara.
  OUTPUT: Devuelve el ROI.
  """

    roi_extract = []

    for i in range(mask.shape[0]):
        img_instance = img[i].copy()
        mask_instance = mask[i].copy()
        index = np.where(mask_instance)
        roi = img_instance[np.unique(index[0])[0]-margin:np.unique(index[0])[-1]+margin, np.unique(index[1])[0]-margin: np.unique(index[1])[-1]+margin]
        roi_extract.append(roi)

    roi_extract = np.array(roi_extract, dtype='object')
    return(roi_extract)


@dataclass
class ExamConfig(tfds.core.BuilderConfig):
  img_type: str = 'pet'

class StanfordDataset(tfds.core.GeneratorBasedBuilder):
  """DatasetBuilder for stanford_ds dataset."""

  VERSION = tfds.core.Version('1.0.0')
  RELEASE_NOTES = {
      '1.0.0': 'Initial release.',
  }

  BUILDER_CONFIGS = [
      ExamConfig(name=f'{type}', description=f'Resultados de tomografia {type.upper()}', img_type=type)
      for type in ['chest_ct', 'ct', 'pet']
  ]

  def _info(self) -> tfds.core.DatasetInfo:
    """Returns the dataset metadata."""
    return self.dataset_info_from_configs(
      features=tfds.features.FeaturesDict({
          # Features of the dataset
          'patient_id': tfds.features.Text(doc='Id of patients of Stanford'),
          'img_exam': tfds.features.Tensor(shape=(None, None),
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
    if self.builder_config.img_type == 'chest_ct': exam_name = 'chest_ct_image'
    if self.builder_config.img_type == 'ct': exam_name = 'ct_image'
    if self.builder_config.img_type == 'pet': exam_name = 'pet_image'

     # Carpeta donde la data se encuentra
    archive_path = 'stanford_data/'

    final_patients = []
    data_file = Path(os.path.join(archive_path, 'stanford_data_info.csv'))
    with data_file.open() as f:
      for row in csv.DictReader(f):
        if row[exam_name] == '1' and row[exam_name][:3] == 'AMC': 
          final_patients.append(row['Case ID'])

    # Create dictionaries of patients with their associated data
    return {patient: self._generate_examples(archive_path, patient) for patient in final_patients} 


  
  def _generate_examples(self, path, patient_list):

    # Lee el archivo csv y retorna los ejemplos de un examen (pet, ct o torax) para una ventana (numero entero).
    data_file = Path(os.path.join(path, 'stanford_data_info.csv'))
    image_folder = os.path.join(path, 'data')
    with data_file.open() as f:
      for row in csv.DictReader(f):
        patient_id = row['Case ID']
        
        if patient_id == patient_list:
          label_value  = 1 if row['EGFR mutation status'] == 'Mutant' else 0
          exam_results = os.path.join(image_folder, patient_id, self.builder_config.img_type) # pytype: disable=attribute-error
          
          if os.path.exists(exam_results):

            image_file_path = os.path.join(exam_results, f'{patient_id}_{self.builder_config.img_type}_image.nrrd')
            label_file_path =  os.path.join(exam_results, f'{patient_id}_{self.builder_config.img_type}_segmentation.nrrd')

            data_exam, _ = nrrd.read(image_file_path)
            mask_exam, _ = nrrd.read(label_file_path)

            # Extrae solo los las imagenes  los niveles que contienen segmentacion
            cut_data_exam, cut_mask_exam = extractImages(data_exam, mask_exam)
            cut_data_roi = roiExtraction(cut_data_exam, cut_mask_exam, margin)
            
            # Convierte el dtype de las imagenes a uint8
            #cut_data_roi = cut_data_roi.astype(np.uint16)
            for i in range(cut_data_roi.shape[0]):
              data_exam_i = cut_data_roi[i].astype(np.uint16)

              print('data shape:', data_exam_i.shape)

              # Create a unique key using the patient_id and the index of the loop
              example_key = f'{patient_id}_{i}'

              yield example_key, {
                  'patient_id': patient_id,
                  'img_exam': data_exam_i,
                  'label': label_value,
              }
