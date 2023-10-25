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
import pydicom


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

class StanfordDataset(tfds.core.GeneratorBasedBuilder):
  """DatasetBuilder for stanford_ds dataset."""

  VERSION = tfds.core.Version('1.0.0')
  RELEASE_NOTES = {
      '1.0.0': 'Initial release.',
  }

  BUILDER_CONFIGS = [
      ExamConfig(name=f'{type}', description=f'Resultados de tomografia {type.upper()}', img_type=type)
      for type in ['pet', 'chest_ct', 'ct']
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
    segmentation_name = None
    if self.builder_config.img_type == 'chest_ct': 
      exam_name = 'chest_ct_image'
      segmentation_name = 'chest_ct_segmentation'
    if self.builder_config.img_type == 'ct': 
      exam_name = 'ct_image'
      segmentation_name = 'ct_segmentation'
    if self.builder_config.img_type == 'pet': 
      exam_name = 'pet_image'
      segmentation_name = 'pet_segmentation'

     # Carpeta donde la data se encuentra
    archive_path = '/media/roberto/TOSHIBA EXT/pet_ct/stanford_data/'

    final_patients = []
    data_file = Path(os.path.join(archive_path, 'stanford_data_info.csv'))
    with data_file.open() as f:
      for row in csv.DictReader(f):
        if row[exam_name] == '1' and row[segmentation_name] == '1': 
          final_patients.append(row['Case ID'])
    
    if self.builder_config.img_type == 'chest_ct':
      final_patients.remove('R01-022')
      final_patients.remove('R01-026')
      final_patients.remove('R01-038')
      final_patients.remove('R01-076')
      final_patients.remove('R01-078')
      
      range_to_remove = range(103, 151)
      case_ids_to_remove = [f'R01-{str(i).zfill(3)}' for i in range_to_remove]

      # Remove the specified 'Case ID' values
      final_patients = [patient for patient in final_patients if patient not in case_ids_to_remove]



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
            label_base_path = os.path.join(exam_results, f'{patient_id}_{self.builder_config.img_type}_segmentation')
            label_file_path = None
            mask_exam = None
            if os.path.exists(label_base_path + '.nrrd'):
                label_file_path = label_base_path + '.nrrd'
                mask_exam, _ = nrrd.read(label_file_path)
            elif os.path.exists(label_base_path + '.dcm'):
                label_file_path = label_base_path + '.dcm'
                mask_exam = pydicom.dcmread(label_file_path).pixel_array
                mask_exam = np.moveaxis(mask_exam, 0, 2)
            
            data_exam, _ = nrrd.read(image_file_path)


            # Extrae solo los las imagenes  los niveles que contienen segmentacion
            cut_data_exam, cut_mask_exam = extractImages(data_exam, mask_exam)
            
            for i in range(cut_data_exam.shape[2]):
              data_exam_i = cut_data_exam[:,:,i].astype(np.uint16)
              mask_exam_i = cut_mask_exam[:,:,i].astype(np.uint16)

              # Create a unique key using the patient_id and the index of the loop
              example_key = f'{patient_id}_{i}'

              yield example_key, {
                  'patient_id': patient_id,
                  'img_exam': data_exam_i,
                  'mask_exam': mask_exam_i,
                  'label': label_value,
              }