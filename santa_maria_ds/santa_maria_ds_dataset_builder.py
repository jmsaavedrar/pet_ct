"""santa_maria_ds_dataset."""

# se separan los pacientes en entrenamiento y testeo dados los id de los pacientes (de forma balanceada)
# separar entrenamiento y testeo
# hacer clases balanceadas
# que todas las clases de una misma persona se encuentren en el conjunto de entrenamiento, o en el de testeo

# agregar el spacing de las imágenes construidas

import tensorflow_datasets as tfds
import os
import nrrd
import csv
import random
import numpy as np
from pathlib import Path
from dataclasses import dataclass


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
                                            dtype=np.float32,
                                            encoding='zlib',
                                            doc = 'Exam Images'),
            'mask_exam': tfds.features.Tensor(shape=(None, None),
                                            dtype=np.int32,
                                            encoding='zlib',
                                            doc = 'Tumor Mask'),
            'egfr_label': tfds.features.ClassLabel(names=['Wildtype', 'Mutant'],
                                            doc='Results on the EGFR Mutation test: 1 is positive and 0 is negative.'),
            'pet_liver': tfds.features.Tensor(shape=(None,), 
                                            dtype=np.float32, 
                                            encoding='zlib', 
                                            doc='Liver PET Images'),
            'exam_metadata': tfds.features.FeaturesDict({
                'space_directions':  tfds.features.Tensor(shape=(3,), 
                                                dtype=np.float64, 
                                                encoding='zlib', 
                                                doc='space directions of exam'),
                'space_origin': tfds.features.Tensor(shape=(3,), 
                                                dtype=np.float64, 
                                                encoding='zlib', 
                                                doc='space origin of exam'),

          })

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
    archive_path = '/media/roberto/TOSHIBA EXT/pet_ct/santa_maria_data/'

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
          pet_liver_path = os.path.join(path, "PET Liver ROI", f'{patient_id}_pet_liver.nrrd')
        
          # Revisa si la imagen y el label existen antes de retornarlos
          if os.path.exists(image_file_path) and os.path.exists(label_file_path):
            data_exam, header = nrrd.read(image_file_path)
            mask_exam, _ = nrrd.read(label_file_path)
            # _ devuelve los metadatos, retornar, modificar código
            

            masked_liver_data = np.array([], dtype=np.float32)
            if self.builder_config.img_type == 'pet':
              pet_liver_mask, _ = nrrd.read(pet_liver_path)
              # Masking using boolean indexing
              
              masked_liver_data = np.where(pet_liver_mask >= 1, data_exam, 0)
              masked_liver_data = masked_liver_data.flatten()
                
            
            for i in range(data_exam.shape[2]):
              data_exam_i = data_exam[:,:,i].astype(np.float32)
              mask_exam_i = mask_exam[:,:,i].astype(np.int32)
                
              if np.max(mask_exam_i) > 0:
                data_exam_i = np.rot90(data_exam_i, k=3)
                data_exam_i = np.fliplr(data_exam_i)  
                    
                mask_exam_i = np.rot90(mask_exam_i, k=3)
                mask_exam_i = np.fliplr(mask_exam_i)
        
                # Create a unique key using the patient_id and the index of the loop
                example_key = f'{patient_id}_{i}'
        
                yield example_key, {
                  'patient_id': patient_id,
                  'img_exam': data_exam_i,
                  'mask_exam': mask_exam_i,
                  'egfr_label': 'Mutant' if row['EGFR'] == '1' else 'Wildtype',
                  'pet_liver': masked_liver_data,
                  'exam_metadata': {'space_directions': np.diag(header['space directions']),
                                    'space_origin': header['space origin']}
                }
