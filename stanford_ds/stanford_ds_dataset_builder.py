"""stanford_ds dataset."""

import tensorflow_datasets as tfds
import os
import nrrd
import csv
import numpy as np
from tqdm import trange
from pathlib import Path
from dataclasses import dataclass
import pydicom


# roi - R01-001, R01-002, R01-014, 033, 052, 049, 042, 039
# 

@dataclass
class ExamConfig(tfds.core.BuilderConfig):
  img_type: str = 'ct'

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
                                          dtype=np.float32,
                                          encoding='zlib',
                                          doc = 'Exam Images'),
        'mask_exam': tfds.features.Tensor(shape=(None, None),
                                          dtype=np.int32,
                                          encoding='zlib',
                                          doc = 'Tumor Mask'),
        'egfr_label': tfds.features.ClassLabel(names=['Wildtype', 'Mutant', 'Unknown', 'Not collected'],
                                          doc='Results on the EGFR Mutation test: 1 is positive and 0 negative'),
        'kras_label': tfds.features.ClassLabel(names=['Wildtype', 'Mutant', 'Unknown', 'Not collected'],
                                          doc='Results on the KRAS Mutation test.'),
        'alk_label': tfds.features.ClassLabel(names=['Wildtype', 'Translocated', 'Unknown', 'Not collected'], 
                                          doc='Results on the ALK translocation test.'),                                  
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

    # Create dictionaries of patients with their associated data
    return {patient: self._generate_examples(archive_path, patient) for patient in final_patients} 

    
  
  def _generate_examples(self, path, patient_list):

    # Lee el archivo csv y retorna los ejemplos de un examen (pet, ct o torax) para una ventana (numero entero).
    data_file = Path(os.path.join(path, 'stanford_data_info.csv'))
    image_folder = os.path.join(path, 'data')
    
    dcm_patients = []
    mutation_status2int = {'Wildtype':0, 'Mutant':1, 'Unknown':2, 'Not collected': 3}
    translocation_status2int = {'Wildtype':0, 'Translocated':1, 'Unknown':2, 'Not collected': 3}
    
    with data_file.open() as f:
      for row in csv.DictReader(f):
        patient_id = row['Case ID']

        if patient_id == patient_list:
          label_value = mutation_status2int[row['EGFR mutation status']] 
          kras_label = mutation_status2int[row['KRAS mutation status']]
          alk_label = translocation_status2int[row['ALK translocation status']]
          
          exam_results = os.path.join(image_folder, patient_id, self.builder_config.img_type) # pytype: disable=attribute-error
          
          if os.path.exists(exam_results):
            
            image_file_path = os.path.join(exam_results, f'{patient_id}_{self.builder_config.img_type}_image.nrrd')
            label_base_path = os.path.join(exam_results, f'{patient_id}_{self.builder_config.img_type}_segmentation')
            pet_liver_path = os.path.join(image_folder, "Liver_pet", f'{patient_id}_pet_liver.nrrd')
            
            label_file_path = None
            mask_exam = None
            nrrd_mask = False
            
            if os.path.exists(label_base_path + '.nrrd'):
              label_file_path = label_base_path + '.nrrd'
              mask_exam, _ = nrrd.read(label_file_path)
              nrrd_mask = True
            elif os.path.exists(label_base_path + '.dcm'):
              dcm_patients.append(patient_id)
              
              label_file_path = label_base_path + '.dcm'
              mask_exam = pydicom.dcmread(label_file_path).pixel_array
              mask_exam = np.moveaxis(mask_exam, 0, 2)
            
            data_exam, header = nrrd.read(image_file_path)
            pet_liver_exam = pet_liver_exam = np.array([], dtype=np.uint16)
              
            
            masked_liver_data = np.array([], dtype=np.float32)
            if self.builder_config.img_type == 'pet':
              pet_liver_mask, _ = nrrd.read(pet_liver_path)
              # Masking using boolean indexing
              
              masked_liver_data = np.where(pet_liver_mask >= 1, data_exam, 0)
              masked_liver_data = masked_liver_data.flatten().astype(np.float32)
              print('type of masked liver data:', masked_liver_data.dtype)

            for i in range(data_exam.shape[2]):
              data_exam_i = data_exam[:,:,i].astype(np.float32)
              mask_exam_i = mask_exam[:,:,i].astype(np.int32)
              
              if np.max(mask_exam_i) > 0:
                data_exam_i = np.rot90(data_exam_i, k=3)
                data_exam_i = np.fliplr(data_exam_i)
               
                if nrrd_mask:
                  # nrrd files
                  mask_exam_i = np.rot90(mask_exam_i, k=3)
                  mask_exam_i = np.fliplr(mask_exam_i)

    
                # Create a unique key using the patient_id and the index of the loop
                example_key = f'{patient_id}_{i}'

                yield example_key, {
                  'patient_id': patient_id,
                  'img_exam': data_exam_i,
                  'mask_exam': mask_exam_i,
                  'egfr_label': label_value,
                  'kras_label': kras_label,
                  'alk_label': alk_label,
                  'pet_liver': masked_liver_data,
                  'exam_metadata': {'space_directions': np.diag(header['space directions']),
                                    'space_origin': header['space origin']}
                
                }
                  
