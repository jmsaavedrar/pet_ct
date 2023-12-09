import os
#import nrrd
import csv
import numpy as np
from pathlib import Path
from dataclasses import dataclass
#import nrrd
#import pydicom
from radiomics import featureextractor
#import SimpleITK as sitk

archive_path = '/media/roberto/TOSHIBA EXT/pet_ct/stanford_data/'

exam_types = ['chest_ct', 'pet', 'ct']
exam_type = 'chest_ct'
patient_id = 'R01-012'

exam_name = {'chest_ct': 'chest_ct_image', 'ct': 'ct_image', 'pet':'pet_image'}[exam_type]
segmentation_name = {'chest_ct': 'chest_ct_segmentation', 'ct': 'ct_segmentation', 'pet':'pet_segmentation'}[exam_type]

exam_results = os.path.join(archive_path, 'data', patient_id, exam_type)

print(exam_results)

image_file_path = os.path.join(exam_results, f'{patient_id}_{exam_type}_image.nrrd')
label_file_path = os.path.join(exam_results, f'{patient_id}_{exam_type}_segmentation.nrrd')

print(image_file_path)
print(label_file_path)

extractor = featureextractor.RadiomicsFeatureExtractor(geometryTolerance=0.1)
extractor2 = featureextractor.RadiomicsFeatureExtractor(geometryTolerance=0.1, label=255)

result = None
try:
    result = extractor.execute(str(image_file_path), str(label_file_path))
except Exception as e:
    print(f"Exception during extractor.execute(): {e}")
    result = extractor2.execute(str(image_file_path), str(label_file_path))

# image = sitk.ReadImage(str(image_file_path))
# mask = sitk.ReadImage(str(label_file_path))

# print("Image Spacing:", image.GetSpacing())
# print("Mask Spacing:", mask.GetSpacing())
# print(image.GetSpacing() == mask.GetSpacing())
# print()
# print("Image Origin:", image.GetOrigin())
# print("Mask Origin:", mask.GetOrigin())
# print(image.GetOrigin() == mask.GetOrigin())
# print()
# print("Image Direction:", image.GetDirection())
# print("Mask Direction:", mask.GetDirection())
# print(image.GetDirection() == mask.GetDirection() )

# import SimpleITK as sitk

# image = sitk.ReadImage(str(image_file_path))
# mask = sitk.ReadImage(str(label_file_path))

# # Resample the mask to match the origin of the image
# mask_resampled = sitk.Resample(mask, image, sitk.Transform(), sitk.sitkNearestNeighbor, 0.0, mask.GetPixelID())

# print(mask_resampled.GetOrigin())
# print(image.GetOrigin() == mask_resampled.GetOrigin())

# new_label_file_path = os.path.join(exam_results, f'{patient_id}_{exam_type}_segmentation_fix.nrrd')
# sitk.WriteImage(mask_resampled, new_label_file_path)




# extractor = featureextractor.RadiomicsFeatureExtractor()
# result = extractor.execute(str(image_file_path), str(new_label_file_path))

#data_exam, _ = nrrd.read(image_file_path)
#mask_exam, _ = nrrd.read(label_file_path)

#print(data_exam.shape, mask_exam.shape)
#print('unicos:', np.unique(mask_exam))
#print('range of values:', np.ptp(mask_exam))