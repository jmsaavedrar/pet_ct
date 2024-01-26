import os
import csv
from pathlib import Path
from radiomics import featureextractor
import SimpleITK as sitk

# Parameters
archive_path = '/media/roberto/TOSHIBA EXT/pet_ct/stanford_data/'
binwidth = 20
sigma = [1, 2, 3]
exam_types = ['body', 'pet', 'torax3d']
exam_types_stanford = ['ct', 'pet', 'torax3d']
normalize = True
interpolator = sitk.sitkBSpline
resampledPixelSpacing = [1,1,1]

# Radiomics feature extractor
settings = {'binWidth': binwidth, 'sigma': sigma, 'normalize':normalize, 'interpolator': interpolator, 
            'resampledPixelSpacing':resampledPixelSpacing}

extractor = featureextractor.RadiomicsFeatureExtractor(**settings, geometryTolerance=0.1)
extractor2 = featureextractor.RadiomicsFeatureExtractor(**settings, geometryTolerance=0.1, label = 255)

# By default, only original is enabled. Optionally enable some image types:
extractor.enableImageTypes(Original={}, LoG={}, Wavelet={}, Square={}, SquareRoot={}, Logarithm={}, Exponential={})
extractor2.enableImageTypes(Original={}, LoG={}, Wavelet={}, Square={}, SquareRoot={}, Logarithm={}, Exponential={})

# Output file path
output_file_path = Path(f'stanford_data_info_all__binwidth_{binwidth}_sigma_{sigma}_changed.csv')
print('Se guarda en:', output_file_path)

# Get fieldnames from feature map
image = Path(archive_path, 'data', 'AMC-001', "ct", 'AMC-001_ct_image.nrrd')
mask = Path(archive_path, 'data', 'AMC-001', "ct", 'AMC-001_ct_segmentation.nrrd')

print(image, mask)
features = extractor.execute(str(image), str(mask))
fieldnames = sorted(key for key in features)

# Get original column names from Santa Maria data
with open(Path(archive_path, 'stanford_data_info.csv')) as csv_file:
    original_col_names = next(csv.DictReader(csv_file)).keys()

# Create new fieldnames for the final CSV
final_fieldnames = [f"{exam_type}_{key}" for exam_type in exam_types for key in fieldnames]
fieldnames = list(original_col_names) + final_fieldnames

# Read the original CSV and create a new one with additional columns
with output_file_path.open('w', newline='') as output_file:
    writer = csv.DictWriter(output_file, fieldnames=fieldnames)
    writer.writeheader()

    with open(Path(archive_path, 'stanford_data_info.csv')) as data_file:
        reader = csv.DictReader(data_file)

        for row in reader:
            patient_id = row['Case ID']

            for exam_type_i in range(len(exam_types)):
                exam_type = exam_types[exam_type_i]
                exam_type_stanford = exam_types_stanford[exam_type_i]
                
                exam_name = {'torax3d': 'chest_ct_image', 'body': 'ct_image', 'pet':'pet_image'}[exam_type]
                segmentation_name = {'torax3d': 'chest_ct_segmentation', 'body': 'ct_segmentation', 'pet':'pet_segmentation'}[exam_type]

                if row[exam_name] == '1' and row[segmentation_name] == '1' and patient_id != 'R01-036' and patient_id != 'R01-058':
                    exam_results = os.path.join(archive_path, 'data', patient_id, exam_type_stanford)
                    image_file_path = os.path.join(exam_results, f'{patient_id}_{exam_type_stanford}_image.nrrd')
                    label_file_path = os.path.join(exam_results, f'{patient_id}_{exam_type_stanford}_segmentation.nrrd')

                    if os.path.exists(label_file_path):
                        result = None
                        try:
                            result = extractor.execute(str(image_file_path), str(label_file_path))
                        except Exception as e:
                            print(f"Exception during extractor.execute(): {e}")
                            result = extractor2.execute(str(image_file_path), str(label_file_path))
                        row.update({f"{exam_type}_{key}": value for key, value in result.items()})
            
            # Write the updated row to the new CSV file
            writer.writerow(row)

print("CSV file with features for all exam types created successfully at: ", str(output_file_path))
