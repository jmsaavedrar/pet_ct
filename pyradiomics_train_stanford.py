import os
import csv
from pathlib import Path
from radiomics import featureextractor

# Parameters
archive_path = '/media/roberto/TOSHIBA EXT/pet_ct/stanford_data/'
binwidth = 20
sigma = [1, 2, 3]
exam_types = ['chest_ct', 'pet', 'ct']

# Radiomics feature extractor
settings = {'binWidth': binwidth, 'sigma': sigma}
extractor = featureextractor.RadiomicsFeatureExtractor(**settings, geometryTolerance=0.1)
extractor2 = featureextractor.RadiomicsFeatureExtractor(**settings, geometryTolerance=0.1, label = 255)

# Output file path
output_file_path = Path(f'stanford_data_info_all__binwidth_{binwidth}_sigma_{sigma}.csv')

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

            for exam_type in exam_types:
                exam_name = {'chest_ct': 'chest_ct_image', 'ct': 'ct_image', 'pet':'pet_image'}[exam_type]
                segmentation_name = {'chest_ct': 'chest_ct_segmentation', 'ct': 'ct_segmentation', 'pet':'pet_segmentation'}[exam_type]

                
                if row[exam_name] == '1' and row[segmentation_name] == '1' and patient_id != 'R01-036' and patient_id != 'R01-058':
                    exam_results = os.path.join(archive_path, 'data', patient_id, exam_type)
                    image_file_path = os.path.join(exam_results, f'{patient_id}_{exam_type}_image.nrrd')
                    label_file_path = os.path.join(exam_results, f'{patient_id}_{exam_type}_segmentation.nrrd')
                    print(image_file_path)
                    print(label_file_path)
                    print()

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

print("CSV file with features for all exam types created successfully.")
