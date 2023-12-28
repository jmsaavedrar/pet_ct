import os
import csv
from pathlib import Path
from radiomics import featureextractor

# Parameters
archive_path = '/media/roberto/TOSHIBA EXT/pet_ct/santa_maria_data'
binwidth = 5
sigma = [1, 2, 3]
exam_types = ['body', 'pet', 'torax3d']
normalize = True

# Radiomics feature extractor
settings = {'binWidth': binwidth, 'sigma': sigma, 'normalize':normalize}
extractor = featureextractor.RadiomicsFeatureExtractor(**settings)

#extractor.enableImageTypeByName('Original')
#extractor.enableImageTypeByName('Wavelet')
#extractor.enableImageTypeByName('LoG')
#extractor.enableImageTypeByName('Square')
#extractor.enableImageTypeByName('SquareRoot')
#extractor.enableImageTypeByName('Logarithm')
#extractor.enableImageTypeByName('Exponential')
# Output file path
output_file_path = Path(f'santamaria_data_all__binwidth_{binwidth}_sigma_{sigma}_normalize_{normalize}.csv')

# Get fieldnames from feature map
image = Path(archive_path, 'EGFR+', 'body', "image", 'sm_001_body_image.nrrd')
mask = Path(archive_path, 'EGFR+', 'body', "label", 'sm_001_body_segmentation.nrrd')


features = extractor.execute(str(image), str(mask))
fieldnames = sorted(key for key in features)

# Get original column names from Santa Maria data
with open(Path(archive_path, 'santamaria_data.csv')) as csv_file:
    original_col_names = next(csv.DictReader(csv_file)).keys()

# Create new fieldnames for the final CSV
final_fieldnames = [f"{exam_type}_{key}" for exam_type in exam_types for key in fieldnames]
fieldnames = list(original_col_names) + final_fieldnames

# Read the original CSV and create a new one with additional columns
with output_file_path.open('w', newline='') as output_file:
    writer = csv.DictWriter(output_file, fieldnames=fieldnames)
    writer.writeheader()

    with open(Path(archive_path, 'santamaria_data.csv')) as data_file:
        reader = csv.DictReader(data_file)

        for row in reader:
            patient_id = row['PATIENT_ID']

            for exam_type in exam_types:
                exam_name = {'torax3d': '3D_TORAX_SEG', 'body': 'BODY_CT_SEG', 'pet': 'PET_SEG'}[exam_type]

                if row[exam_name] == '1':
                    results_folder = os.path.join(archive_path, "EGFR+" if row['EGFR'] == '1' else "EGFR-", exam_type)
                    image_file_path = os.path.join(results_folder, "image", f'{patient_id}_{exam_type}_image.nrrd')
                    label_file_path = os.path.join(results_folder, "label", f'{patient_id}_{exam_type}_segmentation.nrrd')

                    result = extractor.execute(str(image_file_path), str(label_file_path))

                    # Update the row with feature values
                    row.update({f"{exam_type}_{key}": value for key, value in result.items()})
            
            # Write the updated row to the new CSV file
            writer.writerow(row)

print("CSV file with features for all exam types created successfully at ", str(output_file_path))
