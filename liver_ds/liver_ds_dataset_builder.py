"""liver_ds dataset."""


import tensorflow_datasets as tfds
import os
import csv
import random
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from tqdm import trange
from pathlib import Path
from dataclasses import dataclass
import nibabel as nib

class Builder(tfds.core.GeneratorBasedBuilder):
    """DatasetBuilder for liver_ds dataset."""

    VERSION = tfds.core.Version('1.0.0')
    RELEASE_NOTES = {
        '1.0.0': 'Initial release.',
    }

    def _info(self) -> tfds.core.DatasetInfo:
        """Returns the dataset metadata."""
        # TODO(liver_ds): Specifies the tfds.core.DatasetInfo object
        return self.dataset_info_from_configs(
            features=tfds.features.FeaturesDict({
                # features of the dataset
                'image': tfds.features.Tensor(shape=(None, None),
                                              dtype=np.float64,
                                              encoding='zlib',
                                              doc='Exam Images'),
                'mask': tfds.features.Tensor(shape=(None, None),
                                             dtype=np.float64,
                                             encoding='zlib',
                                             doc='Tumor Mask'),
            }),
            # If there's a common (input, target) tuple from the
            # features, specify them here. They'll be used if
            # `as_supervised=True` in `builder.as_dataset`.
            supervised_keys=None,
        )

    def _split_generators(self, dl_manager: tfds.download.DownloadManager):
        """Returns SplitGenerators."""
        # Carpeta donde la data se encuentra
        archive_path = '/media/roberto/TOSHIBA EXT/liver_images/'

        # Get a list of all files in the folder
        all_files = os.listdir(archive_path)

        # Initialize an empty list to store the 'x' values
        x_values = []

        # Iterate through all files
        for file_name in all_files:
            if file_name.startswith("volume-") and file_name.endswith(".nii"):
                x_value = file_name.split("-")[1].split(".")[0]
                segmentation_file = f"segmentation-{x_value}.nii"

                # Check if both volume and segmentation files exist
                if segmentation_file in all_files:
                    x_values.append(x_value)

        # Create dictionaries of patients with their associated data
        return {patient: self._generate_examples(archive_path, patient) for patient in x_values}

    def _generate_examples(self, path, patient):
        """Yields examples."""
        # TODO(liver_ds): Yields (key, example) tuples from the dataset

        image_file_path = os.path.join(path, f'volume-{patient}.nii')
        seg_file_path = os.path.join(path, f'segmentation-{patient}.nii')

        data_exam = nib.load(image_file_path).get_fdata()
        seg_exam = nib.load(seg_file_path).get_fdata()
        
        for i in range(data_exam.shape[2]):
            data_exam_i = data_exam[:, :, i].astype(np.float64)
            seg_exam_i = seg_exam[:, :, i].astype(np.float64)
            
            
	    # Create a unique key using the patient_id and the index of the loop
            example_key = f'patient_{patient}_{i}'
            yield example_key, {
                'image': data_exam_i,
                'mask': seg_exam_i,
            }

