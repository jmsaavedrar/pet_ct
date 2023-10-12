import tensorflow_datasets as tfds
import os
import nrrd
import csv
import numpy as np
import matplotlib.pyplot as plt


sample_dataset = tfds.load('santa_maria_pet_dataset/pet_1')
