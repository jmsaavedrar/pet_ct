import tensorflow_datasets as tfds
import utils 
import random
import tensorflow as tf
import numpy as np
import PIL.Image as pimage

def get_training_testing_sm_datasets(split_val=0.8, random_seed=None):
    # split randomly the data using split_val    

    #reading data
    data_dir = '/mnt/hd-data/Datasets/medseg/pet_ct'
    sample_dataset, info = tfds.load('santa_maria_dataset/torax3d', with_info=True, data_dir = data_dir)
    
    random.seed(random_seed)

    # Define the positive and negative patients
    pos_patients = [f'sm_{str(i).zfill(3)}' for i in range(1, 13)]
    neg_patients = [f'sm_{str(i).zfill(3)}' for i in range(13, 36)]
    
    # Get the split keys (splits) of the dataset
    split_keys = list(info.splits.keys())

    # Find positive and negative patients that are also in the split keys
    pos_patients = [patient for patient in pos_patients if patient in split_keys]
    neg_patients = [patient for patient in neg_patients if patient in split_keys]
    
    # Shuffle the order of positive and negative patients for randomness
    random.shuffle(pos_patients)
    random.shuffle(neg_patients)
    
    # Calculate the number of patients for training and testing
    train_pos_count = int(split_val * len(pos_patients))
    train_neg_count = int(split_val * len(neg_patients))
    
    # Create the training and testing sets
    training_patients = pos_patients[:train_pos_count] + neg_patients[:train_neg_count]
    testing_patients = pos_patients[train_pos_count:] + neg_patients[train_neg_count:]
    
    # Create dictionaries to hold the training and testing data
    training_data = {patient: sample_dataset[patient] for patient in training_patients}
    testing_data = {patient: sample_dataset[patient] for patient in testing_patients}
    
    max_val = 1000
    min_val = -1000
    val_range = max_val - min_val
    padding = 1
    size = 256
    
    # Create a generator for the training dataset
    def generate_data(data_type):
        assert data_type in ['training', 'testing'], "error, incorrect data_type"        
        patient_list = training_patients
        patient_data = training_data
        if data_type == 'testing' :
            patient_list = testing_patients
            patient_data = testing_data

        for patient_id in patient_list:            
            for data in patient_data[patient_id]:
                mask_exam = data['mask_exam']
                img_exam = data['img_exam']
                img_exam = tf.where(img_exam < min_val, min_val, img_exam)
                img_exam = tf.where(img_exam > max_val, max_val, img_exam)
                #img_exam = (img_exam - min_val)/ val_range                 
                _, data_roi = utils.get_roi(img_exam, mask_exam, padding = padding, min_val = min_val)                           
                #print('{} {} {}'.format(np.min(data_roi), np.max(data_roi), data_roi.shape))
                data_roi = tf.expand_dims(data_roi, -1)
                imm = tf.image.resize(data_roi, (size, size))
                yield imm, data['label']

    # Create a TensorFlow Dataset from the generator
    training_dataset = tf.data.Dataset.from_generator(lambda : generate_data('training'), 
                                                      output_signature=(
                                                          tf.TensorSpec(shape=(size, size, 1), dtype=tf.float32, name="image"),
                                                          tf.TensorSpec(shape=(), dtype=tf.int64, name="label")))    
    testing_dataset = tf.data.Dataset.from_generator(lambda: generate_data('testing'), output_signature=(
                                                          tf.TensorSpec(shape=(size, size, 1), dtype=tf.float32, name="image"),
                                                          tf.TensorSpec(shape=(), dtype=tf.int64, name="label")))
        
    return training_dataset, testing_dataset