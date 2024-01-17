import tensorflow as tf
import tensorflow_datasets as tfds
import numpy as np
from sklearn.model_selection import KFold
from tensorflow.keras.metrics import AUC, Precision, Recall
import argparse
import metrics
import models
import random

max_val = 1000
min_val = -1000
val_range = max_val - min_val
margin = 3
initial_learning_rate = 0.001
alpha = 0.00001


def map_fun(image, label) :        
    crop_size = 256    
    image = (image - min_val)/ val_range
    image = tf.image.grayscale_to_rgb(image)
    #size = int(crop_size * 1.15)
    #image = tf.image.resize_with_pad(image, size, size)
    #image = tf.image.random_crop(image, (crop_size, crop_size,3))
    #image = tf.image.random_flip_left_right(image)
    return image, label


def extract_roi(image, mask, margin):
    # Convert mask to boolean tensor
    mask = tf.cast(mask, dtype=tf.bool)

    # Find indices where mask equals 1
    indices = tf.where(mask)
    
    #image = tf.where(mask, image, min_val)

    # Get the minimum and maximum indices along each axis
    min_row = tf.reduce_min(indices[:, 0])
    min_col = tf.reduce_min(indices[:, 1])
    max_row = tf.reduce_max(indices[:, 0])
    max_col = tf.reduce_max(indices[:, 1])

    # Extract the bounding box from the image
    bounding_box = image[min_row-margin:max_row + 1+margin, min_col-margin:max_col + 1+margin]

    return bounding_box
    
    

def cargar_datos(img_type, img_type_sm, n_splits=5, img_size=32, margin=5, batch_size=32, shuffle_buffer_size=1000, random_seed=None):
    # Cargar el conjunto de datos desde TensorFlow Datasets
    stanford_dataset, stanford_info =  tfds.load(f'stanford_dataset/{img_type}', with_info=True, data_dir='/media/roberto/TOSHIBA EXT/tensorflow_ds/')
    santa_maria_dataset, santa_maria_info =  tfds.load(f'santa_maria_dataset/{img_type_sm}', with_info=True, data_dir='/media/roberto/TOSHIBA EXT/tensorflow_ds/')

    # Get the split keys (splits) of the dataset
    stanford_patients = list(stanford_info.splits.keys())
    santa_maria_patients = list(santa_maria_info.splits.keys())

    def generate_data(dataset, patient_ids):
        for patient_id in patient_ids:
            patient_data = dataset[patient_id]
            for data in patient_data:
                if data['egfr_label'] < 2:
                    mask_exam = data['mask_exam']
                    img_exam = data['img_exam']
                
                    # roi value to standarize the image slice
                    if img_type == 'pet':
                        liver_roi_val = tf.cast(tf.reduce_mean(data['pet_liver']), dtype=tf.float32)
                        img_exam  = img_exam / liver_roi_val
                    
                    #img_exam = tf.where(img_exam < min_val, min_val, img_exam)
                    #img_exam = tf.where(img_exam > max_val, max_val, img_exam)
                    #img_exam = (img_exam - min_val)/ val_range                 
                    data_roi = extract_roi(img_exam, mask_exam, margin)                           
                    #print('{} {} {}'.format(np.min(data_roi), np.max(data_roi), data_roi.shape))
                    data_roi = tf.expand_dims(data_roi, -1)
                    imm = tf.image.resize(data_roi, (img_size, img_size))
                    yield imm, data['egfr_label']

    random.shuffle(stanford_patients)
    # Calculate the index for the 80/20 split
    split_index = int(0.8 * len(stanford_patients))

    # Split the list into training and validation sets
    stanford_train_patients = stanford_patients[:split_index]
    stanford_val_patients = stanford_patients[split_index:]

    stanford_train_data = tf.data.Dataset.from_generator(
	lambda: generate_data(stanford_dataset, stanford_train_patients),
	output_signature=(
	    tf.TensorSpec(shape=(img_size, img_size, 1), dtype=tf.float32, name="imagen"),
	    tf.TensorSpec(shape=(), dtype=tf.int64, name="label")
	)
    )

    stanford_val_data = tf.data.Dataset.from_generator(
	lambda: generate_data(stanford_dataset, stanford_val_patients),
	output_signature=(
	    tf.TensorSpec(shape=(img_size, img_size, 1), dtype=tf.float32, name="imagen"),
	    tf.TensorSpec(shape=(), dtype=tf.int64, name="label")
	)
    )

    
    santa_maria_data = tf.data.Dataset.from_generator(
	lambda: generate_data(santa_maria_dataset, santa_maria_patients),
	output_signature=(
	    tf.TensorSpec(shape=(img_size, img_size, 1), dtype=tf.float32, name="imagen"),
	    tf.TensorSpec(shape=(), dtype=tf.int64, name="label")
	)
    )

	

    return stanford_train_data, stanford_val_data, santa_maria_data

def construir_modelo(img_size, train_steps):
    modelo = models.simple_model_v3((img_size, img_size, 3))
    cosdecay = tf.keras.optimizers.schedules.CosineDecay(initial_learning_rate, decay_steps  = train_steps, alpha = alpha)
    optimizer=tf.keras.optimizers.Adam(learning_rate = cosdecay)

    modelo.compile(
        optimizer=optimizer,
        loss=tf.keras.losses.BinaryFocalCrossentropy(gamma=6),
        metrics=['accuracy', AUC(name='auc', curve='PR'), metrics.precision, metrics.recall]
    )
    return modelo



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Entrenar un modelo en el conjunto de datos de Santa Maria.")
    parser.add_argument("-p", "--particion", type=str, default="chest_ct", help="Tipo de partición (pet, body o torax3d)")
    parser.add_argument("-b", "--batch", type=int, default=32, help="Tamaño del lote para entrenamiento")
    parser.add_argument("-s", "--size", type=int, default=32, help="Tamaño de la imagen para extracción de ROI")
    parser.add_argument("-e", "--epochs", type=int, default=10, help="Número de épocas para entrenamiento")
    parser.add_argument("--seed", type=int, default=None, help="Semilla aleatoria para reproducibilidad")

    args = parser.parse_args()
    
    if args.particion not in ['pet', 'ct', 'chest_ct']: raise ValueError('Partición no válida, debe ser pet, ct o chest_ct')
    
    
    sm_particion = {'pet': 'pet', 'ct': 'body', 'chest_ct':'torax3d'}[args.particion]
    
    # Cargar datos
    train_val_test_dataset = cargar_datos(
        args.particion,
        sm_particion,
        batch_size=args.batch,
        img_size=args.size,
        shuffle_buffer_size=1000,
        random_seed=args.seed,
    )

    train_steps = args.epochs * (1200 // args.batch) * 4

    train_ds, val_ds, test_ds = train_val_test_dataset
    
    train_ds = train_ds.shuffle(1024).map(map_fun).batch(args.batch).prefetch(tf.data.experimental.AUTOTUNE)
    val_ds = val_ds.shuffle(1024).map(map_fun).batch(args.batch).prefetch(tf.data.experimental.AUTOTUNE)
    test_ds = test_ds.shuffle(1024).map(map_fun).batch(args.batch).prefetch(tf.data.experimental.AUTOTUNE)
    
    ## Construir el modelo
    modelo = construir_modelo(args.size, train_steps)

    # Entrenar el modelo
    early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)
    history = modelo.fit(train_ds, epochs=args.epochs, validation_data=test_ds, callbacks=[early_stopping])

    # Evaluate on training dataset
    train_accuracy, train_auc, train_precision, train_recall = modelo.evaluate(train_ds)[1:5]

    # Evaluate on validation dataset
    val_accuracy, val_auc, val_precision, val_recall = modelo.evaluate(val_ds)[1:5]

    # Evaluate on testing dataset
    test_accuracy, test_auc, test_precision, test_recall = modelo.evaluate(test_ds)[1:5]

    # Replace NaN values with 0
    test_accuracy, test_auc, test_precision, test_recall = map(lambda x: 0 if np.isnan(x) else x,
                                                           [test_accuracy, test_auc, test_precision, test_recall])

    # Print metrics
    print("Training Metrics:")
    print(f"Accuracy: {train_accuracy:.3f}, AUC: {train_auc:.3f}, Precision: {train_precision:.3f}, Recall: {train_recall:.3f}")
    print("-------------------------------------------")

    print("Validation Metrics:")
    print(f"Accuracy: {val_accuracy:.3f}, AUC: {val_auc:.3f}, Precision: {val_precision:.3f}, Recall: {val_recall:.3f}")
    print("-------------------------------------------")

    print("Testing Metrics:")
    print(f"Accuracy: {test_accuracy:.3f}, AUC: {test_auc:.3f}, Precision: {test_precision:.3f}, Recall: {test_recall:.3f}")

