import tensorflow as tf
import tensorflow_datasets as tfds
import numpy as np
from sklearn.model_selection import KFold
from simple import SimpleModel2
from tensorflow.keras.metrics import AUC, Precision, Recall
import argparse
import metrics
import models


max_val = 1000
min_val = -1000
val_range = max_val - min_val
margin = 3
initial_learning_rate = 0.001
epochs = 20
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
    stanford_dataset, stanford_info =  tfds.load(f'stanford_dataset/{img_type}', with_info=True)
    santa_maria_dataset, santa_maria_info =  tfds.load(f'santa_maria_dataset/{img_type_sm}', with_info=True)

    # Get the split keys (splits) of the dataset
    stanford_patients = list(stanford_info.splits.keys())
    santa_maria_patients = list(santa_maria_info.splits.keys())

    def generate_data(dataset, patient_ids):
        for patient_id in patient_ids:
            patient_data = dataset[patient_id]
            for data in patient_data:
                mask_exam = data['mask_exam']
                img_exam = data['img_exam']
                
                # roi value to standarize the image slice
                if img_type == 'pet':
                    liver_roi_val = tf.cast(tf.reduce_sum(data['pet_liver']), dtype=tf.float32)/619.0
                    if liver_roi_val == 0: liver_roi_val = 1
                    img_exam  = img_exam / liver_roi_val
                #img_exam = tf.where(img_exam < min_val, min_val, img_exam)
                #img_exam = tf.where(img_exam > max_val, max_val, img_exam)
                #img_exam = (img_exam - min_val)/ val_range                 
                data_roi = extract_roi(img_exam, mask_exam, margin)                           
                #print('{} {} {}'.format(np.min(data_roi), np.max(data_roi), data_roi.shape))
                data_roi = tf.expand_dims(data_roi, -1)
                imm = tf.image.resize(data_roi, (img_size, img_size))
                yield imm, data['label']
        
    stanford_data = tf.data.Dataset.from_generator(
	lambda: generate_data(stanford_dataset, stanford_patients),
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

	

    return stanford_data, santa_maria_data

def construir_modelo(img_size, train_steps):
    modelo = models.simple_model((img_size, img_size, 3))
    cosdecay = tf.keras.optimizers.schedules.CosineDecay(initial_learning_rate, decay_steps  = train_steps, alpha = alpha)
    #optimizer=tf.keras.optimizers.AdamW(learning_rate = cosdecay)
    #optimizer=tf.keras.optimizers.SGD(learning_rate = cosdecay, momentum = 0.9)
    optimizer=tf.keras.optimizers.Adam(learning_rate = cosdecay)

    modelo.compile(
        optimizer=optimizer,
        loss=tf.keras.losses.BinaryFocalCrossentropy(), 
        metrics=['accuracy', AUC(name='auc', curve='PR'), metrics.true_positive, metrics.false_positive]
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
    train_test_dataset = cargar_datos(
        args.particion,
        sm_particion,
        batch_size=args.batch,
        img_size=args.size,
        shuffle_buffer_size=1000,
        random_seed=args.seed,
    )

    train_steps = args.epochs * (1200 // args.batch)

    train_ds, test_ds = train_test_dataset
    
    train_ds = train_ds.shuffle(1024).map(map_fun).batch(args.batch).prefetch(tf.data.experimental.AUTOTUNE)
    test_ds = test_ds.shuffle(1024).map(map_fun).batch(args.batch).prefetch(tf.data.experimental.AUTOTUNE)
    
    # Construir el modelo
    modelo = construir_modelo(args.size, train_steps)

    # Entrenar el modelo
    history = modelo.fit(train_ds, epochs=args.epochs, validation_data=test_ds)

    # Save metrics for training dataset
    train_metrics = modelo.evaluate(train_ds)
    train_accuracy, train_auc, train_true_positive, train_false_positive = train_metrics[1:5]

    # Save metrics for testing dataset
    test_metrics = modelo.evaluate(test_ds)
    test_accuracy, test_auc, test_true_positive, test_false_positive = test_metrics[1:5]

    # Replace NaN values with 0
    test_accuracy = 0 if np.isnan(test_accuracy) else test_accuracy
    test_auc = 0 if np.isnan(test_auc) else test_auc
    test_true_positive = 0 if np.isnan(test_true_positive) else test_true_positive
    test_false_positive = 0 if np.isnan(test_false_positive) else test_false_positive
    
     # Print test metrics
    print("Resultados")
    print()
    print(f"Train Accuracy: {train_accuracy}")
    print(f"Train AUC: {train_auc}")
    print(f"Train True Positive: {train_true_positive}")
    print(f"Train False Positive: {train_false_positive}")
    print("-------------------------------------------")
    print(f"Test Accuracy: {test_accuracy}")
    print(f"Test AUC: {test_auc}")
    print(f"Test True Positive: {test_true_positive}")
    print(f"Test False Positive: {test_false_positive}")

