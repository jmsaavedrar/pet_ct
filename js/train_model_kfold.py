import tensorflow as tf
import tensorflow_datasets as tfds
import numpy as np
from sklearn.model_selection import KFold, StratifiedKFold
from tensorflow.keras.metrics import AUC, Precision, Recall
import argparse
import metrics
import models


max_val = 440
min_val = -360
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
    
    bounding_box = tf.clip_by_value(bounding_box, min_val, max_val)

    return bounding_box
    
    

def cargar_datos(dataset, img_type, n_splits=5, img_size=32, margin=5, batch_size=32, shuffle_buffer_size=1000, random_seed=None):
    # Cargar el conjunto de datos desde TensorFlow Datasets

    dataset, info =  tfds.load(f'{dataset}/{img_type}', with_info=True, data_dir='/media/roberto/TOSHIBA EXT/tensorflow_ds/')

    # Get the split keys (splits) of the dataset
    patients = list(info.splits.keys())

    skf = KFold(n_splits=n_splits, shuffle=True, random_state=random_seed)
    fold_datasets = []

    def generate_data(patient_ids):
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

    for i, (train_indices, test_indices) in enumerate(skf.split(patients)):
        training_patients = [patients[i] for i in train_indices]
        testing_patients = [patients[i] for i in test_indices]

        training_dataset = tf.data.Dataset.from_generator(
            lambda: generate_data(training_patients),
            output_signature=(
                tf.TensorSpec(shape=(img_size, img_size, 1), dtype=tf.float32, name="imagen"),
                tf.TensorSpec(shape=(), dtype=tf.int64, name="label")
            )
        )

        testing_dataset = tf.data.Dataset.from_generator(
            lambda: generate_data(testing_patients),
            output_signature=(
                tf.TensorSpec(shape=(img_size, img_size, 1), dtype=tf.float32, name="imagen"),
                tf.TensorSpec(shape=(), dtype=tf.int64, name="label")
            )
        )

        fold_datasets.append((training_dataset, testing_dataset))

    return fold_datasets

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
    parser.add_argument("-d", "--dataset", type=str, default="santa_maria_dataset", help="Conjunto de datos (santa_maria_dataset o stanford_dataset)")
    parser.add_argument("-p", "--particion", type=str, default="torax3d", help="Tipo de partición (pet, body o torax3d)")
    parser.add_argument("-b", "--batch", type=int, default=32, help="Tamaño del lote para entrenamiento")
    parser.add_argument("-s", "--size", type=int, default=32, help="Tamaño de la imagen para extracción de ROI")
    parser.add_argument("-e", "--epochs", type=int, default=10, help="Número de épocas para entrenamiento")
    parser.add_argument("-n", "--splits", type=int, default=5, help="Número de divisiones para KFold")
    parser.add_argument("--seed", type=int, default=None, help="Semilla aleatoria para reproducibilidad")

    args = parser.parse_args()
    
    ds_arg = args.dataset
    if ds_arg not in ['santa_maria_dataset', 'stanford_dataset']:
        raise ValueError('Conjunto de datos no válido')
    
    if ds_arg == 'santa_maria_dataset':
        if args.particion not in ['body', 'pet', 'torax3d']: raise ValueError('Partición no válida, debe ser body, pet o torax3d')
    
    if ds_arg == 'stanford_dataset':
        if args.particion not in ['pet', 'ct', 'chest_ct']: raise ValueError('Partición no válida, debe ser pet, ct o chest_ct')
    # Resuelve el nombre de la función a una función real
    #roi_fn = globals().get(args.roi_fn)

    #if not callable(roi_fn):
    #    raise ValueError(f"Función de extracción de ROI no válida: {args.roi_fn}")

    # Cargar datos
    k_fold_dataset = cargar_datos(
        args.dataset,
        args.particion,
        n_splits=args.splits,
        batch_size=args.batch,
        img_size=args.size,
        shuffle_buffer_size=1000,
        random_seed=args.seed,
    )

    train_steps = args.epochs * (1200 // args.batch)*4

    all_train_accuracies = []
    all_train_aucs = []
    all_train_true_positives = []
    all_train_false_positives = []

    all_test_accuracies = []
    all_test_aucs = []
    all_test_true_positives = []
    all_test_false_positives = []

    # Calculate mean for all folds
    for i, (train_ds, test_ds) in enumerate(k_fold_dataset):
        print(f"Fold {i}:")
        
        train_ds = train_ds.shuffle(1024).map(map_fun).batch(args.batch).prefetch(tf.data.experimental.AUTOTUNE)
        test_ds = test_ds.shuffle(1024).map(map_fun).batch(args.batch).prefetch(tf.data.experimental.AUTOTUNE)
    
    
        # Train the model
        modelo = construir_modelo(args.size, train_steps)
        
        # Entrenar el modelo
        early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)

        history = modelo.fit(train_ds, epochs=args.epochs, validation_data=test_ds, callbacks=[early_stopping])

        # Evaluate on training dataset
        train_accuracy, train_auc, train_precision, train_recall = modelo.evaluate(train_ds)[1:5]

        # Evaluate on testing dataset
        test_accuracy, test_auc, test_precision, test_recall = modelo.evaluate(test_ds)[1:5]

        # Replace NaN values with 0
        test_accuracy, test_auc, test_precision, test_recall = map(lambda x: 0 if np.isnan(x) else x,
                                                               [test_accuracy, test_auc, test_precision, test_recall])
                                                               
        
        # Append metrics to arrays
        all_train_accuracies.append(train_accuracy)
        all_train_aucs.append(train_auc)
        all_train_true_positives.append(train_precision)
        all_train_false_positives.append(train_recall)

        all_test_accuracies.append(test_accuracy)
        all_test_aucs.append(test_auc)
        all_test_true_positives.append(test_precision)
        all_test_false_positives.append(test_recall)
    

        # Print metrics for the current fold
        print(f"Training Metrics: Accuracy: {train_accuracy:.3f}, AUC: {train_auc:.3f}, Precision: {train_precision:.3f}, Recall: {train_recall:.3f}")
        print(f"Testing Metrics: Accuracy: {test_accuracy:.3f}, AUC: {test_auc:.3f}, Precision: {test_precision:.3f}, Recall: {test_recall:.3f}")
        print()

    # Calculate mean for all folds
    mean_train_accuracy = np.mean(all_train_accuracies)
    mean_train_auc = np.mean(all_train_aucs)
    mean_train_true_positive = np.mean(all_train_true_positives)
    mean_train_false_positive = np.mean(all_train_false_positives)

    mean_test_accuracy = np.mean(all_test_accuracies)
    mean_test_auc = np.mean(all_test_aucs)
    mean_test_true_positive = np.mean(all_test_true_positives)
    mean_test_false_positive = np.mean(all_test_false_positives)

# Print mean metrics
print(f"Mean Training Metrics: Accuracy: {mean_train_accuracy:.3f}, AUC: {mean_train_auc:.3f}, Precision: {mean_train_true_positive:.3f}, Recall: {mean_train_false_positive:.3f}")
print(f"Mean Testing Metrics: Accuracy: {mean_test_accuracy:.3f}, AUC: {mean_test_auc:.3f}, Precision: {mean_test_true_positive:.3f}, Recall: {mean_test_false_positive:.3f}")
