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
    
    

def cargar_datos(dataset, img_type, n_splits=5, img_size=32, margin=5, batch_size=32, shuffle_buffer_size=1000, random_seed=None):
    # Cargar el conjunto de datos desde TensorFlow Datasets
    dataset, info =  tfds.load(f'{dataset}/{img_type}', with_info=True)

    # Get the split keys (splits) of the dataset
    patients = list(info.splits.keys())

    skf = KFold(n_splits=n_splits, shuffle=True, random_state=random_seed)
    fold_datasets = []

    def generate_data(patient_ids):
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
    parser.add_argument("-d", "--dataset", type=str, default="santa_maria_dataset", help="Conjunto de daatos (santa_maria_dataset o stanford_dataset)")
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

    train_steps = args.epochs * (1200 // args.batch)

    all_train_accuracies = []
    all_train_aucs = []
    all_train_true_positives = []
    all_train_false_positives = []

    all_test_accuracies = []
    all_test_aucs = []
    all_test_true_positives = []
    all_test_false_positives = []

    for i, (train_ds, test_ds) in enumerate(k_fold_dataset):
        print(f"Fold: {i}")

        train_ds = train_ds.shuffle(1024).map(map_fun).batch(args.batch).prefetch(tf.data.experimental.AUTOTUNE)
        test_ds = test_ds.shuffle(1024).map(map_fun).batch(args.batch).prefetch(tf.data.experimental.AUTOTUNE)

        # Construir el modelo
        modelo = construir_modelo(args.size, train_steps)

        # Entrenar el modelo
        history = modelo.fit(train_ds, epochs=args.epochs, validation_data=test_ds)

        # Save metrics for training dataset
        train_metrics = modelo.evaluate(train_ds)
        train_accuracy, train_auc, train_true_positive, train_false_positive = train_metrics[1:5]
        all_train_accuracies.append(train_accuracy)
        all_train_aucs.append(train_auc)
        all_train_true_positives.append(train_true_positive)
        all_train_false_positives.append(train_false_positive)

        # Save metrics for testing dataset
        test_metrics = modelo.evaluate(test_ds)
        test_accuracy, test_auc, test_true_positive, test_false_positive = test_metrics[1:5]
        
        # Replace NaN values with 0
        test_accuracy = 0 if np.isnan(test_accuracy) else test_accuracy
        test_auc = 0 if np.isnan(test_auc) else test_auc
        test_true_positive = 0 if np.isnan(test_true_positive) else test_true_positive
        test_false_positive = 0 if np.isnan(test_false_positive) else test_false_positive

        all_test_accuracies.append(test_accuracy)
        all_test_aucs.append(test_auc)
        all_test_true_positives.append(test_true_positive)
        all_test_false_positives.append(test_false_positive)

    # Calculate mean for all Folds
    for i in range(len(all_test_accuracies)):
        print(f"Fold {i} - Training Accuracy: {all_train_accuracies[i]:.3f}, \
            Training AUC: {all_train_aucs[i]:.3f}, \
            Training True positives: {all_train_true_positives[i]:.3f}, \
            Training False Positives: {all_train_false_positives[i]:.3f}")
        print()
        print(f"Fold {i} - Test Accuracy: {all_test_accuracies[i]:.3f}, \
            Test AUC: {all_test_aucs[i]:.3f}, \
            Test True positives: {all_test_true_positives[i]:.3f}, \
            Test False Positives: {all_test_false_positives[i]:.3f}")
        print()

    # Calculate mean for all Folds
    mean_train_accuracy = np.mean(all_train_accuracies)
    mean_train_auc = np.mean(all_train_aucs)
    mean_train_true_positive = np.mean(all_train_true_positives)
    mean_train_false_positive = np.mean(all_train_false_positives)

    mean_test_accuracy = np.mean(all_test_accuracies)
    mean_test_auc = np.mean(all_test_aucs)
    mean_test_true_positive = np.mean(all_test_true_positives)
    mean_test_false_positive = np.mean(all_test_false_positives)

    print()
    print(f"Mean Training Accuracy: {mean_train_accuracy:.3f}")
    print(f"Mean Training AUC: {mean_train_auc:.3f}")
    print(f"Mean Training True Positive: {mean_train_true_positive:.3f}")
    print(f"Mean Training False Positive: {mean_train_false_positive:.3f}")

    print()
    print(f"Mean Test Accuracy: {mean_test_accuracy:.3f}")
    print(f"Mean Test AUC: {mean_test_auc:.3f}")
    print(f"Mean Test True Positive: {mean_test_true_positive:.3f}")
    print(f"Mean Test False Positive: {mean_test_false_positive:.3f}")
