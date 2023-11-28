import tensorflow as tf
import tensorflow_datasets as tfds
import numpy as np
from sklearn.model_selection import KFold
from simple import SimpleModel2
from tensorflow.keras.metrics import AUC, Precision, Recall
import argparse

def roiExtractionSize1(img, mask, total_size=None, margin=3):
    """
    Function to extract ROIs from images while ensuring a consistent total size for all ROIs.

    INPUT:
    img: Numpy array of images.
    mask: Numpy array of masks.
    total_size: The desired total size (width and height) of the extracted ROIs.

    OUTPUT: Numpy array containing the ROIs.
    """
    
    img_instance, mask_instance = img.numpy(), mask.numpy()
    index = np.where(mask_instance)

    if total_size == None:
        roi = img_instance[np.unique(index[0])[0]-margin:np.unique(index[0])[-1]+margin, np.unique(index[1])[0]-margin: np.unique(index[1])[-1]+margin]
    else:
    
        # Calculate the center of the mask.
        center_row, center_col = int(np.mean(index[0])), int(np.mean(index[1]))
    
        # Calculate the size of the ROI based on the total size.
        half_size = total_size // 2
    
        # Determine ROI boundaries with the margin.
        min_row = max(0, center_row - half_size)
        max_row = min(mask_instance.shape[0], center_row + half_size)
        min_col = max(0, center_col - half_size)
        max_col = min(mask_instance.shape[1], center_col + half_size)
    
        # Calculate the width and height of the ROI.
        roi_height = max_row - min_row
        roi_width = max_col - min_col
    
        # Case 1: If the ROI is smaller than the total_size, add a margin to make it total_size.
        if roi_height < total_size:
            margin = (total_size - roi_height) // 2
            min_row -= margin
            max_row += margin
    
        if roi_width < total_size:
            margin = (total_size - roi_width) // 2
            min_col -= margin
            max_col += margin
    
        # Case 2: If the ROI is larger than total_size, resize it.
        if roi_height > total_size or roi_width > total_size:
            scale_factor = total_size / max(roi_height, roi_width)
            new_height = int(roi_height * scale_factor)
            new_width = int(roi_width * scale_factor)
            min_row = max(center_row - new_height // 2, 0)
            max_row = min(min_row + new_height, mask_instance.shape[0])
            min_col = max(center_col - new_width // 2, 0)
            max_col = min(min_col + new_width, mask_instance.shape[1])
    
        # Extract the ROI with the desired size.
        roi = img_instance[min_row:max_row, min_col:max_col]
    roi = roi[:,:,np.newaxis]
    return roi


def roiExtractionSize(img, mask, total_size=None, margin=3):
    """
    Function to extract ROIs from images while ensuring a consistent total size for all ROIs.

    INPUT:
    img: Numpy array of images.
    mask: Numpy array of masks.
    total_size: The desired total size (width and height) of the extracted ROIs.

    OUTPUT: Numpy array containing the ROIs.
    """
    
    img_instance, mask_instance = img.numpy(), mask.numpy()
    index = np.where(mask_instance)

    if total_size == None:
        roi = img_instance[np.unique(index[0])[0]-margin:np.unique(index[0])[-1]+margin, np.unique(index[1])[0]-margin: np.unique(index[1])[-1]+margin]
    else:
    
        # Calculate the center of the mask.
        center_row, center_col = int(np.mean(index[0])), int(np.mean(index[1]))
    
        # Calculate the size of the ROI based on the total size.
        half_size = total_size // 2
    
        # Determine ROI boundaries with the margin.
        min_row = max(0, center_row - half_size)
        max_row = min(mask_instance.shape[0], center_row + half_size)
        min_col = max(0, center_col - half_size)
        max_col = min(mask_instance.shape[1], center_col + half_size)
    
        # Calculate the width and height of the ROI.
        roi_height = max_row - min_row
        roi_width = max_col - min_col
    
        # Case 1: If the ROI is smaller than the total_size, add a margin to make it total_size.
        if roi_height < total_size:
            margin = (total_size - roi_height) // 2
            min_row -= margin
            max_row += margin
    
        if roi_width < total_size:
            margin = (total_size - roi_width) // 2
            min_col -= margin
            max_col += margin
    
        # Case 2: If the ROI is larger than total_size, resize it.
        if roi_height > total_size or roi_width > total_size:
            scale_factor = total_size / max(roi_height, roi_width)
            new_height = int(roi_height * scale_factor)
            new_width = int(roi_width * scale_factor)
            min_row = max(center_row - new_height // 2, 0)
            max_row = min(min_row + new_height, mask_instance.shape[0])
            min_col = max(center_col - new_width // 2, 0)
            max_col = min(min_col + new_width, mask_instance.shape[1])
    
        # Extract the ROI with the desired size.
        roi = img_instance[min_row:max_row, min_col:max_col]
    
    return roi



def roiExtractionSize(img, mask, total_size=None, margin=3):
    """
    Function to extract ROIs from images while ensuring a consistent total size for all ROIs.

    INPUT:
    img: Numpy array of images.
    mask: Numpy array of masks.
    total_size: The desired total size (width and height) of the extracted ROIs.

    OUTPUT: Numpy array containing the ROIs.
    """
    
    img_instance, mask_instance = img.numpy(), mask.numpy()
    index = np.where(mask_instance)

    if total_size == None:
        roi = img_instance[np.unique(index[0])[0]-margin:np.unique(index[0])[-1]+margin, np.unique(index[1])[0]-margin: np.unique(index[1])[-1]+margin]

    return roi
    
    

def cargar_datos_santamaria(img_type, roiExtractFn, n_splits=5, img_size=32, margin=5, batch_size=32, shuffle_buffer_size=1000, random_seed=None):
    # Cargar el conjunto de datos desde TensorFlow Datasets
    dataset, info =  tfds.load(f'santa_maria_dataset/{img_type}', with_info=True)

    # Get the split keys (splits) of the dataset
    patients = list(info.splits.keys())

    skf = KFold(n_splits=n_splits, shuffle=True, random_state=random_seed)
    fold_datasets = []

    def generate_data(patient_ids):
        for patient_id in patient_ids:
            patient_data = dataset[patient_id]
            for data in patient_data:
                yield roiExtractFn(data['img_exam'], data['mask_exam'], img_size, margin), data['label']

    for i, (train_indices, test_indices) in enumerate(skf.split(patients)):
        training_patients = [patients[i] for i in train_indices]
        testing_patients = [patients[i] for i in test_indices]

        training_dataset = tf.data.Dataset.from_generator(
            lambda: generate_data(training_patients),
            output_signature=(
                tf.TensorSpec(shape=(img_size, img_size, 1), dtype=tf.float32, name="imagen"),
                tf.TensorSpec(shape=(), dtype=tf.int64, name="label")
            )
        ).batch(batch_size).prefetch(tf.data.experimental.AUTOTUNE)

        testing_dataset = tf.data.Dataset.from_generator(
            lambda: generate_data(testing_patients),
            output_signature=(
                tf.TensorSpec(shape=(img_size, img_size, 1), dtype=tf.float32, name="imagen"),
                tf.TensorSpec(shape=(), dtype=tf.int64, name="label")
            )
        ).batch(batch_size).prefetch(tf.data.experimental.AUTOTUNE)

        fold_datasets.append((training_dataset, testing_dataset))

    return fold_datasets

def construir_modelo():
    modelo = SimpleModel2(1)
    modelo.compile(
        optimizer='adam',
        loss='binary_crossentropy', 
        metrics=['accuracy', AUC(name='auc', curve='PR'), Recall(), Precision()]
    )
    return modelo

def entrenar_modelo(modelo, dataset_entrenamiento, dataset_testeo, epochs=5):
    # Entrenar el modelo
    modelo.fit(dataset_entrenamiento, epochs=epochs, validation_data=dataset_testeo)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Entrenar un modelo en el conjunto de datos de Santa Maria.")
    parser.add_argument("-p", "--particion", type=str, default="body", help="Tipo de partición (pet o torax3d)")
    parser.add_argument("-b", "--batch", type=int, default=32, help="Tamaño del lote para entrenamiento")
    parser.add_argument("-s", "--size", type=int, default=32, help="Tamaño de la imagen para extracción de ROI")
    parser.add_argument("-e", "--epochs", type=int, default=10, help="Número de épocas para entrenamiento")
    parser.add_argument(
        "-r", "--roi_fn", type=str, default="roiExtractionSize1",
        help="Nombre de la función de extracción de ROI",
    )
    parser.add_argument("-n", "--splits", type=int, default=5, help="Número de divisiones para KFold")
    parser.add_argument("--seed", type=int, default=None, help="Semilla aleatoria para reproducibilidad")

    args = parser.parse_args()

    # Resuelve el nombre de la función a una función real
    roi_fn = globals().get(args.roi_fn)

    if not callable(roi_fn):
        raise ValueError(f"Función de extracción de ROI no válida: {args.roi_fn}")

    # Cargar datos
    k_fold_dataset = cargar_datos_santamaria(
        args.particion,
        roi_fn,
        n_splits=args.splits,
        batch_size=args.batch,
        img_size=args.size,
        shuffle_buffer_size=1000,
        random_seed=args.seed,
    )

    for i, (train_ds, test_ds) in enumerate(k_fold_dataset):
        print(f"Fold: {i}")

        # Construir el modelo
        modelo = construir_modelo()

        # Entrenar el modelo
        entrenar_modelo(modelo, train_ds, test_ds, epochs=args.epochs)