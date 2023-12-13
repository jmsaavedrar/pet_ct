import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import numpy as np
import argparse
from segment_anything import sam_model_registry
import tensorflow as tf
import tensorflow_datasets as tfds


max_val = 1000
min_val = -1000
val_range = max_val - min_val
margin = 3

# cargar_encoder
def cargar_encoder():
    model_path = r'medsam_encoder_2.pth'

    # Check if CUDA is available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)

    # Load the model
    model = sam_model_registry['vit_b']()
    
    # If CUDA is available, load the model to the GPU using model.load_state_dict
    if torch.cuda.is_available():
        model.load_state_dict(torch.load(model_path))
        model = model.to(device)
    else:
        # If CUDA is not available, load the model to the i
        model.load_state_dict(torch.load(model_path, map_location='cpu'))

    model.eval()
    return model

# Inicializar codificador
encoder = cargar_encoder()

# Map function

def map_fun(image, label) :        
    crop_size = 256    
    image = (image - min_val)/ val_range
    image = tf.image.grayscale_to_rgb(image)
    image = tf.transpose(image, perm=[2,0,1])
    #size = int(crop_size * 1.15)
    #image = tf.image.resize_with_pad(image, size, size)
    #image = tf.image.random_crop(image, (crop_size, crop_size,3))
    #image = tf.image.random_flip_left_right(image)
    return image, label

# Extract ROI function
def extract_roi(image, mask, margin):
    mask = tf.cast(mask, dtype=tf.bool)
    indices = tf.where(mask)

    min_row = tf.reduce_min(indices[:, 0])
    min_col = tf.reduce_min(indices[:, 1])
    max_row = tf.reduce_max(indices[:, 0])
    max_col = tf.reduce_max(indices[:, 1])

    bounding_box = image[min_row - margin:max_row + 1 + margin, min_col - margin:max_col + 1 + margin]

    return bounding_box

# Cargar datos function
def cargar_datos(img_type, img_type_sm, n_splits=5, img_size=32, margin=5, batch_size=32, shuffle_buffer_size=1000, random_seed=None):
    stanford_dataset, stanford_info = tfds.load(f'stanford_dataset/{img_type}', with_info=True,
                                                 data_dir='/media/roberto/TOSHIBA EXT/tensorflow_ds/')
    santa_maria_dataset, santa_maria_info = tfds.load(f'santa_maria_dataset/{img_type_sm}', with_info=True,
                                                     data_dir='/media/roberto/TOSHIBA EXT/tensorflow_ds/')

    stanford_patients = list(stanford_info.splits.keys())
    santa_maria_patients = list(santa_maria_info.splits.keys())

    def generate_data(dataset, patient_ids):
        for patient_id in patient_ids:
            patient_data = dataset[patient_id]
            for data in patient_data:
                mask_exam = data['mask_exam']
                img_exam = data['img_exam']

                if img_type == 'pet':
                    liver_roi_val = tf.cast(tf.reduce_sum(data['pet_liver']), dtype=tf.float32) / 619.0
                    if liver_roi_val == 0: liver_roi_val = 1
                    img_exam = img_exam / liver_roi_val

                data_roi = extract_roi(img_exam, mask_exam, margin)
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
    
    
def process_dataset(dataset, encoder, is_training, batch_size=32):
    features_list = []
    labels_list = []

    # Use PyTorch DataLoader for batching
    #dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=is_training)
    for i, batch in enumerate(dataset.as_numpy_iterator()):
    #for batch in dataloader:
        print(f'Batch {i}')
        images, labels = batch

        # Convert TensorFlow image to NumPy array
        #image_np = images.numpy()

        # Convert NumPy array to PyTorch tensor
        image_torch = torch.tensor(images, dtype=torch.float32)

        # If CUDA is available, load the model to the GPU using model.load_state_dict
        #if torch.cuda.is_available():
        #    image_torch = image_torch.cuda()
        #else:
        #    image_torch = image_torch.cpu()

        # Pass through the PyTorch encoder
        with torch.no_grad():
            features_tensor = encoder.image_encoder(image_torch)

        # Convert PyTorch features back to NumPy array
        features_np = features_tensor.squeeze().detach().numpy()

        # Append features and labels to the lists
        features_list.append(features_np)
        labels_list.append(labels.numpy())

    # Concatenate features and labels
    features_np = np.concatenate(features_list, axis=0)
    labels_np = np.concatenate(labels_list, axis=0)

    return features_np, labels_np
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Entrenar un modelo en el conjunto de datos de Santa Maria.")
    parser.add_argument("-p", "--particion", type=str, default="chest_ct", help="Tipo de partición (pet, body o torax3d)")
    parser.add_argument("-b", "--batch", type=int, default=8, help="Tamaño del lote para entrenamiento")
    parser.add_argument("-s", "--size", type=int, default=1024, help="Tamaño de la imagen para extracción de ROI")
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

    train_ds, test_ds = train_test_dataset
    
    train_ds = train_ds.shuffle(1024).map(map_fun).batch(args.batch).prefetch(tf.data.experimental.AUTOTUNE)
    test_ds = test_ds.shuffle(1024).map(map_fun).batch(args.batch).prefetch(tf.data.experimental.AUTOTUNE)
    
    # Construir el modelo
    #modelo = construir_modelo(args.size, train_steps)
    
    # Assuming train_ds and test_ds are tf.data.Dataset
    # Assuming encoder is already defined and loaded
    
    # Process training dataset
    train_features_np, train_labels_np = process_dataset(train_ds, encoder, is_training=True)

    # Process testing dataset
    test_features_np, test_labels_np = process_dataset(test_ds, encoder, is_training=False)

    # Save features and labels to a file
    np.savez(f'lung_datasets_tensors_{args.particion}.npz', train_features=train_features_np, train_labels=train_labels_np, test_features=test_features_np, test_labels=test_labels_np)

