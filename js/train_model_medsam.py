import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

import tensorflow as tf
import tensorflow_datasets as tfds
import numpy as np
from sklearn.model_selection import KFold
from simple import SimpleModel2
from tensorflow.keras.metrics import AUC, Precision, Recall
import argparse
import metrics
import models
import numpy as np
from segment_anything import sam_model_registry



max_val = 1000
min_val = -1000
val_range = max_val - min_val
margin = 3
initial_learning_rate = 0.001
epochs = 20
alpha = 0.00001

# cargar encoder
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

    model.train()
    return model

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

                # roi value to standarize the image slice
                if img_type == 'pet':
                    liver_roi_val = tf.cast(tf.reduce_mean(data['pet_liver']), dtype=tf.float32)
                    img_exam  = img_exam / liver_roi_val

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


# Define el modelo completo
class fcModel(nn.Module):
    def __init__(self, encoder_model, fc_input_size):
        super(CustomModel, self).__init__()
        self.encoder_model = encoder_model
        self.fc = nn.Linear(fc_input_size, 1)  # 1 salida para clasificación binaria

    def forward(self, x):
        # Obtén las características de la red principal
        features = self.encoder_model(x)

        # Aplana las características
        features = features.view(features.size(0), -1)

        # Pasa las características a través de la red completamente conectada
        output = self.fc(features)

        return output



# Construir modelo function
def construir_modelo(img_size, train_steps):
    # Inicializar codificador
    modelo = fcModel(cargar_encoder())
    # utilizar dimensionalidad de fc_input_size para la dimensionalidad del vector del encoder

    # Definir la tasa de aprendizaje
    cosdecay = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=train_steps, eta_min=0)

    # Definir el optimizador
    optimizer = optim.Adam(modelo.parameters(), lr=initial_learning_rate)

    # Definir la función de pérdida
    criterion = nn.BCEWithLogitsLoss()

    # Devolver el modelo, el optimizador y la función de pérdida
    return modelo, optimizer, criterion


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Entrenar un modelo en el conjunto de datos de Santa Maria.")
    parser.add_argument("-p", "--particion", type=str, default="chest_ct", help="Tipo de partición (pet, body o torax3d)")
    parser.add_argument("-b", "--batch", type=int, default=8x	x	, help="Tamaño del lote para entrenamiento")
    parser.add_argument("-s", "--size", type=int, default=1024, help="Tamaño de la imagen para extracción de ROI")
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
    #modelo = construir_modelo(args.size, train_steps)
    
    
    for epoch in range(args.epochs):
        print(f"Epoch {epoch + 1}/{args.epochs}")
        
        # Training loop
        for batch in train_ds:
            images, labels = batch
            # Train your model here using images and labels
            image_np = images.numpy()
        
            # Convert NumPy array to PyTorch tensor
            image_torch = torch.tensor(image_np, dtype=torch.float32)
            
            # If CUDA is available, load the model to the GPU using model.load_state_dict
            if torch.cuda.is_available(): image_torch=image_torch.cuda() 
            else: image_torch = image_torch.cpu()
            
            print(image_torch.shape)
            # Pass through the PyTorch encoder
            with torch.no_grad():
                features_tensor = encoder.image_encoder(image_torch)
            
            # Convert PyTorch features back to NumPy array
            features_np = features_tensor.squeeze().detach().numpy()
            print('shape of feature tensor: ', features_np.shape)
            
            # Convert NumPy array to TensorFlow tensor
            features_tensorflow = tf.convert_to_tensor(features_np)
	    
	    
	    

    # Validation loop
    for batch in test_ds:
        images, labels = batch
        # Evaluate your model here using images and labels
        # ...

   



	
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

