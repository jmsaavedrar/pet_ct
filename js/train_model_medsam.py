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
from sklearn.metrics import accuracy_score, roc_auc_score, confusion_matrix
import os


os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
torch.cuda.empty_cache()


max_val = 1000
min_val = -1000
val_range = max_val - min_val
margin = 3
initial_learning_rate = 0.001
epochs = 20
alpha = 0.00001

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Como se entrena el modelo: ", device)

# cargar encoder
def cargar_encoder():
    model_path = r'medsam_encoder_2.pth'

    # Load the model
    model = sam_model_registry['vit_b']()

    # If CUDA is not available, load the model to the i
    model.load_state_dict(torch.load(model_path, map_location=device)) 
    return model

# Map function
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

# Define el modelo completo
class fcModel(nn.Module):
    def __init__(self, encoder_model):
        super(fcModel, self).__init__()
        self.encoder_model = encoder_model
        self.fc = nn.Linear(1048576, 1)  # 1 salida para clasificación binaria
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # Obtén las características de la red principal
        features = self.encoder_model.image_encoder(x)

        print(features.shape)
        
        # Aplana las características
        features = features.view(features.size(0), -1)

        # Pasa las características a través de la red completamente conectada
        output = self.fc(features)
        return_output = self.sigmoid(output)
        print('before after sigmoid: ', output, return_output)

        return return_output

def train_one_epoch(model, optimizer, criterion, epoch_index, training_loader):
    running_loss = 0.
    last_loss = 0.

    # Here, we use enumerate(training_loader) instead of
    # iter(training_loader) so that we can track the batch
    # index and do some intra-epoch reporting
    for i, (inputs, labels) in enumerate(training_loader):
        # Every data instance is an input + label pair

        inputs = torch.as_tensor(inputs, device=device)
        labels = torch.as_tensor(labels, device=device, dtype=torch.float32).view(-1, 1)
 
        # Zero your gradients for every batch!
        optimizer.zero_grad()

        # Make predictions for this batch
        outputs = model(inputs)
        
        # Compute the loss and its gradients
        loss = criterion(outputs, labels)
        loss.backward()

        print('outputs and true labels:', outputs.item(), labels.item(), loss.item())
        print()

        # Adjust learning weights
        optimizer.step()

        # Gather data and report
        running_loss += loss.item()

        del inputs, labels, outputs, loss
        if i % 100 == 99:
            last_loss = running_loss / 100 # loss per batch
            print('  batch {} loss: {}'.format(i + 1, last_loss))
            running_loss = 0.

    return last_loss



# Construir modelo function
def construir_modelo(img_size, lr, train_steps):
    # Inicializar codificador
    encoder = cargar_encoder()
    modelo = fcModel(encoder)
    # utilizar dimensionalidad de fc_input_size para la dimensionalidad del vector del encoder

    # Definir el optimizador
    optimizer = optim.Adam(modelo.parameters(), lr=lr)

    # Definir la tasa de aprendizaje
    cosdecay = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=train_steps, eta_min=0)
    
    # Definir la función de pérdida
    criterion = nn.BCEWithLogitsLoss()

    # Devolver el modelo, el optimizador y la función de pérdida
    return modelo, optimizer, cosdecay, criterion



def calculate_metrics(predictions, targets):
    # Convert predictions to binary values (0 or 1)
    binary_predictions = torch.round(torch.sigmoid(predictions)).long()

    # Calculate metrics
    accuracy = accuracy_score(targets.cpu().numpy(), binary_predictions.cpu().numpy())
    auc = roc_auc_score(targets.cpu().numpy(), torch.sigmoid(predictions).cpu().numpy())
    precision = precision_score(targets.cpu().numpy(), binary_predictions.cpu().numpy())
    recall = recall_score(targets.cpu().numpy(), binary_predictions.cpu().numpy())

    return accuracy, auc, precision, recall




if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Entrenar un modelo en el conjunto de datos de Santa Maria.")
    parser.add_argument("-p", "--particion", type=str, default="chest_ct", help="Tipo de partición (pet, body o torax3d)")
    parser.add_argument("-b", "--batch", type=int, default=100, help="Tamaño del lote para entrenamiento")
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
    
    torch.cuda.empty_cache()
    train_ds, val_ds, test_ds = train_test_dataset
    
    train_ds = tfds.as_numpy(train_ds.shuffle(1024).map(map_fun).batch(args.batch).prefetch(tf.data.experimental.AUTOTUNE))
    test_ds = tfds.as_numpy(test_ds.shuffle(1024).map(map_fun).batch(args.batch).prefetch(tf.data.experimental.AUTOTUNE))
    val_ds = tfds.as_numpy(val_ds.shuffle(1024).map(map_fun).batch(args.batch).prefetch(tf.data.experimental.AUTOTUNE))
    
    # Construir el modelo
    model, optimizer, criterion = construir_modelo(args.size, 0.001, train_steps)
    model = model.to(device)

    # Initialize early stopping parameters
    best_val_loss = float('inf')
    early_stopping_counter = 0
    early_stopping_patience = 5  # Adjust the patience value as needed

    for epoch in range(args.epochs):
        print(f"Epoch {epoch + 1}/{args.epochs}")

        model.train(True)
        avg_loss = train_one_epoch(model, optimizer, criterion, epoch, train_ds)
        
        torch.cuda.empty_cache()
        model.eval()

        # Validation phase
        val_predictions = []
        val_targets = []
    
        with torch.no_grad():
            for inputs, labels in val_ds:
                inputs = torch.as_tensor(inputs, device=device)
                labels = torch.as_tensor(labels, device=device)
    
                outputs = model(inputs)
    
                val_predictions.append(outputs)
                val_targets.append(labels)
    
        val_predictions = torch.cat(val_predictions)
        val_targets = torch.cat(val_targets)
    
        # Calculate validation loss
        val_loss = criterion(val_predictions, val_targets).item()

        # Calculate training loss 
        train_loss = avg_loss.item()

        print(f'Train Loss - {train_loss:.3f}, Val Loss - {val_loss:.3f}')
        # Calculate and print metrics for training data
        train_metrics = calculate_metrics(val_predictions, val_targets)
        print(f'Validation Metrics - Accuracy: {train_metrics[0]:.3f}, AUC: {train_metrics[1]:.3f}, Precision: {train_metrics[2]:.3f}, Recall: {train_metrics[3]:.3f}')
        print()
    
        # Early stopping check
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            early_stopping_counter = 0
        else:
            early_stopping_counter += 1
            if early_stopping_counter >= early_stopping_patience:
                print(f'Early stopping at epoch {epoch + 1} based on validation loss.')
                break

        # Learning rate scheduler step
        cosdecay.step()
                
    # Testing phase
    test_predictions = []
    test_targets = []

    with torch.no_grad():
        for test_inputs, test_labels in test_ds:
            test_inputs = torch.as_tensor(test_inputs, device=device)
            test_labels = torch.as_tensor(test_labels, device=device)

            test_outputs = model(test_inputs)

            test_predictions.append(test_outputs)
            test_targets.append(test_labels)

    test_predictions = torch.cat(test_predictions)
    test_targets = torch.cat(test_targets)

    # Calculate testing loss
    test_loss = criterion(test_predictions, test_targets).item()

    print()
    print()
    # Calculate and print metrics for testing data
    val_metrics = calculate_metrics(val_predictions, val_targets)
    print(f'Validation Metrics - Accuracy: {val_metrics[0]:.3f}, AUC: {val_metrics[1]:.3f}, Precision: {val_metrics[2]:.3f}, Recall: {val_metrics[3]:.3f}')

    # Calculate and print metrics for testing data
    test_metrics = calculate_metrics(test_predictions, test_targets)
    print(f'Testing Metrics - Accuracy: {test_metrics[0]:.3f}, AUC: {test_metrics[1]:.3f}, Precision: {test_metrics[2]:.3f}, Recall: {test_metrics[3]:.3f}')

    # Print or store the results as needed
    print(f'Training Loss: {avg_loss:.3f}, Validation Loss: {val_loss:.3f}, Testing Loss: {test_loss:.3f}')
    print()


