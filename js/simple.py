"""
A simple cnn used for training mnist.
"""
import tensorflow as tf
from tensorflow.keras import regularizers
from tensorflow.keras import layers as layers


class SimpleModel(tf.keras.Model):
    def __init__(self, number_of_classes):
        super(SimpleModel, self).__init__()        
        self.conv_1 = tf.keras.layers.Conv2D(32, (3,3), padding = 'same',  kernel_initializer = 'he_normal', name = 'conv1')
        self.max_pool = tf.keras.layers.MaxPool2D((3,3), strides = 2, padding = 'same')
        self.relu = tf.keras.layers.ReLU();        
        self.bn_conv_1 = tf.keras.layers.BatchNormalization()
        #self.bn_conv_1 = tf.keras.layers.LayerNormalization()        
        self.conv_2 = tf.keras.layers.Conv2D(64, (3,3), padding = 'same',  kernel_initializer='he_normal', name = 'conv2')
        self.bn_conv_2 = tf.keras.layers.BatchNormalization()
        #self.bn_conv_2 = tf.keras.layers.LayerNormalization()
        self.conv_3 = tf.keras.layers.Conv2D(128, (3,3), padding = 'same', kernel_initializer='he_normal', name = 'conv3')
        self.bn_conv_3 = tf.keras.layers.BatchNormalization()     
        #self.bn_conv_3 = tf.keras.layers.LayerNormalization()
        self.fc1 = tf.keras.layers.Dense(256, kernel_initializer='he_normal', name = 'dense1')
        #self.bn_fc_1 = tf.keras.layers.LayerNormalization()
        self.bn_fc_1 = tf.keras.layers.BatchNormalization(name = 'embedding' )
        self.fc2 = tf.keras.layers.Dense(number_of_classes)

    # here, connecting the modules
    def call(self, inputs):        
        #first block
        x = self.conv_1(inputs)    
        x = self.bn_conv_1(x) 
        x = self.relu(x) 
        x = self.max_pool(x)    #14 X 14
        #second block
        x = self.conv_2(x)  
        x = self.bn_conv_2(x) 
        x = self.relu(x) 
        x = self.max_pool(x)  #7X7
        #third block
        x = self.conv_3(x)  
        x = self.bn_conv_3(x)
        x = self.relu(x)  
        x = self.max_pool(x)  #4X4
        #last block        
        x = tf.keras.layers.Flatten()(x) 
        x = self.fc1(x)  
        x = self.bn_fc_1(x) 
        x = self.relu(x) 
        x = self.fc2(x) 
        x = tf.keras.activations.sigmoid(x)
        return x
    
    def model(self, input_shape):
        x = tf.keras.layers.Input(shape = input_shape)
        return tf.keras.Model(inputs = [x], outputs = self.call(x))






class SimpleModel2(tf.keras.Model):
    def __init__(self, number_of_classes, reg_val=0.01):
        super(SimpleModel2, self).__init__()        
        self.conv_1 = tf.keras.layers.Conv2D(32, (3,3), padding = 'same',  kernel_initializer = 'he_normal', name = 'conv1', kernel_regularizer=regularizers.l2(reg_val))
        self.max_pool = tf.keras.layers.MaxPool2D((3,3), strides = 2, padding = 'same')
        self.relu = tf.keras.layers.ReLU();        
        self.bn_conv_1 = tf.keras.layers.BatchNormalization()
        #self.bn_conv_1 = tf.keras.layers.LayerNormalization()        
        self.conv_2 = tf.keras.layers.Conv2D(64, (3,3), padding = 'same',  kernel_initializer='he_normal', name = 'conv2', kernel_regularizer=regularizers.l2(reg_val))
        self.bn_conv_2 = tf.keras.layers.BatchNormalization()
        #self.bn_conv_2 = tf.keras.layers.LayerNormalization()
        self.conv_3 = tf.keras.layers.Conv2D(128, (3,3), padding = 'same', kernel_initializer='he_normal', name = 'conv3', kernel_regularizer=regularizers.l2(reg_val))
        self.bn_conv_3 = tf.keras.layers.BatchNormalization()     
        #self.bn_conv_3 = tf.keras.layers.LayerNormalization()
        self.fc1 = tf.keras.layers.Dense(256, kernel_initializer='he_normal', name = 'dense1')
        #self.bn_fc_1 = tf.keras.layers.LayerNormalization()
        self.bn_fc_1 = tf.keras.layers.BatchNormalization(name = 'embedding' )
        self.fc2 = tf.keras.layers.Dense(number_of_classes)
        self.dropout = tf.keras.layers.Dropout(0.5)

    # here, connecting the modules
    def call(self, inputs):        
        #first block
        x = self.conv_1(inputs)    
        x = self.bn_conv_1(x) 
        x = self.relu(x) 
        x = self.max_pool(x)    #14 X 14
        #second block
        x = self.conv_2(x)  
        x = self.bn_conv_2(x) 
        x = self.relu(x) 
        x = self.max_pool(x)  #7X7
        #third block
        x = self.conv_3(x)  
        x = self.bn_conv_3(x)
        x = self.relu(x)  
        x = self.max_pool(x)  #4X4
        #last block        
        x = tf.keras.layers.Flatten()(x) 
        x = self.fc1(x)  
        x = self.bn_fc_1(x) 
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x) 
        x = tf.keras.activations.sigmoid(x)
        return x
    
    def model(self, input_shape):
        x = tf.keras.layers.Input(shape = input_shape)
        return tf.keras.Model(inputs = [x], outputs = self.call(x))



class BinaryCNN(tf.keras.Model):
    def __init__(self, number_of_classes):
        super(BinaryCNN, self).__init__()

        self.conv1 = layers.Conv2D(32, (3, 3), activation='relu', padding='same', input_shape=(None, None, 1))
        self.maxpool1 = layers.MaxPooling2D((2, 2), strides=(2, 2))
        
        self.conv2 = layers.Conv2D(64, (3, 3), activation='relu', padding='same')
        self.maxpool2 = layers.MaxPooling2D((2, 2), strides=(2, 2))
        
        self.conv3 = layers.Conv2D(128, (3, 3), activation='relu', padding='same')
        self.maxpool3 = layers.MaxPooling2D((2, 2), strides=(2, 2))

        self.flatten = layers.Flatten()
        
        self.fc1 = layers.Dense(256, activation='relu')
        self.fc2 = layers.Dense(number_of_classes, activation='sigmoid')

    def call(self, inputs):
        x = self.conv1(inputs)
        x = self.maxpool1(x)
        
        x = self.conv2(x)
        x = self.maxpool2(x)
        
        x = self.conv3(x)
        x = self.maxpool3(x)

        x = self.flatten(x)
        x = self.fc1(x)
        output = self.fc2(x)

        return output

    def model(self):
        input_shape = (None, None, 1)
        x = layers.Input(shape=input_shape)
        return tf.keras.Model(inputs=[x], outputs=self.call(x))



class SimpleCNNModel(tf.keras.Model):
    def __init__(self, number_of_classes, input_shape):
        super(SimpleCNNModel, self).__init__()
        self.conv1 = tf.keras.layers.Conv2D(32, (3, 3), activation='relu', padding='same', input_shape=input_shape)
        self.maxpool1 = tf.keras.layers.MaxPooling2D((2, 2))
        self.conv2 = tf.keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same')
        self.maxpool2 = tf.keras.layers.MaxPooling2D((2, 2))
        self.flatten = tf.keras.layers.Flatten()
        self.dense1 = tf.keras.layers.Dense(128, activation='relu')
        self.dense2 = tf.keras.layers.Dense(number_of_classes, activation='sigmoid')  # Using 'sigmoid' for binary classification

    def call(self, inputs):
        x = self.conv1(inputs)
        x = self.maxpool1(x)
        x = self.conv2(x)
        x = self.maxpool2(x)
        x = self.flatten(x)
        x = self.dense1(x)
        output = self.dense2(x)
        return output

    def model(self):
        x = tf.keras.layers.Input(shape=(input_shape[0], input_shape[1], input_shape[2]))
        return tf.keras.Model(inputs=[x], outputs=self.call(x))