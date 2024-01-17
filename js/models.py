import tensorflow as tf

def conv_bn_relu(channels) :
    conv = tf.keras.layers.Conv2D(channels, kernel_size=3, strides=1, padding='same', kernel_regularizer = tf.keras.regularizers.L2())
    bn = tf.keras.layers.BatchNormalization()
    relu =  tf.keras.layers.Activation('relu')
    
    def apply(x) :
        x = relu(bn(conv(x)))
        return x
    return apply
    
def simple_model(shape) :
    # defining input-shape
    x_input = tf.keras.layers.Input(shape)
    # block 1
    x = tf.keras.layers.Conv2D(32, kernel_size=3, strides=2, padding='same')(x_input)
    # block 2
    x = conv_bn_relu(32)(x) 
    x = tf.keras.layers.MaxPooling2D()(x)
    # block 3
    x = conv_bn_relu(64)(x)
    x = tf.keras.layers.MaxPooling2D()(x)
    # block 4
    x = conv_bn_relu(128)(x)
    x = tf.keras.layers.MaxPooling2D()(x)
    # block 5
    x = conv_bn_relu(256)(x)
    x = tf.keras.layers.MaxPooling2D()(x)    
    # classification head
    x = tf.keras.layers.GlobalAveragePooling2D()(x)    
    x = tf.keras.layers.Dense(256, activation = 'relu', kernel_regularizer = tf.keras.regularizers.L2())(x)
    x = tf.keras.layers.Dense(128, activation = 'relu', kernel_regularizer = tf.keras.regularizers.L2())(x)
    x = tf.keras.layers.Dense(1, activation = 'sigmoid')(x)
    #x = tf.keras.layers.Dense(2, activation = 'softmax')(x)
    model = tf.keras.models.Model(inputs = x_input, outputs = x)
    return model



def simple_model_simplest(shape):
    x_input = tf.keras.layers.Input(shape)

    # block 1
    x = tf.keras.layers.Conv2D(32, kernel_size=3, strides=2, padding='same')(x_input)

    # block 2
    x = conv_bn_relu(32)(x) 
    x = tf.keras.layers.MaxPooling2D()(x)
    
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    
    x = tf.keras.layers.Dense(128, activation = 'relu', kernel_regularizer = tf.keras.regularizers.L2())(x)
    output = tf.keras.layers.Dense(1, activation='sigmoid')(x)
    
    model = tf.keras.models.Model(inputs=x_input, outputs=output)
    
    return model


def simple_model_v3(shape) :
    # defining input-shape
    x_input = tf.keras.layers.Input(shape)
    # block 1
    x = tf.keras.layers.Conv2D(32, kernel_size=3, strides=2, padding='same')(x_input)
    # block 2
    x = conv_bn_relu(32)(x)
    x = tf.keras.layers.MaxPooling2D()(x)
    # block 3
    x = conv_bn_relu(64)(x)
    x = tf.keras.layers.MaxPooling2D()(x)
    # classification head
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dense(128, activation = 'relu', kernel_regularizer = tf.keras.regularizers.L2())(x)
    x = tf.keras.layers.Dense(1, activation = 'sigmoid')(x)
    #x = tf.keras.layers.Dense(2, activation = 'softmax')(x)
    model = tf.keras.models.Model(inputs = x_input, outputs = x)
    return model
