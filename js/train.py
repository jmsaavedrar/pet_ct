import tensorflow as tf
import data
import models
import metrics 
import matplotlib.pyplot as plt
import numpy as np

CHECK_DATA = False

max_val = 1000
min_val = -1000
val_range = max_val - min_val

def check_data(data) :
    p = 0
    n = 0   
    gca = plt.imshow(np.zeros((512,512), dtype=np.float32), cmap = 'gray', vmin = 0, vmax = 1) 
    for i, sample in enumerate(data):
        # image = sample[0]
        # image = (image - min_val)/ val_range
        # #image = tf.expand_dims(image,-1)    
        # print('{} {}'.format(i, image.shape))
        # print('min {} max {}'.format(np.min(image), np.max(image)))        
        # gca.set_data(image)
        # plt.waitforbuttonpress(0.2)
        if sample[1] == 1 :
            p += 1
        else :
            n += 1
    print('total {} = p {} n {}'.format(p+n, p, n))
    return p + n


def map_fun(image, label) :    
    #TODO 
    #crop_size = 256    
    image = (image - min_val)/ val_range
    image = tf.image.grayscale_to_rgb(image)
    size = int(crop_size * 1.15)
    image = tf.image.resize_with_pad(image, size, size)
    image = tf.image.random_crop(image, (crop_size, crop_size,3))
    image = tf.image.random_flip_left_right(image)
    label =  tf.one_hot(label, 2)
    return image, label


ds_train, ds_test = data.get_training_testing_sm_datasets(random_seed=None)
batch_size = 32
channels = 3
if CHECK_DATA :
    check_data(ds_train)
    check_data(ds_test)
else :
    n_train = 1200
    n_test = check_data(ds_test)
    ds_train = ds_train.shuffle(1024).map(map_fun).batch(batch_size)
    ds_test = ds_test.shuffle(1024).map(map_fun).batch(batch_size)
    #n_train = len(ds_train)
    #n_test = len(ds_test)
    # ds_train = ds_train.shuffle(1024).map(map_fun).batch(batch_size)
    # ds_test = ds_train.shuffle(1024).map(map_fun).batch(batch_size)

    
    initial_learning_rate = 0.001
    epochs = 20
    train_steps = epochs * (n_train // batch_size)
    val_steps = n_test // batch_size
    alpha = 0.00001
    size = 256


    model = models.simple_model((size, size, channels))
    cosdecay = tf.keras.optimizers.schedules.CosineDecay(initial_learning_rate, decay_steps  = train_steps, alpha = alpha)
    #optimizer=tf.keras.optimizers.AdamW(learning_rate = cosdecay)
    #optimizer=tf.keras.optimizers.SGD(learning_rate = cosdecay, momentum = 0.9)
    optimizer=tf.keras.optimizers.Adam(learning_rate = cosdecay)
    model.compile(optimizer, loss='categorical_focal_crossentropy',  metrics=[metrics.true_positive, 
                                                                        metrics.false_positive,  
                                                                        tf.keras.metrics.AUC(multi_label=True, num_labels = 2), 'accuracy']) #  multi_label=True, num_labels = 2)])
    model.fit(ds_train, 
            epochs = epochs,
            validation_data = ds_test, 
            validation_steps = val_steps)
    
    
 
