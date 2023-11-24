import tensorflow as tf
import data
import models
import metrics 


CHECK_DATA = False

def check_data(data) :
    p = 0
    n = 0
    for i, sample in enumerate(data):
        image = sample[0]
        #image = tf.expand_dims(image,-1)    
        print('{} {}'.format(i, image.numpy().shape))
        if sample[1] == 1 :
            p += 1
        else :
            n += 1
    print('total {} = p {} n {}'.format(p+n, p, n))


def map_fun(image, label) :    
    
    #TODO 
    crop_size = 512
    image = tf.expand_dims(image, -1)
    image = tf.image.grayscale_to_rgb(image)
    size = int(crop_size * 1.15)
    image = tf.image.resize_with_pad(image, size, size)
    image = tf.image.random_crop(image, (crop_size, crop_size,3))
    image = tf.image.random_flip_left_right(image)
    label =  tf.one_hot(label, 2)
    return image, label


ds_train, ds_test = data.get_training_testing_sm_datasets(random_seed=None)
batch_size = 64

if CHECK_DATA :
    check_data(ds_test)
else :
    ds_train = ds_train.shuffle(1024).map(map_fun).batch(batch_size)
    ds_test = ds_test.shuffle(1024).map(map_fun).batch(batch_size)
    #n_train = len(ds_train)
    #n_test = len(ds_test)
    # ds_train = ds_train.shuffle(1024).map(map_fun).batch(batch_size)
    # ds_test = ds_train.shuffle(1024).map(map_fun).batch(batch_size)

    n_train = 354
    n_test = 74
    initial_learning_rate = 0.001
    epochs = 30
    train_steps = epochs * (n_train // batch_size)
    val_steps = n_test // batch_size
    alpha = 0.001
    size = 512


    model = models.simple_model((size, size, 3))
    cosdecay = tf.keras.optimizers.schedules.CosineDecay(initial_learning_rate, decay_steps  = train_steps, alpha = alpha)
    optimizer=tf.keras.optimizers.AdamW(learning_rate = cosdecay)
    #optimizer=tf.keras.optimizers.SGD(learning_rate = cosdecay, momentum = 0.9)
    #optimizer=tf.keras.optimizers.Adam(learning_rate = cosdecay)
    model.compile(optimizer, loss='categorical_crossentropy',  metrics=[metrics.true_positive, 
                                                                        metrics.false_positive,  
                                                                        tf.keras.metrics.AUC(multi_label=True, num_labels = 2)])
    model.fit(ds_train, 
            epochs = epochs,
            validation_data = ds_test, 
            validation_steps = val_steps)
    
    
 
