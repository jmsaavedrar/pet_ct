import tensorflow as tf


def true_positive(y_true, y_pred):        
    tp = tf.boolean_mask(y_pred, y_true)        
    tp = tf.reduce_mean(tp)
    return tp


def true_positive_ce(y_true, y_pred):    
    yt = tf.argmax(y_true, axis = 1)            
    tp = tf.boolean_mask(yp, yt)
    tp = tf.reduce_mean(tp)
    return tp

def false_positive(y_true, y_pred):           
    fp = tf.boolean_mask(y_pred, 1 - y_true)
    fp = tf.reduce_mean(fp)
    return fp

def false_positive_ce(y_true, y_pred):    
    yt = tf.argmax(y_true, axis = 1)    
    yp = tf.gather(y_pred, 1 - yt)

    fp = tf.boolean_mask(yp, 1 - yt)
    fp = tf.reduce_mean(fp)
    return fp