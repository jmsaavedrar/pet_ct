import tensorflow as tf


def true_positive(y_true, y_pred):        
    yp = tf.reshape(y_pred, (-1,))
    yt = tf.reshape(y_true, (-1,))
    tp = tf.boolean_mask(yp, yt)        
    tp = tf.reduce_mean(tp)
    return tp


def false_positive(y_true, y_pred):           
    yp = tf.reshape(y_pred, (-1,))
    yt = tf.reshape(y_true, (-1,))
    fp = tf.boolean_mask(yp, 1 - yt)
    fp = tf.reduce_mean(fp)
    return fp
