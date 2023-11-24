import tensorflow as tf


def true_positive(y_true, y_pred):        
    tp = tf.boolean_mask(y_pred, y_true)        
    tp = tf.reduce_mean(tp)
    return tp


def false_positive(y_true, y_pred):           
    fp = tf.boolean_mask(y_pred, 1 - y_true)
    fp = tf.reduce_mean(fp)
    return fp
