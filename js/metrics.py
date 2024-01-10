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



# precision and recall
def precision(y_true, y_pred):
    yp = tf.reshape(y_pred, (-1,))
    yt = tf.reshape(y_true, (-1,))
    
    true_positives = tf.boolean_mask(yp, yt)
    false_positives = tf.boolean_mask(yp, 1 - yt)
    
    if tf.reduce_sum(true_positives) + tf.reduce_sum(false_positives) == 0: return 0.0
    precision = tf.divide(tf.reduce_sum(true_positives), tf.reduce_sum(true_positives) + tf.reduce_sum(false_positives))
    
    return precision

def recall(y_true, y_pred):
    yp = tf.reshape(y_pred, (-1,))
    yt = tf.reshape(y_true, (-1,))
    
    true_positives = tf.boolean_mask(yp, yt)
    false_negatives = tf.boolean_mask(1 - yp, yt)
    
    if tf.reduce_sum(true_positives) + tf.reduce_sum(false_negatives) == 0: return 0.0
    recall = tf.divide(tf.reduce_sum(true_positives), tf.reduce_sum(true_positives) + tf.reduce_sum(false_negatives))
    
    return recall
