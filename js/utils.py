import numpy as np
import skimage.measure as measure
import tensorflow as tf



def merge_boxes(rprops) :    
    x_mins = [r.bbox[0]  for r in rprops]
    y_mins = [r.bbox[1]  for r in rprops]
    x_maxs = [r.bbox[2]  for r in rprops]
    y_maxs = [r.bbox[3]  for r in rprops]

    bb = [np.min(x_mins), np.min(y_mins), np.max(x_maxs), np.max(y_maxs)]
    return bb

def get_biggest_region(rprops) :
    areas = np.array([r.area for r in rprops])    
    return rprops[np.argmax(areas)]

def get_roi(data, mask, padding, min_val) :
    cc = measure.label(mask)
    props = measure.regionprops(cc)
    roi =  merge_boxes(props)
    data_seg = tf.where(mask == 1, data, min_val)
    data_roi = data_seg[roi[0]-padding:roi[2]+padding, roi[1] - padding:roi[3] + padding]
    return roi, data_roi 