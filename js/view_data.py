import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import resource
import tensorflow_datasets as tfds
import random
import os
from pathlib import Path
from PIL import Image
import numpy as np
from copy import copy
import skimage.measure as measure
import matplotlib.patches as patches
import skimage.exposure as exposure


def get_biggest_region(rprops) :
    areas = np.array([r.area for r in rprops])    
    return rprops[np.argmax(areas)]

    
data_dir = '/mnt/hd-data/Datasets/medseg/pet_ct'
santamaria_body_dataset, santamaria_body_info = tfds.load('santa_maria_dataset/body', data_dir = data_dir, with_info=True)
santamaria_torax_dataset, santamaria_torax_info = tfds.load('santa_maria_dataset/torax3d', data_dir = data_dir, with_info=True)
santamaria_pet_dataset, santamaria_pet_info = tfds.load('santa_maria_dataset/pet', data_dir = data_dir, with_info=True)

body_keys = list(santamaria_body_info.splits.keys())
torax_keys = list(santamaria_torax_info.splits.keys())
pet_keys = list(santamaria_pet_info.splits.keys())

# Create a figure with subplots
ct_max_val = 3071
ct_min_val = -3023 
ct_range = ct_max_val - ct_min_val

palette = copy(plt.cm.gray)
palette.set_bad('yellow')

num_examples = len(body_keys)
f, axarr = plt.subplots(2, 3)
ax_ct = axarr[0][0]
ax_ct_chest = axarr[0][1]
ax_pet = axarr[0][2]

ax_roi_ct = axarr[1][0]
ax_roi_ct_chest = axarr[1][1]
ax_roi_pet = axarr[1][2]

im_ct = ax_ct.imshow(np.zeros((512,512), dtype=np.float32), cmap = palette, vmax = 1, vmin = 0)
im_ct_chest = ax_ct_chest.imshow(np.zeros((512,512), dtype=np.float32), cmap = palette, vmax = 1, vmin = 0)
im_pet = ax_pet.imshow(np.zeros((512,512)), cmap='jet')


im_roi_ct = ax_roi_ct.imshow(np.zeros((64,64), dtype=np.float32), cmap = 'gray', vmax = 1, vmin = 0)
im_roi_ct_chest = ax_roi_ct_chest.imshow(np.zeros((64,64), dtype=np.float32), cmap = 'gray', vmax = 1, vmin = 0)


for i, key in enumerate(body_keys): # for each patient
    if key in torax_keys and key in pet_keys:      
        body_data = santamaria_body_dataset[key]
        torax_data = santamaria_torax_dataset[key]
        pet_data = santamaria_pet_dataset[key]
        print("patient: ", key)        
        # # Assuming the datasets are TensorFlow datasets
        for j, (body_sample, torax_sample, pet_sample) in enumerate(zip(body_data, torax_data, pet_data)):
            #pet-ct
            [p.remove() for p in ax_ct.patches]            
            [p.remove() for p in ax_ct_chest.patches]            
            [p.remove() for p in ax_pet.patches]            
            #******************************************* CT
            data_ct = body_sample['img_exam'].numpy()            
            data_ct = (data_ct - ct_min_val)/ ct_range 
            print('{} {}'.format(np.min(data_ct), np.max(data_ct)))
            data_ct = exposure.equalize_adapthist(data_ct)
            mask_ct = body_sample['mask_exam']            
            cc_ct = measure.label(mask_ct)
            props = measure.regionprops(cc_ct)
            roi_ct =  get_biggest_region(props)
            data_roi_ct = data_ct[roi_ct.bbox[0]:roi_ct.bbox[2], roi_ct.bbox[1]:roi_ct.bbox[3]]
            data_ct = np.ma.masked_where(mask_ct, data_ct)
            
            im_ct.set_data(data_ct)
            ax_ct.set_title("Body Exam")            
            ax_ct.add_patch(patches.Rectangle((roi_ct.bbox[1],roi_ct.bbox[0]),roi_ct.bbox[3] - roi_ct.bbox[1] + 1 ,roi_ct.bbox[2] - roi_ct.bbox[0] + 1, edgecolor='red', facecolor='none',lw=2))
            im_roi_ct.set_data(data_roi_ct)

            #******************************************* CHEST CT
            data_ct_chest = torax_sample['img_exam'].numpy()
            data_ct_chest = (data_ct_chest - ct_min_val)/ ct_range 
            data_ct_chest = exposure.equalize_adapthist(data_ct_chest)
            mask_ct_chest = torax_sample['mask_exam']
            
            cc_ct = measure.label(mask_ct_chest)
            props = measure.regionprops(cc_ct)
            roi_ct_chest =  props[0]
            data_roi_ct_chest = data_ct_chest[roi_ct_chest.bbox[0]:roi_ct_chest.bbox[2], roi_ct_chest.bbox[1]:roi_ct_chest.bbox[3]]

            data_ct_chest = np.ma.masked_where(mask_ct_chest, data_ct_chest)
            im_ct_chest.set_data(data_ct_chest)            
            ax_ct_chest.set_title("Torax 3D Exam")            
            ax_ct_chest.add_patch(patches.Rectangle((roi_ct_chest.bbox[1],roi_ct_chest.bbox[0]),roi_ct_chest.bbox[3] - roi_ct_chest.bbox[1] + 1 ,roi_ct_chest.bbox[2] - roi_ct_chest.bbox[0] + 1, edgecolor='red', facecolor='none',lw=2))
            im_roi_ct_chest.set_data(data_roi_ct_chest)

            #pet
            ax_pet.set_title("Pet Exam")
            data_pet = pet_sample['img_exam'].numpy()
            print('{}'.format(body_sample['label']))
            im_pet.set_data(data_pet)
            # plt.show()
            plt.waitforbuttonpress(0.1)

plt.show()
