# Original Code from https://github.com/jun-pac/SNU-fastMRI-22-summer/blob/master/Code/data_preparation.py

import h5py
import numpy as np
import os
import cv2
import tqdm

def get_image_mask(target):
    mask = np.zeros(target.shape)
    mask[target>5e-5] = 1
    kernel = np.ones((3,3), np.uint8)
    mask = cv2.erode(mask, kernel, iterations=1)
    mask = cv2.dilate(mask, kernel, iterations=15)
    mask = cv2.erode(mask, kernel, iterations=14)
    mask = np.array(mask, dtype = np.float32)
    return np.array(mask)

def get_h5_mask(target):
    mask = []
    for i in range(len(target)):
        mask_i = get_image_mask(target[i])
        mask.append(mask_i)
    mask = np.stack(mask, axis = 0)
    return mask

for ftype in ['train', 'val', 'leaderboard/acc4', 'leaderboard/acc8']:
    imagepath = os.path.join('/Data', ftype, 'image')
    kspacepath = os.path.join('/Data', ftype, 'kspace')
    savepath = os.path.join('/root/brain_mask', ftype)
    
    if not os.path.exists(savepath):
        os.makedirs(savepath)
    
    h5list = os.listdir(kspacepath)
    
    h5list.sort()
    
    for hname in tqdm.tqdm(h5list):
        f = h5py.File(os.path.join(imagepath, hname), 'r')
        k = h5py.File(os.path.join(savepath, hname), 'w')
        
        if "image_label" in k:
            del k["image_label"]
        
        if "image_mask" in k:
            del k["image_mask"]
            
        mask = get_h5_mask(f["image_label"][()])
        k.create_dataset("image_label", data=f["image_label"][()])
        k.create_dataset("image_mask", data=mask)
        
        k.attrs["max"] = np.max(f["image_label"][()])
        
        f.close()
        k.close()