import pygrappa
import fastmri
import os
import tqdm
import h5py
import numpy as np
import torch


imagepath = '/Data/train/image'
# kspacepath = '/Data/train/kspace'
savepath = '/root/grappa/train'

if not os.path.exists(savepath):
    os.makedirs(savepath)

h5list = os.listdir(imagepath)
h5list.sort()

for hname in tqdm.tqdm(h5list):
    if 'acc8' in hname:
        continue

    f = h5py.File(os.path.join(imagepath, hname), 'r')
    k = h5py.File(os.path.join(savepath, hname), 'w')


    if "pygrappa" in k:
        del k["pygrappa"]

    k.create_dataset("pygrappa", data = f['image_grappa'])
    
    f.close()
    k.close()
    
