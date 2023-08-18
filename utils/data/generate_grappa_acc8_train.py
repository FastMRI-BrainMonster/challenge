import pygrappa
import fastmri
import os
import tqdm
import h5py
import numpy as np
import torch


# imagepath = '/Data/train/image'
kspacepath = '/Data/train/kspace'
savepath = '/root/grappa/train'

if not os.path.exists(savepath):
    os.makedirs(savepath)

h5list = os.listdir(kspacepath)
h5list.sort()

for hname in tqdm.tqdm(h5list):
    if 'acc4' in hname:
        continue

    f = h5py.File(os.path.join(kspacepath, hname), 'r')
    k = h5py.File(os.path.join(savepath, hname), 'w')

    if "pygrappa" in k:
        del k["pygrappa"]

    grappa_result = []

    for slice_num in range(len(f['kspace'])):
        mask = f['mask']
        calib = f['kspace'][slice_num]      # Something like (14, 768, 396)
        kspace = calib * mask
        
        # GRAPPA
        grappa = pygrappa.grappa(kspace, calib, kernel_size=(7,7), coil_axis = 0)
        grappa_tensor = torch.tensor(grappa, dtype=torch.cfloat)
        grappa_tensor = torch.view_as_real(grappa_tensor)
        grappa_image = fastmri.rss(fastmri.complex_abs(fastmri.ifft2c(grappa_tensor)), dim = 0)

        # CROP
        center = grappa_image.shape
        x = center[1]/2 - 384/2
        y = center[0]/2 - 384/2
        cropped_grappa_image = abs(grappa_image[int(y):int(y+384), int(x):int(x+384)])

        # SAVE
        grappa_result.append(cropped_grappa_image)

    grappa_result = np.stack(grappa_result, axis = 0)
    k.create_dataset("pygrappa", data = grappa_result)
    
    f.close()
    k.close()
    
