import h5py
import random
from unet_utils.data.transforms import DataTransform
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
import numpy as np
import os

class SliceData(Dataset):
    def __init__(self, root, transform,input_key,target_key, mode, is_grappa, grappa_path, forward=False):
        self.transform = transform
        self.input_key = input_key
        self.target_key = target_key
        self.forward = forward
        self.is_grappa = is_grappa
        self.grappa_path = grappa_path
        self.target = []
        self.input = []
        self.input_grappa = []
        if not forward:
            target_files = list(Path("/Data").joinpath(mode,"image").iterdir())
            for fname in sorted(target_files):
                num_slices = self._get_metadata(fname)
                self.target += [
                    (fname, slice_ind) for slice_ind in range(num_slices)
                ]
        input_files = [os.path.join(root,file) for file in os.listdir(root) if file.endswith('.h5')]
        #input_files = list(Path(root).iterdir())
        for fname in sorted(input_files):
            num_slices = self._get_metadata(fname)
            self.input += [
                (fname, slice_ind) for slice_ind in range(num_slices)
            ]
        
        if self.is_grappa == 'y':
            if mode is not None:
                grappa_folder = os.path.join(grappa_path,mode)
            else:
                grappa_folder = grappa_path
            input_grappa_files = [os.path.join(grappa_folder,file) for file in os.listdir(grappa_folder) if file.endswith('.h5')]
            for fname in sorted(input_grappa_files):
                num_slices = self._get_metadata(fname)
                self.input_grappa += [
                    (fname, slice_ind) for slice_ind in range(num_slices)
                ]

    def _get_metadata(self, fname):
        with h5py.File(fname, "r") as hf:
            if self.input_key in hf.keys():
                #image case, "reconstruction"
                num_slices = hf[self.input_key].shape[0]
            elif 'pygrappa' in hf.keys():
                num_slices = hf['pygrappa'].shape[0]
            elif self.target_key in hf.keys():
                #GT case, "image_label"
                num_slices = hf[self.target_key].shape[0]
        return num_slices

    def __len__(self):
        return len(self.input)

    def __getitem__(self, i):
        if not self.forward:
            target_fname, _ = self.target[i]
            
        input_fname, dataslice = self.input[i]
        with h5py.File(input_fname, "r") as hf:
            input_ = hf[self.input_key][dataslice]
            
        if self.is_grappa == 'y':
            input_grappa_fname, dataslice_grappa = self.input_grappa[i]
            assert dataslice == dataslice_grappa, "grappa is not matching with image"
            with h5py.File(input_grappa_fname, "r") as hf:
                input_grappa_ = hf['pygrappa'][dataslice]
        else:
            input_grappa_ = None
        if self.forward:
            target = -1
            attrs = -1
        else:
            with h5py.File(target_fname, "r") as hf:
                target = hf[self.target_key][dataslice]
                attrs = dict(hf.attrs)
        input_fname = input_fname.split('/')[-1]
        return self.transform(input_, input_grappa_, target, attrs, input_fname, dataslice)


def create_data_loaders(data_path, args, mode, shuffle=False, isforward=False):
    if isforward == False:
        target_key_ = args.target_key
        max_key_ = args.max_key
    else:
        max_key_ = -1
        target_key_ = -1
    data_storage = SliceData(
        root=data_path,
        transform=DataTransform(isforward, max_key_),
        input_key=args.input_key,
        target_key=target_key_,
        forward = isforward,
        is_grappa = args.is_grappa,
        grappa_path = args.grappa_path,
        mode = mode
    )

    data_loader = DataLoader(
        dataset=data_storage,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        shuffle=shuffle,
    )
    return data_loader
