import h5py
import random
from unet_utils.data.transforms import DataTransform
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
import numpy as np
import os

class SliceData(Dataset):
    def __init__(self, root, transform,input_key,target_key, mode, forward=False):
        self.transform = transform
        self.input_key = input_key
        self.target_key = target_key
        self.forward = forward
        self.target = []
        self.input = []
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


    def _get_metadata(self, fname):
        with h5py.File(fname, "r") as hf:
            if self.input_key in hf.keys():
                #image case, "reconstruction"
                num_slices = hf[self.input_key].shape[0]
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
            input = hf[self.input_key][dataslice]
        if self.forward:
            target = -1
            attrs = -1
        else:
            with h5py.File(target_fname, "r") as hf:
                target = hf[self.target_key][dataslice]
                attrs = dict(hf.attrs)
        input_fname = input_fname.split('/')[-1]
        return self.transform(input, target, attrs, input_fname, dataslice)


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
        mode = mode
    )

    data_loader = DataLoader(
        dataset=data_storage,
        batch_size=args.batch_size,
        shuffle=shuffle,
    )
    return data_loader
