import numpy as np
import torch

def to_tensor(data):
    """
    Convert numpy array to PyTorch tensor. For complex arrays, the real and imaginary parts
    are stacked along the last dimension.
    Args:
        data (np.array): Input numpy array
    Returns:
        torch.Tensor: PyTorch version of data
    """
    return torch.from_numpy(data)

class DataTransform:
    def __init__(self, isforward, max_key):
        self.isforward = isforward
        self.max_key = max_key
        # [add] for augmentation
        
    def __call__(self, input_, input_grappa, target, attrs, fname, slices):
        if not self.isforward:
            target = to_tensor(target)
            maximum = attrs[self.max_key]
        else:
            target = -1
            maximum = -1
        
        input_ = to_tensor(input_)
        if input_grappa is not None:
            input_grappa = to_tensor(input_grappa)
            input_ = torch.stack((input_, input_grappa), 0)
        
        return input_, target, maximum, fname, slices
