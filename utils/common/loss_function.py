"""
Copyright (c) Facebook, Inc. and its affiliates.
This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class SSIMLoss(nn.Module):
    """
    SSIM loss module.
    """

    def __init__(self, win_size: int = 7, k1: float = 0.01, k2: float = 0.03):
        """
        Args:
            win_size: Window size for SSIM calculation.
            k1: k1 parameter for SSIM calculation.
            k2: k2 parameter for SSIM calculation.
        """
        super().__init__()
        self.win_size = win_size
        self.k1, self.k2 = k1, k2
        self.register_buffer("w", torch.ones(1, 1, win_size, win_size) / win_size ** 2)
        NP = win_size ** 2
        self.cov_norm = NP / (NP - 1)

    def forward(self, X, Y, data_range):
        X = X.unsqueeze(1)
        Y = Y.unsqueeze(1)
        data_range = data_range[:, None, None, None]
        C1 = (self.k1 * data_range) ** 2
        C2 = (self.k2 * data_range) ** 2
        ux = F.conv2d(X, self.w)
        uy = F.conv2d(Y, self.w)
        uxx = F.conv2d(X * X, self.w)
        uyy = F.conv2d(Y * Y, self.w)
        uxy = F.conv2d(X * Y, self.w)
        vx = self.cov_norm * (uxx - ux * ux)
        vy = self.cov_norm * (uyy - uy * uy)
        vxy = self.cov_norm * (uxy - ux * uy)
        A1, A2, B1, B2 = (
            2 * ux * uy + C1,
            2 * vxy + C2,
            ux ** 2 + uy ** 2 + C1,
            vx + vy + C2,
        )
        D = B1 * B2
        S = (A1 * A2) / D

        return 1 - S.mean()

class kspace_loss(nn.Module):
    def __init__(self, loss_type="L2"):
        # args: loss_type: L2 loss or Cosine distance loss
        super().__init__()
        if loss_type == "L2":
            self.loss_func = nn.MSELoss()
        elif loss_type == "Cosine":
            self.loss_func = nn.CosineEmbeddingLoss()
    def forward(self, x, y):
        loss = self.loss_func(x,y)
        return loss.mean()
    
class total_loss(nn.Module):
    def __init__(self, win_size: int = 7, k1: float = 0.01, k2: float = 0.03, loss_type = "L2", alpha = 0.5):
        super().__init__()
        self.loss_func1 = SSIMLoss(win_size, k1, k2)
        self.loss_func2 = kspace_loss(loss_type)
        self.alpha = alpha
    def forward(self, x, y, kspace_x, kspace_y, data_range):
        # x: predicted (image, kspace)
        loss1 = self.loss_func1(x, y, data_range)
        loss2 = self.loss_func2(kspace_x, kspace_y)
        loss = self.alpha*loss1 + (1-self.alpha)*loss2
        return loss
        
    