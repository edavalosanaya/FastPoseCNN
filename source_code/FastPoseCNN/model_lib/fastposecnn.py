import os
import sys
import pathlib

import cv2
import numpy as np

# Torch imports
import torch
import torch.nn as nn
import torch.nn.functional as F

# Local Imports
root = next(path for path in pathlib.Path(os.path.abspath(__file__)).parents if path.name == 'FastPoseCNN')
sys.path.append(str(root))

import model_lib.layers as L
import tools.project as project

#-------------------------------------------------------------------------------
# File Constants

test_image = project.cfg.DATASET_DIR / 'NOCS' / 'camera' / 'val' / '00000' / '0000_color.png'

#-------------------------------------------------------------------------------
# Classes

class FastPoseCNN(nn.Module):

    def __init__(self, n_channels=3, out_channels_id = 6, bilinear=True):
        super().__init__()
        
        """
        This function is for creating all the layers and block of FastPoseCNN
        """

        # Saving input arguments
        self.n_channels = n_channels
        self.out_channels_id = out_channels_id
        self.bilinear = bilinear

        # Encoder phase
        self.in_conv = L.DoubleConv(n_channels, 64)
        self.down1 = L.Down(64, 128)
        self.down2 = L.Down(128, 256)
        self.down3 = L.Down(256, 512)
        factor = 2 if bilinear else 1
        self.down4 = L.Down(512, 1024//factor)

        # ID MASK Decoder 
        self.up1_id = L.Up(1024, 512, bilinear)
        self.up2_id = L.Up(512, 256, bilinear)
        self.up3_id = L.Up(256, 128, bilinear)
        self.up4_id = L.Up(128, 64 * factor, bilinear)
        self.out_conv_id = L.OutConv(64, out_channels_id)

    def forward(self, x):

        # Encoder Phase
        x1 = self.in_conv(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)

        # ID MASK Decoder
        x_id = self.up1_id(x5, x4)
        x_id = self.up2_id(x_id, x3)
        x_id = self.up3_id(x_id, x2)
        x_id = self.up4_id(x_id, x1)
        logits_id = self.out_conv_id(x_id)

        return logits_id

#-------------------------------------------------------------------------------
# Functions

#-------------------------------------------------------------------------------
# File Main

if __name__ == '__main__':

    # Creating testing input
    img = cv2.imread(str(test_image), cv2.IMREAD_UNCHANGED)

    print(f'img.shape: {img.shape}')

    img = np.moveaxis(img, -1, 0) # Requires (D, W, H) instead of (W, H, D)
    img = img.astype(np.float32) # Requires Byte (np.float) instead of integer type (np.int)
    img = torch.from_numpy(img) # Requires torch.Tensor instead of np.array
    img = img.unsqueeze(0) # Requires a dimension (B, D, W, H)

    print(f'tensor img.shape: {img.shape}')

    net = FastPoseCNN()

    output = net(img)

    print(f'output.shape: {output.shape}')


    