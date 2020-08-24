import os
import sys
import pathlib

import cv2
import numpy as np

import pdb

# Torch imports
import torch
import torch.nn as nn
import torch.nn.functional as F

# Local Imports
root = next(path for path in pathlib.Path(os.path.abspath(__file__)).parents if path.name == 'FastPoseCNN')
sys.path.append(str(root))

# Most code reference from 
# (DoubleConv, Down, Up, OutConv) 
# link: https://github.com/yshah43/DPOD/blob/master/unet_model.py
# (unetConv2 and unetUp) 
# link: https://github.com/meetshah1995/pytorch-semseg/blob/master/ptsemseg/models/utils.py

#-------------------------------------------------------------------------------
# File Constants

#-------------------------------------------------------------------------------
# Classes

class DoubleConv(nn.Module):
    """
    (convolution => [BN] => ReLU) * 2
    """

    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()

        if not mid_channels:
            mid_channels = out_channels

        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)

class Down(nn.Module):
    """
    Downscaling with maxpool then double conv
    """

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)

class Up(nn.Module):
    """
    Upscaling then double conv
    """

    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()

        # If bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels // 2, in_channels // 2)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        # input is CHW
        diffY = torch.tensor([x2.size()[2] - x1.size()[2]])
        diffX = torch.tensor([x2.size()[3] - x1.size()[3]])

        x1 = F.pad(x1, [diffX // 2, diffX - diffY // 2,
                        diffY // 2, diffY - diffY // 2])

        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)

class OutConv(nn.Module):
    
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)

class unetConv2(nn.Module):
    def __init__(self, in_size, out_size, is_batchnorm):
        super(unetConv2, self).__init__()

        if is_batchnorm:
            self.conv1 = nn.Sequential(
                nn.Conv2d(in_size, out_size, kernel_size=3, stride=1, padding=1), 
                nn.BatchNorm2d(out_size), 
                nn.ReLU(inplace=True)
            )
            self.conv2 = nn.Sequential(
                nn.Conv2d(out_size, out_size, kernel_size=3, stride=1, padding=1), 
                nn.BatchNorm2d(out_size), 
                nn.ReLU(inplace=True)
            )
        else:
            self.conv1 = nn.Sequential(nn.Conv2d(in_size, out_size, kernel_size=3, stride=1, padding=1), 
                                       nn.ReLU(inplace=True))
            self.conv2 = nn.Sequential(nn.Conv2d(out_size, out_size, kernel_size=3, stride=1, padding=1), 
                                       nn.ReLU(inplace=True))

    def forward(self, inputs):
        outputs = self.conv1(inputs)
        outputs = self.conv2(outputs)
        return outputs

class unetUp(nn.Module):
    # https://github.com/meetshah1995/pytorch-semseg/issues/191#issuecomment-557950791
    def __init__(self, in_size, out_size, is_deconv):
        super(unetUp, self).__init__()
        self.conv = unetConv2(in_size, out_size, False)
        if is_deconv:
            self.up = nn.ConvTranspose2d(in_size, out_size, kernel_size=2, stride=2)
        else:
            self.up = nn.UpsamplingBilinear2d(scale_factor=2)

    def forward(self, inputs1, inputs2):
        outputs2 = self.up(inputs2)
        offsetY = outputs2.size()[2] - inputs1.size()[2]
        offsetX = outputs2.size()[3] - inputs1.size()[3]
        # padding = 2 * [offset // 2, offset // 2]
        # outputs1 = F.pad(inputs1, padding)
        outputs1 = nn.functional.pad(inputs1, (offsetX // 2, offsetX - offsetX//2,
                                    offsetY // 2, offsetY - offsetY//2))
        concat_inputs = torch.cat([outputs1, outputs2], dim=1)
        return self.conv(concat_inputs)

#-------------------------------------------------------------------------------
# Functions

#-------------------------------------------------------------------------------
# File Main

if __name__ == '__main__':
    pass