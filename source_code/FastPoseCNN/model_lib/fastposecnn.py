import os
import sys
import pathlib

import pdb

import cv2
import numpy as np
import skimage.io

# Torch imports
import torch
import torch.nn as nn
import torch.nn.functional as F

# Local Imports
root = next(path for path in pathlib.Path(os.path.abspath(__file__)).parents if path.name == 'FastPoseCNN')
sys.path.append(str(root))

import model_lib.layers as L
import project
import dataset
import data_manipulation
import visualize

#-------------------------------------------------------------------------------
# File Constants

test_image = project.cfg.DATASET_DIR / 'NOCS' / 'camera' / 'val' / '00000' / '0000_color.png'
CAMERA_DATASET = root.parents[1] / 'datasets' / 'NOCS' / 'camera' / 'val'

#-------------------------------------------------------------------------------
# Encoder

class Encoder(nn.Module):

    def __init__(self, in_channels, planes, bilinear=True):
        super().__init__()

        factor = 2 if bilinear else 1

        self.layers = nn.ModuleDict({})
        for id, plane in enumerate(planes):
            
            if id == 0: # Initial layer
                self.layers.update(nn.ModuleDict({f'L{id}': L.DoubleConv(in_channels, planes[0])}))

            elif id == len(planes)-1: # Last Layer
                self.layers.update(nn.ModuleDict({f'L{id}': L.Down(planes[id-1], planes[id]//factor)}))
            
            else:
                self.layers.update(nn.ModuleDict({f'L{id}': L.Down(planes[id-1], planes[id])}))

    def forward(self, x):

        xs = nn.ParameterList([])

        for layer_name in self.layers.keys():
            x = nn.Parameter(self.layers[layer_name](x))
            xs.append(x)

        return xs

#-------------------------------------------------------------------------------
# Decoders

class GenericDecoder(nn.Module):

    def __init__(self, in_channels, out_channels, planes, bilinear=True):
        super().__init__()

        factor = 2 if bilinear else 1

        reversed_planes = planes.copy()
        reversed_planes.reverse()

        self.layers = nn.ModuleDict({})
        for id, plane in enumerate(reversed_planes):

            if id == len(planes)-1: # Last Layer (DEFAULT BUT NOT REQUIRED TO BE USED)
                self.layers.update(nn.ModuleDict({f'Last': L.OutConv(reversed_planes[id], out_channels)}))

            elif id == len(planes)-2: # Second to Last Layer
                self.layers.update(nn.ModuleDict({f'L{id}': L.Up(reversed_planes[id], reversed_planes[id+1] * factor)}))
            
            else:
                self.layers.update(nn.ModuleDict({f'L{id}': L.Up(reversed_planes[id], reversed_planes[id+1])}))

class MaskDecoder(nn.Module):

    def __init__(self, in_channels, out_channels, planes, bilinear=True):
        super().__init__()

        factor = 2 if bilinear else 1

        self.generic_decoder = GenericDecoder(in_channels, out_channels, planes, bilinear)

    def forward(self, x):

        out = x[-1]

        for layer_id, x_id in enumerate(range(-2, -len(x)-1, -1)):
            out = self.generic_decoder.layers[f'L{layer_id}'](out, x[x_id])

        # Last Layer
        logits_mask = self.generic_decoder.layers['Last'](out)

        return logits_mask

class QuatDecoder(nn.Module):

    def __init__(self, in_channels, out_channels, planes, bilinear=True):
        super().__init__()

        factor = 2 if bilinear else 1

        self.generic_decoder = GenericDecoder(in_channels, out_channels, planes, bilinear)

    def forward(self, x):

        out = x[-1]

        for layer_id, x_id in enumerate(range(-2, -len(x)-1, -1)):
            out = self.generic_decoder.layers[f'L{layer_id}'](out, x[x_id])

        # Last Layer
        x_quat = self.generic_decoder.layers['Last'](out)

        return x_quat

class ScaleDecoder(nn.Module):

    def __init__(self, in_channels, out_channels, planes, bilinear=True):
        super().__init__()

        factor = 2 if bilinear else 1

        self.generic_decoder = GenericDecoder(in_channels, out_channels, planes, bilinear)

    def forward(self, x):

        out = x[-1]

        for layer_id, x_id in enumerate(range(-2, -len(x)-1, -1)):
            out = self.generic_decoder.layers[f'L{layer_id}'](out, x[x_id])

        # Last Layer
        x_scale = self.generic_decoder.layers['Last'](out)

        return x_scale

class DepthDecoder(nn.Module):

    def __init__(self, in_channels, out_channels, planes, bilinear=True):
        super().__init__()

        factor = 2 if bilinear else 1

        self.generic_decoder = GenericDecoder(in_channels, out_channels, planes, bilinear)

    def forward(self, x):

        out = x[-1]

        for layer_id, x_id in enumerate(range(-2, -len(x)-1, -1)):
            out = self.generic_decoder.layers[f'L{layer_id}'](out, x[x_id])

        # Last Layer
        x_depth = self.generic_decoder.layers['Last'](out)

        return x_depth

#-------------------------------------------------------------------------------
# Main Module Class

class FastPoseCNN(nn.Module):

    def __init__(self, in_channels=3, num_classes=len(project.constants.NOCS_CLASSES),
                 filter_factor=1, planes=[64,128,256,512,1024], bilinear=True):
        super().__init__()
        
        """
        This function is for creating all the layers and block of FastPoseCNN
        """

        # Saving model name
        self.name = 'fastposecnn'

        # Saving input arguments
        self.in_channels = in_channels
        self.bilinear = bilinear
        self.planes = [plane / filter_factor for plane in planes]

        # Encoder Phase
        self.encoder = Encoder(in_channels, planes, bilinear)

        # Mask Decoder
        self.mask_decoder = MaskDecoder(in_channels, num_classes, planes, bilinear)

        """
        # Quaternion Decoders (real, i, j, and k) components
        self.quat_decoder = nn.ModuleDict({
            'real': QuatDecoder(in_channels, 1, planes, bilinear),
            'i-com': QuatDecoder(in_channels, 1, planes, bilinear),
            'j-com': QuatDecoder(in_channels, 1, planes, bilinear),
            'k-com': QuatDecoder(in_channels, 1, planes, bilinear)
        })

        # Scale decoders (w, h, l) components
        self.scale_decoder = nn.ModuleDict({
            'w-com': ScaleDecoder(in_channels, 1, planes, bilinear),
            'h-com': ScaleDecoder(in_channels, 1, planes, bilinear),
            'l-com': ScaleDecoder(in_channels, 1, planes, bilinear)
        })

        # Depth decoder
        self.depth_decoder = DepthDecoder(in_channels, 1, planes, bilinear)
        #"""

    def forward(self, x):

        ########################################################################
        #                             Encoder Phase
        ########################################################################
        #print('Entering encoder')

        xs = self.encoder(x)

        #print('Finished encoder')
        ########################################################################
        #                             Decoder Phase
        ########################################################################
        
        # Mask
        logits_mask = self.mask_decoder(xs)

        """
        # Depth
        logits_depth = self.depth_decoder(xs)

        # Scales
        scale_out = nn.ParameterDict({'w-com': nn.Parameter(torch.tensor([0.], dtype=torch.float, device='cuda')), 
                                      'h-com': nn.Parameter(torch.tensor([0.], dtype=torch.float, device='cuda')), 
                                      'l-com': nn.Parameter(torch.tensor([0.], dtype=torch.float, device='cuda'))})
        for key in self.scale_decoder.keys():
            scale_out[key] = nn.Parameter(self.scale_decoder[key](xs))

        out_tuple = (scale_out['w-com'], scale_out['h-com'], scale_out['l-com'])
        logits_scale = torch.cat(out_tuple, dim=1)

        # Quaternion
        quat_out = nn.ParameterDict({'real': nn.Parameter(torch.tensor([0.], dtype=torch.float, device='cuda')), 
                                     'i-com': nn.Parameter(torch.tensor([0.], dtype=torch.float, device='cuda')), 
                                     'j-com': nn.Parameter(torch.tensor([0.], dtype=torch.float, device='cuda')), 
                                     'k-com': nn.Parameter(torch.tensor([0.], dtype=torch.float, device='cuda'))})
        for key in self.quat_decoder.keys():
            quat_out[key] = nn.Parameter(self.quat_decoder[key](xs))

        out_tuple = (quat_out['real'], quat_out['i-com'], quat_out['j-com'], quat_out['k-com'])
        logits_quat = torch.cat(out_tuple, dim=1)
        #"""

        return logits_mask

#-------------------------------------------------------------------------------
# Functions

#-------------------------------------------------------------------------------
# File Main

if __name__ == '__main__':

    # This is an simple test case, not for training

    # Loading complete dataset
    complete_dataset = dataset.NOCSDataset(CAMERA_DATASET, 100)

    # Splitting dataset to train and validation
    dataset_num = len(complete_dataset)
    split = 0.2
    train_length, valid_length = int(dataset_num*(1-split)), int(dataset_num*split)

    train_dataset, valid_dataset = torch.utils.data.random_split(complete_dataset,
                                                                [train_length, valid_length])

    # Specifying the criterions
    criterions = {'masks':torch.nn.CrossEntropyLoss(),
                  'depth':torch.nn.BCEWithLogitsLoss(),
                  'scales':torch.nn.BCEWithLogitsLoss(),
                  'quat':torch.nn.BCEWithLogitsLoss()}

    # Creating a Trainer
    my_trainer = trainer.Trainer(FastPoseCNN(in_channels=3, bilinear=True), 
                                 train_dataset,
                                 valid_dataset,
                                 criterions)

    # Testing Neural Network
    my_trainer.test_forward()



    