import os
import sys
import pathlib

import pdb

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
import tools.dataset
import tools.data_manipulation

#-------------------------------------------------------------------------------
# File Constants

test_image = project.cfg.DATASET_DIR / 'NOCS' / 'camera' / 'val' / '00000' / '0000_color.png'
camera_dataset = root.parents[1] / 'datasets' / 'NOCS' / 'camera' / 'val'

#-------------------------------------------------------------------------------
# Encoder

class Encoder(nn.Module):

    def __init__(self, in_channels, planes, bilinear=True):
        super().__init__()

        factor = 2 if bilinear else 1

        self.in_conv = L.DoubleConv(in_channels, 64)
        self.down1 = L.Down(planes[0], planes[1])
        self.down2 = L.Down(planes[1], planes[2])
        self.down3 = L.Down(planes[2], planes[3])
        self.down4 = L.Down(planes[3], planes[4]//factor)

    def forward(self, x):

        x1 = self.in_conv(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)

        return x1, x2, x3, x4, x5

#-------------------------------------------------------------------------------
# Decoders

class MaskDecoder(nn.Module):

    def __init__(self, in_channels, out_channels, planes, bilinear=True):
        super().__init__()

        factor = 2 if bilinear else 1

        self.up1_mask = L.Up(planes[4], planes[3], bilinear)
        self.up2_mask = L.Up(planes[3], planes[2], bilinear)
        self.up3_mask = L.Up(planes[2], planes[1], bilinear)
        self.up4_mask = L.Up(planes[1], planes[0] * factor, bilinear)
        self.out_conv_mask = L.OutConv(planes[0], out_channels)

    def forward(self, x):

        x1, x2, x3, x4, x5 = x # Breaking up tuple

        x_mask = self.up1_mask(x5, x4)
        x_mask = self.up2_mask(x_mask, x3)
        x_mask = self.up3_mask(x_mask, x2)
        x_mask = self.up4_mask(x_mask, x1)
        logits_mask = self.out_conv_mask(x_mask)

        return logits_mask

class QuatDecoder(nn.Module):

    def __init__(self, in_channels, out_channels, planes, bilinear=True):
        super().__init__()

        factor = 2 if bilinear else 1

        self.up1_quat = L.Up(planes[4], planes[3], bilinear)
        self.up2_quat = L.Up(planes[3], planes[2], bilinear)
        self.up3_quat = L.Up(planes[2], planes[1], bilinear)
        self.up4_quat = L.Up(planes[1], planes[0] * factor, bilinear)
        self.out_conv_quat = L.OutConv(planes[0], out_channels)

    def forward(self, x):

        x1, x2, x3, x4, x5 = x # Breaking up tuple

        x_quat = self.up1_quat(x5, x4)
        x_quat = self.up2_quat(x_quat, x3)
        x_quat = self.up3_quat(x_quat, x2)
        x_quat = self.up4_quat(x_quat, x1)
        x_quat = self.out_conv_quat(x_quat)

        return x_quat

class ScaleDecoder(nn.Module):

    def __init__(self, in_channels, out_channels, planes, bilinear=True):
        super().__init__()

        factor = 2 if bilinear else 1

        self.up1_scale = L.Up(planes[4], planes[3], bilinear)
        self.up2_scale = L.Up(planes[3], planes[2], bilinear)
        self.up3_scale = L.Up(planes[2], planes[1], bilinear)
        self.up4_scale = L.Up(planes[1], planes[0] * factor, bilinear)
        self.out_conv_scale = L.OutConv(planes[0], out_channels)

    def forward(self, x):

        x1, x2, x3, x4, x5 = x # Breaking up tuple

        x_scale = self.up1_scale(x5, x4)
        x_scale = self.up2_scale(x_scale, x3)
        x_scale = self.up3_scale(x_scale, x2)
        x_scale = self.up4_scale(x_scale, x1)
        x_scale = self.out_conv_scale(x_scale)

        return x_scale

class DepthDecoder(nn.Module):

    def __init__(self, in_channels, out_channels, planes, bilinear=True):
        super().__init__()

        factor = 2 if bilinear else 1

        self.up1_depth = L.Up(planes[4], planes[3], bilinear)
        self.up2_depth = L.Up(planes[3], planes[2], bilinear)
        self.up3_depth = L.Up(planes[2], planes[1], bilinear)
        self.up4_depth = L.Up(planes[1], planes[0] * factor, bilinear)
        self.out_conv_depth = L.OutConv(planes[0], out_channels)

    def forward(self, x):

        x1, x2, x3, x4, x5 = x # Breaking up tuple

        x_depth = self.up1_depth(x5, x4)
        x_depth = self.up2_depth(x_depth, x3)
        x_depth = self.up3_depth(x_depth, x2)
        x_depth = self.up4_depth(x_depth, x1)
        x_depth = self.out_conv_depth(x_depth)

        return x_depth

#-------------------------------------------------------------------------------
# Main Module Class

class FastPoseCNN(nn.Module):

    def __init__(self, in_channels=3, planes=[64,128,256,512,1024], bilinear=True):
        super().__init__()
        
        """
        This function is for creating all the layers and block of FastPoseCNN
        """

        # Saving input arguments
        self.in_channels = in_channels
        self.bilinear = bilinear
        self.planes = planes

        # Encoder Phase
        self.encoder = Encoder(in_channels, planes, bilinear)

        # Mask Decoder
        self.mask_decoder = MaskDecoder(in_channels, 1, planes, bilinear)

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

        return logits_mask, logits_depth, logits_scale, logits_quat

#-------------------------------------------------------------------------------
# Functions

#-------------------------------------------------------------------------------
# File Main

if __name__ == '__main__':

    # Loading dataset
    print('Loading dataset')
    dataset = tools.dataset.NOCSDataset(camera_dataset, dataset_max_size=10)

    # Creating data loaders
    print('Creating dataloader')
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, num_workers=0)

    # Creating model
    print('Loading NN model')
    net = FastPoseCNN(in_channels=3, bilinear=True)

    # Selecting a criterions
    print('Selecing criterions')
    criterions = {'masks':torch.nn.BCEWithLogitsLoss(),
                  'depth':torch.nn.BCEWithLogitsLoss(),
                  'scales':torch.nn.BCEWithLogitsLoss(),
                  'quat':torch.nn.BCEWithLogitsLoss()}

    # Creating correct environment
    
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    #if torch.cuda.device_count() > 1:
    #    print(f'Using {torch.cuda.device_count()} GPUs!')
    #    net = torch.nn.DataParallel(net)
    
    print(f'Moving net to device: {device}')
    net.to(device)

    # Loading input
    print('Loading single sample')
    sample = next(iter(dataloader))
    color_image, depth_image, zs, masks, coord_map, scales_img, quat_img = sample
    print('Finished loading sample')

    # Model filename
    model_name = 'fastposecnn-design-test'
    model_save_filepath = str(project.cfg.SAVED_MODEL_DIR / (model_name + '.pth') )
    model_logs_dir = str(project.cfg.LOGS / model_name)

    # Creating tensorboard object
    tensorboard_writer = torch.utils.tensorboard.SummaryWriter(model_logs_dir)

    # Saving graph model to Tensorboard
    #tensorboard_writer.add_graph(net, color_image)

    # Forward pass
    print('Forward pass')
    outputs = net(color_image)

    for output, output_type in zip(outputs, ['mask', 'depth_image', 'scales_img', 'quat_img']):
        print(f'Output: {output_type}')
        print(f'max: {torch.max(output)} min: {torch.min(output)} mean: {torch.mean(output)}')

    # Loss propagate
    loss_mask = criterions['masks'](outputs[0], masks)
    loss_depth = criterions['depth'](outputs[1], depth_image)
    loss_scale = criterions['scales'](outputs[2], scales_img)
    loss_quat = criterions['quat'](outputs[3], quat_img)
    print('Successful backprogagation')



    