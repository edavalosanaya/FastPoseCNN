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
import tools.project as project
import tools.dataset
import tools.data_manipulation
import tools.visualize

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

    def __init__(self, in_channels=3, num_classes=len(project.constants.SYNSET_NAMES), planes=[64,128,256,512,1024], bilinear=True):
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

        return logits_mask #, logits_depth, logits_scale, logits_quat

#-------------------------------------------------------------------------------
# Functions

#-------------------------------------------------------------------------------
# File Main

if __name__ == '__main__':

    # Loading dataset
    dataset = tools.dataset.NOCSDataset(camera_dataset, dataset_max_size=10)

    # Creating data loaders
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, num_workers=0)

    # Creating model
    net = FastPoseCNN(in_channels=3, bilinear=True)

    # Selecting a criterions
    criterions = {'masks':torch.nn.CrossEntropyLoss(),
                  'depth':torch.nn.BCEWithLogitsLoss(),
                  'scales':torch.nn.BCEWithLogitsLoss(),
                  'quat':torch.nn.BCEWithLogitsLoss()}

    # Creating correct environment
    
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    #if torch.cuda.device_count() > 1:
    #    print(f'Using {torch.cuda.device_count()} GPUs!')
    #    net = torch.nn.DataParallel(net)
    
    # Moving net to device
    net.to(device)

    # Loading input
    sample = next(iter(dataloader))
    color_image, depth_image, zs, masks, coord_map, scales_img, quat_img = sample

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

    # Visualizing output
    pred = torch.argmax(outputs, dim=1)[0]
    vis_pred = tools.visualize.get_visualized_mask(pred)
    numpy_vis_pred = tools.visualize.torch_to_numpy([vis_pred])[0]
    out_path = project.cfg.TEST_OUTPUT / 'segmentation_output.png'
    skimage.io.imsave(str(out_path), numpy_vis_pred)

    # Testing summary image
    summary_image = tools.visualize.make_summary_image('Summary', sample, outputs)
    out_path = project.cfg.TEST_OUTPUT / 'summary_image.png'
    skimage.io.imsave(str(out_path), summary_image)

    # Loss propagate
    loss_mask = criterions['masks'](outputs, masks)
    """
    loss_depth = criterions['depth'](outputs[1], depth_image)
    loss_scale = criterions['scales'](outputs[2], scales_img)
    loss_quat = criterions['quat'](outputs[3], quat_img)
    #"""
    print('Successful backprogagation')



    