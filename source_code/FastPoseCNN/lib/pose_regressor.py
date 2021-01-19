from typing import Optional, Union
import os
import base64

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

import segmentation_models_pytorch as smp

# Local imports 
import initialization as init
import gpu_tensor_funcs as gtf
import aggregation_layer as al

#-------------------------------------------------------------------------------

class PoseRegressor(torch.nn.Module):

    # Inspired by 
    # https://github.com/qubvel/segmentation_models.pytorch/blob/1f1be174738703af225b6d7c5da90c6c04ce275b/segmentation_models_pytorch/base/model.py#L5
    # https://github.com/qubvel/segmentation_models.pytorch/blob/master/segmentation_models_pytorch/fpn/decoder.py
    # https://github.com/qubvel/segmentation_models.pytorch/blob/1f1be174738703af225b6d7c5da90c6c04ce275b/segmentation_models_pytorch/encoders/__init__.py#L32

    def __init__(
        self,
        HPARAM,
        intrinsics: torch.Tensor,
        architecture: str = 'FPN',
        encoder_name: str = "resnet34",
        encoder_depth: int = 5,
        encoder_weights: Optional[str] = "imagenet",
        decoder_pyramid_channels: int = 256,
        decoder_segmentation_channels: int = 128,
        decoder_merge_policy: str = "add",
        decoder_dropout: float = 0.2,
        in_channels: int = 3,
        classes: int = 2, # bg and one more class
        activation: Optional[str] = None,
        upsampling: int = 4
        ):

        super().__init__()

        # Storing crucial parameters
        self.HPARAM = HPARAM # other algorithm and run hyperparameters
        self.classes = classes # includes background
        self.intrinsics = intrinsics

        # Obtain encoder
        self.encoder = smp.encoders.get_encoder(
            encoder_name,
            in_channels=in_channels,
            depth=encoder_depth,
            weights=encoder_weights,
        )

        # Obtain decoder
        if architecture == 'FPN':

            param_dict = {
                'encoder_channels': self.encoder.out_channels,
                'encoder_depth': encoder_depth,
                'pyramid_channels': decoder_pyramid_channels,
                'segmentation_channels': decoder_segmentation_channels,
                'dropout': decoder_dropout,
                'merge_policy': decoder_merge_policy,
            }

            self.mask_decoder = smp.fpn.decoder.FPNDecoder(**param_dict)
            self.rotation_decoder = smp.fpn.decoder.FPNDecoder(**param_dict)
            self.translation_decoder = smp.fpn.decoder.FPNDecoder(**param_dict)
            self.scales_decoder = smp.fpn.decoder.FPNDecoder(**param_dict)

        # Obtain segmentation head
        self.segmentation_head = smp.base.SegmentationHead(
            in_channels=self.mask_decoder.out_channels,
            out_channels=classes,
            activation=activation,
            kernel_size=1,
            upsampling=upsampling,
        )

        # Creating rotation head (quaternion or rotation matrix)
        self.rotation_head = smp.base.SegmentationHead(
            in_channels=self.rotation_decoder.out_channels,
            out_channels=4*(classes-1), # Removing the background
            activation=activation,
            kernel_size=1,
            upsampling=upsampling,
        )

        # Creating translation head (xyz)
        self.translation_head = smp.base.SegmentationHead(
            in_channels=self.translation_decoder.out_channels,
            out_channels=3*(classes-1), # Removing the background
            activation=activation,
            kernel_size=1,
            upsampling=upsampling,
        )

        # Creating scales head (height, width, and length)
        self.scales_head = smp.base.SegmentationHead(
            in_channels=self.scales_decoder.out_channels,
            out_channels=3*(classes-1), # Removing the background
            activation=activation,
            kernel_size=1,
            upsampling=upsampling,
        )

        # Creating aggregation layer
        self.aggregation_layer = al.AggregationLayer(
            self.HPARAM, 
            self.classes, 
            self.intrinsics
        )

        # initialize the network
        init.initialize_decoder(self.mask_decoder)
        init.initialize_head(self.segmentation_head)

        init.initialize_decoder(self.rotation_decoder)
        init.initialize_head(self.rotation_head)

        init.initialize_decoder(self.translation_decoder)
        init.initialize_head(self.translation_head)

        init.initialize_decoder(self.scales_decoder)
        init.initialize_head(self.scales_head)

    def forward(self, x):

        # Encoder
        features = self.encoder(x)
        
        # Decoders
        mask_decoder_output = self.mask_decoder(*features)
        rotation_decoder_output = self.rotation_decoder(*features)
        translation_decoder_output = self.translation_decoder(*features)
        scales_decoder_output = self.scales_decoder(*features)

        # Heads 
        mask_logits = self.segmentation_head(mask_decoder_output)
        quat_logits = self.rotation_head(rotation_decoder_output)
        xyz_logits = self.translation_head(translation_decoder_output)
        scales_logits = self.scales_head(scales_decoder_output)

        # Spliting the (xyz) to (xy, z) since they will eventually have different
        # ways of computing the loss.
        xy_index = np.array([i for i in range(xyz_logits.shape[1]) if i%3!=0]) - 1
        z_index = np.array([i for i in range(xyz_logits.shape[1]) if i%3==0]) + 2
        xy_logits = xyz_logits[:,xy_index,:,:]
        z_logits = xyz_logits[:,z_index,:,:]

        # Storing all logits in a dictionary
        logits = {
            'quaternion': quat_logits,
            'scales': scales_logits,
            'xy': xy_logits,
            'z': z_logits
        }

        # Create categorical mask
        cat_mask = torch.argmax(torch.nn.LogSoftmax(dim=1)(mask_logits), dim=1)

        # Aggregating the results
        agg_pred, cc_logits = self.aggregation_layer.forward(
            cat_mask, 
            logits=logits
        )

        # Generating complete output
        output = {
            'mask': mask_logits,
            **cc_logits,
            'auxilary': {
                'cat_mask': cat_mask,
                'agg_pred': agg_pred
            }
        }

        return output

#-------------------------------------------------------------------------------
# File Main

if __name__ == '__main__':

    # Load the model
    model = PoseRegressor()

    # Forward propagate
    x1 = np.random.random_sample((3,254,254))
    #x2 = np.random.random_sample((3,254,254))
    
    x1 = torch.from_numpy(x1).unsqueeze(0).float()
    #x2 = torch.from_numpy(x2).unsqueeze(0).float()
    
    #x = torch.cat([x1, x2])
    x = x1

    y = model.forward(x)

    print(y)

