from typing import Optional, Union

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

import segmentation_models_pytorch as smp

# Local imports 
import initialization as init
import gpu_tensor_funcs as gtf

#-------------------------------------------------------------------------------

class PoseRegressor(torch.nn.Module):

    # Inspired by 
    # https://github.com/qubvel/segmentation_models.pytorch/blob/1f1be174738703af225b6d7c5da90c6c04ce275b/segmentation_models_pytorch/base/model.py#L5
    # https://github.com/qubvel/segmentation_models.pytorch/blob/master/segmentation_models_pytorch/fpn/decoder.py
    # https://github.com/qubvel/segmentation_models.pytorch/blob/1f1be174738703af225b6d7c5da90c6c04ce275b/segmentation_models_pytorch/encoders/__init__.py#L32

    def __init__(
        self,
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

        # Obtain encoder
        self.encoder = smp.encoders.get_encoder(
            encoder_name,
            in_channels=in_channels,
            depth=encoder_depth,
            weights=encoder_weights,
        )

        # Obtain decoder
        if architecture == 'FPN':
            self.mask_decoder = smp.fpn.decoder.FPNDecoder(
                encoder_channels=self.encoder.out_channels,
                encoder_depth=encoder_depth,
                pyramid_channels=decoder_pyramid_channels,
                segmentation_channels=decoder_segmentation_channels,
                dropout=decoder_dropout,
                merge_policy=decoder_merge_policy,
            )

            self.quat_decoder = smp.fpn.decoder.FPNDecoder(
                encoder_channels=self.encoder.out_channels,
                encoder_depth=encoder_depth,
                pyramid_channels=decoder_pyramid_channels,
                segmentation_channels=decoder_segmentation_channels,
                dropout=decoder_dropout,
                merge_policy=decoder_merge_policy,
            )
        
        # Obtain segmentation head
        self.segmentation_head = smp.base.SegmentationHead(
            in_channels=self.mask_decoder.out_channels,
            out_channels=classes,
            activation=activation,
            kernel_size=1,
            upsampling=upsampling,
        )

        self.quat_head = smp.base.SegmentationHead(
            in_channels=self.quat_decoder.out_channels,
            out_channels=4*(classes-1), # Removing the background
            activation=activation,
            kernel_size=1,
            upsampling=upsampling,
        )

        # initialize the network
        init.initialize_decoder(self.mask_decoder)
        init.initialize_head(self.segmentation_head)

        init.initialize_decoder(self.quat_decoder)
        init.initialize_head(self.quat_head)

    def forward(self, x):

        # Encoder
        features = self.encoder(x)
        
        # Decoders
        mask_decoder_output = self.mask_decoder(*features)
        quat_decoder_output = self.quat_decoder(*features)

        # Heads 
        mask_logits = self.segmentation_head(mask_decoder_output)
        quat_logits = self.quat_head(quat_decoder_output)

        # Convert ouputs logits to categorical and class compress data
        # Compressing the quaternion output to account for the segmentation output
        quaternion, cat_mask = gtf.class_compress_quaternion(
            mask_logits=mask_logits,
            quaternion=quat_logits
        )

        # Logits
        output = {
            'mask': mask_logits,
            'quaternion': quaternion,
        }

        # Aggregating predictions
        agg_pred = gtf.dense_class_data_aggregation(
            mask=cat_mask,
            dense_class_data=output
        )

        # Attaching all non-logits outputs
        output['auxilary'] = {
            'cat_mask': cat_mask,
            'agg_pred': agg_pred
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

