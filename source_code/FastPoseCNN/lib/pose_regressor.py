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

        # Storing crucial parameters
        self.classes = classes # includes background

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
            in_channels=self.rotation_decoder.out_channels,
            out_channels=3*(classes-1), # Removing the background
            activation=activation,
            kernel_size=1,
            upsampling=upsampling,
        )

        # initialize the network
        init.initialize_decoder(self.mask_decoder)
        init.initialize_head(self.segmentation_head)

        init.initialize_decoder(self.rotation_decoder)
        init.initialize_head(self.rotation_head)

        init.initialize_decoder(self.translation_decoder)
        init.initialize_head(self.translation_head)

    def forward(self, x):

        # Encoder
        features = self.encoder(x)
        
        # Decoders
        mask_decoder_output = self.mask_decoder(*features)
        rotation_decoder_output = self.rotation_decoder(*features)
        translation_decoder_output = self.translation_decoder(*features)

        # Heads 
        mask_logits = self.segmentation_head(mask_decoder_output)
        quat_logits = self.rotation_head(rotation_decoder_output)
        xyz_logits = self.translation_head(translation_decoder_output)

        # Spliting the (xyz) to (xy, z) since they will eventually have different
        # ways of computing the loss.
        xy_index = np.array([i for i in range(xyz_logits.shape[1]) if i%3!=0]) - 1
        z_index = np.array([i for i in range(xyz_logits.shape[1]) if i%3==0]) + 2
        xy_logits = xyz_logits[:,xy_index,:,:]
        z_logits = xyz_logits[:,z_index,:,:]

        # Create categorical mask
        cat_mask = torch.argmax(torch.nn.LogSoftmax(dim=1)(mask_logits), dim=1)

        # Class Compressing (cc) quaternions by masking it with the categorical mask
        # Compressing the quaternion output to account for the segmentation output
        cc_quaternion = gtf.class_compress(
            num_of_classes = self.classes - 1,
            cat_mask = cat_mask,
            data = quat_logits
        )

        cc_xy = gtf.class_compress(
            num_of_classes = self.classes - 1,
            cat_mask = cat_mask,
            data = xy_logits
        )

        cc_z = gtf.class_compress(
            num_of_classes = self.classes - 1,
            cat_mask = cat_mask,
            data = z_logits
        )

        # Squeezing the cc_z (Nx1xHxW) to (NxHxW) to match with ground truth
        cc_z = torch.squeeze(cc_z)

        # Logits
        output = {
            'mask': mask_logits,
            'quaternion': cc_quaternion,
            'xy': cc_xy,
            'z': cc_z
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

