import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

import pretrainedmodels as ptm

# Local imports

import custom_layers as cl

#-------------------------------------------------------------------------------

class PoseRegressor(nn.Module):
    def __init__(self, backbone='resnet18', encoder_weights='imagenet'):
        super().__init__()

        # Using this as reference for image regression models
        # https://medium.com/analytics-vidhya/fastai-image-regression-age-prediction-based-on-image-68294d34f2ed

        # Obtain all the layers of the backbone except the last two
        backbone = ptm.__dict__[backbone](pretrained=encoder_weights)
        backbone_layers = list(backbone.children())[:-2]

        self.backbone = nn.Sequential(*backbone_layers)

        # Add new layers
        self.regressor_head = nn.Sequential(
            cl.AdaptiveConcatPool2d(),
            nn.Flatten(),
            nn.BatchNorm1d(1024, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.Dropout(p=0.5),
            nn.Linear(1024, 512, bias=True),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.Dropout(p=0.5),
            nn.Linear(512, 16, bias=True),
            nn.ReLU(inplace=True),
            nn.Linear(16, 4, bias=True),
        )

        # Add new layers
        self.single_batch_regressor_head = nn.Sequential(
            cl.AdaptiveConcatPool2d(),
            nn.Flatten(),
            nn.Dropout(p=0.5),
            nn.Linear(1024, 512, bias=True),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(512, 16, bias=True),
            nn.ReLU(inplace=True),
            nn.Linear(16, 4, bias=True),
        )

    def forward(self, x):
        # Passing through the backbone
        backbone_x = self.backbone(x)
        
        # If batch size equals 1, avoid using nn.BatchNorm1d (it causes error,
        # requires batch size >= 2).
        if backbone_x.shape[0] == 1:
            logits = self.single_batch_regressor_head(backbone_x)
        else:
            logits = self.regressor_head(backbone_x)

        # Return logits
        return logits

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

