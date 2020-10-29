import math
import torch
import torch.nn as nn

# Local imports
from unet import UNet

class UNetWrapper(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()

        self.name = 'unet-wrapper'

        self.input_batchnorm = nn.BatchNorm2d(kwargs['in_channels'])
        self.unet = UNet(**kwargs)
        self.final = nn.Sigmoid()

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if type(m) in (nn.Linear, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight.data, a=0, mode='fan_out', 
                                        nonlinearity='relu')
                if m.bias is not None:
                    fan_in, fan_out = nn.init._calculate_fan_in_and_fan_out(m.weight.data)
                    bound = 1 / math.sqrt(fan_out)
                    nn.init.normal_(m.bias, -bound, bound)

    def forward(self, x):
        bn_output = self.input_batchnorm(x)
        logits_mask = self.unet(x)
        
        logits_mask = self.final(logits_mask)

        return logits_mask
        