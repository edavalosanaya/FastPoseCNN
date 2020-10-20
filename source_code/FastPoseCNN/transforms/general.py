import albumentations as albu
from albumentations.pytorch import ToTensor

#-------------------------------------------------------------------------------
# General Functions

def to_tensor(x, **kwargs):
    return x.transpose(2,0,1) if len(x.shape) == 3 else x

