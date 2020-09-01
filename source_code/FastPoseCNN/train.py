import os
import sys
import pathlib
import warnings

import pdb

# Ignore annoying warnings
os.environ["TF_CPP_MIN_LOG_LEVEL"]="3"
warnings.filterwarnings('ignore')

import torch
import torch.nn
import torch.optim
import torch.utils
import torch.utils.tensorboard
import torchvision
import kornia # pytorch tensor data augmentation

# How to view tensorboard in the Lambda machine
"""
Do the following in Lamda machine: 

    tensorboard --logdir=logs --port 6006 --host=localhost

    tensorboard --logdir=model_lib/logs --port 6006 --host=localhost

Then run this on the local machine

    ssh -NfL 6006:localhost:6006 edavalos@dp.stmarytx.edu

Then open this on your browser

    http://localhost:6006

"""

import numpy as np

# Local Imports
root = next(path for path in pathlib.Path(os.path.abspath(__file__)).parents if path.name == 'FastPoseCNN')
sys.path.append(str(root))

import project
import dataset
import visualize
import trainer
import model_lib

#-------------------------------------------------------------------------------
# File Constants

CAMERA_TRAIN_DATASET = project.cfg.DATASET_DIR / 'NOCS' / 'camera' / 'train'
CAMERA_VALID_DATASET = project.cfg.DATASET_DIR / 'NOCS' / 'camera' / 'val'

#-------------------------------------------------------------------------------
# Functions

#-------------------------------------------------------------------------------
# Main Code

if __name__ == '__main__':

    """
    # Loading complete dataset
    train_dataset = dataset.NOCSDataset(CAMERA_TRAIN_DATASET, 1000, balance=True)
    valid_dataset = dataset.NOCSDataset(CAMERA_VALID_DATASET, 100)

    # Specifying the criterions
    #criterions = {'masks': kornia.losses.DiceLoss()}
    criterions = {'masks': torch.nn.CrossEntropyLoss()}

    # Creating a Trainer
    #model = model_lib.unet(n_classes = len(project.constants.SYNSET_NAMES), feature_scale=4)
    #model = model_lib.FastPoseCNN(in_channels=3, bilinear=True, filter_factor=4)
    model = model_lib.UNetWrapper(in_channels=3, n_classes=len(project.constants.SYNSET_NAMES),
                                  padding=True, wf=4, depth=4)

    my_trainer = trainer.Trainer(model, 
                                 train_dataset,
                                 valid_dataset,
                                 criterions,
                                 batch_size=4,
                                 num_workers=4)

    # Fitting Trainer
    my_trainer.fit(10)
    """
    