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

    # Loading complete dataset
    train_dataset = dataset.NOCSDataset(CAMERA_TRAIN_DATASET, 1000)
    valid_dataset = dataset.NOCSDataset(CAMERA_VALID_DATASET, 100)

    # Splitting dataset to train and validation
    """
    dataset_num = len(complete_dataset)
    split = 0.2
    train_length, valid_length = int(dataset_num*(1-split)), int(dataset_num*split)

    train_dataset, valid_dataset = torch.utils.data.random_split(complete_dataset,
                                                                [train_length, valid_length])
    """    
    # Specifying the criterions
    """
    criterions = {'masks':torch.nn.CrossEntropyLoss(),
                  'depth':torch.nn.BCEWithLogitsLoss(),
                  'scales':torch.nn.BCEWithLogitsLoss(),
                  'quat':torch.nn.BCEWithLogitsLoss()}
    """
    criterions = {'masks': kornia.losses.DiceLoss()}

    # Creating a Trainer
    #model = model_lib.unet(n_classes = len(project.constants.SYNSET_NAMES))
    model = model_lib.FastPoseCNN(in_channels=3, bilinear=True, filter_factor=4)

    my_trainer = trainer.Trainer(model, 
                                 train_dataset,
                                 valid_dataset,
                                 criterions,
                                 batch_size=10)

    # Fitting Trainer
    my_trainer.fit(10)