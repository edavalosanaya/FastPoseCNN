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

import fastai
import fastai.learner
import fastai.optimizer
import fastai.metrics

import fastai.vision

import fastai.data
import fastai.data.core

import segmentation_models_pytorch as smp

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

#-------------------------------------------------------------------------------
# Functions

def use_custom_trainer(epoch, datasets, dataloaders, criterion, model, optimizer, device, n_classes):

    # Creating trainer
    my_trainer = trainer.Trainer(model, 
                                 datasets,
                                 dataloaders,
                                 criterion,
                                 optimizer,
                                 device,
                                 n_classes,
                                 batch_size=4,
                                 num_workers=4)

    # Fitting Trainer
    my_trainer.fit(epoch)

def use_fastai_learner(epoch, datasets, dataloaders, criterion, model, optimizer, device):

    # Creating fastai dataloaders
    fastai_dataloaders = fastai.data.core.DataLoaders(dataloaders[0], dataloaders[1])

    # Wrapping PyTorch optimizer to be compatible with fastai
    #fastai_optimizer = fastai.optimizer.OptimWrapper(optimizer)
    fastai_optimizer = fastai.optimizer.Adam(model.parameters(), lr=3e-3)

    # Create leaner
    my_learner = fastai.learner.Learner(fastai_dataloaders, model,
                                        loss_func=criterion, metrics=[acc_voc])

    # Fitting Learner
    my_learner.fit(epoch)

def acc_voc(pred, target):
    foreground = (target != 0)
    return (torch.argmax(pred, dim=1)[foreground] == target[foreground]).float().mean()

#-------------------------------------------------------------------------------
# Main Code

if __name__ == '__main__':

    #***************************************************************************
    # Loading dataset 
    #***************************************************************************
    """
    # NOCS
    train_dataset = dataset.NOCSDataset(CAMERA_TRAIN_DATASET, 1000, balance=True)
    valid_dataset = dataset.NOCSDataset(CAMERA_VALID_DATASET, 100)
    n_classes = len(project.constants.SYNSET_NAMES)
    #"""
    
    # VOC
    crop_size = (320, 480)
    train_dataset = dataset.VOCSegDataset(True, crop_size, VOC_DATASET)
    valid_dataset = dataset.VOCSegDataset(False, crop_size, VOC_DATASET)
    datasets = train_dataset, valid_dataset
    n_classes = len(project.constants.VOC_CLASSES)

    #***************************************************************************
    # Creating dataloaders 
    #***************************************************************************
    # Dataloader parameters
    batch_size = 4
    num_workers = 0

    # For using multple CPUs for fast dataloaders
    # More information can be found in the following link:
    # https://github.com/pytorch/pytorch/issues/40403
    if num_workers > 0:
        torch.multiprocessing.set_start_method('spawn') # good solution !!!!

    train_dataloader = torch.utils.data.DataLoader(train_dataset, num_workers=num_workers,
                                                   batch_size=batch_size, shuffle=True)
    valid_dataloader = torch.utils.data.DataLoader(valid_dataset, num_workers=num_workers,
                                                   batch_size=batch_size, shuffle=True)
    dataloaders = train_dataloader, valid_dataloader

    #***************************************************************************
    # Specifying criterions 
    #***************************************************************************
    #criterions = {'masks': kornia.losses.DiceLoss()}
    #criterions = {'masks': torch.nn.CrossEntropyLoss()}
    criterion = torch.nn.CrossEntropyLoss()
    #criterion = kornia.losses.DiceLoss()

    #***************************************************************************
    # Loading model
    #***************************************************************************
    #model = model_lib.unet(n_classes = len(project.constants.SYNSET_NAMES), feature_scale=4)
    #model = model_lib.FastPoseCNN(in_channels=3, bilinear=True, filter_factor=4)
    #model = model_lib.UNetWrapper(in_channels=3, n_classes=len(project.constants.SYNSET_NAMES),
    #                              padding=True, wf=4, depth=4)
    model = smp.Unet('resnet34', encoder_weights='imagenet', classes=n_classes)

    # Using multiple GPUs if avaliable
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    if torch.cuda.device_count() > 1:
        print(f'Using {torch.cuda.device_count()} GPUs!')
        model = torch.nn.DataParallel(model)
    
    model.to(device)

    #***************************************************************************
    # Selecting optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=3e-3, weight_decay=1e-5)
    #***************************************************************************

    #***************************************************************************
    # Selecting Trainer 
    #***************************************************************************
    epoch=10
    use_custom_trainer(epoch, datasets, dataloaders, criterion, model, optimizer, device, n_classes)
    #use_fastai_learner(epoch, datasets, dataloaders, criterion, model, optimizer, device)