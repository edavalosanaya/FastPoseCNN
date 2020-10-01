import os
import sys
import pathlib
import warnings
import collections
import datetime

import pdb

# Ignore annoying warnings
os.environ["TF_CPP_MIN_LOG_LEVEL"]="3"
warnings.filterwarnings('ignore')

import torch
import torch.nn
import torch.optim
import torch.utils
import torch.utils.tensorboard

import catalyst
import catalyst.dl
import catalyst.contrib.nn
import catalyst.dl.callbacks

import sklearn.model_selection

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

To delete hanging Python processes use the following:

    killall -9 python

To delete hanging Tensorboard processes use the following:

    pkill -9 tensorboard

"""

import numpy as np

# Local Imports
root = next(path for path in pathlib.Path(os.path.abspath(__file__)).parents if path.name == 'FastPoseCNN')
sys.path.append(str(root))

import project
import dataset
import visualize
import model_lib
import transforms
import catalyst_callbacks as cc

#-------------------------------------------------------------------------------
# File Constants

# Run hyperparameters
IS_FP16_USED = False

DATASET_NAME = 'NOCS'
BATCH_SIZE = 4
NUM_WORKERS = 8

LEARNING_RATE = 0.001
ENCODER_LEARNING_RATE = 0.0005

ENCODER = 'resnext50_32x4d'
ENCODER_WEIGHTS = 'imagenet'

NUM_EPOCHS = 20
DEVICE = catalyst.utils.get_device()

#-------------------------------------------------------------------------------
# Helper Functions

def load_dataset(DATASET_NAME='VOC'):

    #***************************************************************************
    # Loading dataset 
    #***************************************************************************
    
    # Obtaining the preprocessing_fn depending on the encoder and the encoder
    # weights
    preprocessing_fn = smp.encoders.get_preprocessing_fn(ENCODER, ENCODER_WEIGHTS)

    # NOCS
    if DATASET_NAME == 'NOCS':
        crop_size = 224
        train_dataset = dataset.NOCSDataset(
            dataset_dir=project.cfg.CAMERA_TRAIN_DATASET, 
            max_size=1000,
            classes=project.constants.NOCS_CLASSES,
            augmentation=transforms.get_training_augmentation(height=crop_size, width=crop_size),
            preprocessing=transforms.get_preprocessing(preprocessing_fn),
            balance=True,
            crop_size=crop_size,
            mask_dataformat='HW'
        )

        valid_dataset = dataset.NOCSDataset(
            dataset_dir=project.cfg.CAMERA_VALID_DATASET, 
            max_size=100,
            classes=project.constants.NOCS_CLASSES,
            augmentation=transforms.get_validation_augmentation(height=crop_size, width=crop_size),
            preprocessing=transforms.get_preprocessing(preprocessing_fn),
            balance=False,
            crop_size=crop_size,
            mask_dataformat='HW'
        )
        
        datasets = {'train': train_dataset,
                    'valid': valid_dataset}
    
    # VOC
    if DATASET_NAME == 'VOC':
        
        train_dataset = dataset.VOCDataset(
            voc_dir=project.cfg.VOC_DATASET,
            is_train=True,
            classes=project.constants.VOC_CLASSES,
            augmentation=transforms.get_training_augmentation(),
            preprocessing=transforms.get_preprocessing(preprocessing_fn)
        )

        valid_dataset = dataset.VOCDataset(
            voc_dir=project.cfg.VOC_DATASET,
            is_train=False,
            classes=project.constants.VOC_CLASSES,
            augmentation=transforms.get_validation_augmentation(),
            preprocessing=transforms.get_preprocessing(preprocessing_fn)
        )
        
        datasets = {'train': train_dataset,
                    'valid': valid_dataset}

    # CAMVID
    if DATASET_NAME == 'CAMVID':

        train_dataset = dataset.CAMVIDDataset(
            project.cfg.CAMVID_DATASET,
            train_valid_test='train', 
            classes=project.constants.CAMVID_CLASSES,
            augmentation=transforms.get_training_augmentation(), 
            preprocessing=transforms.get_preprocessing(preprocessing_fn),
            mask_dataformat='HW'
        )

        valid_dataset = dataset.CAMVIDDataset(
            project.cfg.CAMVID_DATASET,
            train_valid_test='val',
            classes=project.constants.CAMVID_CLASSES,
            augmentation=transforms.get_validation_augmentation(), 
            preprocessing=transforms.get_preprocessing(preprocessing_fn),
            mask_dataformat='HW'
        )

        test_dataset = dataset.CAMVIDDataset(
            project.cfg.CAMVID_DATASET,
            train_valid_test='test',
            classes=project.constants.CAMVID_CLASSES,
            augmentation=transforms.get_validation_augmentation(), 
            preprocessing=transforms.get_preprocessing(preprocessing_fn),
            mask_dataformat='HW'
        )

        test_dataset_vis = dataset.CAMVIDDataset(
            project.cfg.CAMVID_DATASET,
            train_valid_test='test',
            classes=project.constants.CAMVID_CLASSES,
            mask_dataformat='HW'
        )

        datasets = {'train': train_dataset,
                    'valid': valid_dataset,
                    'test': test_dataset}

    # CARVANA
    if DATASET_NAME == 'CARVANA':

        train_image_path = pathlib.Path(project.cfg.CARVANA_DATASET) / 'train'
        train_mask_path = pathlib.Path(project.cfg.CARVANA_DATASET) / 'train_masks'
        test_image_path = pathlib.Path(project.cfg.CARVANA_DATASET) / 'test'

        ALL_IMAGES = sorted(train_image_path.glob("*.jpg"))
        ALL_MASKS = sorted(train_mask_path.glob("*.gif"))

        indices = np.arange(len(ALL_IMAGES))
        valid_size=0.2
        random_state = 42

        # Let's divide the data set into train and valid parts.
        train_indices, valid_indices = sklearn.model_selection.train_test_split(
            indices, test_size=valid_size, random_state=random_state, shuffle=True
        )

        np_images = np.array(ALL_IMAGES)
        np_masks = np.array(ALL_MASKS)

        # Creates our train dataset
        train_dataset = dataset.CARVANADataset(
            images = np_images[train_indices].tolist(),
            masks = np_masks[train_indices].tolist(),
            transforms = transforms.train_transforms
        )

        # Creates our valid dataset
        valid_dataset = dataset.CARVANADataset(
            images = np_images[valid_indices].tolist(),
            masks = np_masks[valid_indices].tolist(),
            transforms = transforms.valid_transforms
        )

        datasets = {'train': train_dataset,
                    'valid': valid_dataset}

    return collections.OrderedDict(datasets)

def get_loaders(datasets, batch_size, num_workers):

    loaders = collections.OrderedDict()

    for DATASET_NAME, dataset in datasets.items():
        loader = torch.utils.data.DataLoader(dataset, num_workers=num_workers,
                                             batch_size=batch_size, shuffle=True)

        loaders[DATASET_NAME] = loader

    return loaders

#-------------------------------------------------------------------------------
# Main Code

if __name__ == '__main__':

    #***************************************************************************
    #* Creating datasets and dataloaders 
    #***************************************************************************

    # Load datasets
    datasets = load_dataset(DATASET_NAME)

    # For using multple CPUs for fast dataloaders
    # More information can be found in the following link:
    # https://github.com/pytorch/pytorch/issues/40403
    """
    if NUM_WORKERS > 0:
        torch.multiprocessing.set_start_method('spawn') # good solution !!!!
    """
    loaders = get_loaders(datasets, BATCH_SIZE, NUM_WORKERS)

    #***************************************************************************
    #* Loading model
    #***************************************************************************
    #model = model_lib.unet(n_classes = len(project.constants.NOCS_CLASSES), feature_scale=4)
    #model = model_lib.FastPoseCNN(in_channels=3, bilinear=True, filter_factor=4)
    #model = model_lib.UNetWrapper(in_channels=3, n_classes=len(project.constants.NOCS_CLASSES),
    #                              padding=True, wf=4, depth=4)
    #model = smp.Unet('resnet34', encoder_weights='imagenet', classes=n_classes)
    model = smp.FPN(encoder_name=ENCODER, encoder_weights=ENCODER_WEIGHTS, classes=len(datasets['train'].CLASSES))
    model = torch.nn.DataParallel(model)
    model = model.to(DEVICE)
    model_name = f'FPN-{ENCODER}-{ENCODER_WEIGHTS}'

    #***************************************************************************
    #* Selecting optimizer and learning rate scheduler
    #***************************************************************************

    # Since we use a pre-trained encoder, we will reduce the learning rate on it.
    layerwise_params = {"encoder*": dict(lr=ENCODER_LEARNING_RATE, weight_decay=0.00003)}

    # This function removes weight_decay for biases and applies our layerwise_params
    model_params = catalyst.utils.process_model_params(model, layerwise_params=layerwise_params)

    # Catalyst has new SOTA optimizers out of box
    base_optimizer = catalyst.contrib.nn.RAdam(model_params, lr=LEARNING_RATE, weight_decay=0.0003)
    optimizer = catalyst.contrib.nn.Lookahead(base_optimizer)

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.25, patience=2)

    if IS_FP16_USED:
        fp16_params = dict(opt_level='01')
    else:
        fp16_params = None

    #***************************************************************************
    #* Initialize Runner 
    #***************************************************************************

    runner = catalyst.dl.SupervisedRunner(device=DEVICE, input_key='image', input_target_key='mask')

    #***************************************************************************
    #* Specifying criterions 
    #***************************************************************************
    
    """
    criterion = {
        'dice': catalyst.contrib.nn.DiceLoss(),
        'iou': catalyst.contrib.nn.IoULoss(),
        'bce': torch.nn.BCEWithLogitsLoss()
    }
    """
    criterion = {
        'ce': torch.nn.CrossEntropyLoss(),
        'cce': model_lib.loss.CCE(),
        'focal': model_lib.loss.Focal()
    }

    #***************************************************************************
    #* Create Callbacks 
    #***************************************************************************

    """
    catalyst.dl.callbacks.CriterionCallback(
        input_key="mask",
        prefix="loss_dice",
        criterion_key="dice"
    ),
    catalyst.dl.callbacks.CriterionCallback(
        input_key="mask",
        prefix="loss_iou",
        criterion_key="iou"
    ),
    catalyst.dl.callbacks.CriterionCallback(
        input_key="mask",
        prefix="loss_bce",
        criterion_key="bce"
    ),

    # And only then we aggregate everything into one loss.
    catalyst.dl.callbacks.MetricAggregationCallback(
        prefix="loss",
        mode="weighted_sum", # can be "sum", "weighted_sum" or "mean"
        # because we want weighted sum, we need to add scale for each loss
        metrics={"loss_dice": 1.0, "loss_iou": 1.0, "loss_bce": 0.8},
    ),
    """

    callbacks = [
        # Each criterion is calculated separately.

        catalyst.dl.callbacks.CriterionCallback(
            input_key='mask',
            prefix='loss_ce',
            criterion_key='ce'
        ),
        catalyst.dl.callbacks.CriterionCallback(
            input_key='mask',
            prefix='loss_cce',
            criterion_key='cce'
        ),
        catalyst.dl.callbacks.CriterionCallback(
            input_key='mask',
            prefix='loss_focal',
            criterion_key='focal'
        ),

        catalyst.dl.callbacks.MetricAggregationCallback(
            prefix="loss",
            mode="weighted_sum", # can be "sum", "weighted_sum" or "mean"
            # because we want weighted sum, we need to add scale for each loss
            metrics={"loss_ce": 1.0, "loss_cce": 1.0, "loss_focal": 1.0},
        ),

        # metrics
        #catalyst.dl.callbacks.DiceCallback(input_key="mask"),
        #catalyst.dl.callbacks.IouCallback(input_key="mask"),

        # Visualize mask
        cc.TensorAddImageCallback(colormap=datasets['train'].COLORMAP)
    ]

    # Create tensorboard folder
    now = datetime.datetime.now().strftime('%d-%m-%y--%H-%M')
    run_name = f'{DATASET_NAME}-{model_name}-{now}'
    run_logdir = project.cfg.LOGS / run_name

    if run_logdir.exists() is False:
        os.mkdir(str(run_logdir))

    #***************************************************************************
    #* Training 
    #***************************************************************************

    print('Training')

    runner.train(
        model=model,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        loaders=loaders,
        callbacks=callbacks,
        logdir=str(run_logdir),
        num_epochs=NUM_EPOCHS,
        #main_metric="iou",
        #minimize_metric=False,
        fp16=fp16_params,
        verbose=True,
    )