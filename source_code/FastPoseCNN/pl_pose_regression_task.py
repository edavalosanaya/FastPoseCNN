import os
import warnings
import datetime
import argparse

import pdb

import numpy as np

# Ignore annoying warnings
os.environ["TF_CPP_MIN_LOG_LEVEL"]="3"
warnings.filterwarnings('ignore')

import torch
import torch.nn.functional as F

import pytorch_lightning as pl
import catalyst.contrib.nn

import segmentation_models_pytorch as smp

# Local Imports
import tools
import lib

import pl_logger as pll
import pl_callbacks as plc

#-------------------------------------------------------------------------------
# Documentation

"""
# How to view tensorboard in the Lambda machine

Do the following in Lamda machine: 

    tensorboard --logdir=logs --port 6006 --host=localhost

    tensorboard --logdir=lib/logs --port 6006 --host=localhost

Then run this on the local machine

    ssh -NfL 6006:localhost:6006 edavalos@dp.stmarytx.edu

Then open this on your browser

    http://localhost:6006

To delete hanging Python processes use the following:

    killall -9 python

To delete hanging Tensorboard processes use the following:

    pkill -9 tensorboard

"""

#-------------------------------------------------------------------------------
# File Constants

# Run hyperparameters
class DEFAULT_POSE_HPARAM(argparse.Namespace):
    DATASET_NAME = 'NOCS'
    SELECTED_CLASSES = tools.pj.constants.NUM_CLASSES[DATASET_NAME]
    BATCH_SIZE = 2
    NUM_WORKERS = 36 # 36 total CPUs
    NUM_GPUS = 1
    LEARNING_RATE = 0.001
    ENCODER_LEARNING_RATE = 0.0005
    NUM_EPOCHS = 20
    DISTRIBUTED_BACKEND = None if NUM_GPUS <= 1 else 'ddp'
    BACKBONE_ARCH = 'FPN'
    ENCODER = 'resnext50_32x4d'
    ENCODER_WEIGHTS = 'imagenet'

HPARAM = DEFAULT_POSE_HPARAM()

#-------------------------------------------------------------------------------
# Classes

class PoseRegresssionTask(pl.LightningModule):

    def __init__(self, model, criterion, metrics):
        super().__init__()

        # Saving parameters
        self.model = model

        # Saving the criterion
        self.criterion = criterion

        # Saving the metrics
        self.metrics = metrics

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        
        # Calculate the loss and metrics
        multi_task_losses, multi_task_metrics = self.shared_step('train', batch, batch_idx)

        # Placing the main loss into Train Result to perform backpropagation
        result = pl.TrainResult(minimize=multi_task_losses['total_loss'])

        # Logging the val loss for each task
        for task_name in multi_task_losses.keys():

            # If it is the total loss, skip it, it was already log in the
            # previous line
            if task_name == 'total_loss':
                continue
            
            result.log(f'train_{task_name}_loss', multi_task_losses[task_name]['loss'])

        # Compress the multi_task_losses and multi_task_metrics to make logging easier
        loggable_metrics = tools.dm.compress_dict(multi_task_metrics)

        # Logging the train metrics
        result.log_dict(loggable_metrics)

        return result

    def validation_step(self, batch, batch_idx):
        
        # Calculate the loss
        multi_task_losses, multi_task_metrics = self.shared_step('valid', batch, batch_idx)
        
        # Log the batch loss inside the pl.TrainResult to visualize in the
        # progress bar
        result = pl.EvalResult(checkpoint_on=multi_task_losses['total_loss'])

        # Logging the val loss for each task
        for task_name in multi_task_losses.keys():
            
            # If it is the total loss, skip it, it was already log in the
            # previous line
            if task_name == 'total_loss':
                continue

            result.log(f'val_{task_name}_loss', multi_task_losses[task_name]['loss'])

        # Compress the multi_task_losses and multi_task_metrics to make logging easier
        loggable_metrics = tools.dm.compress_dict(multi_task_metrics)

        # Logging the val metrics
        result.log_dict(loggable_metrics)

        return result

    def shared_step(self, mode, batch, batch_idx):
        # Forward pass the input and generate the prediction of the NN
        output = self.model(batch['image'])

        # Storage for losses and metrics depending on the task
        multi_task_losses = {'total_loss': torch.Tensor([0]).float().to(self.device)}
        multi_task_metrics = {}

        # Calculate separate task losses and metrics
        for task_name in output.keys():
            
            # Calculate the loss based on self.loss_function
            losses, metrics = self.loss_function(
                task_name,
                output[task_name], 
                batch[task_name]
            )

            # Logging the batch loss to Tensorboard
            for loss_name, loss_value in losses.items():
                self.logger.log_metrics(mode, {f'{task_name}/{loss_name}/batch':loss_value}, batch_idx)

            # Logging the metric loss to Tensorboard
            for metric_name, metric_value in metrics.items():
                self.logger.log_metrics(mode, {f'{task_name}/{metric_name}/batch':metric_value}, batch_idx) 

            # Storing the losses and metrics for each task
            multi_task_losses[task_name] = losses
            multi_task_metrics[task_name] = metrics

            # Summing all task total losses
            multi_task_losses['total_loss'] += losses['task_total_loss']

        return multi_task_losses, multi_task_metrics

    def loss_function(self, task_name, pred, gt):
        
        losses = {
            k: v['F'](pred, gt) for k,v in self.criterion[task_name].items()
        }

        with torch.no_grad():
            metrics = {
                k: v(pred, gt) for k,v in self.metrics[task_name].items()
            }
        
        # Calculate total loss
        total_loss = torch.sum(torch.stack(list(losses.values())))

        # Calculate the loss multiplied by its corresponded weight
        weighted_losses = [losses[key] * self.criterion[task_name][key]['weight'] for key in losses.keys()]
        
        # Now calculate the weighted sum
        weighted_sum = torch.sum(torch.stack(weighted_losses))

        # Save the calculated sum in the losses
        losses['loss'] = weighted_sum

        # Saving the total loss
        losses['task_total_loss'] = total_loss

        return losses, metrics

    def configure_optimizers(self):

        # Catalyst has new SOTA optimizers out of box
        base_optimizer = catalyst.contrib.nn.RAdam(self.model.parameters(), lr=HPARAM.LEARNING_RATE, weight_decay=0.0003)
        optimizer = catalyst.contrib.nn.Lookahead(base_optimizer)

        # Solution from here:
        # https://github.com/PyTorchLightning/pytorch-lightning/issues/1598#issuecomment-702038244
        lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer)
        scheduler = {
            'scheduler': lr_scheduler,
            'reduce_on_plateau': True,
            'monitor': 'val_checkpoint_on',
            'patience': 2,
            'mode': 'min',
            'factor': 0.25
        }
        
        return [optimizer], [scheduler]

class PoseRegressionDataModule(pl.LightningDataModule):

    def __init__(
        self, 
        dataset_name='NOCS', 
        batch_size=1, 
        num_workers=0,
        selected_classes=None
        ):

        super().__init__()
        
        # Saving parameters
        self.dataset_name = dataset_name
        self.selected_classes = selected_classes
        self.batch_size = batch_size
        self.num_workers = num_workers

    def setup(self, stage=None):

        # Obtaining the preprocessing_fn depending on the encoder and the encoder
        # weights
        preprocessing_fn = smp.encoders.get_preprocessing_fn(HPARAM.ENCODER, HPARAM.ENCODER_WEIGHTS)

        # NOCS
        if self.dataset_name == 'NOCS':
            
            # Dataset hyperparameters
            train_size=5000
            valid_size=200

            # If no specific classes are used, use all the classes from NOCS
            if self.selected_classes is None:
                self.selected_classes = tools.pj.constants.NOCS_CLASSES

            train_dataset = tools.ds.NOCSPoseRegDataset(
                dataset_dir=tools.pj.cfg.CAMERA_TRAIN_DATASET,
                max_size=train_size,
                classes=self.selected_classes,
                augmentation=tools.transforms.pose.get_training_augmentation(),
                preprocessing=tools.transforms.pose.get_preprocessing(preprocessing_fn)
            )

            valid_dataset = tools.ds.NOCSPoseRegDataset(
                dataset_dir=tools.pj.cfg.CAMERA_VALID_DATASET, 
                max_size=valid_size,
                classes=self.selected_classes,
                augmentation=tools.transforms.pose.get_validation_augmentation(),
                preprocessing=tools.transforms.pose.get_preprocessing(preprocessing_fn)
            )

            self.datasets = {
                'train': train_dataset,
                'valid': valid_dataset
            }        

    def get_loader(self, dataset_key):

        if dataset_key in self.datasets.keys():        
            
            dataloader = torch.utils.data.DataLoader(
                self.datasets[dataset_key],
                num_workers=self.num_workers,
                batch_size=self.batch_size,
                shuffle=True
            )
            return dataloader

        else:

            return None

    def train_dataloader(self):
        return self.get_loader('train')

    def val_dataloader(self):
        return self.get_loader('valid')

    def test_dataloader(self):
        return self.get_loader('test')

#-------------------------------------------------------------------------------
# File Main

if __name__ == '__main__':

    # Modification of hyperparameters
    HPARAM.SELECTED_CLASSES = ['bg','camera']

    # Ensuring that DISTRIBUTED_BACKEND doesn't cause problems
    HPARAM.DISTRIBUTED_BACKEND = None if HPARAM.NUM_GPUS <= 1 else HPARAM.DISTRIBUTED_BACKEND

    # Creating data module
    dataset = PoseRegressionDataModule(
        dataset_name=HPARAM.DATASET_NAME,
        selected_classes=HPARAM.SELECTED_CLASSES,
        batch_size=HPARAM.BATCH_SIZE,
        num_workers=HPARAM.NUM_WORKERS
    )

    # Creating base model
    base_model = lib.models.PoseRegressor(
        architecture=HPARAM.BACKBONE_ARCH,
        encoder_name=HPARAM.ENCODER,
        encoder_weights=HPARAM.ENCODER_WEIGHTS,
        classes=len(HPARAM.SELECTED_CLASSES)
    )

    # Selecting the criterion (specific to each task)
    criterion = {
        'mask': {
            'loss_ce': {'F': torch.nn.CrossEntropyLoss(), 'weight': 0.8},
            'loss_cce': {'F': lib.loss.CCE(), 'weight': 0.8},
            'loss_focal': {'F': lib.loss.Focal(), 'weight': 1.0}
        },
        'quaternion': {
            'loss_mse': {'F': torch.nn.MSELoss(), 'weight': 1.0}
        },
        'scales': {
            'loss_mse': {'F': torch.nn.MSELoss(), 'weight': 1.0}
        },
        'xy': {
            'loss_mse': {'F': torch.nn.MSELoss(), 'weight': 1.0}
        },
        'z': {
            'loss_mse': {'F': torch.nn.MSELoss(), 'weight': 1.0}
        }
    }

    # Selecting metrics
    metrics = {
        'mask': {
            'dice': pl.metrics.functional.dice_score,
            'iou': pl.metrics.functional.iou,
            'f1': pl.metrics.functional.f1_score
        },
        'quaternion': {
            'mae': pl.metrics.functional.regression.mae
        },
        'scales': {
            'mae': pl.metrics.functional.regression.mae
        },
        'xy': {
            'mae': pl.metrics.functional.regression.mae
        },
        'z': {
            'mae': pl.metrics.functional.regression.mae
        }
    }

    # Noting what are the items that we want to see as the training develops
    tracked_data = {
        'minimize': list(tools.dm.compress_dict(criterion, additional_subkey='loss').keys()),
        'maximize': list(tools.dm.compress_dict(metrics).keys())
    }

    # Attaching PyTorch Lightning logic to base model
    model = PoseRegresssionTask(base_model, criterion, metrics)

    # Saving the run
    model_name = f"{HPARAM.ENCODER}-{HPARAM.ENCODER_WEIGHTS}"
    now = datetime.datetime.now().strftime('%y-%m-%d--%H-%M')
    run_name = f"pose-{now}-{HPARAM.DATASET_NAME}-{model_name}"

    # Construct hparams data to send it to MyCallback
    runs_hparams = {
        'model': model_name,
        'dataset': HPARAM.DATASET_NAME,
        'number of GPUS': HPARAM.NUM_GPUS,
        'batch size': HPARAM.BATCH_SIZE,
        'number of workers': HPARAM.NUM_WORKERS,
        'ML abs library': 'pl',
        'distributed_backend': HPARAM.DISTRIBUTED_BACKEND,
    }

    # Creating my own logger
    tb_logger = pll.MyLogger(
        HPARAM,
        pl_module=model,
        save_dir=tools.pj.cfg.LOGS,
        name=run_name
    )

    # Creating my own callback
    custom_callback = plc.MyCallback(
        tasks=['mask', 'quaternion'],
        hparams=runs_hparams,
        tracked_data=tracked_data
    )

    # Training
    trainer = pl.Trainer(
        max_epochs=HPARAM.NUM_EPOCHS,
        gpus=HPARAM.NUM_GPUS,
        num_processes=HPARAM.NUM_WORKERS,
        distributed_backend=HPARAM.DISTRIBUTED_BACKEND, # required to work
        logger=tb_logger,
        callbacks=[custom_callback]
    )

    # Train
    trainer.fit(
        model,
        dataset
    )