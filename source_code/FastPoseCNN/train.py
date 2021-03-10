import sys
import os
import warnings
import datetime
import argparse
import pathlib
import pprint
import random

# DEBUGGING
import pdb
import logging

import numpy as np
import base64

# Ignore annoying warnings
os.environ["TF_CPP_MIN_LOG_LEVEL"]="3"
warnings.filterwarnings('ignore')
#os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

import numpy as np

import torch
import torch.optim
import torch.nn.functional as F

import pytorch_lightning as pl
import pytorch_lightning.metrics.functional
import pytorch_lightning.core.decorators

import segmentation_models_pytorch as smp

# Local Imports
import setup_env
import tools
import lib

import logger as pll
import callbacks as plc
import config as cfg

#-------------------------------------------------------------------------------
# Documentation

"""
# How to view tensorboard in the Lambda machine

Do the following in Lamda machine: 

    tensorboard --logdir=logs --port 6006 --host=localhost

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

HPARAM = cfg.DEFAULT_POSE_HPARAM()
LOGGER = logging.getLogger('fastposecnn')

LOGGER.setLevel(logging.DEBUG)
logging.getLogger('requests').setLevel(logging.DEBUG)
logging.getLogger('PIL').setLevel(logging.INFO)

for log_name, log_obj in logging.Logger.manager.loggerDict.items():
    if log_name != 'fastposecnn':
        log_obj.disabled = True

#-------------------------------------------------------------------------------
# File Main

if __name__ == '__main__':

    # Parse arguments and replace global variables if needed
    parser = argparse.ArgumentParser(description='Train with PyTorch Lightning framework')
    
    # Automatically adding all the attributes of the HPARAM to the parser
    for attr in dir(HPARAM):
        if '__' in attr or attr[0] == '_': # Private or magic attributes
            continue

        if attr == 'EXPERIMENT_NAME':
            parser.add_argument('-e', f'--{attr}', required=True, type=type(getattr(HPARAM, attr)))
        else:
            parser.add_argument(f'--{attr}', type=type(getattr(HPARAM, attr)), default=getattr(HPARAM, attr))

    # Updating the HPARAMs
    parser.parse_args(namespace=HPARAM)

    # Applying environmental HPARAMS
    os.environ['CUDA_VISIBLE_DEVICES'] = HPARAM.CUDA_VISIBLE_DEVICES

    # If not debugging, then make matplotlib use the non-GUI backend to 
    # improve stability and speed, otherwise allow debugging sessions to use 
    # matplotlib figures.
    if not HPARAM.DEBUG:
        import matplotlib
        matplotlib.use('Agg')

    # If deterministic is selected, try our best to make the experiment deterministic
    if HPARAM.DETERMINISTIC:
        # More information here:
        # https://pytorch.org/docs/stable/notes/randomness.html

        # Random seeds
        torch.manual_seed(0)
        random.seed(0)
        np.random.seed(0)

        # CUDA
        torch.backends.cudnn.benchmark = False
        #torch.use_deterministic_algorithms()
        torch.backends.cudnn.deterministic = True

    # Ensuring that DISTRIBUTED_BACKEND doesn't cause problems
    HPARAM.DISTRIBUTED_BACKEND = None if HPARAM.NUM_GPUS <= 1 else HPARAM.DISTRIBUTED_BACKEND

    # Add the numpy instrinsics to the hyperparameters given the dataset name
    HPARAM.NUMPY_INTRINSICS = tools.pj.constants.INTRINSICS[HPARAM.DATASET_NAME]

    # Creating data module
    dataset = tools.ds.PoseRegressionDataModule(
        dataset_name=HPARAM.DATASET_NAME,
        selected_classes=HPARAM.SELECTED_CLASSES,
        batch_size=HPARAM.BATCH_SIZE,
        num_workers=HPARAM.NUM_WORKERS,
        encoder=HPARAM.ENCODER,
        encoder_weights=HPARAM.ENCODER_WEIGHTS,
        train_size=HPARAM.TRAIN_SIZE,
        valid_size=HPARAM.VALID_SIZE,
        is_deterministic=HPARAM.DETERMINISTIC
    )

    # Selecting the criterion (specific to each task)
    criterion = {
        'mask': {
            'loss_ce': {'D': 'pixel-wise', 'F': lib.loss.CE(), 'weight': 5.0},
            'loss_cce': {'D': 'pixel-wise', 'F': lib.loss.CCE(), 'weight': 5.0},
            'loss_focal': {'D': 'pixel-wise', 'F': lib.loss.Focal(), 'weight': 5.0}
        },
        'quaternion': {
            #'loss_mse': {'D': 'pixel-wise', 'F': lib.loss.MaskedMSELoss(key='quaternion'), 'weight': 0.2},
            #'loss_pw_qloss': {'D': 'pixel-wise', 'F': lib.loss.PixelWiseQLoss(key='quaternion'), 'weight': 1.0}
            'loss_quat': {'D': 'matched', 'F': lib.loss.QLoss(key='quaternion'), 'weight': 0.1},
        },
        'xy': {
            #'loss_mse': {'D': 'pixel-wise', 'F': lib.loss.MaskedMSELoss(key='xy'), 'weight': 0.2},
            'loss_xy': {'D': 'matched', 'F': lib.loss.XYLoss(key='xy'), 'weight': 0.01},
        },
        'z': {
            #'loss_mse': {'D': 'pixel-wise', 'F': lib.loss.MaskedMSELoss(key='z'), 'weight': 0.2},
            'loss_z': {'D': 'matched', 'F': lib.loss.ZLoss(key='z'), 'weight': 0.1},
        },
        'scales': {
            #'loss_mse': {'D': 'pixel-wise', 'F': lib.loss.MaskedMSELoss(key='scales'), 'weight': 0.2},
            'loss_scales': {'D': 'matched', 'F': lib.loss.ScalesLoss(key='scales'), 'weight': 0.1},
        },
        #'RT_and_metrics': {
        #    'loss_R': {'D': 'matched', 'F': lib.loss.RLoss(key='R'), 'weight': 1.0},
        #    'loss_T': {'D': 'matched', 'F': lib.loss.TLoss(key='T'), 'weight': 1.0},
        #    'loss_iou3d': {'D': 'matched', 'F': lib.loss.Iou3dLoss(), 'weight': 1.0},
        #    'loss_offset': {'D': 'matched', 'F': lib.loss.OffsetLoss(), 'weight': 1.0}
        #}
    }

    # Selecting metrics
    metrics = {
        'mask': {
            'dice': {'D': 'pixel-wise', 'F': pl.metrics.functional.dice_score},
            'iou': {'D': 'pixel-wise', 'F': pl.metrics.functional.iou},
            'f1': {'D': 'pixel-wise', 'F': pl.metrics.functional.f1_score}
        },
        """
        'quaternion': {
            'mae': {'D': 'pixel-wise', 'F': pl.metrics.functional.mean_absolute_error},
        },
        'xy': {
            'mae': {'D': 'pixel-wise', 'F': pl.metrics.functional.mean_absolute_error}
        },
        'z': {
            'mae': {'D': 'pixel-wise', 'F': pl.metrics.functional.mean_absolute_error}
        },
        'scales': {
            'mae': {'D': 'pixel-wise', 'F': pl.metrics.functional.mean_absolute_error}
        },
        """
        'pose': {
            'degree_error': {'D': 'matched', 'F': lib.metrics.DegreeError()},
            'degree_error_AP_5': {'D': 'matched', 'F': lib.metrics.DegreeErrorMeanAP(5)},
            'iou_3d_mAP_0.25': {'D': 'matched', 'F': lib.metrics.Iou3dAP(0.25)},
            'iou_3d_accuracy': {'D': 'matched', 'F': lib.metrics.Iou3dAccuracy()},
            'offset_error_AP_5cm': {'D': 'matched', 'F': lib.metrics.OffsetAP(5)},
            'offset_error': {'D': 'matched', 'F': lib.metrics.OffsetError()},
        }
    }

    # Construct model (if ckpt=None, it will just load as is)
    base_model = lib.pose_regressor.MODELS[HPARAM.MODEL].load_from_ckpt(
        HPARAM.CHECKPOINT, 
        HPARAM
    )

    # Create PyTorch Lightning Module
    model = lib.pose_regressor.PoseRegressionTask(
        HPARAM,
        model=base_model,
        criterion=criterion,
        metrics=metrics,
        HPARAM=HPARAM
    )

    # If no runs this day, create a runs-of-the-day folder
    date = datetime.datetime.now().strftime('%y-%m-%d')
    run_of_the_day_dir = pathlib.Path(os.getenv("LOGS")) / date
    if run_of_the_day_dir.exists() is False:
        os.mkdir(str(run_of_the_day_dir))

    # Creating run name
    time = datetime.datetime.now().strftime('%H-%M')
    model_name = f"{HPARAM.ENCODER}-{HPARAM.ENCODER_WEIGHTS}"
    run_name = f"{time}-{HPARAM.EXPERIMENT_NAME}-{HPARAM.MODEL}-{HPARAM.DATASET_NAME}-{model_name}"
    
    # Making the run's log path accessible by the environmental variables
    os.environ['RUNS_LOG_DIR'] = str(run_of_the_day_dir / run_name)

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
        save_dir=run_of_the_day_dir,
        name=run_name
    )

    # Add logging for debugging long sessions
    logging.basicConfig(
        filename=str(run_of_the_day_dir / run_name  / 'run.log'),
        level=logging.DEBUG
    )

    # Saving the HPARAMS as a json file in the runs directory for easier
    # troubleshooting
    json_hparam = {k:getattr(HPARAM, k) for k in dir(HPARAM) if (k.find("__") == -1 and k[0] != '_')}
    tools.jt.save_to_json(str(run_of_the_day_dir / run_name / 'HPARAM.json'), json_hparam)

    # A callback for creating the visualization and logging data to Tensorboard
    tensorboard_callback = plc.TensorboardCallback(
        HPARAM=HPARAM,
        tasks=['mask', 'quaternion', 'xy', 'z', 'scales', 'hough voting', 'pose'],
        hparams=runs_hparams,
        #checkpoint_monitor={
        #    'pose/degree_error_AP_5': 'max'
        #}
    )

    # A callback that saves ckpts every N steps (useful for when the test crashes)
    ckpt_save_n_callback = plc.CheckpointEveryNSteps(
        save_step_frequency = HPARAM.CKPT_SAVE_FREQUENCY,
        prefix = 'n-ckpt'
    )

    # Checkpoint callbacks
    # saves a file like: my/path/sample-mnist-epoch=02-val_loss=0.32.ckpt
    loss_checkpoint_callback = pl.callbacks.ModelCheckpoint(
        monitor='checkpoint_on',
        save_top_k=1,
        save_last=True,
        filename='{epoch:02d}-{checkpoint_on:.4f}',
        mode='min'
    )
    
    """
    metric_checkpoint_callback = pl.callbacks.ModelCheckpoint(
        monitor='quaternion/degree_error_AP_5',
        save_top_k=1,
        filename='{epoch:02d}-degree_error_AP_5=?',
        mode='max'
    )
    """

    # Training
    trainer = pl.Trainer(
        max_epochs=HPARAM.NUM_EPOCHS,
        gpus=HPARAM.NUM_GPUS,
        num_processes=HPARAM.NUM_WORKERS,
        distributed_backend=HPARAM.DISTRIBUTED_BACKEND, # required to work
        logger=tb_logger,
        callbacks=[
            tensorboard_callback, 
            loss_checkpoint_callback,
            ckpt_save_n_callback],
        gradient_clip_val=0.15
    )

    # Train
    trainer.fit(
        model,
        dataset
    )