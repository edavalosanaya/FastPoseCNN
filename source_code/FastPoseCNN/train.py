import os
import warnings
import datetime
import argparse
import pathlib
import pprint

# DEBUGGING
import pdb
import logging

import numpy as np
import base64

# Ignore annoying warnings
os.environ["TF_CPP_MIN_LOG_LEVEL"]="3"
warnings.filterwarnings('ignore')
#os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

import torch
import torch.optim
import torch.nn.functional as F

import pytorch_lightning as pl
import pytorch_lightning.metrics.functional
import pytorch_lightning.core.decorators

import catalyst.contrib.nn

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
# Classes

class PoseRegresssionTask(pl.LightningModule):

    def __init__(self, conf, model, criterion, metrics, HPARAM):
        super().__init__()

        # Saving parameters
        self.model = model

        # Saving the configuration (additional hyperparameters)
        self.save_hyperparameters(conf)
        self.HPARAM = HPARAM

        # Saving the criterion
        self.criterion = criterion

        # Saving the metrics
        self.metrics = metrics

    @pl.core.decorators.auto_move_data
    def forward(self, x):

        # Feed in the input to the actual model
        y = self.model(x)

        # Ensuring that the first-level outputs (mostly the image-size outputs)
        # do not have nans for visualization and metric purposes
        for key in y.keys():
            if isinstance(y[key], torch.Tensor):
                # Getting the dangerous conditions
                is_nan = torch.isnan(y[key])
                is_inf = torch.isinf(y[key])
                is_nan_or_inf = torch.logical_or(is_nan, is_inf)

                # Filling with stable information.
                y[key][is_nan_or_inf] = 0

        return y

    def training_step(self, batch, batch_idx):
        
        # Calculate the loss and metrics
        multi_task_losses, multi_task_metrics = self.shared_step('train', batch, batch_idx)

        # Placing the main loss into Train Result to perform backpropagation
        result = pl.core.step_result.TrainResult(minimize=multi_task_losses['pose']['total_loss'])

        # Logging the val loss for each task
        for task_name in multi_task_losses.keys():

            # If it is the total loss, skip it, it was already log in the
            # previous line
            if task_name == 'total_loss' or 'loss' not in multi_task_losses[task_name]:
                continue
            
            result.log(f'train_{task_name}_loss', multi_task_losses[task_name]['task_total_loss'])

        # Compress the multi_task_losses and multi_task_metrics to make logging easier
        loggable_metrics = tools.dm.compress_dict(multi_task_metrics)

        # Logging the train metrics
        result.log_dict(loggable_metrics)

        return result

    def validation_step(self, batch, batch_idx):
        
        # Calculate the loss
        multi_task_losses, multi_task_metrics = self.shared_step('valid', batch, batch_idx)
        
        # Log the batch loss inside the pl.TrainResult to visualize in the progress bar
        result = pl.core.step_result.EvalResult(checkpoint_on=multi_task_losses['pose']['total_loss'])

        # Logging the val loss for each task
        for task_name in multi_task_losses.keys():
            
            # If it is the total loss, skip it, it was already log in the
            # previous line
            if task_name == 'total_loss' or 'loss' not in multi_task_losses[task_name]:
                continue

            result.log(f'val_{task_name}_loss', multi_task_losses[task_name]['loss'])

        # Compress the multi_task_losses and multi_task_metrics to make logging easier
        loggable_metrics = tools.dm.compress_dict(multi_task_metrics)

        # Logging the val metrics
        result.log_dict(loggable_metrics)

        return result

    def shared_step(self, mode, batch, batch_idx):

        LOGGER.info(f'batch id={batch_idx}')
        
        # Forward pass the input and generate the prediction of the NN
        outputs = self.forward(batch['image'])

        # Matching aggregated data between ground truth and predicted
        if self.HPARAM.PERFORM_AGGREGATION and self.HPARAM.PERFORM_MATCHING:
            # Determine matches between the aggreated ground truth and preds
            gt_pred_matches = lib.mg.batchwise_find_matches(
                outputs['auxilary']['agg_pred'],
                batch['agg_data']
            )
        else:
            gt_pred_matches = None

        LOGGER.debug("\nMATCHED DATA")
        LOGGER.debug(pprint.pformat(gt_pred_matches))
        
        # Storage for losses and metrics depending on the task
        multi_task_losses = {'pose': {'total_loss': torch.tensor(0).float().to(self.device)}}
        multi_task_metrics = {}

        # Calculate separate task losses
        for task_name in self.criterion.keys():
            
            # Calculate the losses
            losses = self.calculate_loss_function(
                task_name,
                outputs,
                batch,
                gt_pred_matches
            )

            # Storing the task losses
            if task_name not in multi_task_losses.keys():
                multi_task_losses[task_name] = losses
            else:
                multi_task_losses[task_name].update(losses)

            # Summing all task total losses (if it is not nan)
            if torch.isnan(losses['task_total_loss']) != True:
                if 'total_loss' not in multi_task_losses['pose'].keys():
                    multi_task_losses['pose']['total_loss'] = losses['task_total_loss']
                else:
                    multi_task_losses['pose']['total_loss'] += losses['task_total_loss']

        # ! Debugging what is the loss that has large values!
        LOGGER.debug("\nALL LOSSESS")
        LOGGER.debug(pprint.pformat(multi_task_losses))

        # Logging the losses
        for task_name in multi_task_losses.keys():
            
            # Logging the batch loss to Tensorboard
            for loss_name, loss_value in multi_task_losses[task_name].items():
                self.logger.log_metrics(
                    mode, 
                    {f'{task_name}/{loss_name}/batch':loss_value.detach().clone()}, 
                    batch_idx,
                    use_epoch_num=False
                )

        # Calculate separate task metrics
        for task_name in self.metrics.keys():

            # Calculating the metrics
            metrics = self.calculate_metrics(
                task_name,
                outputs,
                batch,
                gt_pred_matches
            )

            # Storing the task metrics
            multi_task_metrics[task_name] = metrics

        # Logging the metrics
        for task_name in multi_task_metrics.keys():
            for metric_name, metric_value in multi_task_metrics[task_name].items():
                self.logger.log_metrics(
                    mode, 
                    {f'{task_name}/{metric_name}/batch':metric_value.detach().clone()}, 
                    batch_idx,
                    use_epoch_num=False
                ) 

        return multi_task_losses, multi_task_metrics

    def calculate_loss_function(self, task_name, outputs, inputs, gt_pred_matches):
        
        losses = {}
        
        for loss_name, loss_attrs in self.criterion[task_name].items():

            # Determing what type of input data
            if loss_attrs['D'] == 'pixel-wise':
                losses[loss_name] = loss_attrs['F'](outputs, inputs)
            elif loss_attrs['D'] == 'matched' and self.HPARAM.PERFORM_MATCHING and self.HPARAM.PERFORM_AGGREGATION:
                losses[loss_name] = loss_attrs['F'](gt_pred_matches)
        
        # Remove losses that have nan
        true_losses = [x for x in losses.values() if torch.isnan(x) == False]

        # If true losses is not empty, continue calculating losses
        if true_losses:

            # Calculate total loss
            #total_loss = torch.sum(torch.stack(true_losses))

            # Calculate the loss multiplied by its corresponded weight
            weighted_losses = [losses[key] * self.criterion[task_name][key]['weight'] for key in losses.keys() if torch.isnan(losses[key]) == False]
            
            # Now calculate the weighted sum
            weighted_sum = torch.sum(torch.stack(weighted_losses))

            # Save the calculated sum in the losses (for PyTorch progress bar logger)
            #losses['loss'] = weighted_sum

            # Saving the total loss (for customized Tensorboard logger)
            losses['task_total_loss'] = weighted_sum #total_loss

        else: # otherwise, just pass losses as nan

            # Place nan in losses (for PyTorch progress bar logger)
            #losses['loss'] = torch.tensor(float('nan')).to(self.device).float()

            # Notate no task-specific total loss (for customized Tensorboard logger)
            losses['task_total_loss'] = torch.tensor(float('nan')).to(self.device).float()

        # Looking for the invalid tensor in the cpu
        """
        for k,v in losses.items():
            if v.device != self.device:
                raise RuntimeError(f'Invalid tensor for not being in the same device as the pl module: {k}')
        """

        return losses

    def calculate_metrics(self, task_name, outputs, inputs, gt_pred_matches):

        with torch.no_grad():

            metrics = {}

            for metric_name, metric_attrs in self.metrics[task_name].items():

                # Determing what type of input data
                if metric_attrs['D'] == 'pixel-wise':
                    
                    # Indexing the task specific output
                    pred = outputs[task_name]
                    gt = inputs[task_name]

                    # Handling times where metrics handle unexpectedly
                    if metric_name == 'iou' and task_name == 'mask':
                        pred = outputs['auxilary']['cat_mask']

                    metrics[metric_name] = metric_attrs['F'](pred, gt)

                elif metric_attrs['D'] == 'matched' and self.HPARAM.PERFORM_MATCHING and self.HPARAM.PERFORM_AGGREGATION:
                    metrics[metric_name] = metric_attrs['F'](gt_pred_matches)

        return metrics

    def on_after_backward(self) -> None:
        # ! Debugging purposes
        #"""
        for section_name, params in {'translation_decoder': self.model.translation_decoder, 'translation_head': self.model.translation_head}.items():
            LOGGER.debug(f"\nSEEING {section_name} parameters")
            for name, param in params.named_parameters():
                if type(param.grad) != type(None):
                    LOGGER.debug(f'{name}: max={torch.max(torch.abs(param.grad))} nan={torch.isfinite(param.grad).all()}')
                else:
                    LOGGER.debug(f'{name}: {param.grad}')
        #"""

    def configure_optimizers(self):

        # Catalyst has new SOTA optimizers out of box
        base_optimizer = catalyst.contrib.nn.RAdam(self.model.parameters(), lr=self.HPARAM.LEARNING_RATE, weight_decay=self.HPARAM.WEIGHT_DECAY)
        #base_optimizer = torch.optim.SGD(self.model.parameters(), lr=self.HPARAM.LEARNING_RATE, weight_decay=self.HPARAM.WEIGHT_DECAY)
        #base_optimizer = torch.optim.Adagrad(self.model.parameters(), lr=self.HPARAM.LEARNING_RATE, weight_decay=self.HPARAM.WEIGHT_DECAY)
        #base_optimizer = torch.optim.Adam(self.model.parameters(), lr=self.HPARAM.LEARNING_RATE, weight_decay=self.HPARAM.WEIGHT_DECAY)
        optimizer = catalyst.contrib.nn.Lookahead(base_optimizer)

        # Solution from here:
        # https://github.com/PyTorchLightning/pytorch-lightning/issues/1598#issuecomment-702038244
        lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer)
        scheduler = {
            'scheduler': lr_scheduler,
            'reduce_on_plateau': True,
            'monitor': 'checkpoint_on',
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
        selected_classes=None,
        encoder=None,
        encoder_weights=None,
        train_size=None,
        valid_size=None
        ):

        super().__init__()
        
        # Saving parameters
        self.dataset_name = dataset_name
        self.selected_classes = selected_classes
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.encoder = encoder
        self.encoder_weights = encoder_weights
        self.train_size = train_size
        self.valid_size = valid_size

    def setup(self, stage=None):

        # Obtaining the preprocessing_fn depending on the encoder and the encoder
        # weights
        if self.encoder and self.encoder_weights:
            preprocessing_fn = smp.encoders.get_preprocessing_fn(self.encoder, self.encoder_weights)
        else:
            preprocessing_fn = None

        # CAMERA / NOCS
        if self.dataset_name == 'CAMERA':

            train_dataset = tools.ds.CAMERADataset(
                dataset_dir=pathlib.Path(os.getenv("NOCS_CAMERA_TRAIN_DATASET")),
                max_size=self.train_size,
                classes=self.selected_classes,
                augmentation=tools.transforms.pose.get_training_augmentation(),
                preprocessing=tools.transforms.pose.get_preprocessing(preprocessing_fn)
            )

            valid_dataset = tools.ds.CAMERADataset(
                dataset_dir=pathlib.Path(os.getenv("NOCS_CAMERA_VALID_DATASET")), 
                max_size=self.valid_size,
                classes=self.selected_classes,
                augmentation=tools.transforms.pose.get_validation_augmentation(),
                preprocessing=tools.transforms.pose.get_preprocessing(preprocessing_fn)
            )

            self.datasets = {
                'train': train_dataset,
                'valid': valid_dataset
            }

        # REAL / NOCS
        elif self.dataset_name == 'REAL':

            train_dataset = tools.ds.REALDataset(
                dataset_dir=pathlib.Path(os.getenv("NOCS_REAL_TRAIN_DATASET")),
                max_size=self.train_size,
                classes=self.selected_classes,
                augmentation=tools.transforms.pose.get_training_augmentation(),
                preprocessing=tools.transforms.pose.get_preprocessing(preprocessing_fn)
            )

            valid_dataset = tools.ds.REALDataset(
                dataset_dir=pathlib.Path(os.getenv("NOCS_REAL_TEST_DATASET")), 
                max_size=self.valid_size,
                classes=self.selected_classes,
                augmentation=tools.transforms.pose.get_validation_augmentation(),
                preprocessing=tools.transforms.pose.get_preprocessing(preprocessing_fn)
            )

            self.datasets = {
                'train': train_dataset,
                'valid': valid_dataset
            }
        
        # INVALID DATASET
        else:
            raise RuntimeError('Dataset needs to be selected')

        print(f"Training datset size: {len(self.datasets['train'])}")
        print(f"Validation dataset size: {len(self.datasets['valid'])}")

    def get_loader(self, dataset_key):

        if dataset_key in self.datasets.keys():        
            
            dataloader = torch.utils.data.DataLoader(
                self.datasets[dataset_key],
                num_workers=self.num_workers,
                batch_size=self.batch_size,
                collate_fn=tools.ds.my_collate_fn,
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

    # Modification of hyperparameters
    #HPARAM.SELECTED_CLASSES = ['bg','camera','laptop']

    # If not debugging, then make matplotlib use the non-GUI backend to 
    # improve stability and speed, otherwise allow debugging sessions to use 
    # matplotlib figures.
    if not HPARAM.DEBUG:
        import matplotlib
        matplotlib.use('Agg')

    # Ensuring that DISTRIBUTED_BACKEND doesn't cause problems
    HPARAM.DISTRIBUTED_BACKEND = None if HPARAM.NUM_GPUS <= 1 else HPARAM.DISTRIBUTED_BACKEND

    # Creating data module
    dataset = PoseRegressionDataModule(
        dataset_name=HPARAM.DATASET_NAME,
        selected_classes=HPARAM.SELECTED_CLASSES,
        batch_size=HPARAM.BATCH_SIZE,
        num_workers=HPARAM.NUM_WORKERS,
        encoder=HPARAM.ENCODER,
        encoder_weights=HPARAM.ENCODER_WEIGHTS,
        train_size=HPARAM.TRAIN_SIZE,
        valid_size=HPARAM.VALID_SIZE
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

    # Deciding if to use a checkpoint to speed up training 
    if HPARAM.CHECKPOINT: # Not None

        # Loading from checkpoint
        checkpoint = torch.load(HPARAM.CHECKPOINT, map_location='cpu')
        OLD_HPARAM = checkpoint['hyper_parameters']

        # Merge the NameSpaces between the model's hyperparameters and 
        # the evaluation hyperparameters
        for attr in OLD_HPARAM.keys():
            if attr in ['BACKBONE_ARCH', 'ENCODER', 'ENCODER_WEIGHTS', 'SELECTED_CLASSES']:
                setattr(HPARAM, attr, OLD_HPARAM[attr])

        # Decrease the learning rate to simply fine tune parameters
        HPARAM.ENCODER_LEARNING_RATE /= 10
        HPARAM.LEARNING_RATE /= 10

        # Create base model
        base_model = lib.PoseRegressor(
            HPARAM,
            intrinsics=torch.from_numpy(tools.pj.constants.INTRINSICS[HPARAM.DATASET_NAME]).float(),
            architecture=HPARAM.BACKBONE_ARCH,
            encoder_name=HPARAM.ENCODER,
            encoder_weights=HPARAM.ENCODER_WEIGHTS,
            classes=len(HPARAM.SELECTED_CLASSES)
        )

        # Create PyTorch Lightning Module
        model = PoseRegresssionTask.load_from_checkpoint(
            str(HPARAM.CHECKPOINT),
            model=base_model,
            criterion=criterion,
            metrics=metrics,
            HPARAM=HPARAM
        )

    else: # no checkpoint
        
        # Creating base model
        base_model = lib.PoseRegressor(
            HPARAM,
            intrinsics=torch.from_numpy(tools.pj.constants.INTRINSICS[HPARAM.DATASET_NAME]).float(),
            architecture=HPARAM.BACKBONE_ARCH,
            encoder_name=HPARAM.ENCODER,
            encoder_weights=HPARAM.ENCODER_WEIGHTS,
            classes=len(HPARAM.SELECTED_CLASSES),
        )

        # Attaching PyTorch Lightning logic to base model
        model = PoseRegresssionTask(HPARAM, base_model, criterion, metrics, HPARAM)

    # Freeze any components of the model
    if HPARAM.FREEZE_ENCODER:
        lib.gtf.freeze(model.model.encoder)
    if HPARAM.FREEZE_MASK_TRAINING:
        lib.gtf.freeze(model.model.mask_decoder)
        lib.gtf.freeze(model.model.segmentation_head)
    if HPARAM.FREEZE_ROTATION_TRAINING:
        lib.gtf.freeze(model.model.rotation_decoder)
        lib.gtf.freeze(model.model.rotation_head)
    if HPARAM.FREEZE_TRANSLATION_TRAINING:
        lib.gtf.freeze(model.model.translation_decoder)
        lib.gtf.freeze(model.model.translation_head)
    if HPARAM.FREEZE_SCALES_TRAINING:
        lib.gtf.freeze(model.model.scales_decoder)
        lib.gtf.freeze(model.model.scales_head)

    # If no runs this day, create a runs-of-the-day folder
    date = datetime.datetime.now().strftime('%y-%m-%d')
    run_of_the_day_dir = pathlib.Path(os.getenv("LOGS")) / date
    if run_of_the_day_dir.exists() is False:
        os.mkdir(str(run_of_the_day_dir))

    # Creating run name
    time = datetime.datetime.now().strftime('%H-%M')
    model_name = f"{HPARAM.ENCODER}-{HPARAM.ENCODER_WEIGHTS}"
    run_name = f"{time}-{HPARAM.EXPERIMENT_NAME}-{HPARAM.DATASET_NAME}-{model_name}"
    
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