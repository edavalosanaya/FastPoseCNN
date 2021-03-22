import sys
import os
import base64
import abc
import logging
import pprint

from typing import Optional, OrderedDict, Union

import numpy as np
from numpy.core.fromnumeric import compress

import torch
import torch.nn as nn
import torch.nn.functional as F

# PyTorch 3d-party libraries
import segmentation_models_pytorch as smp
import pytorch_lightning as pl
import pytorch_lightning.core.decorators
import catalyst.contrib.nn

# Local imports 
sys.path.append(os.getenv("TOOLS_DIR"))
import timer as tm

import initialization as init
import gpu_tensor_funcs as gtf
import aggregation_layer as al
import hough_voting as hv
import matching as mg

#-------------------------------------------------------------------------------
# Constants

LOGGER = logging.getLogger('fastposecnn')

#-------------------------------------------------------------------------------
# Timers

FORWARD_TIMER = tm.TimerDecorator('forward') # Total Timer
MODEL_TIMER = tm.TimerDecorator('model') # Pure Pytorch Model Timer
AGG_TIMER = tm.TimerDecorator('Aggregation') # Aggregation Timer
HV_TIMER = tm.TimerDecorator('Hough Voting') # Hough Voting Timer
RT_CAL_TIMER = tm.TimerDecorator('RT Calculation') # RT Calculating Timer
CLASS_COMPRESS_TIMER = tm.TimerDecorator('Class Compression') # Class Compression Timer

#-------------------------------------------------------------------------------
# Small Helper Functions

def compress_dict(my_dict, additional_subkey=None):

    new_dict = {}

    for key in my_dict.keys():

        if additional_subkey:
            new_dict[f"{key}/{additional_subkey}"] = None

        for subkey in my_dict[key].keys():
            new_dict[f"{key}/{subkey}"] = my_dict[key][subkey]

    return new_dict

#-------------------------------------------------------------------------------
# PyTorch Class Wrapper for Training

class PoseRegressionTask(pl.LightningModule):

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

        # Error flags
        self.past_inf_flag = False

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

        # If batch is None, skip it
        if not batch:
            LOGGER.debug("EMPTY BATCH, SKIPPED!")
            return None
        
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
        loggable_metrics = compress_dict(multi_task_metrics)

        # Logging the train metrics
        result.log_dict(loggable_metrics)

        return result

    def validation_step(self, batch, batch_idx):

        # If batch is None, skip it
        if not batch:
            LOGGER.debug("EMPTY BATCH, SKIPPED")
            return None
        
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
        loggable_metrics = compress_dict(multi_task_metrics)

        # Logging the val metrics
        result.log_dict(loggable_metrics)

        return result

    def shared_step(self, mode, batch, batch_idx):

        LOGGER.info(f'BATCH ID={batch_idx} ({self.device})')
        LOGGER.info(f"({self.device}) IMAGES={pprint.pformat(batch['path'])}")

        #! Debugging the inf problem
        #batch = torch.load('/home/students/edavalos/GitHub/FastPoseCNN/source_code/FastPoseCNN/logs/21-03-04/18-43-INF_CATCH1-CAMERA-resnet18-imagenet/inf_batch_epoch=1.pth', map_location = self.device)

        # Forward pass the input and generate the prediction of the NN
        outputs = self.forward(batch['image'])

        # Matching aggregated data between ground truth and predicted
        if self.HPARAM.PERFORM_AGGREGATION and self.HPARAM.PERFORM_MATCHING:
            # Determine matches between the aggreated ground truth and preds
            gt_pred_matches = mg.batchwise_find_matches(
                outputs['auxilary']['agg_pred'],
                batch['agg_data']
            )
        else:
            gt_pred_matches = None

        #LOGGER.debug(f"\nMATCHED DATA {self.device}\n" + pprint.pformat(gt_pred_matches))
        
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
        #LOGGER.debug(f"\nALL LOSSESS {self.device}\n" + pprint.pformat(multi_task_losses))

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

        # Before the end of this loop, save a detach instance of the matched data
        self.batch = batch

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
                    if task_name == 'mask':
                        pred = outputs['auxilary']['cat_mask']

                    metrics[metric_name] = metric_attrs['F'](pred, gt)

                elif metric_attrs['D'] == 'matched' and self.HPARAM.PERFORM_MATCHING and self.HPARAM.PERFORM_AGGREGATION:
                    metrics[metric_name] = metric_attrs['F'](gt_pred_matches)

        return metrics

    def on_after_backward(self) -> None:
        # ! Debugging purposes
        #"""
        nan_flag = False
        inf_flag = False

        for name, param in dict(self.model.named_parameters()).items():
            #LOGGER.debug(f"\nSEEING {section_name} PARAMETERS")
            if type(param.grad) != type(None):

                # Catching nans
                any_nans = torch.isnan(param.grad).any()

                if any_nans:
                    nan_flag = True
                
                # Catching the instance that a gradient is not finite (+- inf)
                any_inf = ~torch.isfinite(param.grad).all()

                if any_inf:
                    inf_flag = True

        """
        If an infinite loss is calculated, then we need to save the input batch 
        that caused the infinite loss and the weights and biases of the model to
        easily replicate this issue.
        """

        if inf_flag:

            #print("\nDestructive INF detected!\n")
            LOGGER.debug("DESTRUCTIVE INF DETECTED!")

            # # Saving the most up-to-date batch
            # pth_path = pathlib.Path(os.environ['RUNS_LOG_DIR']) / f'inf_batch_epoch={self.trainer.current_epoch+1}.pth'
            # torch.save(self.batch, str(pth_path))

            # # Saving the model's weights and biases
            # ckpt_path = pathlib.Path(os.environ['RUNS_LOG_DIR']) / f'inf_ckpt_epoch={self.trainer.current_epoch+1}.pth'
            # self.trainer.save_checkpoint(ckpt_path)

            # Clearing gradients
            # https://stackoverflow.com/questions/48001598/why-do-we-need-to-call-zero-grad-in-pytorch
            #print("Clearing gradients\n")
            LOGGER.debug("CLEARING GRADIENTS")
            self.model.zero_grad()

            # Keeping track of past inf
            self.past_inf_flag = True

        elif nan_flag and self.past_inf_flag:

            # # Saving the most up-to-date batch
            # pth_path = pathlib.Path(os.environ['RUNS_LOG_DIR']) / f'nan_batch_epoch={self.trainer.current_epoch+1}.pth'
            # torch.save(self.batch, str(pth_path))

            # # Saving the model's weights and biases
            # ckpt_path = pathlib.Path(os.environ['RUNS_LOG_DIR']) / f'nan_ckpt_epoch={self.trainer.current_epoch+1}.pth'
            # self.trainer.save_checkpoint(ckpt_path)

            # Stopping training 
            print("\nNans procceding desctructive INF detected. Stopping training!\n")
            LOGGER.debug("NANS PROCCEDING DESRUCTIVE INF DETECTED. STOPPING TRAINING!")
            sys.exit(0)

        else:

            # If nans are not detected after the inf, then clear the history of the 
            # of the flag.
            if self.past_inf_flag:
                LOGGER.debug("STABITLY DETECTED AFTER INF CORRECTION. CONTINUING TRAINING")
                self.past_inf_flag = False
        
        #"""
        return None

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

#-------------------------------------------------------------------------------
# Model's Abstract Class

class Model(object):

    @CLASS_COMPRESS_TIMER
    def class_compression(self, logits):

        # Create categorical mask
        cat_mask = torch.argmax(torch.nn.LogSoftmax(dim=1)(logits['mask']), dim=1)

        # Class compression of the data
        cc_logits = gtf.class_compress3(self.classes, cat_mask, logits)

        return cat_mask, cc_logits

    @AGG_TIMER
    def aggregate(self, cat_mask, data):
        
        # Perform aggregation
        agg_data = self.aggregation_layer.forward(cat_mask, data)

        return agg_data

    @HV_TIMER
    def hough_voting(self, agg_data):

        # Perform hough voting
        agg_data = self.hough_voting_layer(agg_data)

        return agg_data

    @RT_CAL_TIMER
    def perform_RT_calculation(self, agg_data):

        # Perform RT calculation
        agg_data = gtf.samplewise_get_RT(agg_data, self.inv_intrinsics)

        return agg_data

    # Shared aggregation, hough voting and RT generation function
    def agg_hough_and_generate_RT(self, cat_mask, data):

        # If aggregation is wanted, perform it
        if self.HPARAM.PERFORM_AGGREGATION:
            # Aggregating the results
            agg_data = self.aggregate(cat_mask, data)

            # If hough voting is wanted, perform it
            if self.HPARAM.PERFORM_HOUGH_VOTING:
                # Perform hough voting
                agg_data = self.hough_voting(agg_data)

                # If RT calculation is wanted, perform it
                if self.HPARAM.PERFORM_RT_CALCULATION:
                    # Calculate RT
                    agg_data = self.perform_RT_calculation(agg_data)

        else:
            return None 

        return agg_data

    @classmethod
    def load_from_ckpt(self, ckpt_path, HPARAM):

        # Catching the scenario where no ckpt is selected
        if type(ckpt_path) != type(None):

            # Loading from checkpoint
            checkpoint = torch.load(ckpt_path, map_location='cpu')
            
            # Getting the old parameters
            OLD_HPARAM = checkpoint['hyper_parameters']

            # Merge the NameSpaces between the model's hyperparameters and 
            # the evaluation hyperparameters
            for attr in OLD_HPARAM.keys():
                if attr in ['MODEL', 'BACKBONE_ARCH', 'ENCODER', 'ENCODER_WEIGHTS', 'SELECTED_CLASSES']:
                    setattr(HPARAM, attr, OLD_HPARAM[attr])

            # Constructing a new model with the new HPARAMS
            model = self.construct_model(HPARAM)

            # Renaming the state_dict to remove the 'model.' component that is saved by
            # PyTorch-Lightning. This is to make the model not rely on PyTorch-pl for 
            # everything
            striped_state_dict = OrderedDict([(k.replace('model.',''),v) for (k,v) in checkpoint['state_dict'].items()])

            # Loading the weights to the new model
            model.load_state_dict(striped_state_dict)

        else: # just construct the model and return it

            model = self.construct_model(HPARAM)

        return model

    @classmethod
    def construct_model(self, HPARAM):

        # Create base model
        model = self(
            HPARAM=HPARAM,
            architecture=HPARAM.BACKBONE_ARCH,
            encoder_name=HPARAM.ENCODER,
            encoder_weights=HPARAM.ENCODER_WEIGHTS,
            classes=len(HPARAM.SELECTED_CLASSES)
        )

        # Appending the timers to calculate runtime
        model.TIMERS = [MODEL_TIMER, AGG_TIMER, HV_TIMER, RT_CAL_TIMER, CLASS_COMPRESS_TIMER, FORWARD_TIMER]

        # Enable timers bases on HPARA.RUNTIME_TIMING param
        if HPARAM.RUNTIME_TIMING:
            for timer in model.TIMERS:
                timer.enabled = True

        return model

    def report_runtime(self):

        if self.HPARAM.RUNTIME_TIMING:
            for timer in self.TIMERS:
                print(f"{timer.name}: {timer.average:.3f} ms - {timer.fps} fps")

        else:
            print("Incapable of Runtime calculation: Set RUNTIME_TIMING = True next time.")

#-------------------------------------------------------------------------------
# Implemented Models

class PoseRegressor(Model, torch.nn.Module):

    # Inspired by 
    # https://github.com/qubvel/segmentation_models.pytorch/blob/1f1be174738703af225b6d7c5da90c6c04ce275b/segmentation_models_pytorch/base/model.py#L5
    # https://github.com/qubvel/segmentation_models.pytorch/blob/master/segmentation_models_pytorch/fpn/decoder.py
    # https://github.com/qubvel/segmentation_models.pytorch/blob/1f1be174738703af225b6d7c5da90c6c04ce275b/segmentation_models_pytorch/encoders/__init__.py#L32

    def __init__(
        self,
        HPARAM,
        architecture: str = 'FPN',
        encoder_name: str = "resnet34",
        encoder_depth: int = 5,
        encoder_weights: Optional[str] = "imagenet",
        decoder_pyramid_channels: int = 256,
        decoder_segmentation_channels: int = 128,
        decoder_merge_policy: str = "add",
        decoder_dropout: float = 0.2,
        in_channels: int = 3,
        classes: int = 2, # bg and one more class
        activation: Optional[str] = None,
        upsampling: int = 4
        ):

        torch.nn.Module.__init__(self)

        # Storing crucial parameters
        self.HPARAM = HPARAM # other algorithm and run hyperparameters
        self.classes = classes # includes background
        self.intrinsics = torch.from_numpy(HPARAM.NUMPY_INTRINSICS).float()
        self.inv_intrinsics = torch.inverse(self.intrinsics)

        # Obtain encoder
        self.encoder = smp.encoders.get_encoder(
            encoder_name,
            in_channels=in_channels,
            depth=encoder_depth,
            weights=encoder_weights,
        )

        # Obtain decoder
        if architecture == 'FPN':

            param_dict = {
                'encoder_channels': self.encoder.out_channels,
                'encoder_depth': encoder_depth,
                'pyramid_channels': decoder_pyramid_channels,
                'segmentation_channels': decoder_segmentation_channels,
                'dropout': decoder_dropout,
                'merge_policy': decoder_merge_policy,
            }

            self.mask_decoder = smp.fpn.decoder.FPNDecoder(**param_dict)
            self.rotation_decoder = smp.fpn.decoder.FPNDecoder(**param_dict)
            self.translation_decoder = smp.fpn.decoder.FPNDecoder(**param_dict)
            self.scales_decoder = smp.fpn.decoder.FPNDecoder(**param_dict)

        # Obtain segmentation head
        self.segmentation_head = smp.base.SegmentationHead(
            in_channels=self.mask_decoder.out_channels,
            out_channels=classes,
            activation=activation,
            kernel_size=1,
            upsampling=upsampling,
        )

        # Creating rotation head (quaternion or rotation matrix)
        self.rotation_head = smp.base.SegmentationHead(
            in_channels=self.rotation_decoder.out_channels,
            out_channels=4*(classes-1), # Removing the background
            activation=activation,
            kernel_size=1,
            upsampling=upsampling,
        )

        # Creating translation head (xyz)
        self.translation_head = smp.base.SegmentationHead(
            in_channels=self.translation_decoder.out_channels,
            out_channels=3*(classes-1), # Removing the background
            activation=activation,
            kernel_size=1,
            upsampling=upsampling,
        )

        # Creating scales head (height, width, and length)
        self.scales_head = smp.base.SegmentationHead(
            in_channels=self.scales_decoder.out_channels,
            out_channels=3*(classes-1), # Removing the background
            activation=activation,
            kernel_size=1,
            upsampling=upsampling,
        )

        # Creating aggregation layer
        self.aggregation_layer = al.AggregationLayer(
            self.HPARAM, 
            self.classes
        )

        # Creating hough voting layer
        self.hough_voting_layer = hv.HoughVotingLayer(
            self.HPARAM
        )

        # initialize the network
        init.initialize_decoder(self.mask_decoder)
        init.initialize_head(self.segmentation_head)

        init.initialize_decoder(self.rotation_decoder)
        init.initialize_head(self.rotation_head)

        init.initialize_decoder(self.translation_decoder)
        init.initialize_head(self.translation_head)

        init.initialize_decoder(self.scales_decoder)
        init.initialize_head(self.scales_head)

        # Use the HPARAM variables to freeze certain components
        # Freeze any components of the model
        if HPARAM.FREEZE_ENCODER:
            gtf.freeze(self.encoder)
        if HPARAM.FREEZE_MASK_TRAINING:
            gtf.freeze(self.mask_decoder)
            gtf.freeze(self.segmentation_head)
        if HPARAM.FREEZE_ROTATION_TRAINING:
            gtf.freeze(self.rotation_decoder)
            gtf.freeze(self.rotation_head)
        if HPARAM.FREEZE_TRANSLATION_TRAINING:
            gtf.freeze(self.translation_decoder)
            gtf.freeze(self.translation_head)
        if HPARAM.FREEZE_SCALES_TRAINING:
            gtf.freeze(self.scales_decoder)
            gtf.freeze(self.scales_head)

    @MODEL_TIMER
    def pure_model_forward(self, x):

        # Encoder
        features = self.encoder(x)
        
        # Decoders
        mask_decoder_output = self.mask_decoder(*features)
        rotation_decoder_output = self.rotation_decoder(*features)
        translation_decoder_output = self.translation_decoder(*features)
        scales_decoder_output = self.scales_decoder(*features)

        # Heads 
        mask_logits = self.segmentation_head(mask_decoder_output)
        quat_logits = self.rotation_head(rotation_decoder_output)
        xyz_logits = self.translation_head(translation_decoder_output)
        scales_logits = self.scales_head(scales_decoder_output)

        # Spliting the (xyz) to (xy, z) since they will eventually have different
        # ways of computing the loss.
        xy_index = np.array([i for i in range(xyz_logits.shape[1]) if i%3!=0]) - 1
        z_index = np.array([i for i in range(xyz_logits.shape[1]) if i%3==0]) + 2
        xy_logits = xyz_logits[:,xy_index,:,:]
        z_logits = xyz_logits[:,z_index,:,:]

        # Storing all logits in a dictionary
        logits = {
            'mask': mask_logits,
            'quaternion': quat_logits,
            'scales': scales_logits,
            'xy': xy_logits,
            'z': z_logits
        }

        return logits

    @FORWARD_TIMER
    def forward(self, x):

        # Ensuring that intrinsics is in the same device
        if self.intrinsics.device != x.device:
            self.intrinsics = self.intrinsics.to(x.device)
            self.inv_intrinsics = torch.inverse(self.intrinsics)

        # Pass through the Pure Pytorch Model and get the raw logits
        logits = self.pure_model_forward(x)

        # Perform Class Compression
        cat_mask, cc_logits = self.class_compression(logits)

        # Perform aggregation, hough voting, and generate RT matrix given the 
        # results o f previous operations.
        agg_pred = self.agg_hough_and_generate_RT(
            cat_mask,
            cc_logits
        )

        # Generating complete output
        output = {
            'mask': logits['mask'],
            **cc_logits,
            'auxilary': {
                'cat_mask': cat_mask,
                'agg_pred': agg_pred
            }
        }

        return output

class PoseRegressor2(Model, torch.nn.Module):

    # Inspired by 
    # https://github.com/qubvel/segmentation_models.pytorch/blob/1f1be174738703af225b6d7c5da90c6c04ce275b/segmentation_models_pytorch/base/model.py#L5
    # https://github.com/qubvel/segmentation_models.pytorch/blob/master/segmentation_models_pytorch/fpn/decoder.py
    # https://github.com/qubvel/segmentation_models.pytorch/blob/1f1be174738703af225b6d7c5da90c6c04ce275b/segmentation_models_pytorch/encoders/__init__.py#L32

    def __init__(
        self,
        HPARAM,
        architecture: str = 'FPN',
        encoder_name: str = "resnet34",
        encoder_depth: int = 5,
        encoder_weights: Optional[str] = "imagenet",
        decoder_pyramid_channels: int = 256,
        decoder_segmentation_channels: int = 128,
        decoder_merge_policy: str = "add",
        decoder_dropout: float = 0.2,
        in_channels: int = 3,
        classes: int = 2, # bg and one more class
        activation: Optional[str] = None,
        upsampling: int = 4
        ):

        super().__init__()

        # Storing crucial parameters
        self.HPARAM = HPARAM # other algorithm and run hyperparameters
        self.classes = classes # includes background
        self.intrinsics = torch.from_numpy(HPARAM.NUMPY_INTRINSICS).float()
        self.inv_intrinsics = torch.inverse(self.intrinsics)

        # Obtain encoder
        self.encoder = smp.encoders.get_encoder(
            encoder_name,
            in_channels=in_channels,
            depth=encoder_depth,
            weights=encoder_weights,
        )

        # Obtain decoder
        if architecture == 'FPN':

            param_dict = {
                'encoder_channels': self.encoder.out_channels,
                'encoder_depth': encoder_depth,
                'pyramid_channels': decoder_pyramid_channels,
                'segmentation_channels': decoder_segmentation_channels,
                'dropout': decoder_dropout,
                'merge_policy': decoder_merge_policy,
            }

            self.decoder = smp.fpn.decoder.FPNDecoder(**param_dict)

        # Obtain segmentation head
        self.segmentation_head = smp.base.SegmentationHead(
            in_channels=self.decoder.out_channels,
            out_channels=classes,
            activation=activation,
            kernel_size=1,
            upsampling=upsampling,
        )

        # Creating rotation head (quaternion or rotation matrix)
        self.rotation_head = smp.base.SegmentationHead(
            in_channels=self.decoder.out_channels,
            out_channels=4*(classes-1), # Removing the background
            activation=activation,
            kernel_size=1,
            upsampling=upsampling,
        )

        # Creating translation head (xyz)
        self.translation_head = smp.base.SegmentationHead(
            in_channels=self.decoder.out_channels,
            out_channels=3*(classes-1), # Removing the background
            activation=activation,
            kernel_size=1,
            upsampling=upsampling,
        )

        # Creating scales head (height, width, and length)
        self.scales_head = smp.base.SegmentationHead(
            in_channels=self.decoder.out_channels,
            out_channels=3*(classes-1), # Removing the background
            activation=activation,
            kernel_size=1,
            upsampling=upsampling,
        )

        # Creating aggregation layer
        self.aggregation_layer = al.AggregationLayer(
            self.HPARAM, 
            self.classes
        )

        # Creating hough voting layer
        self.hough_voting_layer = hv.HoughVotingLayer(
            self.HPARAM
        )

        # initialize the network
        init.initialize_decoder(self.decoder)
        init.initialize_head(self.segmentation_head)
        init.initialize_head(self.rotation_head)
        init.initialize_head(self.translation_head)
        init.initialize_head(self.scales_head)

        # Use the HPARAM variables to freeze certain components
        # Freeze any components of the model
        if HPARAM.FREEZE_ENCODER:
            gtf.freeze(self.encoder)
        if HPARAM.FREEZE_MASK_TRAINING:
            gtf.freeze(self.segmentation_head)
        if HPARAM.FREEZE_ROTATION_TRAINING:
            gtf.freeze(self.rotation_head)
        if HPARAM.FREEZE_TRANSLATION_TRAINING:
            gtf.freeze(self.translation_head)
        if HPARAM.FREEZE_SCALES_TRAINING:
            gtf.freeze(self.scales_head)

    @MODEL_TIMER
    def pure_model_forward(self, x):

        # Encoder
        features = self.encoder(x)
        decoder_output = self.decoder(*features)

        # Heads 
        mask_logits = self.segmentation_head(decoder_output)
        quat_logits = self.rotation_head(decoder_output)
        xyz_logits = self.translation_head(decoder_output)
        scales_logits = self.scales_head(decoder_output)

        # Spliting the (xyz) to (xy, z) since they will eventually have different
        # ways of computing the loss.
        xy_index = np.array([i for i in range(xyz_logits.shape[1]) if i%3!=0]) - 1
        z_index = np.array([i for i in range(xyz_logits.shape[1]) if i%3==0]) + 2
        xy_logits = xyz_logits[:,xy_index,:,:]
        z_logits = xyz_logits[:,z_index,:,:]

        # Storing all logits in a dictionary
        logits = {
            'mask': mask_logits,
            'quaternion': quat_logits,
            'scales': scales_logits,
            'xy': xy_logits,
            'z': z_logits
        }

        return logits

    @FORWARD_TIMER
    def forward(self, x):

        # Ensuring that intrinsics is in the same device
        if self.intrinsics.device != x.device:
            self.intrinsics = self.intrinsics.to(x.device)
            self.inv_intrinsics = torch.inverse(self.intrinsics)

        logits = self.pure_model_forward(x)

        # Perform Class Compression
        cat_mask, cc_logits = self.class_compression(logits)

        # Perform aggregation, hough voting, and generate RT matrix given the 
        # results o f previous operations.
        agg_pred = self.agg_hough_and_generate_RT(
            cat_mask,
            cc_logits
        )

        # Generating complete output
        output = {
            'mask': logits['mask'],
            **cc_logits,
            'auxilary': {
                'cat_mask': cat_mask,
                'agg_pred': agg_pred
            }
        }

        return output

#-------------------------------------------------------------------------------
# Available models (after model definitions)

MODELS = {
    'PoseRegressor': PoseRegressor,
    'Experimental': PoseRegressor2
}

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

    print(y)

