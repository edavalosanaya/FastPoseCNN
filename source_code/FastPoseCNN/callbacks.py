import pdb
import shutil
import os
import operator

import numpy as np

import torch
import torch.nn

import matplotlib.pyplot as plt

import pytorch_lightning as pl
from pytorch_lightning.utilities import rank_zero_only

# Local import
import tools
import lib

#-------------------------------------------------------------------------------
# Classes

class MyCallback(pl.callbacks.Callback):

    @rank_zero_only
    def __init__(self, tasks, hparams, tracked_data, checkpoint_monitor):
        super().__init__()

        # Saving parameters
        self.tasks = tasks
        self.hparams = hparams
        self.tracked_data = tracked_data

        # Creating dictionary containing information about checkpoint
        self.checkpoint_monitor = {}
        for k,v in checkpoint_monitor.items():
            if v == 'min':
                self.checkpoint_monitor[k] = {
                    'mode': 'min',
                    'best_value': np.inf,
                    'saved_checkpoint_fp': None
                }
            elif v == 'max': # max
                self.checkpoint_monitor[k] = {
                    'mode': 'max',
                    'best_value': 0,
                    'saved_checkpoint_fp': None
                }
            else:
                raise RuntimeError('Invalid mode: needs to be min or max')

        # Setting initial values for the metrics/losses
        max_metrics = {k:0 for k in self.tracked_data['maximize']}
        min_metrics = {k:np.inf for k in self.tracked_data['minimize']}
        
        self.metric_dict = max_metrics
        self.metric_dict.update(min_metrics)

    @rank_zero_only
    def on_train_epoch_end(self, trainer, pl_module, outputs):

        # Performing the shared functions of logging after end of epoch
        self.shared_epoch_end('train', trainer, pl_module)

    @rank_zero_only
    def on_validation_epoch_end(self, trainer, pl_module):

        # Performing the shared functions of logging after end of epoch
        self.shared_epoch_end('valid', trainer, pl_module)

        # Creating variable for possible metrics to monitor
        # Removing /batch from the metrics
        monitorable_metrics = [x.replace('/batch', '') for x in list(pl_module.logger.log['valid'].keys())]

        # If there is nothing logged yet, this must be the sanity test. Skip this.
        if monitorable_metrics == []:
            return

        # Save checkpoint depending on logged metrics/losses
        for monitor_name, monitor_data in self.checkpoint_monitor.items():

            # if the monitor name is not in the list of metrics, then raise error
            if monitor_name not in monitorable_metrics:
                raise RuntimeError(f'Invalid monitor name: possible include {monitorable_metrics}')

            # Given the mode ('min' or 'max'), determine if the current epoch 
            # scores better than the previous saved checkpoint
            if monitor_data['mode'] == 'min':
                comparison = operator.le
            else: # 'max'
                comparison = operator.ge

            # Obtaining the values that need to be compared
            current_value = pl_module.logger.log['valid'][monitor_name+'/batch'][0]
            best_value = monitor_data['best_value']

            # If current is better than best, then update
            if comparison(current_value, best_value):

                # Delete previous checkpoint (if it exist)
                if monitor_data['saved_checkpoint_fp']:
                    os.remove(monitor_data['saved_checkpoint_fp'])

                # Generating new checkpoint filepath
                safe_monitor_name = monitor_name.replace("/", "_")
                new_checkpoint_fp = f'epoch={trainer.current_epoch+1}--{safe_monitor_name}={current_value:.2f}.ckpt'
                
                # Avoid using trainer.log_dir, it causes the program to
                # unexplaniably freeze right before training epoch.
                #complete_checkpoint_path = trainer.log_dir + '/checkpoints/' + new_checkpoint_fp
                complete_checkpoint_path = os.getenv('RUNS_LOG_DIR') + '/_/checkpoints/' + new_checkpoint_fp

                # Saving new checkpoint
                #trainer.save_checkpoint(complete_checkpoint_path)

                # Saving the checkpoints location
                self.checkpoint_monitor[monitor_name]['saved_checkpoint_fp'] = complete_checkpoint_path

                # Save the current value as the best value now
                self.checkpoint_monitor[monitor_name]['best_value'] = current_value

        return None

    @rank_zero_only
    def shared_epoch_end(self, mode, trainer, pl_module):

        # Log the average for the metrics for each epoch
        self.log_epoch_average(mode, trainer, pl_module)

        # Depending on the task, create the correct visualization
        if 'mask' in self.tasks:
            # Log visualization of the mask
            self.log_epoch_mask(mode, trainer, pl_module)
        
        if 'quaternion' in self.tasks:
            # Log visualization of quat
            self.log_epoch_quat(mode, trainer, pl_module)

        if 'xy' in self.tasks:
            # Log visualization of xy
            self.log_epoch_xy(mode, trainer, pl_module)

        if 'z' in self.tasks:
            # Log visualization of z
            self.log_epoch_z(mode, trainer, pl_module)

        if 'pose' in self.tasks:
            # Log visualization of pose
            self.log_epoch_pose(mode, trainer, pl_module)

    @rank_zero_only
    def log_epoch_average(self, mode, trainer, pl_module):

        log = pl_module.logger.log[mode]

        for log_name in log.keys():

            tb_log_name = log_name.replace('/batch', '')

            # if the log name is from the desired tensorboard metrics, then use it
            if tb_log_name in sum(self.tracked_data.values(), []):

                # Move all items inside the logger into cuda:0
                cuda_0_log = [x.to('cuda:0') for x in log[log_name]]

                # Obtain the average first
                average = torch.mean(torch.stack(cuda_0_log))

                # Log the average
                trainer.logger.log_metrics(
                    mode, 
                    {f'{tb_log_name}/epoch': average},
                    trainer.current_epoch+1,
                    store=False
                )

                # Save the best value (epoch level) to later log into hparams
                # If maximize, take the maximum value
                if tb_log_name in self.tracked_data['maximize']:
                    if self.metric_dict[tb_log_name] < average:
                        self.metric_dict[tb_log_name] = average
                
                # If minimize, take the minimum value
                elif tb_log_name in self.tracked_data['minimize']:
                    if self.metric_dict[tb_log_name] > average:
                        self.metric_dict[tb_log_name] = average  

                # Store the average as the initial value of the next epoch log
                pl_module.logger.log[mode][log_name] = [average]

        # After an epoch, we clear out the log
        #pl_module.logger.clear_metrics(mode)

    #---------------------------------------------------------------------------
    # Visualizations

    # MASK 
    @rank_zero_only
    def log_epoch_mask(self, mode, trainer, pl_module):

        # Obtaining the LightningDataModule
        datamodule = trainer.datamodule

        # Accessing the corresponding dataset
        dataset = datamodule.datasets[mode]

        # Obtaining the corresponding colormap from the dataset
        colormap = dataset.COLORMAP

        # Get random sample
        sample = dataset.get_random_batched_sample(batch_size=3)

        # Given the sample, make the prediction with the PyTorch Lightning Module
        with torch.no_grad():
            outputs = pl_module(torch.from_numpy(sample['image']).float().to(pl_module.device))
        
        # Obtaining the categorical predicted mask
        pred_cat_mask = outputs['auxilary']['cat_mask'].cpu().numpy()

        # Create the summary figure
        summary_fig = tools.vz.compare_mask_performance(sample, pred_cat_mask, colormap)

        # Log the figure to tensorboard
        pl_module.logger.writers[mode].add_figure(f'mask_gen/{mode}', summary_fig, trainer.global_step)
    
    # QUAT
    @rank_zero_only
    def log_epoch_quat(self, mode, trainer, pl_module):
        
        # Obtaining the LightningDataModule
        datamodule = trainer.datamodule

        # Accessing the corresponding dataset
        dataset = datamodule.datasets[mode]

        # Get random sample
        sample = dataset.get_random_batched_sample(batch_size=3)

        # Given the sample, make the prediciton with the PyTorch Lightning Moduel
        with torch.no_grad():
            outputs = pl_module(torch.from_numpy(sample['image']).float().to(pl_module.device))
        
        # Applying activation function to the mask
        pred_cat_mask = outputs['auxilary']['cat_mask'].cpu().numpy()

        # Selecting the quaternion from the output
        # https://pytorch.org/docs/stable/nn.functional.html?highlight=activation%20functions
        pred_quaternion = outputs['quaternion'].cpu().numpy()
        pred_quaternion /= np.max(np.abs(pred_quaternion))

        # Create the pose figure
        summary_fig = tools.vz.compare_quat_performance(
            sample, 
            pred_quaternion, 
            pred_cat_mask=pred_cat_mask, 
            mask_colormap=dataset.COLORMAP
        )

        # Log the figure to tensorboard
        pl_module.logger.writers[mode].add_figure(f'quat_gen/{mode}', summary_fig, trainer.global_step)      

    # XY
    @rank_zero_only
    def log_epoch_xy(self, mode, trainer, pl_module):

        # Obtaining the LightningDataModule
        datamodule = trainer.datamodule

        # Accessing the corresponding dataset
        dataset = datamodule.datasets[mode]

        # Get random sample
        sample = dataset.get_random_batched_sample(batch_size=3)

        # Given the sample, make the prediciton with the PyTorch Lightning Moduel
        with torch.no_grad():
            outputs = pl_module(torch.from_numpy(sample['image']).float().to(pl_module.device))
        
        # Applying activation function to the mask
        pred_cat_mask = outputs['auxilary']['cat_mask'].cpu().numpy()

        # Selecting the quaternion from the output
        # https://pytorch.org/docs/stable/nn.functional.html?highlight=activation%20functions
        pred_xy = outputs['xy'].cpu().numpy()

        # Create the pose figure
        summary_fig = tools.vz.compare_xy_performance(
            sample, 
            pred_xy, 
            pred_cat_mask=pred_cat_mask, 
            mask_colormap=dataset.COLORMAP
        )

        # Log the figure to tensorboard
        pl_module.logger.writers[mode].add_figure(f'xy_gen/{mode}', summary_fig, trainer.global_step)      

    # Z
    def log_epoch_z(self, mode, trainer, pl_module):
        
        # Obtaining the LightningDataModule
        datamodule = trainer.datamodule

        # Accessing the corresponding dataset
        dataset = datamodule.datasets[mode]

        # Get random sample
        sample = dataset.get_random_batched_sample(batch_size=3)

        # Given the sample, make the prediciton with the PyTorch Lightning Moduel
        with torch.no_grad():
            outputs = pl_module(torch.from_numpy(sample['image']).float().to(pl_module.device))
        
        # Applying activation function to the mask
        pred_cat_mask = outputs['auxilary']['cat_mask'].cpu().numpy()

        # Selecting the quaternion from the output
        # https://pytorch.org/docs/stable/nn.functional.html?highlight=activation%20functions
        pred_z = outputs['z'].cpu().numpy()

        # Create the pose figure
        summary_fig = tools.vz.compare_z_performance(
            sample, 
            pred_z, 
            pred_cat_mask=pred_cat_mask, 
            mask_colormap=dataset.COLORMAP
        )

        # Log the figure to tensorboard
        pl_module.logger.writers[mode].add_figure(f'z_gen/{mode}', summary_fig, trainer.global_step)      

    # POSE
    @rank_zero_only
    def log_epoch_pose(self, mode, trainer, pl_module):

        # Obtaining the LightningDataModule
        datamodule = trainer.datamodule

        # Accessing the corresponding dataset
        dataset = datamodule.datasets[mode]

        # Get random sample
        sample = dataset.get_random_batched_sample(batch_size=2)

        # Convert the random sample (numpy) to a batch (torch)
        batch = {k:torch.from_numpy(v).to(pl_module.device) for k,v in sample.items()}

        # Given the sample, make the prediciton with the PyTorch Lightning Moduel
        with torch.no_grad():
            outputs = pl_module(batch['image'].float())

        # Obtain the matches between aggregated predictions and ground truth data
        agg_gt = lib.gtf.dense_class_data_aggregation(
            mask=batch['mask'],
            dense_class_data=batch
        )

        # Determine matches between the aggreated ground truth and preds
        gt_pred_matches = lib.gtf.find_matches_batched(
            outputs['auxilary']['agg_pred'],
            agg_gt
        )

        # Create summary for the pose
        """
        summary_fig = tools.vz.compare_pose_performance_v4(
            sample,
            outputs,
            gt_pred_matches,
            dataset.INTRINSICS
        )

        # Log the figure to tensorboard
        pl_module.logger.writers[mode].add_figure(f'pose_gen/{mode}', summary_fig, trainer.global_step)      
        """
        
    #---------------------------------------------------------------------------
    # End of Training

    @rank_zero_only
    def teardown(self, trainer, pl_module, stage):
        
        """
        # Log hyper parameters:
            - Using the initialization parameter self.hparams, we use that as the
              hparam_dict while the logger will contain the metrics_dict.
        """
        pl_module.logger.writers['base'].add_hparams(
            hparam_dict=self.hparams,
            metric_dict=self.metric_dict
        )

        # Remove the additional folder for this hyperparameter entry
        base_log_dir = pl_module.logger.base_dir

        # For all the items inside the base_log_dir of the run
        for child in base_log_dir.iterdir():

            # If the child is a directory
            if child.is_dir():

                # Take all the files out into the log_dir
                for file_in_child in child.iterdir():
                    shutil.move(str(file_in_child), str(base_log_dir))

                # Then delete the ugly folder >:(
                os.rmdir(str(child))
