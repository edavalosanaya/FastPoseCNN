import os
import pathlib
from pathlib import Path

from torch.utils.tensorboard.writer import SummaryWriter

import pytorch_lightning as pl
from pytorch_lightning.utilities import rank_zero_only

#-------------------------------------------------------------------------------
# Classes

class MyLogger(pl.loggers.LightningLoggerBase):

    experiment = 'FastPoseCNN'
    version = '_'
    name = 'logger'
    save_dir='.'

    @rank_zero_only
    def __init__(
        self, 
        HPARAM,
        pl_module,
        save_dir: Path = Path(pathlib.Path.cwd() / 'logs'), 
        name = 'model',
        **kwargs
        ):

        # Super class initialization
        super().__init__(**kwargs)

        # Saving parameters
        self.save_dir = str(save_dir)
        self.name = name
        self.HPARAM = HPARAM
        self.pl_module = pl_module

        # Creating the necessary directories
        self.run_dir = save_dir / self.name
        self.base_dir = save_dir / self.name / '_base_log'
        self.train_dir = save_dir / self.name / 'train_log'
        self.valid_dir = save_dir / self.name / 'valid_log'

        necessary_dirs = [self.run_dir, self.base_dir, self.train_dir, self.valid_dir]

        for necessary_dir in necessary_dirs:
            if necessary_dir.exists() is False:
                os.mkdir(str(necessary_dir))

        # Creating Summary Writer for both base, training and validation
        self.writers = {
            'base': SummaryWriter(log_dir=str(self.base_dir)),
            'train': SummaryWriter(log_dir=str(self.train_dir)),
            'valid': SummaryWriter(log_dir=str(self.valid_dir))
        }

        # Creating logging variables
        self.log = {
            'base': {},
            'train': {},
            'valid': {}
        }

    @rank_zero_only
    def calculate_global_step(self, mode, batch_idx):

        # Calculate the actual global step
        if self.pl_module.trainer.train_dataloader is not None:
            num_of_train_batchs = len(self.pl_module.trainer.train_dataloader)
        else:
            num_of_train_batchs = 0

        if self.pl_module.trainer.val_dataloaders[0] is not None:
            num_of_valid_batchs = len(self.pl_module.trainer.val_dataloaders[0])
        else:
            num_of_valid_batchs = 0

        total_batchs = num_of_train_batchs + num_of_valid_batchs - 1

        # Calculating the effective batch_size
        if self.pl_module.use_ddp:
            effective_batch_size = self.HPARAM.BATCH_SIZE * self.HPARAM.NUM_GPUS
        elif self.pl_module.use_single_gpu or self.pl_module.use_dp:
            effective_batch_size = self.HPARAM.BATCH_SIZE
        else:
            raise NotImplementedError('Invalid distributed backend option')

        # Calculating the effective batch_idx
        if self.pl_module.use_ddp:
            effective_batch_idx = batch_idx * self.HPARAM.NUM_GPUS + 1
        elif self.pl_module.use_single_gpu or self.pl_module.use_dp:
            effective_batch_idx = batch_idx
        else:
            raise NotImplementedError('Invalid distributed backend option')

        if mode == 'train':
            epoch_start_point = self.pl_module.current_epoch * total_batchs * effective_batch_size
            global_step = epoch_start_point + effective_batch_idx * self.HPARAM.BATCH_SIZE + 1
        elif mode == 'valid':
            epoch_start_point = self.pl_module.current_epoch * total_batchs * effective_batch_size + num_of_train_batchs * effective_batch_size
            global_step = epoch_start_point + effective_batch_idx * self.HPARAM.BATCH_SIZE + 1
        else:
            epoch_start_point = 0
            global_step = self.pl_module.global_step
        
        return global_step

    @rank_zero_only
    def log_hyperparams(self, params):
        # params is an argparse.Namespace
        # your code to record hyperparameters goes here
        pass

    @rank_zero_only
    def log_metrics(self, mode, metrics, batch_idx, store=True):

        # Calculate global step, since given information regarding the step is 
        # given in batch_idx
        step = self.calculate_global_step(mode, batch_idx)

        # Logging the metrics invidually to tensorboard
        for metric_name, metric_value in metrics.items():
            self.writers[mode].add_scalar(metric_name, metric_value, step)

        # Saving the metrics for later use
        if store:
            self.save_metrics(mode, metrics)

    @rank_zero_only
    def save_metrics(self, mode, metrics):

        for metric_name, metric_value in metrics.items():

            # Save the information into the logs
            # If log metric has been saved before, simply just append it to the list
            if metric_name in self.log[mode].keys():
                self.log[mode][metric_name].append(metric_value)
            # else the log metric is new so we need to create a list for it
            else:
                self.log[mode][metric_name] = [metric_value]

    @rank_zero_only
    def clear_metrics(self, mode=None):

        # If no mode is specified, just completely reset the log
        if mode is None:
            self.log = {
            'base': {},
            'train': {},
            'valid': {}
        }

        # else reset that specific mode
        else:
            self.log[mode] = {}

    @rank_zero_only
    def agg_and_log_metrics(self, scalar_metrics, step):
        # Necessary empty function
        pass

    @rank_zero_only
    def save(self):
        
        # Flushing all the writers to make sure all the logs are completed
        for writer in self.writers.values():
            writer.flush()

    @rank_zero_only
    def finalize(self, status):
        
        # Flushing and closing all the writers
        for writer in self.writers.values():
            writer.flush()
            writer.close()