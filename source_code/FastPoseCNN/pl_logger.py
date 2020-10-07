import os
import pathlib
from pathlib import Path

from torch.utils.tensorboard.writer import SummaryWriter

import pytorch_lightning as pl
from pytorch_lightning.utilities import rank_zero_only

import pdb

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
        save_dir: Path = Path(pathlib.Path.cwd() / 'logs'), 
        name = 'model',
        **kwargs
        ):

        # Super class initialization
        super().__init__(**kwargs)

        # Saving parameters
        self.save_dir = str(save_dir)
        self.name = name

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
    def log_hyperparams(self, params):
        # params is an argparse.Namespace
        # your code to record hyperparameters goes here
        pass

    @rank_zero_only
    def log_metrics(self, mode, metrics, step, store=True):

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