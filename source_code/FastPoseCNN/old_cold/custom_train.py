import os
import sys
import pathlib
import shutil
import datetime
import tqdm
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

import numpy as np

import segmentation_models_pytorch as smp

# Local imports

import project
import dataset
import model_lib
import visualize
import transforms

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

#-------------------------------------------------------------------------------
# File Constants

CAMERA_TRAIN_DATASET = project.cfg.DATASET_DIR / 'NOCS' / 'camera' / 'train'
CAMERA_VALID_DATASET = project.cfg.DATASET_DIR / 'NOCS' / 'camera' / 'val'

VOC_DATASET = project.cfg.DATASET_DIR / 'VOC2012'

#-------------------------------------------------------------------------------
# Classes

class Trainer():

    def __init__(self, 
                 run_name, 
                 model, 
                 datasets, 
                 dataloaders, 
                 loss,
                 metrics,
                 optimizer, 
                 device, 
                 batch_size=1):

        # Saving the name of the run
        self.run_name = run_name

        # Saving dataset into model
        self.train_dataset = datasets[0]
        self.valid_dataset = datasets[1]
        self.test_dataset = datasets[2]
        self.test_dataset_vis = datasets[3]

        # Create data loaders
        self.train_dataloader = dataloaders[0]
        self.valid_dataloader = dataloaders[1]

        # Load model
        self.model = model

        # Load loss functions
        self.loss = loss

        # Saving the metrics
        self.metrics = metrics

        # Specify optimizer
        self.optimizer = optimizer

        # Saving optional parameters
        self.batch_size = batch_size

        # Saving the selected device
        self.device = device

        # Saving the number of classes
        self.n_classes = self.train_dataset.CLASSES

    def loop(self, dataloader, train=True):

        # Tracked metrics
        self.logs = {}
        self.loss_meter = smp.utils.meter.AverageValueMeter()
        self.metrics_meters = {metric.__name__: smp.utils.meter.AverageValueMeter() for metric in self.metrics}

        # Overating overall all samples in the epoch
        with tqdm.tqdm(dataloader, desc = f'{self.epoch}/{self.num_epoch}') as iterator:
            
            for sample in iterator:
                
                # Moving the sample to the device
                for key in sample:
                    sample[key] = sample[key].to(self.device)

                # clearing gradients
                self.optimizer.zero_grad()

                # Forward pass
                prediction = self.model(sample['image'])

                # calculate loss with both the input and output
                loss_mask = self.loss(prediction, sample['mask'])

                # total loss
                loss = loss_mask #+ loss_depth + loss_scale + loss_quat

                if train: # only for training
                    # compute gradient of the loss
                    loss.backward()

                    # backpropagate the loss
                    self.optimizer.step()

                # Update loss log
                loss_value = loss.cpu().detach().numpy()
                self.loss_meter.add(loss_value)
                
                loss_log = {self.loss.__name__: self.loss_meter.mean}
                self.logs.update(loss_log)

                # Updating metrics
                for metric_fn in self.metrics:
                    metric_value = metric_fn(prediction, sample['mask']).cpu().detach().numpy()
                    self.metrics_meters[metric_fn.__name__].add(metric_value)
                
                metrics_logs = {k: v.mean for k, v in self.metrics_meters.items()}
                self.logs.update(metrics_logs)

                # Updating tqdm string
                str_logs = ['{} - {:.4}'.format(k, v) for k, v in self.logs.items()]
                summary_text = ', '.join(str_logs)
                iterator.set_postfix_str(summary_text)

        torch.cuda.empty_cache()

    def fit(self, num_epoch):

        # Saving num_epoch
        self.num_epoch = num_epoch

        # Epoch Loop
        for self.epoch in range(1, self.num_epoch+1):

            # Calculate the global set (len(train_dataset) * epoch)
            self.global_step = len(self.train_dataset) * self.epoch

            if self.epoch % 20 == 0:
                self.optimizer.param_groups[0]['lr'] /= 2
                print(f"lr decreased to {self.optimizer.param_groups[0]['lr']}")

            #*******************************************************************
            #                               TRAINING
            #*******************************************************************

            # Set model to train 
            self.model.train()

            # Feed training input
            print('Training loop') 
            self.loop(self.train_dataloader, train=True)

            # If epoch is the first, then initialize tensorboard right before the 
            # first time we write to the tensorboard
            if self.epoch == 1:
                self.initialize_tensorboard()

            self.log_scalar_metrics_tb(train_valid='train')

            #*******************************************************************
            #                              VALIDATE MODEL
            #*******************************************************************

            # Set model to eval
            self.model.eval()

            # Feed validating input
            print('Validation Loop')
            with torch.no_grad():
                self.loop(self.valid_dataloader, train=False)
                self.log_scalar_metrics_tb(train_valid='valid')
                
            summary_fig = self.mask_check_tb()
            self.tensorboard_writer.add_figure(f'test/summary', summary_fig, self.global_step)

            #*******************************************************************
            #                            TENSORBOARD REPORT
            #*******************************************************************

            #***********************************************************************
            #                               SAVE MODEL
            #***********************************************************************

            # Save model if validation loss is minimum
            
            #torch.save(net.state_dict(), model_save_filepath)

        # Outside Epoch Loop

        """
        # Saving run information
        run_hparams = {'run_name': self.run_name,
                       'model': self.model_name,
                       'batch_size': self.batch_size,
                       'loss': repr(self.loss),
                       'optimizer': repr(self.optimizer),
                       'train dataset size': len(self.train_dataset),
                       'valid dataset size': len(self.valid_dataset)}

        # Obtaining the best metrics to report at the end
        best_metrics = {}
        for metric_name, metric_values in self.per_epoch_metrics.items():
            
            new_metric_name = metric_name.capitalize()

            if 'loss' in metric_name:
                best_metrics[new_metric_name] = min(metric_values)
            else: # Accuracy
                best_metrics[new_metric_name] = max(metric_values)

        # Converge hparams and best metrics to avoid the add_scalars aspect of
        # the add_hparams function
        run_hparams.update(best_metrics)

        # Writing to tensorboard the hparams
        self.tensorboard_writer.add_hparams(run_hparams, {})

        # Removing ugly additional tensorboard folder
        self.remove_folder_in_tensorboard_report()
        """

    ############################################################################
    # Tensorboard
    ############################################################################

    def log_scalar_metrics_tb(self, train_valid):

        for metric_name, metric_value in self.logs.items():
            self.tensorboard_writer.add_scalar(f'{metric_name}/{train_valid}', metric_value, self.global_step)

    def mask_check_tb(self):

        # Selecting a random indicie from the dataset
        n = np.random.choice(len(self.test_dataset))

        # Randomly selecting a sample from the dataset
        sample_vis = self.test_dataset_vis[n]
        image_vis = sample_vis['image'].astype(np.uint8)
        gt_mask_vis = sample_vis['mask']
        
        sample = self.test_dataset[n]
        image = sample['image']
        gt_mask = sample['mask']

        # Molding the input
        gt_mask = gt_mask.squeeze().round()
        x_tensor = torch.from_numpy(image).to(self.device).unsqueeze(0)
        
        # Obtaining the output of the neural network 
        with torch.no_grad():
            pr_mask = self.model(x_tensor)
            pr_mask = (pr_mask.squeeze().cpu().numpy().round())

        pr_mask = np.argmax(pr_mask, axis=0)
        gt_mask_vis = np.argmax(gt_mask_vis, axis=2)

        # Obtaining colormap given the dataset

        # Creating a matplotlib figure illustrating the inputs vs outputs
        summary_fig = visualize.make_summary_figure(
            colormap=self.test_dataset.COLORMAP,
            image=image_vis,
            ground_truth_mask=gt_mask_vis,
            predicited_mask=pr_mask)

        return summary_fig

    def initialize_tensorboard(self):

        # Saving model name and logs
        self.run_name = datetime.datetime.now().strftime('%d-%m-%y--%H-%M') + '-' + self.run_name
        self.run_save_filepath = str(project.cfg.SAVED_MODEL_DIR / (self.run_name + '.pth') )
        self.run_logs_dir = str(project.cfg.LOGS / self.run_name)

        # Creating a special model logs directory
        if os.path.exists(self.run_logs_dir) is False:
            os.mkdir(self.run_logs_dir)

        # Creating tensorboard object
        # Info here: https://pytorch.org/docs/master/tensorboard.html
        self.tensorboard_writer = torch.utils.tensorboard.SummaryWriter(self.run_logs_dir)
        print(f'Tensorboard folder: {self.run_name}')

    def remove_folder_in_tensorboard_report(self):

        log_dir = pathlib.Path(self.run_logs_dir)

        # For all the items inside the log_dir of the run
        for child in log_dir.iterdir():

            # If the child is a directory
            if child.is_dir():

                # Take all the files out into the log_dir
                for file_in_child in child.iterdir():
                    shutil.move(str(file_in_child), str(log_dir))

                # Then delete the ugly folder >:(
                os.rmdir(str(child))           

#-------------------------------------------------------------------------------
# Functions

#-------------------------------------------------------------------------------
# File Main

if __name__ == '__main__':

    # Run constants
    encoder = 'se_resnext50_32x4d'
    encoder_weights = 'imagenet'
    preprocessing_fn = smp.encoders.get_preprocessing_fn(encoder, encoder_weights)

    run_name = f'smp-{encoder}-{encoder_weights}'

    #***************************************************************************
    # Loading dataset 
    #***************************************************************************
    
    """ # NOCS (OLD)
    train_dataset = dataset.NOCSDataset(CAMERA_TRAIN_DATASET, 1000, balance=True)
    valid_dataset = dataset.NOCSDataset(CAMERA_VALID_DATASET, 100)
    #"""
    
    """ # VOC (OLD)
    crop_size = (320, 480)
    train_dataset = dataset.VOCSegDataset(True, crop_size, VOC_DATASET)
    valid_dataset = dataset.VOCSegDataset(False, crop_size, VOC_DATASET)
    """

    # Collecting the datasets into tuple
    datasets = train_dataset, valid_dataset, test_dataset, test_dataset_vis

    #***************************************************************************
    # Creating dataloaders 
    #***************************************************************************
    # Dataloader parameters
    batch_size = 8
    num_workers = 2

    # For using multple CPUs for fast dataloaders
    # More information can be found in the following link:
    # https://github.com/pytorch/pytorch/issues/40403
    if num_workers > 0:
        torch.multiprocessing.set_start_method('spawn') # good solution !!!!

    train_dataloader = torch.utils.data.DataLoader(train_dataset, num_workers=num_workers*2,
                                                   batch_size=batch_size, shuffle=True)
    valid_dataloader = torch.utils.data.DataLoader(valid_dataset, num_workers=num_workers,
                                                   batch_size=1, shuffle=True)
    dataloaders = train_dataloader, valid_dataloader

    #***************************************************************************
    # Loading model
    #***************************************************************************
    #model = model_lib.unet(n_classes = self.n_classes, feature_scale=4)
    #model = model_lib.FastPoseCNN(in_channels=3, bilinear=True, filter_factor=4)
    #model = model_lib.UNetWrapper(in_channels=3, n_classes=self.n_classes,
    #                              padding=True, wf=4, depth=4)

    activation = 'sigmoid'
    model = smp.FPN(
        encoder_name=encoder, 
        encoder_weights=encoder_weights, 
        classes=len(train_dataset.CLASSES),
        activation=activation
    )

    # Using multiple GPUs if avaliable
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    if torch.cuda.device_count() > 1:
        print(f'Using {torch.cuda.device_count()} GPUs!')
        model = torch.nn.DataParallel(model)
    
    model.to(device)

    #***************************************************************************
    # Selecting loss function, metrics and optimizer
    loss = smp.utils.losses.DiceLoss()  
    #loss = {'masks': kornia.losses.DiceLoss()}
    #loss = {'masks': torch.nn.CrossEntropyLoss()}
    #loss = torch.nn.CrossEntropyLoss()
    #loss = kornia.losses.DiceLoss()

    metrics = [
        smp.utils.metrics.IoU(threshold=0.5),
        smp.utils.metrics.Fscore(threshold=0.5),
        #smp.utils.metrics.Accuracy(threshold=0.5),
        smp.utils.metrics.Recall(threshold=0.5),
        smp.utils.metrics.Precision(threshold=0.5)
    ]

    optimizer = torch.optim.Adam([ 
        dict(params=model.parameters(), lr=0.0001),
    ])
    #***************************************************************************

    #***************************************************************************
    # Selecting Trainer 
    #***************************************************************************
    epoch = 20
    
    # Creating trainer
    my_trainer = Trainer(run_name,
                         model, 
                         datasets,
                         dataloaders,
                         loss,
                         metrics,
                         optimizer,
                         device,
                         batch_size=batch_size)

    # Fitting Trainer
    my_trainer.fit(epoch)
