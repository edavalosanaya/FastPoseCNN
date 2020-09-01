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

    def __init__(self, model, datasets, dataloaders, criterion, optimizer, device, 
                 n_classes, batch_size=2, num_workers=0, transform=None):

        # Saving dataset into model
        self.train_dataset = datasets[0]
        self.valid_dataset = datasets[1]

        # Create data loaders
        self.train_dataloader = dataloaders[0]
        self.valid_dataloader = dataloaders[1]

        # Load model
        self.model = model
        
        # Obtain the model name
        try:
            self.model_name = self.model.name
        except:
            self.model_name = 'no-name-net'

        # Load loss functions
        self.criterion = criterion

        # Specify optimizer
        self.optimizer = optimizer

        # Saving optional parameters
        self.batch_size = batch_size
        self.num_workers = num_workers

        # Saving transform
        if transform == 'default':
            transform = 1
        
        self.transform = transform

        # Saving the selected device
        self.device = device

        # Saving the number of classes
        self.n_classes = n_classes

    def loop(self, dataloader, backpropagate=True):

        # Initialize metrics
        self.per_sample_metrics['loss'] = 0.0
        
        self.per_sample_metrics['P'] = torch.zeros(self.n_classes, device=self.device)
        self.per_sample_metrics['N'] = torch.zeros(self.n_classes, device=self.device)

        self.per_sample_metrics['TP'] = torch.zeros(self.n_classes, device=self.device)
        self.per_sample_metrics['TN'] = torch.zeros(self.n_classes, device=self.device)
        
        self.per_sample_metrics['FP'] = torch.zeros(self.n_classes, device=self.device)
        self.per_sample_metrics['FN'] = torch.zeros(self.n_classes, device=self.device)

        self.per_sample_metrics['accuracy'] = torch.zeros(self.n_classes, device=self.device)
        self.per_sample_metrics['precision'] = torch.zeros(self.n_classes, device=self.device)
        self.per_sample_metrics['recall'] = torch.zeros(self.n_classes, device=self.device)

        # Overating overall all samples in the epoch
        for sample in tqdm.tqdm(dataloader, desc = f'{self.epoch}/{self.num_epoch}'):

            # clearing gradients
            self.optimizer.zero_grad()

            # Forward pass
            logits_masks = self.model(sample['color_image'])

            # calculate loss with both the input and output
            loss_mask = self.criterion(logits_masks, sample['masks'])

            # total loss
            loss = loss_mask #+ loss_depth + loss_scale + loss_quat

            if backpropagate: # only for training
                # compute gradient of the loss
                loss.backward()

                # backpropagate the loss
                self.optimizer.step()

            # Update per_sample metrics
            self.per_sample_metrics['loss'] += loss.item()

            # Calculate the pred masks
            pred_masks = torch.argmax(logits_masks, dim=1)
            
            # Per sample in batch
            for pred_mask, mask in zip(pred_masks, sample['masks']):
                # Per class inside the mask
                for class_id in torch.unique(mask):

                    # Selecting the class' target and prediction mask
                    target_class_mask = sample['masks'] == class_id
                    pred_class_mask = pred_mask == class_id
                    
                    # Positive & Negative
                    self.per_sample_metrics['P'][class_id] = torch.sum(target_class_mask)
                    self.per_sample_metrics['N'][class_id] = torch.sum(~target_class_mask)

                    # True
                    self.per_sample_metrics['TP'][class_id] = torch.sum(torch.logical_and(target_class_mask, pred_class_mask))
                    self.per_sample_metrics['TN'][class_id] = torch.sum(torch.logical_and(~target_class_mask, ~pred_class_mask))

                    # False
                    self.per_sample_metrics['FP'][class_id] = torch.sum(torch.logical_and(pred_class_mask, ~target_class_mask))
                    self.per_sample_metrics['FN'][class_id] = torch.sum(torch.logical_and(~pred_class_mask, target_class_mask))

                    # Calculating accuracy, precision, recall
                    accuracy = self.per_sample_metrics['TP'][class_id] / self.per_sample_metrics['P'][class_id]
                    
                    if self.per_sample_metrics['accuracy'][class_id] == 0:
                        self.per_sample_metrics['accuracy'][class_id] = accuracy
                    else:
                        self.per_sample_metrics['accuracy'][class_id] += accuracy
                        self.per_sample_metrics['accuracy'][class_id] /= 2

                    precision = self.per_sample_metrics['TP'][class_id] / (self.per_sample_metrics['TP'][class_id] + self.per_sample_metrics['FP'][class_id])

                    if self.per_sample_metrics['precision'][class_id] == 0:
                        self.per_sample_metrics['precision'][class_id] = precision
                    else:
                        self.per_sample_metrics['precision'][class_id] += precision
                        self.per_sample_metrics['precision'][class_id] /= 2

                    recall = self.per_sample_metrics['TP'][class_id] / (self.per_sample_metrics['TP'][class_id] + self.per_sample_metrics['FN'][class_id])

                    if self.per_sample_metrics['recall'][class_id] == 0:
                        self.per_sample_metrics['recall'][class_id] = recall
                    else:
                        self.per_sample_metrics['recall'][class_id] += recall
                        self.per_sample_metrics['recall'][class_id] /= 2

        # Update per_batch metrics
        if backpropagate: # Training
            mode = 'train'
        else:
            mode = 'valid'
            
        """
        # Calculating the average of all the classes
        self.per_sample_metrics['accuracy'] = torch.div(self.per_sample_metrics['TP'], self.per_sample_metrics['P'])
        self.per_sample_metrics['precision'] = torch.div(self.per_sample_metrics['TP'], self.per_sample_metrics['TP'] + self.per_sample_metrics['FP'])
        self.per_sample_metrics['recall'] = torch.div(self.per_sample_metrics['TP'], self.per_sample_metrics['TP'] + self.per_sample_metrics['FN'])
        """

        # Saving per batch metrics NOTE: [1:] is to ignore the background
        average_accuracy = float(torch.mean(self.per_sample_metrics['accuracy'][1:]))
        average_precision = float(torch.mean(self.per_sample_metrics['precision'][1:]))
        average_recall = float(torch.mean(self.per_sample_metrics['recall'][1:]))
        average_loss = float(self.per_sample_metrics['loss'] / len(dataloader))

        self.per_epoch_metrics[mode]['accuracy'].append(average_accuracy)
        self.per_epoch_metrics[mode]['precision'].append(average_precision) 
        self.per_epoch_metrics[mode]['recall'].append(average_recall)
        self.per_epoch_metrics[mode]['loss'].append(average_loss)

        torch.cuda.empty_cache()

    def fit(self, num_epoch):

        # Saving num_epoch
        self.num_epoch = num_epoch

        # Tracked metrics
        self.per_epoch_metrics = {'train':{'loss': [], 'accuracy': [], 'precision': [], 'recall': []},
                                  'valid':{'loss': [], 'accuracy': [], 'precision': [], 'recall': []}}
        self.per_sample_metrics = {'loss': None, 'P': None, 'N': None, 'TP': None, 'TN': None, 'FP': None, 'FN': None,
                                   'accuracy': None, 'precision': None, 'recall': None}

        # Epoch Loop
        for self.epoch in range(1, self.num_epoch+1):

            #*******************************************************************
            #                               TRAINING
            #*******************************************************************

            # Set model to train 
            self.model.train()

            # Feed training input
            print('Training loop') 
            self.loop(self.train_dataloader, backpropagate=True)

            #*******************************************************************
            #                              VALIDATE MODEL
            #*******************************************************************

            # Set model to eval
            self.model.eval()

            # Feed validating input
            print('Validation Loop')
            with torch.no_grad():
                self.loop(self.valid_dataloader, backpropagate=False)

            #*******************************************************************
            #                             DATA PROCESSING
            #*******************************************************************

            # If epoch is the first, then initialize tensorboard right before the 
            # first time we write to the tensorboard
            if self.epoch == 1:
                self.initialize_tensorboard()

            # Calculate the global set (len(train_dataset) * epoch)
            global_step = len(self.train_dataset) * self.epoch

            # Foward propagating random sample
            for mode_name, dataloader in zip(['train', 'valid'],[self.train_dataset, self.valid_dataset]):
                
                # Randomly selecting a sample from the dataset
                random_sample = dataloader.get_random_sample(batched=True)

                # Obtaining the output of the neural network 
                with torch.no_grad():
                    logits_masks = self.model(random_sample['color_image'])

                # Calculate the pred masks
                pred_masks = torch.argmax(logits_masks, dim=1)

                # Creating a matplotlib figure illustrating the inputs vs outputs
                summary_fig = visualize.make_summary_figure(random_sample, pred_masks)
                self.tensorboard_writer.add_figure(f'{mode_name}/summary', summary_fig, global_step)

            #*******************************************************************
            #                             TERMINAL REPORT
            #*******************************************************************

            # Priting the metrics for the epoch to the terminal
            print(f'Epoch: {self.epoch}\t', end ='')
            for mode in self.per_epoch_metrics.keys():
                print(f'{mode}\t', end ='')
                for metric_name, metric_values in self.per_epoch_metrics[mode].items():
                    print(f'{metric_name}: {metric_values[-1]:.3f}\t', end='')
                print('\n\t\t', end='')
            print()

            #*******************************************************************
            #                            TENSORBOARD REPORT
            #*******************************************************************

            # Writing the metrics for the epoch to TensorBoard
            for mode in self.per_epoch_metrics.keys():
                for metric_name, metric_values in self.per_epoch_metrics[mode].items():
                    self.tensorboard_writer.add_scalar(f'{metric_name}/{mode}', metric_values[-1], global_step)

            #***********************************************************************
            #                               SAVE MODEL
            #***********************************************************************

            # Save model if validation loss is minimum
            
            #torch.save(net.state_dict(), model_save_filepath)

        # Outside Epoch Loop

        # Saving run information
        run_hparams = {'run_name': self.run_name,
                       'model': self.model_name,
                       'batch_size': self.batch_size,
                       'criterions': repr(self.criterions),
                       'optimizer': repr(self.optimizer),
                       'train dataset size': len(self.train_dataset),
                       'valid dataset size': len(self.valid_dataset)}

        """
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
        """

        # Writing to tensorboard the hparams
        self.tensorboard_writer.add_hparams(run_hparams, {})

        # Removing ugly additional tensorboard folder
        self.remove_folder_in_tensorboard_report()

    def test_forward(self):

        # Loading one sample from train_dataloader
        sample = next(iter(self.train_dataloader))

        # Saving graph model to Tensorboard
        #tensorboard_writer.add_graph(net, color_image)

        # Forward pass
        print('Forward pass')
        outputs = self.model(sample['color_image'])

        """
        # Visualizing output
        pred = torch.argmax(outputs, dim=1)[0]
        vis_pred = visualize.get_visualized_mask(pred)
        numpy_vis_pred = visualize.torch_to_numpy([vis_pred])[0]
        out_path = project.cfg.TEST_OUTPUT / 'segmentation_output.png'
        skimage.io.imsave(str(out_path), numpy_vis_pred)

        # Testing summary image
        summary_image = visualize.make_summary_image('Summary', sample, outputs)
        out_path = project.cfg.TEST_OUTPUT / 'summary_image.png'
        skimage.io.imsave(str(out_path), summary_image)
        """

        # Loss propagate
        loss_mask = self.criterions['masks'](outputs, sample['masks'])
        """
        loss_depth = self.criterions['depth'](outputs[1], depth_image)
        loss_scale = self.criterions['scales'](outputs[2], scales_img)
        loss_quat = self.criterions['quat'](outputs[3], quat_img)
        #"""

        print('Successful backprogagation')

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

    def initialize_tensorboard(self):

        # Saving model name and logs
        self.run_name = datetime.datetime.now().strftime('%d-%m-%y--%H-%M') + '-' + self.model_name
        self.run_save_filepath = str(project.cfg.SAVED_MODEL_DIR / (self.run_name + '.pth') )
        self.run_logs_dir = str(project.cfg.LOGS / self.run_name)

        # Creating a special model logs directory
        if os.path.exists(self.run_logs_dir) is False:
            os.mkdir(self.run_logs_dir)

        # Creating tensorboard object
        # Info here: https://pytorch.org/docs/master/tensorboard.html
        self.tensorboard_writer = torch.utils.tensorboard.SummaryWriter(self.run_logs_dir)

#-------------------------------------------------------------------------------
# Functions

#-------------------------------------------------------------------------------
# File Main

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
    #model = model_lib.unet(n_classes = self.n_classes, feature_scale=4)
    #model = model_lib.FastPoseCNN(in_channels=3, bilinear=True, filter_factor=4)
    #model = model_lib.UNetWrapper(in_channels=3, n_classes=self.n_classes,
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
    
    # Creating trainer
    my_trainer = Trainer(model, 
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
