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

# Local imports

import project
import dataset
import model_lib
import visualize

#-------------------------------------------------------------------------------
# File Constants

CAMERA_DATASET = project.cfg.DATASET_DIR / 'NOCS' / 'camera' / 'val'

#-------------------------------------------------------------------------------
# Classes

class Trainer():

    def __init__(self, model, train_dataset, valid_dataset, criterions, 
                 batch_size=2, num_workers=0, transform=None):

        # Saving dataset into model
        self.train_dataset = train_dataset
        self.valid_dataset = valid_dataset

        # Create data loaders
        self.train_dataloader = torch.utils.data.DataLoader(self.train_dataset, batch_size=batch_size,
                                                            num_workers=num_workers)
        self.valid_dataloader = torch.utils.data.DataLoader(self.valid_dataset, batch_size=batch_size,
                                                            num_workers=num_workers)

        # Load model
        self.model = model
        self.model_name = self.model.name

        # Saving model name and logs
        self.run_name = datetime.datetime.now().strftime('%d-%m-%y--%H-%M') + '-' + self.model.name
        self.run_save_filepath = str(project.cfg.SAVED_MODEL_DIR / (self.run_name + '.pth') )
        self.run_logs_dir = str(project.cfg.LOGS / self.run_name)

        # Creating a special model logs directory
        if os.path.exists(self.run_logs_dir) is False:
            os.mkdir(self.run_logs_dir)

        # Creating tensorboard object
        # Info here: https://pytorch.org/docs/master/tensorboard.html
        self.tensorboard_writer = torch.utils.tensorboard.SummaryWriter(self.run_logs_dir)
        
        # Using multiple GPUs if avaliable
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        if torch.cuda.device_count() > 1:
            print(f'Using {torch.cuda.device_count()} GPUs!')
            self.model = torch.nn.DataParallel(self.model)
        
        self.model.to(self.device)

        # Load loss functions
        self.criterions = criterions

        # Specify optimizer
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-4, weight_decay=1e-5)

        # Saving optional parameters
        self.batch_size = batch_size
        self.num_workers = num_workers

        # Saving transform
        if transform == 'default':
            transform = 1
        
        self.transform = transform

    def fit(self, num_epoch):

        # Saving num_epoch
        self.num_epoch = num_epoch

        # Tracked metrics
        self.tracked_metrics = {'loss/train': [], 'loss/valid': []}

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

            # Foward propagating random sample
            random_sample = iter(self.valid_dataloader).next()
            with torch.no_grad():
                outputs = self.model(random_sample['color_image'])

            # Creating summary image to visualize the learning of the model
            summary_image = visualize.make_summary_image('Summary', random_sample, outputs)
            self.tensorboard_writer.add_image('summary', summary_image, self.epoch, dataformats='HWC')

            #*******************************************************************
            #                             TERMINAL REPORT
            #*******************************************************************

            # Priting the metrics for the epoch to the terminal
            print(f'Epoch: {self.epoch}\t', end ='')
            for metric_name, metric_values in self.tracked_metrics.items():
                print(f'{metric_name}: {metric_values[-1]:.6f}\t', end='')
            print()

            #*******************************************************************
            #                            TENSORBOARD REPORT
            #*******************************************************************

            # Writing the metrics for the epoch to TensorBoard
            for metric_name, metric_values in self.tracked_metrics.items():
                self.tensorboard_writer.add_scalar(metric_name, metric_values[-1], self.epoch)

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
                       'train dataset size': len(self.train_dataloader),
                       'valid dataset size': len(self.valid_dataloader)}

        # Obtaining the best metrics to report at the end
        best_metrics = {}
        for metric_name, metric_values in self.tracked_metrics.items():
            
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

    def loop(self, dataloader, backpropagate=True):

        running_loss = 0.0

        for sample in tqdm.tqdm(dataloader, desc = f'{self.epoch}/{self.num_epoch}'):

            # clearing gradients
            self.optimizer.zero_grad()

            # Forward pass
            outputs = self.model(sample['color_image'])

            # calculate loss with both the input and output
            loss_mask = self.criterions['masks'](outputs, sample['masks'])
            """
            loss_depth = self.criterions['depth'](outputs[1], depth_image)
            loss_scale = self.criterions['scales'](outputs[2], scales_img)
            loss_quat = self.criterions['quat'](outputs[3], quat_img)
            #"""

            # total loss
            loss = loss_mask #+ loss_depth + loss_scale + loss_quat

            if backpropagate: # only for training
                # compute gradient of the loss
                loss.backward()

                # backpropagate the loss
                self.optimizer.step()

            # update loss
            running_loss += loss.item()

        # Updating tracked metrics

        if backpropagate: # Training
            self.tracked_metrics['loss/train'].append(running_loss / len(dataloader))
        else: # validation
            self.tracked_metrics['loss/valid'].append(running_loss / len(dataloader))

        torch.cuda.empty_cache()

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

#-------------------------------------------------------------------------------
# Functions

#-------------------------------------------------------------------------------
# File Main

if __name__ == '__main__':

    # This is an simple test case, not for training

    # Loading complete dataset
    complete_dataset = dataset.NOCSDataset(CAMERA_DATASET, 10)

    # Splitting dataset to train and validation
    dataset_num = len(complete_dataset)
    split = 0.2
    train_length, valid_length = int(dataset_num*(1-split)), int(dataset_num*split)

    train_dataset, valid_dataset = torch.utils.data.random_split(complete_dataset,
                                                                [train_length, valid_length])

    # Specifying the criterions
    """
    criterions = {'masks':torch.nn.CrossEntropyLoss(),
                  'depth':torch.nn.BCEWithLogitsLoss(),
                  'scales':torch.nn.BCEWithLogitsLoss(),
                  'quat':torch.nn.BCEWithLogitsLoss()}
    #"""
    #criterions = {'masks': model_lib.loss.GDiceLoss(apply_nonlin = torch.nn.Identity())}
    criterions = {'masks': kornia.losses.DiceLoss()}

    # Creating a Trainer
    my_trainer = Trainer(model_lib.FastPoseCNN(in_channels=3, bilinear=True), 
                         train_dataset,
                         valid_dataset,
                         criterions)

    # Fitting Trainer
    my_trainer.fit(2)
