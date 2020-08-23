import os
import sys
import pathlib
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

import numpy as np
import cv2

# Local Imports
root = next(path for path in pathlib.Path(os.path.abspath(__file__)).parents if path.name == 'FastPoseCNN')
sys.path.append(str(root))

import project
import dataset
import visualize
from model_lib.fastposecnn import FastPoseCNN as Net

#-------------------------------------------------------------------------------
# File Constants

DEBUG = False
camera_dataset = project.cfg.DATASET_DIR / 'NOCS' / 'camera' / 'val'

#-------------------------------------------------------------------------------
# Functions

def setup(val_split=0.2, dataset_max_size=100, batch_size=10, num_workers=0):

    # Model filename
    model_name = datetime.datetime.now().strftime('%d-%m-%y--%H-%M') + '-fastposecnn'
    model_save_filepath = str(project.cfg.SAVED_MODEL_DIR / (model_name + '.pth') )
    model_logs_dir = str(project.cfg.LOGS / model_name)

    # Creating tensorboard object
    tensorboard_writer = torch.utils.tensorboard.SummaryWriter(model_logs_dir)

    if os.path.exists(model_logs_dir) is False:
        os.mkdir(model_logs_dir)

    # Loading dataset
    dataset = tools.dataset.NOCSDataset(camera_dataset, dataset_max_size=dataset_max_size)

    # Splitting dataset to train and validation
    dataset_num = len(dataset)
    
    indices = list(range(dataset_num))
    np.random.shuffle(indices)

    split = int(np.floor(val_split * dataset_num))
    train_idx, valid_idx = indices[split:], indices[:split]

    # Obtain samplers for training and validation batches
    #print('Creating samplers')
    train_sampler = torch.utils.data.sampler.SubsetRandomSampler(train_idx)
    valid_sampler = torch.utils.data.sampler.SubsetRandomSampler(valid_idx)

    # Create data loaders
    train_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size,
                                               sampler=train_sampler, num_workers=num_workers)
    valid_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size,
                                               sampler=valid_sampler, num_workers=num_workers)

    # Load model
    net = Net(in_channels=3, bilinear=True)
    
    # Using multiple GPUs if avaliable
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    if torch.cuda.device_count() > 1:
        print(f'Using {torch.cuda.device_count()} GPUs!')
        net = torch.nn.DataParallel(net)
    
    net.to(device)

    # Load loss functions
    criterions = {'masks':torch.nn.CrossEntropyLoss(),
                  'depth':torch.nn.BCEWithLogitsLoss(),
                  'scales':torch.nn.BCEWithLogitsLoss(),
                  'quat':torch.nn.BCEWithLogitsLoss()}

    # Specify optimizer
    optimizer = torch.optim.Adam(net.parameters(), lr=1e-4, weight_decay=1e-5)

    return net, criterions, optimizer, train_loader, valid_loader, tensorboard_writer

def train(num_epoch=5):

    net, criterions, optimizer, train_loader, valid_loader, tensorboard_writer = setup(dataset_max_size=None)

    # Epoch Loop
    for epoch in range(1, num_epoch+2):

        # Signaling epoch
        #print(f'Entering epoch: {epoch}')
        epoch_info = f'Epoch {epoch}/{num_epoch}'

        # Keepting track of training and validation loss
        tracked_metrics = {'loss/train':0.0, 'loss/valid':0.0}

        #***********************************************************************
        #                               TRAINING
        #***********************************************************************

        # Set model to train 
        net.train()

        # Feed training input
        print('Training loop') 
        tracked_metrics['loss/train'] = training_loop(epoch_info, net, criterions, optimizer, train_loader, backpropagate=True)

        #***********************************************************************
        #                              VALIDATE MODEL
        #***********************************************************************

        # Set model to eval
        net.eval()

        # Feed validating input
        print('Validation Loop')
        tracked_metrics['loss/valid'] = training_loop(epoch_info, net, criterions, optimizer, valid_loader, backpropagate=False)

        #***********************************************************************
        #                             DATA PROCESSING
        #***********************************************************************

        # calculate average loss
        tracked_metrics['loss/train'] /= (len(train_loader.sampler))
        tracked_metrics['loss/valid'] /= (len(valid_loader.sampler))

        # Foward propagating random sample
        random_sample = iter(train_loader).next()
        color_image, depth_image, zs, masks, coord_map, scales_img, quat_img = random_sample
        outputs = net(color_image)

        title = f'Summary'
        summary_image = tools.visualize.make_summary_image(title, random_sample, outputs)
        tensorboard_writer.add_image(title, summary_image, epoch, dataformats='HWC')

        #***********************************************************************
        #                             TERMINAL REPORT
        #***********************************************************************

        print('Epoch: {} \tloss/train: {:.6f} \tloss/valid: {:.6f}'.format(epoch, tracked_metrics['loss/train'], tracked_metrics['loss/valid']))

        #***********************************************************************
        #                            TENSORBOARD REPORT
        #***********************************************************************

        for metric_name, metric_value in tracked_metrics.items():
            tensorboard_writer.add_scalar(metric_name, metric_value, epoch)

        #***********************************************************************
        #                               SAVE MODEL
        #***********************************************************************

        # Save model if validation loss is minimum
        
        #torch.save(net.state_dict(), model_save_filepath)

    return None

def training_loop(epoch_info, net, criterions, optimizer, dataloader, backpropagate=True):

    running_loss = 0.0

    for sample in tqdm.tqdm(dataloader, desc = epoch_info):

            # Deconstructing the sample tuple
            color_image, depth_image, zs, masks, coord_map, scales_img, quat_img = sample
            
            # clearing gradients
            optimizer.zero_grad()

            # Forward pass
            outputs = net(color_image)

            # calculate loss with both the input and output
            loss_mask = criterions['masks'](outputs, masks)
            """
            loss_depth = criterions['depth'](outputs[1], depth_image)
            loss_scale = criterions['scales'](outputs[2], scales_img)
            loss_quat = criterions['quat'](outputs[3], quat_img)
            #"""

            # total loss
            loss = loss_mask #+ loss_depth + loss_scale + loss_quat

            if backpropagate: # only for training
                # compute gradient of the loss
                loss.backward()

                # backpropagate the loss
                optimizer.step()

            # update loss
            running_loss += loss.item()

    torch.cuda.empty_cache()

    return running_loss

#-------------------------------------------------------------------------------
# Main Code

if __name__ == '__main__':

    train(num_epoch=25)