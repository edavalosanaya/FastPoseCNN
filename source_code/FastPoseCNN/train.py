import os
import sys
import pathlib

import torch
import torch.nn
import torch.optim

import numpy as np

# Local Imports
root = next(path for path in pathlib.Path(os.path.abspath(__file__)).parents if path.name == 'FastPoseCNN')
sys.path.append(str(root))

import tools.project as project
import tools.dataset
from model_lib.fastposecnn import FastPoseCNN as Net

#-------------------------------------------------------------------------------
# File Constants

DEBUG = False
camera_dataset = root.parents[1] / 'datasets' / 'NOCS' / 'camera' / 'val'
model_path = project.cfg.SAVED_MODEL_DIR / 'test_model.pth'

#-------------------------------------------------------------------------------
# Functions

def train(num_epoch, PATH):

    # Loading dataset
    print('Loading dataset')
    dataset = tools.dataset.NOCSDataset(camera_dataset, dataset_max_size=10)

    # Splitting dataset to train and validation
    print('Creating dataset split (train/validation)')
    val_split = 0.2
    dataset_num = len(dataset)
    
    indices = list(range(dataset_num))
    np.random.shuffle(indices)

    split = int(np.floor(val_split * dataset_num))
    train_idx, valid_idx = indices[split:], indices[:split]

    # Obtain samplers for training and validation batches
    print('Creating samplers')
    train_sampler = torch.utils.data.sampler.SubsetRandomSampler(train_idx)
    valid_sampler = torch.utils.data.sampler.SubsetRandomSampler(valid_idx)

    # Create data loaders
    batch_size = 1
    num_workers = 0

    print('Creating data loaders')
    train_loader = tools.dataset.NOCSDataLoader(dataset, batch_size=batch_size,
                                                sampler=train_sampler, num_workers=num_workers)
    valid_loader = tools.dataset.NOCSDataLoader(dataset, batch_size=batch_size,
                                                sampler=valid_sampler, num_workers=num_workers)

    # Load model
    print('Loading model')
    net = Net(n_channels=3, out_channels_mask=6, bilinear=True)
    
    # Using multiple GPUs if avaliable
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    if torch.cuda.device_count() > 1:
        print(f'Using {torch.cuda.device_count()} GPUs!')
        net = torch.nn.DataParallel(net)
    
    net.to(device)

    # Load loss functions
    print('Initializing loss function')
    #criterion_masks = torch.nn.CrossEntropyLoss()
    criterion_masks = torch.nn.BCEWithLogitsLoss()

    # Specify optimizer
    print('Specifying optimizer')
    optimizer = torch.optim.Adam(net.parameters(), lr=1e-4, weight_decay=1e-5)

    # Training loop

    for epoch in range(1, num_epoch+1):

        # Signaling epoch
        #print(f'Entering epoch: {epoch}')

        # Keepting track of training and validation loss
        train_loss = 0.0
        valid_loss = 0.0

        #***********************************
        #             TRAINING
        #***********************************

        # Set model to train 
        net.train()

        # Feed training input
        #print('Entering training loop') 
        for i, sample in enumerate(train_loader):

            # Deconstructing dictionary for simple usage
            color_img, masks, coords, scales, quaternions, norm_factors = [data.to(device) for data in sample.values()]
            
            # clearing gradients
            optimizer.zero_grad()

            # Forward pass
            masks_pred = net(color_img)

            # calculate loss with both the input and output
            masks_loss = criterion_masks(masks_pred, masks)
            
            # total loss
            loss = masks_loss 

            # compute gradient of the loss
            loss.backward()

            # backpropagate the loss
            optimizer.step()

            # update training loss
            train_loss += loss.item()

        #***********************************
        #           VALIDATE MODEL
        #***********************************

        # Set model to eval
        net.eval()

        # Feed validating input
        #print('Entering validation loop')
        for i, sample in enumerate(valid_loader):

            # Deconstructing dictionary for simple usage
            color_img, masks, coords, scales, quaternions, norm_factors = [data.to(device) for data in sample.values()]

            # Forward pass
            masks_pred = net(color_img)

            # calculate loss with both the input and output
            masks_loss = criterion_masks(masks_pred, masks)

            # total loss 
            loss = masks_loss

            # Calculate valid loss
            valid_loss += loss.item()

        # calculate average loss
        train_loss = train_loss/(len(train_loader.sampler))
        valid_loss = valid_loss/(len(valid_loader.sampler))

        #***********************************
        #             REPORT
        #***********************************
        print('Epoch: {} \ttrain_loss: {:.6f} \tvalid_loss: {:.6f}'.format(epoch, train_loss, valid_loss))

        # Report training and validation loss

        #***********************************
        #             SAVE MODEL
        #***********************************

        # Save model if validation loss is minimum
        torch.save(net.state_dict(), PATH)

    return None

#-------------------------------------------------------------------------------
# Main Code

if __name__ == '__main__':

    train(10, model_path)