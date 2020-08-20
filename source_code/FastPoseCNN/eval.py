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

def eval(PATH):

    # Loading dataset
    print('Loading dataset')
    dataset = tools.dataset.NOCSDataset(camera_dataset, dataset_max_size=10)

    # Create data loaders
    batch_size = 1
    num_workers = 0

    print('Creating data loaders')
    dataloader = tools.dataset.NOCSDataLoader(dataset, batch_size=batch_size,
                                              num_workers=num_workers)

    # Load model
    print('Loading model')
    net = Net(n_channels=3, out_channels_mask=6, bilinear=True)
    
    # Using multiple GPUs if avaliable
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    if torch.cuda.device_count() > 1:
        print(f'Using {torch.cuda.device_count()} GPUs!')
        net = torch.nn.DataParallel(net)
    
    net.to(device)

    # Setting network to evaluate
    net.eval()
    net.load_state_dict(torch.load(PATH))

    # Load loss functions
    print('Initializing loss function')
    #criterion_masks = torch.nn.CrossEntropyLoss()
    criterion_masks = torch.nn.BCEWithLogitsLoss()

    # Specify optimizer
    print('Specifying optimizer')
    optimizer = torch.optim.Adam(net.parameters(), lr=1e-4, weight_decay=1e-5)

    valid_loss = 0.0

    for i, sample in enumerate(dataloader):

        #print(f'id: {i} \n {sample.keys()}\n')

        # Deconstructing dictionary for simple usage
        color_img, depth, zs, masks, coords, scales, quaternions = [data.to(device) for data in sample.values()]

        # Forward pass
        masks_pred = net(color_img)

        # calculate loss with both the input and output
        masks_loss = criterion_masks(masks_pred, masks)
        
        # total loss
        loss = masks_loss 

        # update training loss
        valid_loss += loss.item()

    #***********************************
    #           VALIDATE MODEL
    #***********************************

    valid_loss = valid_loss/(len(dataloader.sampler))

    #***********************************
    #             REPORT
    #***********************************
    print('valid_loss: {:.6f}'.format(valid_loss))

    return None

#-------------------------------------------------------------------------------
# Main Code

if __name__ == '__main__':

    eval(model_path)