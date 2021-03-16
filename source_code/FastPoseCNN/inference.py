# External Imports
import os
import sys
import pathlib
import torch
import argparse
import tqdm
import matplotlib.pyplot as plt
import skimage
import skimage.io
import time

import numpy as np

# Local Imports
import setup_env
import tools
import config
import lib

#-------------------------------------------------------------------------------
# Constants

PATH = pathlib.Path('/home/students/edavalos/GitHub/FastPoseCNN/source_code/FastPoseCNN/logs/21-03-06/16-36-BASE_TEST-CAMERA-resnet18-imagenet/_/checkpoints/last.ckpt')
HPARAM = config.DEFAULT_POSE_HPARAM()

#-------------------------------------------------------------------------------
# File Main

# Parse arguments and replace global variables if needed
parser = argparse.ArgumentParser(description='Train with PyTorch Lightning framework')

# Automatically adding all the attributes of the HPARAM to the parser
for attr in dir(HPARAM):
    if '__' in attr or attr[0] == '_': # Private or magic attributes
        continue
    
    parser.add_argument(f'--{attr}', type=type(getattr(HPARAM, attr)), default=getattr(HPARAM, attr))

# Updating the HPARAMs
parser.parse_args(namespace=HPARAM)

# Getting the intrinsics for the dataset selected
HPARAM.NUMPY_INTRINSICS = tools.pj.constants.INTRINSICS[HPARAM.DATASET_NAME]
HPARAM.BATCH_SIZE = 1
HPARAM.VALID_SIZE = 20

# Construct model (if ckpt=None, it will just load as is)
model = lib.pose_regressor.MODELS[HPARAM.MODEL].load_from_ckpt(
    PATH,
    HPARAM
)

# Put the model into evaluation mode
#model.to('cuda') # ! Make it work with multiple GPUs
model.eval()

# Load the PyTorch Lightning dataset
datamodule = tools.ds.PoseRegressionDataModule(
    dataset_name=HPARAM.DATASET_NAME,
    selected_classes=HPARAM.SELECTED_CLASSES,
    batch_size=HPARAM.BATCH_SIZE,
    num_workers=HPARAM.NUM_WORKERS,
    encoder=HPARAM.ENCODER,
    encoder_weights=HPARAM.ENCODER_WEIGHTS,
    train_size=HPARAM.TRAIN_SIZE,
    valid_size=HPARAM.VALID_SIZE
)

# Setup the dataset
datamodule.setup()

# Obtaining the valid dataset
valid_dataset = datamodule.datasets['valid']

# Numpy container for all the data, divided per class
#pred_gt_data = np.zeros((len(HPARAM.SELECTED_CLASSES)))
all_matches = []

# image counter
image_counter = 0

# Pass through all the test data of the dataset and collect the predictions
# and the ground truths
for batch_id, batch in tqdm.tqdm(enumerate(datamodule.val_dataloader())):

    if type(batch) == type(None):
        continue

    # Forward pass
    tic = time.time()
    
    with torch.no_grad():
        outputs = model.forward(batch['image'])
    
    toc = time.time()

    # Determine matches between the aggreated ground truth and preds
    gt_pred_matches = lib.mg.batchwise_find_matches(
        outputs['auxilary']['agg_pred'],
        batch['agg_data']
    )

    # Visualize ground truth data
    # gt_data_fig = tools.vz.visualize_gt_pose(batch, tools.pj.constants.INTRINSICS['CAMERA'])

    # Visualize the poses
    # pose_data_fig = tools.vz.compare_pose_performance_v5(
    #     batch['clean_image'],
    #     gt_pred_matches,
    #     outputs['auxilary']['cat_mask'],
    #     tools.pj.constants.COLORMAP[HPARAM.DATASET_NAME],
    #     HPARAM.NUMPY_INTRINSICS
    # )

    # Saving visualization in temp_folder
    # if gt_data_fig:
    #     gt_data_fig.savefig(
    #         str(pathlib.Path(os.getenv('TEST_OUTPUT')) / f'{batch_id}_gt_data.png'),
    #         dpi=300
    #     )
    # if pose_data_fig:
    #     pose_data_fig.savefig(
    #         str(pathlib.Path(os.getenv('TEST_OUTPUT')) / f'{batch_id}_pose_data.png'),
    #         dpi=300
    #     )

    gt_images, pred_images, pose_images = tools.vz.compare_all_performance(
        batch,
        outputs,
        gt_pred_matches,
        HPARAM.NUMPY_INTRINSICS,
        tools.pj.constants.COLORMAP[HPARAM.DATASET_NAME],
        return_as_fig=False
    )

    # Saving the input RGB image
    rgb = np.squeeze(batch['clean_image'].cpu().numpy())
    rgb_path = pathlib.Path(os.getenv('TEST_OUTPUT')) / f'{batch_id}-rgb.png'
    skimage.io.imsave(str(rgb_path), rgb)

    for group_images in [gt_images, pred_images, pose_images]:
        for image_type_name, image in group_images.items():
            
            # Removing any unnecessary dimension
            image = np.squeeze(image)

            # Saving the image
            image_path = pathlib.Path(os.getenv('TEST_OUTPUT')) / f'{batch_id}-{image_type_name}.png'
            skimage.io.imsave(str(image_path), image)
