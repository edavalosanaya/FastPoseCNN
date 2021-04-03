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

PATH = pathlib.Path('/home/students/edavalos/GitHub/FastPoseCNN/source_code/FastPoseCNN/logs/good_saved_runs/all_object/19-52-PVNET_HV-PoseRegressor-CAMERA-resnet18-imagenet/_/checkpoints/epoch=37-checkpoint_on=1.0512.ckpt')
HPARAM = config.INFERENCE()
DRAW_IMAGE = 20

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

# Construct model (if ckpt=None, it will just load as is)
model = lib.pose_regressor.MODELS[HPARAM.MODEL].load_from_ckpt(
    PATH,
    HPARAM
)

# Put the model into evaluation mode
model = model.to('cuda') # ! Make it work with multiple GPUs
model = model.eval()

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

    # Skip if the batch is empty
    if type(batch) == type(None):
        continue

    # Move the batch to the device
    batch = tools.ds.move_batch_to(batch, 'cuda:0')
    
    # Forward Propagating
    with torch.no_grad():
        outputs = model.forward(batch['image'])

    if batch_id > DRAW_IMAGE:
        break

    # Determine matches between the aggreated ground truth and preds
    gt_pred_matches = lib.mg.batchwise_find_matches(
        outputs['aggregated'],
        batch['agg_data']
    )

    # # Visualize ground truth data
    # gt_data_fig = tools.vz.visualize_gt_pose(batch, tools.pj.constants.INTRINSICS['CAMERA'])

    # Visualize the poses
    pose_data_fig = tools.vz.compare_pose_performance_v5(
        batch['clean_image'],
        gt_pred_matches,
        outputs['categorical']['mask'],
        tools.pj.constants.COLORMAP[HPARAM.DATASET_NAME],
        HPARAM.NUMPY_INTRINSICS
    )

    # # Saving visualization in temp_folder
    # if gt_data_fig:
    #     gt_data_fig.savefig(
    #         str(pathlib.Path(os.getenv('TEST_OUTPUT')) / f'{batch_id}_gt_data.png'),
    #         dpi=300
    #     )
    if pose_data_fig:
        pose_data_fig.savefig(
            str(pathlib.Path(os.getenv('TEST_OUTPUT')) / f'{batch_id}_pose_data.png'),
            dpi=300
        )

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

    # Saving the raw mask
    # mask_img = outputs['categorical']['mask'].cpu().numpy()
    # mask_path = pathlib.Path(os.getenv('TEST_OUTPUT')) / f'{batch_id}-raw_mask.npy'
    # np.save(str(mask_path), mask_img)

    # # Saving the raw hough voting vectors!
    # hv_img = outputs['aggregated']['xy_mask'].cpu().numpy()
    # hv_path = pathlib.Path(os.getenv('TEST_OUTPUT')) / f'{batch_id}-raw_hv.npy'
    # np.save(str(hv_path), hv_img)

# At the end of running loop, calculate the runtime of each model
if HPARAM.RUNTIME_TIMING:
    model.report_runtime()