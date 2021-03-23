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
HPARAM = config.INFERENCE()
DRAW_IMAGE = 10

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
#model = model.to('cuda') # ! Make it work with multiple GPUs
model = model.eval()

# Save the model as a ONNX
onnx_path = pathlib.Path(os.getenv('SAVED_ONNX_MODELS')) / 'test.onnx'
tools.ot.export_onnx_model(model, (1, 3, 480, 640), str(onnx_path))
sys.exit(-1)

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

# At the end of running loop, calculate the runtime of each model
if HPARAM.RUNTIME_TIMING:
    model.report_runtime()