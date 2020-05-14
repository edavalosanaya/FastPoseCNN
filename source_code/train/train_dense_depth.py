# Library Import
import os
import sys
import glob
import pathlib
import time
import socket
import datetime
import argparse

from matplotlib import pyplot as plt
import numpy as np
import cv2

import tqdm 
import json
import pickle

# Keras / TensorFlow
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '5'
import keras
import tensorflow

# Determing the ROOT_DIR from CWD_DIR
CWD_DIR = pathlib.Path.cwd()
ROOT_DIR = CWD_DIR.parents[len(CWD_DIR.parts) - CWD_DIR.parts.index('MastersProject') - 2]
NETWORKS_DIR = ROOT_DIR / 'networks'

# Getting NOCS paths
NOCS_ROOT_DIR = NETWORKS_DIR / "NOCS_CVPR2019"
NOCS_MODEL_DIR = NOCS_ROOT_DIR / "logs"
NOCS_COCO_MODEL_PATH = NOCS_MODEL_DIR / "mask_rcnn_coco.h5"
NOCS_WEIGHTS_PATH = NOCS_MODEL_DIR / "nocs_rcnn_res50_bin32.h5"

# Getting DenseDepth paths
DENSE_ROOT_DIR = NETWORKS_DIR / "DenseDepth"
DENSE_TF_ROOT_DIR = DENSE_ROOT_DIR / "Tensorflow"

# Source Code path
SOURCE_CODE_DIR = ROOT_DIR / "source_code"

# Appending necessary paths
sys.path.append(str(SOURCE_CODE_DIR))
sys.path.append(str(NETWORKS_DIR))
sys.path.append(str(NOCS_ROOT_DIR))
sys.path.append(str(DENSE_ROOT_DIR))

# Local Imports

import tools
from DenseDepth import layers
from DenseDepth import utils as dd_utils
from DenseDepth import model as dd_model
from DenseDepth import loss as dd_loss

#-----------------------------------------------------------------------
# Main Code

# Parameters
parser = argparse.ArgumentParser()
parser.add_argument('--epochs', '-e', required=True, type=int)
parser.add_argument('--batch', '-b', default=1, type=int)
parser.add_argument('--lr', '-l', default=0.0001, type=int)
parser.add_argument('--dataset_size', '-ds', default=0, type=int)
args = parser.parse_args()

epochs = args.epochs
batch_size = args.batch
learning_rate = args.lr
dataset_size = None if args.dataset_size == 0 else args.dataset_size

# Paths
dense_depth_checkpoint = DENSE_ROOT_DIR / 'logs' / 'nyu.h5'
dataset_path = ROOT_DIR / 'datasets' / 'NOCS' / 'real'
output_path = pathlib.Path.cwd() / 'models'

# Create model 
model = dd_model.create_model(existing=str(dense_depth_checkpoint))

# Data loaders
train_generator, test_generator = tools.training.data.get_training_data(dataset_path, batch_size, dataset_size)

# Training session details and saving model name
date = datetime.datetime.now()
timestamp = date.strftime("%d/%b/%Y-%H:%M-")
model_name = "time-{}-e-{}-size-{}-bsize-{}-lr-{}.h5".format(timestamp,
                                                             epochs,
                                                             len(train_generator),
                                                             batch_size,
                                                             learning_rate)
model_save_location = output_path / model_name

# The following line get me an error
# Generate model plot (SVG) 
#keras.utils.vis_utils.plot_model(model, to_file=str(output_path / 'model_plot.svg'), show_shapes=True, show_layer_names=True)

# Multi-gpu setup
#os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2,3' # using all four cores
basemodel = model

# for the lambda machine
if socket.gethostname() == 'lambda-quad': # if the lambda machine, use all cores
    model = keras.utils.multi_gpu_model(model, gpus=4)

# Optimizer
optimizer = keras.optimizers.Adam(lr=learning_rate, amsgrad=True)

# Compile the Model
model.compile(loss=dd_loss.depth_loss_function, optimizer=optimizer)

print("Ready for training!\n")

# Callbacks
callbacks = tools.training.callbacks.get_callbacks(model, basemodel, train_generator, test_generator, output_path)

# Start training
print("Training!\n")
model.fit_generator(train_generator, callbacks=callbacks, validation_data=test_generator, epochs=epochs, shuffle=True)
print("Finished Training!\n")

# Save the final trained model
model.save(str(model_save_location))