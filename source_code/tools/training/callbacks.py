# Imports
import os
import sys
import pathlib
import random
import PIL
import io

import tensorflow as tf
import keras
import numpy as np
import sklearn
import skimage

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
sys.path.append(str(DENSE_TF_ROOT_DIR))

# Local Imports

from DenseDepth import utils as dd_utils

#------------------------------------------------------------
# Functions

def get_callbacks(model, basemodel, train_generator, test_generator, output_path):

    # Output
    callbacks = []

    # Getting the tensorboard object
    callbacks.append(MyTensorBoard(str(output_path / 'logs'), model, basemodel, train_generator, test_generator, output_path))

    # Getting the learning rate scheduler
    learning_rate_schedule = keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.7, patience=5, min_lr=0.000009, min_delta=1e-2)
    callbacks.append(learning_rate_schedule)

    # Getting the checkpoint saver
    checkpoint_saver = keras.callbacks.ModelCheckpoint(str(output_path / 'weights.{epoch:02d}-{val_loss:.2f}.hdf5'),
                                                       monitor='val_loss', verbose=1, save_best_only=False, save_weights_only=False,
                                                       mode='min', period=5)
    callbacks.append(checkpoint_saver)

    return callbacks

def make_tensor_to_image(tensor):
    height, width, channel = tensor.shape
    image = PIL.Image.fromarray(tensor.astype('uint8'))
    output = io.BytesIO()
    image.shape(output, format='JPEG', quality=90)
    image_string = output.getvalue()
    output.close()
    return tf.Summary.Image(height=height, width=width, colorspace=channel, encoded_image_string=image_string)

#------------------------------------------------------------
# Classes

class MyTensorBoard(keras.callbacks.TensorBoard):

    def __init__(self, log_dir, model, basemodel, train_generator, test_generator, output_path):
        super().__init__(log_dir=log_dir)

        # Saving inputs
        self.model = model
        self.basemode = basemodel
        self.train_generator = train_generator
        self.test_generator = test_generator
        
        if isinstance(output_path,str) is True: # not a pathlib.Path, make it one
            self.output_path = pathlib.Path(output_path)
        else:
            self.output_path = output_path

        # main parameters
        self.num_samples = 6
        self.train_idx = np.random.randint(low=0, high=len(self.train_generator), size=10)
        self.test_idx = np.random.randint(low=0, high=len(self.test_generator), size=10)

        return None

    def on_epoch_end(self, epoch, logs=None):

        # Samples using current model
        plasma = plt.get_cmap('plasma')
        min_depth, max_depth = 10, 1000

        train_samples = []
        test_samples = []

        for i in range(self.num_samples):
            
            # Loading the color and depth for the training and testing set
            color_train, depth_train = self.train_generator[self.train_idx[i]]
            color_test, depth_test = self.test_generator[self.test_idx[i]]

            # Clipping the images
            color_train, depth_train = color_train[0], np.clip(depth_train[0] / 1000, min_depth, max_depth) / max_depth
            color_test, depth_test = color_test[0], np.clip(depth_test[0] / 1000, min_depth, max_depth) / max_depth

            # Getting the images shape
            height, width = depth_train.shape[0], depth_train.shape[1]

            # Resizing color to match depth size
            resized_color_train = skimage.transform.resize(color_train, (height, width), preserver_range=True, mode='reflect', anti_aliasing=True)
            resized_color_test = skimage.transform.resize(color_test, (height, width), preserver_range=True, mode='reflect', anti_aliasing=True)

            # Getting the ground truth images
            gt_train = plasma(depth_train[:,:,0])[:,:,:3]
            gt_test = plasma(depth_test[:,:,0])[:,:,:3]

            # Make the prediction
            predict_train = plasma(dd_utils.predict(model, color_train, minDepth=min_depth, maxDepth=max_depth)[0,:,:,0])[:,:,:3]
            predict_test = plasma(dd_utils.predict(model, color_test, minDepth=min_depth, maxDepth=max_depth)[0,:,:,0])[:,:,:3]

            # Combining color and depth into a sample
            train_samples.append(np.vstack([resized_color_train, gt_train, predict_train]))
            test_samples.append(np.vstack([resized_color_test, gt_test, predict_test]))

        # Outside of the for i in range(self.num_samples) loop
        self.writer.add_summary(tf.Summary(value=[tf.Summary.Value(tag='Train', image=make_tensor_to_image(255 * np.hstack(train_samples)))]), epoch)
        self.writer.add_summary(tf.Summary(value=[tf.Summary.Value(tag='Test', image=make_tensor_to_image(255 * np.hstack(test_samples)))]), epoch)

        super().on_epoch_end(epoch, logs)

#------------------------------------------------------------
# Main Code