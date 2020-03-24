import sys
import os

import cv2
import glob

# Keras / TensorFlow
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '5'
import tensorflow as tf
import keras

ROOT_DIR = os.getcwd()
DENSE_ROOT_DIR = os.path.join(ROOT_DIR, "DenseDepth")
DENSE_TF_ROOT_DIR = os.path.join(DENSE_ROOT_DIR, "Tensorflow")

sys.path.append(DENSE_TF_ROOT_DIR)

from DenseDepth import layers
from DenseDepth import utils

#----------------------------------------------------------------
# Main Code

# test parameter
custom_objects = {'BilinearUpSampling2D': layers.BilinearUpSampling2D, 'depth_loss_function': None}
image_path = r"E:\MASTERS_STUFF\workspace\bts\dataset\nyu_depth_v2\official_splits\test\computer_lab\rgb_00332.jpg"
checkpoint_path = r"E:\MASTERS_STUFF\workspace\DenseDepth\logs\nyu.h5"

# Loading Model
print("Loading model...")
model = keras.models.load_model(checkpoint_path, custom_objects=custom_objects, compile=False)
print("\n Model loaded ({})".format(checkpoint_path))

# Input images
inputs = utils.load_images( glob.glob(image_path) )

# Network Output
outputs = utils.predict(model, inputs)

# Showing input and output (assuming only one input)
cv2.imshow("Input Image", inputs[0])
cv2.imshow("Output depth", outputs[0])
cv2.waitKey(0)
cv2.destroyAllWindows()


