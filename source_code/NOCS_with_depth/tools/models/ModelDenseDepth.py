# Library Imports
import os
import sys
import pathlib
import glob

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' # Basically, SHUT UP TENSORFLOW
import tensorflow
import keras
import numpy as np

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

# Appending necessary paths
sys.path.append(str(NETWORKS_DIR))
sys.path.append(str(NOCS_ROOT_DIR))
sys.path.append(str(DENSE_TF_ROOT_DIR))

# Local Imports
from DenseDepth import layers
from DenseDepth import utils as dd_utils

#---------------------------------------------------------------------------
# Class

class DenseDepth():

    def __init__(self, checkpoint_path=False, load_model=True):

        if checkpoint_path == False:
            checkpoint_path = str(DENSE_ROOT_DIR / "logs" / "nyu.h5")

        self.checkpoint_path = checkpoint_path

        if load_model is True:
            print("\nLoading DenseDepth model\n")
            self.load_model()
            print("\nFinished loading model\n")

        return None

    def load_model(self):

        self.custom_objects = {'BilinearUpSampling2D': layers.BilinearUpSampling2D, 'depth_loss_function': None}
        self.model = keras.models.load_model(self.checkpoint_path, custom_objects=self.custom_objects, compile=False)

        return None

    def predict(self, input):

        # Input can be an image path or image anarray
        assert type(input) == str or type(input) == np.ndarray

        # If input is an image path, make sure the image exist
        if type(input) == str:
            assert os.path.isfile(input)
            inputs = dd_utils.load_images(glob.glob(input))

        else: # Only supports uint8 to float32
            inputs = np.clip(input / 255, 0, 1).astype(np.float32)
        
        output = dd_utils.predict(self.model, inputs, minDepth=10, maxDepth=1000)[0]

        return output

