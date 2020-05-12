# Imports
import os
import sys
import pathlib
import random
import PIL.Image
import tqdm

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
from DenseDepth import augment as dd_augment

#-------------------------------------------------------------------
# Classes

class BasicRGBSequence(keras.utils.Sequence):

    def __init__(self, dataset_filepaths, batch_size):

        # Placing the dataset's information into the Sequence class
        self.dataset_filepaths = dataset_filepaths
        self.batch_size = batch_size
        self.N = len(self.dataset_filepaths)
        
        # set information
        self.max_depth = 1000.0
        self.rgb_shape = [480, 640, 3]
        self.depth_shape = [240, 320, 1]

        return None

    def __len__(self):
        return int(np.ceil(self.N / self.batch_size))

    def __getitem__(self, idx):

        # batching handling
        batch_color = np.zeros( [self.batch_size] + self.rgb_shape )
        batch_depth = np.zeros( [self.batch_size] + self.depth_shape )

        # for image in the batch, get the information of the image
        for i in range(self.batch_size):
            
            # Catching possible KeyError, accessing past the available images
            index = min((idx * self.batch_size) + 1, self.N - 1)

            # Getting the color and depth images
            image_pair = self.dataset_filepaths[index]
            color_image = np.clip(np.asarray(PIL.Image.open(image_pair[0])).reshape(480,640,3)/255,0,1)
            depth_image = skimage.transform.resize(np.asarray(PIL.Image.open(image_pair[1]), dtype=np.float32), (240, 320, 1)).copy().astype(float) / 10.0
            #depth_image = self.max_depth / depth_image # depth normalization

            batch_color[i] = color_image
            batch_depth[i] = depth_image

        return batch_color, batch_depth

class BasicAugmentRGBSequence(BasicRGBSequence):

    def __init__(self, dataset_filepaths, batch_size, is_flip=False, is_addnoise=False, is_erase=False):
        # Using parent's __init__ + additional special conditions
        super().__init__(dataset_filepaths, batch_size)

        # Additional special conditions
        self.dataset_filepaths = sklearn.utils.shuffle(self.dataset_filepaths, random_state=0)
        self.policy = dd_augment.BasicPolicy(color_change_ratio=0.50, mirror_ratio=0.50, flip_ratio=0.0 if not is_flip else 0.2, add_noise_peak=0 if not is_addnoise else 20, erase_ratio=-1.0 if not is_erase else 0.5)

        return None

    def __getitem__(self, idx, is_apply_policy=True):

        # Getting the non-augmented batch
        batch_color, batch_depth = super().__getitem__(idx)

        # If no augmentation is asked, do not augment and return simple images
        if is_apply_policy is False:
            return batch_color, batch_depth

        # For image in the batch, apply the augmentation
        for i in range(self.batch_size):

            # Applying augment policy
            batch_color[i], batch_depth[i] = self.policy(batch_color[i], batch_depth[i])           

        return batch_color, batch_depth

#-------------------------------------------------------------------
# Functions

def get_training_data(dataset_path, batch_size=1):

    # Making sure that dataset_path is a pathlib.Path object
    if isinstance(dataset_path, str):
        dataset_path = pathlib.Path(dataset_path)

    # Load filepaths for train and test datasets/generators
    print("Obtaining file pathnames ...")
    train_filepath_list, test_filepath_list = get_dataset_image_filepaths(dataset_path, False)
    print("Train database size:", len(train_filepath_list), "Test database size:", len(test_filepath_list))

    # Create the generators
    print("Creating database generators")
    train_generator = BasicAugmentRGBSequence(train_filepath_list, batch_size)
    test_generator = BasicRGBSequence(test_filepath_list, batch_size)

    return train_generator, test_generator

def get_dataset_image_filepaths(dataset_path, automatic_test_train_split=False, train_percentage=0.5):

    # Output variables
    test_filepath_list = [] # list of strings
    train_filepath_list = [] # list of strings

    # if a test/train split has been implemented in the dataset
    if automatic_test_train_split is False: 
        test_dataset_path = dataset_path / 'test'
        train_dataset_path = dataset_path / 'train'

        # if split data exist, use it
        if train_dataset_path.exists() and test_dataset_path.exists():
            print("Using pre-existing test/train split ...")
            test_filepath_list = get_image_paths_in_dir(test_dataset_path)
            train_filepath_list = get_image_paths_in_dir(train_dataset_path)

    else:
        # else, we have to load the entire dataset
        print("Using automatic test/train split ...")
        dataset_filepath_list = get_image_paths_in_dir(dataset_path)
        train_filepath_list, test_filepath_list = split_dataset(dataset_filepath_list, train_percentage)    

    return train_filepath_list, test_filepath_list

def get_image_paths_in_dir(dir_path):

    # Output
    total_filepath_list = [] # [[color, depth], ...]
    
    # handling additional directories
    eval_paths = [dir_path]
    i = 0

    while True:

        # Break condition: if no more directories to evaluate, end
        if len(eval_paths) <= i:
            break

        # Selecting the new path to evaluate
        eval_path = eval_paths[i]
        i += 1

        # for all the evaluate paths, do the following
        files = [x for x in eval_path.iterdir() if x.is_file()]

        # Coupling corresponding color and depth images
        color_images = [x for x in files if x.name.find('color') != -1 and x.name.endswith('.png')]

        # Finding color's matching depth image
        for color_image in color_images:
            depth_image = color_image.parent / color_image.name.replace("color","depth")
            
            # if the expected depth image exist, save the couple to the dataset
            if depth_image.exists():
                total_filepath_list.append([str(color_image),str(depth_image)])

        directories = [x for x in eval_path.iterdir() if x.is_dir()]
        eval_paths += directories

    return total_filepath_list

def split_dataset(dataset_filepath_list, train_percentage=0.5):

    # Obtain the quantity of files for a train_percentage split
    train_file_quantity = int(len(dataset) * train_percentage)

    # Shuffle the dataset_filepath_list
    for i in range(2):
        random.shuffle(dataset_filepath_list)

    # Split the test and training data
    train_filepath_list, test_filepath_list = dataset_filepath_list[:train_filepath_list], dataset_filepath_list[train_filepath_list]

    return train_filepath_list, test_filepath_list

#------------------------------------------------------------------------
# Main Code

if __name__ == "__main__":
    # This section is for pure troubleshooting purposes

    dataset_path = ROOT_DIR / 'datasets' / 'NOCS' / 'real'

    # Testing get_training_data
    print("\n\nGetting training data ...")
    train_generator, test_generator = get_training_data(dataset_path, batch_size=2)
    print("Finished getting training data\n\n")

    # Testing the output generators
    print(train_generator[0])