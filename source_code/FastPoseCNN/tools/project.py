import os
import sys
import pathlib

import matplotlib.cm

from easydict import EasyDict

import numpy as np

"""
################################################################################
# Path Config
################################################################################

cfg = EasyDict()

cfg.ROOT_DIR = pathlib.Path(os.path.dirname(os.path.abspath(__file__))).parent
cfg.SRC_DIR = cfg.ROOT_DIR.parent
cfg.BEDROCK_DIR = cfg.SRC_DIR.parent

cfg.DATASET_DIR = cfg.BEDROCK_DIR / 'datasets'
cfg.NETS_DIR = cfg.ROOT_DIR / 'lib'
cfg.SAVED_MODEL_DIR = cfg.NETS_DIR / 'saved_model_logs'
cfg.LOGS = cfg.NETS_DIR / 'logs'
cfg.TEST_OUTPUT = cfg.ROOT_DIR / 'tests_output'

# Specific datasets
# NOCS
cfg.CAMERA_TRAIN_DATASET = cfg.DATASET_DIR / 'NOCS' / 'camera' / 'train'
cfg.CAMERA_VALID_DATASET = cfg.DATASET_DIR / 'NOCS' / 'camera' / 'val'

# VOC
cfg.VOC_DATASET = cfg.DATASET_DIR / 'VOC2012'

# CAMVID
cfg.CAMVID_DATASET = cfg.DATASET_DIR / 'CAMVID'

# CARVANA
cfg.CARVANA_DATASET = cfg.DATASET_DIR / 'CARVANA'
"""

################################################################################
# Data constants
################################################################################

constants = EasyDict()

#-------------------------------------------------------------------------------
# Functions regarding the constants

def generate_colormap(num_classes, cmap=matplotlib.cm.get_cmap('hsv'), bg_index=0):

    colormap = np.zeros((num_classes, 3))

    for x in range(num_classes):

        fraction = x/num_classes
        rgb = (np.array(cmap(fraction)[:3]))

        if x == bg_index:
            rgb = np.array([0,0,0])
        
        colormap[x] = rgb

    return colormap

#-------------------------------------------------------------------------------
# All dataset variables

constants.NUM_CLASSES = {}
constants.INTRINSICS = {}

#-------------------------------------------------------------------------------
# NOCS Dataset Constants

constants.INTRINSICS['NOCS'] = np.array([[577.5, 0, 319.5], [0., 577.5, 239.5], [0., 0., 1.]]) # CAMERA intrinsics

constants.NOCS_CLASSES = ['bg', #0
                          'bottle', #1
                          'bowl', #2
                          'camera', #3
                          'can',  #4
                          'laptop',#5
                          'mug'#6
                          ]

constants.NOCS_COLORMAP = generate_colormap(len(constants.NOCS_CLASSES))

constants.CLASS_MAP = {
    'bottle': 'bottle',
    'bowl':'bowl',
    'cup':'mug',
    'laptop': 'laptop',
}

constants.NUM_CLASSES['NOCS'] = len(constants.NOCS_CLASSES)

#-------------------------------------------------------------------------------
# PASCAL VOC Dataset Constants

#"""
constants.VOC_COLORMAP = [[0, 0, 0], [128, 0, 0], [0, 128, 0], [128, 128, 0],
                          [0, 0, 128], [128, 0, 128], [0, 128, 128], [128, 128, 128],
                          [64, 0, 0], [192, 0, 0], [64, 128, 0], [192, 128, 0],
                          [64, 0, 128], [192, 0, 128], [64, 128, 128], [192, 128, 128],
                          [0, 64, 0], [128, 64, 0], [0, 192, 0], [128, 192, 0],
                          [0, 64, 128]]
#"""

constants.VOC_CLASSES = ['background', 'aeroplane', 'bicycle', 'bird', 'boat',
                         'bottle', 'bus', 'car', 'cat', 'chair', 'cow',
                         'diningtable', 'dog', 'horse', 'motorbike', 'person',
                         'potted plant', 'sheep', 'sofa', 'train', 'tv/monitor']

#constants.VOC_COLORMAP = generate_colormap(len(constants.VOC_CLASSES))

constants.NUM_CLASSES['VOC'] = len(constants.VOC_CLASSES)

#-------------------------------------------------------------------------------
# CAMVID Dataset Constants

constants.CAMVID_CLASSES = ['sky', 'building', 'pole', 'road', 'pavement', 
                            'tree', 'signsymbol', 'fence', 'car', 
                            'pedestrian', 'bicyclist', 'unlabelled']

constants.CAMVID_COLORMAP = generate_colormap(len(constants.CAMVID_CLASSES))

constants.NUM_CLASSES['CAMVID'] = len(constants.CAMVID_CLASSES)
