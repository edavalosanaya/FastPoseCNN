import os
import sys
import pathlib

from easydict import EasyDict

import numpy as np

"""
Path Config
"""

cfg = EasyDict()

cfg.ROOT_DIR = pathlib.Path(os.path.dirname(os.path.abspath(__file__)))
cfg.SRC_DIR = cfg.ROOT_DIR.parent
cfg.BEDROCK_DIR = cfg.SRC_DIR.parent

cfg.DATASET_DIR = cfg.BEDROCK_DIR / 'datasets'
cfg.NETS_DIR = cfg.ROOT_DIR / 'model_lib'
cfg.SAVED_MODEL_DIR = cfg.NETS_DIR / 'saved_model_logs'
cfg.LOGS = cfg.NETS_DIR / 'logs'
cfg.TEST_OUTPUT = cfg.ROOT_DIR / 'tests_output'

"""
for key,value in cfg.items():
    if 'DIR' in key:
        sys.path.append(value)
"""

"""
Data constants
"""

constants = EasyDict()

#-------------------------------------------------------------------------------
# NOCS Dataset Constants

constants.INTRINSICS = np.array([[577.5, 0, 319.5], [0., 577.5, 239.5], [0., 0., 1.]]) # CAMERA intrinsics

constants.SYNSET_NAMES = ['BG', #0
                          'bottle', #1
                          'bowl', #2
                          'camera', #3
                          'can',  #4
                          'laptop',#5
                          'mug'#6
                          ]

constants.SYNSET_COLORS = [(0,0,0), # background (black)
                           (237,27,36), # bottle (red)
                           (247,143,30), # bowl (orange)
                           (254,242,0), # camera (yellow)
                           (1,168,96), # can (green)
                           (1,86,164), # laptop (blue)
                           (166,68,153) # mug (purple)
                           ]

constants.CLASS_MAP = {
    'bottle': 'bottle',
    'bowl':'bowl',
    'cup':'mug',
    'laptop': 'laptop',
}

#-------------------------------------------------------------------------------
# PASCAL VOC Dataset Constants

constants.VOC_COLORMAP = [[0, 0, 0], [128, 0, 0], [0, 128, 0], [128, 128, 0],
                          [0, 0, 128], [128, 0, 128], [0, 128, 128], [128, 128, 128],
                          [64, 0, 0], [192, 0, 0], [64, 128, 0], [192, 128, 0],
                          [64, 0, 128], [192, 0, 128], [64, 128, 128], [192, 128, 128],
                          [0, 64, 0], [128, 64, 0], [0, 192, 0], [128, 192, 0],
                          [0, 64, 128]]

constants.VOC_CLASSES = ['background', 'aeroplane', 'bicycle', 'bird', 'boat',
                         'bottle', 'bus', 'car', 'cat', 'chair', 'cow',
                         'diningtable', 'dog', 'horse', 'motorbike', 'person',
                         'potted plant', 'sheep', 'sofa', 'train', 'tv/monitor']
