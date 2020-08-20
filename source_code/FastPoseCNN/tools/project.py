import os
import sys
import pathlib

from easydict import EasyDict

import numpy as np

"""
Path Config
"""

cfg = EasyDict()

cfg.TOOL_DIR = pathlib.Path(os.path.abspath(__file__)).parent
cfg.ROOT_DIR = cfg.TOOL_DIR.parent
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
Data constantsants
"""

constants = EasyDict()

constants.INTRINSICS = np.array([[577.5, 0, 319.5], [0., 577.5, 239.5], [0., 0., 1.]]) # CAMERA intrinsics


constants.DATA_NAMES = ['bottle',
                        'bowl',
                        'camera',
                        'can',
                        'laptop',
                        'mug']

constants.SYNSET_NAMES = ['BG', #0
                          'bottle', #1
                          'bowl', #2
                          'camera', #3
                          'can',  #4
                          'laptop',#5
                          'mug'#6
                          ]

constants.CLASS_MAP = {
    'bottle': 'bottle',
    'bowl':'bowl',
    'cup':'mug',
    'laptop': 'laptop',
}