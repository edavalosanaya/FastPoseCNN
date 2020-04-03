# Library Imports
import os
import time
import sys
import glob

import cv2
import numpy as np

import tensorflow
import keras

import pickle

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
sys.path.append(SOURCE_CODE_DIR)
sys.path.append(NETWORKS_DIR)
sys.path.append(NOCS_ROOT_DIR)
sys.path.append(DENSE_TF_ROOT_DIR)

# Local Imports
from NOCS_CVPR2019 import model as modelib
from NOCS_CVPR2019 import utils as nocs_utils
from NOCS_CVPR2019 import config
from DenseDepth import layers
from DenseDepth import utils as dd_utils

import tools

#-------------------------------------------------------------------------------------
# Main Code

image_path = os.path.join(r"E:\MASTERS_STUFF\workspace\NOCS_CVPR2019\data\real_test\real_test\scene_1", "0000_color.png")
depth_path = os.path.join(r"E:\MASTERS_STUFF\workspace\NOCS_CVPR2019\data\real_test\real_test\scene_1", "0000_depth.png")
checkpoint_path = r"E:\MASTERS_STUFF\workspace\DenseDepth\logs\nyu.h5"

# Loading files in
image = cv2.imread(image_path)

# DenseDepth prediction
"""
dense_depth_net = tools.DenseDepth(checkpoint_path)
depth = dense_depth_net.predict(image_path)
depth = tools.dense_depth_to_nocs_depth(depth)
depth = tools.nocs_depth_formatting(depth)
cv2.imwrite(os.path.join(ROOT_DIR, "case_study_data", "generated_0000_depth.png"), depth)
"""

# NOCS prediction
"""
nocs_model = tools.NOCS()
result, r = nocs_model.predict_coord(image, image_path)
data = {"result": result, "r": r}

with open(os.path.join(ROOT_DIR, "case_study_data", "0000_color_nocs_result.pickle"), 'wb') as f:
    pickle.dump(data, f, pickle.HIGHEST_PROTOCOL)
"""

# Loading data
with open(r"E:\MASTERS_STUFF\workspace\case_study_data\real_test_scene_1\0000\0000_color_nocs_result.pickle",'rb') as f:
    data = pickle.load(f)

result, r = data['result'], data['r']

generated_depth = cv2.imread(r"E:\MASTERS_STUFF\workspace\case_study_data\real_test_scene_1\0000\generated_0000_depth.png", -1)
source_depth = cv2.imread(depth_path, -1)

# Making both depths have the same range (0-255)
print("\n", "*"*35, "\n", "Generated depth info")
tools.print_cv2_data_info(generated_depth)

print("\n", "*"*35, "\n", "Source depth info (before normalization)")
tools.print_cv2_data_info(source_depth)

print("\n", "*"*35, "\n", "Source depth info (after normalization)")
source_depth = cv2.normalize(src=source_depth, dst=None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_16U)
tools.print_cv2_data_info(source_depth)

# Modifing generated_depth histogram to match source
mod_generated_depth = tools.hist_matching(generated_depth, tools.calculate_cdf(source_depth))

# Applying both depths to the ICP algorithm
nocs_model = tools.NOCS()

gen_depth_result = nocs_model.icp_algorithm(image, image_path, generated_depth, result, r)
sou_depth_result = nocs_model.icp_algorithm(image, image_path, source_depth, result, r)
mod_gen_depth_result = nocs_model.icp_algorithm(image, image_path, mod_generated_depth, result, r)

# Visualizing output and input
visual_depth_1 = tools.making_depth_easier_to_see(source_depth)
visual_depth_2 = tools.making_depth_easier_to_see(generated_depth)
visual_depth_3 = tools.making_depth_easier_to_see(mod_generated_depth)

tools.visualize(["source depth", "RT (source depth)", "generated depth", "RT (generated depth)", "modified generated depth", "RT (modified generated depth)"],
                visual_depth_1, sou_depth_result, visual_depth_2, gen_depth_result, visual_depth_3, mod_gen_depth_result)

