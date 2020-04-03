# Library Imports
import os
import sys
import pathlib

import numpy as np
import cv2
from matplotlib import pyplot as plt

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

import tools

#-------------------------------------------------------------------------------------
# Main Code

source_depth = cv2.imread(r"E:\MASTERS_STUFF\workspace\case_study_data\real_test_scene_1\0000\0000_depth.png",-1)
generated_depth = cv2.imread(r"E:\MASTERS_STUFF\workspace\case_study_data\real_test_scene_1\0000\generated_0000_depth.png",-1)

generated_depth = tools.hist_matching(generated_depth, tools.calculate_cdf(source_depth))

visual_depth = tools.making_depth_easier_to_see(generated_depth)
tools.visualize(["Histogram Modified Generated Depth"], visual_depth)

depth_dict = {"source": {"image": source_depth}, "generated": {"image": generated_depth}}

counter = 0
fig, ax = plt.subplots(1, 2, sharey=True, tight_layout=False)
fig.suptitle("Depth Comparision Histograms")

for key, value in depth_dict.items():

    print("\n", "*" * 35)
    print("{} depth analysis".format(key))

    # Normalizing to range (0-255) to make fair comparison
    depth_dict[key]["image"] = cv2.normalize(src=depth_dict[key]["image"], dst=None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_16U)

    # Printing image's information
    tools.print_cv2_data_info(depth_dict[key]["image"])

    # Creating histogram
    #hist = cv2.calcHist([depth_dict[key]["image"]], [0], None, [256], [0,256])

    ax[counter].hist(depth_dict[key]['image'].ravel(), 256, [0,256])
    ax[counter].set_title("{}".format(key))

    counter += 1
    
plt.show()
