# Library Import
import os
import sys
import glob
import pathlib

from matplotlib import pyplot as plt
import numpy as np
import cv2

import tqdm 
import json
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

import tools
from NOCS_CVPR2019 import dataset as nocs_dataset
from NOCS_CVPR2019 import utils as nocs_utils

#---------------------------------------------------------------------
# Constants

gen_aps_path = r"E:\MASTERS_STUFF\workspace\case_study_data\output_logs\generated_aps"
sou_aps_path = r"E:\MASTERS_STUFF\workspace\case_study_data\output_logs\source_aps"
output_path = r"E:\MASTERS_STUFF\workspace\case_study_data\output_logs"
"""
# Getting the entire aps
with open(os.path.join(sou_aps_path, 'aps_source.pickle'), 'rb') as f:
    aps = pickle.load(f)

iou_3d_aps, pose_aps = aps

print(iou_3d_aps)
print("*"*35)
print(pose_aps)
"""

ap_dict = {}

for title, depth_type_path in {"source": sou_aps_path, "generated": gen_aps_path}.items():

    ap_dict[title] = {}

    # Getting the iou_dict
    with open(os.path.join(depth_type_path, 'IoU_3D_AP_0.0-1.0.pkl'), 'rb') as f:
        iou_dict = pickle.load(f)

    #print(iou_dict)

    iou_thres_list = iou_dict['thres_list']

    iou_thres_list = np.around(iou_thres_list, 2)

    _5_index = np.where(iou_thres_list == 0.5)[0][0]
    _25_index = np.where(iou_thres_list == 0.25)[0][0]

    for thresh_value, thresh_index in {"0.25": _25_index, "0.50": _5_index}.items():

        ap_dict[title][thresh_value] = {}
        mAP = 0

        for cls_id in range(1, len(tools.synset_names)):

            class_name = tools.synset_names[cls_id]

            ap_dict[title][thresh_value][class_name] = "{:.3f}".format(iou_dict['aps'][cls_id, thresh_index])
            mAP += iou_dict['aps'][cls_id, thresh_index]

        mAP = mAP / 6

        ap_dict[title][thresh_value]['mAP'] = '{:.3f}'.format(mAP)

print(ap_dict)

with open(os.path.join(output_path, "classes_AP_and_mAP.json"), 'w') as f:
    json.dump(ap_dict, f)

