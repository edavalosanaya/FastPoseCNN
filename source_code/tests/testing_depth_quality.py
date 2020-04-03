# Library Import

import os
import sys
import glob

import pickle
import json

import numpy as np
import cv2
import open3d as o3d
from matplotlib import pyplot as plt

# Getting Root directory
ROOT_DIR = os.getcwd()
NETWORKS_DIR = os.path.join(ROOT_DIR, 'networks')

# Getting NOCS paths
NOCS_ROOT_DIR = os.path.join(NETWORKS_DIR, "NOCS_CVPR2019")
NOCS_MODEL_DIR = os.path.join(NOCS_ROOT_DIR, "logs")
NOCS_COCO_MODEL_PATH = os.path.join(NOCS_MODEL_DIR, "mask_rcnn_coco.h5")
NOCS_WEIGHTS_PATH = os.path.join(NOCS_MODEL_DIR, "nocs_rcnn_res50_bin32.h5")

# Getting DenseDepth paths
DENSE_ROOT_DIR = os.path.join(NETWORKS_DIR, "DenseDepth")
DENSE_TF_ROOT_DIR = os.path.join(DENSE_ROOT_DIR, "Tensorflow")

# Appending necessary paths
sys.path.append(NETWORKS_DIR)
sys.path.append(NOCS_ROOT_DIR)
sys.path.append(DENSE_TF_ROOT_DIR)

# Local Imports

import tools
#from NOCS_CVPR2019 import dataset as nocs_dataset
#from NOCS_CVPR2019 import utils as nocs_utils

#---------------------------------------------------------------------
# Functions

def depth_analysis(color_path, src_depth_path):

    # Checking Depth Size and Dtype every step of the way
    # Loading color image
    print('\n', '*' *  35)
    print('COLOR IMAGE')
    color_raw = cv2.imread(color_path)
    tools.print_cv2_data_info(color_raw)

    # Loading depth image
    print('\n', '*' *  35)
    print('SOURCE DEPTH IMAGE')
    src_depth = cv2.imread(src_depth_path, cv2.IMREAD_UNCHANGED)
    tools.print_cv2_data_info(src_depth)

    # Loading models before iteration 
    print("\nLoading models...\n")
    tools.disable_print()
    dense_depth_net = tools.DenseDepth(checkpoint_path)
    tools.enable_print()
    print("\nFinished loading models\n")

    # Generating depth
    print('\n', '*' *  35)
    print('ORIGINAL GENERATED DEPTH')
    gen_depth = dense_depth_net.predict(color_path)
    tools.print_cv2_data_info(gen_depth)

    # Cleaning depth
    print('\n', '*' *  35)
    print('MODIFIED GENERATED DEPTH')
    norm_gen_depth = tools.modify_image(gen_depth, size=src_depth.shape, dtype=src_depth.dtype)
    scale_gen_depth = tools.modify_image(gen_depth, size=src_depth.shape)
    
    scale_gen_depth_uint8 = (scale_gen_depth * 255).astype(np.uint8)
    scale_gen_depth_uint16 = (scale_gen_depth * 65535).astype(np.uint16)

    tools.print_cv2_data_info(norm_gen_depth)
    tools.print_cv2_data_info(scale_gen_depth_uint8)
    tools.print_cv2_data_info(scale_gen_depth_uint16)

    print("PLOT HISTOGRAMS")
    tools.plot_histogram([scale_gen_depth_uint8])

    print('\n', "#"*35)
    print("COMPUTED WITH GENERATED DEPTH MODIFICATIONS")
    errors = dense_depth_net.compute_errors(src_depth, scale_gen_depth_uint16)
    print(errors)

    # Displaying output
    cv2.imshow('color', color_raw)
    cv2.imshow('source depth', src_depth)
    cv2.imshow('generated depth', gen_depth)
    cv2.imshow('clean generated depth', scale_gen_depth_uint8)

    cv2.waitKey(0)
    cv2.destroyAllWindows()

    cv2.imwrite(os.path.join(output_dir, 'norm_depth.png'), norm_gen_depth)
    cv2.imwrite(os.path.join(output_dir, 'gen_depth_uint8.png'), scale_gen_depth_uint8)
    cv2.imwrite(os.path.join(output_dir, 'gen_depth_uint16.png'), scale_gen_depth_uint16)

    return None

def create_exp_depth():

    color_raw = cv2.imread(r"E:\MASTERS_STUFF\workspace\datasets\NOCS\real\test\scene_1\0000_depth.png")
    target_depth = cv2.imread(r"E:\MASTERS_STUFF\workspace\datasets\NOCS\real\test\scene_1\0000_depth.png", cv2.IMREAD_UNCHANGED)
    depth = cv2.imread(os.path.join(output_dir, 'kitti_clean_generated_depth.png'), cv2.IMREAD_UNCHANGED)
    
    target_depth = tools.match_image(target_depth, color_raw, size=False, dtype=True)
    depth = tools.match_image(depth, color_raw, size=False, dtype=True)

    tools.print_cv2_data_info(target_depth)
    tools.print_cv2_data_info(depth)

    new_depth = tools.hist_matching(depth, tools.calculate_cdf(target_depth))
    tools.print_cv2_data_info(new_depth)

    cv2.imshow('new_depth', new_depth)

    cv2.waitKey(0)
    cv2.destroyAllWindows()

    cv2.imwrite(os.path.join(output_dir, 'mod_kitti_clean_generated_depth_2.png'), new_depth)

    return None

def open3d_test(color_path, depth_path):

    tools.open3d_plot(color_path, depth_path)
    tools.pcd_visualization(color_path, depth_path)

    return None

#---------------------------------------------------------------------
# Main Code


# REAL DATASET
r"""
color_path = r"E:\MASTERS_STUFF\workspace\datasets\NOCS\real\test\scene_1\0000_color.png"
src_depth_path = r"E:\MASTERS_STUFF\workspace\datasets\NOCS\real\test\scene_1\0000_depth.png"
intrinsics = [[591.0125, 0, 322.525], [0, 590.16775, 244.11084], [0, 0, 1]] 
"""

# NYU v2 DATASET
color_path = r"E:\MASTERS_STUFF\workspace\case_study_data\open3d_tests\depth_quality\rgb_00045.jpg"
src_depth_path = r"E:\MASTERS_STUFF\workspace\case_study_data\open3d_tests\depth_quality\sync_depth_00045.png"
checkpoint_path = r"E:\MASTERS_STUFF\workspace\networks\DenseDepth\logs\nyu.h5" # or kitti.h5
output_dir = r"E:\MASTERS_STUFF\workspace\case_study_data\open3d_tests\depth_quality"

# Test Routines
#depth_analysis(color_path, src_depth_path)

open3d_test(color_path, os.path.join(output_dir, 'gen_depth_uint16_2_8.png'))
#img = cv2.imread(r'E:\MASTERS_STUFF\workspace\case_study_data\open3d_tests\depth_quality\gen_depth_uint16_2_8.png', cv2.IMREAD_UNCHANGED)
#img2 = cv2.imread(r'E:\MASTERS_STUFF\workspace\case_study_data\open3d_tests\depth_quality\src_depth_uint8.png', cv2.IMREAD_UNCHANGED)
#cv2.imshow('uint16 depth generated', img)
#tools.plot_histogram([img])
