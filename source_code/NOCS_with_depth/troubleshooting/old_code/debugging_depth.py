# Library Import

import os
import sys
import glob
import pathlib

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' # Basically, SHUT UP TENSORFLOW

import pickle
import json

import numpy as np
import cv2
import open3d as o3d
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

#-----------------------------------------------------------------------------
# Functions

def demo_npy_2_png():

    demo_depth = np.load(demo_depth_path)
    demo_rgb = np.load(demo_rgb_path)

    print("Demo RGB")
    tools.print_cv2_data_info(demo_rgb)

    print("Demo Depth")
    tools.print_cv2_data_info(demo_depth)

    # Modifications

    print('\n', 'MODIFICATIONS', '\n')
    demo_rgb = (demo_rgb * 255).round().astype(np.uint8)
    demo_depth = (demo_depth * 255).round().astype(np.uint8)

    # Outputs

    print("Demo RGB")
    tools.print_cv2_data_info(demo_rgb)

    print("Demo Depth")
    tools.print_cv2_data_info(demo_depth)

    cv2.imshow('demo depth', demo_depth)
    cv2.imshow('demo rgb', demo_rgb)

    cv2.imwrite(os.path.join(data_directory, 'demo_depth.png'), demo_depth)
    cv2.imwrite(os.path.join(data_directory, 'demo_rgb.png'), demo_rgb)

    cv2.waitKey(0)
    cv2.destroyAllWindows()

    return None

def gen_float32_2_uint():

    dense_depth_net = tools.DenseDepth(checkpoint_path)
    gen_depth_float32 = dense_depth_net.predict(rgb_path)

    np.save(os.path.join(data_directory, '0000_gen_depth_float32.npy'), gen_depth_float32)
    #gen_depth_float32 = np.load(os.path.join(data_directory, '0000_gen_depth_float32_3.npy'))

    #gen_depth_float32 = np.load(os.path.join(data_directory_2, 'scene_1_0000_gen_depth_float32.npy'))

    tools.print_cv2_data_info(gen_depth_float32)

    gen_depth_float32 = tools.scale_up(2, [gen_depth_float32])[0]

    gen_depth_uint8 = (gen_depth_float32 * 255).astype(np.uint8)
    gen_depth_uint16 = (gen_depth_float32 * 65535).astype(np.uint16)

    tools.print_cv2_data_info(gen_depth_uint8)
    tools.print_cv2_data_info(gen_depth_uint16)

    cv2.imwrite(os.path.join(data_directory, '0000_gen_depth_uint8.png'), gen_depth_uint8)
    cv2.imwrite(os.path.join(data_directory, '0000_gen_depth_uint16.png'), gen_depth_uint16)

    return None

def hist_match():

    depth_uint16 = cv2.imread(os.path.join(data_directory, '0000_depth.png'), cv2.IMREAD_UNCHANGED)
    depth_uint8 = ( depth_uint16 * (256/65536) ).astype(np.uint8)

    gen_depth_uint8 = cv2.imread(os.path.join(data_directory, '0000_gen_depth_uint8.png'), cv2.IMREAD_UNCHANGED)
    gen_depth_uint16 = cv2.imread(os.path.join(data_directory, '0000_gen_depth_uint16.png'), cv2.IMREAD_UNCHANGED)


    hist_match_gen_depth_uint8 = tools.hist_matching(gen_depth_uint8, depth_uint8)
    hist_match_gen_depth_uint16 = tools.hist_matching(gen_depth_uint16, depth_uint16)
    
    tools.print_cv2_data_info(hist_match_gen_depth_uint8)
    tools.print_cv2_data_info(hist_match_gen_depth_uint16)
    #tools.plot_histogram(hist_match_gen_depth)

    cv2.imshow("hist_match_gen_depth_uint8", hist_match_gen_depth_uint8)
    cv2.imshow("hist_match_gen_depth_uint16", hist_match_gen_depth_uint16)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    cv2.imwrite(os.path.join(data_directory, '0000_hist_match_gen_depth_uint8.png'), hist_match_gen_depth_uint8)
    cv2.imwrite(os.path.join(data_directory, '0000_hist_match_gen_depth_uint16.png'), hist_match_gen_depth_uint16)

    return None

def create_RT_image():
    
    with open(pkl_path, 'rb') as f:
        data = pickle.load(f)

    result, r = data['result'], data['r']
    rgb = cv2.imread(rgb_path)
    generated_depth = cv2.imread(os.path.join(data_directory, '0000_gen_depth_uint16.png'), cv2.IMREAD_UNCHANGED)
    source_depth = cv2.imread(src_depth_path, cv2.IMREAD_UNCHANGED)

    nocs_model = tools.NOCS(load_model=False)

    generated_depth_result = nocs_model.icp_algorithm(rgb, rgb_path, generated_depth, result, r)
    source_depth_result = nocs_model.icp_algorithm(rgb, rgb_path, source_depth, result, r)
    
    cv2.imshow('source', source_depth_result)
    cv2.imshow('generated', generated_depth_result)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    cv2.imwrite(os.path.join(data_directory,'0000_gen_depth_uint16_RT_2.png'), generated_depth_result)

    return None

#-----------------------------------------------------------------------------
# Main Code

# Data Paths
data_directory = r"E:\MASTERS_STUFF\workspace\case_study_data\debugging_depth"
data_directory_2 = r'E:\MASTERS_STUFF\workspace\case_study_data\output_logs_2\images'
checkpoint_path = r"E:\MASTERS_STUFF\workspace\networks\DenseDepth\logs\nyu.h5" # or kitti.h5


rgb_path = os.path.join(data_directory, '0000_color.png')
src_depth_path = os.path.join(data_directory, '0000_depth.png')
pkl_path = os.path.join(data_directory, '0000_color_nocs_result.pickle')
gen_depth_float32_path = os.path.join(data_directory, '0000_depth_gen_float32.npy')
demo_depth_path = os.path.join(data_directory, 'demo_depth.png')
demo_rgb_path = os.path.join(data_directory, 'demo_rgb.png')


# Beginning of Code

#gen_float32_2_uint()

hist_match()

#create_RT_image()

#tools.rgbd_visualize(os.path.join(data_directory, '0000_color.png'), os.path.join(data_directory, '0000_hist_match_gen_depth_uint8.png'))

