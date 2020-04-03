# Library Import

import os
import sys
import glob
import pathlib
import time
import statistics

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
sys.path.append(str(SOURCE_CODE_DIR))
sys.path.append(str(NETWORKS_DIR))
sys.path.append(str(NOCS_ROOT_DIR))
sys.path.append(str(DENSE_TF_ROOT_DIR))

# Local Imports

import tools

#-----------------------------------------------------------------------
# File-Specific Constants

data_dir = ROOT_DIR / 'case_study_data' / 'small_tests' / "debug_depth" / 'real_dataset'
output_dir = data_dir / 'generated_images'

rgb_path = data_dir / 'color.png'
depth_path = data_dir / 'depth.png'

rgb = cv2.imread(str(rgb_path))
depth = cv2.imread(str(depth_path), cv2.IMREAD_UNCHANGED)

#-----------------------------------------------------------------------
# Functions

# Creating the right depth

def gen_depth():

    # Generate original depth

    dense_depth_net = tools.models.DenseDepth()
    gen_depth_float32 = dense_depth_net.predict(rgb)

    np.save(str(output_dir / 'gen_depth_float32.npy'), gen_depth_float32)

    return None

def convert_float32_depth():

    # Convert original depth to different types
    gen_depth_float32 = np.load(str(output_dir / 'gen_depth_float32.npy'))

    new_depth = {"uint8": None, "uint16": None}

    # Changed dtype
    for key in new_depth.keys():
        new_depth[key] = tools.img_aug.img_change_dtype(gen_depth_float32, key)
        
    #tools.visualize.img_show(list(new_depth.keys()), list(new_depth.values()))

    # Saving image
    for key in new_depth.keys():
        cv2.imwrite(str(output_dir / 'gen_depth_{}.png'.format(key)), new_depth[key])

    return None

def scale_up_uint16():

    new_depth = {'uint8': None, 'uint16': None}

    # Reading image
    for key in new_depth.keys():
        new_depth[key] = cv2.imread(str(output_dir / 'gen_depth_{}.png'.format(key)), cv2.IMREAD_UNCHANGED)
        new_depth[key] = tools.img_aug.scale_up(2, new_depth[key])
        cv2.imwrite(str(output_dir / 'gen_depth_{}_s2.png'.format(key)), new_depth[key])

    return None

# Checking if it is the right depth

def get_depth_data():

    #new_depth = {'uint8': None, 'uint16': None}
    new_depth = {'uint16': None}

    # Reading image
    for key in new_depth.keys():
        new_depth[key] = cv2.imread(str(output_dir / 'gen_depth_{}_s2_factored.png'.format(key)), cv2.IMREAD_UNCHANGED)
        tools.visualize.print_img_info(new_depth[key])

    src_depth = cv2.imread(str(data_dir / 'depth.png'), cv2.IMREAD_UNCHANGED)
    tools.visualize.print_img_info(src_depth)

    tools.visualize.img_show(list(new_depth.keys()) + ['src'], list(new_depth.values()) + [src_depth])

    tools.visualize.dd_compute_errors(src_depth, new_depth['uint16'])

    return None

def click_on_depth():

    def click(event, x, y, flags, param):

        if event == cv2.EVENT_LBUTTONDOWN:
            print("GEN DEPTH ({},{}): {}\t SRC DEPTH ({},{}): {}".format(x,y,gen_depth[y,x],x,y,src_depth[y,x]))

    gen_depth = cv2.imread(str(output_dir / 'gen_depth_uint16_s2.png'), cv2.IMREAD_UNCHANGED)
    src_depth = cv2.imread(str(data_dir / 'depth.png'), cv2.IMREAD_UNCHANGED)

    tools.visualize.print_img_info(gen_depth)
    tools.visualize.print_img_info(src_depth)

    cv2.imshow("gen_depth", gen_depth)
    cv2.setMouseCallback("gen_depth", click)

    while True:

        time.sleep(0.1)
        key = cv2.waitKey(1) & 0xFF

        if key == ord("q"):
            break

    return None

def determine_relative_scale():

    text_files = os.listdir(str(data_dir / 'depth_value_comparison'))

    for text_file in text_files:
        file = open(str(data_dir / 'depth_value_comparison' / text_file), 'r')

        print(text_file)

        ratios = []

        lines = file.readlines()

        for line in lines:
            gen_data, src_data = line.split("SRC")

            gen_data = int(gen_data.split(":")[1].strip())
            src_data = int(src_data.split(":")[1].strip())
            ratio = round(gen_data/src_data, 2)
            ratios.append(ratio)

            print("gen: {} src: {} ratio: {}".format(gen_data, src_data, ratio))

        # Overal File Analysis
        print("MAX: {} \tMIN: {} \tMEAN: {:.2f}\tMEDIAN: {}".format(max(ratios), min(ratios), statistics.mean(ratios), statistics.median(ratios)))

def difference_visualize():

    gen_depth = cv2.imread(str(output_dir / 'gen_depth_uint16_s2.png'), cv2.IMREAD_UNCHANGED)
    src_depth = cv2.imread(str(data_dir / 'depth.png'), cv2.IMREAD_UNCHANGED)

    # Changing all zeros in src_depth to 1
    src_depth[src_depth == 0] = np.median(src_depth[src_depth > 0])

    division = (gen_depth / src_depth).astype(np.float32)

    median = np.median(division[division > 0])
    maximum = np.max(division[division > 0])
    minimum = np.min(division[division > 0])
    mean = np.mean(division[division > 0])
    print("Median: {} Maximum: {} Minimum: {} Mean: {}".format(median, maximum, minimum, mean))

    #print(division)

    tools.visualize.img_show(['division'], [(division * 5).astype(np.uint8)])

    print("\nNOW MODIFYING GEN DATA\n")

    # Now modifying generated depth to make it match better
    gen_depth = (gen_depth / mean).astype(np.uint16)

    division = (gen_depth / src_depth).astype(np.float32)

    median = np.median(division[division > 0])
    maximum = np.max(division[division > 0])
    minimum = np.min(division[division > 0])
    mean = np.mean(division[division > 0])
    print("Median: {} Maximum: {} Minimum: {} Mean: {}".format(median, maximum, minimum, mean))

    #print(division)

    tools.visualize.img_show(['division'], [(division * 5).astype(np.uint8)])

    cv2.imwrite(str(output_dir / 'gen_depth_uint16_s2_factored.png'), gen_depth)

    # The best scale is the average: 14.037758827209473

    return None

#-----------------------------------------------------------------------
# Main Code

#get_depth_data()
#click_on_depth()
#determine_relative_scale()
#difference_visualize()

#tools.visualize.rgbd_visualize(str(data_dir / 'color.png'), str(output_dir / 'gen_depth_uint16_s2_factored.png'))
#tools.visualize.rgbd_visualize(str(data_dir / 'color.png'), str(data_dir / 'depth.png'))
