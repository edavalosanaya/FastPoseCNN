# Library Imports
import os
import time
import sys
import pathlib

import skimage
import imutils
import cv2
import numpy as np
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

# Appending necessary paths
sys.path.append(str(NETWORKS_DIR))
sys.path.append(str(NOCS_ROOT_DIR))
sys.path.append(str(DENSE_TF_ROOT_DIR))

# Local Imports

from DenseDepth import utils as dd_utils
import constants

#-------------------------------------------------------------------------------------
# Functions

def scale_up(scale, image):

    output_shape = (scale * image.shape[0], scale * image.shape[1])
    output_image = imutils.resize(image, width=output_shape[1])

    return output_image

def img_change_dtype(image, new_dtype):

    info_new_dtypes = np.iinfo(new_dtype)

    try:
        info_old_dtypes = np.iinfo(images.dtype)
        old_mx = info_old_dtypes.max
        old_mn = info_old_dtypes.min
    except:
        print("old dtype not valid, assuming range from 0 - 1")
        old_mx = 1
        old_mn = 0

    new_mx = info_new_dtypes.max
    new_mn = info_new_dtypes.min

    new_image = np.clip(image * (new_mx / old_mx), new_mn, new_mx).astype(new_dtype)

    return new_image

def dd_to_nocs_depth(dd_depth):

    dd_depth_uint16 = img_change_dtype(dd_depth, np.uint16)
    dd_depth_uint16_resized = scale_up(2, dd_depth_uint16)
    dd_depth_uint16_resized_factored = (dd_depth_uint16_resized / 14.037758827209473).astype(np.uint16)

    return dd_depth_uint16_resized_factored

def nocs_depth_formatting(depth):

    if len(depth.shape) == 3:
        # This is encoded depth image, let's convert
        formatted_depth = np.uint16(depth[:, :, 1]*256) + np.uint16(depth[:, :, 2]) # NOTE: RGB is actually BGR in opencv
        formatted_depth = depth16.astype(np.uint16)
    elif len(depth.shape) == 2 and depth.dtype == 'uint16':
        formatted_depth = depth
    else:
        assert False, '[ Error ]: Unsupported depth type.'

    return formatted_depth

def calculate_cdf(image):

    print("image dtype: {}".format(image.dtype))

    info_dtype = np.iinfo(image.dtype)
    max_val = info_dtype.max

    cdf, b = skimage.exposure.cumulative_distribution(image)
    cdf = np.insert(cdf, 0, [0]*b[0])
    cdf = np.append(cdf, [1]*(max_val-b[-1]))

    return cdf

def hist_matching(image, target_image):

    assert image.dtype == target_image.dtype
    
    info_dtype = np.iinfo(image.dtype)
    max_val = info_dtype.max

    pixels = np.arange(max_val + 1)
    image_cdf = calculate_cdf(image)
    target_cdf = calculate_cdf(target_image)

    new_pixels = np.interp(image_cdf, target_cdf, pixels)
    new_image = (np.reshape(new_pixels[image.ravel()], image.shape)).astype(image.dtype)

    return new_image
