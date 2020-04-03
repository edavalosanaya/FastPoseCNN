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

#-------------------------------------------------------------------------------------
# Functions

#### General Open3D Functions

def pcd_visualization(color_path, depth_path, intrinsics = False):

    transformation_matrix = [[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]]

    if intrinsics == False:
        intrinsics = o3d.camera.PinholeCameraIntrinsic(o3d.camera.PinholeCameraIntrinsicParameters.PrimeSenseDefault)

    # Visualizing src depth and color
    color_raw = o3d.io.read_image(color_path)
    src_depth_raw = o3d.io.read_image(depth_path)

    rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(color_raw, src_depth_raw)
    pcd = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd_image, intrinsics)
    pcd.transform(transformation_matrix)

    o3d.visualization.draw_geometries([pcd])

    return None

def open3d_plot(color_path, depth_path):

    # Visualizing src depth and color
    color_raw = o3d.io.read_image(color_path)
    depth_raw = o3d.io.read_image(depth_path)
    rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(color_raw, depth_raw)

    plt.subplot(1, 2, 1)
    plt.title('Grayscale Image')
    plt.imshow(rgbd_image.color)
    plt.subplot(1, 2, 2)
    plt.title('Depth Image')
    plt.imshow(rgbd_image.depth)
    plt.show()

    return None

def rgbd_visualize(rgb_path, depth_path):

    open3d_plot(rgb_path, depth_path)
    pcd_visualization(rgb_path, depth_path)

    return None

# Network Specific Functions

def dd_compute_errors(gt, pred):
    
    e = dd_utils.compute_errors(gt, pred)
    print("{:>10}, {:>10}, {:>10}, {:>10}, {:>10}, {:>10}".format('a1', 'a2', 'a3', 'rel', 'rms', 'log_10'))
    print("{:10.4f}, {:10.4f}, {:10.4f}, {:10.4f}, {:10.4f}, {:10.4f}".format(e[0],e[1],e[2],e[3],e[4],e[5]))

    return None

# CV2 Functions

def print_img_info(img): # Could be an image (RGB) or depth

    print("cv2 data information:")
    print("Shape: {}".format(img.shape))
    print("Type: {}".format(img.dtype))

    try:
        #min_val, max_val, _, _ = cv2.minMaxLoc(img) # Causes errors sometimes
        
        numpy_data = img.ravel()
        min_val = np.amin(numpy_data)
        max_val = np.amax(numpy_data)
        
        print("Min: {} Max: {}".format(min_val, max_val))
    except:
        pass

    return None

def img_show(titles_list, img_list):

    for i, image in enumerate(img_list):

        if titles_list is None:
            title = str(i)
        else:
            title = titles_list[i]
        
        cv2.imshow("{}".format(title), image)

    cv2.waitKey(0)
    cv2.destroyAllWindows()

    return None

#### Histogram Functions

def plot_histogram(images):

    if type(images) is not list and type(images) is np.ndarray: # Single image
        images = list(images)
    elif type(images) is list: # Multiple images
        pass
    else:
        raise RuntimeError("Input invalid")

    print("Plotting Histogram")

    for image in images:
        _, dtype_range = dtype_numpy2cv2(image.dtype)
        max_val = dtype_range[1]
        
        if max_val >= 500:
            n_bins = 500
        else:
            n_bins = max_val

        plt.hist(image.ravel(), n_bins, [0,max_val+1])

    plt.show()     

    return 

def draw_hist_image(hist, hist_height, hist_width, nbins):

    # Normalize histogram
    cv2.normalize(hist, hist, hist_height, cv2.NORM_MINMAX)

    bin_width = int(hist_width/nbins)

    h = np.zeros((hist_height, hist_width))
    bins = np.arange(nbins, dtype=np.int32).reshape(nbins,1)

    for x, y in enumerate(hist):

        cv2.rectangle(h, (x*bin_width,y * 2), (x*bin_width + bin_width-1, hist_height), (255), -1)

    h = np.flipud(h)

    return h

#### Miscellanious Functions

def disable_print():
    sys.stdout = open(os.devnull, 'w')

def enable_print():
    sys.stdout = sys.__stdout__

def deco_silence(func):

    def wrapper(func):

        disable_print()
        func()
        enable_print()

        return None
    
    return wrapper(func)