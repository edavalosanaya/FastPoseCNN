import os
import sys
import pathlib
import warnings

from skimage import color
warnings.filterwarnings('ignore')

import pdb

import cv2
import numpy as np
import PIL
import io
import random
import skimage.io
import scipy.ndimage

import torch
import torchvision

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.cm

# Local Imports
root = next(path for path in pathlib.Path(os.path.abspath(__file__)).parents if path.name == 'FastPoseCNN')
sys.path.append(str(pathlib.Path(__file__).parent))

import project as pj
import draw as dr
import data_manipulation as dm

#-------------------------------------------------------------------------------
# File's Constants

#-------------------------------------------------------------------------------
# Class

#-------------------------------------------------------------------------------
# Visualization of data

def get_visualized_mask(mask, colormap):

    colorized_mask = np.zeros((3,mask.shape[-2], mask.shape[-1]))

    for class_id, class_color in enumerate(colormap):
        for i in range(3):

            X = np.array(class_color[i])
            Y = np.zeros((1,))
            
            if len(mask.shape) == 2:
                colorized_mask[i,:,:] += np.where(mask == class_id, X, Y)

    return colorized_mask

def get_visualized_masks(masks, colormap):

    # If HW data format
    if len(masks.shape) == len('HW'):

        colorized_masks = get_visualized_mask(masks, colormap)

    # If CHW data format
    elif len(masks.shape) == len('CHW'):

        colorized_masks = np.zeros((masks.shape[0],3,masks.shape[-2],masks.shape[-1]))

        for id, mask in enumerate(masks):
            colorized_masks[id,:,:,:] = get_visualized_mask(mask, colormap)

    else:
        raise RuntimeError('Wrong mask dataformat')

    return colorized_masks

def get_visualized_unit_vector(mask, unit_vector, colormap='hsv'):

    # Determing the angle of the unit vectors: f: R^2 -> R^1
    angle = np.arctan2(unit_vector[:,:,0], unit_vector[:,:,1])

    # Create norm function to shift data to [0:1]
    norm = matplotlib.colors.Normalize(vmin=np.min(angle), vmax=np.max(angle))

    # Obtain the colormap of choice
    my_colormap = matplotlib.cm.get_cmap(colormap)
    
    # Normalize data, apply the colormap, make it bytes (np.array), and remove the alpha channel
    colorized_angle = my_colormap(norm(angle), bytes=True)[:,:,:3] # removing alpha channel

    # Removing background
    colorized_angle = np.where(np.expand_dims(mask, axis=-1) == 0, 0, colorized_angle)

    return colorized_angle

def get_visualized_simple_center_2d(center_2d):

    # Create a holder of the data
    norm_center_2d = np.zeros((center_2d.shape[0], center_2d.shape[1], 3))

    # Convert the integer data into float [0,1]
    norm_center_2d[:,:,0] = center_2d[:,:,0] # Y (Red)
    norm_center_2d[:,:,2] = center_2d[:,:,1] # X (Blue)

    return norm_center_2d

def get_visualized_quaternion(quaternion):

    # Selecting the i,j, and k components
    ijk_component = quaternion[:,:,1:]

    # creating norm function
    norm = matplotlib.colors.Normalize(vmin=-1, vmax=1)

    # Normalize data, apply the colormap, make it bytes (np.array), and remove the alpha channel
    colorized_quat = norm(ijk_component)

    # Remove background
    black = np.zeros_like(colorized_quat)
    colorized_quat = np.where(ijk_component == [0,0,0], black, colorized_quat)

    return colorized_quat

#-------------------------------------------------------------------------------
# General Matplotlib Functions

def make_summary_figure(**images):

    # Initializing the figure and axs
    fig = plt.figure(figsize=(16,5))
    nr = len(images)
    nc = images[list(images.keys())[0]].shape[0]

    for i, (name, image) in enumerate(images.items()):
        if len(image.shape) >= 3: # NHW or NCHW
            for j, img in enumerate(image):

                plt.subplot(nr, nc, 1 + j + nc*i)
                plt.xticks([])
                plt.yticks([])
                if j == 0:
                    plt.ylabel(' '.join(name.split('_')).title())

                """
                if len(img.shape) == 3: # CHW to HWC
                    img = np.moveaxis(img, 0, -1)
                """
                img = dm.standardize_image(img)

                plt.imshow(img)
        else: # HW only

            plt.subplot(nr, nc, i + 1)
            plt.xticks([])
            plt.yticks([])
            plt.title(' '.join(name.split('_')).title())
            plt.imshow(image)

    # Overall figure configurations

    return fig    

def debug_show(**images):

    num_rows = num_columns = round(len(list(images.keys()))/2)

    if num_rows == 0:
        num_rows = num_columns = 1

    fig = plt.figure(figsize=(3*num_rows,3*num_columns))

    for i, (name, image) in enumerate(images.items()):

        plt.subplot(num_rows, num_columns, i+1)
        #plt.xticks([])
        #plt.yticks([])
        plt.xlabel(' '.join(name.split('_')).title())

        image = dm.standardize_image(image)
        plt.imshow(image)

    plt.show()

#-------------------------------------------------------------------------------
# Batch Ground Truth and Prediction Visualization

def compare_mask_performance(sample, pred_mask, colormap):

    # Selecting clean image and mask if available
    image_key = 'clean image' if 'clean image' in sample.keys() else 'image'
    mask_key = 'clean mask' if 'clean mask' in sample.keys() else 'mask'
    
    # Converting visual images into np.uint8 for matplotlib compatibility
    image_vis = sample[image_key].astype(np.uint8)
    gt_mask = sample[mask_key].astype(np.uint8)

    # Target (ground truth) data format 
    if len(gt_mask.shape) == len('BCHW'):

        if pred_mask.shape[1] == 1: # Binary segmentation
            pred_mask = pred_mask[:,0,:,:]
            gt_mask = gt_mask[:,0,:,:]

        else: # Multi-class segmentation
            pred_mask = np.argmax(pred_mask, axis=1)
            gt_mask = np.argmax(gt_mask, axis=1)

    elif len(gt_mask.shape) == len('BHW'):

        if pred_mask.shape[1] == 1: # Binary segmentation
            pred_mask = pred_mask[:,0,:,:]

        else: # Multi-class segmentation
            pred_mask = np.argmax(pred_mask, axis=1)

    # Colorized the binary masks
    gt_mask_vis = get_visualized_masks(gt_mask, colormap)
    pred_mask = get_visualized_masks(pred_mask, colormap)

    # Creating a matplotlib figure illustrating the inputs vs outputs
    summary_fig = make_summary_figure(
        image=image_vis,
        ground_truth_mask=gt_mask_vis,
        predicited_mask=pred_mask)

    return summary_fig

def compare_pose_performance(sample, pred_quaternion):

    # Selecting clean image and mask if available
    image_key = 'clean image' if 'clean image' in sample.keys() else 'image'
    mask_key = 'clean mask' if 'clean mask' in sample.keys() else 'mask'
    depth_key = 'clean depth' if 'clean depth' in sample.keys() else 'depth'

    # Getting the image, mask, and depth
    image, mask, depth = sample[image_key], sample[mask_key], sample[depth_key]

    # Creating the translation vector
    modified_intrinsics = pj.constants.INTRINSICS.copy()
    modified_intrinsics[0,2] = sample['image'].shape[1] / 2
    modified_intrinsics[1,2] = sample['image'].shape[0] / 2

    # Create the drawn poses
    gt_poses = []
    pred_poses = []

    for batch_id in range(image.shape[0]):
        
        #"""
        # Obtain the centroids (x,y)
        centroids = dm.get_masks_centroids(sample['mask'][batch_id])

        # If no centroids are found, just skip
        if not centroids:
            continue
        
        # Obtain the depth located at the centroid (z)
        zs = dm.get_data_from_centroids(centroids, sample['depth'][batch_id]) * 100000
        
        # Create translation vector given the (x,y,z)
        translation_vectors = dm.create_translation_vectors(centroids, zs, modified_intrinsics)

        # Selecting the first translation vector
        translation_vector = translation_vectors[0]
        #"""
        """
        translation_vector = dm.extract_translation_vector_from_RT(
            sample['RT'], 
            modified_intrinsics
        )
        """

        # Draw the poses
        gt_pose = dr.draw_quat(
            image = image[batch_id],
            quaternion = sample['quaternion'][batch_id],
            translation_vector = translation_vector,
            norm_scale = sample['scale'][batch_id],
            norm_factor = sample['norm_factor'][batch_id],
            intrinsics = modified_intrinsics,
            zoom = sample['zoom'][batch_id]
        )

        pred_pose = dr.draw_quat(
            image = image[batch_id],
            quaternion = pred_quaternion[batch_id],
            translation_vector = translation_vector,
            norm_scale = sample['scale'][batch_id],
            norm_factor = sample['norm_factor'][batch_id],
            intrinsics = modified_intrinsics,
            zoom = sample['zoom'][batch_id]
        )

        # Store the drawn pose to list
        gt_poses.append(gt_pose)
        pred_poses.append(pred_pose)

    # Convert list to array 
    gt_poses = np.array(gt_poses, dtype=np.uint8)
    pred_poses = np.array(pred_poses, dtype=np.uint8)

    # Creating a matplotlib figure illustrating the inputs vs outputs
    summary_fig = make_summary_figure(
        image=image,
        gt_pose=gt_poses,
        pred_pose=pred_poses
    )

    return summary_fig  





