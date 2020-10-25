import os
import sys
import pathlib
import warnings
warnings.filterwarnings('ignore')

import pdb

import cv2
import numpy as np
import PIL
import io
import random
import skimage.io

import torch
import torchvision

import matplotlib
if os.environ.get('DISPLAY', '') == '':
    matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.cm

# Local Imports
root = next(path for path in pathlib.Path(os.path.abspath(__file__)).parents if path.name == 'FastPoseCNN')
sys.path.append(str(pathlib.Path(__file__).parent))

import project
import dataset
import draw
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
        plt.xticks([])
        plt.yticks([])
        plt.xlabel(' '.join(name.split('_')).title())

        image = dm.standardize_image(image)
        plt.imshow(image)

    plt.show()

#-------------------------------------------------------------------------------
# Batch Ground Truth and Prediction Visualization

def compare_mask_performance(sample, pl_module, colormap):

    # Selecting clean image and mask if available
    image_key = 'clean image' if 'clean image' in sample.keys() else 'image'
    mask_key = 'clean mask' if 'clean mask' in sample.keys() else 'mask'
    
    # Converting visual images into np.uint8 for matplotlib compatibility
    image_vis = sample[image_key].astype(np.uint8)
    gt_mask = sample[mask_key].astype(np.uint8)
    
    # Given the sample, make the prediction with the PyTorch Lightning Module
    logits = pl_module(torch.from_numpy(sample['image']).float().to(pl_module.device)).detach()
    pr_mask = torch.nn.functional.sigmoid(logits).cpu().numpy()

    # Target (ground truth) data format 
    if len(gt_mask.shape) == len('BCHW'):

        if pr_mask.shape[1] == 1: # Binary segmentation
            pr_mask = pr_mask[:,0,:,:]
            gt_mask = gt_mask[:,0,:,:]

        else: # Multi-class segmentation
            pr_mask = np.argmax(pr_mask, axis=1)
            gt_mask = np.argmax(gt_mask, axis=1)

    elif len(gt_mask.shape) == len('BHW'):

        if pr_mask.shape[1] == 1: # Binary segmentation
            pr_mask = pr_mask[:,0,:,:]

        else: # Multi-class segmentation
            pr_mask = np.argmax(pr_mask, axis=1)

    # Colorized the binary masks
    gt_mask_vis = get_visualized_masks(gt_mask, colormap)
    pr_mask = get_visualized_masks(pr_mask, colormap)

    # Creating a matplotlib figure illustrating the inputs vs outputs
    summary_fig = make_summary_figure(
        image=image_vis,
        ground_truth_mask=gt_mask_vis,
        predicited_mask=pr_mask)

    return summary_fig

def compare_pose_performance(sample, pl_module):

    # Selecting clean image and mask if available
    image_key = 'clean image' if 'clean image' in sample.keys() else 'image'
    mask_key = 'clean mask' if 'clean mask' in sample.keys() else 'mask'
    depth_key = 'clean depth' if 'clean depth' in sample.keys() else 'depth'

    # Getting the image, mask, and depth
    image, mask, depth = sample[image_key], sample[mask_key], sample[depth_key]

    # Given the sample, make the prediciton with the PyTorch Lightning Moduel
    logits = pl_module(torch.from_numpy(sample['image']).float().to(pl_module.device)).detach()
    pred_quaternion = logits.cpu().numpy()

    # Creating the translation vector
    modified_intrinsics = project.constants.INTRINSICS.copy()
    modified_intrinsics[0,2] = sample['image'].shape[1] / 2
    modified_intrinsics[1,2] = sample['image'].shape[0] / 2

    # Create the drawn poses
    gt_poses = []
    pred_poses = []

    for batch_id in range(image.shape[0]):
        
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

        # Draw the poses
        gt_pose = draw.draw_quat(
            image = image[batch_id],
            quaternion = sample['quaternion'][batch_id],
            translation_vector = translation_vector,
            norm_scale = sample['scale'][batch_id],
            norm_factor = sample['norm_factor'][batch_id],
            intrinsics = modified_intrinsics,
            zoom = sample['zoom'][batch_id]
        )

        pred_pose = draw.draw_quat(
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

#-------------------------------------------------------------------------------
# File's Main

if __name__ == "__main__":

    # Loading a test dataset
    dataset = dataset.NOCSDataset(project.cfg.CAMERA_DATASET, dataset_max_size=2)

    # Creating a dataloader to include the batch dimension
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=True)

    # Obtaining a test sample
    test_sample = next(iter(dataloader))

    # Getting specific inputs
    color_images = test_sample[0]
    masks = test_sample[3]

    # Create similar to the nets output
    outputs = torch.stack((masks,masks),dim=1)




