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

#-------------------------------------------------------------------------------
# File's Constants

#-------------------------------------------------------------------------------
# Class

#-------------------------------------------------------------------------------
# Dimensional Compression Functions

def vstack_multichannel_masks(multi_c_image, binary=True):

    output = multi_c_image[0]

    # Stacking
    for i in range(1, multi_c_image.shape[0]):
        output = np.concatenate([output, multi_c_image[i,:,:]])

    # Converting to binary
    output = (output != 0) * 255

    return output

def vstack_multichaknnel_quat_or_scales(multi_c_image):

    n, _, h, w = multi_c_image.shape
    output = np.empty((h*n,w))

    for c in range(n):
        for i in range(h):
            for j in range(w):
                output[i+h*c,j] = (multi_c_image[c,:,i,j] != 0).all() * 255

    return output

def collapse_multichannel_masks(multi_c_image):

    output = multi_c_image[0]

    # Collapsing
    for i in range(1, multi_c_image.shape[0]):
        output += multi_c_image[i,:,:]

    # Converting to binary
    output = (output != 0) * 255

    output = torch.clamp(output, 0, 255)

    return output

def collapse_multichannel_quat_or_scales(multi_c_image):

    n, _, h, w = multi_c_image.shape
    output = multi_c_image[0]

    for c in range(1, n):

        for i in range(h):
            for j in range(w):
                output[i,j] += (multi_c_image[c,:,i,j] != 0).all() * 255

    output = torch.clamp(output, 0, 255)

    return output

#-------------------------------------------------------------------------------
# Main Visualization Functions

def make_summary_figure(colormap=None, **images):

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

                if len(img.shape) == 3: # CHW to HWC
                    img = np.moveaxis(img, 0, -1)

                plt.imshow(img)
        else: # HW only

            plt.subplot(nr, nc, i + 1)
            plt.xticks([])
            plt.yticks([])
            plt.title(' '.join(name.split('_')).title())
            plt.imshow(image)

    # Overall figure configurations

    return fig    

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

    colorized_masks = np.zeros((masks.shape[0],3,masks.shape[-2],masks.shape[-1]))

    for id, mask in enumerate(masks):
        colorized_masks[id,:,:,:] = get_visualized_mask(mask, colormap)

    return colorized_masks

#-------------------------------------------------------------------------------
# PASCAL Functions
# From this link: 
# https://d2l.ai/chapter_computer-vision/semantic-segmentation-and-dataset.html

def read_voc_images(voc_dir, is_train=True):
    """Read all VOC feature and label images."""

    txt_fname = os.path.join(voc_dir, 'ImageSets', 'Segmentation',
                             'train.txt' if is_train else 'val.txt')
    with open(txt_fname, 'r') as f:
        images = f.read().split()
    features, labels = [], []
    for i, fname in enumerate(images):
        features.append(skimage.io.imread(os.path.join(
            voc_dir, 'JPEGImages', f'{fname}.jpg')))
        labels.append(skimage.io.imread(os.path.join(
            voc_dir, 'SegmentationClass', f'{fname}.png')))
    return features, labels

def build_colormap2label():
    """Build an RGB color to label mapping for segmentation."""
    colormap2label = np.zeros(256 ** 3)

    for i, colormap in enumerate(project.constants.VOC_COLORMAP):
        colormap2label[(colormap[0]*256 + colormap[1])*256 + colormap[2]] = i
    
    return colormap2label

def voc_label_indices(colormap, colormap2label):
    """Map an RGB color to a label."""

    colormap = colormap.astype(np.int32)
    
    idx = ((colormap[:, :, 0] * 256 + colormap[:, :, 1]) * 256
           + colormap[:, :, 2])
    
    return colormap2label[idx]

def voc_rand_crop(feature, label, height, width):
    """Randomly crop for both feature and label images."""
    from mxnet import image

    feature, rect = image.random_crop(feature, (width, height))
    label = image.fixed_crop(label, *rect)
    
    return feature, label

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




