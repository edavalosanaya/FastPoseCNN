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

# Local Imports
root = next(path for path in pathlib.Path(os.path.abspath(__file__)).parents if path.name == 'FastPoseCNN')
sys.path.append(str(pathlib.Path(__file__).parent))

import project
import dataset

#-------------------------------------------------------------------------------
# File's Constants

CAMERA_DATASET = root.parents[1] / 'datasets' / 'NOCS' / 'camera' / 'val'

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

def torch_to_numpy(imgs):

    formatted_images = []

    for img in imgs:

        # Converting images with C,H,W to H,W,C
        if len(img.shape) == 3: # C,H,W
            new_img = img.permute(1,2,0)

        # Converting tensors to numpy arrays
        new_img = new_img.cpu().numpy()

        # Convert to the uint8 dataset
        new_img = new_img.astype(np.uint8)
        
        # Saving formatted image
        formatted_images.append(new_img)

    return formatted_images

def make_summary_figure(random_sample, pred_masks):

    #pdb.set_trace()
    # Randomly selecting a sample from the batch
    batch_size = random_sample['color_image'].shape[0]
    random_choice = random.choice(list(range(batch_size)))

    # Select one sample from the batch
    color_image = random_sample['color_image'][random_choice]
    mask = random_sample['masks'][random_choice]
    vis_gt_mask = get_visualized_mask(mask)
    gt_images = [color_image, vis_gt_mask]

    # Predicted output of the neural network
    pred_mask = pred_masks[random_choice]
    vis_pred_mask = get_visualized_mask(pred_mask)
    pred_images = [vis_pred_mask]

    # Convert torch into numpy
    formatted_gt_images = torch_to_numpy(gt_images)
    formatted_pred_images = torch_to_numpy(pred_images)

    # Initializing the figure and axs
    col = len(gt_images)
    fig, axs = plt.subplots(2, col)

    # Inputting the ground truth images into the subplot 
    for i, gt_img in enumerate(formatted_gt_images):
        axs[0, i].imshow(gt_img)
        axs[0, i].axis('off')

    # remove the non-utilized slot of the subplot
    axs[1, 0].axis('off')

    # Inputting the predicted images into the subplot 
    for j, pred_img in enumerate(formatted_pred_images):
        axs[1, j+1].imshow(pred_img)
        axs[1, j+1].axis('off')

    # Overall figure configurations
    fig.tight_layout(rect=[0, 0.03, 1, 0.95])
    fig.suptitle('Summary')

    return fig    

def get_visualized_mask(mask, PIL_transform=False):

    colorized_mask = torch.zeros((3,mask.shape[-2], mask.shape[-1])).cuda()

    for class_id, class_color in enumerate(project.constants.SYNSET_COLORS):
        for i in range(3):
            X = torch.tensor(class_color[i]).cuda()
            Y = torch.tensor(0).cuda()
            if len(mask.shape) == 2:
                colorized_mask[i,:,:] += torch.where(mask == class_id, X, Y)
            elif len(mask.shape) == 3:
                colorized_mask[i,:,:] += torch.where(mask[0,:,:] == class_id, X, Y)

    if PIL_transform:
        img = torchvision.transforms.ToPILImage()(colorized_mask).convert("RGB")
        return img

    return colorized_mask

def get_visualized_masks(masks, PIL_transform=False):

    colorized_masks = torch.zeros((masks.shape[0],3,masks.shape[-2],masks.shape[-1]))

    for id, mask in enumerate(masks):
        colorized_masks[id,:,:,:] = get_visualized_mask(mask)

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
    dataset = dataset.NOCSDataset(CAMERA_DATASET, dataset_max_size=2)

    # Creating a dataloader to include the batch dimension
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=True)

    # Obtaining a test sample
    test_sample = next(iter(dataloader))

    # Getting specific inputs
    color_images = test_sample[0]
    masks = test_sample[3]

    # Create similar to the nets output
    outputs = torch.stack((masks,masks),dim=1)




