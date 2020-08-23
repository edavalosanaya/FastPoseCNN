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

import torch
import torchvision

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

def get_img_from_fig(fig, dpi=180):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=dpi)
    buf.seek(0)
    img_arr = np.frombuffer(buf.getvalue(), dtype=np.uint8)
    buf.close()
    img = cv2.imdecode(img_arr, 1)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    return img

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

def make_summary_image(title, random_sample, outputs):

    # Desconstructing the random sample
    color_image, depth_image, zs, masks, coord_map, scales_img, quat_img = random_sample

    # Select one sample from the batch
    color_image = color_image[0]
    mask = masks[0]

    # Testing mask
    """
    pdb.set_trace()
    mask = torch.where(mask != 0, torch.tensor([255]).cuda(), torch.tensor([0]).cuda())
    out_path = project.cfg.TEST_OUTPUT / 'test_mask.png'
    torchvision.utils.save_image(mask.type(torch.FloatTensor), out_path)
    #"""

    vis_gt_mask = get_visualized_mask(mask)

    gt_images = [color_image, vis_gt_mask]

    # Then creating matplotlib figure with all the images like this 
    # https://www.tensorflow.org/tensorboard/image_summaries

    # Converting probabilities to classes
    pred = torch.argmax(outputs, dim=1)[0]

    # Converting classes into colors
    vis_pred_mask = get_visualized_mask(pred)
    vis_pred_output = [vis_pred_mask]

    # Formatting the torch images into numpy images with the right H,W,C layout
    formatted_gt_images = torch_to_numpy(gt_images)
    formatted_pred_images = torch_to_numpy(vis_pred_output)

    # Now creating single image with all the formatted images using this example: 
    # https://stackoverflow.com/questions/46615554/how-to-display-multiple-images-in-one-figure-correctly/46616645
    
    col = len(formatted_gt_images)

    fig, axs = plt.subplots(2, col)

    for i, gt_img in enumerate(formatted_gt_images):
        axs[0, i].imshow(gt_img)
        axs[0, i].axis('off')

    axs[1, 0].axis('off')

    for j, pred_img in enumerate(formatted_pred_images):
        axs[1, j+1].imshow(pred_img)
        axs[1, j+1].axis('off')

    # Now saving saving figure as an image
    # Using this method: https://stackoverflow.com/a/57988387/13231446

    # matplotlib
    fig.tight_layout(rect=[0, 0.03, 1, 0.95])
    fig.suptitle(title)

    # Creating the image
    image_from_plot = get_img_from_fig(fig)

    return image_from_plot

def get_visualized_mask(mask, PIL_transform=False):

    colorized_mask = torch.zeros((3,mask.shape[-2], mask.shape[-1])).cuda()

    for class_id, class_color in enumerate(project.constants.SYNSET_COLORS):
        for i in range(3):
            X = torch.tensor(class_color[i]).cuda()
            Y = torch.tensor(0).cuda()
            colorized_mask[i,:,:] += torch.where(mask == class_id, X, Y)

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
# File's Main

if __name__ == "__main__":

    # Loading a test dataset
    dataset = dataset.NOCSDataset(CAMERA_DATASET, dataset_max_size=5)

    # Creating a dataloader to include the batch dimension
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=True)

    # Obtaining a test sample
    test_sample = next(iter(dataloader))

    # Getting specific inputs
    color_images = test_sample[0]
    masks = test_sample[3]

    # Create similar to the nets output
    outputs = torch.stack((masks,masks,masks,masks,masks,masks,masks),dim=1)

    # Testing test run image
    #"""
    epoch = 1
    num_epoch = 100
    title = f'Input and Outputs: Epoch {epoch}/{num_epoch}'
    
    summary_image = make_summary_image(title, test_sample, outputs)
    summary_image = cv2.cvtColor(summary_image, cv2.COLOR_RGB2BGR)
    out_path = project.cfg.TEST_OUTPUT / 'summary_image.png'
    cv2.imwrite(str(out_path), summary_image)
    #"""




