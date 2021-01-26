import os
import sys
import pathlib
import warnings
from pprint import pprint

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
#matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.cm

# Local Imports
root = pathlib.Path(os.getenv("ROOT_DIR"))
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

def get_visualized_u_vector_xy(mask, xy, colormap='hsv'):

    # Make channels_last in the image
    xy = dm.set_image_data_format(xy, 'channels_last')

    # Determing the angle of the unit vectors: f: R^2 -> R^1
    angle = np.arctan2(xy[:,:,0], xy[:,:,1])

    # Create norm function to shift data to [0:1]
    norm = matplotlib.colors.Normalize(vmin=np.min(angle), vmax=np.max(angle))

    # Obtain the colormap of choice
    my_colormap = matplotlib.cm.get_cmap(colormap)
    
    # Normalize data, apply the colormap, make it bytes (np.array), and remove the alpha channel
    colorized_angle = my_colormap(norm(angle), bytes=True)[:,:,:3] # removing alpha channel

    # Removing background
    colorized_angle = np.where(np.expand_dims(mask, axis=-1) == 0, 0, colorized_angle)

    return colorized_angle

def get_visualized_u_vector_xys(mask, xys, colormap='hsv'):

    colorized_xys = np.zeros((xys.shape[0], xys.shape[2], xys.shape[3], 3), dtype=np.uint8)

    for id, xy in enumerate(xys):
        colorized_xys[id,:,:,:] = get_visualized_u_vector_xy(mask[id], xy, colormap)

    return colorized_xys

def get_visualized_simple_xy(xy):

    # Make channels_last in the image
    xy = dm.set_image_data_format(xy, 'channels_last')

    # Create a holder of the data
    norm_xy = np.zeros((xy.shape[0], xy.shape[1], 3))

    # Convert the integer data into float [0,1]
    norm_xy[:,:,0] = xy[:,:,0] # Y (Red)
    norm_xy[:,:,2] = xy[:,:,1] # X (Blue)

    return norm_xy

def get_visualized_simple_xys(xys):
    """
    Arguments:
        xys: np.array (Nx2xHxW)
    Returns:
        colorized_xys: np.array (NxHxWx3)
    """

    colorized_xys = np.zeros((xys.shape[0], xys.shape[2], xys.shape[3], 3))

    for id, xy in enumerate(xys):
        colorized_xys[id,:,:,:] = get_visualized_simple_xy(xy)

    return colorized_xys

def get_visualized_z(z, colormap='viridis'):

    # Create norm function to shift data to [0:1]
    norm = matplotlib.colors.Normalize(vmin=0, vmax=8)

    # Obtain the colormap of choice
    my_colormap = matplotlib.cm.get_cmap(colormap)
    
    # Normalize data, apply the colormap, make it bytes (np.array), and remove the alpha channel
    colorized_z = my_colormap(norm(z), bytes=True)[:,:,:3] # removing alpha channel
    
    return colorized_z

def get_visualized_zs(zs, colormap='viridis'):

    colorized_zs = np.zeros((zs.shape[0], zs.shape[1], zs.shape[2], 3), dtype=np.uint8)

    for id, z in enumerate(zs):
        colorized_zs[id,:,:,:] = get_visualized_z(z)

    return colorized_zs

def get_visualized_quaternion(quaternion, bg='black'):

    """
    # METHOD 1
    # Make channels_last in the image
    quaternion = dm.set_image_data_format(quaternion, 'channels_last')

    # Selecting the i,j, and k components
    ijk_component = quaternion[:,:,1:]

    # creating norm function
    norm = matplotlib.colors.Normalize(vmin=-1, vmax=1)

    # Normalize data
    colorized_quat = norm(ijk_component)

    # Remove background
    black = np.zeros_like(colorized_quat)
    colorized_quat = np.where(ijk_component == [0,0,0], black, colorized_quat)
    
    return colorized_quat
    """

    # METHOD 2
    # Make channels_last in the image
    quaternion = dm.set_image_data_format(quaternion, 'channels_last')
    empty_quaternion = np.zeros_like(quaternion)

    # creating norm function
    norm = matplotlib.colors.Normalize(vmin=-1, vmax=1)

    # Normalize data
    norm_quat = norm(quaternion)
    empty_norm_quat = norm(empty_quaternion)

    # Colorized quat
    colorized_quat = d4_to_d3(norm_quat)
    empty_colorized_norm_quat = d4_to_d3(empty_norm_quat)

    # Remove background
    black = np.zeros_like(colorized_quat)
    white = np.ones_like(colorized_quat)

    if bg == 'black':
        colorized_quat = np.where(colorized_quat == empty_colorized_norm_quat, black, colorized_quat)
    elif bg == 'white':
        colorized_quat = np.where(colorized_quat == empty_colorized_norm_quat, white, colorized_quat)

    return colorized_quat

def get_visualized_quaternions(quaternions):

    if len(quaternions.shape) == 3: # A single quaternion 

        colorized_quaternions = get_visualized_quaternion(quaternions)

    elif len(quaternions.shape) == 4: # Batched 3 channel quaternions

        if dm.image_data_format(quaternions[0]) == 'channels_last':
            b, h, w, c = quaternions.shape
        elif dm.image_data_format(quaternions[0]) == 'channels_first':
            b, c, h, w = quaternions.shape
        else:
            raise RuntimeError('Invalid quaternion image input: 1')

        colorized_quaternions = np.zeros((b,h,w,3))

        for id in range(b):
            colorized_quaternions[id,:,:,:] = get_visualized_quaternion(quaternions[id])

    else:
        raise RuntimeError('Invalid quaternion image input: 2')
        
    return colorized_quaternions

def get_visualized_scale(scale):
    """
    Since scale is composed three quantities, each one can be map to the a 
    different colormap.
    """

    # Changing (3xHxW) to (HxWx3)
    colorized_scale = np.moveaxis(scale, 0, -1)

    # Normalize data to improve visualization

    return colorized_scale

def get_visualized_scales(scales):
    
    colorized_scales = np.zeros((scales.shape[0], scales.shape[2], scales.shape[3], 3), dtype=np.float32)

    for id, scale in enumerate(scales):
        colorized_scales[id,:,:,:] = get_visualized_scale(scale)

    return colorized_scales

def d4_to_d3(image):

    new_image = np.zeros((image.shape[0], image.shape[1], 3))

    # Red Channel: R = 255 * (1-C) * (1-K)
    new_image[:,:,0] = (image[:,:,0]) * (image[:,:,3])

    # Green Channel: G = 255 * (1 - M) * (1 - K)
    new_image[:,:,1] = (image[:,:,1]) * (image[:,:,3])

    # Blue Channel: B = 255 * (1 -Y) * (1 - K)
    new_image[:,:,2] = (image[:,:,2]) * (image[:,:,3])

    return new_image

def get_visualized(image, key, mask_colormap):

    if key == 'mask':
        return get_visualized_mask(image, mask_colormap)
    elif key == 'xy':
        return get_visualized_simple_xy(image)
        #return get_visualized_u_vector_xy()
    elif key == 'quaternion':
        return get_visualized_quaternion(image)
    elif key == 'z':
        return get_visualized_z(image)
    else:
        return image

#-------------------------------------------------------------------------------
# Hough Voting Visualization

def get_visualized_hough_voting(
    hypothesis, 
    pruned_hypothesis,
    pixel_xy,
    mask
    ):

    # Shape information
    h,w = mask.shape

    # Flip the x and y to match the style of visualization in this function
    pixel_xy = pixel_xy[[1,0]]
    safe_pixel_xy = torch.squeeze(
        make_pts_index_friendly(torch.unsqueeze(pixel_xy, dim=0), h, w)
    )

    colors = torch.tensor([
        [0,0.5,0], # mask
        [0,1,1], # original hypothesis
        [1,0,1], # pruned hypothesis
        [1,0,0], # final conclusion
    ], dtype=torch.float, device=mask.device)

    bg = torch.zeros((h,w,3), dtype=torch.float)

    # Drawing the mask (blue)
    expand_mask = torch.unsqueeze(mask, dim=-1).expand((h,w,3))
    draw_image = torch.where(expand_mask != 0.0, colors[0], bg)

    # Drawing the initial hypothesis (red)
    draw_image = draw_pts(
        draw_image,
        hypothesis.long(),
        colors[1],
        t=2
    )

    # Draw the prun hypothesis
    draw_image = draw_pts(
        draw_image,
        pruned_hypothesis.long(),
        colors[2],
        t=1
    )

    # Draw the selected center pts
    draw_image = draw_pts(
        draw_image,
        torch.unsqueeze(safe_pixel_xy, dim=0),
        colors[3],
        t=3
    )

    return draw_image

def make_pts_index_friendly(pts, h, w):

    # Creating a out of frame shift to more easily visualize pts outside 
    # the image
    out_of_frame_shift = 5

    # Ensuring the dtype is correct
    pts = pts.long()

    # Height : Y
    pts[:,0] = torch.where(pts[:,0] >= h-out_of_frame_shift, h-out_of_frame_shift, pts[:,0])
    pts[:,0] = torch.where(pts[:,0] < 0, out_of_frame_shift, pts[:,0])
    
    # Width: X
    pts[:,1] = torch.where(pts[:,1] >= w-out_of_frame_shift, w-out_of_frame_shift, pts[:,1])
    pts[:,1] = torch.where(pts[:,1] < 0, out_of_frame_shift, pts[:,1])

    return pts

def draw_pts(draw_image, pts, color, t=1):

    h, w, _ = draw_image.shape

    # Making the pts safe to begin with to allow for better visualization 
    pts = make_pts_index_friendly(pts, h, w)

    ys = pts[:,0]
    xs = pts[:,1]

    s = 2*t+1 # size
    a = (torch.arange(0, s**2, device=draw_image.device) % s).reshape((s, s)) - t
    kernel = torch.stack((a, a.t()), dim=-1).reshape((-1,2))
    
    for i in range(xs.shape[0]):
        
        dilate_x = kernel[:,0] + xs[i]
        dilate_y = kernel[:,1] + ys[i]

        dilate_pts = torch.stack((dilate_y, dilate_x)).t()
        
        """
        safe_pts = self.make_pts_index_friendly(
            dilate_pts, h, w
        )
        """

        draw_image[dilate_pts[:,0], dilate_pts[:,1]] = color

    return draw_image

#-------------------------------------------------------------------------------
# General Matplotlib Functions

def make_summary_figure(**images):

    # Calculating the number of rows and columns
    nr = len(images)
    nc = images[list(images.keys())[0]].shape[0]
    
    h = nr * images[list(images.keys())[0]][0].shape[0]
    w = nc * images[list(images.keys())[0]][0].shape[1]

    largest_dim = max(h,w)
    largest_size_in_inches = 8

    h_ratio = h/largest_dim
    w_ratio = w/largest_dim

    cal_h = h_ratio * largest_size_in_inches
    cal_w = w_ratio * largest_size_in_inches

    # Initializing the figure and axs
    fig = plt.figure(figsize=(cal_w, cal_h))
    fig.subplots_adjust(
        left=0,
        bottom=0,
        right=1,
        top=1,
        wspace=0,
        hspace=0
    )

    for i, (name, image) in enumerate(images.items()):
        if len(image.shape) >= 3: # NHW or NCHW
            for j, img in enumerate(image):

                plt.subplot(nr, nc, 1 + j + nc*i)
                plt.xticks([])
                plt.yticks([])
                if j == 0:
                    plt.ylabel(' '.join(name.split('_')).title())
                    plt.text(25, 25, ' '.join(name.split('_')).title(), bbox=dict(facecolor='white', alpha=0.5), color=(0,0,0))

                img = dm.set_image_data_format(img, "channels_last")

                plt.imshow(img)
        else: # HW only

            plt.subplot(nr, nc, i + 1)
            plt.xticks([])
            plt.yticks([])
            plt.title(' '.join(name.split('_')).title())
            plt.imshow(image)

    # Overall figure configurations
    #plt.tight_layout()

    return fig    

def debug_show(**images):

    fig = make_summary_figure(**images)
    plt.show()

def show_tensor(tensor: torch.Tensor):

    # Standardizing all the data
    img = dm.standardize_image(tensor)
    plt.imshow(img)
    plt.show()

# #-----------------------------------------------------------------------------
# Single Sample Ground Truth and Prediction Visualization

def compare_all(preds, gts, mask_colormap):
    """
    Compare the performance of all matching attributes between the predictions 
    and the ground truth data (within a single sample). Grabbing all of the 
    images within the single sample and stacking them into a single image
    container to later create a figure with the images.
    """

    # Selecting clean image if available
    image_key = 'clean_image' if 'clean_image' in gts.keys() else 'image'

    pred_imgs = None
    gt_imgs = None 

    for key in gts.keys():

        # If the predictions do not include it, then ignore it for now.
        if key not in preds.keys() or key == 'image' or key == 'clean_image':
            continue

        # Get the corresponding image for the key
        pred_img = get_visualized(preds[key], key, mask_colormap)
        gt_img = get_visualized(gts[key], key, mask_colormap)

        # Convert the pred_img and gt_img to dataformat HWC
        pred_img = dm.set_image_data_format(pred_img, "channels_last")
        gt_img = dm.set_image_data_format(gt_img, "channels_last")

        # Convert image from 0-255 to 0-1
        pred_img = skimage.img_as_float32(pred_img)
        gt_img = skimage.img_as_float32(gt_img)

        # For the first one, just copy the image and expand the dims
        if isinstance(pred_imgs, type(None)) and isinstance(gt_imgs, type(None)):
            pred_imgs = pred_img
            gt_imgs = gt_img
            pred_imgs = np.expand_dims(pred_imgs, axis=0)
            gt_imgs = np.expand_dims(gt_imgs, axis=0)

        else: # concatenate other images
            pred_imgs = np.concatenate((pred_imgs, np.expand_dims(pred_img, axis=0)), axis=0)
            gt_imgs = np.concatenate((gt_imgs, np.expand_dims(gt_img, axis=0)), axis=0)

    # Create figure of the prediction and ground truths
    fig = make_summary_figure(
        gt = gt_imgs,
        pred = pred_imgs
    )

    return fig    

#-------------------------------------------------------------------------------
# Visualization of Individual Task

def compare_mask_performance(sample, pred_cat_mask, colormap):

    # Selecting clean image and mask if available
    image_key = 'clean_image' if 'clean_image' in sample.keys() else 'image'
    mask_key = 'clean_mask' if 'clean_mask' in sample.keys() else 'mask'
    
    # Converting visual images into np.uint8 for matplotlib compatibility
    image_vis = sample[image_key]#.astype(np.uint8)
    gt_mask = sample[mask_key].astype(np.uint8)

    # Colorized the binary masks
    gt_mask_vis = get_visualized_masks(gt_mask, colormap)
    pred_mask_vis = get_visualized_masks(pred_cat_mask, colormap)

    # Creating a matplotlib figure illustrating the inputs vs outputs
    summary_fig = make_summary_figure(
        image=image_vis,
        ground_truth_mask=gt_mask_vis,
        predicted_mask=pred_mask_vis)

    return summary_fig

def compare_quat_performance(sample, pred_quaternion, pred_cat_mask, mask_colormap):

    # Selecting clean image and mask if available
    image_key = 'clean_image' if 'clean_image' in sample.keys() else 'image'

    # Converting visual images into np.uint8 for matplotlib compatibility
    image_vis = sample[image_key]#.astype(np.uint8)
    pred_mask_vis = get_visualized_masks(pred_cat_mask, mask_colormap)

    # Get colorized dense quaternion info
    gt_quat_vis = get_visualized_quaternions(sample['quaternion'])
    pred_quat_vis = get_visualized_quaternions(pred_quaternion)

    # Create a matplotlib figure illustrating the inputs vs outputs
    summary_fig = make_summary_figure(
        image = image_vis,
        pred_mask=pred_mask_vis,
        gt_quaternion = gt_quat_vis,
        pred_quaternion = pred_quat_vis
    )

    return summary_fig

def compare_xy_performance(sample, pred_xy, pred_cat_mask, mask_colormap):

    # Selecting clean image and mask if available
    image_key = 'clean_image' if 'clean_image' in sample.keys() else 'image'

    # Converting visual images into np.uint8 for matplotlib compatibility
    image_vis = sample[image_key]#.astype(np.uint8)
    pred_mask_vis = get_visualized_masks(pred_cat_mask, mask_colormap)

    # Get colorized dense xy info
    #gt_xy_vis = get_visualized_simple_xys(sample['xy'])
    #pred_xy_vis = get_visualized_simple_xys(pred_xy)
    gt_xy_vis = get_visualized_u_vector_xys(sample['mask'], sample['xy'])
    pred_xy_vis = get_visualized_u_vector_xys(pred_cat_mask, pred_xy)

    # Create a matplotlib figure illustrating the inputs vs outputs
    summary_fig = make_summary_figure(
        image = image_vis,
        pred_mask=pred_mask_vis,
        gt_xy = gt_xy_vis,
        pred_xy = pred_xy_vis
    )

    return summary_fig

def compare_z_performance(sample, pred_z, pred_cat_mask, mask_colormap):

    # Selecting clean image and mask if available
    image_key = 'clean_image' if 'clean_image' in sample.keys() else 'image'

    # Converting visual images into np.uint8 for matplotlib compatibility
    image_vis = sample[image_key]#.astype(np.uint8)
    pred_mask_vis = get_visualized_masks(pred_cat_mask, mask_colormap)

    # Get colorized dense z info
    gt_z_vis = get_visualized_zs(sample['z'])
    pred_z_vis = get_visualized_zs(pred_z)

    # Create a matplotlib figure illustrating the inputs vs outputs
    summary_fig = make_summary_figure(
        image = image_vis,
        pred_mask=pred_mask_vis,
        gt_z = gt_z_vis,
        pred_z = pred_z_vis
    )

    return summary_fig

def compare_scales_performance(sample, pred_scales, pred_cat_mask, mask_colormap):

    # Selecting clean image and mask if available
    image_key = 'clean_image' if 'clean_image' in sample.keys() else 'image'

    # Converting visual images into np.uint8 for matplotlib compatibility
    image_vis = sample[image_key]#.astype(np.uint8)
    pred_mask_vis = get_visualized_masks(pred_cat_mask, mask_colormap)

    # Get colorized dense z info
    gt_scales_vis = get_visualized_scales(sample['scales'])
    pred_scales_vis = get_visualized_scales(pred_scales)

    # Create a matplotlib figure illustrating the inputs vs outputs
    summary_fig = make_summary_figure(
        image = image_vis,
        pred_mask=pred_mask_vis,
        gt_scales = gt_scales_vis,
        pred_z = pred_scales_vis
    )

    return summary_fig

def compare_hough_voting_performance(image, gt_pred_matches):

    # Obtaining image shape information
    b, h, w, _ = image.shape


    # Container for all the drawn images 
    drawn_gt_uv = torch.zeros_like(image, device=image.device)
    drawn_pred_uv = torch.zeros_like(image, device=image.device)
    drawn_gt_hv = torch.zeros_like(image, device=image.device)
    drawn_pred_hv = torch.zeros_like(image, device=image.device)

    # Per class visualization
    for class_id in range(len(gt_pred_matches)):

        # Obtaining the sample ids
        try:
            sample_ids = gt_pred_matches[class_id]['sample_ids']
        except KeyError:
            # No instances for this class, skip it
            continue

        for sequence_id, sample_id in enumerate(sample_ids):

            # Visualize the gt uv
            gt_vis_uv_img = torch.from_numpy(get_visualized_u_vector_xy(
                gt_pred_matches[class_id]['instance_masks'][0][sequence_id].cpu().numpy(),
                gt_pred_matches[class_id]['xy_mask'][0][sequence_id].cpu().numpy()
            ))

            # Visualize the pred uv
            pred_vis_uv_img = torch.from_numpy(get_visualized_u_vector_xy(
                gt_pred_matches[class_id]['instance_masks'][1][sequence_id].cpu().numpy(),
                gt_pred_matches[class_id]['xy_mask'][1][sequence_id].cpu().numpy()
            ))

            # Visualize gt hypothesis (casting from float to uint8)
            gt_vis_hypo_img = (get_visualized_hough_voting(
                gt_pred_matches[class_id]['hypothesis'][0][sequence_id],
                gt_pred_matches[class_id]['pruned_hypothesis'][0][sequence_id],
                gt_pred_matches[class_id]['xy'][0][sequence_id],
                gt_pred_matches[class_id]['instance_masks'][0][sequence_id],
            )*255).type(torch.uint8)

            # Visualize gt hypothesis (casting from float to uint8)
            pred_vis_hypo_img = (get_visualized_hough_voting(
                gt_pred_matches[class_id]['hypothesis'][1][sequence_id],
                gt_pred_matches[class_id]['pruned_hypothesis'][1][sequence_id],
                gt_pred_matches[class_id]['xy'][1][sequence_id],
                gt_pred_matches[class_id]['instance_masks'][1][sequence_id],
            )*255).type(torch.uint8)

            drawn_gt_uv[sample_id] = torch.where(
                drawn_gt_uv[sample_id] == 0, gt_vis_uv_img, drawn_gt_uv[sample_id]
            )
            drawn_pred_uv[sample_id] = torch.where(
                drawn_pred_uv[sample_id] == 0, pred_vis_uv_img, drawn_pred_uv[sample_id]
            )
            drawn_gt_hv[sample_id] = torch.where(
                drawn_gt_hv[sample_id] == 0, gt_vis_hypo_img, drawn_gt_hv[sample_id]
            )
            drawn_pred_hv[sample_id] = torch.where(
                drawn_pred_hv[sample_id] == 0, pred_vis_hypo_img, drawn_pred_hv[sample_id]
            )

    summary_fig = make_summary_figure(
        gt_uv=drawn_gt_uv.cpu().numpy(),
        gt_hv=drawn_gt_hv.cpu().numpy(),
        pred_uv=drawn_pred_uv.cpu().numpy(),
        pred_hv=drawn_pred_hv.cpu().numpy()
    )

    return summary_fig

#-------------------------------------------------------------------------------
# Visualization of Complete Task

def compare_pose_performance(sample, pred_quaternion):

    # Selecting clean image and mask if available
    image_key = 'clean_image' if 'clean_image' in sample.keys() else 'image'
    mask_key = 'clean_mask' if 'clean_mask' in sample.keys() else 'mask'
    depth_key = 'clean_depth' if 'clean_depth' in sample.keys() else 'depth'

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

def compare_pose_performance_v2(preds, gts, intrinsics):

    # Selecting clean image if available
    image_key = 'clean_image' if 'clean_image' in gts.keys() else 'image'

    # Create the drawn poses
    gt_poses = []
    pred_poses = []

    # For each sample inside the batch
    for i in range(gts['mask'].shape[0]):

        # Obtain the single sample data
        single_preds = {k:v[i] for k,v in preds.items()}
        single_gts = {k:v[i] for k,v in gts.items()}

        # Obtain the data for the predictions via aggregation
        preds_aggregated_data = dm.aggregate_dense_sample(single_preds, intrinsics)

        # Obtain the data for the gts via aggregation
        gts_aggregated_data = dm.aggregate_dense_sample(single_gts, intrinsics)

        # Draw a sample's poses
        pred_pose = dr.draw_quats(
            image = single_preds[image_key], 
            intrinsics = intrinsics,
            quaternions = preds_aggregated_data['quaternion'],
            translation_vectors = preds_aggregated_data['translation_vector'],
            scales = preds_aggregated_data['scales'],
            color=(0,255,255)
        )

        # Draw a sample's poses
        gt_pose = dr.draw_quats(
            image = single_gts[image_key], 
            intrinsics = intrinsics,
            quaternions = gts_aggregated_data['quaternion'],
            translation_vectors = gts_aggregated_data['translation_vector'],
            scales = gts_aggregated_data['scales'],
            color=(0,255,255)
        )

        # Store the drawn pose to list
        gt_poses.append(gt_pose)
        pred_poses.append(pred_pose)

    # Convert list to array 
    gt_poses = np.array(gt_poses)
    pred_poses = np.array(pred_poses)

    # Creating a matplotlib figure illustrating the inputs vs outputs
    summary_fig = make_summary_figure(
        gt_pose=gt_poses,
        pred_pose=pred_poses
    )

    return summary_fig

def compare_pose_performance_v3(preds, gts, intrinsics, pred_mask=None, mask_colormap=None):

    # Selecting clean image if available
    image_key = 'clean_image' if 'clean_image' in gts.keys() else 'image'

    if pred_mask is not None:
        pred_mask = np.argmax(pred_mask, axis=1)
        pred_mask = get_visualized_masks(pred_mask, mask_colormap)

    # Create the drawn poses
    poses = []

    # For each sample inside the batch
    for i in range(gts['mask'].shape[0]):

        # Obtain the single sample data
        single_preds = {k:v[i] for k,v in preds.items()}
        single_gts = {k:v[i] for k,v in gts.items()}

        # Obtain the data for the predictions via aggregation
        preds_aggregated_data = dm.aggregate_dense_sample(single_preds, intrinsics)

        # Obtain the data for the gts via aggregation
        gts_aggregated_data = dm.aggregate_dense_sample(single_gts, intrinsics)

        # Draw a sample's poses
        gt_pose = dr.draw_RTs(
            image = single_gts[image_key], 
            intrinsics = intrinsics,
            RTs = gts_aggregated_data['RT'],
            scales = gts_aggregated_data['scales'],
            color=(0,255,255)
        )

        # Draw a sample's poses
        pose = dr.draw_RTs(
            image = gt_pose, 
            intrinsics = intrinsics,
            RTs = preds_aggregated_data['RT'],
            scales = preds_aggregated_data['scales'],
            color=(255,0,255)
        )

        # Store the drawn pose to list
        poses.append(pose)

    # Convert list to array 
    poses = np.array(poses)

    # Creating a matplotlib figure illustrating the inputs vs outputs
    summary_fig = make_summary_figure(
        poses=poses,
        pred_mask=pred_mask
    )

    return summary_fig

def compare_pose_performance_v4(sample, pred_cat_mask, pred_gt_matches, intrinsics, mask_colormap):
    
    # Create colorized mask
    pred_mask_vis = get_visualized_masks(pred_cat_mask, mask_colormap)

    # Spliting matches based on their sample id
    per_sample_matches = {}
    for match in pred_gt_matches:

        if match['sample_id'] not in per_sample_matches.keys():
            per_sample_matches[match['sample_id']] = [match]

        else:
            per_sample_matches[match['sample_id']].append(match)

    draw_images = []

    # Iterating through all the samples
    for i in per_sample_matches.keys():

        # Selecting the clean image for the specific sample
        draw_image = sample['clean_image'][i]

        # Then selecting the matches associated with that sample
        sample_matches = per_sample_matches[i]

        # Now visualize each match if 'iou_2d_mask' is greater than 0
        for match in sample_matches:

            # Skip non-true matches
            #if match['iou_2d_mask'] <= 0:
            #    continue

            # Exponentiating z to undo torch.log 
            match['z'] = torch.exp(match['z'])

            # Convert match (tensor) to numpy
            np_match = {}
            for k,v in match.items():
                if type(v) == torch.Tensor:
                    np_match[k] = v.cpu().numpy()
                else:
                    np_match[k] = v

            # Creating the translation vectors
            #gt_T = dm.create_translation_vector(np_match['xy'][0].reshape((-1,1)), np_match['z'][0], intrinsics)
            #pred_T = dm.create_translation_vector(np_match['xy'][1].reshape((-1,1)), np_match['z'][1], intrinsics)

            # Creating rotation matrix
            #gt_RT = dm.quat_2_RT_given_T_in_world(np_match['quaternion'][0], gt_T)
            #pred_RT = dm.quat_2_RT_given_T_in_world(np_match['quaternion'][1], pred_T)

            # Drawing the ground truth pose
            draw_image = dr.draw_RT(
                image = draw_image, 
                intrinsics = intrinsics,
                RT = np_match['RT'][0],
                scale = np_match['scales'][0],
                color=(0,255,255)
            )

            # Drawing the predicted pose
            draw_image = dr.draw_RT(
                image = draw_image, 
                intrinsics = intrinsics,
                RT = np_match['RT'][1],
                scale = np_match['scales'][1],
                color=(255,0,255)
            )

        # After drawing the image, added it to the list of drawn_images
        draw_images.append(draw_image)

    # Stack the drawn images
    draw_images = np.stack(draw_images)

    # Creating a matplotlib figure illustrating the inputs vs outputs
    summary_fig = make_summary_figure(
        poses=draw_images,
        pred_mask=pred_mask_vis
    )

    return summary_fig

def compare_pose_performance_v5(
    image: torch.Tensor, 
    gt_pred_matches: dict, 
    pred_cat_mask: torch.Tensor, 
    mask_colormap,
    intrinsics: np.ndarray
    ):

    # Draw image
    draw_image = image.cpu().numpy()

    # Create colorized mask
    pred_mask_vis = get_visualized_masks(pred_cat_mask, mask_colormap)

    # Per class visualization
    for class_id in range(len(gt_pred_matches)):

        # Obtaining the sample ids
        try:
            sample_ids = gt_pred_matches[class_id]['sample_ids']
        except KeyError:
            # No instances for this class, skip it
            continue

        # Drawing per sample
        for sequence_id, sample_id in enumerate(sample_ids):

            # Draw the ground truth pose
            gt_pose = dr.draw_RT(
                image=draw_image[sample_id],
                intrinsics=intrinsics,
                RT = gt_pred_matches[class_id]['RT'][0][sequence_id].cpu().numpy(),
                scale = gt_pred_matches[class_id]['scales'][0][sequence_id].cpu().numpy(),
                color=(0,255,255)
            )

            # Draw the pred pose 
            gt_pred_pose = dr.draw_RT(
                image=gt_pose,
                intrinsics=intrinsics,
                RT = gt_pred_matches[class_id]['RT'][1][sequence_id].cpu().numpy(),
                scale = gt_pred_matches[class_id]['scales'][1][sequence_id].cpu().numpy(),
                color=(255,0,255)
            )

            # Overwrite the older draw image
            draw_image[sample_id] = gt_pred_pose            

    summary_fig = make_summary_figure(
        poses=draw_image,
        pred_mask=pred_mask_vis
    )

    return summary_fig

#-------------------------------------------------------------------------------
# Plot metrics

def plot_ap(
    aps, 
    x_range, 
    cls_names,
    title,
    x_axis_label
    ):

    # Initializing the matplotlib figure
    fig = plt.figure()
    plt.title(title)
    plt.xlabel(x_axis_label)

    # Iterating through all the classes data
    for i, ap in enumerate(aps):

        # Obtaining the name of the class
        class_name = cls_names[i]

        # Add the data to the plot
        if class_name == 'mean':
            plt.plot(x_range, ap, '--', label=f'{class_name}')
        else:
            plt.plot(x_range, ap, '-o', label=f'{class_name}')

    plt.legend()

    return fig

def plot_aps(
    aps,
    titles,
    x_ranges,
    cls_names,
    x_axis_labels
    ):

    # Initializing the matplotlib figure
    fig, axs = plt.subplots(1,3, sharey=True, figsize=(10, 5))

    # Placing shared ylabel for AP
    axs[0].set_ylabel('AP%')

    for ap_id, ap_name in enumerate(aps.keys()):

        # Setting plot title, xlabel, grid and xy limits
        axs[ap_id].set_title(titles[ap_id])
        axs[ap_id].set_xlabel(x_axis_labels[ap_id])
        axs[ap_id].grid(True)
        axs[ap_id].set_ylim([0, 100])
        axs[ap_id].set_xlim([x_ranges[ap_id][0], x_ranges[ap_id][-1]])

        for class_id, class_ap in enumerate(aps[ap_name]):

            class_name = cls_names[class_id]

            # Add the data to the plot
            if class_name == 'mean':
                axs[ap_id].plot(x_ranges[ap_id], class_ap * 100, '--', label=f'{class_name}')
            else:
                axs[ap_id].plot(x_ranges[ap_id], class_ap * 100, '-o', label=f'{class_name}')

    plt.legend()

    return fig
