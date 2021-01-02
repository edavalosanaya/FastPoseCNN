import os
import sys
import warnings
warnings.filterwarnings('ignore')

import cv2
import imutils

from pyquaternion import Quaternion

import math
import numpy as np
import skimage

import scipy.spatial
import scipy.linalg
import sklearn.preprocessing
import scipy.spatial.transform

import matplotlib
import matplotlib.pyplot as plt

import torch

# Local Imports

import project

#-------------------------------------------------------------------------------
# Classes   

#-------------------------------------------------------------------------------
# Simple Tool Function

def image_data_format(image):

    if len(image.shape) == 4: # Batched sample
        if image.shape[1] > image.shape[-1]:
            return 'channels_last'
        else:
            return 'channels_first'

    elif len(image.shape) == 3: # Simple RGB image
        if image.shape[0] > image.shape[-1]:
            return 'channels_last'
        else:
            return 'channels_first'

    else: # 2 channel (HxW)
        return 'channels_none'

def get_number_of_channels(image):

    if image_data_format(image) == 'channels_last':
        return image.shape[-1]

    elif image_data_format(image) == 'channels_first':
        return image.shape[0]

    else:
        return -1

def set_image_data_format(image, dataformat):

    if isinstance(image, np.ndarray): # numpy

        current_dataformat = image_data_format(image)

        if current_dataformat == dataformat or current_dataformat == 'channels_none':
            return image
        else:
            if len(image.shape) == 4:
                return np.moveaxis(image, 1, -1)
            elif len(image.shape) == 3:
                return np.moveaxis(image, 0, -1)

    elif isinstance(image, torch.Tensor): # tensor

        current_dataformat = image_data_format(image)

        if current_dataformat == dataformat or current_dataformat == 'channels_none':
            return image
        else:
            if len(image.shape) == 4:
                return image.permute(0, 3, 2, 1)
            elif len(image.shape) == 3:
                return image.permute(2, 0, 1)

def standardize_image(image, get_original_data=False):

    # ensure that it is numpy (np.uint8)
    was_tensor = isinstance(image, torch.Tensor)
    if was_tensor:
        image = image.cpu().numpy()
        was_tensor = True

    """
    # Ensure the right datatype
    original_dtype = image.dtype
    if image.dtype != np.uint8:
        image = skimage.img_as_ubyte(image)
    """

    # ensure the correct dataformat
    original_image_dataformat = image_data_format(image)
    if original_image_dataformat != 'channels_last':
        image = set_image_data_format(image, 'channels_last')

    if get_original_data:
        return image, was_tensor, original_image_dataformat
    else:
        return image

def dec_correct_image_dataformat(function):

    restore_like_input_image = False

    def wrapper_function(*args, **kwargs):

        # Determine if args were used, instead of kwargs
        args_used = True if len(list(args)) else False

        # Finding the image in args and kwargs
        if 'image' in list(kwargs.keys()):
            image = kwargs['image']
        else:
            image = args[0]

        # Standarize image to numpy np.uint8
        image, was_tensor, original_image_dataformat = standardize_image(image, get_original_data=True)

        # Store back the altered image
        if args_used:
            new_args = list(args)
            new_args[0] = image
            args = tuple(new_args)
        else:
            kwargs['image'] = image
        
        # Apply function
        drawn_image = function(*args, **kwargs)

        # Restore the image to the exact same as the input
        if restore_like_input_image:
            drawn_image = set_image_data_format(drawn_image, original_image_dataformat)
            #drawn_image = drawn_image.astype(original_dtype)

            if was_tensor:
                drawn_image = torch.from_numpy(drawn_image)

        return drawn_image

    return wrapper_function

def standardize_depth(depth):

    # Converting depth to the correct shape and dtype
    if len(depth.shape) == 3: # encoded depth image
        new_depth = np.uint16(depth[:, :, 1] * 256) + np.uint16(depth[:,:,2])
        new_depth = new_depth.astype(np.uint16)
        return new_depth
    elif len(depth.shape) == 2 and depth.dtype == 'uint16':
        return depth
    else:
        assert False, '[ Error ]: Unsupported depth type'

def compress_dict(my_dict, additional_subkey=None):


    new_dict = {}

    for key in my_dict.keys():

        if additional_subkey:
            new_dict[f"{key}/{additional_subkey}"] = None

        for subkey in my_dict[key].keys():
            new_dict[f"{key}/{subkey}"] = my_dict[key][subkey]

    return new_dict

#-------------------------------------------------------------------------------
# create Functions

def create_uniform_dense_image(mask, array):
    
    # Create the image for the output
    output = np.zeros((mask.shape[0], mask.shape[1], array.shape[0]))

    for i in range(array.shape[0]):

        output[:,:,i] = np.where(mask == 1, array[i], 0)

    return output

def create_dense_quaternion(mask, json_data):

    # Ultimately the output
    quaternions = np.zeros((mask.shape[0], mask.shape[1], 4))

    # Given the data, modify any data necessary
    instance_dict = json_data['instance_dict']

    for instance_id in instance_dict.keys():

        selected_class_id = instance_dict[instance_id]
        location_id = list(instance_dict.keys()).index(instance_id)
        selected_quaternion = np.asarray(json_data['quaternions'][location_id], dtype=np.float32)
        
        instance_mask = np.equal(mask, int(instance_id)) * 1
        instance_mask = instance_mask.astype(np.uint8)

        # If the instance mask is empty, simply skip this instance
        if (np.unique(instance_mask) == np.array([0])).all():
            continue

        dense_quaternion = create_uniform_dense_image(instance_mask, selected_quaternion)
        quaternions += dense_quaternion

    return quaternions

def create_dense_scales(mask, json_data):

    # Ultimately the output
    scales = np.zeros((mask.shape[0], mask.shape[1], 3))

    # Given the data, modify any data necessary
    instance_dict = json_data['instance_dict']

    for instance_id in instance_dict.keys():

        selected_class_id = instance_dict[instance_id]
        location_id = list(instance_dict.keys()).index(instance_id)
        selected_scale = np.asarray(json_data['scales'][location_id], dtype=np.float32)
        selected_norm = np.asarray(json_data['norm_factors'][location_id], dtype=np.float32)
        
        selected_norm_scale = selected_scale / selected_norm

        instance_mask = np.equal(mask, int(instance_id)) * 1
        instance_mask = instance_mask.astype(np.uint8)

        # If the instance mask is empty, simply skip this instance
        if (np.unique(instance_mask) == np.array([0])).all():
            continue

        dense_scale = create_uniform_dense_image(instance_mask, selected_norm_scale)
        scales += dense_scale

    return scales

def create_dense_3d_centers(mask, json_data):

    # Ultimately the output
    xys = np.zeros((mask.shape[0], mask.shape[1], 2))
    zs = np.zeros_like(mask)

    # Getting mask shape
    h, w = mask.shape

    # Given the data, modify any data necessary
    instance_dict = json_data['instance_dict']

    for instance_id in instance_dict.keys():

        selected_class_id = instance_dict[instance_id]
        location_id = list(instance_dict.keys()).index(instance_id)
        selected_RT = np.asarray(json_data['RTs'][location_id], dtype=np.float32)

        # Creating the instance mask for the object
        instance_mask = np.equal(mask, int(instance_id)) * 1
        instance_mask = instance_mask.astype(np.uint8)

        # If the instance mask is empty, simply skip this instance
        if (np.unique(instance_mask) == np.array([0])).all():
            continue

        # Obtaining the 2D projection of the 3D center point
        center = np.array([[0,0,0]]).transpose()
        center_2d = transform_3d_camera_coords_to_2d_quantized_projections(
            center,
            selected_RT,
            project.constants.CAMERA_INTRINSICS
        )[0]
        center_2d = np.flip(center_2d)

        # Constructing the unit vectors pointing to the center
        x_coord = np.mod(np.arange(w*h), w).reshape((h,w))
        y_coord = np.mod(np.arange(w*h), h).reshape((w,h)).transpose()
        coord = np.dstack([y_coord,x_coord])
        vector = np.divide((center_2d - coord), np.expand_dims(np.linalg.norm(center_2d - coord, axis=-1), axis=-1))

        # Obtaining the z component for the 3D centroid
        z_value = extract_z_from_RT(selected_RT)

        # Applying the mask on the depth and the unit vectors
        z = instance_mask * z_value
        y = np.where(instance_mask, vector[:,:,0], 0)
        x = np.where(instance_mask, vector[:,:,1], 0)
        xy = np.dstack([y,x])

        # Removing infinities and nan values
        xy = np.nan_to_num(xy)
        z = np.nan_to_num(z)

        xys += xy
        zs += z

    return xys, zs

def create_simple_dense_3d_centers(mask, json_data):

    # Ultimately the output
    xys = np.zeros((mask.shape[0], mask.shape[1], 2))
    zs = np.zeros_like(mask)

    # Getting mask shape
    h, w = mask.shape

    # Given the data, modify any data necessary
    instance_dict = json_data['instance_dict']

    for instance_id in instance_dict.keys():

        selected_class_id = instance_dict[instance_id]
        location_id = list(instance_dict.keys()).index(instance_id)
        selected_RT = np.asarray(json_data['RTs'][location_id], dtype=np.float32)

        # Creating the instance mask for the object
        instance_mask = np.equal(mask, int(instance_id)) * 1
        instance_mask = instance_mask.astype(np.uint8)

        # If the instance mask is empty, simply skip this instance
        if (np.unique(instance_mask) == np.array([0])).all():
            continue

        # Obtaining the 2D projection of the 3D center point
        center = np.array([[0,0,0]]).transpose()
        center_2d = transform_3d_camera_coords_to_2d_quantized_projections(
            center,
            selected_RT,
            project.constants.CAMERA_INTRINSICS
        )[0]

        # Constructing the unit vectors pointing to the center
        x_coord = np.ones_like(mask, dtype=np.float32) * (center_2d[0] / w)
        y_coord = np.ones_like(mask, dtype=np.float32) * (center_2d[1] / h)
        coord = np.dstack([x_coord,y_coord])

        # Obtaining the z component for the 3D centroid
        z_value = extract_z_from_RT(selected_RT)

        # Applying the mask on the depth and the unit vectors
        z = instance_mask * z_value
        y = np.where(instance_mask, coord[:,:,0], 0)
        x = np.where(instance_mask, coord[:,:,1], 0)
        xy = np.dstack([x,y])

        # Removing infinities and nan values
        xy = np.nan_to_num(xy)
        z = np.nan_to_num(z)

        xys += xy
        zs += z

    return xys, zs

#-------------------------------------------------------------------------------
# get Functions

def extract_2d_bboxes_from_masks(masks):
    """
    Computing 2D bounding boxes from mask(s).
    Input
        masks: [height, width, number of mask]. Mask pixel are either 1 or 0
    Output
        bboxes [number of masks, (y1, x1, y2, x2)]
    """

    num_instances = masks.shape[-1]
    bboxes = np.zeros([num_instances, 4], dtype=np.int32)

    for i in range(num_instances):
        mask = masks[:, :, i]

        # Bounding box
        horizontal_indicies = np.where(np.any(mask, axis=0))[0]
        vertical_indicies = np.where(np.any(mask, axis=1))[0]

        if horizontal_indicies.shape[0]:
            x1, x2 = horizontal_indicies[[0, -1]]
            y1, y2 = vertical_indicies[[0, -1]]

            # x2 and y2 should not be part of the box. Increment by 1
            x2 += 1
            y2 += 1

        else:
            # No mask for this instance. Error might have occurred
            x1, x2, y1, y2 = 0,0,0,0

        bboxes[i] = np.array([y1, x1, y2, x2]) 

    return bboxes

def get_3d_bbox(scale, shift = 0):
    """
    Input: 
        scale: [3] or scalar
        shift: [3] or scalar
    Return 
        bbox_3d: [3, N]

    """
    if hasattr(scale, "__iter__"):
        bbox_3d = np.array([[scale[0] / 2, +scale[1] / 2, scale[2] / 2],
                  [scale[0] / 2, +scale[1] / 2, -scale[2] / 2],
                  [-scale[0] / 2, +scale[1] / 2, scale[2] / 2],
                  [-scale[0] / 2, +scale[1] / 2, -scale[2] / 2],
                  [+scale[0] / 2, -scale[1] / 2, scale[2] / 2],
                  [+scale[0] / 2, -scale[1] / 2, -scale[2] / 2],
                  [-scale[0] / 2, -scale[1] / 2, scale[2] / 2],
                  [-scale[0] / 2, -scale[1] / 2, -scale[2] / 2]]) + shift
    else:
        bbox_3d = np.array([[scale / 2, +scale / 2, scale / 2],
                  [scale / 2, +scale / 2, -scale / 2],
                  [-scale / 2, +scale / 2, scale / 2],
                  [-scale / 2, +scale / 2, -scale / 2],
                  [+scale / 2, -scale / 2, scale / 2],
                  [+scale / 2, -scale / 2, -scale / 2],
                  [-scale / 2, -scale / 2, scale / 2],
                  [-scale / 2, -scale / 2, -scale / 2]]) +shift

    bbox_3d = bbox_3d.transpose()
    return bbox_3d

def get_3d_bboxes(scales, shift = 0):

    bboxes_3d = []

    for i in range(scales.shape[0]):

        bboxes_3d.append(get_3d_bbox(scales[i,:], shift))

    return bboxes_3d

def get_mask_centroids(mask, class_id=None):

    try:
        _, cnts, hie = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    except Exception as e:
        cnts, hie = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    # Hierarchy representation in OpenCV: [Next, Previous, First_Child, Parent]

    centroids = []

    for i, c in enumerate(cnts):

        # if the contour has a parent, meaning inside another contour, skip it.
        if hie[0][i][3] != -1: 
            continue

        try:
            M = cv2.moments(c)
            cX = int(M["m10"] / M["m00"])
            cY = int(M["m01"] / M["m00"])
            centroids.append(np.array([cX, cY]))
        except ZeroDivisionError:
            pass

    return centroids

def get_masks_centroids(masks):

    total_centroids = []

    # Converting to the right dtype
    if type(masks) == torch.Tensor:
        masks = masks.numpy().astype(np.uint8)
    elif masks.dtype != np.uint8:
        masks = skimage.img_as_ubyte(masks)

    for class_id in np.unique(masks):

        if class_id == 0: # background, skip
            continue
        
        class_mask = (np.equal(masks, class_id) * 1).astype(np.uint8)
        centroids = get_mask_centroids(class_mask, class_id)

        if centroids: # not empty
            total_centroids += centroids

    return total_centroids

def get_data_from_centroids(centroids, image):

    if isinstance(image, torch.Tensor):
        image = image.cpu().numpy()

    total_data = []

    for centroid in centroids:

        data = image[centroid[1], centroid[0]]

        total_data.append(data)

    return np.array(total_data)

def aggregate_dense_sample(sample, intrinsics):

    # Sample contains the following keys: 
        # image, mask, quaternions, scales, xy, and z

    # Ensuring that the dataformats are channels_last
    sample = {k:set_image_data_format(v, "channels_last") for k,v in sample.items()}

    # The output should be an array of quaternions, translation vectors, and scales
    output = {
        'class_id': [],
        'instance_id': [],
        'instance_mask': [],
        'z': [],
        'xy': [],
        'quaternion': [],
        'RT': [],
        'scales': []
    }

    # Process data per class
    for class_id in np.unique(sample['mask']):

        # Ignore the background
        if class_id == 0:
            continue

        # Selecting the class mask
        class_mask = np.equal(sample['mask'], class_id) * 1

        # Then shatter the segmentation into instances
        instance_masks, num_of_instances = scipy.ndimage.label(class_mask)

        # Process data per instance
        for instance_id in range(1,num_of_instances+1):

            # Selecting the instance mask
            instance_mask = np.equal(instance_masks, instance_id) * 1

            # Obtaining the z
            z_img = np.where(instance_mask, sample['z'], 0)

            # Obtaining the pertaining xy
            xy_mask = np.dstack([instance_mask, instance_mask])
            xy_img = np.where(xy_mask, sample['xy'], 0)

            # Obtaining the pertaining scales
            scales_mask = np.concatenate([xy_mask, np.expand_dims(instance_mask, axis=-1)], axis=-1)
            scales_img = np.where(scales_mask, sample['scales'], 0)

            # Obtaning the pertaining quaternion
            quaternion_mask = np.concatenate([scales_mask, np.expand_dims(instance_mask, axis=-1)], axis=-1)
            quaternion_img = np.where(quaternion_mask, sample['quaternion'], 0)

            # For now use the naive average
            z = np.sum(z_img, axis=(0,1)) / np.sum(instance_mask)
            xy = np.sum(xy_img, axis=(0,1)) / np.sum(instance_mask)
            scales = np.sum(scales_img, axis=(0,1)) / np.sum(instance_mask)
            quaternion = np.sum(quaternion_img, axis=(0,1)) / np.sum(instance_mask)

            # Calculating the translation vector
            pixel_xy = xy.copy()

            # Converting image ratio to pixel location
            pixel_xy[0] = xy[1] * instance_mask.shape[1]
            pixel_xy[1] = xy[0] * instance_mask.shape[0]
            pixel_xy = pixel_xy.reshape((-1,1))

            # Creating the translation RT matrix
            translation_vector = create_translation_vector(pixel_xy, z, intrinsics)
            RT = quat_2_RT_given_T_in_world(quaternion, translation_vector)

            # Storing data
            output['class_id'].append(class_id)
            output['instance_id'].append(instance_id)
            output['instance_mask'].append(instance_mask)
            output['z'].append(z)
            output['xy'].append(pixel_xy)
            output['quaternion'].append(quaternion)
            output['RT'].append(RT)
            output['scales'].append(scales)

    return output

def find_matches(preds, gts, image_tag=None):
    """ 
    Match the predictions and ground truth data given the decomposed
    dense representations. 

    output = {
            'class_id': [],
            'instance_id': [],
            'instance_mask': [],
            'z': [],
            'xy': [],
            'quaternion': [],
            'RT': [],
            'scales': []
    }
    """

    pred_gt_matches = []

    taken_pred_ids = []

    # For all instances in the ground truth data
    for gt_id in range(len(gts['instance_id'])):

        # Match the ground truth and preds given the 2D IoU between the 
        # instance mask
        all_iou_2d = np.zeros((len(preds['instance_id'])))

        for pred_id in range(len(preds['instance_id'])):

            # If the pred id has been used, avoid reusing
            if pred_id in taken_pred_ids:
                all_iou_2d[pred_id] = -1
                continue

            # If the classes do not match, skip it!
            if gts['class_id'][gt_id] != preds['class_id'][pred_id]:
                continue
            
            # Calculating the 2d IoU
            iou_2d_mask = get_2d_iou(
                preds['instance_mask'][pred_id],
                gts['instance_mask'][gt_id]
            )

            # Storing the 2d IoU
            all_iou_2d[pred_id] = iou_2d_mask

        # If the maximum value is 0 or less (also -1), avoid it
        if np.max(all_iou_2d) <= 0:
            continue

        # Use the mask with the highest 2d iou score
        max_id = np.argmax(all_iou_2d)

        # Add it to the taken pred ids
        taken_pred_ids.append(max_id)

        # Store it into the matches
        match = {
            'class_id':  gts['class_id'][gt_id],
            'z': np.stack((gts['z'][gt_id], preds['z'][max_id])),
            'xy': np.stack((gts['xy'][gt_id], preds['xy'][max_id])),
            'quaternion': np.stack((gts['quaternion'][gt_id], preds['quaternion'][max_id])),
            'RT': np.stack((gts['RT'][gt_id], preds['RT'][max_id])),
            'scales': np.stack((gts['scales'][gt_id], preds['scales'][max_id])),
            'image_tag': image_tag
        }

        pred_gt_matches.append(match)

    return pred_gt_matches

#-------------------------------------------------------------------------------
# Get Raw Metrics

def get_2d_iou(mask_1, mask_2):

    intersection = np.sum(np.logical_and(mask_1, mask_2))
    union = np.sum(np.logical_or(mask_1, mask_2))
    return intersection / union

def get_asymmetric_3d_iou(RT_1, RT_2, scales_1, scales_2):

    noc_cube_1 = get_3d_bbox(scales_1, 0)
    bbox_3d_1 = transform_3d_camera_coords_to_3d_world_coords(noc_cube_1, RT_1)

    noc_cube_2 = get_3d_bbox(scales_2, 0)
    bbox_3d_2 = transform_3d_camera_coords_to_3d_world_coords(noc_cube_2, RT_2)

    bbox_1_max = np.amax(bbox_3d_1, axis=0)
    bbox_1_min = np.amin(bbox_3d_1, axis=0)
    bbox_2_max = np.amax(bbox_3d_2, axis=0)
    bbox_2_min = np.amin(bbox_3d_2, axis=0)

    overlap_min = np.maximum(bbox_1_min, bbox_2_min)
    overlap_max = np.minimum(bbox_1_max, bbox_2_max)

    # intersections and union
    if np.amin(overlap_max - overlap_min) <0:
        intersections = 0
    else:
        intersections = np.prod(overlap_max - overlap_min)
    union = np.prod(bbox_1_max - bbox_1_min) + np.prod(bbox_2_max - bbox_2_min) - intersections
    overlaps = intersections / union
    
    return overlaps

def get_symmetric_3d_iou(RT_1, RT_2, scales_1, scales_2):

    def y_rotation_matrix(theta):
            return np.array([[np.cos(theta), 0, np.sin(theta), 0],
                             [0, 1, 0 , 0], 
                             [-np.sin(theta), 0, np.cos(theta), 0],
                             [0, 0, 0 , 1]])

    n = 20
    max_iou = 0
    for i in range(n):
        rotated_RT_1 = RT_1 @ y_rotation_matrix(2*math.pi*i/float(n))
        max_iou = max(
            max_iou, 
            get_asymmetric_3d_iou(
                rotated_RT_1, 
                RT_2, 
                scales_1, 
                scales_2
            )
        )
        
    return max_iou

def get_3d_iou(RT_1, RT_2, scales_1, scales_2):

    #symmetry_flag = (class_name_1 in ['bottle', 'bowl', 'can'] and class_name_1 == class_name_2) or (class_name_1 == 'mug' and class_name_1 == class_name_2 and handle_visibility==0)
    symetry_flag = False
    if symetry_flag:
        return get_symmetric_3d_iou(RT_1, RT_2, scales_1, scales_2)
    else:
        return get_asymmetric_3d_iou(RT_1, RT_2, scales_1, scales_2)
    
def get_R_degree_error(quaternion_1, quaternion_2):
    
    rad = Quaternion.absolute_distance(
        Quaternion(quaternion_1),
        Quaternion(quaternion_2)
    )

    return np.rad2deg(rad)

def get_T_offset_error(center_3d_1, center_3d_2):
    diff = center_3d_1 - center_3d_2
    return np.sqrt(np.sum(np.power(diff,2)))

#-------------------------------------------------------------------------------
# Get Performance Metrics (AP, mAP)

def calculate_aps(
    cls_metrics,
    metrics_thresholds,
    metrics_operators
    ):

    aps = {}

    for metric_name, cls_metric in cls_metrics.items():

        # Obtain the range of the metric
        metric_operator = metrics_operators[metric_name]

        thresholds = metrics_thresholds[metric_name]
        metric_aps = np.zeros((len(cls_metric), thresholds.shape[0]))

        # For each class
        for i, cls_data in enumerate(cls_metric):

            # If there is not instances, skip it
            if not cls_data:
                continue
            
            # Convert the volatile list of data into an effective np.array
            np_cls_data = np.array(cls_data)
        
            # For each threshold, determine the average precision
            for j, threshold in enumerate(thresholds):
                
                # Calculating the average precision
                ap = np.sum(metric_operator(np_cls_data, threshold)) / np_cls_data.shape[0]

                # Storing the average precision
                metric_aps[i,j] = ap

        # Calculating the mean
        mean_aps = np.mean(metric_aps, axis=0)

        # Storing the mean
        metric_aps = np.concatenate((metric_aps, np.expand_dims(mean_aps,  axis=0)), axis=0)

        # Storing the metric aps into the overall aps
        aps[metric_name] = metric_aps

    return aps

#-------------------------------------------------------------------------------
# Pure Geometric Functions

def cartesian_2_homogeneous_coord(cartesian_coord):
    """
    Input: 
        Cartesian Vector/Matrix of size [N,M]
    Process:
        Transforms a cartesian vector/matrix into a homogeneous by appending ones
        as a new row to vector/matrix, making it a homogeneous vector/matrix.
    Output:
        Homogeneous Vector/Matrix of size [N+1, M]
    """

    homogeneous_coord = np.vstack([cartesian_coord, np.ones((1, cartesian_coord.shape[1]), dtype=cartesian_coord.dtype)])
    return homogeneous_coord

def homogeneous_2_cartesian_coord(homogeneous_coord):
    """
    Input: 
        Homogeneous Vector/Matrix of size [N,M]
    Process:
        Transforms a homogeneous vector/matrix into a cartesian by removing the 
        last row and dividing all the other rows by the last row, making it a 
        cartesian vector/matrix.
    Output:
        Cartesian Vector/Matrix of size [N-1, M]
    """

    cartesian_coord = homogeneous_coord[:-1, :] / homogeneous_coord[-1, :]
    return cartesian_coord

#-------------------------------------------------------------------------------
# RT Functions

def transform_2d_quantized_projections_to_3d_camera_coords(cartesian_projections_2d, RT, intrinsics, z):
    # Math comes from: https://robotacademy.net.au/lesson/image-formation/
    # and: https://stackoverflow.com/questions/12299870/computing-x-y-coordinate-3d-from-image-point
    """
    Input:
        cartesian projections: [2, N] (N number of 2D points)
        RT: [4, 4] Transformation matrix
        intrinsics: [3, 3] intrinsics parameters of the camera
        z: [1, N] (N number of 2D points)
    Process:
        Given the cartesian 2D projections, RT, intrinsics, and Z value @ the (x,y)
        locations of the projections, we do the following routine
            (1) Integrate the Z component into the projection, resulting in making
                the cartesian vector/matrix into a homogeneous vector/matrix
            (2) Convert the resulting homogeneous vector/matrix (which is in the 
                world coordinates space) into the camera coordinates space
            (3) Convert the homogeneous camera coordinates into cartesian camera
                coordinates.k
    Output:
        cartesian camera coordinates: [3, N] (N number of 3D points)
    """
    

    # Including the Z component into the projection (2D to 3D)
    cartesian_projections_2d = cartesian_projections_2d.astype(np.float32)
    cartesian_projections_2d[0, :] = cartesian_projections_2d[0, :] * (z/1000)
    cartesian_projections_2d[1, :] = cartesian_projections_2d[1, :] * (z/1000)
    
    homogeneous_projections_2d = np.vstack([cartesian_projections_2d, z/1000])

    cartesian_world_coordinates_3d = np.linalg.inv(intrinsics) @ homogeneous_projections_2d

    homogeneous_world_coordinates_3d = cartesian_2_homogeneous_coord(cartesian_world_coordinates_3d)

    homogeneous_camera_coordinates_3d = RT @ homogeneous_world_coordinates_3d

    cartesian_camera_coordinated_3d = homogeneous_2_cartesian_coord(homogeneous_camera_coordinates_3d)

    return cartesian_camera_coordinated_3d

def transform_3d_camera_coords_to_2d_quantized_projections(cartesian_camera_coordinates_3d, RT, intrinsics):
    # Math comes from: https://robotacademy.net.au/lesson/image-formation/
    """
    Input:
        cartesian camera coordinates: [3, N]
        RT: transformation matrix [4, 4]
        intrinsics: camera intrinsics parameters [3, 3]
    Process:
        Given the inputs, we do the following routine:
            (1) Convert cartesian camera coordinates into homogeneous camera
                coordinates.
            (2) Create the K matrix with the intrinsics.
            (3) Create the Camera matrix with the K matrix and RT.
            (4) Convert homogeneous camera coordinate directly to homogeneous 2D
                projections.
            (5) Convert homogeneous 2D projections into cartesian 2D projections.
    Output:
        cartesian_projections_2d [2, N]
    """

    # Converting cartesian 3D coordinates to homogeneous 3D coordinates
    homogeneous_camera_coordinates_3d = cartesian_2_homogeneous_coord(cartesian_camera_coordinates_3d)

    # Creating proper K matrix (including Pu, Pv, Uo, Vo, and f)
    K_matrix = np.hstack([intrinsics, np.zeros((intrinsics.shape[0], 1), dtype=np.float32)])

    """
    # METHOD 1
    # Creating camera matrix (including intrinsic and external paramaters)
    camera_matrix = K_matrix @ np.linalg.inv(RT)
    
    # Obtaining the homogeneous 2D projection
    homogeneous_projections_2d = camera_matrix @ homogeneous_camera_coordinates_3d
    print(f'homo projections: \n{homogeneous_projections_2d}\n')
    """
    
    # METHOD 2
    # Obtaining the homogeneous world coordinates 3d
    homogeneous_world_coordinates_3d = np.linalg.inv(RT) @ homogeneous_camera_coordinates_3d

    # Convert homogeneous world coordinates 3d to homogeneous 2d projections
    homogeneous_projections_2d = K_matrix @ homogeneous_world_coordinates_3d

    # Converting the homogeneous projection into a cartesian projection
    cartesian_projections_2d = homogeneous_2_cartesian_coord(homogeneous_projections_2d)

    # Converting projections from float32 into int32 (quantizing to from continuous to integers)
    cartesian_projections_2d = cartesian_projections_2d.astype(np.int32)

    # Transposing cartesian_projections_2d to have matrix in row major fashion
    cartesian_projections_2d = cartesian_projections_2d.transpose()

    return cartesian_projections_2d

def transform_3d_camera_coords_to_3d_world_coords(cartesian_camera_coordinates_3d, RT):
    """
    Input:
        cartesian camera coordinates: [3, N]
        RT: transformation matrix [4, 4]
    Process:
        Given the inputs, we do the following routine:
            (1) Convert cartesian camera coordinates into homogeneous camera
                coordinates.
            (4) Convert homogeneous camera coordinate directly to homogeneous 3D
                world coordinate by multiplying by the inverse of RT.
            (5) Convert homogeneous 3D world coordinates into cartesian 3D world coordinates.
    Output:
        cartesian_projections_2d [2, N]
    """

    # Converting cartesian 3D coordinates to homogeneous 3D coordinates
    homogeneous_camera_coordinates_3d = cartesian_2_homogeneous_coord(cartesian_camera_coordinates_3d)

    # Obtaining the homogeneous world coordinates 3d
    homogeneous_world_coordinates_3d = np.linalg.inv(RT) @ homogeneous_camera_coordinates_3d

    # Converting the homogeneous projection into a cartesian projection
    cartesian_world_coordinates_3d = homogeneous_2_cartesian_coord(homogeneous_world_coordinates_3d)

    return cartesian_world_coordinates_3d    

#-------------------------------------------------------------------------------
# Translation Vector Functions

def extract_z_from_RT(RT):

    z = (np.linalg.inv(RT)[2, 3] * 1000)

    return z

def extract_translation_vector_from_RT(RT, intrinsics):

    xyz_axis = 0.3*np.array([[0, 0, 0], [0, 0, 1], [0, 1, 0], [1, 0, 0]]).transpose()
    perfect_projected_axes = transform_3d_camera_coords_to_2d_quantized_projections(xyz_axis, RT, intrinsics)
    
    projected_origin = perfect_projected_axes[0,:].reshape((-1, 1))
    origin_z = extract_z_from_RT(RT)

    translation_vector = create_translation_vector(projected_origin, origin_z, intrinsics)

    return translation_vector

def create_translation_vector(cartesian_projections_2d_xy_origin, z, intrinsics):
    """
    Inputs: 
        cartesian projections @ (x,y) point: [2, 1]
        z, depth @ (x,y) point: [1,]
        intrinsics: camera intrinsics parameters: [3, 3]
    Process:
        Given the inputs, this function performs the following routine:
            (1) Includes the Z component into the cartesian projection, converting
                it into a homogeneous 2D projection.
            (2) With the inverse of the intrinsics matrix, the homogeneous 2D projection
                is transformed into the cartesian world coordinates
            (3) The translation vector is the cartesian world coordinates
            (4) Profit :)
    Output:
        translation vector: [3, 1]
    """

    # Including the Z component into the projection (2D to 3D)
    cartesian_projections_2d_xy_origin = cartesian_projections_2d_xy_origin.astype(np.float32)
    cartesian_projections_2d_xy_origin[0, :] = cartesian_projections_2d_xy_origin[0, :] * (z/1000)
    cartesian_projections_2d_xy_origin[1, :] = cartesian_projections_2d_xy_origin[1, :] * (z/1000)

    homogeneous_projections_2d_xyz_origin = np.vstack([cartesian_projections_2d_xy_origin, z/1000])

    # Converting projectins to world 3D coordinates
    cartesian_world_coordinates_3d_xyz_origin = np.linalg.inv(intrinsics) @ homogeneous_projections_2d_xyz_origin

    # The cartesian world coordinates of the origin are the translation vector
    translation_vector = cartesian_world_coordinates_3d_xyz_origin

    return translation_vector

def create_translation_vectors(centroids, zs, intrinsics):

    translation_vectors = []

    for i, centroid in enumerate(centroids):

        z = zs[i]
        formatted_centroid = centroid.reshape((-1, 1))

        translation_vector = create_translation_vector(formatted_centroid, z, intrinsics)
        translation_vectors.append(translation_vector)

    return translation_vectors

def overwrite_RTs_translation_vector(translation_vector, RT):

    #"""
    inv_RT = np.linalg.inv(RT)
    inv_RT[:-1, -1] = translation_vector.flatten()
    new_RT = np.linalg.inv(inv_RT)
    #"""

    #RT[:-1, -1] = translation_vector.flatten()
    #new_RT = RT
    
    return new_RT

#-------------------------------------------------------------------------------
# RT-Quaternion Functions

def RT_2_quat(RT, normalize=True):
    """
    Inputs:
        RT, transformation matrix: [4, 4]
    Process:
        Given the inputs, this function does the following routine:
            (1) Normalize the RT to reduce as much as possible scaling errors
                generated by the orthogonalization of the rotation matrix
            (2) Inputing the rotation matrix into the scipy.spatial.transform.Rotation
                object and performing orthogonalization (determinant = +1).
            (3) Constructing the quaternion with the scipy...Rotation object
            (4) Taking the translation vector as output to not lose information
    Output:
        quaternion: [1, 4]
        translation vector: [3, 1]
        normalizing factor: [1,]
    """

    # normalizing
    if normalize:
        normalizing_factor = np.amax(RT)
        RT[:3, :] = RT[:3, :] / normalizing_factor
    else:
        normalizing_factor = 1

    # Rotation Matrix
    rotation_matrix = RT[:3, :3]

    smart_rotation_object = scipy.spatial.transform.Rotation.from_matrix(rotation_matrix)

    quaternion = smart_rotation_object.as_quat()

    # Translation Matrix
    translation_vector = RT[:3, -1].reshape((-1, 1))

    return quaternion, translation_vector, normalizing_factor

def quat_2_RT_given_T_in_camera(quaternion, translation_vector):
    """
    Inputs:
        quaternion: [1, 4]
        translation_vector, cartesian camera coordinates space: [3, 1]
    Process:
        Given the inputs, this function does the following routine:
            (1) By feeding the quaternion to the scipy...Rotation object, the
                quaternion is converted into a rotation matrix
            (2) Reconstruct RT joint matrix by joining rotation and translation 
                matrices and adding the [0,0,0,1] section of the matrix. 
    Output:
        RT: transformation matrix: [4, 4]
    """

    smart_rotation_object = scipy.spatial.transform.Rotation.from_quat(quaternion)
    rotation_matrix = smart_rotation_object.as_matrix()
    
    RT = np.vstack([np.hstack([rotation_matrix, translation_vector]), [0,0,0,1]])

    return RT

def quat_2_RT_given_T_in_world(quaternion, translation_vector):
    """
    Inputs:
        quaternion: [1, 4]
        translation_vector, cartesian world coordinates space: [3, 1]
    Process:
        Given the inputs, this function does the following:
            (1) Convert the quaternion into a rotation matrix with scipy...Rotation.
            (2) Inverting the rotation matrix to match the translation vector's 
                coordinate space.
            (3) Joining the inverse rotation matrix with the translation vector 
                and adding [0,0,0,1] to the matrix.
            (4) Returning the inverse RT matrix into RT.
    Output:
        RT: [4,4]
    """

    try:
        smart_rotation_object = scipy.spatial.transform.Rotation.from_quat(quaternion)
    except ValueError: # zero norm quaternion 
        smart_rotation_object = scipy.spatial.transform.Rotation.from_quat(np.array([1,0,0,0]))
    
    rotation_matrix = smart_rotation_object.as_matrix()
    inv_rotation_matrix = np.linalg.inv(rotation_matrix)

    inv_RT =  np.vstack([np.hstack([inv_rotation_matrix, translation_vector]), [0,0,0,1]])
    RT = np.linalg.inv(inv_RT)

    return RT

def quats_2_RTs_given_Ts_in_world(quaternions, translation_vectors):

    RTs = {}
    print(f'quaternions: {quaternions}')
    print(f'translation_vectors: {translation_vectors}')

    for class_id, class_quaternions in quaternions.items():

        class_RTs = []

        for instance_id, quaternion in enumerate(class_quaternions):

            translation_vector = translation_vectors[class_id][instance_id]

            RT = quat_2_RT_given_T_in_world(quaternion, translation_vector)
            class_RTs.append(RT)

        if class_RTs: # if not empty
            RTs[class_id] = class_RTs

    return RTs

#-------------------------------------------------------------------------------
# Error Correction

def get_new_RT_error(perfect_projected_pts, quat_projected_pts):
    """
    Inputs:
        perfect_projected_pts: [N, 2]
        quat_projected_pts: [N, 2]
    Process:
        Given the inputs, this function does the following routine:
            (1) Obtain the difference between the perfect and quaternion projected
                axes points.
            (2) Calculate overall error by taking the sum of the absolute value 
                of the difference.
    Output:
        error: error of the quat_projected_pts compared to the perfect_projected_pts,
               size = scalar
    """

    #print(f'\nperfect_projected_pts: \n{perfect_projected_pts}\n')
    #print(f'quat_projected_pts: \n{quat_projected_pts}\n')

    diff_pts = quat_projected_pts - perfect_projected_pts
    #print(f'diff_pts: \n{diff_pts}\n')

    error = np.average(np.absolute(diff_pts))
    #print(f'error: {error}')

    return error

def fix_quaternion_T(intrinsics, original_RT, quat_RT, normalizing_factor):
    """
    Inputs:
        intrinsics: camera intrinsics parameters: [3, 3]
        original_RT: original transformation matrix: [4,4]
        quat_RT: quaternion-generated transformation matrix: [4, 4]
        normalizing_factor: normalizing factor used to do the RT2Quat conversion: scalar
    Process:
        Given the inputs, this function does the following routine:
            (1) Calculate the perfect_projected_axes
            (2) Determines the baseline error between the quaternion-based projected
                axes and the perfect projected axes
            (3) Inside a for loop, determine which axes (x,y,z) and which direction
                along each axes (+, -) will result in a reduce error.
            (4) Keep iterating until either (a) the error is reduced below a
                threshold of 1 or (b) 50 iterations are completed.
    Output:
        quat_RT, hopefully a fixed RT: [4,4]
    """

    # Creating a xyz axis and converting 3D coordinates into 2D projections
    xyz_axis = 0.3*np.array([[0, 0, 0], [0, 0, 1], [0, 1, 0], [1, 0, 0]]).transpose()
    perfect_projected_axes = transform_3d_camera_coords_to_2d_quantized_projections(xyz_axis, original_RT, intrinsics)

    norm_xyz_axis = xyz_axis / normalizing_factor
    quat_projected_axes = transform_3d_camera_coords_to_2d_quantized_projections(norm_xyz_axis, quat_RT, intrinsics)

    baseline_error = get_new_RT_error(perfect_projected_axes, quat_projected_axes)

    step_size = 0.000001 # 0.01
    test_RT = quat_RT.copy()
    min_error = baseline_error
    past_error = np.zeros((3, 2), np.int)

    for i in range(100): # only 100 iterations allowed

        errors = np.zeros((3, 2), np.int)

        # try each dimension
        for xyz in range(3):

            # try both positive and negative addition
            for operation_index, operation in enumerate([add, subtract]):

                test_RT[xyz,-1] = operation(quat_RT[xyz,-1], step_size)
                test_projected_axes = transform_3d_camera_coords_to_2d_quantized_projections(norm_xyz_axis, test_RT, intrinsics)

                error = get_new_RT_error(perfect_projected_axes, test_projected_axes)
                errors[xyz, operation_index] = error

        # Process error
        if np.amin(errors) > baseline_error:
            step_size /= 10
            print('divided by 10')

        elif (past_error == errors).all():
            print(f'past_error: {past_error}')
            step_size *= 10
            print('multiplied by 10')

        else:
            # perform the action that reduces the error the most
            min_error = np.amin(errors)
            min_error_index = divmod(errors.argmin(), errors.shape[1])
            min_xyz, operation = min_error_index[0], add if min_error_index[1] == 0 else subtract
            quat_RT[min_xyz, -1] = operation(quat_RT[min_xyz,-1], step_size)
            print(f'new min_error: {min_error}')

        if min_error <= 1:
            print('min_error of <= 1 achieved!')
            break
        else:
            past_error = errors.copy()

    return quat_RT

def fix_quat_RT_matrix(intrinsics, original_RT, quat_RT, pts=None):

    # Creating an xyz axis and converting 3D coordinates into 2D projections
    if isinstance(pts, type(None)):
        pts = 0.3*np.array([[0, 0, 0], [0, 0, 1], [0, 1, 0], [1, 0, 0]]).transpose()
    
    perfect_projected_pts = transform_3d_camera_coords_to_2d_quantized_projections(pts, original_RT, intrinsics)
    quat_projected_pts = transform_3d_camera_coords_to_2d_quantized_projections(pts, quat_RT, intrinsics)

    # Noting the base error
    baseline_error = get_new_RT_error(perfect_projected_pts, quat_projected_pts)
    min_error = baseline_error
    print(f'baseline_error: {baseline_error}')

    #return quat_RT

    # Getting the quaternion version of the quat_RT affect it
    quat = Quaternion(matrix=quat_RT)
    
    # Fixing hyper-parameters
    angles = np.array([0.001,0.005,0.01,0.05,0.1,0.25,0.5,1,2,3]) * (math.pi/180)
    axes = [[1,0,0],[0,1,0],[0,0,1]]
    direction = [1,-1]

    for i in range(5):

        errors = np.zeros((len(angles),len(axes),len(direction)), np.int)

        for an_id, angle in enumerate(angles):

            for ax_id, axis in enumerate(axes): # x, y, z

                for di_id, di_coefficient in enumerate(direction): # +1, and -1 degrees 

                    # Get this test's quat, RT, and projected axes
                    temp_quat = quat * Quaternion(axis=axis, angle=angle*di_coefficient)
                    temp_RT = temp_quat.transformation_matrix
                    temp_RT[:,-1] = quat_RT[:,-1]
                    test_projected_pts = transform_3d_camera_coords_to_2d_quantized_projections(pts, temp_RT, intrinsics)

                    # Get the error
                    error = get_new_RT_error(perfect_projected_pts, test_projected_pts)
                    errors[an_id, ax_id, di_id] = error
                

        #print(errors)

        # Get error information
        temp_min_error = np.amin(errors)
        min_angle_id, min_error_residue = divmod(errors.argmin(), errors.shape[1] * errors.shape[2])
        min_axis_id, min_direction_id = divmod(min_error_residue, errors.shape[2])
        min_error_index = (min_angle_id, min_axis_id, min_direction_id)

        if temp_min_error < min_error:
            min_error = temp_min_error
            print('new min_error:', min_error)
            quat *= Quaternion(axis=axes[min_axis_id], angle=angles[min_angle_id]*direction[min_direction_id])

        else:
            temp_RT = temp_quat.transformation_matrix
            temp_RT[:,-1] = quat_RT[:,-1]
            quat_RT = temp_RT
            break

    return quat_RT

#-------------------------------------------------------------------------------
# Quaternion Functions