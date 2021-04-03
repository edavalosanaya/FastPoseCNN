import os
import sys
import gc

import numpy as np
import scipy.ndimage

import torch
import torch.nn as nn
import torch.utils.dlpack

import cupy as cp
import cupyx as cpx
import cupyx.scipy.ndimage

# Local imports
sys.path.append(os.getenv("TOOLS_DIR"))

try:
    import visualize as vz
except ImportError:
    pass

import type_hinting as th

#-------------------------------------------------------------------------------
# Helper Functions

def freeze(dict_of_params):
    
    for param in dict_of_params.parameters():
            param.requires_grad = False

#-------------------------------------------------------------------------------
# Native PyTorch Functions

def normalize(data, dim):

    # Determine the norm along that dimension
    norm_data = data.norm(dim=dim, keepdim=True)

    # Replace zero norm by one to avoid division by zero error
    safe_norm_data = torch.where(
        norm_data != 0, norm_data.float(), torch.tensor(1.0, device=norm_data.device).float()
    )

    # Normalize data
    normalized_data = data / safe_norm_data

    return normalized_data

def class_compress(
    num_of_classes: int, 
    cat_mask: torch.Tensor, 
    logits: th.LogitData) -> th.CategoricalData:

    # Output:
    class_compress_logits = {}

    # Obtaining size information from cat_mask
    b, h, w = cat_mask.shape

    # Creating class masks
    class_masks = torch.zeros((b, num_of_classes, h, w), device=cat_mask.device)
    class_masks = class_masks.scatter(1, torch.unsqueeze(cat_mask, dim=1), 1)[:,1:]

    # Creating class data for the logits
    class_chunks_logits = {k:torch.stack(torch.chunk(v, num_of_classes-1, dim=1), dim=1) for k,v in logits.items() if k != 'mask'}

    # Perform class compression on the logits for pixel-wise regression
    for logit_key in logits.keys():

        # If dealing with the mask logits, skip it
        if logit_key == 'mask':
            continue

        # Applying the class masks to the logits
        masked_class_chunk = torch.where(
            torch.unsqueeze(class_masks, dim=2).bool(), 
            class_chunks_logits[logit_key].double(),
            0.0
        ).float()

        # Compress the classes into one
        class_compress_logit = torch.sum(masked_class_chunk, dim=1)

        # Need to squeeze when logit_key == z in dim = 1 to match categorical
        # ground truth data
        if logit_key == 'z':
            class_compress_logit = torch.squeeze(class_compress_logit, dim=1)

        # Normalize quaternion and xy
        elif logit_key == 'quaternion' or logit_key == 'xy':
            class_compress_logit = normalize(class_compress_logit, dim=1)

        # Store data
        class_compress_logits[logit_key] = class_compress_logit
    
    return class_compress_logits

#-------------------------------------------------------------------------------
# Camera/World Transformation Functions

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

    homogeneous_coord = torch.vstack([cartesian_coord, torch.ones((1, cartesian_coord.shape[1]), device=cartesian_coord.device, dtype=cartesian_coord.dtype)])
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

def create_translation_vector(pixel_xy, exp_z, intrinsics):

    # Including the Z component into the projection (2D to 3D)
    projected_xy = pixel_xy.clone()
    projected_xy = projected_xy * (exp_z/1000)

    # Converting xy and z to xyz in homogenous form
    homogenous_xyz = torch.vstack([projected_xy, exp_z.clone()/1000])

    # Convert projections to world 3D coordinates
    translation_vector = torch.inverse(intrinsics.to(pixel_xy.device)) @ homogenous_xyz

    return translation_vector

def quat_2_RT_given_T_in_world(q, T):

    # First ensure that the quaternion is normalized
    norm = q.norm()
    if norm > 0:
        q = q / norm
    
    # Convert quaternion to rotation matrix
    R = quat_2_rotation_matrix(q)

    # Modifying the rotation matrix to work
    R = torch.rot90(R, 2).T
    x = torch.tensor([
        [1,-1,1],
        [1,-1,1],
        [-1,1,-1]
    ], device=R.device, dtype=R.dtype)
    R = torch.mul(R, x)

    # Then invert the rotation matrix
    inv_R = torch.inverse(R)

    # Then combine to generate the inverse transformation matrix
    inv_RT = torch.vstack([torch.hstack([inv_R, T]), torch.tensor([0,0,0,1], device=q.device, dtype=q.dtype)])

    # Then undo the inverse to obtain the correct transformation matrix
    RT = torch.inverse(inv_RT)

    return RT

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
    homogeneous_world_coordinates_3d = torch.inverse(RT) @ homogeneous_camera_coordinates_3d

    # Converting the homogeneous projection into a cartesian projection
    cartesian_world_coordinates_3d = homogeneous_2_cartesian_coord(homogeneous_world_coordinates_3d)

    return cartesian_world_coordinates_3d 

def batchwise_get_RT(q, xys, exp_zs, inv_intrinsics):

    # q = quaternion
    # Including the Z component into the projection (2D to 3D)
    projected_xys = xys * (exp_zs/1000)

    # First construct the translation vector matrix
    homogenous_xyzs = torch.vstack([projected_xys.T, exp_zs.T/1000])
    T = inv_intrinsics @ homogenous_xyzs # torch.inverse(intrinsics) @ homo_xyz

    # Then create R
    norm = q.norm(dim=1)
    safe_norm = torch.where(norm > 0, norm, torch.ones_like(norm, device=q.device))
    q = q / torch.unsqueeze(safe_norm, dim=1)

    # Creating container for the all R
    R = torch.zeros((q.shape[0], 3, 3), device=q.device, dtype=q.dtype)

    # Correction matrix
    x = torch.tensor([
        [1,-1,1],
        [1,-1,1],
        [-1,1,-1]
    ], device=q.device, dtype=q.dtype)

    # Processing each quaternion individually (too hard)
    for i in range(q.shape[0]):
        r = quat_2_rotation_matrix(q[i])
        R[i] = torch.mul(torch.rot90(r, 2).T, x)
    
    # Need the inverse version to combine it with the translation
    inv_R = torch.inverse(R)

    # Then combining it to the translation vector
    inv_RT = torch.cat(
        [
            torch.cat([inv_R, torch.unsqueeze(T.T, dim=-1)], dim=-1), 
            torch.tensor([0,0,0,1], device=q.device, dtype=q.dtype).expand((q.shape[0],1,4))
        ], dim=1)

    # Inversing it to get the actual RT
    RT = torch.inverse(inv_RT)

    return R, T.t(), RT

def samplewise_get_RT(agg_data, inv_intrinsics):

    # Once all the raw data has been aggregated, we need to calculate the 
    # rotation matrix of each instance.
    R_data, T_data, RT_data = batchwise_get_RT(
        agg_data['quaternion'],
        agg_data['xy'],
        agg_data['z'],
        inv_intrinsics
    )

    # Storing generated RT
    agg_data['R'] = R_data
    agg_data['T'] = T_data
    agg_data['RT'] = RT_data

    return agg_data

#-------------------------------------------------------------------------------
# Generative/Conversion Functions

def quat_2_rotation_matrix(quaternion):
    # Code translated from here:
    #https://github.com/KieranWynn/pyquaternion/blob/99025c17bab1c55265d61add13375433b35251af/pyquaternion/quaternion.py#L981

    # Then perform the dot product between the q matrix and the q bar matrix
    q_matrix = get_q_matrix(quaternion)
    q_bar_matrix = get_q_bar_matrix(quaternion)
    product = torch.mm(q_matrix, torch.conj(q_bar_matrix).T)

    # Selecting the right components of the product creates the rotation matrix
    R = product[1:][:,1:]

    return R

def get_q_matrix(q):

    return torch.tensor([
        [q[0], -q[1], -q[2], -q[3]],
        [q[1],  q[0], -q[3],  q[2]],
        [q[2],  q[3],  q[0], -q[1]],
        [q[3], -q[2],  q[1],  q[0]]
    ], device=q.device, dtype=q.dtype)

def get_q_bar_matrix(q):

    return torch.tensor([
        [q[0], -q[1], -q[2], -q[3]],
        [q[1],  q[0],  q[3], -q[2]],
        [q[2], -q[3],  q[0],  q[1]],
        [q[3],  q[2], -q[1],  q[0]]
    ], device=q.device, dtype=q.dtype)

def quat_2_rotation_matrix2(q):

    qw, qx, qy, qz = q

    return torch.tensor([
        [1-2*(torch.pow(qy,2) - torch.pow(qz,2)), 2*(qx*qy - qz*qw)                      , 2*(qx*qz + qy*qw)],
        [2*(qx*qy + qz*qw)                      , 1-2*(torch.pow(qx,2) - torch.pow(qz,2)), 2*(qy*qz - qx*qw)],
        [2*(qx*qz - qy*qw)                      , 2*(qy*qz + qx*qw)                      , 1 - 2*(torch.pow(qx,2) - torch.pow(qy,2))]
    ], device=q.device, dtype=q.dtype)

def get_3d_bbox(scale, shift = 0):
    """
    Input: 
        scale: [3] or scalar
        shift: [3] or scalar
    Return 
        bbox_3d: [3, N]

    """
    # if hasattr(scale, "__iter__"):
    #     bbox_3d = torch.tensor([
    #         [scale[0] / 2, scale[1] / 2, scale[2] / 2],
    #         [scale[0] / 2, scale[1] / 2, -scale[2] / 2],
    #         [-scale[0] / 2, scale[1] / 2, scale[2] / 2],
    #         [-scale[0] / 2, scale[1] / 2, -scale[2] / 2],
    #         [scale[0] / 2, -scale[1] / 2, scale[2] / 2],
    #         [scale[0] / 2, -scale[1] / 2, -scale[2] / 2],
    #         [-scale[0] / 2, -scale[1] / 2, scale[2] / 2],
    #         [-scale[0] / 2, -scale[1] / 2, -scale[2] / 2]], device=scale.device, dtype=scale.dtype)
    # else:
    #     bbox_3d = torch.tensor([
    #         [scale / 2, scale / 2, scale / 2],
    #         [scale / 2, scale / 2, -scale / 2],
    #         [-scale / 2, scale / 2, scale / 2],
    #         [-scale / 2, scale / 2, -scale / 2],
    #         [scale / 2, -scale / 2, scale / 2],
    #         [scale / 2, -scale / 2, -scale / 2],
    #         [-scale / 2, -scale / 2, scale / 2],
    #         [-scale / 2, -scale / 2, -scale / 2]], device=scale.device, dtype=scale.dtype)

    unit_bbox_3d = torch.tensor([
        [1, 1, 1],
        [1, 1, -1],
        [-1, 1, 1],
        [-1, 1, -1],
        [1, -1, 1],
        [1, -1, -1],
        [-1, -1, 1],
        [-1, -1, -1]
    ], device=scale.device, dtype=scale.dtype) / 2
    
    e_scales = torch.unsqueeze(scale, axis=0)

    bbox_3d = unit_bbox_3d * e_scales

    bbox_3d += shift
    bbox_3d = bbox_3d.T
    return bbox_3d 

#-------------------------------------------------------------------------------
# Comparison Functions

def torch_get_2d_iou(tensor1: torch.Tensor, tensor2: torch.Tensor) -> torch.Tensor:

    intersection = torch.sum(torch.logical_and(tensor1, tensor2))
    union = torch.sum(torch.logical_or(tensor1, tensor2))
    return torch.true_divide(intersection,union)

def batchwise_get_2d_iou(batch_masks1, batch_masks2):

    # Number of masks
    n_of_m1, h, w = batch_masks1.shape
    n_of_m2 = batch_masks2.shape[0]

    # Expand the masks to match size
    expanded_b_masks1 = torch.unsqueeze(batch_masks1, dim=1).expand((n_of_m1, n_of_m2, h, w))
    expanded_b_masks2 = batch_masks2.expand((n_of_m1, n_of_m2, h, w))

    # Calculating the intersection and union
    intersection = torch.sum(
        torch.logical_and(expanded_b_masks1, expanded_b_masks2),
        dim=(2,3)
    )
    union = torch.sum(
        torch.logical_or(expanded_b_masks1, expanded_b_masks2),
        dim=(2,3)
    )

    # Calculating iou
    iou = intersection / union
    
    return iou

def get_quat_distance(q0, q1, symmetric_ids=None):

    # If symmetric ids are given, then use them

    if type(symmetric_ids) != type(None):
        # Separate symmetric and non-symmetric items
        non_sym_i = torch.where(symmetric_ids == 0)[0]
        sym_i = torch.where(symmetric_ids != 0)[0]

        # Process the non-symmetric instances easily
        non_sym_distances = get_raw_quat_distance(q0[non_sym_i], q1[non_sym_i])
        sym_distances = get_symmetric_quat_distance(q0[sym_i], q1[sym_i])

        # Getting all the distances together
        distances = torch.cat((non_sym_distances, sym_distances), dim=0)

        # Removing any nans
        clean_distances = distances[torch.isnan(distances) == False]

        return clean_distances

    else:    
        return get_raw_quat_distance(q0, q1)

def get_raw_quat_distance(q0, q1):

    # Catching zero data case
    if q0.shape[0] == 0:
        return torch.tensor([float('nan')], device=q0.device)

    # Determine the difference
    q0_minus_q1 = q0 - q1
    q0_plus_q1  = q0 + q1
    
    # Obtain the norm
    d_minus = q0_minus_q1.norm(dim=-1)
    d_plus  = q0_plus_q1.norm(dim=-1)

    # Compare the norms and select the one with the smallest norm
    ds = torch.stack((d_minus, d_plus))
    rad_distance = torch.min(ds, dim=0).values

    # Converting the rad to degree
    degree_distance = torch.rad2deg(rad_distance)

    return degree_distance

def get_symmetric_quat_distance(q0, q1):

    # If already constructed, simply use for calculating the best distance

    # Catching zero data case
    if q0.shape[0] == 0:
        return torch.tensor([float('nan')], device=q0.device)

    # Expanding q1 to account for 0-360 degrees of rotation to account for z-axis
    # symmetric and expanding the q0 to match its size for later comparison
    rot_e_q1, e_q0 = quat_symmetric_tf(q1, q0)

    # Calculating the distance
    distance = get_raw_quat_distance(e_q0, rot_e_q1)

    # Determing the best distance
    best_distance = torch.min(distance, dim=-1).values

    return best_distance

def y_rotation_matrix(theta):
    return torch.tensor([
        [torch.cos(theta), 0, torch.sin(theta), 0],
        [0, 1, 0, 0],
        [-torch.sin(theta), 0, torch.cos(theta), 0],
        [0, 0, 0, 1]
    ], device=theta.device, dtype=theta.dtype)

def get_symmetric_3d_iou(RT_1, RT_2, scales_1, scales_2):

    n = 20
    max_iou = 0
    for i in range(n):
        theta = torch.tensor(2*np.pi*i/float(n), device=RT_1.device, dtype=RT_1.dtype)
        rotated_RT1 = RT_1 @ y_rotation_matrix(theta)
        max_iou = max(
            max_iou,
            get_asymmetric_3d_iou(
                rotated_RT1,
                RT_2,
                scales_1,
                scales_2
            )
        )

def get_asymmetric_3d_iou(RT_1, RT_2, scales_1, scales_2):
    
    noc_cube_1 = get_3d_bbox(scales_1, 0)
    bbox_3d_1 = transform_3d_camera_coords_to_3d_world_coords(noc_cube_1, RT_1)

    noc_cube_2 = get_3d_bbox(scales_2, 0)
    bbox_3d_2 = transform_3d_camera_coords_to_3d_world_coords(noc_cube_2, RT_2)

    bbox_1_max = torch.amax(bbox_3d_1, dim=0)
    bbox_1_min = torch.amin(bbox_3d_1, dim=0)
    bbox_2_max = torch.amax(bbox_3d_2, dim=0)
    bbox_2_min = torch.amin(bbox_3d_2, dim=0)

    overlap_min = torch.maximum(bbox_1_min, bbox_2_min)
    overlap_max = torch.minimum(bbox_1_max, bbox_2_max)

    # intersections and union
    if torch.amin(overlap_max - overlap_min) <0:
        intersections = 0
    else:
        intersections = torch.prod(overlap_max - overlap_min)
    
    union = torch.prod(bbox_1_max - bbox_1_min) + torch.prod(bbox_2_max - bbox_2_min) - intersections
    iou_3d = intersections / union
    
    return iou_3d

def get_3d_iou(RT_1, RT_2, scales_1, scales_2):

    symetry_flag = False
    if symetry_flag:
        return get_symmetric_3d_iou(RT_1, RT_2, scales_1, scales_2)
    else:
        return get_asymmetric_3d_iou(RT_1, RT_2, scales_1, scales_2)

def get_3d_ious(RTs_1, RTs_2, scales_1, scales_2):

    # For the number of entries, find the 3d_iou
    ious_3d = []
    for i in range(RTs_1.shape[0]):
        ious_3d.append(get_3d_iou(RTs_1[i], RTs_2[i], scales_1[i], scales_2[i]))

    ious_3d = torch.stack(ious_3d)

    return ious_3d

def get_T_offset_errors(centers3d_1, centers3d_2):
    
    offset_errors = []
    for i in range(centers3d_1.shape[0]):
        offset_error = get_offset_error_from_centroids(centers3d_1[i], centers3d_2[i])
        offset_errors.append(offset_error)

    offset_errors = torch.stack(offset_errors)
    return offset_errors

def get_offset_error_from_centroid(center3d_1, center3d_2):
    diff = center3d_1 - center3d_2
    return torch.sqrt(torch.sum(torch.pow(diff,2)))

def from_Ts_get_offset_error(gt_Ts, pred_Ts):

    return torch.linalg.norm(gt_Ts - pred_Ts, dim=1) * 10

def from_RTs_get_T_offset_errors(gt_RTs, pred_RTs):

    # Creating the value for the camera 3d center for later computations
    camera_coord_3d_center = torch.tensor(
        [[0,0,0]], 
        device=gt_RTs.device,
        dtype=gt_RTs.dtype
    ).T

    # Calculating the world centers of the objects (gt and preds)
    # per RT
    gt_world_coord_3d_centers = []
    pred_world_coord_3d_centers = []
    
    for i in range(gt_RTs.shape[0]):

        gt_world_coord_3d_center = transform_3d_camera_coords_to_3d_world_coords(
            camera_coord_3d_center,
            gt_RTs[i]
        )

        pred_world_coord_3d_center = transform_3d_camera_coords_to_3d_world_coords(
            camera_coord_3d_center,
            pred_RTs[i]
        )
        
        gt_world_coord_3d_centers.append(gt_world_coord_3d_center)
        pred_world_coord_3d_centers.append(pred_world_coord_3d_center)

    # Combinding all the 3d centers
    gt_world_coord_3d_centers = [x.flatten() for x in gt_world_coord_3d_centers]
    pred_world_coord_3d_centers = [x.flatten() for x in pred_world_coord_3d_centers]

    gt_world_coord_3d_centers = torch.stack(gt_world_coord_3d_centers)
    pred_world_coord_3d_centers = torch.stack(pred_world_coord_3d_centers)

    # Calculating the distance between the gt and pred points
    offset_errors = get_offset_error_from_centroid(
        gt_world_coord_3d_centers,
        pred_world_coord_3d_centers
    ) * 10 # Converting units

    return offset_errors

def calculate_aps(raw_data, metrics_threshold, metrics_operator):

    # Container for all aps
    aps = {}

    # Iteration over the metrics
    for data_key in raw_data.keys():

        # Creating container for metrics values
        aps[data_key] = {}

        # Select metrics information
        data = raw_data[data_key]
        thresholds = metrics_threshold[data_key]
        operator = metrics_operator[data_key]

        # Determine number of thresholds for later operations
        n_of_t = thresholds.shape[0]

        # Iterating over the classes
        for class_id in data.keys():

            # Selecting the class' data
            class_data = data[class_id]

            # Remove Nans from the calculations
            class_data = class_data[torch.isnan(class_data) == False]
            n_of_d = class_data.shape[0]

            # Expanding data and thresholds to single-handed compute the metric value
            expanded_class_data = torch.unsqueeze(class_data, dim=0).expand(n_of_t, n_of_d)
            expanded_thresholds = torch.unsqueeze(thresholds, dim=1).expand(n_of_t, n_of_d)

            # Applying operator
            class_ap = operator(expanded_class_data, expanded_thresholds)

            # Collapse result to the sizeof the thresholds
            class_ap = torch.sum(class_ap, dim=1) / n_of_d

            # Storing class aps
            aps[data_key][class_id] = class_ap

        # Calculating mean for class
        aps[data_key]['mean'] = torch.mean(torch.stack(list(aps[data_key].values())).float(), dim=0)

    return aps

def calculate_complex_aps(raw_data, metrics_threshold, metrics_operator):

    # Conatiner for all aps
    aps = {}

    # Iteration over the metrics
    for data_key in metrics_threshold.keys():

        # Creating subcontainer for metrics values
        aps[data_key] = {}

        # Select metrics information
        thresholds = metrics_threshold[data_key]
        keys = [k for k in raw_data.keys() if k in data_key]
        
        # Concatinating data sets
        data = {}
        for key in keys:
            for class_id in raw_data[key].keys():
                if class_id in list(data.keys()):
                    data[class_id] = torch.stack(
                        (data[class_id], raw_data[key][class_id])
                    )
                else:
                    data[class_id] = raw_data[key][class_id]

        # Iterating over the classes
        for class_id in data.keys():

            # Selecting the class' data
            class_data = data[class_id]

            # Remove Nans from the calculations
            # class_data = class_data[torch.isnan(class_data) == False]
            n_of_d = class_data.shape[1]
            
            # Expanding data to make comparison easy
            e_class_data = torch.unsqueeze(class_data, dim=1)
            e_threshold = torch.unsqueeze(thresholds, dim=-1)

            # Applying operator (currently only using less than)
            applied_threshold = torch.less(e_class_data, e_threshold)

            # Collapsing the metrics to a single value per sample
            threshold_mixed = (torch.sum(applied_threshold, dim=0) == applied_threshold.shape[0]).bool()

            # Futher collapsing now all the samples into a single class value
            class_ap = torch.sum(threshold_mixed, dim=1) / n_of_d

            # Storing class aps
            aps[data_key][class_id] = class_ap

        # Calculating mean for class
        aps[data_key]['mean'] = torch.mean(torch.stack(list(aps[data_key].values())).float(), dim=0)

    return aps

#-------------------------------------------------------------------------------
# Operations 

def quaternion_raw_multiply(a, b):
    # Take from https://github.com/facebookresearch/pytorch3d/blob/e13e63a811438c250c1760cbcbcbe6c034a8570d/pytorch3d/transforms/rotation_conversions.py#L339
    """
    Multiply two quaternions.
    Usual torch rules for broadcasting apply.
    Args:
        a: Quaternions as tensor of shape (..., 4), real part first.
        b: Quaternions as tensor of shape (..., 4), real part first.
    Returns:
        The product of a and b, a tensor of quaternions shape (..., 4).
    """
    aw, ax, ay, az = torch.unbind(a, -1)
    bw, bx, by, bz = torch.unbind(b, -1)
    ow = aw * bw - ax * bx - ay * by - az * bz
    ox = aw * bx + ax * bw + ay * bz - az * by
    oy = aw * by - ax * bz + ay * bw + az * bx
    oz = aw * bz + ax * by - ay * bx + az * bw
    return torch.stack((ow, ox, oy, oz), -1)

def quaternion_multiply(a, b):
    # Take from https://github.com/facebookresearch/pytorch3d/blob/e13e63a811438c250c1760cbcbcbe6c034a8570d/pytorch3d/transforms/rotation_conversions.py#L339
    """
    Multiply two quaternions representing rotations, returning the quaternion
    representing their composition, i.e. the versor with nonnegative real part.
    Usual torch rules for broadcasting apply.
    Args:
        a: Quaternions as tensor of shape (..., 4), real part first.
        b: Quaternions as tensor of shape (..., 4), real part first.
    Returns:
        The product of a and b, a tensor of quaternions of shape (..., 4).
    """
    ab = quaternion_raw_multiply(a, b)
    return normalize(ab, dim=-1)

def quat_symmetric_tf(tf_q, ex_q):
    """
    This function applies a rotation transformation on tf_q to account for z 
    symmetry, while ex_q is simply expanded to match the size resulting tf_q
    """

    # Construct rotated quaternions
    if not hasattr(quat_symmetric_tf, 'rot_q'):
        
        # Creating all the degrees of rotation
        # degrees = torch.tensor([0,90,180,270], device=tf_q.device).float() #
        # degrees = torch.arange(0, 360, 15).float()
        degrees = torch.arange(0, 360).float()
        factor = torch.sin(torch.deg2rad(degrees) / 2)
        
        # Creating the values for the quaternions 4 elements
        w = torch.cos(torch.deg2rad(degrees) / 2)
        ones = (1 * factor)
        zeros = (0 * factor)
        
        # x, y, z = ones, zeros, zeros
        x, y, z = zeros, ones, zeros # This is the correct one!
        # x, y, z = zeros, zeros, ones

        
        # Constructing the overall rotation quaternions
        quat_symmetric_tf.rot_q = torch.vstack((w, x, y, z)).T

        # Expanding the data to prepare it for larger-scale quaternion multiplication
        quat_symmetric_tf.rot_q = torch.unsqueeze(quat_symmetric_tf.rot_q, dim=0)

    # Ensuring that rot_q is in the same device as tf_q
    if quat_symmetric_tf.rot_q.device != tf_q.device:
        quat_symmetric_tf.rot_q = quat_symmetric_tf.rot_q.to(tf_q.device)

    # Getting information of the size of data
    n_of_quat = tf_q.shape[0]
    n_of_r_quat = quat_symmetric_tf.rot_q.shape[1]

    # Expanding data to make data match in size
    e_tf_q = torch.unsqueeze(tf_q, dim=1).expand((n_of_quat, n_of_r_quat, 4))
    e_ex_q = torch.unsqueeze(ex_q, dim=1).expand((n_of_quat, n_of_r_quat, 4))
    e_q = quat_symmetric_tf.rot_q.expand((n_of_quat, n_of_r_quat, 4))

    # Applying transformations onto the pred data
    rot_tf_q = quaternion_multiply(e_tf_q.double(), e_q.double())

    return rot_tf_q, e_ex_q

#-------------------------------------------------------------------------------
# GPU Memory Functions

def count_tensors():

    count = 0
    empty_counter = 0

    for obj in gc.get_objects():
        try:
            if torch.is_tensor(obj) or (hasattr(obj, 'data') and torch.is_tensor(obj.data)):
                #print(type(obj), obj.size())
                count += 1
                if torch.empty(obj):
                    empty_counter += 1
        except: pass

    return f'total: {count} - empty: {empty_counter}'

def memory_leak_check():
    memory_percentage = torch.cuda.memory_allocated()/torch.cuda.max_memory_allocated()
    print(f"Memory used: {memory_percentage:.3f}")


    dict2 = {}

    # Obtaining the first cupy in the dict
    first_cupy = dict2[list(dict2.keys())[0]]

    # For each item in the dict1, convert it to cupy
    for key in dict1.keys():
        dict2[key] = torch.as_tensor(cp.unpackbits(dict1[key]), device=f'cuda:{first_cupy.device.id}')

    return dict2