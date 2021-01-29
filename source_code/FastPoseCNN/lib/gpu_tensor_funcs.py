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

import hough_voting as hv

#-------------------------------------------------------------------------------
# Native PyTorch Functions

def normalize(data, dim):

    # Determine the norm along that dimension
    norm_data = data.norm(dim=dim, keepdim=True)

    # Replace zero norm by one to avoid division by zero error
    safe_norm_data = torch.where(
        norm_data != 0, norm_data, torch.tensor(1.0, device=norm_data.device).float()
    )

    # Normalize data
    normalized_data = data / safe_norm_data

    return normalized_data

def class_compress(num_of_classes, cat_mask, data):
    """
    Args:
        num_of_classes: int (already removed the background)
        cat_mask: NxHxW
        data: NxACxHxW [A = 2,3,4]
    Returns:
        compressed_data: NxAxHxW [A = 2,3,4]
        mask: NxHxW
    """

    # Divide the data by the number of classes
    all_class_data = torch.chunk(data, num_of_classes, dim=1)

    # Constructing the final compressed datas
    compressed_data = torch.zeros_like(all_class_data[0], requires_grad=data.requires_grad)

    # Creating container for all class quats
    class_datas = []

    # Filling the compressed_data from all available objects in mask
    for object_class_id in torch.unique(cat_mask):

        if object_class_id == 0: # if the background, skip
            continue

        # Create class mask (NxHxW)
        class_mask = cat_mask == object_class_id

        # Unsqueeze the class mask to make it (Nx1xHxW) making broadcastable with
        # the (NxAxHxW) data
        class_data = torch.unsqueeze(class_mask, dim=1) * all_class_data[object_class_id-1]

        # Storing class_data into class_datas
        class_datas.append(class_data)

    # If not empty datas
    if class_datas:
        # Calculating the total datas
        compressed_data = torch.sum(torch.stack(class_datas, dim=0), dim=0)

    return compressed_data

def mask_gradients(to_be_masked, mask):

    # Creating a binary mask of all objects
    binary_mask = (mask != 0)

    # Unsqueeze and expand to match the shape of the quaternion
    binary_mask = torch.unsqueeze(binary_mask, dim=1).expand_as(to_be_masked)

    # Make only the components that match the mask regressive in the quaternion
    if to_be_masked.requires_grad:
        to_be_masked.register_hook(lambda grad: grad * binary_mask.float())

def dense_class_data_aggregation(mask, dense_class_data, intrinsics):
    """
    Args:
        mask: NxHxW (categorical)
        dense_class_data: dict
            quaternion: Nx4xHxW
            xy: Nx2xHxW
            z: NxHxW
            scales: Nx3xHxW

    Returns:
        outputs: list
            single_sample_output: dict
                class_ids: list
                instance_ids: list
                instance_mask: HxW
                z: list (1)
                xy: list (2)
                quaternion: list (4)
                T: list (3)
                RT: list (4x4)
                scales: list (3)
    """
    outputs = []

    for n in range(mask.shape[0]):

        # Per sample outputs
        single_sample_output = {
            'class_id': [],
            'instance_id': [],
            'instance_mask': [],
            'quaternion': [],
            'xy': [],
            'z': [],
            'scales': [],
            'T': [],
            'RT': []
        }

        # Selecting the samples mask
        sample_mask = mask[n]

        # Process data per class
        for class_id in torch.unique(sample_mask):

            # Ignore the background
            if class_id == 0:
                continue

            # Selecting the class mask
            class_mask = (sample_mask == class_id) * torch.Tensor([1]).float().to(sample_mask.device)

            # Breaking a class segmentation mask into instance masks
            num_of_instances, instance_masks = break_segmentation_mask(class_mask)

            # Process data per instance
            for instance_id in range(1, num_of_instances+1):

                # Selecting the instance mask (HxW)
                instance_mask = (instance_masks == instance_id) # BoolTensor

                # Obtaining the pertaining quaternion
                quaternion_mask = instance_mask.expand((4, instance_mask.shape[0], instance_mask.shape[1]))
                quaternion_img = torch.where(quaternion_mask, dense_class_data['quaternion'][n], torch.zeros_like(quaternion_mask).float().to(sample_mask.device))

                # Obtaining the pertaining xy
                xy_mask = instance_mask.expand((2, instance_mask.shape[0], instance_mask.shape[1]))
                xy_img = torch.where(xy_mask, dense_class_data['xy'][n], torch.zeros_like(xy_mask).float().to(sample_mask.device))

                # Obtaining the pertaining z
                z_mask = instance_mask.expand((1, instance_mask.shape[0], instance_mask.shape[1]))
                z_img = torch.where(z_mask, dense_class_data['z'][n], torch.zeros_like(z_mask).float().to(sample_mask.device))

                # Obtaining the pertaining scales
                scales_mask = instance_mask.expand((3, instance_mask.shape[0], instance_mask.shape[1]))
                scales_img = torch.where(scales_mask, dense_class_data['scales'][n], torch.zeros_like(scales_mask).float().to(sample_mask.device))

                # Aggregate the values via naive average
                quaternion = torch.sum(quaternion_img, dim=(1,2)) / torch.sum(instance_mask)
                z = torch.sum(z_img, dim=(1,2)) / torch.sum(instance_mask)
                scales = torch.sum(scales_img, dim=(1,2)) / torch.sum(instance_mask)

                # Convert xy (unit vector) to xy (pixel)
                #pixel_xy = hv.hough_voting(xy_img, instance_mask)
                h,w = instance_mask.shape
                xy = torch.tensor([0,0], device=xy_img.device).float()
                pixel_xy = create_pixel_xy(xy, h, w)

                # Create translation vector
                T = create_translation_vector(pixel_xy, torch.exp(z), intrinsics)

                # Create rotation matrix
                RT = quat_2_RT_given_T_in_world(quaternion, T)

                # Storing per-sample data
                single_sample_output['class_id'].append(class_id)
                single_sample_output['instance_id'].append(instance_id)
                single_sample_output['instance_mask'].append(instance_mask)
                single_sample_output['quaternion'].append(quaternion)
                single_sample_output['xy'].append(pixel_xy)
                single_sample_output['z'].append(z)
                single_sample_output['scales'].append(scales)
                single_sample_output['T'].append(T)
                single_sample_output['RT'].append(RT)

        # Storing per sample data
        outputs.append(single_sample_output)

    return outputs

def find_matches_batched(preds, gts):
    """Find the matches between predictions and ground truth data

    Args:
        preds/gts ([list]):
            single_sample_output: dict
                class_id: list
                instance_id: list
                instance_mask: HxW
                z: list (1)
                xy: list (2)
                quaternion: list (4)
                T: list (3)
                RT: list (4x4)
                scales: list (3)

    Returns:
        pred_gt_matches [list]: 
            match ([dict]):
                sample_id: int
                class_id: torch.Tensor
                quaternion: torch.Tensor
                xy: torch.Tensor
                z: torch.Tensor
                scales: torch.Tensor
                RT: torch.Tensor
                T: torch.Tensor
    """

    pred_gt_matches = []

    # For each sample in the batch
    for n in range(len(preds)):

        # If it this sample their was no predictions, skip it
        if len(preds[n]['class_id']) == 0:
            continue

        # Avoid using already matched predictions
        used_pred_ids = []

        # For all instances in the ground truth data
        for gt_id in range(len(gts[n]['instance_id'])):

            # Match the ground truth and preds given the 2D IoU between the 
            # instance mask
            all_iou_2d = torch.zeros((len(preds[n]['instance_id']))).to(preds[n]['class_id'][0].device)

            for pred_id in range(len(preds[n]['instance_id'])):

                # If the pred id has been matched, avoid reusing
                if pred_id in used_pred_ids:
                    all_iou_2d[pred_id] = -1
                    continue

                # If the classes do not match, skip it!
                if gts[n]['class_id'][gt_id] != preds[n]['class_id'][pred_id]:
                    continue

                # Determing the 2d IoU
                iou_2d_mask = torch_get_2d_iou(
                    preds[n]['instance_mask'][pred_id],
                    gts[n]['instance_mask'][gt_id]
                )

                # Storing the 2d IoU for later comparison
                all_iou_2d[pred_id] = iou_2d_mask

            # After determing all the 2d IoU, select the largest, given that it 
            # is valid (greater than 0)
            max_iou_2d_mask = torch.max(all_iou_2d)
            
            # If no match is found, use a standard quaternion to notify failure
            if max_iou_2d_mask <= 0:

                # Creating the standard quaternion
                standard_quaternion = torch.tensor(
                    [1,0,0,0],
                    dtype=gts[n]['quaternion'][gt_id].dtype,
                    device=gts[n]['quaternion'][gt_id].device
                )
                standard_xy = torch.zeros_like(
                    gts[n]['xy'][gt_id],
                    dtype=gts[n]['xy'][gt_id].dtype,
                    device=gts[n]['xy'][gt_id].device
                )
                standard_z = torch.zeros_like(
                    gts[n]['z'][gt_id],
                    dtype=gts[n]['z'][gt_id].dtype,
                    device=gts[n]['z'][gt_id].device
                )
                standard_scales = torch.zeros_like(
                    gts[n]['scales'][gt_id],
                    dtype=gts[n]['scales'][gt_id].dtype,
                    device=gts[n]['scales'][gt_id].device
                )
                standard_T = torch.zeros_like(
                    gts[n]['T'][gt_id],
                    dtype=gts[n]['T'][gt_id].dtype,
                    device=gts[n]['T'][gt_id].device
                )
                standard_RT = torch.zeros_like(
                    gts[n]['RT'][gt_id],
                    dtype=gts[n]['RT'][gt_id].dtype,
                    device=gts[n]['RT'][gt_id].device
                )

                # Create a match container
                match = {
                    'sample_id': n,
                    'class_id': gts[n]['class_id'][gt_id],
                    'iou_2d_mask': max_iou_2d_mask,
                    'quaternion': torch.stack((gts[n]['quaternion'][gt_id], standard_quaternion)),
                    'xy': torch.stack((gts[n]['xy'][gt_id], standard_xy)),
                    'z': torch.stack((gts[n]['z'][gt_id], standard_z)),
                    'scales': torch.stack((gts[n]['scales'][gt_id], standard_scales)),
                    'T': torch.stack((gts[n]['T'][gt_id], standard_T)),
                    'RT': torch.stack((gts[n]['RT'][gt_id], standard_RT)),
                }

            # Else, use the best possible matched quaternion
            else:
                # use the mask with the highest 2d iou score
                max_pred_id = torch.argmax(all_iou_2d)

                # Mark the mask_id as taken
                used_pred_ids.append(max_pred_id)

                # Create a match container
                match = {
                    'sample_id': n,
                    'class_id': gts[n]['class_id'][gt_id],
                    'iou_2d_mask': max_iou_2d_mask,
                    'quaternion': torch.stack((gts[n]['quaternion'][gt_id], preds[n]['quaternion'][max_pred_id])),
                    'xy': torch.stack((gts[n]['xy'][gt_id], preds[n]['xy'][max_pred_id])),
                    'z': torch.stack((gts[n]['z'][gt_id], preds[n]['z'][max_pred_id])),
                    'scales': torch.stack((gts[n]['scales'][gt_id], preds[n]['scales'][max_pred_id])),
                    'T': torch.stack((gts[n]['T'][gt_id], preds[n]['T'][max_pred_id])),
                    'RT': torch.stack((gts[n]['RT'][gt_id], preds[n]['RT'][max_pred_id])),
                }

            # Store the container
            pred_gt_matches.append(match)

    return pred_gt_matches

def batchwise_find_matches(preds, gts):

    pred_gt_matches = []
    keys_to_stack = [
        'instance_masks', # Class
        'quaternion', # Rotation
        'scales', # Size
        'xy', 'z', # Translation
        'RT', # Transformation
        'xy_mask', 'hypothesis', 'pruned_hypothesis' # Hough Voting
    ]

    # For each class
    for class_id in range(len(preds)):

        class_data = {'class_id': class_id}

        # Check the number of instances in pred and gts dicts
        n_of_m1 = gts[class_id]['instance_masks'].shape[0]
        n_of_m2 = preds[class_id]['instance_masks'].shape[0]

        # If there is no instances of this class to begin with
        if n_of_m1 == 0 or n_of_m2 == 0:
            pred_gt_matches.append(class_data)
            continue

        # Find the 2D Iou between the pred and gt instance masks
        iou_2ds = batchwise_get_2d_iou(
            gts[class_id]['instance_masks'],
            preds[class_id]['instance_masks']
        )

        #iou_2ds = torch.zeros((n_of_m1, n_of_m2))
        #min_n = min(n_of_m1, n_of_m2)
        #iou_2ds[torch.arange(min_n), torch.arange(min_n)] = 1

        # Pair ground truth and predictions based on their iou_2ds
        max_v, max_pred_id = torch.max(iou_2ds, dim=1)
        max_gt_id = torch.arange(n_of_m1)

        # Removing those whose max iou2d was zero
        valid_max_id = max_v > 0

        # Check that there is true matches to begin
        if (valid_max_id == False).all():
            pred_gt_matches.append(class_data)
            continue            

        # Keep only good matches
        max_v = max_v[valid_max_id]
        max_pred_id = max_pred_id[valid_max_id]
        max_gt_id = max_gt_id[valid_max_id]

        # Storing shared data
        class_data['sample_ids'] = gts[class_id]['sample_ids'][max_gt_id]

        # Select the match data and combined them together!
        for data_key in keys_to_stack:
            stacked_data = torch.stack(
                (
                    gts[class_id][data_key][max_gt_id],
                    preds[class_id][data_key][max_pred_id]
                )
            )

            # Store stacked data
            class_data[data_key] = stacked_data

        # Store the class-specific data into the multi-class data container
        pred_gt_matches.append(class_data)

    return pred_gt_matches        

def stack_class_matches(matches, key):

    # Given the key, aggregated all the matches of that type by class
    class_data = {}

    # Iterating through all the matches
    for match in matches:

        # If this match is the first of its class object, then add new item
        if int(match['class_id']) not in class_data.keys():
            class_data[int(match['class_id'])] = [match[key]]

        # else, just append it to the pre-existing list
        else:
            class_data[int(match['class_id'])].append(match[key])

    # Container for per class ground truths and predictions
    stacked_class_data = {}

    # Once the quaternions have been separated by class, stack them all to
    # formally compute the QLoss
    for class_number, class_data in class_data.items():

        # Stacking all matches in one class
        stacked_class_data[class_number] = torch.stack(class_data)

    return stacked_class_data

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

    return RT

def samplewise_get_RT(agg_data, inv_intrinsics):

    for class_id in range(len(agg_data)):

        # Once all the raw data has been aggregated, we need to calculate the 
        # rotation matrix of each instance.
        RT_data = batchwise_get_RT(
            agg_data[class_id]['quaternion'],
            agg_data[class_id]['xy'],
            agg_data[class_id]['z'],
            inv_intrinsics
        )

        # Storing generated RT
        agg_data[class_id]['RT'] = RT_data

    return agg_data

#-------------------------------------------------------------------------------
# Generative/Conversion Functions

def break_segmentation_mask(class_mask):

    # Converting torch to GPU numpy (cupy)
    if class_mask.is_cuda:
        with cp.cuda.Device(class_mask.device.index):
            cupy_class_mask = cp.asarray(class_mask)

        # Shattering the segmentation into instances
        cupy_instance_masks, num_of_instances = cupyx.scipy.ndimage.label(cupy_class_mask)

        # Convert cupy to tensor
        # https://discuss.pytorch.org/t/convert-torch-tensors-directly-to-cupy-tensors/2752/7
        instance_masks = torch.utils.dlpack.from_dlpack(cupy_instance_masks.toDlpack())

    else:
        numpy_class_mask = np.asarray(class_mask)

        # Shattering the segmentation into instances
        numpy_instance_masks, num_of_instances = scipy.ndimage.label(numpy_class_mask)

        # Convert numpy to tensor
        instance_masks = torch.from_numpy(numpy_instance_masks)

    return num_of_instances, instance_masks

def create_pixel_xy(xy, h, w):

    pixel_xy = xy.clone()
    pixel_xy[0] = xy[1] * w
    pixel_xy[1] = xy[0] * h
    pixel_xy = pixel_xy.reshape((-1,1))

    return pixel_xy

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
    if hasattr(scale, "__iter__"):
        bbox_3d = torch.tensor([
            [scale[0] / 2, scale[1] / 2, scale[2] / 2],
            [scale[0] / 2, scale[1] / 2, -scale[2] / 2],
            [-scale[0] / 2, scale[1] / 2, scale[2] / 2],
            [-scale[0] / 2, scale[1] / 2, -scale[2] / 2],
            [scale[0] / 2, -scale[1] / 2, scale[2] / 2],
            [scale[0] / 2, -scale[1] / 2, -scale[2] / 2],
            [-scale[0] / 2, -scale[1] / 2, scale[2] / 2],
            [-scale[0] / 2, -scale[1] / 2, -scale[2] / 2]], device=scale.device, dtype=scale.dtype)
    else:
        bbox_3d = torch.tensor([
            [scale / 2, scale / 2, scale / 2],
            [scale / 2, scale / 2, -scale / 2],
            [-scale / 2, scale / 2, scale / 2],
            [-scale / 2, scale / 2, -scale / 2],
            [scale / 2, -scale / 2, scale / 2],
            [scale / 2, -scale / 2, -scale / 2],
            [-scale / 2, -scale / 2, scale / 2],
            [-scale / 2, -scale / 2, -scale / 2]], device=scale.device, dtype=scale.dtype)

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

def torch_quat_distance(q0, q1):

    # Determine the difference
    q0_minus_q1 = q0 - q1
    q0_plus_q1  = q0 + q1
    
    # Obtain the norm
    d_minus = q0_minus_q1.norm(dim=1)
    d_plus  = q0_plus_q1.norm(dim=1)

    # Compare the norms and select the one with the smallest norm
    ds = torch.stack((d_minus, d_plus))
    rad_distance = torch.min(ds, dim=0).values

    # Converting the rad to degree
    degree_distance = torch.rad2deg(rad_distance)

    return degree_distance

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
        offset_error = get_T_offset_error(centers3d_1[i], centers3d_2[i])
        offset_errors.append(offset_error)

    offset_errors = torch.stack(offset_errors)
    return offset_errors

def get_T_offset_error(center3d_1, center3d_2):
    diff = center3d_1 - center3d_2
    return torch.sqrt(torch.sum(torch.pow(diff,2)))

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
    offset_errors = get_T_offset_errors(
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

#-------------------------------------------------------------------------------
# CuPy Functions

def tensor2cupy_dict(dict1):

    # Convert PyTorch Tensor to CuPy:
    # https://discuss.pytorch.org/t/convert-torch-tensors-directly-to-cupy-tensors/2752/5

    dict2 = {}

    # Obtaining the first tensor in the dict
    first_tensor = dict2[list(dict2.keys())[0]]

    # Imply which cuda to use based on the device of the tensor
    with cp.cuda.Device(first_tensor.device.index):

        # For each item in the dict1, convert it to cupy
        for key in dict1.keys():
            dict2[key] = cp.asarray(dict1[key])

    return dict2

def cupy2tensor_dict(dict1):

    dict2 = {}

    # Obtaining the first cupy in the dict
    first_cupy = dict2[list(dict2.keys())[0]]

    # For each item in the dict1, convert it to cupy
    for key in dict1.keys():
        dict2[key] = torch.as_tensor(cp.unpackbits(dict1[key]), device=f'cuda:{first_cupy.device.id}')

    return dict2