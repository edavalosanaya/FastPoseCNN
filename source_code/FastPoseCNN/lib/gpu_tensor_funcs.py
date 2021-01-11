import os
import sys
import gc

import torch
import torch.nn as nn
import torch.utils.dlpack

import cupy as cp
import cupyx as cpx
import cupyx.scipy.ndimage

# Local imports
sys.path.append(os.getenv("TOOLS_DIR"))
import visualize as vz

#-------------------------------------------------------------------------------
# Native PyTorch Functions

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

            # Converting torch to GPU numpy (cupy)
            with cp.cuda.Device(class_mask.device.index):
                cupy_class_mask = cp.asarray(class_mask)

            # Shattering the segmentation into instances
            cupy_instance_masks, num_of_instances = cupyx.scipy.ndimage.label(cupy_class_mask)

            # Convert cupy to tensor
            # https://discuss.pytorch.org/t/convert-torch-tensors-directly-to-cupy-tensors/2752/7
            instance_masks = torch.utils.dlpack.from_dlpack(cupy_instance_masks.toDlpack())

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
                xy = torch.sum(xy_img, dim=(1,2)) / torch.sum(instance_mask)
                z = torch.sum(z_img, dim=(1,2)) / torch.sum(instance_mask)
                scales = torch.sum(scales_img, dim=(1,2)) / torch.sum(instance_mask)

                # Convert xy (embedded) to xy (pixel)
                h,w = instance_mask.shape
                pixel_xy = create_pixel_xy(xy, 'simple', h, w)

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
                max_id = torch.argmax(all_iou_2d)

                # Mark the mask_id as taken
                used_pred_ids.append(max_id)

                # Create a match container
                match = {
                    'sample_id': n,
                    'class_id': gts[n]['class_id'][gt_id],
                    'iou_2d_mask': max_iou_2d_mask,
                    'quaternion': torch.stack((gts[n]['quaternion'][gt_id], preds[n]['quaternion'][max_id])),
                    'xy': torch.stack((gts[n]['xy'][gt_id], preds[n]['xy'][max_id])),
                    'z': torch.stack((gts[n]['z'][gt_id], preds[n]['z'][max_id])),
                    'scales': torch.stack((gts[n]['scales'][gt_id], preds[n]['scales'][max_id])),
                    'T': torch.stack((gts[n]['T'][gt_id], preds[n]['T'][max_id])),
                    'RT': torch.stack((gts[n]['RT'][gt_id], preds[n]['RT'][max_id])),
                }

            # Store the container
            pred_gt_matches.append(match)

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
# Generative/Conversion Functions

def create_pixel_xy(xy, input_type, h, w):

    if input_type == 'simple':
        pixel_xy = xy.clone()
        pixel_xy[0] = xy[1] * w
        pixel_xy[1] = xy[0] * h
        pixel_xy = pixel_xy.reshape((-1,1))
    else:
        raise NotImplementedError("Other input_types for xy are not implemented yet!")

    return pixel_xy

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

#-------------------------------------------------------------------------------
# Comparison Functions

def torch_get_2d_iou(tensor1: torch.Tensor, tensor2: torch.Tensor) -> torch.Tensor:

    intersection = torch.sum(torch.logical_and(tensor1, tensor2))
    union = torch.sum(torch.logical_or(tensor1, tensor2))
    return torch.true_divide(intersection,union)

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