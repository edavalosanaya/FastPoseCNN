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

import gpu_tensor_funcs as gtf

import hough_voting as hvk

#-------------------------------------------------------------------------------

def find_matches_batched(preds, gts):
    """
    OLD CODE! Previous matching method
    Find the matches between predictions and ground truth data

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
                iou_2d_mask = gtf.torch_get_2d_iou(
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

    pred_gt_matches = {
        'sample_ids': [],
        'class_ids': [],
        'symmetric_ids': []
    }

    keys_to_stack = [
        'instance_masks', # Class
        'quaternion', 'R', # Rotation
        'scales', # Size
        'xy', 'z', 'T', # Translation
        'RT', # Transformation
        'xy_mask', 'hypothesis', 'pruned_hypothesis' # Hough Voting
    ]

    # If there is no classes in the predicted, then return None
    if preds['class_ids'].shape[0] == 0:
        return None

    # Determine all the types of classes present
    preds_classes = torch.unique(preds['class_ids'])
    gts_classes = torch.unique(gts['class_ids'])
    
    # Expanding data to allow for fast comparison
    e_preds_classes = torch.unsqueeze(preds_classes, dim=-1).expand(preds_classes.shape[0], gts_classes.shape[0])
    e_gts_classes = gts_classes.expand(preds_classes.shape[0], gts_classes.shape[0])
    
    # Finding shared values
    shared = (e_preds_classes == e_gts_classes)
    
    # Compressing results to use as index
    index = torch.where(torch.sum(shared, dim=0) == 1)[0]
    
    # Shared classes
    shared_classes = gts_classes[index]

    # If there is no shared classes, then return None
    if shared_classes.shape[0] == 0:
        return None

    # For each class
    for class_id in shared_classes:

        # Finding the specific indexes for the class
        gts_class_instances = torch.where(gts['class_ids'] == class_id)[0]
        preds_class_instances = torch.where(preds['class_ids'] == class_id)[0]

        # Check the number of instances in pred and gts dicts
        n_of_m1 = gts_class_instances.shape[0]
        n_of_m2 = preds_class_instances.shape[0]

        # If there is no instances of this class to begin with
        if n_of_m1 == 0 or n_of_m2 == 0:
            continue

        # Find the 2D Iou between the pred and gt instance masks
        iou_2ds = gtf.batchwise_get_2d_iou(
            gts['instance_masks'][gts_class_instances],
            preds['instance_masks'][preds_class_instances]
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
            continue            

        # Keep only good matches
        max_v = max_v[valid_max_id]
        max_pred_id = max_pred_id[valid_max_id]
        max_gt_id = max_gt_id[valid_max_id]

        # Storing shared data
        class_instance_identifiers = torch.tensor(
            [class_id], device=gts_class_instances.device
        ).repeat(max_gt_id.shape[0])

        # Storing data that is shared between ground truth and pred agg data
        pred_gt_matches['sample_ids'].append(gts['sample_ids'][gts_class_instances][max_gt_id])
        pred_gt_matches['symmetric_ids'].append(gts['symmetric_ids'][gts_class_instances][max_gt_id])
        pred_gt_matches['class_ids'].append(class_instance_identifiers)

        # Select the match data and combined them together!
        for data_key in gts.keys():
            if data_key in keys_to_stack:
                # Stack data
                stacked_data = torch.stack(
                    (
                        gts[data_key][gts_class_instances][max_gt_id],
                        preds[data_key][preds_class_instances][max_pred_id]
                    )
                )

                # Store stacked data
                if data_key in pred_gt_matches.keys():
                    pred_gt_matches[data_key].append(stacked_data)
                else:
                    pred_gt_matches[data_key] = [stacked_data]

    # Concatinating the results
    for key in pred_gt_matches.keys():
        if key in ['sample_ids', 'class_ids', 'symmetric_ids']:
            axis = 0
        else:
            axis = 1
        pred_gt_matches[key] = torch.cat(pred_gt_matches[key], dim=axis)

    return pred_gt_matches        
