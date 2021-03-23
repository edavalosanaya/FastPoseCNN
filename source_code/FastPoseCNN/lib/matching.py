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
import type_hinting as th

#-------------------------------------------------------------------------------
# Constants

KEYS_TO_STACK = [
    'instance_masks', # Class
    'quaternion', 'R', # Rotation
    'scales', # Size
    'xy', 'z', 'T', # Translation
    'RT', # Transformation
]

#-------------------------------------------------------------------------------
# Utility Routines

def stack_and_store_data(pred_gt_matches, gts, preds, gts_instances, preds_instances):
    
    # Select the match data and combined them together!
    for data_key in gts.keys():
        if data_key in KEYS_TO_STACK:
            
            # Stack data
            stacked_data = torch.stack(
                (
                    gts[data_key][gts_instances],
                    preds[data_key][preds_instances]
                )
            )

            # Store stacked data
            if data_key in pred_gt_matches.keys():
                pred_gt_matches[data_key].append(stacked_data)
            else:
                pred_gt_matches[data_key] = [stacked_data]

#-------------------------------------------------------------------------------
# Filling-Missing-Pred Routines

def batchwise_find_matches2(preds: th.AggData, gts: th.AggData) -> th.MatchedData:

    pred_gt_matches = {
        'sample_ids': [],
        'class_ids': [],
        'symmetric_ids': []
    }

    # For each class
    for class_id in torch.unique(gts['class_ids']):

        # Finding the specific indexes for the class
        gts_class_instances = torch.where(gts['class_ids'] == class_id)[0]
        preds_class_instances = torch.where(preds['class_ids'] == class_id)[0]

        # Check the number of instances in pred and gts dicts
        n_of_m1 = gts_class_instances.shape[0]
        n_of_m2 = preds_class_instances.shape[0]
        
        # Storing shared data
        class_instance_identifiers = torch.tensor(
            [class_id], device=gts_class_instances.device
        ).repeat(n_of_m1)

        # If there is no instances of this class to begin with, just fill everthing
        # with standard quat, xyz, and scales data
        if n_of_m2 == 0:

            # Placing all the class instances in
            pred_gt_matches['sample_ids'].append(gts['sample_ids'][gts_class_instances])
            pred_gt_matches['symmetric_ids'].append(gts['symmetric_ids'][gts_class_instances])
            pred_gt_matches['class_ids'].append(class_instance_identifiers)

            # Generate standard preds for all class instances
            standard_preds = get_standard_preds(gts, n_of_m1)

            # Stacking and storing data
            stack_and_store_data(
                pred_gt_matches,
                gts,
                standard_preds,
                gts_class_instances,
                torch.arange(0, n_of_m1, device=gts_class_instances.device)
            )
        
        # Else if there are instances to compare, compare them!
        else:

            # Find the 2D Iou between the pred and gt instance masks
            iou_2ds = gtf.batchwise_get_2d_iou(
                gts['instance_masks'][gts_class_instances],
                preds['instance_masks'][preds_class_instances]
            )

            #iou_2ds = torch.zeros((n_of_m1, n_of_m2))
            #min_n = min(n_of_m1, n_of_m2)
            #iou_2ds[torch.arange(min_n), torch.arange(min_n)] = 1

            # Pair ground truth and predictions based on their iou_2ds
            max_v, max_pred_ids = torch.max(iou_2ds, dim=1)
            gt_ids = torch.arange(n_of_m1, device=gts_class_instances.device)

            # Obtaining the valid (true matches)
            valid_max_id = max_v > 0         

            # Obtaining the ids for the valid samples
            valid_pred_ids = max_pred_ids[valid_max_id]
            valid_gt_ids = gt_ids[valid_max_id]

            # Storing data that is shared between ground truth and pred agg data
            pred_gt_matches['sample_ids'].append(gts['sample_ids'][gts_class_instances][valid_gt_ids])
            pred_gt_matches['symmetric_ids'].append(gts['symmetric_ids'][gts_class_instances][valid_gt_ids])
            pred_gt_matches['class_ids'].append(class_instance_identifiers[valid_gt_ids])

            # Stacking and storing data
            stack_and_store_data(
                pred_gt_matches,
                gts,
                preds,
                gts_class_instances[valid_gt_ids],
                preds_class_instances[valid_pred_ids]
            )

            # Now obtaining the ids for the invalid samples
            invalid_gt_ids = gt_ids[~valid_max_id]
            #invalid_pred_ids = max_pred_ids[~valid_max_id]

            # If there is no missed ground truth data, skip this part
            if invalid_gt_ids.shape[0] == 0:
                continue

            # Storing data that is not shared between ground truth and pred agg data
            # We plan on using the standard pred to fill in this invalid match data
            # to punish the model for not capturing this instances
            pred_gt_matches['sample_ids'].append(gts['sample_ids'][gts_class_instances][invalid_gt_ids])
            pred_gt_matches['symmetric_ids'].append(gts['symmetric_ids'][gts_class_instances][invalid_gt_ids])
            pred_gt_matches['class_ids'].append(class_instance_identifiers[invalid_gt_ids])

            # Generating the standard preds to matched the invalid gts ids
            standard_preds = get_standard_preds(gts, invalid_gt_ids.shape[0])

            # Stacking and storing data
            stack_and_store_data(
                pred_gt_matches,
                gts,
                standard_preds,
                invalid_gt_ids,
                torch.arange(0, invalid_gt_ids.shape[0], device=gts_class_instances.device)
            )

    # Concatinating the results
    for key in pred_gt_matches.keys():
        if key in ['sample_ids', 'class_ids', 'symmetric_ids']:
            axis = 0
        else:
            axis = 1
        pred_gt_matches[key] = torch.cat(pred_gt_matches[key], dim=axis)

    return pred_gt_matches 

def get_standard_preds(gts, n_of_data):

    # Construct standard preds base (scalar)
    if not hasattr(get_standard_preds, 'standard_preds'):

        # Creating the standard quaternion
        get_standard_preds.standard_preds = {}

        # Filling all zeros
        for data_key in gts.keys():
            if data_key in KEYS_TO_STACK:
                get_standard_preds.standard_preds[data_key] = torch.zeros_like(
                    gts[data_key][0],
                    dtype=gts[data_key].dtype,
                    device=gts[data_key].device,
                    #requires_grad=gts[data_key].requires_grad
                )

        # Exception to zeros is quaternions (to make the data have a norm = 1)
        get_standard_preds.standard_preds['quaternion'][0] = 1
        get_standard_preds.standard_preds['RT'] = torch.eye(4, device=gts['RT'].device)
        get_standard_preds.standard_preds['z'][0] = 1000

    # Creating the standard preds for this specific gts
    std_preds = {}

    # Once constructed the data, expanded it to match the gts correctly
    for data_key in gts.keys():
        if data_key in KEYS_TO_STACK:
            
            # Expanding it to match the dimensions of gts, repeat it to
            # the needed amount of instances, and move it to the same device
            std_preds[data_key] = torch.unsqueeze(
                get_standard_preds.standard_preds[data_key],
                dim=0
            ).repeat_interleave(n_of_data, dim=0).to(gts['instance_masks'].device)

    return std_preds
            
#-------------------------------------------------------------------------------
# Non-Filling-Missing-Pred Routines

def batchwise_find_matches(preds, gts):

    # If preds == None or gts == None, skip it!
    if not preds or not gts:
        return None

    # If there is no classes in the predicted, then return None
    if preds['class_ids'].shape[0] == 0:
        return None

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

    # For each class
    for class_id in torch.unique(gts['class_ids']):

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

        # Stacking and storing data
        stack_and_store_data(
            pred_gt_matches,
            gts,
            preds,
            gts_class_instances[max_gt_id],
            preds_class_instances[max_pred_id]
        )

    # Concatinating the results
    for key in pred_gt_matches.keys():
        if key in ['sample_ids', 'class_ids', 'symmetric_ids']:
            axis = 0
        else:
            axis = 1
        
        # Catching a rather odd error
        if len(pred_gt_matches[key]) == 0:
            return None

        # Concatinating results here
        pred_gt_matches[key] = torch.cat(pred_gt_matches[key], dim=axis)

    return pred_gt_matches   
