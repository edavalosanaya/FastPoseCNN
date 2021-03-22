import os
import sys
from typing import Union

import torch
import torch.nn as nn

import numpy as np
import scipy
import scipy.ndimage

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
import gpu_tensor_funcs as gtf

class AggregationLayer(nn.Module):

    def __init__(self, HPARAM, classes):
        super().__init__()
        self.HPARAM = HPARAM
        self.classes = classes # including background
        self.hough_voting_layer = hv.HoughVotingLayer(self.HPARAM)        

        # Creating binary structure that does not attach batchwise data
        self.s = torch.tensor([
            [
                [False, False, False],
                [False, False, False],
                [False, False, False],
            ],
            [
                [False, True, False],
                [True,  True, True],
                [False, True, False],
            ],
            [
                [False, False, False],
                [False, False, False],
                [False, False, False],
            ],
        ])

    def forward(
        self,
        cat_mask: torch.Tensor,
        data: Union[dict], # categorical data
        ):

        # Outputs
        complete_agg_data = {
            'class_ids': [],
            'instance_masks': [],
            'sample_ids': []
        }

        # Breaking the overall mask into instances
        instance_masks, total_num_of_instances = self.batchwise_break_segmentation_mask(cat_mask != 0)

        # Obtain the shape of the masks
        b,h,w = cat_mask.shape

        # Per batch
        for bi in range(b):

            # Obtaining the number of instances in the batch
            batch_number_of_instances = (torch.unique(instance_masks[bi]) != 0).sum()
            sample_to_mask = torch.ones(
                (batch_number_of_instances,),
                dtype=torch.int64,
                device=cat_mask.device
            ) * bi

            # Storing array indicating sample-to-mask correspondence
            complete_agg_data['sample_ids'].append(sample_to_mask)

            # Converting the categorical instance mask into binary instance masks
            binary_instance_masks = torch.zeros((total_num_of_instances+1, h, w), device=cat_mask.device)
            binary_instance_masks = binary_instance_masks.scatter(0, torch.unsqueeze(instance_masks[bi], dim=0).type(torch.int64), 1)[1:]

            # Removing all-zero image-planes (extra were made for safety reasons)
            binary_instance_masks = binary_instance_masks[torch.sum(binary_instance_masks, dim=(-2, -1)) != 0]

            # Storing the binary instance masks
            complete_agg_data['instance_masks'].append(binary_instance_masks)

            # Obtaining the class_ids for the instance masks
            class_instance_masks = torch.unsqueeze(cat_mask[bi], dim=0) * binary_instance_masks.bool()
            instance_classes = torch.stack([torch.unique(x)[1] for x in torch.unbind(class_instance_masks)])

            # Storing the classes for the instance masks
            complete_agg_data['class_ids'].append(instance_classes)

        # Concatenate all of the information
        for key in complete_agg_data.keys():
            complete_agg_data[key] = torch.cat(complete_agg_data[key], dim=0)

        # Obaint the instances' values      
        for data_key in ['quaternion', 'scales', 'xy', 'z']:

            # Matching the categorical data to the instance' quantity and order
            instance_data = data[data_key][complete_agg_data['sample_ids']]

            # Need to expand the instance_data when data_key == 'z'
            if data_key == 'z':
                instance_data = torch.unsqueeze(instance_data, dim=1)

            # Obtaining the data for the instance in respect to their mask
            masked_data = torch.unsqueeze(complete_agg_data['instance_masks'], dim=1) * instance_data

            # Use native average as the instance-wise data
            if data_key in ['quaternion', 'scales', 'z']:
                total_val = torch.sum(masked_data, dim=(-2, -1))
                mask_size = torch.sum(complete_agg_data['instance_masks'], dim=(-2, -1))
                agg_data = torch.div(total_val, torch.unsqueeze(mask_size.T, dim=1))

                # Undoing the torch.log in data embedding
                if data_key == 'z':
                    agg_data = torch.exp(agg_data)

                # Normalizing quaternion
                elif data_key == 'quaternion':
                    agg_data = gtf.normalize(agg_data, dim=1)

            # Not performing aggregation via averaging. Instead allow hough voting to use data
            elif data_key == 'xy':
                agg_data = masked_data

            # Store the information into the container
            complete_agg_data[data_key] = agg_data

        return complete_agg_data

    def forward2(
        self, 
        cat_mask: torch.Tensor, 
        data: Union[dict], # categorical data
        ):

        # Outputs
        complete_agg_data = {
            'class_ids': [],
            'instance_masks': [],
            'sample_ids': []
        }

        # Obtain the height and width of the masks
        b,h,w = cat_mask.shape

        # Per class
        for class_id in range(self.classes):

            # Skipping the background
            if class_id == 0:
                continue

            # Selecting the class mask
            class_mask = (cat_mask == class_id) *  torch.Tensor([1]).float().to(cat_mask.device)

            # Breaking the class mask into instances (number from 1 to num_of_instances)
            instance_masks, total_num_of_instances = self.batchwise_break_segmentation_mask(class_mask)

            # Creating class identifies
            class_instance_identifiers = torch.tensor(
                [class_id], device=cat_mask.device
            ).repeat(total_num_of_instances)

            # Storing the number of instances to keep record
            complete_agg_data['class_ids'].append(class_instance_identifiers)

            # Construct pure instance masks
            sample_id_for_instances = torch.zeros(
                (total_num_of_instances,), 
                dtype=torch.int64, 
                device=cat_mask.device
            )
            pure_instance_masks = torch.empty(
                (total_num_of_instances, h, w), 
                device=cat_mask.device
            )

            # For each instance, find its corresponding element and grab its mask
            for i in range(1, total_num_of_instances+1):

                # Determine which sample within the entire batch the instance is located
                element_id = torch.where(instance_masks == i)[0][0]

                # Storing the specific instance mask into the pure_instance_masks
                pure_instance_masks[i-1] = (instance_masks[element_id] == i)

                # Storing the element id of the instance masks into a tensor for record
                sample_id_for_instances[i-1] = element_id

            # Storing the instances masks and their sample ids
            complete_agg_data['sample_ids'].append(sample_id_for_instances)
            complete_agg_data['instance_masks'].append(pure_instance_masks)

        # Concatenate all of the classes data into a single batch container
        for key in complete_agg_data.keys():
            complete_agg_data[key] = torch.cat(complete_agg_data[key], dim=0)

        # Obtain the instance's values (quaternion, z, scales)
        for data_key in ['quaternion', 'scales', 'xy', 'z']:

            # Matching the categorical data to the instance's quantity and order
            instance_data = data[data_key][complete_agg_data['sample_ids']]

            # Need to expand the instance_data when data_key == z
            if data_key == 'z':
                instance_data = torch.unsqueeze(instance_data, dim=1)

            # Obtaining the data for the instance
            masked_data = torch.unsqueeze(complete_agg_data['instance_masks'], dim=1) * instance_data

            # Take the average of quaternions, scales and z's logit value
            if data_key in ['quaternion', 'scales', 'z']:
                total_val = torch.sum(masked_data, dim=(-2, -1))
                mask_size = torch.sum(complete_agg_data['instance_masks'], dim=(-2, -1))
                agg_data = torch.div(total_val, torch.unsqueeze(mask_size.T, dim=1))

                # Undoing the torch.log in data embedding
                if data_key == 'z':
                    agg_data = torch.exp(agg_data)

                # Normalizing data
                elif data_key == 'quaternion':
                    agg_data = gtf.normalize(agg_data, dim=1)

            # Saving the masked unit vector mask since we need to perform hough
            # voting for this section.
            elif data_key == 'xy':
                agg_data = masked_data

            # Storing the mean of the instances to complete_agg_data
            complete_agg_data[data_key] = agg_data
 
        return complete_agg_data

    def batchwise_break_segmentation_mask(self, class_mask):

        # Converting torch to GPU numpy (cupy)
        if class_mask.is_cuda:
            with cp.cuda.Device(class_mask.device.index):
                cupy_class_mask = cp.asarray(class_mask.float())

            # Shattering the segmentation into instances
            cupy_instance_masks, num_of_instances = cupyx.scipy.ndimage.label(cupy_class_mask, structure=self.s)

            # Convert cupy to tensor
            # https://discuss.pytorch.org/t/convert-torch-tensors-directly-to-cupy-tensors/2752/7
            instance_masks = torch.utils.dlpack.from_dlpack(cupy_instance_masks.toDlpack())

        else:
            numpy_class_mask = np.asarray(class_mask)

            # Shattering the segmentation into instances
            numpy_instance_masks, num_of_instances = scipy.ndimage.label(numpy_class_mask, structure=self.s)

            # Convert numpy to tensor
            instance_masks = torch.from_numpy(numpy_instance_masks)

        return instance_masks, num_of_instances
