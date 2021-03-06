import os
import sys
import typing
import pathlib

import torch
import torch.nn as nn

import skimage.io
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
import type_hinting as th

# COUNTER = 0

#-------------------------------------------------------------------------------

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

    def forward(self, cat_data: th.CategoricalData) -> th.AggData:

        global COUNTER

        # Outputs
        complete_agg_data = {
            'class_ids': [],
            'instance_masks': [],
            'sample_ids': []
        }

        # Grabbing the categorical mask
        cat_mask = cat_data['mask']

        # Breaking the overall mask into instances
        instance_masks, total_num_of_instances = self.batchwise_break_segmentation_mask(cat_mask != 0)

        # ! For now, I am using for visualization of the instance masks
        # instance_mask_path = pathlib.Path(os.getenv('TEST_OUTPUT')) / f'{COUNTER}-instance_mask.png'
        # skimage.io.imsave(str(instance_mask_path), instance_masks[0].cpu().numpy())
        # COUNTER+=1

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
            try:
                instance_classes = torch.stack([torch.unique(x)[1] for x in torch.unbind(class_instance_masks)])
            except RuntimeError: # No instances                
                instance_classes = torch.empty((0,), device=cat_mask.device)
                
            # Storing the classes for the instance masks
            complete_agg_data['class_ids'].append(instance_classes)

        # Concatenate all of the information
        for key in complete_agg_data.keys():
            complete_agg_data[key] = torch.cat(complete_agg_data[key], dim=0)

        # Obaint the instances' values      
        for data_key in ['quaternion', 'scales', 'xy', 'z']:

            # Matching the categorical data to the instance' quantity and order
            instance_data = cat_data[data_key][complete_agg_data['sample_ids']]

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
