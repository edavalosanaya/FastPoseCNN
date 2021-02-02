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
        agg_pred = []

        # Obtain the height and width of the masks
        b,h,w = cat_mask.shape

        # Per class
        for class_id in range(self.classes):

            # Container for all the instances within the class
            class_data = {}

            # Skipping the background
            if class_id == 0:
                continue

            # Selecting the class mask
            class_mask = (cat_mask == class_id) *  torch.Tensor([1]).float().to(cat_mask.device)

            # Breaking the class mask into instances (number from 1 to num_of_instances)
            instance_masks, total_num_of_instances = self.batchwise_break_segmentation_mask(class_mask)

            # Storing the number of instances to keep record
            class_data['total_num_of_instances'] = total_num_of_instances

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
            class_data['sample_ids'] = sample_id_for_instances
            class_data['instance_masks'] = pure_instance_masks

            # Obtain the instance's values (quaternion, z, scales)
            for data_key in ['quaternion', 'scales', 'xy', 'z']:

                # Matching the categorical data to the instance's quantity and order
                instance_data = data[data_key][sample_id_for_instances]

                # Need to expand the instance_data when data_key == z
                if data_key == 'z':
                    instance_data = torch.unsqueeze(instance_data, dim=1)

                # Obtaining the data for the instance
                masked_data = torch.unsqueeze(pure_instance_masks, dim=1) * instance_data

                # Take the average of quaternions, scales and z's logit value
                if data_key in ['quaternion', 'scales', 'z']:
                    total_val = torch.sum(masked_data, dim=(-2, -1))
                    mask_size = torch.sum(pure_instance_masks, dim=(-2, -1))
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

                # Storing the mean of the instances to agg_pred
                class_data[data_key] = agg_data

            # Storing the finished data of one class into the multi-class container
            agg_pred.append(class_data) 
 
        return agg_pred

    def batchwise_break_segmentation_mask(self, class_mask):

        # Converting torch to GPU numpy (cupy)
        if class_mask.is_cuda:
            with cp.cuda.Device(class_mask.device.index):
                cupy_class_mask = cp.asarray(class_mask)

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

    def batchwise_hough_voting(self, uv_imgs, masks):

        pixel_xys = torch.zeros((masks.shape[0], 2), device=masks.device).float()

        for instance_id in range(masks.shape[0]):
            pixel_xy = self.hough_voting_layer.forward(uv_imgs[instance_id], masks[instance_id])
            pixel_xys[instance_id] = pixel_xy

        return pixel_xys