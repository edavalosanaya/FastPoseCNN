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

    def __init__(self, classes, intrinsics):
        super().__init__()
        self.classes = classes # including background
        self.intrinsics = intrinsics
        self.inv_intrinsics = torch.inverse(intrinsics)

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
        logits: Union[dict, None] = None, # logical data
        categos: Union[dict, None] = None # categorical data
        ):

        # Can only use either logits or categorical data
        assert (logits == None) ^ (categos == None), "XOR: logits or categos"

        # Ensuring that intrinsics is in the same device
        if self.intrinsics.device != cat_mask.device:
            self.intrinsics = self.intrinsics.to(cat_mask.device)
            self.inv_intrinsics = torch.inverse(self.intrinsics)

        # Outputs
        agg_pred = []

        # Splitting all the logits into class chunks
        if logits:
            class_compress_logits = {}
            class_chunks_logits = {k:torch.chunk(v, self.classes-1, dim=1) for k,v in logits.items()}

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
            
            # If logits used, then perform class compression on logits
            if logits:
                # Perform class compression on the logits for pixel-wise regression
                for logit_key in logits.keys():

                    # Applying the class mask on the logits class chunk
                    masked_class_chunk = class_chunks_logits[logit_key][class_id-1] * torch.unsqueeze(class_mask, dim=1)
                    
                    # Need to squeeze when logit_key == z in dim = 1 to match 
                    # categorical ground truth data
                    if logit_key == 'z':
                        masked_class_chunk = torch.squeeze(masked_class_chunk, dim=1)

                    if logit_key in class_compress_logits.keys():
                        class_compress_logits[logit_key] += masked_class_chunk
                    else:
                        class_compress_logits[logit_key] = masked_class_chunk

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
            for logit_key in ['quaternion', 'scales', 'xy', 'z']:

                if logits:
                    # Indexing the class logits
                    class_logits = class_chunks_logits[logit_key][class_id-1]

                    # Matching the class logits to the instances' quantity and order
                    instance_logits = class_logits[sample_id_for_instances]

                    # Applying the mask to the instance logits
                    masked_data = torch.unsqueeze(pure_instance_masks, dim=1) * instance_logits
                
                elif categos:

                    # Matching the categorical data to the instance's quantity and order
                    instance_categos = categos[logit_key][sample_id_for_instances]

                    # Need to expand the instance_categos when logit_key == z
                    if logit_key == 'z':
                        instance_categos = torch.unsqueeze(instance_categos, dim=1)

                    # Obtaining the data for the instance
                    masked_data = torch.unsqueeze(pure_instance_masks, dim=1) * instance_categos

                else:
                    raise RuntimeError(f"Invalid input data for {logit_key}")

                # Take the average of quaterins, scales and z's logit value
                if logit_key in ['quaternion', 'scales', 'z']:
                    total_val = torch.sum(masked_data, dim=(-2, -1))
                    mask_size = torch.sum(pure_instance_masks, dim=(-2, -1))
                    agg_data = torch.div(total_val, torch.unsqueeze(mask_size.T, dim=1))

                else: # for xy, we need hough voting
                    agg_data = self.batchwise_hough_voting(masked_data, pure_instance_masks)

                # Storing the mean of the instances to agg_pred
                class_data[logit_key] = agg_data 

            # Once all the raw data has been aggregated, we need to calculate the 
            # rotation matrix of each instance.
            RT_data = gtf.batchwise_get_RT(
                class_data['quaternion'],
                class_data['xy'],
                    torch.exp(class_data['z']),
                    self.inv_intrinsics
                )
            class_data['RT'] = RT_data

            # Storing the finished data of one class into the multi-class container
            agg_pred.append(class_data) 

        if logits:
            return agg_pred, class_compress_logits
        elif categos:
            return agg_pred
        else:
            raise RuntimeError("Invalid data: needs to be logits/categos")

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

    def batchwise_hough_voting(self, uv_imgs, masks, N=25):

        pixel_xys = torch.zeros((masks.shape[0], 2), device=masks.device).float()

        for instance_id in range(masks.shape[0]):
            pixel_xy = hv.hough_voting(uv_imgs[instance_id], masks[instance_id], N)
            pixel_xys[instance_id] = pixel_xy

        return pixel_xys