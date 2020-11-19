import torch
import torch.nn as nn
import cupy as cp
import cupyx as cpx

#-------------------------------------------------------------------------------
# Native PyTorch Functions

def class_compress_quaternion(mask_logits, quaternion):
    """
    Args:
        mask_logits: NxCxHxW
        quaternion: Nx4CxHxW
    Returns:
        compressed_quat: Nx4xHxW
    """

    # Determing the number of classes
    num_of_classes = mask_logits.shape[1]

    # First convert mask logits to mask values
    lsf_mask = nn.LogSoftmax(dim=1)(mask_logits.clone())

    # Convert lsf mask into a categorical mask (NxHxW)
    mask = torch.argmax(lsf_mask, dim=1)

    # Divide the quaternion to 4 (4 for each class) (Nx4xHxW)
    class_quaternions = torch.chunk(quaternion, num_of_classes-1, dim=1)

    # Constructing the final compressed quaternion
    compressed_quat = torch.zeros_like(class_quaternions[0], requires_grad=quaternion.requires_grad)

    # Creating container for all class quats
    class_quats = []

    # Filling the compressed_quat from all available objects in mask
    for object_class_id in torch.unique(mask):

        if object_class_id == 0: # if the background, skip
            continue

        # Create class mask (NxHxW)
        class_mask = mask == object_class_id

        # Unsqueeze the class mask to make it (Nx1xHxW) making broadcastable with
        # the (Nx4xHxW) quaternion
        class_quat = torch.unsqueeze(class_mask, dim=1) * class_quaternions[object_class_id-1]

        # Storing class_quat into class_quats
        class_quats.append(class_quat)

    # If not empty quats
    if class_quats:
        # Calculating the total quats
        compressed_quat = torch.sum(torch.stack(class_quats, dim=0), dim=0)

    return compressed_quat, mask

def mask_gradients(to_be_masked, mask):

    # Creating a binary mask of all objects
    binary_mask = (mask != 0)

    # Unsqueeze and expand to match the shape of the quaternion
    binary_mask = torch.unsqueeze(binary_mask, dim=1).expand_as(to_be_masked)

    # Make only the components that match the mask regressive in the quaternion
    if to_be_masked.requires_grad:
        to_be_masked.register_hook(lambda grad: grad * binary_mask.float())

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