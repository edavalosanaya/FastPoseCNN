import os
import sys
import shutil
import pathlib
import collections

import pdb

import random
import numpy as np
import cv2
import imutils
import skimage.io

import torch
import torchvision
import torch.utils
import torch.utils.tensorboard

# Local Imports

root = next(path for path in pathlib.Path(os.path.abspath(__file__)).parents if path.name == 'FastPoseCNN')
sys.path.append(str(pathlib.Path(__file__).parent))

import json_tools
import data_manipulation
import project
import draw
import np_transforms

#-------------------------------------------------------------------------------
# File Constants

camera_dataset = root.parents[1] / 'datasets' / 'NOCS' / 'camera' / 'val'

#-------------------------------------------------------------------------------
# Collective Data Classes

class NOCSDataset(torch.utils.data.Dataset):

    """
    Doc-string
    """

    ############################################################################
    # Functions required for torch.utils.data.Dataset to work
    ############################################################################

    def __init__(self, dataset_path, dataset_max_size=None, transform=None):
        """
        Args:
            dataset_path: (string or pathlib object): Path to the dataset
            transform (callable, optional): Optional transform to be applied on 
                a sample.
        """

        if isinstance(dataset_path, str):
            dataset_path = pathlib.Path(dataset_path)

        # Saving the dataset path and transform
        self.dataset_path = dataset_path
        self.transform = transform

        # Determing the paths of the all the color images inside the dataset_path
        self.color_image_path_list = self.get_image_paths_in_dir(self.dataset_path, dataset_max_size)

        # Saving the size of the dataset
        self.dataset_size = len(self.color_image_path_list)

        # Transformations
        self.transforms = trf = np_transforms.Compose([
            np_transforms.Scale(size=(256, 256)),
            np_transforms.RandomCrop(size=(224, 224)),
            np_transforms.RandomVerticalFlip(prob=0.5),
            np_transforms.RandomHorizontalFlip(prob=0.5),
            np_transforms.RotateImage(angles=(-15, 15)),
            np_transforms.ToTensor(),
        ])
        
        return None

    def __len__(self):
        """
        Returns the size of the datasekt
        """
        return self.dataset_size

    def __getitem__(self, idx):
        """
        Args: idx (tensor/list): A collection of ids to retrieve datasample 
        Objective:
            Fetches an entry (color image, depth image, mask image, coord image, and
            meta data) given an idx of the dataset.
        Output:
            total_data (list of dictionaries): A collection of data sample(s) to
            be used as training samples
        """

        if torch.is_tensor(idx):
            idx = idx.tolist()

        # Loading data
        sample = self.get_data_sample(self.color_image_path_list[idx])

        if self.transform:
            sample = self.transform(sample) 

        return sample

    ############################################################################
    # Functions used to create the dataset and data samples
    ############################################################################

    def get_image_paths_in_dir(self, dir_path, max_size=None):
        """
        Args:
            dir_path (pathlib object): The directory to be search for all possible
                color images
        Objective:
            Perform a search inside dir_path to collect all available color images
            and save them into a list
        Output:
            total_path_list (list): A list of all collected color images
        """

        total_path_list = [] # [[color, depth], ...]
        
        # handling additional directories
        eval_paths = [dir_path]
        i = 0

        while True:

            # Break condition: if no more directories to evaluate, end
            if len(eval_paths) <= i:
                break

            # Selecting the new path to evaluate
            eval_path = eval_paths[i]
            i += 1

            # for all the evaluate paths, do the following
            files = [x for x in eval_path.iterdir() if x.is_file()]

            # Adding all the color images into the total_path_list
            color_images = [x for x in files if x.name.find('color') != -1 and x.suffix == '.png']
            total_path_list += color_images

            directories = [x for x in eval_path.iterdir() if x.is_dir()]
            eval_paths += directories

            # If max size is reached or overpassed, break from loop
            if max_size != None:
                if len(total_path_list) >= max_size:
                    break

        # removing any falsy elements
        total_path_list = [x for x in total_path_list if x]

        # Trimming excess if dataset_max_size is set
        if max_size != None:
            total_path_list = total_path_list[:max_size]

        return total_path_list

    def get_data_sample(self, color_image_path):

        """
        Args:
            color_image_path (pathlib object): Path to the color image 
        Objective:
            Collect all the information pertaining to the color image path given.
        Output:
            data (dictionary): Contains all the information pertaining to that
            color image.
        """

        # Getting the data set ID and obtaining the corresponding file paths for the
        # mask, coordinate map, depth, and meta files
        data_id = color_image_path.name.replace('_color.png', '')
        mask_path = color_image_path.parent / f'{data_id}_mask.png'
        depth_path = color_image_path.parent / f'{data_id}_depth.png'
        coord_path = color_image_path.parent / f'{data_id}_coord.png'
        meta_plus_path = color_image_path.parent / f'{data_id}_meta+.json'

        # Loading data (color, mask, coord, depth)
        color_image = skimage.io.imread(str(color_image_path))
        mask_image = skimage.io.imread(str(mask_path))[:, :, 0]
        coord_map = skimage.io.imread(str(coord_path))[:, :, :3]
        depth_image = skimage.io.imread(str(depth_path))

        # Loading JSON data
        json_data = json_tools.load_from_json(meta_plus_path)
        instance_dict = json_data['instance_dict']
        scales = np.asarray(json_data['scales'], dtype=np.float32)
        RTs = np.asarray(json_data['RTs'], dtype=np.float32)
        norm_factors = np.asarray(json_data['norm_factors'], dtype=np.float32)
        quaternions = np.asarray(json_data['quaternions'], dtype=np.float32)

        # Converting depth to the correct shape and dtype
        if len(depth_image.shape) == 3: # encoded depth image
            new_depth = np.uint16(depth_image[:, :, 1] * 256) + np.uint16(depth_image[:,:,2])
            new_depth = new_depth.astype(np.uint16)
            depth_image = new_depth
        elif len(depth_image.shape) == 2 and depth_image.dtype == 'uint16':
            pass # depth is perfecto!
        else:
            assert False, '[ Error ]: Unsupported depth type'

        # Converting depth again from np.uint16 to np.int16 because np.uint16 
        # is not supported by PyTorch
        depth_image = depth_image.astype(np.int16)

        # Using the exact depth from the RTs
            
        # flip z axis of coord map
        coord_map = np.array(coord_map, dtype=np.float32) / 255
        coord_map[:, :, 2] = 1 - coord_map[:, :, 2]

        # Converting the mask into a typical mask
        mask_cdata = np.array(mask_image, dtype=np.int32)
        mask_cdata[mask_cdata==255] = -1

        # NOTE!
        """
        RTs follow this convention
        CAMERA SPACE --- inverse RT ---> WORLD SPACE
        WORLD SPACE  ---     RT     ---> CAMERA SPACE
        """

        # Creating data into multi-label structure
        num_classes = len(project.constants.SYNSET_NAMES)
        class_ids = list(instance_dict.values())
        h, w = mask_cdata.shape

        # Shifting color channel to the beginning to match PyTorch's format
        color_image = np.moveaxis(color_image, -1, 0)
        coord_map = np.moveaxis(coord_map, -1, 0)
        depth_image = np.expand_dims(depth_image, axis=0)

        zs = np.zeros([1, h, w], dtype=np.float32)
        masks = np.zeros([h, w], dtype=np.float32)
        quat_img = np.zeros([4, h, w], dtype=np.float32)
        scales_img = np.zeros([3, h, w], dtype=np.float32)

        for e_id, (i_id, c_id) in enumerate(instance_dict.items()):

            # Obtaining the mask for each class
            instance_mask = np.equal(mask_cdata, int(i_id)) * 1

            # Contour filling to avoid possible error
            instance_mask = instance_mask.astype(np.uint8)
            _, contour, hei = cv2.findContours(instance_mask, cv2.RETR_TREE ,cv2.CHAIN_APPROX_SIMPLE)
            for cnt in contour:
                cv2.drawContours(instance_mask, [cnt],0,1,-1)

            assert c_id > 0 and c_id < len(project.constants.SYNSET_NAMES), f'Invalid class id: {c_id}'
            #print(c_id)
            
            # Converting the instances in the mask to classes
            masks = np.where(instance_mask == 1, c_id, masks)
            zs[:,:] += instance_mask * np.linalg.inv(RTs[e_id])[2,3] * 1000

            # Storing quaterions
            
            quaternion = quaternions[e_id,:]
            for i in range(4): # real, i, j, and k component
                quat_img[i,:,:] = np.where(instance_mask != 0, quaternion[i], 0)

            # Storing scales
            scale = scales[e_id,:] / norm_factors[e_id]
            for i in range(3): # real, i, j, and k component
                scales_img[i,:,:] = np.where(instance_mask != 0, scale[i], 0)

        # Performing image transformations
        #pdb.set_trace() # Test image transformations

        # Perform numpy to PyTorch conversion
        color_image = self.numpy_to_torch(color_image)
        depth_image = self.numpy_to_torch(depth_image)
        zs = self.numpy_to_torch(zs)
        masks = self.numpy_to_torch(masks, dtype=torch.LongTensor)
        coord_map = self.numpy_to_torch(coord_map)
        scales_img = self.numpy_to_torch(scales_img)
        quat_img = self.numpy_to_torch(quat_img)

        # Always return tuple
        return color_image, depth_image, zs, masks, coord_map, scales_img, quat_img

    def numpy_to_torch(self, numpy_object, dtype=torch.FloatTensor):
        """
        Moves the numpy object to a PyTorch FloatTensor
        """

        torch_object = torch.from_numpy(numpy_object).type(dtype)

        if torch.cuda.is_available():
            torch_object = torch_object.cuda()

        return torch_object

    ############################################################################
    # Additional functionalitity
    ############################################################################

    def get_random_sample(self, from_this_idx=None):

        if from_this_idx:
            random_idx = random.choice(from_this_idx)
        else:
            random_idx = random.choice(range(self.dataset_size))
        
        # Loading data
        sample = self.get_data_sample(self.color_image_path_list[random_idx])

        return sample

#-------------------------------------------------------------------------------
# Functions

#-------------------------------------------------------------------------------
# Main Code

if __name__ == '__main__':
    # For testing the dataset

    # For randomly splitting datasets: https://pytorch.org/docs/stable/data.html#torch.utils.data.random_split

    print('Loading dataset')
    dataset = NOCSDataset(camera_dataset, dataset_max_size=10)
    print('Finished loading dataset')

    # Testing custom Dataset (Working)
    #"""
    print("\n\nTesting dataset loading\n\n")
    test_sample = dataset[0]

    #print(test_sample)

    for item in test_sample:
        print(item.shape)
    #"""

    # Testing built-in dataset (Working)
    #"""
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=2, shuffle=True)

    sample = next(iter(dataloader))

    for item in sample:
        print(item.shape)

    #"""

        
