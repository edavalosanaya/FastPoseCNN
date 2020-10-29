import os
from os import replace
import sys
import shutil
import pathlib
import collections
from typing import List

import pdb

import random
import numpy as np
import cv2
import imutils
import skimage
import skimage.io
import skimage.transform

import torch
import torchvision
import torch.utils
import torch.utils.tensorboard
import torchvision.transforms.functional

import catalyst
import segmentation_models_pytorch as smp

import matplotlib
if os.environ.get('DISPLAY', '') == '':
    matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.cm

# Local Imports

root = next(path for path in pathlib.Path(os.path.abspath(__file__)).parents if path.name == 'FastPoseCNN')
sys.path.append(str(pathlib.Path(__file__).parent))

import json_tools as jt
import data_manipulation as dm
import project as pj
import draw as dr
import visualize as vz
import transforms

#-------------------------------------------------------------------------------
# File Constants

ENCODER = 'resnext50_32x4d'
ENCODER_WEIGHTS = 'imagenet'

#-------------------------------------------------------------------------------
# Old Data Classes

class OLDNOCSDataset(torch.utils.data.Dataset):

    """
    Doc-string
    """

    ############################################################################
    # Functions required for torch.utils.data.Dataset to work
    ############################################################################

    def __init__(self, dataset_path, dataset_max_size=None, device='automatic',
                 balance=None):
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

        # Determing the paths of the all the color images inside the dataset_path
        self.color_image_path_list = self.get_image_paths_in_dir(self.dataset_path, dataset_max_size)

        # Saving the size of the dataset
        self.dataset_size = len(self.color_image_path_list)

        # Selecting device
        if device == 'automatic':
            self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = device

        # Selecting balance (cropping)
        self.balance = balance

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

        # Balancing of dataset
        if self.balance:
            sample = self.balance_sample(sample)
        
        # Convert all numpy arrays into PyTorch Tensors (with correct dtypes)
        sample = self.numpy_sample_to_torch(sample) 

        # Moving sample to GPU if possible
        for key in sample.keys():
            sample[key] = sample[key].to(self.device)

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
        depth = skimage.io.imread(str(depth_path))

        # Loading JSON data
        json_data = jt.load_from_json(meta_plus_path)
        instance_dict = json_data['instance_dict']
        scales = np.asarray(json_data['scales'], dtype=np.float32)
        RTs = np.asarray(json_data['RTs'], dtype=np.float32)
        norm_factors = np.asarray(json_data['norm_factors'], dtype=np.float32)
        quaternions = np.asarray(json_data['quaternions'], dtype=np.float32)

        # Converting depth to the correct shape and dtype
        if len(depth.shape) == 3: # encoded depth image
            new_depth = np.uint16(depth[:, :, 1] * 256) + np.uint16(depth[:,:,2])
            new_depth = new_depth.astype(np.uint16)
            depth = new_depth
        elif len(depth.shape) == 2 and depth.dtype == 'uint16':
            pass # depth is perfecto!
        else:
            assert False, '[ Error ]: Unsupported depth type'

        # Converting depth again from np.uint16 to np.int16 because np.uint16 
        # is not supported by PyTorch
        depth = depth.astype(np.int16)

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
        num_classes = len(pj.constants.NOCS_CLASSES)
        class_ids = list(instance_dict.values())
        h, w = mask_cdata.shape

        # Shifting color channel to the beginning to match PyTorch's format
        #depth = np.expand_dims(depth, axis=0)
        color_image = np.moveaxis(color_image, -1, 0)
        coord_map = np.moveaxis(coord_map, -1, 0)

        zs = np.zeros([1, h, w], dtype=np.float32)
        masks = np.zeros([h, w], dtype=np.uint8)
        #masks = np.zeros([1,h,w], dtype=np.uint8)
        quat_image = np.zeros([4, h, w], dtype=np.float32)
        scales_image = np.zeros([3, h, w], dtype=np.float32)

        for e_id, (i_id, c_id) in enumerate(instance_dict.items()):

            # Obtaining the mask for each class
            instance_mask = np.equal(mask_cdata, int(i_id)) * 1

            # Contour filling to avoid possible error
            instance_mask = instance_mask.astype(np.uint8)
            _, contour, hei = cv2.findContours(instance_mask, cv2.RETR_TREE ,cv2.CHAIN_APPROX_SIMPLE)
            for cnt in contour:
                cv2.drawContours(instance_mask,[cnt],0,1,-1)

            assert c_id > 0 and c_id < len(pj.constants.NOCS_CLASSES), f'Invalid class id: {c_id}'
            
            # Converting the instances in the mask to classes
            masks = np.where(instance_mask == 1, c_id, masks)
            #zs[:,:] += instance_mask * np.linalg.inv(RTs[e_id])[2,3] * 1000
            zs = np.where(instance_mask == 1, np.linalg.inv(RTs[e_id])[2,3] * 1000, zs)

            # Storing quaternions
            quaternion = quaternions[e_id,:]
            for i in range(4): # real, i, j, and k component
                quat_image[i,:,:] = np.where(instance_mask != 0, quaternion[i], quat_image[i,:,:])

            # Storing scales
            scale = scales[e_id,:] / norm_factors[e_id]
            for i in range(3): # real, i, j, and k component
                scales_image[i,:,:] = np.where(instance_mask != 0, scale[i], scales_image[i,:,:])

        sample = {'color_image': color_image,
                  'depth': depth,
                  'zs': zs,
                  'masks': masks,
                  'coord_map': coord_map,
                  'scales_image': scales_image,
                  'quat_image': quat_image}

        return sample 

    def numpy_sample_to_torch(self, sample):
        """
        Moves the numpy object to a PyTorch FloatTensor
        """

        # Transform to tensor
        for key in sample.keys():
            sample[key] = torch.from_numpy(sample[key])

        # Making all inputs and outputs float expect of classification tasks (masks)
        for key in sample.keys():
            if key == 'masks':
                sample[key] = sample[key].type(torch.LongTensor)
            else:
                sample[key] = sample[key].type(torch.FloatTensor)

        return sample

    ############################################################################
    # Augmentations
    ############################################################################
    
    def balance_sample(self, sample):

        # First determing if there is an object in the image
        if len(np.unique(sample['masks'])) == 1:
            # If only background, just crop the size of the image match the size
            sample = self.crop_sample(sample)
            return sample

        # Selecting a random class, while not selecting the background
        selected_class = np.random.choice(np.unique(sample['masks'])[1:])

        # Create a mask specific to the selected class
        class_mask = (sample['masks'] == selected_class) * 1
        class_mask = class_mask.astype(np.uint8)

        try:
            # In the case of multiple objects within the same class, we find the contours
            _, cnts, hei = cv2.findContours(class_mask, cv2.RETR_TREE ,cv2.CHAIN_APPROX_SIMPLE)

            # If multple contours (multiple objects), select one randomly
            rand_choice = random.choice(range(len(cnts)))
            cnt = cnts[rand_choice]

            # Find the centroid of the contour
            M = cv2.moments(cnt)
            cX = int(M["m10"] / M["m00"])
            cY = int(M["m01"] / M["m00"])
            centroid = (cX, cY)
        except:
            centroid = None

        # Crop the sample given the center and crop_size
        cropped_sample = self.crop_sample(sample, center=centroid)

        return cropped_sample

    def crop_sample(self, sample, center=None, crop_size=128):

        # Creating a container for the output
        cropped_sample = {}

        # H,W of the images
        _, h, w = sample['color_image'].shape

        # No center is provided, then assume the center of the image
        if center is None:
            center = (w // 2, h // 2)

        # Obtaing half of the crop_size
        hcs = crop_size // 2 # half crop size

        # Determing the left, right, top and bottom of the image
        left = center[0] - hcs
        right = center[0] + hcs
        top = center[1] - hcs
        bottom = center[1] + hcs

        # Check if the left, right, top and bottom are outside the image
        left_pad = right_pad = top_pad = bottom_pad = 0

        if left < 0:
            left_pad = -left
            left = 0
        if right > w:
            right_pad = right - w
            right = w
        if top < 0:
            top_pad = -top
            top = 0
        if bottom > h:
            bottom_pad = bottom - h
            bottom = h 

        """
        if left_pad != 0 or right_pad != 0 or top_pad != 0 or bottom_pad != 0:
            pdb.set_trace()
        """

        # Crop the images in the sample
        for key in sample.keys():
            if len(sample[key].shape) == 2:
                crop = sample[key][top:bottom,left:right]
                crop_and_pad = np.pad(crop, ((top_pad,bottom_pad),(left_pad, right_pad)), mode='constant')
                cropped_sample[key] = crop_and_pad
            elif len(sample[key].shape) == 3:
                crop = sample[key][:,top:bottom,left:right]
                crop_and_pad = np.pad(crop, ((0,0),(top_pad,bottom_pad),(left_pad, right_pad)), mode='constant')
                cropped_sample[key] = crop_and_pad

        return cropped_sample

    ############################################################################
    # Additional functionalitity
    ############################################################################

    def get_random_sample(self, from_this_idx=None, batched=False):

        if from_this_idx:
            random_idx = random.choice(from_this_idx)
        else:
            random_idx = random.choice(range(self.dataset_size))
        
        # Loading data
        sample = self.__getitem__(random_idx)

        if batched:
            for key in sample.keys():
                sample[key] = torch.unsqueeze(sample[key], 0)

        return sample

#-------------------------------------------------------------------------------
# Segmentation Datasets

class CAMVIDSegDataset(torch.utils.data.Dataset):
    """CamVid Dataset. Read images, apply augmentation and preprocessing transformations.
    
    Args:
        images_dir (str): path to images folder
        masks_dir (str): path to segmentation masks folder
        class_values (list): values of classes to extract from segmentation mask
        augmentation (albumentations.Compose): data transfromation pipeline 
            (e.g. flip, scale, etc.)
        preprocessing (albumentations.Compose): data preprocessing 
            (e.g. normalization, shape manipulation, etc.)
    
    """
    
    CLASSES = pj.constants.CAMVID_CLASSES
    COLORMAP = pj.constants.CAMVID_COLORMAP
    
    def __init__(
            self, 
            dataset_dir,
            train_valid_test='train',
            classes=None, 
            augmentation=None, 
            preprocessing=None,
            mask_dataformat='CHW'
    ):
        # Determing the images and masks directory
        images_dir = os.path.join(dataset_dir, train_valid_test)
        masks_dir = os.path.join(dataset_dir, train_valid_test+'annot')

        self.ids = os.listdir(images_dir)
        self.images_fps = [os.path.join(images_dir, image_id) for image_id in self.ids]
        self.masks_fps = [os.path.join(masks_dir, image_id) for image_id in self.ids]
        
        # convert str names to class values on masks
        self.class_values = [self.CLASSES.index(cls.lower()) for cls in classes]
        
        self.augmentation = augmentation
        self.preprocessing = preprocessing
        self.mask_dataformat = mask_dataformat
    
    def __getitem__(self, i):
        
        # read data
        image = skimage.io.imread(self.images_fps[i])
        mask = skimage.io.imread(self.masks_fps[i])
        
        if self.mask_dataformat == 'CHW':
            # extract certain classes from mask (e.g. cars)
            masks = [(mask == v) for v in self.class_values]
            mask = np.stack(masks, axis=-1).astype('float')
        
        elif self.mask_dataformat == 'HW':
            pass
        else:
            raise RuntimeError('Invalid mask dataformat')
        
        # apply augmentations
        if self.augmentation:
            sample = self.augmentation(image=image, mask=mask)
            image, mask = sample['image'], sample['mask']
        
        # apply preprocessing
        if self.preprocessing:
            sample = self.preprocessing(image=image, mask=mask)
            image, mask = sample['image'], sample['mask']
            
        sample = {'image': image.astype('float32'), 'mask': mask.astype('long')}
        return sample
        
    def __len__(self):
        return len(self.ids)

    def get_random_batched_sample(self, batch_size=1):

        batched_sample = {}

        for sample_id in np.random.choice(np.arange(self.__len__()), size=batch_size, replace=False):

            sample = self.__getitem__(sample_id)

            for key in sample.keys():

                if key in batched_sample.keys():
                    batched_sample[key] = np.concatenate([batched_sample[key], np.expand_dims(sample[key], axis=0)], axis=0)

                else:
                    batched_sample[key] = np.expand_dims(sample[key], axis=0)

        return batched_sample

class VOCSegDataset(torch.utils.data.Dataset):
    """PASCAL VOC 2012 Dataset. Read images, apply augmentation and preprocessing transformations.
    
    Args:
        voc_dir (str): filepath to the VOC dataset
        augmentation (albumentations.Compose): data transfromation pipeline 
            (e.g. flip, scale, etc.)
        preprocessing (albumentations.Compose): data preprocessing 
            (e.g. noralization, shape manipulation, etc.)
    
    """
    
    CLASSES = pj.constants.VOC_CLASSES
    COLORMAP = pj.constants.VOC_COLORMAP
    
    def __init__(
            self, 
            voc_dir,
            is_train=False,
            classes=None, 
            augmentation=None, 
            preprocessing=None,
    ):
        # Reading text file containing details about the segmantation and images
        txt_fname = os.path.join(voc_dir, 'ImageSets', 'Segmentation', 'train.txt' if is_train else 'val.txt')
        with open(txt_fname, 'r') as f:
            images = f.read().split()

        self.is_train = is_train
        self.ids = images
        self.images_fps = [os.path.join(voc_dir, 'JPEGImages', f'{image_id}.jpg') for image_id in self.ids]
        self.masks_fps = [os.path.join(voc_dir, 'SegmentationClass', f'{image_id}.png') for image_id in self.ids]
        
        # convert str names to class values on masks
        self.class_values = [self.CLASSES.index(cls.lower()) for cls in classes]
        
        self.augmentation = augmentation
        self.preprocessing = preprocessing
    
    def __getitem__(self, i):
        
        pdb.set_trace()

        # read data
        image = skimage.io.imread(self.images_fps[i])
        mask = skimage.io.imread(self.masks_fps[i])
        
        # extract certain classes from mask (e.g. cars)
        masks = [(mask == v) for v in self.class_values]
        mask = np.stack(masks, axis=-1).astype('float')

        pdb.set_trace()
        
        # apply augmentations
        if self.augmentation:
            sample = self.augmentation(image=image, mask=mask)
            image, mask = sample['image'], sample['mask']

        pdb.set_trace()
        
        # apply preprocessing
        if self.preprocessing:
            sample = self.preprocessing(image=image, mask=mask)
            image, mask = sample['image'], sample['mask']

        pdb.set_trace()
            
        sample = {'image': image, 'mask': mask}
        return sample
        
    def __len__(self):
        return len(self.ids)

class NOCSSegDataset(torch.utils.data.Dataset):
    """NOCS Dataset. Read images, apply augmentation and preprocessing transformations.
    
    Args:
        dataset_dir (str): filepath to the dataset (train, valid, or test)
        max_size (int): maximum number of samples
        augmentation (albumentations.Compose): data transfromation pipeline 
            (e.g. flip, scale, etc.)
        preprocessing (albumentations.Compose): data preprocessing 
            (e.g. noralization, shape manipulation, etc.)
    """
    
    CLASSES = pj.constants.NOCS_CLASSES
    COLORMAP = pj.constants.NOCS_COLORMAP
    
    def __init__(
            self, 
            dataset_dir,
            max_size=None,
            classes=None, 
            augmentation=None, 
            preprocessing=None,
            balance=False,
            crop_size=128,
            mask_dataformat='CHW'
    ):
        self.images_fps = self.get_image_paths_in_dir(dataset_dir, max_size=max_size)
        
        # convert str names to class values on masks
        self.class_values = [self.CLASSES.index(cls.lower()) for cls in classes]
        
        self.augmentation = augmentation
        self.preprocessing = preprocessing
        self.balance = balance
        self.crop_size = crop_size
        self.mask_dataformat = mask_dataformat
    
    def __getitem__(self, i):
        
        # Reading data
        image = skimage.io.imread(str(self.images_fps[i]))

        mask_fp = str(self.images_fps[i]).replace('_color.png', '_mask.png')
        mask = skimage.io.imread(mask_fp, 0)[:,:,0].astype('float')
        mask[mask == 255] = 0

        json_fp = str(self.images_fps[i]).replace('_color.png', '_meta+.json')
        json_data = jt.load_from_json(json_fp)

        # Given the data, modify any data necessary
        instance_dict = json_data['instance_dict']

        # Creating the new mask
        if self.mask_dataformat == 'CHW':
        
            new_mask = np.zeros((mask.shape[0], mask.shape[1], len(self.class_values)))
            
            for i_id, c_id in instance_dict.items():
                new_mask[:,:,c_id] += np.where(mask == int(i_id), 1, 0)

            mask = new_mask.astype('float')

        elif self.mask_dataformat == 'HW':

            new_mask = np.zeros((mask.shape[0], mask.shape[1]))

            for i_id, c_id in instance_dict.items():
                new_mask[:,:] += np.where(mask == int(i_id), c_id, 0)

            mask = new_mask.astype('float')

        #pdb.set_trace()

        # Applying class balancing
        if self.balance:
            sample = {'image': image, 'mask': mask}
            sample = self.balance_sample(sample)
            image = sample['image']
            mask = sample['mask']

        # apply augmentations
        if self.augmentation:
            sample = self.augmentation(image=image, mask=mask)
            image, mask = sample['image'], sample['mask']

        clean_image = np.moveaxis(image.copy(), -1, 0)
        clean_mask = np.moveaxis(mask.copy(), -1, 0)

        # apply preprocessing
        if self.preprocessing:
            sample = self.preprocessing(image=image, mask=mask)
            image, mask = sample['image'], sample['mask']

        sample = {'image': image.astype('float32'), 'mask': mask.astype('long'), 'clean image': clean_image, 'clean mask': clean_mask}

        #print(sample['image'].shape, sample['mask'].shape)
            
        return sample
        
    def __len__(self):
        return len(self.images_fps)

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

    def balance_sample(self, sample):

        # First determing if there is an object in the image
        class_has_object = []

        # If mask dataformat is CHW
        if self.mask_dataformat == 'CHW':
            for i in range(sample['mask'].shape[2]):
                if len(np.unique(sample['mask'][:,:,i])) == 2:
                    class_has_object.append(i)
        
        # elif mask dataformat is HW
        elif self.mask_dataformat == 'HW':
            class_has_object = np.unique(sample['mask'])

        # error if other mask dataformat
        else:
            raise RuntimeError('Invalid data format')

        if len(class_has_object) == 0:
            centroid = None
        else:
            # Selecting a random class, while not selecting the background
            selected_class = np.random.choice(np.array(class_has_object))

            # Create a mask specific to the selected class
            if self.mask_dataformat == 'CHW':
                class_mask = sample['mask'][:,:,selected_class]
            elif self.mask_dataformat == 'HW':
                class_mask = (sample['mask'] == selected_class) * 1
            else:
                raise RuntimeError('Invalid data format')
            
            class_mask = class_mask.astype(np.uint8)

            try:
                # In the case of multiple objects within the same class, we find the contours
                cnts, hei = cv2.findContours(class_mask, cv2.RETR_TREE ,cv2.CHAIN_APPROX_SIMPLE)

                # If multple contours (multiple objects), select one randomly
                rand_choice = random.choice(range(len(cnts)))
                cnt = cnts[rand_choice]

                # Find the centroid of the contour
                M = cv2.moments(cnt)
                cX = int(M["m10"] / M["m00"])
                cY = int(M["m01"] / M["m00"])
                centroid = (cX, cY)
            except Exception as e:
                #print(e)
                centroid = None

        # Crop the sample given the center and crop_size
        cropped_sample = self.crop_sample(sample, center=centroid)

        return cropped_sample

    def crop_sample(self, sample, center=None):

        # Creating a container for the output
        cropped_sample = {}

        # H,W of the images
        h, w, _ = sample['image'].shape

        # No center is provided, then assume the center of the image
        if center is None:
            center = (w // 2, h // 2)

        # Obtaing half of the crop_size
        hcs = self.crop_size // 2 # half crop size

        # Determing the left, right, top and bottom of the image
        left = center[0] - hcs
        right = center[0] + hcs
        top = center[1] - hcs
        bottom = center[1] + hcs

        # Check if the left, right, top and bottom are outside the image
        left_pad = right_pad = top_pad = bottom_pad = 0

        if left < 0:
            left_pad = -left
            left = 0
        if right > w:
            right_pad = right - w
            right = w
        if top < 0:
            top_pad = -top
            top = 0
        if bottom > h:
            bottom_pad = bottom - h
            bottom = h 

        """
        if left_pad != 0 or right_pad != 0 or top_pad != 0 or bottom_pad != 0:
            pdb.set_trace()
        #"""

        # Crop the images in the sample
        for key in sample.keys():
            if len(sample[key].shape) == 2:
                crop = sample[key][top:bottom,left:right]
                crop_and_pad = np.pad(crop, ((top_pad,bottom_pad),(left_pad, right_pad)), mode='constant')
                cropped_sample[key] = crop_and_pad
            elif len(sample[key].shape) == 3:
                crop = sample[key][top:bottom,left:right,:]
                crop_and_pad = np.pad(crop, ((top_pad,bottom_pad),(left_pad, right_pad), (0,0)), mode='constant')
                cropped_sample[key] = crop_and_pad

        return cropped_sample

    def get_random_batched_sample(self, batch_size=1):

        batched_sample = {}

        for sample_id in np.random.choice(np.arange(self.__len__()), size=batch_size, replace=False):

            sample = self.__getitem__(sample_id)

            for key in sample.keys():

                if key in batched_sample.keys():
                    batched_sample[key] = np.concatenate([batched_sample[key], np.expand_dims(sample[key], axis=0)], axis=0)

                else:
                    batched_sample[key] = np.expand_dims(sample[key], axis=0)

        return batched_sample

class CARVANASegDataset(torch.utils.data.Dataset):

    CLASSES = ['car']
    
    def __init__(
        self,
        images: List[pathlib.Path],
        masks: List[pathlib.Path] = None,
        transforms=None
    ) -> None:

        self.images = images
        self.masks = masks
        self.transforms = transforms

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, idx: int) -> dict:
        image_path = self.images[idx]
        image = catalyst.utils.imread(image_path)
        
        result = {"image": image}
        
        if self.masks is not None:
            mask = skimage.io.imread(self.masks[idx])
            result["mask"] = mask
        
        if self.transforms is not None:
            result = self.transforms(**result)
        
        result["filename"] = image_path.name

        return result

#-------------------------------------------------------------------------------
# Pose Regression Datasets

class NOCSPoseRegDataset(torch.utils.data.Dataset):
    """NOCS Dataset. Read images, apply augmentation and preprocessing transformations.
    
    Args:
        dataset_dir (str): filepath to the dataset (train, valid, or test)
        max_size (int): maximum number of samples
        augmentation (albumentations.Compose): data transfromation pipeline 
            (e.g. flip, scale, etc.)
        preprocessing (albumentations.Compose): data preprocessing 
            (e.g. noralization, shape manipulation, etc.)
    """

    CLASSES = pj.constants.NOCS_CLASSES
    COLORMAP = pj.constants.NOCS_COLORMAP
    
    def __init__(
        self,
        dataset_dir,
        max_size=None,
        classes=None,
        augmentation=None,
        preprocessing=None,
        crop_size=100
        ):

        # Obtaining the filepaths for the images
        self.images_fps = self.get_image_paths_in_dir(dataset_dir, max_size=max_size)

        # convert str names to class values on masks
        self.class_values = [self.CLASSES.index(cls.lower()) for cls in classes]

        # Saving parameters
        self.augmentation = augmentation
        self.preprocessing = preprocessing
        self.crop_size = crop_size

    def __getitem__(self, i):

        # Reading data
        # Image
        image = skimage.io.imread(str(self.images_fps[i]))

        # Mask
        mask_fp = str(self.images_fps[i]).replace('_color.png', '_mask.png')
        mask = skimage.io.imread(mask_fp, 0)[:,:,0].astype('float')
        mask[mask == 255] = 0

        # Depth
        """
        depth_fp = str(self.images_fps[i]).replace('_color.png', '_depth.png')
        depth = skimage.io.imread(depth_fp)
        depth = dm.standardize_depth(depth)
        """

        # Other data
        json_fp = str(self.images_fps[i]).replace('_color.png', '_meta+.json')
        json_data = jt.load_from_json(json_fp)

        # Create dense quaternion
        quaternions = dm.create_dense_quaternion(mask, json_data)
        scales = dm.create_dense_scales(mask, json_data)
        #xy, z = dm.create_dense_3d_centers(mask, json_data)
        xy, z = dm.create_simple_dense_3d_centers(mask, json_data)

        # Remove objects not found within the instance_id
        correct_obj_mask = np.zeros_like(mask)
        
        for instance_id in json_data['instance_dict'].keys():
            class_id = json_data['instance_dict'][instance_id]
            correct_obj_mask += np.equal(mask, int(instance_id)) * int(class_id)

        mask = correct_obj_mask

        # Storing mask and image into sample
        sample = {
            'image': image, 
            'mask': mask, 
            'quaternions': quaternions,
            'scales': scales,
            'xy': xy,
            'z': z
        }

        # apply augmentations
        """
        if self.augmentation:
            sample = self.augmentation(**sample)
        """

        # apply preprocessing
        if self.preprocessing:
            sample = self.preprocessing(**sample)

        # Normalize to maintain the -1 to +1 magnitude
        if sample['image'].dtype != np.uint8:
            sample['image'] /= np.max(np.abs(sample['image']))

        # Changing dtype
        sample.update({
            'image': skimage.img_as_float32(sample['image']),
            'mask': sample['mask'].astype('long')
        })

        return sample

    def __len__(self):
        return len(self.images_fps)

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

            # Removing empty color images that do not have instances
            good_color_images = self.remove_empty_samples(color_images)
            total_path_list += good_color_images

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

    def remove_empty_samples(self, file_paths):

        good_samples_fps = []

        for i in range(len(file_paths)):

            # Obtain the json data
            json_fp = str(file_paths[i]).replace('_color.png', '_meta+.json')
            json_data = jt.load_from_json(json_fp)

            # Get the instance data
            instance_dict = json_data['instance_dict']

            # If instances, save them as good samples
            if instance_dict:
                good_samples_fps.append(file_paths[i])

        return good_samples_fps

    def get_random_batched_sample(self, batch_size=1):

        batched_sample = {}

        for sample_id in np.random.choice(np.arange(self.__len__()), size=batch_size, replace=False):

            sample = self.__getitem__(sample_id)

            for key in sample.keys():

                if key in batched_sample.keys():
                    batched_sample[key] = np.concatenate([batched_sample[key], np.expand_dims(sample[key], axis=0)], axis=0)

                else:
                    batched_sample[key] = np.expand_dims(sample[key], axis=0)

        return batched_sample

#-------------------------------------------------------------------------------
# Functions

def test_seg_camvid_dataset():

    preprocessing_fn = smp.encoders.get_preprocessing_fn(ENCODER, ENCODER_WEIGHTS)

    dataset = CAMVIDSegDataset(
        pj.cfg.CAMVID_DATASET,
        train_valid_test='train', 
        classes=pj.constants.CAMVID_CLASSES,
        augmentation=transforms.get_training_augmentation(), 
        preprocessing=transforms.get_preprocessing(preprocessing_fn),
        mask_dataformat='HW'
    )

    sample = dataset[1]
    pdb.set_trace()

def test_seg_nocs_dataset():

    preprocessing_fn = smp.encoders.get_preprocessing_fn(ENCODER, ENCODER_WEIGHTS)

    crop_size = 224

    print('Loading dataset')
    dataset = NOCSSegDataset(
        dataset_dir=pj.cfg.CAMERA_TRAIN_DATASET, 
        max_size=1000,
        classes=pj.constants.NOCS_CLASSES,
        augmentation=transforms.get_training_augmentation(height=crop_size, width=crop_size),
        preprocessing=transforms.get_preprocessing(preprocessing_fn),
        balance=True,
        crop_size=crop_size,
        mask_dataformat='HW'
    )
    print('Finished loading dataset')

    print("\n\nTesting dataset loading\n\n")
    sample = dataset[1]
    #sample = dataset.get_random_sample(batched=True)

    # Testing built-in dataset (Working)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=True)
    sample = next(iter(dataloader))

def test_seg_voc_dataset():

    # Creating train and valid dataset
    crop_size = (320, 480)
    voc_train = VOCSegDataset(True, crop_size, voc_dir)
    voc_valid = VOCSegDataset(False, crop_size, voc_dir)

    # Create dataloader
    batch_size = 4
    train_dataloader = torch.utils.data.DataLoader(voc_train, batch_size=batch_size, shuffle=True)
    valid_dataloader = torch.utils.data.DataLoader(voc_valid, batch_size=batch_size, shuffle=True)
    
    """
    # Test train dataloader output
    for X, Y in train_dataloader:
        print(X.shape)
        print(Y.shape)
        print(torch.unique(Y[0]))
        torchvision.utils.save_image(X[0], 'voc_train_input.png')
        torchvision.utils.save_image(Y[0], 'voc_train_label.png')
        break

    for X, Y in valid_dataloader:
        print(X.shape)
        print(Y.shape)
        print(torch.unique(Y[0]))
        torchvision.utils.save_image(X[0], 'voc_valid_input.png')
        torchvision.utils.save_image(Y[0], 'voc_valid_label.png')
        break
    #"""

def test_pose_nocs_dataset():

    preprocessing_fn = smp.encoders.get_preprocessing_fn(ENCODER, ENCODER_WEIGHTS)

    dataset = NOCSPoseRegDataset(
        dataset_dir=pj.cfg.CAMERA_TRAIN_DATASET,
        max_size=1000,
        classes=pj.constants.NOCS_CLASSES,
        augmentation=transforms.pose.get_training_augmentation(),
        preprocessing=transforms.pose.get_preprocessing(preprocessing_fn)
    )

    for id in range(20):

        sample = dataset[id]

        #vis_test = vz.get_visualized_unit_vector(sample['mask'], sample['xy'])
        #vis_test = vz.get_visualized_quaternion(sample['quaternions'])
        #vis_test = vz.get_visualized_simple_center_2d(sample['xy'])
        #vis_test = vz.get_visualized_pose(sample)
        output_data = dm.decompose_dense_representations(sample, pj.constants.CAMERA_INTRINSICS)
        
        vis_test = dr.draw_quats(
            image = sample['image'], 
            intrinsics = pj.constants.CAMERA_INTRINSICS,
            quaternions = output_data['quaternion'],
            translation_vectors = output_data['translation_vector'],
            norm_scales = output_data['scales'],
            color=(0,255,255)
        )

        fig = plt.figure()
        plt.imshow(vis_test)
        fig.savefig(f'/home/students/edavalos/GitHub/MastersProject/source_code/FastPoseCNN/test_output/global_pose/{id}.png')

    # Testing dataloader
    #dataloader = torch.utils.data.DataLoader(dataset, batch_size=2, shuffle=True)
    #sample = next(iter(dataloader))

    return 0

#-------------------------------------------------------------------------------
# Main Code

if __name__ == '__main__':
    
    # For testing the dataset
    
    # Segmentation Datasets
    #test_seg_camvid_dataset()
    #test_seg_nocs_dataset()
    #test_seg_voc_dataset()

    # Pose Datsets
    test_pose_nocs_dataset()

