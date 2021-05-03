import os
import sys
import shutil
import pathlib
import collections
from typing import List, Union
import abc
import pprint
import tqdm

import pdb
import logging

import random
import numpy as np
import cv2
import imutils
from numpy.lib.arraysetops import isin
import skimage
import skimage.io
import skimage.transform
import albumentations as albu

import torch
import torchvision
import torch.utils
import torch.utils.tensorboard
import torchvision.transforms.functional

import segmentation_models_pytorch as smp
import pytorch_lightning as pl

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.cm

# Local Imports
sys.path.append("/home/students/edavalos/GitHub/FastPoseCNN/source_code/FastPoseCNN")
import setup_env

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
LOGGER = logging.getLogger('fastposecnn')

#-------------------------------------------------------------------------------
# Dataset-related functions

def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)

#-------------------------------------------------------------------------------
# Custom Abstract Dataset

class Dataset(abc.ABC):

    @abc.abstractmethod
    def get_random_batched_sample(self):
        ...

    @property
    @abc.abstractmethod
    def CLASSES(self) -> List:
        ...

    @property
    @abc.abstractmethod
    def SYMMETRIC_CLASSES(self) -> List:
        ...

    @property
    @abc.abstractmethod
    def COLORMAP(self) -> np.ndarray:
        ...

    @property
    @abc.abstractmethod
    def INTRINSICS(self) -> np.ndarray:
        ...

#-------------------------------------------------------------------------------
# Pose Regression Datasets

class NOCSDataset(Dataset, torch.utils.data.Dataset):
    """NOCS Dataset. Read images, apply augmentation and preprocessing transformations.
    
    Args:
        dataset_dir (str): filepath to the dataset (train, valid, or test)
        max_size (int): maximum number of samples
        augmentation (albumentations.Compose): data transfromation pipeline 
            (e.g. flip, scale, etc.)
        preprocessing (albumentations.Compose): data preprocessing 
            (e.g. noralization, shape manipulation, etc.)
    """
    
    def __init__(
        self,
        dataset_dir: pathlib.Path,
        max_size: Union[int, None] = None,
        classes: Union[List, None] = None,
        augmentation: Union[albu.Compose, None] = None,
        preprocessing: Union[albu.Compose, None] = None
        ):

        # ! Debugging
        self.counter = 0

        # If None or just all the classes, no nead of class values map
        if classes is None:
            self.classes = self.CLASSES
        
        # then create class values map
        self.classes = classes
        self.class_values_map = {self.CLASSES.index(cls.lower()):self.classes.index(cls) for cls in self.classes}
        self.symmetric_classes = [self.classes.index(cls.lower()) for cls in self.SYMMETRIC_CLASSES if cls in self.classes]

        # Obtaining the filepaths for the images
        self.images_fps = self.get_image_paths_in_dir(dataset_dir, max_size=max_size)

        # Saving parameters
        self.augmentation = augmentation
        self.preprocessing = preprocessing

    def __getitem__(self, i):

        # DEBUGGING
        """
        if self.counter == 0:
            image_path = '/home/students/edavalos/GitHub/FastPoseCNN/datasets/NOCS/camera/train/27102/0001_color.png'
        elif self.counter == 1:
            image_path = '/home/students/edavalos/GitHub/FastPoseCNN/datasets/NOCS/camera/train/16523/0008_color.png'
        elif self.counter == 2:
            image_path = '/home/students/edavalos/GitHub/FastPoseCNN/datasets/NOCS/camera/train/10915/0000_color.png'
        
        self.images_fps[i] = pathlib.Path(image_path)
        self.counter += 1
        self.counter = (self.counter % 3)
        #"""

        #LOGGER.debug(f"LOADING IMAGE: {str(self.images_fps[i])}")

        # Reading data
        # Image
        image = skimage.io.imread(str(self.images_fps[i]))

        # Mask
        mask_fp = str(self.images_fps[i]).replace('_color.png', '_mask.png')
        
        if isinstance(self, CAMERADataset):
            mask = skimage.io.imread(mask_fp, 0)[:,:,0].astype('float')
        elif isinstance(self, REALDataset):
            mask = skimage.io.imread(mask_fp, 0).astype('float')
        else:
            raise NotImplementedError("Invalid dataset type")
        
        # Change the background from 255 to 0
        mask[mask == 255] = 0

        # Depth
        depth_fp = str(self.images_fps[i]).replace('_color.png', '_depth.png')
        depth = cv2.imread(depth_fp, -1) # has to be cv2, not skimage
        depth = dm.standardize_depth(depth)

        # Other data
        json_fp = str(self.images_fps[i]).replace('_color.png', '_meta+.json')
        json_data = jt.load_from_json(json_fp)

        # Removing distraction objects
        instances_mask = np.zeros_like(mask)
        for instance_id in json_data['instance_dict'].keys():
            instances_mask[mask == int(instance_id)] = int(instance_id)

        # Removing unwanted classes
        good_json_data = {'instance_dict': {}}
        good_instances_mask = np.zeros_like(instances_mask)

        # Iterating through all json data and keeping only data that is the selected class
        for enumerate_id, (id_value, class_id) in enumerate(json_data['instance_dict'].items()):

            # If the class is in the wanted class values, then keep it
            if class_id in self.class_values_map.keys():
                good_instances_mask[instances_mask == int(id_value)] = int(id_value)

                good_json_data['instance_dict'][int(id_value)] = self.class_values_map[class_id]

                for key in json_data.keys():
                    if key != 'instance_dict':
                        if key in good_json_data.keys():
                            good_json_data[key].append(json_data[key][enumerate_id])
                        else:
                            good_json_data[key] = [json_data[key][enumerate_id]]

        # Stacking data
        for key in good_json_data.keys():
            if key != 'instance_dict':
                good_json_data[key] = np.stack(good_json_data[key])

        # Generate the aggregated data (instance-lead data)
        agg_data = self.generate_agg_data(good_instances_mask, good_json_data)

        # Check if any data is invalid, if it is, simply return None for the sample
        if (agg_data['z'] <= 0).any():
            LOGGER.debug(f"{str(self.images_fps[i])} -> INVALID/CORRUPT SAMPLE: z <= 0, class_ids: {agg_data['class_ids']}")
            return None

        # Create dense representation of the data (class type is instances)
        #quaternions = dm.create_dense_quaternion(good_instances_mask, good_json_data)
        #scales = dm.create_dense_scales(good_instances_mask, good_json_data)
        #xy, z = dm.create_dense_3d_centers(good_instances_mask, good_json_data, self.INTRINSICS)

        # Generating class mask with the desired object class
        class_mask = np.zeros_like(good_instances_mask)
        for instance_id, class_id in good_json_data['instance_dict'].items():
            class_mask[good_instances_mask == int(instance_id)] = class_id

        # Storing mask and image into sample
        sample = {
            'clean_image': image,
            'image': image, 
            'mask': class_mask, 
            'depth': depth # For testing purposes only
            #'quaternion': quaternions,
            #'scales': scales,
            #'xy': xy,
            #'z': z
        }

        # apply augmentations
        """
        if self.augmentation:
            sample = self.augmentation(**sample)
        """

        # apply preprocessing
        if self.preprocessing:
            sample = self.preprocessing(**sample)

        # Converting numpyt to Torch dataformat convention
        sample = transforms.pose.numpy_to_torch()(**sample)

        # Normalize to maintain the -1 to +1 magnitude
        if sample['image'].dtype != np.uint8:
            sample['image'] /= np.max(np.abs(sample['image']))

        # Changing dtype
        sample.update({
            'path': self.images_fps[i],
            'image': skimage.img_as_float32(sample['image']),
            'mask': sample['mask'].astype('long'),
            'depth': sample['depth'].astype('float32'),
            #'quaternion': skimage.img_as_float32(sample['quaternion']),
            #'scales': skimage.img_as_float32(sample['scales']),
            #'xy': skimage.img_as_float32(sample['xy']),
            #'z': skimage.img_as_float32(sample['z']),
            'agg_data': agg_data
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

            # Trim the classes that are not wanted
            good_instance_dict = {}
            for id_value, class_value in json_data['instance_dict'].items():

                # If the class is in the wanted class values, then keep it
                if class_value in self.class_values_map.keys():
                    good_instance_dict[int(id_value)] = self.class_values_map[class_value]

            # If instances, save them as good samples
            if good_instance_dict:
                good_samples_fps.append(file_paths[i])

        return good_samples_fps

    def get_random_batched_sample(self, batch_size=1, device=None):

        all_samples = []

        # Obtain all the sample needed
        for sample_id in np.random.choice(np.arange(self.__len__()), size=batch_size, replace=False):
            sample = self.__getitem__(sample_id)
            all_samples.append(sample)

        # Collate the samples
        batch = my_collate_fn(all_samples, device)

        return batch        

    def generate_agg_data(self, instances_mask, json_data):

        # Obtaing the size of the images
        h, w = instances_mask.shape

        # Determining the number of instances
        num_of_instances = np.unique(instances_mask).shape[0] - 1

        # Creating a pure instances image
        agg_data = {
            'class_ids': np.zeros((num_of_instances,)),
            'symmetric_ids': np.zeros((num_of_instances,)),
            'instance_masks': np.zeros((num_of_instances, h, w)),
            'quaternion': np.zeros((num_of_instances, 4)),
            'scales': np.zeros((num_of_instances, 3)),
            'xy': np.zeros((num_of_instances, 2)),
            'z': np.zeros((num_of_instances, 1)),
            'T': np.zeros((num_of_instances, 3)),
            'R': np.zeros((num_of_instances, 3, 3)),
            'RT': np.zeros((num_of_instances, 4, 4))
        }

        # Modify the json data to work well (missing xy, z, R, and T, and 
        # renaming quaternions to quaternion)
        json_data['quaternion'] = json_data['quaternions']
        json_data['RT'] = json_data['RTs']
        xyz_R_T = dm.extract_xyz_R_T_from_RTs(json_data['RTs'], self.INTRINSICS)
        json_data.update(xyz_R_T)

        # Filling in the data for each instance
        for enumerate_id, (instance_id, class_id) in enumerate(json_data['instance_dict'].items()):
            for data_name in agg_data.keys():
                
                # Different to account for datatype mismatch
                if data_name == 'class_ids': 
                    agg_data[data_name][enumerate_id] = json_data['instance_dict'][instance_id]
                
                # Marking 1 for symmetric, 0 for non-symmetric classes
                elif data_name == 'symmetric_ids':
                    symmetric_ids = 1 * (json_data['instance_dict'][instance_id] in self.symmetric_classes)
                    agg_data[data_name][enumerate_id] = symmetric_ids

                # Use the mask instead of json_data
                elif data_name == 'instance_masks': 
                    agg_data[data_name][enumerate_id] = np.where(instances_mask == instance_id, 1, 0)
                
                else:
                    # Grabbing the data to test data type
                    insert_data = json_data[data_name]
                    # Ensuring that the data is an numpy array
                    if isinstance(instance_id, np.ndarray) is False:
                        insert_data = np.array(insert_data)
                    # Storing data into larger numpy array (instances)
                    agg_data[data_name][enumerate_id] = insert_data[enumerate_id]

        # Reducing the size of the object
        agg_data['scales'] /= np.expand_dims(json_data['norm_factors'], axis=1)

        # Flip the xy to match the other style
        agg_data['xy'] = np.flip(agg_data['xy'], axis=1)

        return agg_data

class CAMERADataset(NOCSDataset):

    CLASSES = pj.constants.CAMERA_CLASSES
    SYMMETRIC_CLASSES = pj.constants.CAMERA_SYMMETRIC_CLASSES
    COLORMAP = pj.constants.COLORMAP['CAMERA']
    INTRINSICS = pj.constants.INTRINSICS['CAMERA']

class REALDataset(NOCSDataset):

    CLASSES = pj.constants.CAMERA_CLASSES
    SYMMETRIC_CLASSES = pj.constants.REAL_SYMMETRIC_CLASSES
    COLORMAP = pj.constants.COLORMAP['REAL']
    INTRINSICS = pj.constants.INTRINSICS['REAL']

#-------------------------------------------------------------------------------
# Dataloader

def my_collate_fn(batch, device=None):

    # Filtering any samples that were deemed corrupted or invalid
    batch = list(filter(lambda x : x is not None, batch))

    # If the batch is empty, simply return None
    if not batch:
        return None

    collate_batch = {}
    agg_data = {
        'sample_ids': []
    }

    # Iterating overall the data input data
    for sample_id, single_sample in enumerate(batch):
        for key in single_sample.keys():
            
            # Instance-based data
            if key == 'agg_data':
                continue
            
            # Placing all the same type of data into their containers
            if key in collate_batch.keys():
                collate_batch[key].append(single_sample[key])
            else:
                collate_batch[key] = [single_sample[key]]

        # If there is agg_data, the np.concat it
        if 'agg_data' in single_sample.keys():

            # Get subkey from the agg_data dictionary
            for subkey in single_sample['agg_data'].keys():

                # Placing all the same type of data into their containers
                if subkey in agg_data.keys():
                    agg_data[subkey].append(single_sample['agg_data'][subkey])
                else:
                    agg_data[subkey] = [single_sample['agg_data'][subkey]]

            # Appending sample_ids key to this to mark the sample to instances
            agg_data['sample_ids'].append(np.repeat(np.array(sample_id), single_sample['agg_data']['class_ids'].shape[0]))

    # Containers for stacked and concatinated data
    stacked_collate_batch = {}
    concated_agg_data = {}

    # If device is specified, 
    if device:
        # Now stacking all the uniform data
        for key in collate_batch.keys():
            if isinstance(collate_batch[key][0], np.ndarray):
                stacked_collate_batch[key] = torch.from_numpy(np.stack(collate_batch[key])).to(device)
            else:
                stacked_collate_batch[key] = collate_batch[key]

        # Now concatinating all non-uniform data
        for subkey in agg_data.keys():
            concated_agg_data[subkey] = torch.from_numpy(np.concatenate(agg_data[subkey], axis=0)).to(device)
    
    # If no device is specified
    else:
        # Now stacking all the uniform data
        for key in collate_batch.keys():
            if isinstance(collate_batch[key][0], np.ndarray):
                stacked_collate_batch[key] = torch.from_numpy(np.stack(collate_batch[key]))
            else:
                stacked_collate_batch[key] = collate_batch[key]

        # Now concatinating all non-uniform data
        for subkey in agg_data.keys():
            concated_agg_data[subkey] = torch.from_numpy(np.concatenate(agg_data[subkey], axis=0))

    # Now storing agg data into the collate batch
    stacked_collate_batch['agg_data'] = concated_agg_data

    return stacked_collate_batch

#-------------------------------------------------------------------------------
# PyTorch-Lightning DataModule

class PoseRegressionDataModule(pl.LightningDataModule):

    def __init__(
        self,
        dataset_name='NOCS', 
        batch_size=1, 
        num_workers=0,
        selected_classes=None,
        encoder=None,
        encoder_weights=None,
        train_size=None,
        valid_size=None,
        is_deterministic=False
        ):

        super().__init__()
        
        # Saving parameters
        self.dataset_name = dataset_name
        self.selected_classes = selected_classes
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.encoder = encoder
        self.encoder_weights = encoder_weights
        self.train_size = train_size
        self.valid_size = valid_size
        self.is_deterministic = is_deterministic

    def setup(self, stage=None):

        # Obtaining the preprocessing_fn depending on the encoder and the encoder
        # weights
        if self.encoder and self.encoder_weights:
            preprocessing_fn = smp.encoders.get_preprocessing_fn(self.encoder, self.encoder_weights)
        else:
            preprocessing_fn = None

        # CAMERA / NOCS
        if self.dataset_name == 'CAMERA':

            train_dataset = CAMERADataset(
                dataset_dir=pathlib.Path(os.getenv("NOCS_CAMERA_TRAIN_DATASET")),
                max_size=self.train_size,
                classes=self.selected_classes,
                augmentation=transforms.pose.get_training_augmentation(),
                preprocessing=transforms.pose.get_preprocessing(preprocessing_fn)
            )

            valid_dataset = CAMERADataset(
                dataset_dir=pathlib.Path(os.getenv("NOCS_CAMERA_VALID_DATASET")), 
                max_size=self.valid_size,
                classes=self.selected_classes,
                augmentation=transforms.pose.get_validation_augmentation(),
                preprocessing=transforms.pose.get_preprocessing(preprocessing_fn)
            )

            self.datasets = {
                'train': train_dataset,
                'valid': valid_dataset
            }

        # REAL / NOCS
        elif self.dataset_name == 'REAL':

            train_dataset = REALDataset(
                dataset_dir=pathlib.Path(os.getenv("NOCS_REAL_TRAIN_DATASET")),
                max_size=self.train_size,
                classes=self.selected_classes,
                augmentation=transforms.pose.get_training_augmentation(),
                preprocessing=transforms.pose.get_preprocessing(preprocessing_fn)
            )

            valid_dataset = REALDataset(
                dataset_dir=pathlib.Path(os.getenv("NOCS_REAL_TEST_DATASET")), 
                max_size=self.valid_size,
                classes=self.selected_classes,
                augmentation=transforms.pose.get_validation_augmentation(),
                preprocessing=transforms.pose.get_preprocessing(preprocessing_fn)
            )

            self.datasets = {
                'train': train_dataset,
                'valid': valid_dataset
            }
        
        # INVALID DATASET
        else:
            raise RuntimeError('Dataset needs to be selected')

        print(f"Training datset size: {len(self.datasets['train'])}")
        print(f"Validation dataset size: {len(self.datasets['valid'])}")

    def get_loader(self, dataset_key):

        if dataset_key in self.datasets.keys():        
            
            # Constructing general params
            params = {
                'dataset': self.datasets[dataset_key],
                'num_workers': self.num_workers,
                'batch_size': self.batch_size,
                'collate_fn': my_collate_fn,
                'pin_memory': True
            }

            # If deterministic, don't shuffle and provide seed_worker function
            if self.is_deterministic:
                params.update({
                    'worker_init_fn': seed_worker,
                    'shuffle': False
                })
            else: # else, enable shuffle
                params.update({
                    'shuffle': True
                })

            # Passing parameters
            dataloader = torch.utils.data.DataLoader(**params)
            
            return dataloader

        else:

            return None

    def train_dataloader(self):
        return self.get_loader('train')

    def val_dataloader(self):
        return self.get_loader('valid')

    def test_dataloader(self):
        return self.get_loader('test')

#-------------------------------------------------------------------------------
# Functions

def move_dict_to(a: dict, device):

    for key in a.keys():
        if isinstance(a[key], torch.Tensor):
            a[key] = a[key].to(device)

    return a

def move_batch_to(batch: dict, device):

    device_batch = {}

    for key in batch.keys():

        if key == 'agg_data':
            agg_data_dict = move_dict_to(batch[key], device)
            device_batch[key] = agg_data_dict

        else:

            if isinstance(batch[key], torch.Tensor):
                device_batch[key] = batch[key].to(device)
            else:
                device_batch[key] = batch[key]

    return device_batch

def test_pose_camera_dataset():

    preprocessing_fn = smp.encoders.get_preprocessing_fn(ENCODER, ENCODER_WEIGHTS)

    dataset = CAMERADataset(
        dataset_dir=pathlib.Path(os.getenv("NOCS_CAMERA_TRAIN_DATASET")),
        max_size=300,
        classes=pj.constants.CAMERA_CLASSES,
        augmentation=transforms.pose.get_training_augmentation(),
        preprocessing=transforms.pose.get_preprocessing(preprocessing_fn)
    )

    # Testing dataloader
    dataloader = torch.utils.data.DataLoader(
        dataset, 
        batch_size=3,
        collate_fn=my_collate_fn,
        shuffle=False
    )

    # Iterative through data batches
    for i, batch in tqdm.tqdm(enumerate(dataloader)):
        fig = vz.visualize_gt_pose(batch, dataset.COLORMAP, dataset.INTRINSICS)
        fig_path = pathlib.Path(os.getenv("TEST_OUTPUT")) / f'quat_pose{i}.png'
        fig.savefig(str(fig_path), dpi=400)

#-------------------------------------------------------------------------------
# Main Code

if __name__ == '__main__':
    
    # For testing the datasets

    # camera dataset
    test_pose_camera_dataset()

