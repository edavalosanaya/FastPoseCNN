import os
import sys
import pathlib

import numpy as np
import cv2
import imutils

import torch
import torchvision

# Local Imports
root = next(path for path in pathlib.Path(os.path.abspath(__file__)).parents if path.name == 'FastPoseCNN')
sys.path.append(str(pathlib.Path(__file__).parent))

import abc123
import json_tools
import data_manipulation
import project

#-------------------------------------------------------------------------------
# File Constants

DEBUG = False
camera_dataset = root.parents[1] / 'datasets' / 'NOCS' / 'camera' / 'val'

#-------------------------------------------------------------------------------
# Class

class NOCSDataset(torch.utils.data.Dataset):

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
        
        return None

    def __len__(self):
        """
        Returns the size of the dataset
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
        sample = self.new_get_data_sample(self.color_image_path_list[idx])

        if self.transform:
            sample = self.transform(sample) 

        return sample
        
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

        abc123.disable_print(DEBUG)
        print("Getting all image paths inside dataset path given")

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

        print(f"total_path_list length: {len(total_path_list)}")
        abc123.enable_print(DEBUG)

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
        abc123.disable_print(DEBUG)

        # Getting the data set ID and obtaining the corresponding file paths for the
        # mask, coordinate map, depth, and meta files
        print('Constructing all the datas file paths')
        data_id = color_image_path.name.replace('_color.png', '')
        mask_path = color_image_path.parent / f'{data_id}_mask.png'
        depth_path = color_image_path.parent / f'{data_id}_depth.png'
        coord_path = color_image_path.parent / f'{data_id}_coord.png'
        meta_plus_path = color_image_path.parent / f'{data_id}_meta+.json'

        # Loading data (color, mask, coord, depth)
        print('Loading raw image data')
        color_image = cv2.imread(str(color_image_path), cv2.IMREAD_UNCHANGED)
        mask_image = cv2.imread(str(mask_path), cv2.IMREAD_UNCHANGED)[:, :, 2]
        coord_map = cv2.imread(str(coord_path), cv2.IMREAD_UNCHANGED)[:, :, :3]
        coord_map = coord_map[:, :, (2, 1, 0)]
        depth_image = cv2.imread(str(depth_path), cv2.IMREAD_UNCHANGED)

        # Converting depth to the correct shape and dtype
        print('Converting depth into valid depth data')
        if len(depth_image.shape) == 3: # encoded depth image
            new_depth = np.uint16(depth_image[:, :, 1] * 256) + np.uint16(depth_image[:,:,2])
            new_depth = new_depth.astype(np.uint16)
            depth_image = new_depth
        elif len(depth_image.shape) == 2 and depth_image.dtype == 'uint16':
            pass # depth is perfecto!
        else:
            assert False, '[ Error ]: Unsupported depth type'
            
        # flip z axis of coord map
        print('Converting coordinate map into a valid one')
        coord_map = np.array(coord_map, dtype=np.float32) / 255
        coord_map[:, :, 2] = 1 - coord_map[:, :, 2]

        # Converting the mask into a typical mask
        print('Converting mask map into a valid one')
        mask_cdata = np.array(mask_image, dtype=np.int32)
        mask_cdata[mask_cdata==255] = -1

        # Loading data
        print('Loading JSON data')
        json_data = json_tools.load_from_json(meta_plus_path)
        instance_dict = json_data['instance_dict']
        scales = np.asarray(json_data['scales'], dtype=np.float32)
        RTs = np.asarray(json_data['RTs'], dtype=np.float32)
        norm_factors = np.asarray(json_data['norm_factors'], dtype=np.float32)
        quaternions = np.asarray(json_data['quaternions'], dtype=np.float32)

        # NOTE!
        """
        RTs follow this convention
        CAMERA SPACE --- inverse RT ---> WORLD SPACE
        WORLD SPACE  ---     RT     ---> CAMERA SPACE
        """

        # Reorganizing data to match instance ids order
        print('Organizing data to match instance ids')
        h, w = mask_cdata.shape
        num_instance = len(instance_dict.keys())
        masks = np.zeros([h, w, num_instance], dtype=np.uint8)
        coords = np.zeros((h, w, num_instance, 3), dtype=np.float32)
        class_ids = np.asarray(list(instance_dict.values()), dtype=np.int_)

        for i, instance_id in enumerate(instance_dict.keys()):
            
            instance_mask = np.equal(mask_cdata, int(instance_id))
            masks[:,:,i] = instance_mask
            coords[:,:,i,:] = np.multiply(coord_map, np.expand_dims(instance_mask, axis=-1))

        # Obtain last information
        bboxes = data_manipulation.extract_2d_bboxes_from_masks(masks)
        scores = np.asarray([100 for j in range(len(class_ids))], dtype=np.float32) # 100% for ground truth data

        # Placing all information into a convenient data structure
        print('Constructing sample dictionary')
        sample = {'color_image': color_image,
                  'instance_dict': instance_dict,
                  'class_ids': class_ids,
                  'scores': scores,
                  'bboxes': bboxes,
                  'masks': masks,
                  'coords': coords,
                  'scales': scales,
                  'norm_factors': norm_factors,
                  'RTs': RTs,
                  'quaternions': quaternions}

        abc123.enable_print(DEBUG)

        return sample

    def new_get_data_sample(self, color_image_path):
        """
        Args:
            color_image_path (pathlib object): Path to the color image 
        Objective:
            Collect all the information pertaining to the color image path given.
        Output:
            data (dictionary): Contains all the information pertaining to that
            color image.
        """
        abc123.disable_print(DEBUG)

        # Getting the data set ID and obtaining the corresponding file paths for the
        # mask, coordinate map, depth, and meta files
        print('Constructing all the datas file paths')
        data_id = color_image_path.name.replace('_color.png', '')
        mask_path = color_image_path.parent / f'{data_id}_mask.png'
        depth_path = color_image_path.parent / f'{data_id}_depth.png'
        coord_path = color_image_path.parent / f'{data_id}_coord.png'
        meta_plus_path = color_image_path.parent / f'{data_id}_meta+.json'

        # Loading data (color, mask, coord, depth)
        print('Loading raw image data')
        color_image = cv2.imread(str(color_image_path), cv2.IMREAD_UNCHANGED)
        mask_image = cv2.imread(str(mask_path), cv2.IMREAD_UNCHANGED)[:, :, 2]
        coord_map = cv2.imread(str(coord_path), cv2.IMREAD_UNCHANGED)[:, :, :3]
        coord_map = coord_map[:, :, (2, 1, 0)]
        depth_image = cv2.imread(str(depth_path), cv2.IMREAD_UNCHANGED)

        #print(f'original coord_map.shape: {coord_map.shape}')

        # Loading data
        print('Loading JSON data')
        json_data = json_tools.load_from_json(meta_plus_path)
        instance_dict = json_data['instance_dict']
        scales = np.asarray(json_data['scales'], dtype=np.float32)
        RTs = np.asarray(json_data['RTs'], dtype=np.float32)
        norm_factors = np.asarray(json_data['norm_factors'], dtype=np.float32)
        quaternions = np.asarray(json_data['quaternions'], dtype=np.float32)

        # Converting depth to the correct shape and dtype
        print('Converting depth into valid depth data')
        if len(depth_image.shape) == 3: # encoded depth image
            new_depth = np.uint16(depth_image[:, :, 1] * 256) + np.uint16(depth_image[:,:,2])
            new_depth = new_depth.astype(np.uint16)
            depth_image = new_depth
        elif len(depth_image.shape) == 2 and depth_image.dtype == 'uint16':
            pass # depth is perfecto!
        else:
            assert False, '[ Error ]: Unsupported depth type'
            
        # flip z axis of coord map
        print('Converting coordinate map into a valid one')
        coord_map = np.array(coord_map, dtype=np.float32) / 255
        coord_map[:, :, 2] = 1 - coord_map[:, :, 2]

        # Converting the mask into a typical mask
        print('Converting mask map into a valid one')
        mask_cdata = np.array(mask_image, dtype=np.int32)
        mask_cdata[mask_cdata==255] = -1

        # NOTE!
        """
        RTs follow this convention
        CAMERA SPACE --- inverse RT ---> WORLD SPACE
        WORLD SPACE  ---     RT     ---> CAMERA SPACE
        """

        num_classes = len(project.constants.SYNSET_NAMES) - 1 # Removing BG
        class_ids = list(instance_dict.values())
        h, w = mask_cdata.shape

        color_image = np.moveaxis(color_image, -1, 0)

        masks = np.zeros([num_classes, h, w], dtype=np.float32)
        coords = np.zeros([num_classes, 3, h, w], dtype=np.float32)
        quat_img = np.zeros([num_classes, 4, h, w], dtype=np.float32)
        scales_img = np.zeros([num_classes, 3, h, w], dtype=np.float32)
        
        for e_id, (i_id, c_id) in enumerate(instance_dict.items()):

            c_id -= 1 # shifting left to remove BG

            instance_mask = np.equal(mask_cdata, int(i_id)) * 1 # the * 1 is to convert boolean to binary
            masks[c_id,:,:] += instance_mask
            coords[c_id,:,:,:] += np.moveaxis(np.multiply(coord_map, np.expand_dims(instance_mask, axis=-1)), -1, 0)
            
            quaternion = quaternions[e_id,:]
            #print(f'quaternion: {quaternion} - quaternion.shape: {quaternion.shape}')
            temp_quaterion = np.expand_dims(np.expand_dims(quaternion, axis=-1), axis=-1)
            #print(f'tem_quaterion.shape: {temp_quaterion}')
            #print(f'quat_img[c_id,:,:,:].shape: {quat_img[c_id,:,:,:].shape}')
            quat_img[c_id,:,:,:] = np.where(np.expand_dims(masks[c_id,:,:], axis=0), temp_quaterion, quat_img[c_id,:,:,:])

            scale = scales[e_id,:]
            temp_scales = np.expand_dims(np.expand_dims(scale, axis=-1), axis=-1)
            scales_img[c_id,:,:,:] = np.where(np.expand_dims(masks[c_id,:,:], axis=0), temp_scales, scales_img[c_id,:,:,:])


        # Placing all information into a convenient data structure
        print('Constructing sample dictionary')
        sample = {'color_image': color_image,
                  'masks': masks,
                  'coords': coords,
                  'scales': scales_img,
                  'quaternions': quat_img,
                  'norm_factors': norm_factors}

        abc123.enable_print(DEBUG)

        return sample 

class NOCSDataLoader(torch.utils.data.DataLoader):

    def __init__(self, *args, **kwargs):

        super().__init__(*args, collate_fn = self.my_collate, **kwargs)

        return None

    def my_collate(self, batch):
        """
        Args: batch (list), a list of data samples
        Objective: collate data samples into a single batch
        Output: collate_batch (dictionary)
        Help here: https://discuss.pytorch.org/t/how-to-create-a-dataloader-with-variable-size-input/8278/3

        sample = {'color_image': color_image,
                  'masks': masks,
                  'coords': coords,
                  'scales': scales,
                  'quaternions': quat_img,
                  'norm_factors': norm_factors}

        """

        collate_batch = {}
        all_data = {}

        for key in batch[0].keys():
            all_data[key] = [sample[key] for sample in batch]

        # Format all data
        all_data = self.format_data(all_data)

        # Stacking all the data in all_data dictionary, but while considering datatype
        collate_batch = self.stack_data(all_data, collate_batch)

        return collate_batch

    def format_data(self, all_data):
        """
        Converting all data from numpy to torch
        """
        
        for key in all_data.keys():

            for id in range(len(all_data[key])):

                x = all_data[key][id]
                x = torch.from_numpy(x)
                x = x.type(torch.FloatTensor)
                all_data[key][id] = x

        return all_data

    def stack_data(self, all_data, collate_batch):
        """
        Stacking all data
        """

        for key in all_data.keys():

            if key == 'norm_factors':
                collate_batch[key] = all_data[key]
                continue

            collate_batch[key] = torch.stack(all_data[key])

        return collate_batch

#-------------------------------------------------------------------------------
# Functions

def my_collate(batch):
    """
    Args: batch (list), a list of data samples
    Objective: collate data samples into a single batch
    Output: collate_batch (dictionary)
    Help here: https://discuss.pytorch.org/t/how-to-create-a-dataloader-with-variable-size-input/8278/3

    sample = {'color_image': color_image,
              'instance_dict': instance_dict,
              'class_ids': class_ids,
              'scores': scores,
              'bboxes': bboxes,
              'masks': masks,
              'coords': coords,
              'scales': scales,
              'norm_factors': norm_factors,
              'RTs': RTs}

    """

    collate_batch = {}
    
    # If the batch size is greater than 1
    if len(batch) > 1: 

        #print('batch size is larger than 1')

        # Grab the other unused samples and append them to collate_batch
        # while accounting for the data type associated with the key
        all_data = {}
        for key in batch[0].keys():
            all_data[key] = [sample[key] for sample in batch]

        #print(f'all_data: {all_data}')

        # Stacking all the data in all_data dictionary, but while considering datatype
        for key in all_data.keys():
            if key in ['color_image']: # Stacking images (input to the neural network)
                collate_batch[key] = torch.from_numpy(np.stack(all_data[key]))
            else: # Putting everything else in list because they don't necessary have the same size
                collate_batch[key] = all_data[key]

    else: # batch size = 1

        #print('batch size is 1')
        collate_batch = batch

    return collate_batch

#-------------------------------------------------------------------------------
# Main Code

if __name__ == '__main__':
    # For testing the dataset

    # For randomly splitting datasets: https://pytorch.org/docs/stable/data.html#torch.utils.data.random_split

    print('Loading dataset')
    dataset = NOCSDataset(camera_dataset, dataset_max_size=10)
    print('Finished loading dataset')

    # Testing dataset
    print("\n\nTesting dataset loading\n\n")
    test_sample = dataset[0]

    """
    for key in test_sample.keys():
        shape_info = test_sample[key].shape if type(test_sample[key]) == np.ndarray else len(test_sample[key])
        unique = np.unique(test_sample[key]) if key in ['masks'] else None
        print(f'key: {key} - type: {type(test_sample[key])} - shape: {shape_info} - unique: {unique}')
    """

    dataloader = NOCSDataLoader(dataset, batch_size=2, shuffle=True, num_workers=0)

    # Testing dataloader
    print("\n\nTesting dataloader loading\n\n")
    for i, batched_sample in enumerate(dataloader):

        print(batched_sample.keys())

        color_image = batched_sample['color_image'][0]
        color_image = color_image.permute(1, 2, 0).numpy().astype(np.int32).copy()
        print(color_image.shape)
        print(color_image.dtype)

        batched_masks = batched_sample['masks']
        masks = batched_masks[0]
        masks = masks * 255
        print(masks.shape)

        class_centroids = data_manipulation.get_masks_centroids(masks)
    
        for c_id in range(len(class_centroids)):
            centroids = class_centroids[c_id]

            for i_id, centroid in enumerate(centroids):

                cX, cY = centroid

                cv2.circle(color_image, centroid, 4, (0,255,0), -1)
                label = f'{project.constants.SYNSET_NAMES[c_id+1]}: {i_id}'
                cv2.putText(color_image, label, (cX-20, cY-20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 1)

        cv2.imwrite('color_image.png', color_image)
        break
