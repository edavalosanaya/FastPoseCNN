import os
import sys
import pathlib
import tqdm
import pdb
import argparse

import cv2
import numpy as np
from pyquaternion import Quaternion

import matplotlib.pyplot as plt

# Acknowledge the root of the project
root = next(path for path in pathlib.Path(os.path.abspath(__file__)).parents if path.name == 'FastPoseCNN')

# NOCS Imports
sys.path.append(str(root.parents[1] / 'networks' / 'NOCS_CVPR2019'))
import utils as nocs_utils

# local Imports (new source code)
sys.path.append(str(root))

import project as pj
import json_tools as jt
import data_manipulation as dm
import draw as dr

# ! from matplotlib.mlab import PCA -> matplotlib==2.2.2

#-------------------------------------------------------------------------------
# File Constants

DEBUG = False
FAULTY_IMAGES = []

#-------------------------------------------------------------------------------
# Small Helper Functions

def disable_print():
    sys.stdout = open(os.devnull, 'w')

def enable_print():
    sys.stdout = sys.__stdout__

def get_image_paths_in_dir(dir_path, max_size=None):
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

#-------------------------------------------------------------------------------
# Original Dataset Access (NOCS)

def get_camera_original_information(color_image):

    # Getting the data set ID and obtaining the corresponding file paths for the
    # mask, coordinate map, depth, and meta files
    data_id = color_image.name.replace('_color.png', '')
    mask_path = color_image.parent / f'{data_id}_mask.png'
    depth_path = color_image.parent / f'{data_id}_depth.png'
    coord_path = color_image.parent / f'{data_id}_coord.png'
    meta_path = color_image.parent / f'{data_id}_meta.txt'

    # Loading data
    color_image = cv2.imread(str(color_image), cv2.IMREAD_UNCHANGED)
    mask_image = cv2.imread(str(mask_path), cv2.IMREAD_UNCHANGED)[:, :, 2]
    coord_map = cv2.imread(str(coord_path), cv2.IMREAD_UNCHANGED)[:, :, :3]
    depth_image = cv2.imread(str(depth_path), cv2.IMREAD_UNCHANGED)

    # Normalizing data
    coord_map = coord_map[:, :, (2, 1, 0)]

    # Converting depth to the correct shape and dtype
    if len(depth_image.shape) == 3: # encoded depth image
        new_depth = np.uint16(depth_image[:, :, 1] * 256) + np.uint16(depth_image[:,:,2])
        new_depth = new_depth.astype(np.uint16)
        depth_image = new_depth
    elif len(depth_image.shape) == 2 and depth_image.dtype == 'uint16':
        pass # depth is perfecto!
    else:
        assert False, '[ Error ]: Unsupported depth type'

    # flip z axis of coord map
    coord_map = np.array(coord_map, dtype=np.float32) / 255
    coord_map[:, :, 2] = 1 - coord_map[:, :, 2]

    instance_dict = {}

    # Loading instance information from meta file
    with open(str(meta_path), 'r') as f:
        for line in f:
            line_info = line.split(' ')
            instance_id = int(line_info[0])
            class_id = int(line_info[1])
            instance_dict[instance_id] = class_id

            #print(f"Instance #{instance_id}: {class_id} = {pj.constants.NOCS_CLASSES[class_id]}")

    cdata = np.array(mask_image, dtype=np.int32)

    # The value of the mask, from 0 to 255, is the class id
    instance_ids = list(np.unique(cdata))
    instance_ids = sorted(instance_ids)

    #print(f'instance_ids: {instance_ids}')

    # removing background
    assert instance_ids[-1] == 255
    del instance_ids[-1]

    cdata[cdata==255] = -1
    assert(np.unique(cdata).shape[0] < 20)

    num_instance = len(instance_ids)
    h, w = cdata.shape

    masks = np.zeros([h, w, num_instance], dtype=np.uint8)
    coords = np.zeros((h, w, num_instance, 3), dtype=np.float32)
    class_ids = np.zeros([num_instance], dtype=np.int_)
    scales = np.zeros([num_instance, 3], dtype=np.float32)

    # Determing the scales (of the 3d bounding boxes)
    with open(str(meta_path), 'r') as f:

        lines = f.readlines()
        scale_factor = np.zeros((len(lines), 3), dtype=np.float32)

        for i, line in enumerate(lines):
            words = line[:-1].split(' ')

            symmetry_id = words[2]
            reference_id = words[3]

            bbox_file = OBJ_MODEL_DIR / symmetry_id / reference_id / 'bbox.txt'
            bbox = np.loadtxt(str(bbox_file))

            value = bbox[0, :] - bbox[1, :]
            scale_factor[i, :] = value # [a b c] for scale of NOCS [x y z]

    # Deleting background objects and non-existing objects
    instance_dict = {instance_id: class_id for instance_id, class_id in instance_dict.items() if (class_id != 0 and instance_id in instance_ids)}

    i = 0
    for instance_id in instance_ids:

        if instance_id not in instance_dict.keys():
            continue
        instance_mask = np.equal(cdata, instance_id)
        assert np.sum(instance_mask) > 0
        assert instance_dict[instance_id]

        masks[:,:, i] = instance_mask
        coords[:,:,i,:] = np.multiply(coord_map, np.expand_dims(instance_mask, axis=-1))

        # class ids are also one-indexed
        class_ids[i] = instance_dict[instance_id]
        scales[i, :] = scale_factor[instance_id - 1, :]
        i += 1

    masks = masks[:, :, :i]
    coords = coords[:, :, :i, :]
    coords = np.clip(coords, 0, 1) # normalize
    class_ids = class_ids[:i]
    scales = scales[:i]
    bboxes = dm.extract_2d_bboxes_from_masks(masks)
    scores = [100 for j in range(len(class_ids))]

    # RT pickled
    RT_json_path = pathlib.Path.cwd() / f'RT_{data_id}.json'

    # now obtaining the rotation and translation
    if RT_json_path.exists():
        data = jt.load_from_json(RT_json_path)
        RTs = data['RTs']
        
    else:

        # Disable print
        disable_print()

        RTs, _, _, _ = nocs_utils.align(class_ids, masks, coords, depth_image, pj.constants.INTRINSICS['CAMERA'],
                                        pj.constants.CAMERA_CLASSES, ".", None)

        # Enable print
        enable_print()

        data = {'RTs': RTs}
        #jt.save_to_json(RT_json_path, data)

    return class_ids, bboxes, masks, coords, RTs, scores, scales, instance_dict

def get_real_original_information(color_path):

    # Getting the data set ID and obtaining the corresponding file paths for the
    # mask, coordinate map, depth, and meta files
    data_id = color_path.name.replace('_color.png', '')
    mask_path = color_path.parent / f'{data_id}_mask.png'
    depth_path = color_path.parent / f'{data_id}_depth.png'
    coord_path = color_path.parent / f'{data_id}_coord.png'
    meta_path = color_path.parent / f'{data_id}_meta.txt'

    # Loading data
    color_image = cv2.imread(str(color_path), cv2.IMREAD_UNCHANGED)
    mask_image = cv2.imread(str(mask_path), cv2.IMREAD_UNCHANGED)
    coord_map = cv2.imread(str(coord_path), cv2.IMREAD_UNCHANGED)[:, :, :3]
    depth_image = cv2.imread(str(depth_path), cv2.IMREAD_UNCHANGED)

    # Normalizing data
    coord_map = coord_map[:, :, (2, 1, 0)]

    # Converting depth to the correct shape and dtype
    if len(depth_image.shape) == 3: # encoded depth image
        new_depth = np.uint16(depth_image[:, :, 1] * 256) + np.uint16(depth_image[:,:,2])
        new_depth = new_depth.astype(np.uint16)
        depth_image = new_depth
    elif len(depth_image.shape) == 2 and depth_image.dtype == 'uint16':
        pass # depth is perfecto!
    else:
        assert False, '[ Error ]: Unsupported depth type'

    # flip z axis of coord map
    coord_map = np.array(coord_map, dtype=np.float32) / 255
    coord_map[:, :, 2] = 1 - coord_map[:, :, 2]

    instance_dict = {}

    # Loading instance information from meta file
    with open(str(meta_path), 'r') as f:
        for line in f:
            line_info = line.split(' ')
            instance_id = int(line_info[0])
            class_id = int(line_info[1])
            instance_dict[instance_id] = class_id

            #print(f"Instance #{instance_id}: {class_id} = {pj.constants.NOCS_CLASSES[class_id]}")

    cdata = np.array(mask_image, dtype=np.int32)

    # The value of the mask, from 0 to 255, is the class id
    instance_ids = list(np.unique(cdata))
    instance_ids = sorted(instance_ids)

    #print(f'instance_ids: {instance_ids}')

    # removing background
    assert instance_ids[-1] == 255
    del instance_ids[-1]

    cdata[cdata==255] = -1
    assert(np.unique(cdata).shape[0] < 20)

    num_instance = len(instance_ids)
    h, w = cdata.shape

    masks = np.zeros([h, w, num_instance], dtype=np.uint8)
    coords = np.zeros((h, w, num_instance, 3), dtype=np.float32)
    class_ids = np.zeros([num_instance], dtype=np.int_)
    scales = np.zeros([num_instance, 3], dtype=np.float32)

    # Determing the scales (of the 3d bounding boxes)
    with open(str(meta_path), 'r') as f:

        lines = f.readlines()
        scale_factor = np.zeros((len(lines), 3), dtype=np.float32)

        for i, line in enumerate(lines):
            words = line[:-1].split(' ')

            object_instance_name = words[2]

            bbox_file = OBJ_MODEL_DIR / f'{object_instance_name}.txt'
            bbox = np.loadtxt(str(bbox_file))

            value = bbox #bbox[0, :] - bbox[1, :]
            scale_factor[i, :] = value # [a b c] for scale of NOCS [x y z]
            scale_factor[i, :] /= np.linalg.norm(scale_factor[i, :])

    # Deleting background objects and non-existing objects
    instance_dict = {instance_id: class_id for instance_id, class_id in instance_dict.items() if (class_id != 0 and instance_id in instance_ids)}

    i = 0
    for instance_id in instance_ids:

        if instance_id not in instance_dict.keys():
            continue
        instance_mask = np.equal(cdata, instance_id)
        assert np.sum(instance_mask) > 0
        assert instance_dict[instance_id]

        masks[:,:, i] = instance_mask
        coords[:,:,i,:] = np.multiply(coord_map, np.expand_dims(instance_mask, axis=-1))

        # class ids are also one-indexed
        class_ids[i] = instance_dict[instance_id]
        scales[i, :] = scale_factor[instance_id - 1, :]
        i += 1

    masks = masks[:, :, :i]
    coords = coords[:, :, :i, :]
    coords = np.clip(coords, 0, 1) # normalize
    class_ids = class_ids[:i]
    scales = scales[:i]
    bboxes = dm.extract_2d_bboxes_from_masks(masks)
    scores = [100 for j in range(len(class_ids))]

    # RT pickled
    RT_json_path = pathlib.Path.cwd() / f'RT_{data_id}.json'

    # now obtaining the rotation and translation
    if RT_json_path.exists():
        data = jt.load_from_json(RT_json_path)
        RTs = data['RTs']
        
    else:

        # Disable print
        disable_print()

        RTs, _, _, _ = nocs_utils.align(class_ids, masks, coords, depth_image, pj.constants.INTRINSICS['REAL'],
                                        pj.constants.REAL_CLASSES, ".", None)

        # Enable print
        enable_print()

        data = {'RTs': RTs}
        #jt.save_to_json(RT_json_path, data)

    return class_ids, bboxes, masks, coords, RTs, scores, scales, instance_dict

    return None

def test_original_data_organization():

    color_image = cv2.imread(str(image_path), cv2.IMREAD_UNCHANGED)

    for name, data_handling_function in {'manual': manual_data_handling, 'nocs_util': nocs_util_data_handling}.items():

        print('\n\n')
        print('#'*50)
        print(f'DATA HANDLING METHOD: {name}')

        class_ids, bboxes, masks, coords, RTs, scores, scales, instance_dict = data_handling_function()

        print("\n\n")
        print("ALL RTS")
        print(RTs)

        # Drawing the output

        """
        # Version 1 (drawing)
        drawn_image = tools.visualize.draw_detections(color_image, image_id, pj.constants.INTRINSICS, pj.constants.NOCS_CLASSES, 
                                                    bboxes, class_ids, masks, coords, RTs, scores, scales,
                                                    draw_coord=True, draw_tag=False, draw_RT=True)

        cv2.imshow("output", drawn_image)
        cv2.waitKey(0)

        """

        # Version 2 (drawing)
        nocs_model = tools.models.NOCS(load_model=False)
        output = nocs_model.draw_detections(color_image, pj.constants.INTRINSICS, pj.constants.NOCS_CLASSES, bboxes, class_ids,
                                            masks, coords, RTs, scores, scales, draw_coord=True, draw_tag=True, draw_RT=True)

        cv2.imshow(f'{name} output', output)
        break

    cv2.waitKey(0)

    # Saving data into a json file
    data = {'instance_dict': instance_dict, 'scales': scales, 'RTs': RTs}
    jt.save_to_json(pathlib.Path.cwd() / 'first_json.json', data)

def nocs_util_data_handling():

    local_image_id = int(image_id)

    config = nocs_eval.InferenceConfig()
    config.OBJ_MODEL_DIR = r'E:\MASTERS_STUFF\MastersProject\networks\NOCS_CVPR2019\data\obj_models'
    dataset = nocs_dataset.NOCSDataset(pj.constants.NOCS_CLASSES, 'val', config)
    dataset.load_camera_scenes(str(CAMERA_DATASET))
    dataset.prepare(pj.constants.CLASS_MAP)

    image = dataset.load_image(local_image_id)
    depth = dataset.load_depth(local_image_id)
    gt_mask, gt_coord, gt_class_ids, gt_scales, gt_domain_label = dataset.load_mask(local_image_id)
    gt_bbox = nocs_utils.extract_bboxes(gt_mask)
    gt_RTs, _, _, _ = nocs_utils.align(gt_class_ids, gt_mask, gt_coord, depth, pj.constants.INTRINSICS,
                            pj.constants.NOCS_CLASSES, None, None)

    # Printing all information to determine issue
    print(f'image.shape: {image.shape}')
    print(f'depth.shape: {depth.shape} depth.dtype: {depth.dtype}')
    print(f'gt_mask.shape: {gt_mask.shape}')
    print(f'gt_coord.shape: {gt_coord.shape}')
    print(f'gt_class_ids: {gt_class_ids}')
    print(f'gt_domain_label: {gt_domain_label}')
    print(f'gt_bbox: {gt_bbox}')
    print(f'gt_scales: {gt_scales}')
    print(f'gt_RTs: {gt_RTs}')

    return gt_class_ids, gt_bbox, gt_mask, gt_coord, gt_RTs, [100 for i in range(len(gt_class_ids))], gt_scales, None

def manual_data_handling():

    # Loading all raw information
    print("Loading raw data")

    instance_dict = {}

    # Loading instance information from meta file
    with open(str(meta_path), 'r') as f:
        for line in f:
            line_info = line.split(' ')
            instance_id = int(line_info[0])
            class_id = int(line_info[1])
            instance_dict[instance_id] = class_id

            #print(f"Instance #{instance_id}: {class_id} = {pj.constants.NOCS_CLASSES[class_id]}")

    # Loading data
    color_image = cv2.imread(str(image_path), cv2.IMREAD_UNCHANGED)
    mask_image = cv2.imread(str(mask_path), cv2.IMREAD_UNCHANGED)[:, :, 2]
    coord_map = cv2.imread(str(coord_path), cv2.IMREAD_UNCHANGED)[:, :, :3]
    coord_map = coord_map[:, :, (2, 1, 0)]
    depth_image = cv2.imread(str(depth_path), cv2.IMREAD_UNCHANGED)

    # Converting depth to the correct shape and dtype
    if len(depth_image.shape) == 3: # encoded depth image
        new_depth = np.uint16(depth_image[:, :, 1] * 256) + np.uint16(depth_image[:,:,2])
        new_depth = new_depth.astype(np.uint16)
        depth_image = new_depth
    elif len(depth_image.shape) == 2 and depth_image.dtype == 'uint16':
        pass # depth is perfecto!
    else:
        assert False, '[ Error ]: Unsupported depth type'

    """
    cv2.imshow('mask_image', mask_image)
    cv2.imshow('coord_map', coord_map)
    cv2.imshow('depth_image', depth_image)
    cv2.waitKey(0)
    """

    # Processing data
    print("Processing data")

    cdata = np.array(mask_image, dtype=np.int32)

    # The value of the mask, from 0 to 255, is the class id
    instance_ids = list(np.unique(cdata))
    instance_ids = sorted(instance_ids)

    #print(f'instance_ids: {instance_ids}')

    # removing background
    assert instance_ids[-1] == 255
    del instance_ids[-1]

    cdata[cdata==255] = -1
    assert(np.unique(cdata).shape[0] < 20)

    num_instance = len(instance_ids)
    h, w = cdata.shape

    # flip z axis of coord map
    coord_map = np.array(coord_map, dtype=np.float32) / 255
    coord_map[:, :, 2] = 1 - coord_map[:, :, 2]

    masks = np.zeros([h, w, num_instance], dtype=np.uint8)
    coords = np.zeros((h, w, num_instance, 3), dtype=np.float32)
    class_ids = np.zeros([num_instance], dtype=np.int_)
    scales = np.zeros([num_instance, 3], dtype=np.float32)

    # Determing the scales (of the 3d bounding boxes)
    with open(str(meta_path), 'r') as f:

        lines = f.readlines()
        scale_factor = np.zeros((len(lines), 3), dtype=np.float32)

        for i, line in enumerate(lines):
            words = line[:-1].split(' ')

            symmetry_id = words[2]
            reference_id = words[3]

            bbox_file = obj_model_dir / symmetry_id / reference_id / 'bbox.txt'
            bbox = np.loadtxt(str(bbox_file))

            value = bbox[0, :] - bbox[1, :]
            scale_factor[i, :] = value # [a b c] for scale of NOCS [x y z]

    # Deleting background objects and non-existing objects
    instance_dict = {instance_id: class_id for instance_id, class_id in instance_dict.items() if (class_id != 0 and instance_id in instance_ids)}

    i = 0
    for instance_id in instance_ids:

        if instance_id not in instance_dict.keys():
            continue
        instance_mask = np.equal(cdata, instance_id)
        assert np.sum(instance_mask) > 0
        assert instance_dict[instance_id]

        masks[:,:, i] = instance_mask
        coords[:,:,i,:] = np.multiply(coord_map, np.expand_dims(instance_mask, axis=-1))

        # class ids are also one-indexed
        class_ids[i] = instance_dict[instance_id]
        scales[i, :] = scale_factor[instance_id - 1, :]
        i += 1

    masks = masks[:, :, :i]
    coords = coords[:, :, :i, :]
    coords = np.clip(coords, 0, 1) # normalize
    class_ids = class_ids[:i]
    scales = scales[:i]
    bboxes = nocs_utils.extract_bboxes(masks)
    scores = [100 for j in range(len(class_ids))]

    """
    print(f"masks.shape: {masks.shape}")
    print(f"coords.shape: {coords.shape}")
    print(f"scales: {scales}")

    for i in range(masks.shape[2]):
        visual_mask = masks[:,:,i]
        visual_mask[visual_mask==0] = 255
        cv2.imshow(f"({pj.constants.NOCS_CLASSES[class_ids[i]]}) - Mask {i}", visual_mask)
        cv2.imshow(f"({pj.constants.NOCS_CLASSES[class_ids[i]]}) - Coords {i}", coords[:,:,i,:])

    cv2.waitKey(0)
    """

    # now obtaining the rotation and translation
    RTs, _, _, _ = nocs_utils.align(class_ids, masks, coords, depth_image, pj.constants.INTRINSICS,
                                    pj.constants.NOCS_CLASSES, ".", None)

    print(f'color_image.shape: {color_image.shape}')
    print(f'depth_image.shape: {depth_image.shape} depth_image.dtype: {depth_image.dtype}')
    print(f'mask_image.shape: {mask_image.shape} mask_image.dtype: {mask_image.dtype}')
    print(f'coord_map.shape: {coord_map.shape} coord_map.dtype: {coord_map.dtype}')
    print(f'class_ids: {class_ids}')
    print(f'domain_label: {None}')
    print(f'bboxes: {bboxes}')
    print(f'scales: {scales}')
    print(f'RTs: {RTs}')

    return class_ids, bboxes, masks, coords, RTs, scores, scales, instance_dict

#-------------------------------------------------------------------------------
# Generating Modified Dataset

def create_new_dataset():

    all_color_img_paths = get_image_paths_in_dir(DATASET_DIR, max_size=None)

    print(f'dataset size: {len(all_color_img_paths)}')

    for color_path in tqdm.tqdm(all_color_img_paths, bar_format='{l_bar}{bar:40}{r_bar}{bar:-10b}'):

        #disable_print()
        data_id = color_path.name.replace('_color.png', '')
        new_meta_filepath = color_path.parent / f'{data_id}_meta+.json'

        if new_meta_filepath.exists():
            continue

        # Obtain the ground truth data for the image
        if DATASET_NAME == 'camera':
            class_ids, bboxes, masks, coords, RTs, scores, scales, instance_dict = get_camera_original_information(color_path)
            intrinsics = pj.constants.INTRINSICS['CAMERA']
        elif DATASET_NAME == 'real':
            class_ids, bboxes, masks, coords, RTs, scores, scales, instance_dict = get_real_original_information(color_path)
            intrinsics = pj.constants.INTRINSICS['REAL']
        else:
            raise NotImplementedError("Invalid dataset_type parameter value")

        new_RTs = []
        old_RTs = []
        normalizing_factors = []
        quaternions = []
        norm_scales = np.zeros(scales.shape)
        translation_vectors = []

        # Convert RT into quaternion
        for i, RT in enumerate(RTs):

            # Making RT match the following scheme
            """
            CAMERA SPACE --- inverse RT ---> WORLD SPACE
            WORLD SPACE  ---     RT     ---> CAMERA SPACE
            """
            RT = np.linalg.inv(RT)

            # Obtaining the 3D objects that will be drawn
            xyz_axis = 0.3*np.array([[0, 0, 0], [0, 0, 1], [0, 1, 0], [1, 0, 0]]).transpose()
            bbox_3d = dm.get_3d_bbox(scales[i,:],0)

            perfect_projected_axes = dm.transform_3d_camera_coords_to_2d_quantized_projections(xyz_axis, RT, intrinsics)

            # Converting transformation matrix into quaterion with translation vector
            quaternion, translation_vector, normalizing_factor = dm.RT_2_quat(RT.copy(), normalize=True)
            
            # Need to fix the translation vector, to account for the orthogonalization of the rotation matrix
            new_RT = dm.quat_2_RT_given_T_in_camera(quaternion, translation_vector)

            # Method 1 (low accuracy)
            #new_RT = dm.fix_quaternion_T(pj.constants.INTRINSICS, RT, new_RT, normalizing_factor)
            
            # Method 2 (high accuracy)
            projected_origin = perfect_projected_axes[0,:].reshape((-1, 1))
            #origin_z = (np.linalg.inv(new_RT)[2, 3] * 1000)
            origin_z = (np.linalg.inv(new_RT)[2, 3] * 1000)


            new_translation_vector = dm.create_translation_vector(projected_origin, origin_z, intrinsics)
            new_RT = dm.quat_2_RT_given_T_in_world(quaternion, new_translation_vector)

            # Need to fix the rotation matrix, to account for the small error introducted by the orthogonalization of the rotation matrix
            #new_RT = dm.fix_quat_RT_matrix(pj.constants.INTRINSICS, RT, new_RT, pts=bbox_3d)

            # Saving parameters
            old_RTs.append(RT)
            new_RTs.append(new_RT)
            quaternions.append(quaternion)
            translation_vectors.append(new_translation_vector)
            normalizing_factors.append(normalizing_factor)
            norm_scales[i,:] = scales[i,:] / normalizing_factor
        
        """
        # Checking output
        image = cv2.imread(str(color_path), cv2.IMREAD_UNCHANGED)
        output = dr.draw_detections(image, intrinsics, None, None, class_ids,
                                            None, None, old_RTs, None, scales, [1 for i in range(len(old_RTs))],
                                            (0,0,255), False, False, True)
        output = dr.draw_detections(output, intrinsics, None, None, class_ids,
                                            None, None, new_RTs, None, scales, normalizing_factors,
                                            (255,0,0), False, False, True)
        #output = dr.draw_quat_detections(image, intrinsics, quaternions, translation_vectors, norm_scales)
        
        plt.imshow(output); plt.show()
        continue
        """
        
        # Saving output into a meta+.json
        data = {'instance_dict': instance_dict, 'scales': scales, 'RTs': new_RTs, 'norm_factors': normalizing_factors, 'quaternions': quaternions}
        jt.save_to_json(new_meta_filepath, data)

        enable_print()

    return None

#-------------------------------------------------------------------------------
# Modified Dataset Access (NOCS+)

def get_new_dataset_information(color_image):

    # Getting the data set ID and obtaining the corresponding file paths for the
    # mask, coordinate map, depth, and meta files
    data_id = color_image.name.replace('_color.png', '')
    mask_path = color_image.parent / f'{data_id}_mask.png'
    depth_path = color_image.parent / f'{data_id}_depth.png'
    coord_path = color_image.parent / f'{data_id}_coord.png'
    meta_plus_path = color_image.parent / f'{data_id}_meta+.json'

    # Loading data
    color_image = cv2.imread(str(color_image), cv2.IMREAD_UNCHANGED)
    mask_image = cv2.imread(str(mask_path), cv2.IMREAD_UNCHANGED)[:, :, 2]
    coord_map = cv2.imread(str(coord_path), cv2.IMREAD_UNCHANGED)[:, :, :3]
    coord_map = coord_map[:, :, (2, 1, 0)]
    depth_image = cv2.imread(str(depth_path), cv2.IMREAD_UNCHANGED)

    print("Processing basic data")
    # Converting depth to the correct shape and dtype
    if len(depth_image.shape) == 3: # encoded depth image
        new_depth = np.uint16(depth_image[:, :, 1] * 256) + np.uint16(depth_image[:,:,2])
        new_depth = new_depth.astype(np.uint16)
        depth_image = new_depth
    elif len(depth_image.shape) == 2 and depth_image.dtype == 'uint16':
        pass # depth is perfecto!
    else:
        assert False, '[ Error ]: Unsupported depth type'
        
    # flip z axis of coord map
    coord_map = np.array(coord_map, dtype=np.float32) / 255
    coord_map[:, :, 2] = 1 - coord_map[:, :, 2]

    # Converting the mask into a typical mask
    mask_cdata = np.array(mask_image, dtype=np.int32)
    mask_cdata[mask_cdata==255] = -1

    # Loading data
    print("Reading json data")
    json_data = jt.load_from_json(meta_plus_path)
    instance_dict = json_data['instance_dict']
    scales = np.asarray(json_data['scales'], dtype=np.float32)
    RTs = np.asarray(json_data['RTs'], dtype=np.float32)
    normalization_factors = np.asarray(json_data['norm_factors'], dtype=np.float32)

    # Reorganizing data to match instance ids order
    print("Reorganizing basic data to match order")
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
    bboxes = dm.extract_2d_bboxes_from_masks(masks)
    scores = [100 for j in range(len(class_ids))] # 100% for ground truth data

    return class_ids, bboxes, masks, coords, RTs, scores, scales, instance_dict, normalization_factors

def test_new_data_organization():

    print("Loading all basic data")
    color_image = cv2.imread(str(image_path), cv2.IMREAD_UNCHANGED)
    mask_image = cv2.imread(str(mask_path), cv2.IMREAD_UNCHANGED)[:, :, 2]
    coord_map = cv2.imread(str(coord_path), cv2.IMREAD_UNCHANGED)[:, :, :3]
    coord_map = coord_map[:, :, (2, 1, 0)]
    depth_image = cv2.imread(str(depth_path), cv2.IMREAD_UNCHANGED)

    print("Processing basic data")
    # Converting depth to the correct shape and dtype
    if len(depth_image.shape) == 3: # encoded depth image
        new_depth = np.uint16(depth_image[:, :, 1] * 256) + np.uint16(depth_image[:,:,2])
        new_depth = new_depth.astype(np.uint16)
        depth_image = new_depth
    elif len(depth_image.shape) == 2 and depth_image.dtype == 'uint16':
        pass # depth is perfecto!
    else:
        assert False, '[ Error ]: Unsupported depth type'
        
    # flip z axis of coord map
    coord_map = np.array(coord_map, dtype=np.float32) / 255
    coord_map[:, :, 2] = 1 - coord_map[:, :, 2]

    # Converting the mask into a typical mask
    mask_cdata = np.array(mask_image, dtype=np.int32)
    mask_cdata[mask_cdata==255] = -1

    # Loading data
    print("Reading json data")
    json_data = jt.load_from_json(pathlib.Path.cwd() / 'first_json.json')
    instance_dict = json_data['instance_dict']
    scales = np.asarray(json_data['scales'], dtype=np.float32)
    RTs = np.asarray(json_data['RTs'], dtype=np.float32)

    # Reorganizing data to match instance ids order
    print("Reorganizing basic data to match order")
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
    bboxes = dm.extract_2d_bboxes_from_masks(masks)
    scores = [100 for j in range(len(class_ids))] # 100% for ground truth data

    print("Creating drawn image")
    output = dr.draw_detections(color_image, pj.constants.INTRINSICS, pj.constants.NOCS_CLASSES, bboxes, class_ids,
                                         masks, coords, RTs, scores, scales, draw_coord=True, draw_tag=True, draw_RT=True)

    cv2.imshow(f'My own data structure output', output)
    cv2.waitKey(0)

def load_new_dataset(dataset_path, obj_model_dir):

    all_color_images = get_image_paths_in_dir(dataset_path)

    for color_image in all_color_images:

        # Obtain the ground truth data for the image
        class_ids, bboxes, masks, coords, RTs, scores, scales, instance_dict, normalization_factors = get_new_dataset_information(color_image)

        image = cv2.imread(str(color_image), cv2.IMREAD_UNCHANGED)
        output = dr.draw_detections(image, pj.constants.INTRINSICS, None, None, class_ids,
                                             None, None, RTs, None, scales, normalization_factors,
                                             (255,0,0), False, False, True)
        
        cv2.imshow('output', output)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    return None

#-------------------------------------------------------------------------------
# Main Code

if __name__ == '__main__':

    # Parse arguments and replace global variables
    parser = argparse.ArgumentParser(description='Modify NOCS datasets')
    parser.add_argument('--DATASET_NAME', required=True, type=str)
    parser.add_argument('--SUBSET_DATASET_NAME', required=True, type=str)
    args = parser.parse_args()

    # Obtaining information paths
    DATASET_NAME = args.DATASET_NAME
    SUBSET_DATASET_NAME = args.SUBSET_DATASET_NAME

    DATASET_DIR = root.parents[1] / 'datasets' / 'NOCS' / DATASET_NAME / SUBSET_DATASET_NAME
    OBJ_MODEL_DIR = root.parents[1] / 'datasets' / 'NOCS' / 'obj_models' / DATASET_NAME / SUBSET_DATASET_NAME

    # Create new dataset 
    create_new_dataset()