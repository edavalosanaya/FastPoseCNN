import os
import sys
import pathlib

import cv2
import numpy as np
import scipy.spatial
import scipy.linalg
import sklearn.preprocessing
from dual_quaternions import DualQuaternion

# local Imports (new source code)

root = next(path for path in pathlib.Path(os.path.abspath(__file__)).parents if path.name == 'FastPoseCNN')
sys.path.append(str(root))

import constants

# data-category imports
import data_json_tools
import data_manipulation

# visual-category imports
import visual_draw

sys.path.append(r'E:\MASTERS_STUFF\MastersProject\networks\NOCS_CVPR2019')
import utils as nocs_utils

# Obtaining information paths
camera_dataset = pathlib.Path(r'E:\MASTERS_STUFF\MastersProject\datasets\NOCS\small_camera')
dataset_path = camera_dataset / 'val' / '00000' 

image_id = "0009"
image_path = dataset_path / f'{image_id}_color.png'
mask_path = dataset_path / f'{image_id}_mask.png'
depth_path = dataset_path / f'{image_id}_depth.png'
coord_path = dataset_path / f'{image_id}_coord.png'
meta_path = dataset_path / f'{image_id}_meta.txt'

obj_model_dir = pathlib.Path(r'E:\MASTERS_STUFF\MastersProject\networks\NOCS_CVPR2019\data\obj_models\val')

#-------------------------------------------------------------------------------
# Getting data Functions

def get_original_information(color_image, obj_model_dir):

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

            #print(f"Instance #{instance_id}: {class_id} = {constants.synset_names[class_id]}")

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
    bboxes = data_manipulation.extract_2d_bboxes_from_masks(masks)
    scores = [100 for j in range(len(class_ids))]

    # now obtaining the rotation and translation
    RTs, _, _, _ = nocs_utils.align(class_ids, masks, coords, depth_image, constants.intrinsics,
                                    constants.synset_names, ".", None)

    return class_ids, bboxes, masks, coords, RTs, scores, scales, instance_dict

#-------------------------------------------------------------------------------
# Simple Tool Function

def add(a,b):

    return a + b

def subtract(a,b):

    return a - b

def enable_print():
    sys.stdout = sys.__stdout__

def disable_print():
    sys.stdout = open(os.devnull, 'w')

#-------------------------------------------------------------------------------
# Algorithm Functions

def transform_2d_quantized_projections_to_3d_camera_coords(cartesian_projections_2d, RT, intrinsics, z):
    # Math comes from: https://robotacademy.net.au/lesson/image-formation/
    # and: https://stackoverflow.com/questions/12299870/computing-x-y-coordinate-3d-from-image-point
    
    disable_print()

    # Including the Z component into the projection (2D to 3D)
    cartesian_projections_2d = cartesian_projections_2d.astype(np.float32)
    cartesian_projections_2d[0, :] = cartesian_projections_2d[0, :] * (z/1000)
    cartesian_projections_2d[1, :] = cartesian_projections_2d[1, :] * (z/1000)
    
    homogeneous_projections_2d = np.vstack([cartesian_projections_2d, z/1000])
    print(f'homogeneous_projections_2d: \n{homogeneous_projections_2d}\n')

    cartesian_world_coordinates_3d = np.linalg.inv(intrinsics) @ homogeneous_projections_2d
    print(f'cartesian_world_coordinates_3d: \n{cartesian_world_coordinates_3d}\n')

    homogeneous_world_coordinates_3d = data_manipulation.cartesian_2_homogeneous_coord(cartesian_world_coordinates_3d)
    print(f'homogeneous_world_coordinates_3d: \n{homogeneous_world_coordinates_3d}\n')

    homogeneous_camera_coordinates_3d = RT @ homogeneous_world_coordinates_3d
    print(f'homogeneous_camera_coordinates_3d: \n{homogeneous_camera_coordinates_3d}\n')

    cartesian_camera_coordinated_3d = data_manipulation.homogeneous_2_cartesian_coord(homogeneous_camera_coordinates_3d)

    enable_print()

    return cartesian_camera_coordinated_3d

def transform_3d_camera_coords_to_2d_quantized_projections(cartesian_camera_coordinates_3d, RT, intrinsics):
    # Math comes from: https://robotacademy.net.au/lesson/image-formation/

    disable_print()

    # Converting cartesian 3D coordinates to homogeneous 3D coordinates
    homogeneous_camera_coordinates_3d = data_manipulation.cartesian_2_homogeneous_coord(cartesian_camera_coordinates_3d)
    print(f'homo camera coordinates: \n{homogeneous_camera_coordinates_3d}\n')

    # Creating proper K matrix (including Pu, Pv, Uo, Vo, and f)
    K_matrix = np.hstack([intrinsics, np.zeros((intrinsics.shape[0], 1), dtype=np.float32)])

    ############################################################################
    # METHOD 1
    ############################################################################
    print("METHOD 1")

    # Creating camera matrix (including intrinsic and external paramaters)
    camera_matrix = K_matrix @ np.linalg.inv(RT)
    
    # Obtaining the homogeneous 2D projection
    homogeneous_projections_2d = camera_matrix @ homogeneous_camera_coordinates_3d
    print(f'homo projections: \n{homogeneous_projections_2d}\n')

    """
    ############################################################################
    # METHOD 2
    ############################################################################
    print('METHOD 2')

    homogeneous_world_coordinates_3d = np.linalg.inv(RT) @ homogeneous_camera_coordinates_3d
    print(f'homo world coordinates: \n{homogeneous_world_coordinates_3d}\n')

    homogeneous_projections_2d = K_matrix @ homogeneous_world_coordinates_3d
    print(f'homo projections: \n{homogeneous_projections_2d}\n')

    homogeneous_world_coordinates_3d = np.linalg.inv(intrinsics) @ homogeneous_projections_2d
    print(f'back to homo world coordinates from projections: \n{homogeneous_world_coordinates_3d}\n')

    cartesian_projections_2d = data_manipulation.homogeneous_2_cartesian_coord(homogeneous_projections_2d)
    print(f'cartesian projections: \n{cartesian_projections_2d}\n')

    homogeneous_projections_2d = data_manipulation.cartesian_2_homogeneous_coord(cartesian_projections_2d)
    homogeneous_world_coordinates_3d = np.linalg.inv(intrinsics) @ homogeneous_projections_2d
    cartesian_world_coordinates_3d = data_manipulation.homogeneous_2_cartesian_coord(homogeneous_world_coordinates_3d)
    print(f'back to cartesian camera coordinates from projections: \n{cartesian_world_coordinates_3d}\n')
    """

    # Converting the homogeneous projection into a cartesian projection
    cartesian_projections_2d = data_manipulation.homogeneous_2_cartesian_coord(homogeneous_projections_2d)
    print(f'cartesian projections: \n{cartesian_projections_2d}\n')

    # Converting projections from float32 into int32 (quantizing to from continuous to integers)
    cartesian_projections_2d = cartesian_projections_2d.astype(np.int32)

    # Transposing cartesian_projections_2d to have matrix in row major fashion
    cartesian_projections_2d = cartesian_projections_2d.transpose()

    enable_print()

    return cartesian_projections_2d

def create_translation_vector(cartesian_projections_2d_xy_origin, z, intrinsics):

    disable_print()

    # Checking inputs
    print(f'cartesian_projections_2d_xy_origin: \n{cartesian_projections_2d_xy_origin}\n')
    print(f'z: {z}')

    # Including the Z component into the projection (2D to 3D)
    cartesian_projections_2d_xy_origin = cartesian_projections_2d_xy_origin.astype(np.float32)
    cartesian_projections_2d_xy_origin[0, :] = cartesian_projections_2d_xy_origin[0, :] * (z/1000)
    cartesian_projections_2d_xy_origin[1, :] = cartesian_projections_2d_xy_origin[1, :] * (z/1000)

    homogeneous_projections_2d_xyz_origin = np.vstack([cartesian_projections_2d_xy_origin, z/1000])
    print(f'homogeneous_projections_2d_xyz_origin: \n{homogeneous_projections_2d_xyz_origin}\n')

    # Converting projectins to world 3D coordinates
    cartesian_world_coordinates_3d_xyz_origin = np.linalg.inv(intrinsics) @ homogeneous_projections_2d_xyz_origin
    print(f'cartesian_world_coordinates_3d_xyz_origin: \n{cartesian_world_coordinates_3d_xyz_origin}\n')

    # The cartesian world coordinates of the origin are the translation vector
    translation_vector = cartesian_world_coordinates_3d_xyz_origin

    enable_print()

    return translation_vector

def convert_RT_to_quaternion(RT):

    # Verbose
    disable_print()

    print(f'RT before normalization: \n{RT}\n')

    # normalizing
    normalizing_factor = np.amax(RT)
    print(f'normalizing_factor: {normalizing_factor}')
    RT[:3, :] = RT[:3, :] / normalizing_factor

    # Testing quaternion representation
    print(f'original RT matrix: \n{RT}\n')

    # Rotation Matrix
    rotation_matrix = RT[:3, :3]

    smart_rotation_object = scipy.spatial.transform.Rotation.from_matrix(rotation_matrix)
    print(f'registered original rotation matrix: \n{smart_rotation_object.as_matrix()}\n')
    print(f'det(registed_original_rotation_matrix) = {np.linalg.det(smart_rotation_object.as_matrix())}\n')

    quaternion = smart_rotation_object.as_quat()
    print(f'constructed quaternion: \n{quaternion}\n')

    # Translation Matrix
    translation_vector = RT[:3, -1].reshape((-1, 1))

    # Returning print to enabled
    enable_print()

    return quaternion, translation_vector, normalizing_factor

def convert_quaternion_to_RT(quaternion, translation_vector):

    smart_rotation_object = scipy.spatial.transform.Rotation.from_quat(quaternion)
    rotation_matrix = smart_rotation_object.as_matrix()
    
    RT = np.vstack([np.hstack([rotation_matrix, translation_vector]), [0,0,0,1]])

    return RT

def get_new_RT_error(perfect_projected_axes, quat_projected_axes):

    #print(f'\nperfect_projected_axes: \n{perfect_projected_axes}\n')
    #print(f'quat_projected_axes: \n{quat_projected_axes}\n')

    diff_axes = quat_projected_axes - perfect_projected_axes
    #print(f'diff_axes: \n{diff_axes}\n')

    error = np.sum(np.absolute(diff_axes))
    #print(f'error: {error}')

    return error

def fix_quaternion_RT(intrinsics, original_RT, quad_RT, normalizing_factor):

    # Verbose
    #disable_print()
    enable_print()

    # Creating a xyz axis and converting 3D coordinates into 2D projections
    xyz_axis = 0.3*np.array([[0, 0, 0], [0, 0, 1], [0, 1, 0], [1, 0, 0]]).transpose()
    perfect_projected_axes = transform_3d_camera_coords_to_2d_quantized_projections(xyz_axis, original_RT, intrinsics)

    norm_xyz_axis = xyz_axis / normalizing_factor
    quat_projected_axes = transform_3d_camera_coords_to_2d_quantized_projections(norm_xyz_axis, quad_RT, intrinsics)
    #disable_print()

    baseline_error = get_new_RT_error(perfect_projected_axes, quat_projected_axes)

    print(f'baseline_error: {baseline_error}')

    test_RT = quad_RT.copy()

    original_step_size = 0.00001
    step_size = original_step_size # 0.01
    correction_coef = 10

    max_step_size = 0.01
    maxing_error_flag = False
    min_step_size = 0.00000001
    minimum_error_flag = False
    
    min_error = baseline_error
    past_error = np.zeros((3, 2), np.int)
    
    past_min_errors = [] # Holds last 10 min errors
    past_min_flag = False

    for i in range(25): # only 100 iterations allowed

        print("%"*50)
        errors = np.zeros((3, 2), np.int)

        # try each dimension
        for xyz in range(3):

            # try both positive and negative addition
            for operation_index, operation in enumerate([add, subtract]):

                test_RT[xyz,-1] = operation(quad_RT[xyz,-1], step_size)
                test_projected_axes = transform_3d_camera_coords_to_2d_quantized_projections(norm_xyz_axis, test_RT, intrinsics)
                #disable_print()

                error = get_new_RT_error(perfect_projected_axes, test_projected_axes)
                errors[xyz, operation_index] = error

        print(f'errors: \n{errors} step_size: {step_size}')
        past_min_errors.append(np.amin(errors))
        past_min_errors = past_min_errors[:15]

        if min_error <= np.amin(errors):
            
            # Process error
            # Chaning step_size to match error
            if len(np.unique(past_min_errors)) <= 3 and len(past_min_errors) >= 10:
                past_min_errors = []

                if maxing_error_flag is True:
                    step_size /= correction_coef
                else:
                    step_size *= correction_coef

                print(f'A (repetition) - multiplied by {correction_coef}')

            elif len(np.unique(past_min_errors)) >= 8 and len(past_min_errors) >= 10:

                if step_size * correction_coef >= max_step_size:
                    pass

                past_min_errors = []
                step_size *= correction_coef
                print(f'B (all unique) - multiplied by {correction_coef}')

            elif np.amin(errors) > min_error:

                step_size /= correction_coef
                print(f'C (more error) - divided by {correction_coef}')

            elif (past_error == errors).all():
                print(f'past_error: \n{past_error}')
                step_size *= correction_coef
                print(f'D (all same) - multiplied by {correction_coef}')

            elif np.amin(errors) == min_error:

                step_size *= correction_coef
                print(f'E (static) - multiplied by {correction_coef}')

            # Checking if step_size had not diverted too much

            if step_size > max_step_size:
                step_size = original_step_size
                maxing_error_flag = True
                minimum_error_flag = False

            elif step_size < min_step_size:
                step_size = original_step_size
                minimum_error_flag = True
                maxing_error_flag = False
        else:
            # perform the action that reduces the error the most
            min_error = np.amin(errors)
            min_error_index = divmod(errors.argmin(), errors.shape[1])
            min_xyz, operation = min_error_index[0], add if min_error_index[1] == 0 else subtract
            quad_RT[min_xyz, -1] = operation(quad_RT[min_xyz,-1], step_size)
            print(f'new min_error: {min_error}')

        

        if min_error <= 1:
            print('min_error of <= 1 achieved!')
            break
        else:
            past_error = errors.copy()

    # Returning printing
    enable_print()

    return quad_RT

def reconstruct_RT(quaternion, translation_vector):

    smart_rotation_object = scipy.spatial.transform.Rotation.from_quat(quaternion)
    rotation_matrix = smart_rotation_object.as_matrix()
    inv_rotation_matrix = np.linalg.inv(rotation_matrix)

    inv_RT =  np.vstack([np.hstack([inv_rotation_matrix, translation_vector]), [0,0,0,1]])
    RT = np.linalg.inv(inv_RT)

    return RT

def draw_detections(image, intrinsics, synset_names, bbox, class_ids, masks, coords,
                    RTs, scores, scales, depth, draw_coord=False, draw_tag=False, draw_RT=True):

    """
    This function draws the coordinate image, the class and score tag, and the 
    rotation and translation information into an image if given the right data.
    """

    draw_image = image.copy()

    for i in range(len(class_ids)):

        print('*' * 50)
        print(f'instance: {i}')

        if draw_coord:
            mask = masks[:, :, i]
            cind, rind = np.where(mask == 1)
            coord_data = coords[:, :, i, :].copy()
            coord_data[:, :, 2] = 1 - coord_data[:, :, 2]
            draw_image[cind,rind] = coord_data[cind, rind] * 255

        # Tag data 
        if draw_tag:
            text = f'{i}-' + constants.synset_names[class_ids[i]]+'({:.2f})'.format(scores[i])
            draw_image = visual_draw.draw_text(draw_image, bbox[i], text, draw_box=True)

        # Rotation and Translation data
        if draw_RT:

            RT = RTs[i]
            inv_RT = RT.copy()
            print(f'RT: \n{RT}\n')
            RT = np.linalg.inv(RT)

            # Obtaining the 3D objects that will be drawn
            xyz_axis = 0.3*np.array([[0, 0, 0], [0, 0, 1], [0, 1, 0], [1, 0, 0]]).transpose()
            bbox_3d = data_manipulation.get_3d_bbox(scales[i,:],0)
            
            # Getting the original projected pts with the old RTs
            perfect_projected_axes = transform_3d_camera_coords_to_2d_quantized_projections(xyz_axis, RT, constants.intrinsics)
            perfect_projected_bbox = transform_3d_camera_coords_to_2d_quantized_projections(bbox_3d, RT, constants.intrinsics)

            # Drawing the original pts
            #draw_image = visual_draw.draw_axes(draw_image, perfect_projected_axes)
            #draw_image = visual_draw.draw_3d_bbox(draw_image, perfect_projected_bbox, (0,0,255))

            print('\n')
            print('-' * 50)
            print('-' * 50)
            print('\n')

            # Converting transformation matrix to quaternion with a translation vector
            quaternion, translation_vector, normalizing_factor = convert_RT_to_quaternion(RT.copy())
            #print(f'old translation vector: \n{translation_vector}\n')

            # Converting back the quaternion & translation vector to transformation matrix
            new_RT = convert_quaternion_to_RT(quaternion, translation_vector)

            # Fixing translation error introduced by the conversion
            #new_RT = fix_quaternion_RT(intrinsics, RT, new_RT, normalizing_factor)

            #old_normalized_translation_vector = np.linalg.inv(RT)[:3, -1].reshape((-1, 1))
            #print(f'old_normalized_translation_vector: \n{old_normalized_translation_vector}\n')
            
            # Creating my own transformation matrix given only the centroid and depth
            projected_origin = perfect_projected_axes[0,:].reshape((-1, 1))
            #origin_z = np.array(depth[tuple(np.flip(perfect_projected_axes[0,:]))])
            #inv_RT = np.linalg.inv(RT)
            print(f'normalizing_factor: {normalizing_factor}')
            origin_z = (np.linalg.inv(new_RT)[2, 3] * 1000)
            #print(origin_z)
            new_translation_vector = create_translation_vector(projected_origin, origin_z, intrinsics)
            print(f'new translation vector: \n{new_translation_vector}\n')

            # Using the created translation vector
            new_RT = reconstruct_RT(quaternion, new_translation_vector)

            # Normalizing the XYZ coordinates to match the normalized new_RT
            norm_xyz_axis = xyz_axis / normalizing_factor
            norm_bbox_3d = bbox_3d / normalizing_factor

            # Obtaining the projected pts with the new_RT
            projected_axes = transform_3d_camera_coords_to_2d_quantized_projections(norm_xyz_axis, new_RT, constants.intrinsics)
            projected_bbox = transform_3d_camera_coords_to_2d_quantized_projections(norm_bbox_3d, new_RT, constants.intrinsics)

            # Drawing the new pts
            draw_image = visual_draw.draw_3d_bbox(draw_image, projected_bbox, (255, 0, 0))
            draw_image = visual_draw.draw_axes(draw_image, projected_axes)

            # Converting projections back to 3D coordinates            
            #points = projected_axes[:1, :]
            #z = np.array([depth[tuple(np.flip(coord))] for coord in points])

            # Perfect values
            #z = np.array([1.1467362]) * 1000
            #z = np.array([1.1467362, 1.16376528, 1.04470176, 1.24229696]) * 1000

            #calculated_origin_3d = transform_2d_quantized_projections_to_3d_camera_coords(points.transpose(), new_RT, constants.intrinsics, z)
            #print(f'calculated_origin_3d: \n{calculated_origin_3d}')
            

    return draw_image

#-------------------------------------------------------------------------------
# Main Code

if __name__ == '__main__':

    class_ids, bboxes, masks, coords, RTs, scores, scales, instance_dict = get_original_information(image_path, obj_model_dir)

    depth_image = cv2.imread(str(depth_path), cv2.IMREAD_UNCHANGED)
    image = cv2.imread(str(image_path), cv2.IMREAD_UNCHANGED)

    #print("Creating drawn image")
    output = draw_detections(image, constants.intrinsics, constants.synset_names, bboxes, class_ids,
                             masks, coords, RTs, scores, scales, depth_image, draw_coord=True, draw_tag=True, draw_RT=True)
    

    #fixed_RT = fix_quaternion_RT(color_image, constants.intrinsics, RTs)

    cv2.imshow(f'output', output)
    cv2.waitKey(0)