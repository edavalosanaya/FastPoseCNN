import os
import sys

import numpy as np
import scipy.spatial
import scipy.linalg
import sklearn.preprocessing
import scipy.spatial.transform
#from dual_quaternions import DualQuaternion

#-------------------------------------------------------------------------------
# Classes

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
# Mask/BBOX Functions

def extract_2d_bboxes_from_masks(masks):
    """
    Computing 2D bounding boxes from mask(s).
    Input
        masks: [height, width, number of mask]. Mask pixel are either 1 or 0
    Output
        bboxes [number of masks, (y1, x1, y2, x2)]
    """

    num_instances = masks.shape[-1]
    bboxes = np.zeros([num_instances, 4], dtype=np.int32)

    for i in range(num_instances):
        mask = masks[:, :, i]

        # Bounding box
        horizontal_indicies = np.where(np.any(mask, axis=0))[0]
        vertical_indicies = np.where(np.any(mask, axis=1))[0]

        if horizontal_indicies.shape[0]:
            x1, x2 = horizontal_indicies[[0, -1]]
            y1, y2 = vertical_indicies[[0, -1]]

            # x2 and y2 should not be part of the box. Increment by 1
            x2 += 1
            y2 += 1

        else:
            # No mask for this instance. Error might have occurred
            x1, x2, y1, y2 = 0,0,0,0

        bboxes[i] = np.array([y1, x1, y2, x2]) 

    return bboxes

def get_3d_bbox(scale, shift = 0):
    """
    Input: 
        scale: [3] or scalar
        shift: [3] or scalar
    Return 
        bbox_3d: [3, N]

    """
    if hasattr(scale, "__iter__"):
        bbox_3d = np.array([[scale[0] / 2, +scale[1] / 2, scale[2] / 2],
                  [scale[0] / 2, +scale[1] / 2, -scale[2] / 2],
                  [-scale[0] / 2, +scale[1] / 2, scale[2] / 2],
                  [-scale[0] / 2, +scale[1] / 2, -scale[2] / 2],
                  [+scale[0] / 2, -scale[1] / 2, scale[2] / 2],
                  [+scale[0] / 2, -scale[1] / 2, -scale[2] / 2],
                  [-scale[0] / 2, -scale[1] / 2, scale[2] / 2],
                  [-scale[0] / 2, -scale[1] / 2, -scale[2] / 2]]) + shift
    else:
        bbox_3d = np.array([[scale / 2, +scale / 2, scale / 2],
                  [scale / 2, +scale / 2, -scale / 2],
                  [-scale / 2, +scale / 2, scale / 2],
                  [-scale / 2, +scale / 2, -scale / 2],
                  [+scale / 2, -scale / 2, scale / 2],
                  [+scale / 2, -scale / 2, -scale / 2],
                  [-scale / 2, -scale / 2, scale / 2],
                  [-scale / 2, -scale / 2, -scale / 2]]) +shift

    bbox_3d = bbox_3d.transpose()
    return bbox_3d

#-------------------------------------------------------------------------------
# Pure Geometric Functions

def cartesian_2_homogeneous_coord(cartesian_coord):
    """
    Input: 
        Cartesian Vector/Matrix of size [N,M]
    Process:
        Transforms a cartesian vector/matrix into a homogeneous by appending ones
        as a new row to vector/matrix, making it a homogeneous vector/matrix.
    Output:
        Homogeneous Vector/Matrix of size [N+1, M]
    """

    homogeneous_coord = np.vstack([cartesian_coord, np.ones((1, cartesian_coord.shape[1]), dtype=cartesian_coord.dtype)])
    return homogeneous_coord

def homogeneous_2_cartesian_coord(homogeneous_coord):
    """
    Input: 
        Homogeneous Vector/Matrix of size [N,M]
    Process:
        Transforms a homogeneous vector/matrix into a cartesian by removing the 
        last row and dividing all the other rows by the last row, making it a 
        cartesian vector/matrix.
    Output:
        Cartesian Vector/Matrix of size [N-1, M]
    """

    cartesian_coord = homogeneous_coord[:-1, :] / homogeneous_coord[-1, :]
    return cartesian_coord

def transform_2d_quantized_projections_to_3d_camera_coords(cartesian_projections_2d, RT, intrinsics, z):
    # Math comes from: https://robotacademy.net.au/lesson/image-formation/
    # and: https://stackoverflow.com/questions/12299870/computing-x-y-coordinate-3d-from-image-point
    """
    Input:
        cartesian projections: [2, N] (N number of 2D points)
        RT: [4, 4] Transformation matrix
        intrinsics: [3, 3] intrinsics parameters of the camera
        z: [1, N] (N number of 2D points)
    Process:
        Given the cartesian 2D projections, RT, intrinsics, and Z value @ the (x,y)
        locations of the projections, we do the following routine
            (1) Integrate the Z component into the projection, resulting in making
                the cartesian vector/matrix into a homogeneous vector/matrix
            (2) Convert the resulting homogeneous vector/matrix (which is in the 
                world coordinates space) into the camera coordinates space
            (3) Convert the homogeneous camera coordinates into cartesian camera
                coordinates.
    Output:
        cartesian camera coordinates: [3, N] (N number of 3D points)
    """
    
    disable_print()

    # Including the Z component into the projection (2D to 3D)
    cartesian_projections_2d = cartesian_projections_2d.astype(np.float32)
    cartesian_projections_2d[0, :] = cartesian_projections_2d[0, :] * (z/1000)
    cartesian_projections_2d[1, :] = cartesian_projections_2d[1, :] * (z/1000)
    
    homogeneous_projections_2d = np.vstack([cartesian_projections_2d, z/1000])
    print(f'homogeneous_projections_2d: \n{homogeneous_projections_2d}\n')

    cartesian_world_coordinates_3d = np.linalg.inv(intrinsics) @ homogeneous_projections_2d
    print(f'cartesian_world_coordinates_3d: \n{cartesian_world_coordinates_3d}\n')

    homogeneous_world_coordinates_3d = cartesian_2_homogeneous_coord(cartesian_world_coordinates_3d)
    print(f'homogeneous_world_coordinates_3d: \n{homogeneous_world_coordinates_3d}\n')

    homogeneous_camera_coordinates_3d = RT @ homogeneous_world_coordinates_3d
    print(f'homogeneous_camera_coordinates_3d: \n{homogeneous_camera_coordinates_3d}\n')

    cartesian_camera_coordinated_3d = homogeneous_2_cartesian_coord(homogeneous_camera_coordinates_3d)

    enable_print()

    return cartesian_camera_coordinated_3d

def transform_3d_camera_coords_to_2d_quantized_projections(cartesian_camera_coordinates_3d, RT, intrinsics):
    # Math comes from: https://robotacademy.net.au/lesson/image-formation/
    """
    Input:
        cartesian camera coordinates: [3, N]
        RT: transformation matrix [4, 4]
        intrinsics: camera intrinsics parameters [3, 3]
    Process:
        Given the inputs, we do the following routine:
            (1) Convert cartesian camera coordinates into homogeneous camera
                coordinates.
            (2) Create the K matrix with the intrinsics.
            (3) Create the Camera matrix with the K matrix and RT.
            (4) Convert homogeneous camera coordinate directly to homogeneous 2D
                projections.
            (5) Convert homogeneous 2D projections into cartesian 2D projections.
    Output:
    """

    disable_print()

    # Converting cartesian 3D coordinates to homogeneous 3D coordinates
    homogeneous_camera_coordinates_3d = cartesian_2_homogeneous_coord(cartesian_camera_coordinates_3d)
    print(f'homo camera coordinates: \n{homogeneous_camera_coordinates_3d}\n')

    # Creating proper K matrix (including Pu, Pv, Uo, Vo, and f)
    K_matrix = np.hstack([intrinsics, np.zeros((intrinsics.shape[0], 1), dtype=np.float32)])

    # Creating camera matrix (including intrinsic and external paramaters)
    camera_matrix = K_matrix @ np.linalg.inv(RT)
    
    # Obtaining the homogeneous 2D projection
    homogeneous_projections_2d = camera_matrix @ homogeneous_camera_coordinates_3d
    print(f'homo projections: \n{homogeneous_projections_2d}\n')

    # Converting the homogeneous projection into a cartesian projection
    cartesian_projections_2d = homogeneous_2_cartesian_coord(homogeneous_projections_2d)
    print(f'cartesian projections: \n{cartesian_projections_2d}\n')

    # Converting projections from float32 into int32 (quantizing to from continuous to integers)
    cartesian_projections_2d = cartesian_projections_2d.astype(np.int32)

    # Transposing cartesian_projections_2d to have matrix in row major fashion
    cartesian_projections_2d = cartesian_projections_2d.transpose()

    enable_print()

    return cartesian_projections_2d

def create_translation_vector(cartesian_projections_2d_xy_origin, z, intrinsics):
    """
    Inputs: 
        cartesian projections @ (x,y) point: [2, 1]
        z, depth @ (x,y) point: [1,]
        intrinsics: camera intrinsics parameters: [3, 3]
    Process:
        Given the inputs, this function performs the following routine:
            (1) Includes the Z component into the cartesian projection, converting
                it into a homogeneous 2D projection.
            (2) With the inverse of the intrinsics matrix, the homogeneous 2D projection
                is transformed into the cartesian world coordinates
            (3) The translation vector is the cartesian world coordinates
            (4) Profit :)
    Output:
        translation vector: [3, 1]
    """

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
    """
    Inputs:
        RT, transformation matrix: [4, 4]
    Process:
        Given the inputs, this function does the following routine:
            (1) Normalize the RT to reduce as much as possible scaling errors
                generated by the orthogonalization of the rotation matrix
            (2) Inputing the rotation matrix into the scipy.spatial.transform.Rotation
                object and performing orthogonalization (determinant = +1).
            (3) Constructing the quaternion with the scipy...Rotation object
            (4) Taking the translation vector as output to not lose information
    Output:
        quaternion: [1, 4]
        translation vector: [3, 1]
        normalizing factor: [1,]
    """

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
    """
    Inputs:
        quaternion: [1, 4]
        translation_vector: [3, 1]
    Process:
        Given the inputs, this function does the following routine:
            (1) By feeding the quaternion to the scipy...Rotation object, the
                quaternion is converted into a rotation matrix
            (2) Reconstruct RT joint matrix by joining rotation and translation 
                matrices and adding the [0,0,0,1] section of the matrix. 
    Output:
        RT: transformation matrix: [4, 4]
    """

    smart_rotation_object = scipy.spatial.transform.Rotation.from_quat(quaternion)
    rotation_matrix = smart_rotation_object.as_matrix()
    
    RT = np.vstack([np.hstack([rotation_matrix, translation_vector]), [0,0,0,1]])

    return RT

def get_new_RT_error(perfect_projected_axes, quat_projected_axes):
    """
    Inputs:
        perfect_projected_axes: [N, 2]
        quat_projected_axes: [N, 2]
    Process:
        Given the inputs, this function does the following routine:
            (1) Obtain the difference between the perfect and quaternion projected
                axes points.
            (2) Calculate overall error by taking the sum of the absolute value 
                of the difference.
    Output:
        error: error of the quat_projected_axes compared to the perfect_projected_axes,
               size = scalar
    """

    #print(f'\nperfect_projected_axes: \n{perfect_projected_axes}\n')
    #print(f'quat_projected_axes: \n{quat_projected_axes}\n')

    diff_axes = quat_projected_axes - perfect_projected_axes
    #print(f'diff_axes: \n{diff_axes}\n')

    error = np.sum(np.absolute(diff_axes))
    #print(f'error: {error}')

    return error

def fix_quaternion_RT(intrinsics, original_RT, quad_RT, normalizing_factor):
    """
    Inputs:
        intrinsics: camera intrinsics parameters: [3, 3]
        original_RT: original transformation matrix: [4,4]
        quad_RT: quaternion-generated transformation matrix: [4, 4]
        normalizing_factor: normalizing factor used to do the RT2Quat conversion: scalar
    Process:
        Given the inputs, this function does the following routine:
            (1) Calculate the perfect_projected_axes
            (2) Determines the baseline error between the quaternion-based projected
                axes and the perfect projected axes
            (3) Inside a for loop, determine which axes (x,y,z) and which direction
                along each axes (+, -) will result in a reduce error.
            (4) Keep iterating until either (a) the error is reduced below a
                threshold of 1 or (b) 50 iterations are completed.
    Output:
        quat_RT, hopefully a fixed RT: [4,4]
    """

    # Verbose
    disable_print()

    # Creating a xyz axis and converting 3D coordinates into 2D projections
    xyz_axis = 0.3*np.array([[0, 0, 0], [0, 0, 1], [0, 1, 0], [1, 0, 0]]).transpose()
    perfect_projected_axes = transform_3d_camera_coords_to_2d_quantized_projections(xyz_axis, original_RT, intrinsics)

    norm_xyz_axis = xyz_axis / normalizing_factor
    quat_projected_axes = transform_3d_camera_coords_to_2d_quantized_projections(norm_xyz_axis, quad_RT, intrinsics)
    disable_print()

    baseline_error = get_new_RT_error(perfect_projected_axes, quat_projected_axes)

    print(f'baseline_error: {baseline_error}')

    step_size = 0.000001 # 0.01
    test_RT = quad_RT.copy()
    min_error = baseline_error
    past_error = np.zeros((3, 2), np.int)

    for i in range(100): # only 100 iterations allowed

        print("%"*50)
        errors = np.zeros((3, 2), np.int)

        # try each dimension
        for xyz in range(3):

            # try both positive and negative addition
            for operation_index, operation in enumerate([add, subtract]):

                test_RT[xyz,-1] = operation(quad_RT[xyz,-1], step_size)
                test_projected_axes = transform_3d_camera_coords_to_2d_quantized_projections(norm_xyz_axis, test_RT, intrinsics)
                disable_print()

                error = get_new_RT_error(perfect_projected_axes, test_projected_axes)
                errors[xyz, operation_index] = error

        print(errors)

        # Process error
        if np.amin(errors) > baseline_error:
            step_size /= 10
            print('divided by 10')

        elif (past_error == errors).all():
            print(f'past_error: {past_error}')
            step_size *= 10
            print('multiplied by 10')

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
    """
    Inputs:
        quaternion: [1, 4]
        translation_vector, cartesian camera coordinates space: [3, 1]
    Process:
        Given the inputs, this function does the following:
            (1) Convert the quaternion into a rotation matrix with scipy...Rotation.
            (2) Inverting the rotation matrix to match the translation vector's 
                coordinate space.
            (3) Joining the inverse rotation matrix with the translation vector 
                and adding [0,0,0,1] to the matrix.
            (4) Returning the inverse RT matrix into RT.
    Output:
        RT: [4,4]
    """

    smart_rotation_object = scipy.spatial.transform.Rotation.from_quat(quaternion)
    rotation_matrix = smart_rotation_object.as_matrix()
    inv_rotation_matrix = np.linalg.inv(rotation_matrix)

    inv_RT =  np.vstack([np.hstack([inv_rotation_matrix, translation_vector]), [0,0,0,1]])
    RT = np.linalg.inv(inv_RT)

    return RT
