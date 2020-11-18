import os
import sys
import pathlib

import math
import numpy as np
import cv2
from pyquaternion import Quaternion

import matplotlib.pyplot as plt

import scipy.spatial.transform
import scipy.spatial

import torch

import skimage

# Local Imports

import project as pj
import data_manipulation as dm

#-------------------------------------------------------------------------------
# Draw Constants

line_thickness_factor = 800
font_size_factor = 1000

#-------------------------------------------------------------------------------
# Complete (old data structure) Routine Functions

@dm.dec_correct_image_dataformat
def draw_detections(image, intrinsics, synset_names, bbox, class_ids, masks, coords,
                    RTs, scores, scales, normalizing_factors, RT_color, draw_coord=False, draw_tag=False, draw_RT=True):

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
            text = pj.constants.NOCS_CLASSES[class_ids[i]]+'({:.2f})'.format(scores[i])
            draw_image = draw_text(draw_image, bbox[i], text, draw_box=True)

        # Rotation and Translation data
        if draw_RT:
            RT = RTs[i]
            class_id = class_ids[i]
            normalizing_factor = normalizing_factors[i]

            # Creating a xyz axis and converting 3D coordinates into 2D projections
            xyz_axis = 0.3*np.array([[0, 0, 0], [0, 0, 1], [0, 1, 0], [1, 0, 0]]).transpose()
            norm_xyz_axis = xyz_axis / normalizing_factor
            projected_axes = dm.transform_3d_camera_coords_to_2d_quantized_projections(norm_xyz_axis, RT, intrinsics)

            # Creating a 8-point bounding box and converting it into a its 2D projection
            bbox_3d = dm.get_3d_bbox(scales[i,:],0)
            norm_bbox_3d = bbox_3d / normalizing_factor
            projected_bbox = dm.transform_3d_camera_coords_to_2d_quantized_projections(norm_bbox_3d, RT, intrinsics)

            # Drawing the projections by using the resulting points
            draw_image = draw_3d_bbox(draw_image, projected_bbox, RT_color)
            draw_image = draw_axes(draw_image, projected_axes)

    return draw_image

#-------------------------------------------------------------------------------
# Drawing of Quaternion

@dm.dec_correct_image_dataformat
def draw_quat(
    image, 
    quaternion, 
    translation_vector, 
    scale,
    intrinsics,
    color=(0,0,255)
    ):

    # Convert quaternion to RT matrix
    RT = dm.quat_2_RT_given_T_in_world(quaternion, translation_vector)
    
    draw_image = draw_RT(
        image = image,
        RT = RT,
        scale = scale,
        intrinsics = intrinsics,
        color = color
    )

    return draw_image

@dm.dec_correct_image_dataformat
def draw_quats(
    image, 
    intrinsics, 
    quaternions, 
    translation_vectors, 
    scales,
    color=(0,0,255)
    ):

    draw_image = image.copy()

    for i in range(len(quaternions)):

        try:
            draw_image = draw_quat(
                image = draw_image, 
                quaternion = quaternions[i], 
                translation_vector = translation_vectors[i], 
                scale = scales[i],
                intrinsics = intrinsics,
                color=color
            )
        except Exception as e:
            print(f'draw_quat error: {e}')

    return draw_image

#-------------------------------------------------------------------------------
# Drawing of Rotation Matrix

@dm.dec_correct_image_dataformat
def draw_RT(
    image, 
    RT, 
    scale, 
    intrinsics,
    color=(0,0,255)
    ):

    # Pts that will be displayed
    xyz = 0.075*np.array([[0,0,0],[1,0,0],[0,1,0],[0,0,1]], dtype=np.float32).transpose()
    
    bbox_3d = dm.get_3d_bbox(scale, 0)

    # Apply RT into a set of points
    xyz_projection = dm.transform_3d_camera_coords_to_2d_quantized_projections(xyz, RT, intrinsics)
    bbox_3d_projection = dm.transform_3d_camera_coords_to_2d_quantized_projections(bbox_3d, RT, intrinsics)

    # Drawing projections
    draw_image = draw_3d_bbox(image, bbox_3d_projection, color=color)
    draw_image = draw_axes(image, xyz_projection)

    return draw_image

@dm.dec_correct_image_dataformat
def draw_RTs(image, RTs, scales, intrinsics, color):

    draw_image = image.copy()

    for i in range(len(RTs)):

        draw_image = draw_RT(
            draw_image, 
            RT = RTs[i], 
            scale = scales[i], 
            intrinsics = intrinsics,
            color=color
        )            

    return draw_image

#-------------------------------------------------------------------------------
# Small-helper Functions

@dm.dec_correct_image_dataformat
def draw_centroids(image, centroids, color=(0,255,0), thickness=4):

    draw_image = image.copy()

    for centroid in centroids:

        cv2.circle(draw_image, (centroid[0], centroid[1]), thickness, color, -1)
        label = f'{pj.constants.NOCS_CLASSES[centroid.class_id]}'
        cv2.putText(draw_image, label, (centroid[0]-20, centroid[1]-20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

    return draw_image

@dm.dec_correct_image_dataformat
def draw_3d_bbox(image, bbox_pts, color):

    # Determine a good thickness
    size = np.sqrt(image.shape[0]**2 + image.shape[1]**2)
    thickness = int(math.ceil(size/line_thickness_factor))

    # draw ground layer in darker color
    color_ground = (int(color[0] * 0.7), int(color[1] * 0.7), int(color[2] * 0.7))
    for i, j in zip([4, 5, 6, 7],[5, 7, 4, 6]):
        image = cv2.line(image, tuple(bbox_pts[i]), tuple(bbox_pts[j]), color_ground, thickness, cv2.LINE_AA)

    # draw pillars in blue color
    color_pillar = (int(color[0]*0.9), int(color[1]*0.9), int(color[2]*0.9))
    for i, j in zip([0, 1, 2, 3],[4, 5, 6, 7]):
        image = cv2.line(image, tuple(bbox_pts[i]), tuple(bbox_pts[j]), color_pillar, thickness, cv2.LINE_AA)

    # finally, draw top layer in color
    for i, j in zip([0, 1, 2, 3],[1, 3, 0, 2]):
        image = cv2.line(image, tuple(bbox_pts[i]), tuple(bbox_pts[j]), color, thickness, cv2.LINE_AA)

    # Drawing the corners as circles
    for bbox_pt in bbox_pts:
        cv2.circle(image, tuple(bbox_pt), thickness*2, color, -1)

    return image

@dm.dec_correct_image_dataformat
def draw_axes(image, axes):

    font = cv2.FONT_HERSHEY_TRIPLEX

    # Determine a good thickness
    size = np.sqrt(image.shape[0]**2 + image.shape[1]**2)
    thickness = int(math.ceil(size/line_thickness_factor))
    font_scale = size/font_size_factor

    # draw axes
    # axes[0] = center of axis, axes[X] = end point of an axis
    image = cv2.line(image, tuple(axes[0]), tuple(axes[1]), (100, 100, 255), 2*thickness) # Red (x)
    image = cv2.line(image, tuple(axes[0]), tuple(axes[3]), (255, 100, 100), 2*thickness) # Blue (y)
    image = cv2.line(image, tuple(axes[0]), tuple(axes[2]), (100, 255, 100), 2*thickness) # Green (z)
    image = cv2.circle(image, tuple(axes[0]), 2*thickness, (255,255,255), -1) # Center axis

    """
    # Calculating the x,y,z text locations
    for letter, color, axis_index in zip(['x', 'y', 'z'], [(100,100,255),(255,100,100),(100,255,100)], [1,3,2]):
        text_size = cv2.getTextSize(letter, font, 1, 1)[0]
        text_center_shift = (np.array([-1 * text_size[0], text_size[1]]) / 3).astype(np.int)
        letter_place = ((axes[axis_index] - axes[0]) * np.array([1.2, 1.3])).astype(np.int) + axes[0] + text_center_shift

        # drawing letter (x, y, z) corresponding to the axes
        cv2.putText(image, letter, tuple(letter_place), font, font_scale, color)
    """

    return image

@dm.dec_correct_image_dataformat
def draw_text(image, bbox, text, draw_box=False):

    font_face = cv2.FONT_HERSHEY_TRIPLEX

    size = np.sqrt(image.shape[0]**2 + image.shape[1]**2)
    thickness = int(math.ceil(size/line_thickness_factor))
    font_scale = size/font_size_factor
    
    retval, baseline = cv2.getTextSize(text, font_face, font_scale, thickness)
    
    bbox_margin = 10
    text_margin = 10
    
    text_box_pos_tl = (min(bbox[1] + bbox_margin, 635 - retval[0] - 2* text_margin) , min(bbox[2] + bbox_margin, 475 - retval[1] - 2* text_margin)) 
    text_box_pos_br = (text_box_pos_tl[0] + retval[0] + 2* text_margin,  text_box_pos_tl[1] + retval[1] + 2* text_margin)

    # text_pose is the bottom-left corner of the text
    text_pos = (text_box_pos_tl[0] + text_margin, text_box_pos_br[1] - text_margin - 3)
    
    if draw_box:
        cv2.rectangle(image, 
                      (bbox[1], bbox[0]),
                      (bbox[3], bbox[2]),
                      (255, 0, 0), 2)

    cv2.rectangle(image, 
                  text_box_pos_tl,
                  text_box_pos_br,
                  (255,0,0), -1)
    
    cv2.rectangle(image, 
                  text_box_pos_tl,
                  text_box_pos_br,
                  (0,0,0), 1)

    cv2.putText(image, text, text_pos,
                font_face, font_scale, (255,255,255), thickness)

    return image