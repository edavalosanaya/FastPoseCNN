# Library Import

import os
import sys
import cv2
from matplotlib import pyplot as plt
import pickle
import numpy as np
import open3d as o3d

# Getting Root directory
ROOT_DIR = os.getcwd()
NETWORKS_DIR = os.path.join(ROOT_DIR, 'networks')

# Getting NOCS paths
NOCS_ROOT_DIR = os.path.join(NETWORKS_DIR, "NOCS_CVPR2019")
NOCS_MODEL_DIR = os.path.join(NOCS_ROOT_DIR, "logs")
NOCS_COCO_MODEL_PATH = os.path.join(NOCS_MODEL_DIR, "mask_rcnn_coco.h5")
NOCS_WEIGHTS_PATH = os.path.join(NOCS_MODEL_DIR, "nocs_rcnn_res50_bin32.h5")

# Getting DenseDepth paths
DENSE_ROOT_DIR = os.path.join(NETWORKS_DIR, "DenseDepth")
DENSE_TF_ROOT_DIR = os.path.join(DENSE_ROOT_DIR, "Tensorflow")

# Appending necessary paths
sys.path.append(NETWORKS_DIR)
sys.path.append(NOCS_ROOT_DIR)
sys.path.append(DENSE_TF_ROOT_DIR)

# Local Imports

import tools

#---------------------------------------------------------------------------------
# Main Code

# Parameters
normalizing = False
histogram_matching = False

clipping_enabled = False
brightness_shift = False # bs
brightness_multiplication = False

bs_value = 0
bm_value = 1

save_photos = False

# Inputs
image_path = r"E:\MASTERS_STUFF\workspace\case_study_data\debugging_depth\0000_color.png"
source_depth_path = r"E:\MASTERS_STUFF\workspace\case_study_data\debugging_depth\0000_depth.png"
generated_depth_path = r"E:\MASTERS_STUFF\workspace\case_study_data\real_dataset_results\scene_1\0000\generated_0000_depth.png"
pickle_data_path = r"E:\MASTERS_STUFF\workspace\case_study_data\real_dataset_results\scene_1\0000\0000_color_nocs_result.pickle"

mod_generated_depth_path = r"E:\MASTERS_STUFF\workspace\case_study_data\real_dataset_results\scene_1\0000\modified_generated_0000_depth.png"

source_depth_original = cv2.imread(source_depth_path, -1)
generated_depth_original = cv2.imread(generated_depth_path,-1)
image = cv2.imread(image_path)

with open(pickle_data_path, 'rb') as f:
    data = pickle.load(f)

result, r = data['result'], data['r']

# Loading models
print("\nLoading models...\n")
nocs_model = tools.NOCS(load_model=False)
print("\nFinished loading models\n")

while True:

    # Key Information
    key = cv2.waitKey() & 0xFF

    # Key Parameters
    if key == ord('q'):
        break
    elif key == ord('n'): # normalizing
        normalizing = not normalizing
        print("Normalizing: {}".format(normalizing))
    
    elif key == ord("h"): # Histogram Matching
        histogram_matching = not histogram_matching
        print("Histogram Matching: {}".format(histogram_matching))

    elif key == ord('c'): # Clipping
        clipping_enabled = not clipping_enabled
        print("Clippping Enabled: {}".format(clipping_enabled))
    
    elif key == ord('s'): # Brightness Shift 
        brightness_shift = not brightness_shift
        print("Brightness Shift: {}".format(brightness_shift))
    elif key == ord("u"): # up
        bs_value += 5
        print("bs_value: {}".format(bs_value))
    elif key == ord("d"): # down
        bs_value -= 5
        print("bs_value: {}".format(bs_value))
    
    elif key == ord('m'): # Brightness Multiplication
        brightness_multiplication = not brightness_multiplication
        print("Brightness Multiplication: {}".format(brightness_multiplication))
    elif key == ord("+"):
        bm_value += 0.1
        print("bm_value: {}".format(bm_value))
    elif key == ord("-"):
        bm_value -= 0.1
        print("bm_value: {}".format(bm_value))

    elif key == ord('a'): # Save photos
        save_photos = True

    # Depth Modifications
    source_depth = source_depth_original
    generated_depth = generated_depth_original

    # Normalize depth
    if normalizing is True:
        source_depth = tools.normalize_depth(source_depth)
        generated_depth = tools.normalize_depth(generated_depth)

    if brightness_multiplication is True:
        generated_depth = generated_depth * bm_value
        generated_depth = generated_depth.astype('uint16')
        if clipping_enabled is True:
            generated_depth = np.clip(generated_depth, 0, 255)

    if brightness_shift is True:
        generated_depth = generated_depth + bs_value
        if clipping_enabled is True:
            generated_depth = np.clip(generated_depth, 0, 255)

    # Histogram Matching
    if histogram_matching is True:
        generated_depth = tools.hist_matching(generated_depth, tools.calculate_cdf(source_depth))

    # RGBD Image Creation
    source_rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(o3d.io.read_image(image_path), o3d.io.read_image(source_depth_path))
    generated_rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(o3d.io.read_image(image_path), o3d.io.read_image(mod_generated_depth_path))

    # Drawing Histogram
    _, max_value, _, _ = cv2.minMaxLoc(generated_depth)
    max_value = int(max_value)

    if max_value < 256:
        max_value = 256

    nbins = hist_width = max_value
    hist_height = 256

    hist = cv2.calcHist([generated_depth], [0], None, [max_value], [0,max_value])
    draw_hist = tools.draw_hist_image(hist, hist_height, hist_width, nbins)

    # ICP Calculations
    tools.disable_print()
    source_depth_result = nocs_model.icp_algorithm(image, image_path, source_depth, result, r)
    generated_depth_result = nocs_model.icp_algorithm(image, image_path, generated_depth, result, r)
    tools.enable_print()

    # Visualizing Depth
    source_depth = tools.making_depth_easier_to_see(source_depth)
    generated_depth = tools.making_depth_easier_to_see(generated_depth)

    # OpenCV visualizing
    cv2.imshow("source_depth", source_depth)
    cv2.imshow("generated_depth", generated_depth)
    cv2.imshow("source_depth_result", source_depth_result)
    cv2.imshow("generated_depth_result", generated_depth_result)
    cv2.imshow("generated depth histogram", draw_hist)

    # Open3D Visualizing (Look Identical to OpenCV Information)
    #cv2.imshow("RGBD Source Color", np.asarray(source_rgbd_image.color))
    #cv2.imshow("RGBD Source Depth", np.asarray(source_rgbd_image.depth))
    #cv2.imshow("RGBD Generated Color", np.asarray(generated_rgbd_image.color))
    #cv2.imshow("RGBD Generated Depth", np.asarray(generated_rgbd_image.depth))

    point_cloud = o3d.geometry.PointCloud.create_from_rgbd_image(source_rgbd_image,
                                                                 o3d.camera.PinholeCameraIntrinsic(o3d.camera.PinholeCameraIntrinsicParameters.PrimeSenseDefault))
    point_cloud.transform([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])
    o3d.visualization.draw_geometries([point_cloud])

    point_cloud = o3d.geometry.PointCloud.create_from_rgbd_image(generated_rgbd_image,
                                                                 o3d.camera.PinholeCameraIntrinsic(o3d.camera.PinholeCameraIntrinsicParameters.PrimeSenseDefault))
    point_cloud.transform([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])
    o3d.visualization.draw_geometries([point_cloud])

    print("*")

    # Saving photos
    if save_photos:

        save_photos = False
        save_dir_path = r"E:\MASTERS_STUFF\workspace\case_study_data\histogram_visualize"

        if histogram_matching is False:
            
            bs_value = int(bs_value)
            bm_value_print = str(bm_value)
            bm_value_print = "_".join(bm_value_print.split("."))

            print("\nSaving photos with shift {} and multiplication {}\n".format(bs_value, bm_value_print))

            shift_multiply_dir = os.path.join(save_dir_path, "shift_{}_multiply_{}".format(bs_value, bm_value_print))

            if os.path.exists(shift_multiply_dir) is False:
                os.mkdir(shift_multiply_dir)
        
            cv2.imwrite(os.path.join(shift_multiply_dir, "generated_depth_shift_{}_multiply_{}.png".format(bs_value, bm_value_print)), generated_depth)
            cv2.imwrite(os.path.join(shift_multiply_dir, "generated_depth_result_shift_{}_multiply_{}.png".format(bs_value, bm_value_print)), generated_depth_result)
            cv2.imwrite(os.path.join(shift_multiply_dir, "generated_depth_histo_shift_{}_mutliply_{}.png".format(bs_value, bm_value_print)), draw_hist)

        else:

            print("\nSaving photos with histogram Matching")

            cv2.imwrite(os.path.join(save_dir_path, "generated_depth_hist_matching.png"), generated_depth)
            cv2.imwrite(os.path.join(save_dir_path, "generated_depth_result_histo_matching.png"), generated_depth_result)
            cv2.imwrite(os.path.join(save_dir_path, "generated_depth_histo_shift_histo_matching.png"), draw_hist)

cv2.waitKey(0)
cv2.destroyAllWindows()

