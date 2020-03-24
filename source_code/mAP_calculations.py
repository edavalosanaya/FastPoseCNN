# Library Import

import os
import sys
import cv2
from matplotlib import pyplot as plt
import pickle
import numpy as np
import tqdm 
import glob

# Getting Root directory
ROOT_DIR = os.getcwd()

# Getting NOCS paths
NOCS_ROOT_DIR = os.path.join(ROOT_DIR, "NOCS_CVPR2019")
NOCS_MODEL_DIR = os.path.join(NOCS_ROOT_DIR, "logs")
NOCS_COCO_MODEL_PATH = os.path.join(NOCS_MODEL_DIR, "mask_rcnn_coco.h5")
NOCS_WEIGHTS_PATH = os.path.join(NOCS_MODEL_DIR, "nocs_rcnn_res50_bin32.h5")

# Getting DenseDepth paths
DENSE_ROOT_DIR = os.path.join(ROOT_DIR, "DenseDepth")
DENSE_TF_ROOT_DIR = os.path.join(DENSE_ROOT_DIR, "Tensorflow")

# Appending necessary paths
sys.path.append(NOCS_ROOT_DIR)
sys.path.append(DENSE_TF_ROOT_DIR)

# Local Imports

import tools
from NOCS_CVPR2019 import dataset as nocs_dataset
from NOCS_CVPR2019 import utils as nocs_utils

#---------------------------------------------------------------------
# Functions

def generate_data():

    # Dataset Configuration
    tools.disable_print()

    config = tools.InferenceConfig()

    # Ground Dataset
    dataset = nocs_dataset.NOCSDataset(tools.synset_names, 'test', config)
    dataset.load_real_scenes(original_dataset_path)
    dataset.prepare(tools.class_map)

    tools.enable_print()

    # Loading the models

    print("\nLoading models...\n")
    tools.disable_print()
    #dense_depth_net = tools.DenseDepth(checkpoint_path)
    nocs_model = tools.NOCS(load_model=True)
    tools.enable_print()
    print("\nFinished loading models\n")


    #image_ids = dataset.image_ids[:num_eval]
    image_ids = dataset.image_ids

    for image_id in tqdm.tqdm(image_ids):

        image_path = dataset.image_info[image_id]["path"]
        image_path_parse = image_path.split("\\")
        scene_info = image_path_parse[-2]
        image_large_id = image_path_parse[-1].replace("_color.png", "")
        title = 'generated'

        pickle_filename = "{}_{}_{}_results.pickle".format(scene_info, image_large_id, title)
        pickle_dest = os.path.join(output_dir, 'results_pickled', pickle_filename)

        if os.path.isfile(pickle_dest):
            continue

        #print(image_path)
        
        # Loading image and source depth
        image = dataset.load_image(image_id)
        source_depth = dataset.load_depth(image_id)
        source_depth = tools.nocs_depth_formatting(source_depth)

        # Generating the new depth
        #generated_depth_float32 = dense_depth_net.predict(image_path + "_color.png")
        #np.save(os.path.join(output_dir, 'images', '{}_{}_gen_depth_float32.npy'.format(scene_info, image_large_id)), generated_depth_float32)
        generated_depth_float32 = np.load(os.path.join(previous_output_dir, 'images', '{}_{}_gen_depth_float32.npy'.format(scene_info, image_large_id)))

        #tools.print_cv2_data_info(generated_depth_float32)
        
        generated_depth_float32 = tools.scale_up(2, [generated_depth_float32])[0]
        generated_depth_uint16 = (generated_depth_float32 * 65535).astype(np.uint16)
        generated_depth_uint16 = generated_depth_uint16.reshape((generated_depth_uint16.shape[0], generated_depth_uint16.shape[1]))

        cv2.imwrite(os.path.join(output_dir, 'images', '{}_{}_gen_depth_uint16.png'.format(scene_info, image_large_id)), generated_depth_uint16)

        # Getting all ground truth data
        gt_mask, gt_coord, gt_class_ids, gt_scales, gt_domain_label = dataset.load_mask(image_id)
        gt_bbox = nocs_utils.extract_bboxes(gt_mask)
        gt_RTs, _, _, _ = nocs_utils.align(gt_class_ids,
                                        gt_mask,
                                        gt_coord,
                                        source_depth,
                                        intrinsics,
                                        tools.synset_names,
                                        image_path)
        gt_handle_visibility = np.ones_like(gt_class_ids)

        # Generating results that is the same for both source and generated depth
        result, r = nocs_model.predict_coord(image, image_path)
        
        # Image Information
        result['image_id'] = image_id
        result['image_path'] = image_path
        
        # Ground truth information
        result['gt_class_ids'] = gt_class_ids
        result['gt_bboxes'] = gt_bbox
        result['gt_RTs'] = gt_RTs
        result['gt_scales'] = gt_scales
        result['gt_handle_visibility'] = gt_handle_visibility

        # Predicted information
        result['pred_class_ids'] = r['class_ids']
        result['pred_bboxes'] = r['rois']
        result['pred_scores'] = r['scores']  
        result['r'] = r
            
        result['pred_RTs'], result['pred_scales'], _, _ = nocs_utils.align(r['class_ids'],
                                                                        r['masks'],
                                                                        r['coords'],
                                                                        generated_depth_uint16,
                                                                        intrinsics,
                                                                        tools.synset_names,
                                                                        image_path)
    
        
        # Save results into pickle
        with open(pickle_dest, 'wb') as f:
            pickle.dump(result, f)

        gen_depth_result = nocs_model.icp_algorithm(image, image_path, generated_depth_uint16, result, r)
        cv2.imwrite(os.path.join(output_dir, 'images', '{}_{}_gen_depth_RT.png'.format(scene_info, image_large_id)), gen_depth_result)

    return None

def evaluate_data():

    # Evaluating the degree cm mAP of the generated result, stored in pickles

    for depth_type in ["source", "generated"]:

        
        if depth_type == "source":
            continue
        

        result_pkl_list = glob.glob(os.path.join(output_dir, 'results_pickled', '*_{}_results.pickle'.format(depth_type)))
        result_pkl_list = sorted(result_pkl_list)[:num_eval]
        assert len(result_pkl_list)

        #print(result_pkl_list)

        final_result = []

        for pkl_path in tqdm.tqdm(result_pkl_list):

            with open(pkl_path, 'rb') as f:
                try:
                    result = pickle.load(f)
                    result['gt_handle_visibility'] = np.ones_like(result['gt_class_ids'])
                except:
                    print("ERROR WITH {} pickle".format(pkl_path))
                    continue

            if type(result) is list:
                final_result += result
            elif type(result) is dict:
                final_result.append(result)
            else:
                assert False

        output_path = os.path.join(output_dir, "{}_aps".format(depth_type))

        aps = nocs_utils.compute_degree_cm_mAP(final_result, tools.synset_names, output_path,
                                               degree_thresholds = [5, 10, 15],#range(0, 61, 1), 
                                               shift_thresholds= [5, 10, 15], #np.linspace(0, 1, 31)*15, 
                                               iou_3d_thresholds=np.linspace(0, 1, 101),
                                               iou_pose_thres=0.1,
                                               use_matches_for_pose=True)

        with open(os.path.join(output_path, "aps_{}.pickle".format(depth_type)), 'wb') as f:
            pickle.dump(aps,f)

    return None

#---------------------------------------------------------------------
# Constants 

intrinsics = np.array([[591.0125, 0, 322.525], [0, 590.16775, 244.11084], [0, 0, 1]])

#---------------------------------------------------------------------
# Main Code

# Paths
original_dataset_path = r"E:\MASTERS_STUFF\workspace\datasets\NOCS\real"
checkpoint_path = r"E:\MASTERS_STUFF\workspace\networks\DenseDepth\logs\nyu.h5"
previous_output_dir = r'E:\MASTERS_STUFF\workspace\case_study_data\output_logs_2'
output_dir = r'E:\MASTERS_STUFF\workspace\case_study_data\output_logs_3'

# Parameters
generate_data_flag = False
num_eval = 100

if generate_data_flag is True:
    
    generate_data()

else:

    evaluate_data()
    