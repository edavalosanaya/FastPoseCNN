# Library Import
import os
import sys
import pathlib

import cv2
from matplotlib import pyplot as plt
import pickle
import numpy as np
import tqdm 
import glob

# Determing the ROOT_DIR from CWD_DIR
CWD_DIR = pathlib.Path.cwd()
ROOT_DIR = CWD_DIR.parents[len(CWD_DIR.parts) - CWD_DIR.parts.index('MastersProject') - 2]
NETWORKS_DIR = ROOT_DIR / 'networks'

# Getting NOCS paths
NOCS_ROOT_DIR = NETWORKS_DIR / "NOCS_CVPR2019"
NOCS_MODEL_DIR = NOCS_ROOT_DIR / "logs"
NOCS_COCO_MODEL_PATH = NOCS_MODEL_DIR / "mask_rcnn_coco.h5"
NOCS_WEIGHTS_PATH = NOCS_MODEL_DIR / "nocs_rcnn_res50_bin32.h5"

# Getting DenseDepth paths
DENSE_ROOT_DIR = NETWORKS_DIR / "DenseDepth"
DENSE_TF_ROOT_DIR = DENSE_ROOT_DIR / "Tensorflow"

# Source Code path
SOURCE_CODE_DIR = ROOT_DIR / "source_code"

# Appending necessary paths
sys.path.append(str(SOURCE_CODE_DIR))
sys.path.append(str(NETWORKS_DIR))
sys.path.append(str(NOCS_ROOT_DIR))
sys.path.append(str(DENSE_TF_ROOT_DIR))

# Local Imports

import tools
from NOCS_CVPR2019 import dataset as nocs_dataset
from NOCS_CVPR2019 import utils as nocs_utils

# Paths

original_dataset_path = ROOT_DIR / 'datasets' / 'NOCS' / 'real'
checkpoint_path = ROOT_DIR / 'networks' / 'DenseDepth' / 'logs' / 'nyu.h5'
depth_npys_location = ROOT_DIR / 'case_study_data' / 'large_tests' / 'output_logs_2' / 'images'
tests_output_dir = ROOT_DIR / 'case_study_data' / 'large_tests' / 'output_logs_4'

#---------------------------------------------------------------------
# Functions

def generate_data():

    # Dataset Configuration
    config = tools.models.InferenceConfig()

    # Ground Dataset
    dataset = nocs_dataset.NOCSDataset(tools.constants.synset_names, 'test', config)
    dataset.load_real_scenes(original_dataset_path)
    dataset.prepare(tools.constants.class_map)

    # Loading the models

    print("\nLoading models...\n")
    dense_depth_net = tools.models.DenseDepth(str(checkpoint_path))
    nocs_model = tools.models.NOCS(load_model=True)
    print("\nFinished loading models\n")

    image_ids = dataset.image_ids[:num_eval]
    #image_ids = dataset.image_ids

    for image_id in tqdm.tqdm(image_ids):

        # Obtaining images information
        image_path = dataset.image_info[image_id]["path"]
        pathlib_image_path = pathlib.Path(image_path)
        scene_info = pathlib_image_path.parts[-2]
        image_large_id = pathlib_image_path.parts[-1].replace("_color.png", "")
        
        # Loading image and source depth
        image = dataset.load_image(image_id)
        source_depth = dataset.load_depth(image_id)

        # Generating the new depth
        generated_depth_path = depth_npys_location / '{}_{}_gen_depth_float32.npy'.format(scene_info, image_large_id)

        if generated_depth_path.exists(): # If generated data exists, do not regenerate it again
            generated_depth_float32 = np.load(str(generated_depth_path))
        else: 
            generated_depth_float32 = dense_depth_net.predict(image_path + "_color.png")

        #tools.print_cv2_data_info(generated_depth_float32)
        
        generated_depth = tools.img_aug.dd_to_nocs_depth(generated_depth_float32)
        generated_depth = tools.img_aug.factor_dd_depth_by_nocs_depth(generated_depth, source_depth)

        tools.visualize.print_img_info(generated_depth)

        # Getting all ground truth data
        gt_mask, gt_coord, gt_class_ids, gt_scales, gt_domain_label = dataset.load_mask(image_id)
        gt_bbox = nocs_utils.extract_bboxes(gt_mask)
        gt_RTs, _, _, _ = nocs_utils.align(gt_class_ids,
                                        gt_mask,
                                        gt_coord,
                                        source_depth,
                                        intrinsics,
                                        tools.constants.synset_names,
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

        for title, depth in {"source": source_depth, "generated": generated_depth}.items():
            
            result['pred_RTs'], result['pred_scales'], _, _ = nocs_utils.align(r['class_ids'],
                                                                               r['masks'],
                                                                               r['coords'],
                                                                               depth,
                                                                               intrinsics,
                                                                               tools.constants.synset_names,
                                                                               image_path)

            pickle_filename = "{}_{}_{}_results.pickle".format(scene_info, image_large_id, title)
            pickle_dest = tests_output_dir / 'results_pickled' / pickle_filename

            # Save results into pickle
            with open(str(pickle_dest), 'wb') as f:
                pickle.dump(result, f)

            # Save image results
            image_result = nocs_model.icp_algorithm(image, image_path, depth, result, r)
            image_result_filename = '{}_{}_gen_depth_RT.png'.format(scene_info, image_large_id)
            image_result_path = tests_output_dir / 'images' / image_result_filename
            cv2.imwrite(str(image_result_path), image_result)

    return None

def evaluate_data():

    # Evaluating the degree cm mAP of the generated result, stored in pickles

    for depth_type in ["source", "generated"]:        

        result_pkl_path = tests_output_dir / 'results_pickled' / '*_{}_results.pickle'.format(depth_type)
        result_pkl_list = glob.glob(str(result_pkl_path))
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

        aps_output_dir = tests_output_dir / 'aps' / depth_type
        aps_pickle_path = aps_output_dir / 'aps_{}.pickle'.format(depth_type)

        aps = nocs_utils.compute_degree_cm_mAP(final_result, tools.constants.synset_names, str(aps_output_dir),
                                               degree_thresholds = [5, 10, 15],#range(0, 61, 1), 
                                               shift_thresholds= [5, 10, 15], #np.linspace(0, 1, 31)*15, 
                                               iou_3d_thresholds=np.linspace(0, 1, 101),
                                               iou_pose_thres=0.1,
                                               use_matches_for_pose=True)

        with open(str(aps_pickle_path), 'wb') as f:
            pickle.dump(aps,f)

    return None

#---------------------------------------------------------------------
# Constants 

intrinsics = np.array([[591.0125, 0, 322.525], [0, 590.16775, 244.11084], [0, 0, 1]])

#---------------------------------------------------------------------
# Main Code

# Parameters
generate_data_flag = False
num_eval = 10

if generate_data_flag is True:
    
    generate_data()

else:

    evaluate_data()
    