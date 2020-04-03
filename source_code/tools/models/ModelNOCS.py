# Library Imports
import os
import time
import sys
import pathlib

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' # Basically, SHUT UP TENSORFLOW
import tensorflow
import keras

import skimage
import imutils
import cv2
import numpy as np
import open3d as o3d
from matplotlib import pyplot as plt

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
sys.path.append(str(NETWORKS_DIR))
sys.path.append(str(NOCS_ROOT_DIR))
sys.path.append(str(DENSE_TF_ROOT_DIR))
sys.path.append(str(SOURCE_CODE_DIR))

# Local Imports
from NOCS_CVPR2019 import model as modelib
from NOCS_CVPR2019 import utils as nocs_utils
from NOCS_CVPR2019 import config

import tools

#---------------------------------------------------------------------------------------
# Classes

class NOCS():

    def __init__(self, load_model=False):

        if load_model is True:
            print("\nLoading NOCS Model\n")
            self.load_model()
            print('\nFinishing Loading Model\n')

        # If data is from real dataset 
        self.intrinsics = np.array([[591.0125, 0, 322.525], [0, 590.16775, 244.11084], [0, 0, 1]])

        # If data is from CAMERA dataset
        #self.intrinsics = np.array([[577.5, 0, 319.5], [0., 577.5, 239.5], [0., 0., 1.]])

        return None

    def load_model(self):

        # Models Configuration Class
        config = InferenceConfig()
        #config.display()

        # Model Loading
        self.model = modelib.MaskRCNN(mode="inference",
                                      config=config,
                                      model_dir=str(NOCS_MODEL_DIR))

        # Model's weights loading
        self.model.load_weights(str(NOCS_WEIGHTS_PATH), by_name=True)

        return None

    def predict_coord(self, image, image_path):

        # Making the prediction (detection)
        detect_result = self.model.detect([image], verbose=0)
        r = detect_result[0]

        # Formatting the results to determine the RT (rotation and translations)
        result = {}
        result["image_id"] = image_id = 1
        result["image_path"] = image_path
        result["pred_class_ids"] = r["class_ids"]
        result["pred_bboxes"] = r["rois"]
        result["pred_RTs"] = None
        result["pred_scores"] = r["scores"]

        if len(r["class_ids"]) == 0:
            print("No instances were detected")

        return result, r

    def icp_algorithm(self, image, image_path, depth, result, r):

        # Using ICP algorith to determine RT
        result['pred_RTs'], result['pred_scales'], error_message, elapses = nocs_utils.align(r['class_ids'],
                                                                                             r['masks'],
                                                                                             r['coords'],
                                                                                             depth,
                                                                                             self.intrinsics,
                                                                                             tools.constants.synset_names,
                                                                                             image_path)

        # Drawing the output
        draw_image = self.draw_detections(image, self.intrinsics, tools.constants.synset_names, r['rois'], r['class_ids'],
                                          r['masks'], r['coords'], result['pred_RTs'], r['scores'], result['pred_scales'])

        return draw_image

    def predict(self, image, image_path, depth):

        result, r = self.predict_coord(image, image_path)
        draw_image = self.icp_algorithm(image, image_path, depth, result, r)

        return draw_image

    def draw_detections(self, image, intrinsics, synset_names, pred_bbox, pred_class_ids,
                        pred_mask, pred_coord, pred_RTs, pred_scores, pred_scales, 
                        draw_coord = False, draw_tag = False, draw_RT = True):

        draw_image = image.copy()

        num_pred_instances = len(pred_class_ids)

        for i in range(num_pred_instances):

            # Mask and Coord data
            if draw_coord:
                mask = pred_mask[:, :, i]
                cind, rind = np.where(mask == 1)
                coord_data = pred_coord[:, :, i, :].copy()
                coord_data[:, :, 2] = 1 - coord_data[:, :, 2]
                draw_image[cind,rind] = coord_data[cind, rind] * 255

            # Tag data 
            if draw_tag:
                text = tools.constants.synset_names[pred_class_ids[i]]+'({:.2f})'.format(pred_scores[i])
                draw_image = nocs_utils.draw_text(draw_image, pred_bbox[i], text, draw_box=True)

            # Rotation and Translation data
            if draw_RT:
                RT = pred_RTs[i]
                class_id = pred_class_ids[i]

                xyz_axis = 0.3*np.array([[0, 0, 0], [0, 0, 1], [0, 1, 0], [1, 0, 0]]).transpose()
                transformed_axes = nocs_utils.transform_coordinates_3d(xyz_axis, RT)
                projected_axes = nocs_utils.calculate_2d_projections(transformed_axes, intrinsics)

                bbox_3d = nocs_utils.get_3d_bbox(pred_scales[i,:],0)
                transformed_bbox_3d = nocs_utils.transform_coordinates_3d(bbox_3d, RT)
                projected_bbox = nocs_utils.calculate_2d_projections(transformed_bbox_3d, intrinsics)
                draw_image = nocs_utils.draw(draw_image, projected_bbox, projected_axes, (255, 0, 0))

        return draw_image

class ScenesConfig(config.Config):
    
    """
    Configuration for training on the toy shapes dataset.
    Derives from the base Config class and overrides values specific
    to the toy shapes dataset.
    """
    # Give the configuration a recognizable name
    NAME = "ShapeNetTOI"
    OBJ_MODEL_DIR = os.path.join(NOCS_ROOT_DIR, 'data', 'obj_models')
    # Train on 1 GPU and 8 images per GPU. We can put multiple images on each
    # GPU because the images are small. Batch size is 8 (GPUs * images/GPU).
    GPU_COUNT = 1
    IMAGES_PER_GPU = 2

    # Number of classes (including background)
    NUM_CLASSES = 1 + 6  # background + 6 object categories
    MEAN_PIXEL = np.array([[ 120.66209412, 114.70348358, 105.81269836]])

    IMAGE_MIN_DIM = 480
    IMAGE_MAX_DIM = 640

    RPN_ANCHOR_SCALES = (16, 32, 48, 64, 128)  # anchor side in pixels

    # Reduce training ROIs per image because the images are small and have
    # few objects. Aim to allow ROI sampling to pick 33% positive ROIs.
    TRAIN_ROIS_PER_IMAGE = 64

    # Use a small epoch since the data is simple
    STEPS_PER_EPOCH = 1000

    # use small validation steps since the epoch is small
    VALIDATION_STEPS = 50

    WEIGHT_DECAY = 0.0001
    LEARNING_RATE = 0.001
    LEARNING_MOMENTUM = 0.9

    COORD_LOSS_SCALE = 1
    
    COORD_USE_BINS = True
    if COORD_USE_BINS:
         COORD_NUM_BINS = 32
    else:
        COORD_REGRESS_LOSS   = 'Soft_L1'
   
    COORD_SHARE_WEIGHTS = False
    COORD_USE_DELTA = False

    COORD_POOL_SIZE = 14
    COORD_SHAPE = [28, 28]

    USE_BN = True
    USE_SYMMETRY_LOSS = True


    RESNET = "resnet50"
    TRAINING_AUGMENTATION = True
    SOURCE_WEIGHT = [3, 1, 1] #'ShapeNetTOI', 'Real', 'coco'

class InferenceConfig(ScenesConfig):
    """Configuration for training on the toy shapes dataset.
    Derives from the base Config class and overrides values specific
    to the toy shapes dataset.
    """
    # Give the configuration a recognizable name
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1
