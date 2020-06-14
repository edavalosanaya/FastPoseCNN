# Library Imports
import os
import time
import cv2
import numpy as np
import sys

# Getting NOCS Root directory
NOCS_ROOT_DIR = os.path.join(os.getcwd(), "NOCS_CVPR2019")
sys.path.append(NOCS_ROOT_DIR)

# Getting the logs directory within NOCS
NOCS_MODEL_DIR = os.path.join(NOCS_ROOT_DIR, "logs")

# Getting COCO trained weights path
NOCS_COCO_MODEL_PATH = os.path.join(NOCS_MODEL_DIR, "mask_rcnn_coco.h5")

# Getting NOCS trained weights path
NOCS_WEIGHTS_PATH = os.path.join(NOCS_MODEL_DIR, "nocs_rcnn_res50_bin32.h5")

# Local Imports
from NOCS_CVPR2019 import model as modelib
from NOCS_CVPR2019 import utils
from NOCS_CVPR2019 import config

#-----------------------------------------------------------------------------------------------
# Classes

class ScenesConfig(config.Config):
    """Configuration for training on the toy shapes dataset.
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
#     if COORD_SHARE_WEIGHTS:
#         USE_BN = False

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

#-----------------------------------------------------------------------------------------------
# Constants

coco_names = ['BG', 'person', 'bicycle', 'car', 'motorcycle', 'airplane',
                'bus', 'train', 'truck', 'boat', 'traffic light',
                'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird',
                'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear',
                'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie',
                'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
                'kite', 'baseball bat', 'baseball glove', 'skateboard',
                'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup',
                'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
                'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza',
                'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed',
                'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote',
                'keyboard', 'cell phone', 'microwave', 'oven', 'toaster',
                'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors',
                'teddy bear', 'hair drier', 'toothbrush']


synset_names = ['BG', #0
                'bottle', #1
                'bowl', #2
                'camera', #3
                'can',  #4
                'laptop',#5
                'mug'#6
                ]

class_map = {
    'bottle': 'bottle',
    'bowl':'bowl',
    'cup':'mug',
    'laptop': 'laptop',
}

#-----------------------------------------------------------------------------------------------
# Functions

def draw_detections(image, image_id, intrinsics, synset_names, pred_bbox, pred_class_ids,
                    pred_mask, pred_coord, pred_RTs, pred_scores, pred_scales, 
                    draw_coord = False, draw_tag = False, draw_RT = False):

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
            text = synset_names[pred_class_ids[i]]+'({:.2f})'.format(pred_scores[i])
            draw_image = utils.draw_text(draw_image, pred_bbox[i], text, draw_box=True)

        # Rotation and Translation data
        if draw_RT:
            RT = pred_RTs[i]
            class_id = pred_class_ids[i]

            xyz_axis = 0.3*np.array([[0, 0, 0], [0, 0, 1], [0, 1, 0], [1, 0, 0]]).transpose()
            transformed_axes = utils.transform_coordinates_3d(xyz_axis, RT)
            projected_axes = utils.calculate_2d_projections(transformed_axes, intrinsics)

            bbox_3d = utils.get_3d_bbox(pred_scales[i,:],0)
            transformed_bbox_3d = utils.transform_coordinates_3d(bbox_3d, RT)
            projected_bbox = utils.calculate_2d_projections(transformed_bbox_3d, intrinsics)
            draw_image = utils.draw(draw_image, projected_bbox, projected_axes, (255, 0, 0))

    return draw_image

#-----------------------------------------------------------------------------------------------
# Main Code

if __name__ == "__main__":

    # Models Configuration Class
    config = InferenceConfig()
    config.display()

    # Model Loading
    model = modelib.MaskRCNN(mode="inference",
                             config=config,
                             model_dir=NOCS_MODEL_DIR)

    # Model's weights loading
    model.load_weights(NOCS_WEIGHTS_PATH, by_name=True)

    # Detecting image, while considering the image's intrinsics
    image_path = os.path.join(r"E:\MASTERS_STUFF\workspace\NOCS_CVPR2019\data\real_test\real_test\scene_1", "0000_color.png")
    depth_path = os.path.join(r"E:\MASTERS_STUFF\workspace\NOCS_CVPR2019\data\real_test\real_test\scene_1", "0000_depth.png")
    image, depth = cv2.imread(image_path)[:, :, :3], cv2.imread(depth_path, -1)
    image_id = 1

    depth_original = depth.copy()
    
    # Formatting image and depth to fit the model's input
    image = image[:, :, ::-1]
    
    if len(depth.shape) == 3:
        # This is encoded depth image, let's convert
        depth16 = np.uint16(depth[:, :, 1]*256) + np.uint16(depth[:, :, 2]) # NOTE: RGB is actually BGR in opencv
        depth16 = depth16.astype(np.uint16)
    elif len(depth.shape) == 2 and depth.dtype == 'uint16':
        depth16 = depth
    else:
        assert False, '[ Error ]: Unsupported depth type.'
    
    depth = depth16
    
    # If data is from real dataset 
    #intrinsics = np.array([[591.0125, 0, 322.525], [0, 590.16775, 244.11084], [0, 0, 1]])

    # If data is from CAMERA dataset
    intrinsics = np.array([[577.5, 0, 319.5], [0., 577.5, 239.5], [0., 0., 1.]])

    # Making the prediction (detection)
    start = time.time()
    detect_result = model.detect([image], verbose=0)
    r = detect_result[0]
    end = time.time()
    delta = end - start

    print("Delta time for image detection: {}".format(delta))

    # Formatting the results to determine the RT (rotation and translations)
    result = {}
    result["image_id"] = 1
    result["image_path"] = image_path
    result["pred_class_ids"] = r["class_ids"]
    result["pred_bboxes"] = r["rois"]
    result["pred_RTs"] = None
    result["pred_scores"] = r["scores"]

    if len(r["class_ids"]) == 0:
        print("No instances were detected")

    # Using ICP algorith to determine RT
    print("Aligning predictions...")
    start = time.time()
    result['pred_RTs'], result['pred_scales'], error_message, elapses = utils.align(r['class_ids'],
                                                                                    r['masks'],
                                                                                    r['coords'],
                                                                                    depth,
                                                                                    intrinsics,
                                                                                    synset_names,
                                                                                    image_path)
    end = time.time()
    delta = end - start
    print("Delta time for ICP algorithm: {}".format(delta))

    # Drawing the output
    draw_image = draw_detections(image, image_id, intrinsics, synset_names, r['rois'], r['class_ids'],
                                  r['masks'], r['coords'], result['pred_RTs'], r['scores'], result['pred_scales'],
                                  draw_coord=False, draw_tag=False, draw_RT=True)

    # Formating depth to be more visible
    min_val, max_val, _, _ = cv2.minMaxLoc(depth_original)
    depth_original = cv2.convertScaleAbs(depth_original, depth_original, 255 / max_val)

    cv2.imshow('Input RGB', image)
    cv2.imshow('Input Depth', depth)
    cv2.imshow('Formatted Depth', depth_original)
    cv2.imshow('Output', draw_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

