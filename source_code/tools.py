# Library Imports
import os
import time
import glob
import sys

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' # Basically, SHUT UP TENSORFLOW
import tensorflow
import keras

import skimage
import imutils
import cv2
import numpy as np
import open3d as o3d
from matplotlib import pyplot as plt

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
from NOCS_CVPR2019 import model as modelib
from NOCS_CVPR2019 import utils as nocs_utils
from NOCS_CVPR2019 import config
from DenseDepth import layers
from DenseDepth import utils as dd_utils

#-----------------------------------------------------------------------------------------------
# Constants

#https://stackoverflow.com/a/36785314  CV2 Flags and their values

cv2_flags_range = {(cv2.CV_8U, 'uint8'): [0,255],
                   (cv2.CV_8S, 'sint8'): [-128,127],
                   (cv2.CV_16U,'uint16'): [0,65535],
                   (cv2.CV_16S,'sint16'): [-32768,32767]}

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

#-------------------------------------------------------------------------------------
# Functions

#### General Open3D Functions

def pcd_visualization(color_path, depth_path, intrinsics = False):

    transformation_matrix = [[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]]

    if intrinsics == False:
        intrinsics = o3d.camera.PinholeCameraIntrinsic(o3d.camera.PinholeCameraIntrinsicParameters.PrimeSenseDefault)

    # Visualizing src depth and color
    color_raw = o3d.io.read_image(color_path)
    src_depth_raw = o3d.io.read_image(depth_path)

    rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(color_raw, src_depth_raw)
    pcd = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd_image, intrinsics)
    pcd.transform(transformation_matrix)

    o3d.visualization.draw_geometries([pcd])

    return None

def open3d_plot(color_path, depth_path):

    # Visualizing src depth and color
    color_raw = o3d.io.read_image(color_path)
    depth_raw = o3d.io.read_image(depth_path)
    rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(color_raw, depth_raw)

    plt.subplot(1, 2, 1)
    plt.title('Grayscale Image')
    plt.imshow(rgbd_image.color)
    plt.subplot(1, 2, 2)
    plt.title('Depth Image')
    plt.imshow(rgbd_image.depth)
    plt.show()

    return None

def rgbd_visualize(rgb_path, depth_path):

    open3d_plot(rgb_path, depth_path)
    pcd_visualization(rgb_path, depth_path)

    return None

#### General CV2 Functions

def scale_up(scale, image):

    return dd_utils.scale_up(scale, image) 

def print_cv2_data_info(cv2_data): # Could be an image (RGB) or depth

    print("cv2 data information:")
    print("Shape: {}".format(cv2_data.shape))
    print("Type: {}".format(cv2_data.dtype))

    try:
        #min_val, max_val, _, _ = cv2.minMaxLoc(cv2_data) # Causes errors sometimes
        
        numpy_data = cv2_data.ravel()
        min_val = np.amin(numpy_data)
        max_val = np.amax(numpy_data)
        
        print("Min: {} Max: {}".format(min_val, max_val))
    except:
        pass

    return None

def visualize(titles_list, *argv):

    for i, image in enumerate(argv):

        if titles_list is None:
            title = str(i)
        else:
            title = titles_list[i]
        
        cv2.imshow("{}".format(title), image)

    cv2.waitKey(0)
    cv2.destroyAllWindows()

    return None

def match_image(src_image, target_image, size=True, dtype=True):

    new_image = src_image.copy()

    if dtype == True:
        
        # Changing type and normalizing
        cv2_flag, cv2_flag_range = dtype_numpy2cv2(target_image.dtype)
        new_image = cv2.normalize(src=src_image, dst=None, alpha=cv2_flag_range[0], beta=cv2_flag_range[1], norm_type=cv2.NORM_MINMAX, dtype=cv2_flag)

    if size == True:
        # Matching size (width and height)
        w = target_image.shape[1]
        h = target_image.shape[0]
        dim = (w, h)
        new_image = cv2.resize(new_image, dim, interpolation = cv2.INTER_AREA)

    return new_image

def modify_image(src_image, size=False, dtype=False, max_val=False, min_val=False):

    new_image = src_image.copy()

    if dtype is not False:

        cv2_flag, cv2_flag_range = dtype_numpy2cv2(dtype)
        
        if max_val is False: max_val = cv2_flag_range[1]
        if min_val is False: min_val = cv2_flag_range[0]

        new_image = cv2.normalize(src=src_image, dst=None, alpha=min_val, beta=max_val, norm_type=cv2.NORM_MINMAX, dtype=cv2_flag)

    if size is not False:
        # Matching size (width and height)
        w = size[1]
        h = size[0]
        dim = (w, h)
        new_image = cv2.resize(new_image, dim, interpolation = cv2.INTER_AREA)

    return new_image

#### Depth-related Functions

def dense_depth_to_nocs_depth(depth):

    if depth.dtype != "uint16":
        depth = cv2.normalize(src=depth, dst=None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_16U)

    depth = scale_up(2, depth)

    return depth

def nocs_depth_formatting(depth):

    if len(depth.shape) == 3:
        # This is encoded depth image, let's convert
        formatted_depth = np.uint16(depth[:, :, 1]*256) + np.uint16(depth[:, :, 2]) # NOTE: RGB is actually BGR in opencv
        formatted_depth = depth16.astype(np.uint16)
    elif len(depth.shape) == 2 and depth.dtype == 'uint16':
        formatted_depth = depth
    else:
        assert False, '[ Error ]: Unsupported depth type.'

    return formatted_depth

def normalize_depth(depth):

    return cv2.normalize(src=depth, dst=None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_16U)

def making_depth_easier_to_see(depth):

    # Formating depth to be more visible
    min_val, max_val, _, _ = cv2.minMaxLoc(depth)
    new_depth = cv2.convertScaleAbs(depth, depth, 255 / max_val)

    return new_depth

#### Histogram Functions

def plot_histogram(images):

    if type(images) is not list and type(images) is np.ndarray: # Single image
        images = list(images)
    elif type(images) is list: # Multiple images
        pass
    else:
        raise RuntimeError("Input invalid")

    print("Plotting Histogram")

    for image in images:
        _, dtype_range = dtype_numpy2cv2(image.dtype)
        max_val = dtype_range[1]
        
        if max_val >= 500:
            n_bins = 500
        else:
            n_bins = max_val

        plt.hist(image.ravel(), n_bins, [0,max_val+1])

    plt.show()     

    return 

def calculate_cdf(image):

    print("image dtype: {}".format(image.dtype))

    _, dtype_range = dtype_numpy2cv2(image.dtype)
    max_val = dtype_range[1]

    cdf, b = skimage.exposure.cumulative_distribution(image)
    cdf = np.insert(cdf, 0, [0]*b[0])
    cdf = np.append(cdf, [1]*(max_val-b[-1]))

    return cdf

def hist_matching(image, target_image):

    assert image.dtype == target_image.dtype

    _, dtype_range = dtype_numpy2cv2(image.dtype)
    max_val = dtype_range[1]

    pixels = np.arange(max_val + 1)
    image_cdf = calculate_cdf(image)
    target_cdf = calculate_cdf(target_image)

    new_pixels = np.interp(image_cdf, target_cdf, pixels)
    new_image = (np.reshape(new_pixels[image.ravel()], image.shape)).astype(image.dtype)

    return new_image

def draw_hist_image(hist, hist_height, hist_width, nbins):

    # Normalize histogram
    cv2.normalize(hist, hist, hist_height, cv2.NORM_MINMAX)

    bin_width = int(hist_width/nbins)

    h = np.zeros((hist_height, hist_width))
    bins = np.arange(nbins, dtype=np.int32).reshape(nbins,1)

    for x, y in enumerate(hist):

        cv2.rectangle(h, (x*bin_width,y * 2), (x*bin_width + bin_width-1, hist_height), (255), -1)

    h = np.flipud(h)

    return h

#### Miscellanious Functions

def disable_print():
    sys.stdout = open(os.devnull, 'w')

def enable_print():
    sys.stdout = sys.__stdout__

def deco_silence(func):

    def wrapper(func):

        disable_print()
        func()
        enable_print()

        return None
    
    return wrapper(func)

def dtype_numpy2cv2(numpy_dtype):

    target_dtype = str(numpy_dtype) # Converting numpy.dtype to string

    # Converting numpy dtype to cv2 dtype flags with their ranges
    for dtype_tuple in cv2_flags_range:
        if dtype_tuple[1].find(target_dtype) != -1:

            cv2_flag = dtype_tuple[0]
            cv2_flag_range = cv2_flags_range[dtype_tuple]

    return cv2_flag, cv2_flag_range

#-------------------------------------------------------------------------------------
# Classes

class NOCS():

    def __init__(self, load_model=False):

        if load_model is True:
            self.load_model()

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
                                      model_dir=NOCS_MODEL_DIR)

        # Model's weights loading
        self.model.load_weights(NOCS_WEIGHTS_PATH, by_name=True)

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
                                                                                             synset_names,
                                                                                             image_path)

        # Drawing the output
        draw_image = self.draw_detections(image, self.intrinsics, synset_names, r['rois'], r['class_ids'],
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
                text = synset_names[pred_class_ids[i]]+'({:.2f})'.format(pred_scores[i])
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

class DenseDepth():

    def __init__(self, checkpoint_path, load_model=True):

        self.checkpoint_path = checkpoint_path

        if load_model is True:
            self.load_model()

        return None

    def load_model(self):

        self.custom_objects = {'BilinearUpSampling2D': layers.BilinearUpSampling2D, 'depth_loss_function': None}
        self.model = keras.models.load_model(self.checkpoint_path, custom_objects=self.custom_objects, compile=False)

        return None

    def predict(self, image_path):

        inputs = dd_utils.load_images(glob.glob(image_path))

        output = dd_utils.predict(self.model, inputs, minDepth=10, maxDepth=1000)[0]
        #flip_output = dd_utils.predict(self.model, inputs[::-1,:] / 255, minDepth=0, maxDepth=1000)[0] * 10

        #true_output = ( (0.5 * outputs) + (0.5 * flip_output) )

        return output

    def compute_errors(self, gt, pred):
        
        return dd_utils.compute_errors(gt, pred)

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
