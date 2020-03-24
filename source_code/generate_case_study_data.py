# Library Import

import os
import sys
import cv2
from matplotlib import pyplot as plt
import pickle

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

#---------------------------------------------------------------------
# Main Code

# Neural Network Parameters
checkpoint_path = r"E:\MASTERS_STUFF\workspace\DenseDepth\logs\nyu.h5"

# Data Parameters
database_path = r"E:\MASTERS_STUFF\workspace\datasets\NOCS\real\test"
output_data_path = r"E:\MASTERS_STUFF\workspace\case_study_data\real_dataset_results_2"

number_of_output = 5
skip_repetitions = True

# Loading models before iteration 
print("\nLoading models...\n")
dense_depth_net = tools.DenseDepth(checkpoint_path)
nocs_model = tools.NOCS(load_model=True)
print("\nFinished loading models\n")

# Beginning of actual code
scene_list = [os.path.join(database_path, scene_dir) for scene_dir in os.listdir(database_path)] # making the list have absolute paths

for scene_path in scene_list:

    scene_destination_folder = os.path.join(output_data_path, scene_path.split("\\")[-1])

    if os.path.isdir(scene_destination_folder) is False:
        os.mkdir(scene_destination_folder)

    file_list = os.listdir(scene_path)

    # Trimming file list to only color images and to a smaller size
    file_list = [filename for filename in file_list if filename.endswith("color.png") is True]
    file_list = file_list[:number_of_output]

    for image_filename in file_list:

        # Loading source image and depth 
        image_id = image_filename.split("_")[0]
        depth_filename = image_id + "_depth.png"

        # Saving output to destination folder
        destination_folder_path = os.path.join(scene_destination_folder, image_id)

        # If new image, create dest folder. Elif reptition, avoid rework.
        if os.path.isdir(destination_folder_path) is False: # Create new folder
            os.mkdir(destination_folder_path)
        else:
            print("{} has been skipped".format(image_id))

        # Finish loading images
        image_path = os.path.join(scene_path, image_filename)
        depth_path = os.path.join(scene_path, depth_filename)

        image = cv2.imread(image_path)
        depth = cv2.imread(depth_path, -1)
        depth = tools.normalize_depth(depth)

        # Checking if the image loaded correctly (possible wrong image path)
        assert len(image.shape) >= 3

        # DenseDepth - Creating generated depth
        print("\nDenseDepth - Predicting Depth\n")
        gen_depth = dense_depth_net.predict(image_path)
        gen_depth = tools.dense_depth_to_nocs_depth(gen_depth)
        gen_depth = tools.nocs_depth_formatting(gen_depth)
        print("\nDenseDepth - Finished\n")

        # Histogram Matching Generated Depth to Source Depth
        mod_gen_depth = tools.hist_matching(gen_depth, tools.calculate_cdf(depth))

        # NOCS - Creating result and r values
        print("\nNOCS - Predicting results and r\n")
        result, r = nocs_model.predict_coord(image, image_path)
        data = {"result": result, "r": r}
        print("\nNOCS - Finished\n")

        # ICP with both depths
        print("\nRunning ICP algorithm\n")
        sou_depth_result = nocs_model.icp_algorithm(image, image_path, depth, result, r)
        gen_depth_result = nocs_model.icp_algorithm(image, image_path, gen_depth, result, r)
        mod_gen_depth_result = nocs_model.icp_algorithm(image, image_path, mod_gen_depth, result, r)

        # Creating visual version of both depths
        vis_sou_depth = tools.making_depth_easier_to_see(depth)
        vis_gen_depth = tools.making_depth_easier_to_see(gen_depth)
        vis_mod_gen_depth = tools.making_depth_easier_to_see(mod_gen_depth)

        # Creating historgram of the non-visual version of both depths
        print("\nCreating histograms\n")
        fig, axes = plt.subplots(1, 3, sharey=True, tight_layout=False)
        fig.suptitle("Depth Comparision Histograms\n")

        axes[0].hist(depth.ravel(), 256, [0,256])
        axes[1].hist(gen_depth.ravel(), 256, [0,256])
        axes[2].hist(mod_gen_depth.ravel(), 256, [0,256])

        axes[0].set_title("Source")
        axes[1].set_title("Generated")
        axes[2].set_title("Histogram Matched Gen.")

        #print("\Visualizing input and output\n")
        #tools.visualize(["Image", "Source Depth", "Generated Depth", "Source Depth Results", "Generated Depth Results"], image, vis_sou_depth, vis_gen_depth, sou_depth_result, gen_depth_result)
        #plt.show()

        # Now with destination folder present, store all output into destination folder
        print("\nSaving outputs\n")
        print("\nSaving images first\n")
        cv2.imwrite(os.path.join(destination_folder_path, image_filename), image)
        cv2.imwrite(os.path.join(destination_folder_path, depth_filename), depth)

        cv2.imwrite(os.path.join(destination_folder_path, "generated_{}_depth.png".format(image_id)), gen_depth)
        cv2.imwrite(os.path.join(destination_folder_path, "modified_generated_{}_depth.png".format(image_id)), mod_gen_depth)
        
        cv2.imwrite(os.path.join(destination_folder_path, "generated_{}_depth_visualized.png".format(image_id)), vis_gen_depth)
        cv2.imwrite(os.path.join(destination_folder_path, "modified_generated_{}_depth_visualized.png".format(image_id)), vis_mod_gen_depth)
        cv2.imwrite(os.path.join(destination_folder_path, "source_{}_depth_visualized.png".format(image_id)), vis_sou_depth)

        cv2.imwrite(os.path.join(destination_folder_path, "generated_{}_RT.png".format(image_id)), gen_depth_result)
        cv2.imwrite(os.path.join(destination_folder_path, "modified_generated_{}_RT.png".format(image_id)), mod_gen_depth_result)
        cv2.imwrite(os.path.join(destination_folder_path, "source_{}_RT.png".format(image_id)), sou_depth_result)

        print("\nSaving results and r pickle\n")
        with open(os.path.join(destination_folder_path, "{}_color_nocs_result.pickle".format(image_id)), 'wb') as f:
            pickle.dump(data, f, pickle.HIGHEST_PROTOCOL)

        print("\nSaving histogram figure\n")
        fig.savefig(os.path.join(destination_folder_path, "depth_comparision_histogram.png"))

        print("\nFinished one iteration")
