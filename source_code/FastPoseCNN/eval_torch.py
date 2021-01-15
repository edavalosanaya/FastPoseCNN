# Imports
import os
import sys
import argparse
import pathlib
from pprint import pprint
import tqdm
import pandas as pd

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import skimage.io

import pytorch_lightning.overrides.data_parallel as pl_o_d

os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2,3'

# Local Imports
import setup_env
import tools
import lib
import train

#-------------------------------------------------------------------------------
# Constants

PATH = pathlib.Path(os.getenv("LOGS")) / '20-12-24' / '19-09-PW_QLOSS_NEW_LOSS_FUNC-NOCS-resnext50_32x4d-imagenet' / '_' / 'checkpoints' / 'epoch=7.ckpt'

# Run hyperparameters
class DEFAULT_POSE_HPARAM(argparse.Namespace):
    DATASET_NAME = 'NOCS'
    SELECTED_CLASSES = ['bg','camera','laptop'] #tools.pj.constants.NUM_CLASSES[DATASET_NAME]
    BATCH_SIZE = 8
    NUM_WORKERS = 36 # 36 total CPUs
    NUM_GPUS = 4
    DISTRIBUTED_BACKEND = None if NUM_GPUS <= 1 else 'ddp'
    TRAIN_SIZE = 1
    VALID_SIZE = 2000
    DRAW = True
    COLLECT_DATA = False

HPARAM = DEFAULT_POSE_HPARAM()

#-------------------------------------------------------------------------------
# File Main

if __name__ == '__main__':

    # Construct the json path depending on the PATH string
    json_path = PATH.parent.parent / f'pred_gt_{HPARAM.VALID_SIZE}_results.json'

    # Constructing folder of images if not existent
    images_path = PATH.parent.parent / 'images'
    if images_path.exists() is False:
        os.mkdir(str(images_path))

    # Load from checkpoint
    checkpoint = torch.load(PATH)
    OLD_HPARAM = checkpoint['hyper_parameters']

    # Merge the NameSpaces between the model's hyperparameters and 
    # the evaluation hyperparameters
    for attr in OLD_HPARAM.keys():
        setattr(HPARAM, attr, OLD_HPARAM[attr])

    # Determining if collect model's performance data
    # or visualizing the results of the model's performance
    if HPARAM.COLLECT_DATA:

        # Create model
        base_model = lib.PoseRegressor(
            architecture=HPARAM.BACKBONE_ARCH,
            encoder_name=HPARAM.ENCODER,
            encoder_weights=HPARAM.ENCODER_WEIGHTS,
            classes=len(HPARAM.SELECTED_CLASSES),
        )

        # Create PyTorch Lightning Module
        model = train.PoseRegresssionTask.load_from_checkpoint(
            str(PATH),
            model=base_model,
            criterion=None,
            metrics=None
        )

        # Freezing weights to avoid updating the weights
        model.freeze()

        # Put the model into evaluation mode
        model.to('cuda') # ! Make it work with multiple GPUs
        model.eval()

        # Load the PyTorch Lightning dataset
        datamodule = train.PoseRegressionDataModule(
            dataset_name=HPARAM.DATASET_NAME,
            selected_classes=HPARAM.SELECTED_CLASSES,
            batch_size=HPARAM.BATCH_SIZE,
            num_workers=HPARAM.NUM_WORKERS,
            encoder=HPARAM.ENCODER,
            encoder_weights=HPARAM.ENCODER_WEIGHTS,
            train_size=HPARAM.TRAIN_SIZE,
            valid_size=HPARAM.VALID_SIZE
        )

        # Setup the dataset
        datamodule.setup()

        # Obtaining the valid dataset
        valid_dataset = datamodule.datasets['valid']

        # Numpy container for all the data, divided per class
        #pred_gt_data = np.zeros((len(HPARAM.SELECTED_CLASSES)))
        all_matches = []

        # image counter
        image_counter = 0

        # Pass through all the test data of the dataset and collect the predictions
        # and the ground truths
        for batch in tqdm.tqdm(datamodule.val_dataloader()):

            # Forward pass
            with torch.no_grad():
                output = model.forward(batch['image'])

            # Removing auxilary outputs
            aux_outputs = output.pop('auxilary')

            # Convert inputs and outputs to numpy arrays
            numpy_aux_outputs = {k:v.cpu().numpy() for k,v in aux_outputs.items()}
            numpy_inputs = {k:v.cpu().numpy() for k,v in batch.items()}
            numpy_outputs = {k:v.cpu().numpy() for k,v in output.items()}

            # Obtaining the mask
            numpy_outputs['mask'] = numpy_aux_outputs['cat_mask'] #np.argmax(torch.nn.functional.sigmoid(output['mask']).cpu().numpy(), axis=1)

            # Placing the predicted information into the numpy outputs
            numpy_outputs['quaternion'] = output['quaternion'].cpu().numpy()
            
            # Ensure that the value of quaternion is between -1 and 1
            numpy_outputs['quaternion'] /= np.max(np.abs(numpy_outputs['quaternion']))

            # ! Only for now we use the ground truth data that we have not regressed yet
            for key in numpy_inputs.keys():
                if key not in numpy_outputs.keys():
                    numpy_outputs[key] = numpy_inputs[key]

            # Iterate for each sample in the batch to obtain its information
            for i in range(numpy_inputs['mask'].shape[0]):

                # Obtain the single sample data and convert it to dataformat HWC        
                single_preds = {k:tools.dm.set_image_data_format(v[i], 'channels_last') for k,v in numpy_outputs.items()}
                single_gts = {k:tools.dm.set_image_data_format(v[i], 'channels_last') for k,v in numpy_inputs.items()}

                # Obtain the data for the predictions via aggregation
                preds_aggregated_data = tools.dm.aggregate_dense_sample(single_preds, valid_dataset.INTRINSICS)

                # Obtain the data for the gts via aggregation
                gts_aggregated_data = tools.dm.aggregate_dense_sample(single_gts, valid_dataset.INTRINSICS)

                # Selecting clean image if available
                image_key = 'clean_image' if 'clean_image' in single_gts.keys() else 'image'

                # If the neural network did not make a prediction, ignore, else compare
                if len(preds_aggregated_data['instance_id']) == 0:
                    continue

                # Find the matches between pred and gt data
                pred_gt_matches = tools.dm.find_matches(
                    preds_aggregated_data, 
                    gts_aggregated_data,
                    image_tag=image_counter
                )

                # Accumulate all the matches in the sample
                all_matches.extend(pred_gt_matches)

                # If requested to draw the preds and gts
                if HPARAM.DRAW and image_counter < 25:        

                    # Draw a sample's poses
                    gt_pose = tools.dr.draw_RTs(
                        image = single_gts[image_key], 
                        intrinsics = valid_dataset.INTRINSICS,
                        RTs = gts_aggregated_data['RT'],
                        scales = gts_aggregated_data['scales'],
                        color=(0,255,255)
                    )

                    # Draw a sample's poses
                    pose = tools.dr.draw_RTs(
                        image = gt_pose, 
                        intrinsics = valid_dataset.INTRINSICS,
                        RTs = preds_aggregated_data['RT'],
                        scales = preds_aggregated_data['scales'],
                        color=(255,0,255)
                    )

                    # Save the image of the pose
                    skimage.io.imsave(
                        str(images_path / f'{image_counter}_pose.png'),
                        pose
                    )

                    # Visually compare all inputs and outputs of this single sample
                    single_sample_performance_fig = tools.vz.compare_all(
                        single_preds, 
                        single_gts,
                        mask_colormap = valid_dataset.COLORMAP
                    )

                    # Save the figure into the images folder
                    single_sample_performance_fig.savefig(
                        str(images_path / f'{image_counter}.png')
                    )

                    # Clear matplotlib
                    plt.clf()

                # Update image counter
                image_counter += 1

        # Store the simple data into a json file
        tools.jt.save_to_json(json_path, all_matches)

    else:

        # Load the json
        all_matches = tools.jt.load_from_json(json_path)
        cls_metrics = {
            '3d_iou': [[] for cls in HPARAM.SELECTED_CLASSES],
            'degree': [[] for cls in HPARAM.SELECTED_CLASSES],
            'offset': [[] for cls in HPARAM.SELECTED_CLASSES]
        }

        # For each match calculate the 3D IoU, degree error, and offset error 
        for match in tqdm.tqdm(all_matches):

            # Determine representative data of the ground truth and prediction data
            output_data = {
                '3d_bbox': [],
                '3d_center': []
            }

            # For the pred and gt
            for i in range(2):

                # Determine the 3D bounding box for 3D IoU
                """
                camera_coord_3d_bbox = tools.dm.get_3d_bbox(match['scales'][i], 0)
                world_coord_3d_bbox = tools.dm.transform_3d_camera_coords_to_3d_world_coords(
                    camera_coord_3d_bbox,
                    match['RT'][i]
                )
                """

                camera_coord_3d_center = np.array([[0,0,0]]).transpose()
                world_coord_3d_center = tools.dm.transform_3d_camera_coords_to_3d_world_coords(
                    camera_coord_3d_center,
                    match['RT'][i]
                )

                #output_data['3d_bbox'].append(world_coord_3d_bbox.transpose())
                output_data['3d_center'].append(world_coord_3d_center)

            # Calculate the performance 
            iou_3d = tools.dm.get_3d_iou(*match['RT'], *match['scales'])
            degree_error = tools.dm.get_R_degree_error(*match['quaternion'])
            offset_error = tools.dm.get_T_offset_error(*output_data['3d_center'])

            # Store performance metrics depending on the class
            cls_metrics['3d_iou'][match['class_id']].append(iou_3d)
            cls_metrics['degree'][match['class_id']].append(degree_error)
            cls_metrics['offset'][match['class_id']].append(offset_error)

        # Remove background entry
        for key in cls_metrics.keys():
            cls_metrics[key].pop(0)

        # Defining the nature of the metric (higher/lower is better)
        metrics_operators = {
            '3d_iou': np.greater,
            'degree': np.less,
            'offset': np.less
        }

        ########################################################################
        # Generating plots of aps
        num_of_points = 30

        metrics_thresholds = {
            '3d_iou': np.linspace(0, 1, num_of_points),
            'degree': np.linspace(0, 60, num_of_points),
            'offset': np.linspace(0, 10, num_of_points)
        }

        # Calculate the aps for each metric
        aps = tools.dm.calculate_aps(
            cls_metrics, 
            metrics_thresholds,
            metrics_operators
        )

        # Save the raw aps to a JSON file
        #aps_json_path = PATH.parent.parent / f'{HPARAM.VALID_SIZE}_aps_values_plot.json'
        #tools.jt.save_to_json(aps_json_path, aps_complete_data)

        # Save the raw aps to Excel file
        excel_path = PATH.parent.parent / f'{HPARAM.VALID_SIZE}_aps_values_plot.xlsx'
        tools.et.save_aps_to_excel(excel_path, metrics_thresholds, aps)
        
        # Plotting aps
        fig = tools.vz.plot_aps(
            aps,
            titles=['3D Iou AP', 'Rotation AP', 'Translation AP'],
            x_ranges=list(metrics_thresholds.values()),
            cls_names=HPARAM.SELECTED_CLASSES[1:] + ['mean'],
            x_axis_labels=['3D IoU %', 'Rotation error/degree', 'Translation error/cm']
        )

        # Saving the plot
        fig.savefig(
            str(PATH.parent.parent / f'all_metrics_{HPARAM.VALID_SIZE}_aps.png')
        )

        ########################################################################
        # Generating tabular data for comparision with state-of-the-art 
        # methods

        metrics_thresholds = {
            '3d_iou': np.array([0.25, 0.50]),
            'degree': np.array([5, 10]),
            'offset': np.array([.05, .10])
        }

        # Calculate the aps for each metric
        aps = tools.dm.calculate_aps(
            cls_metrics, 
            metrics_thresholds,
            metrics_operators
        )

        # Saving the output data into excel
        excel_path = PATH.parent.parent / f'{HPARAM.VALID_SIZE}_aps_values_table.xlsx'
        tools.et.save_aps_to_excel(excel_path, metrics_thresholds, aps)