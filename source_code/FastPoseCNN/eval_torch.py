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

os.environ['CUDA_VISIBLE_DEVICES'] = ''

# Local Imports
import setup_env
import tools
import lib
import train
from config import DEFAULT_POSE_HPARAM

#-------------------------------------------------------------------------------
# Constants

PATH = pathlib.Path(os.getenv("LOGS")) / 'good_saved_runs' / '07-43-MSE+AGG-NOCS-resnet18-imagenet' / '_' / 'checkpoints' / 'last.ckpt'

HPARAM = DEFAULT_POSE_HPARAM()
HPARAM.VALID_SIZE = 2000
HPARAM.HV_NUM_OF_HYPOTHESES = 501

COLLECT_DATA = False
DRAW = True
TOTAL_DRAW_IMAGES = 50
APS_NUM_OF_POINTS = 50

#-------------------------------------------------------------------------------
# File Main

if __name__ == '__main__':

    # Construct the json path depending on the PATH string
    pth_path = PATH.parent.parent / f'pred_gt_{HPARAM.VALID_SIZE}_results.pth'

    # Constructing folder of images if not existent
    images_path = PATH.parent.parent / 'images'
    if images_path.exists() is False:
        os.mkdir(str(images_path))

    # Load from checkpoint
    checkpoint = torch.load(PATH, map_location=torch.device('cpu'))
    OLD_HPARAM = checkpoint['hyper_parameters']

    # Merge the NameSpaces between the model's hyperparameters and 
    # the evaluation hyperparameters
    for attr in OLD_HPARAM.keys():
        setattr(HPARAM, attr, OLD_HPARAM[attr])

    # Determining if collect model's performance data
    # or visualizing the results of the model's performance
    if COLLECT_DATA:

        # Create model
        base_model = lib.PoseRegressor(
            HPARAM,
            intrinsics=torch.from_numpy(tools.pj.constants.INTRINSICS[HPARAM.DATASET_NAME]).float(),
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
            metrics=None,
            HPARAM=HPARAM
        )

        # Freezing weights to avoid updating the weights
        model.freeze()

        # Put the model into evaluation mode
        #model.to('cuda') # ! Make it work with multiple GPUs
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
                outputs = model.forward(batch['image'])

            # Obtaining the aggregated values for the both the ground truth
            agg_gt = model.model.agg_hough_and_generate_RT(
                batch['mask'],
                data=batch
            )

            # Determine matches between the aggreated ground truth and preds
            gt_pred_matches = lib.gtf.batchwise_find_matches(
                outputs['auxilary']['agg_pred'],
                agg_gt
            )

            # Draw the output of the model
            if DRAW and image_counter <= TOTAL_DRAW_IMAGES:
                
                # Accounting for the new image
                image_counter += 1
                
                # Generating massive figure
                gt_fig, pred_fig, poses_fig = tools.vz.compare_all_performance(
                    batch,
                    outputs,
                    gt_pred_matches,
                    valid_dataset.INTRINSICS,
                    valid_dataset.COLORMAP
                )

                # Saving figure
                gt_fig.savefig(
                    str(images_path / f'{image_counter}_gt.png'), 
                    dpi=600
                )
                pred_fig.savefig(
                    str(images_path / f'{image_counter}_pred.png'), 
                    dpi=600
                )
                poses_fig.savefig(
                    str(images_path / f'{image_counter}_poses.png'), 
                    dpi=600
                )


            # Saving the matched data
            all_matches.append(gt_pred_matches)

        # Store the simple data into a json file
        torch.save(all_matches, pth_path)

    else:

        # Raw data
        raw_data = {
            '3d_iou': {},
            'degree_error': {},
            'offset_error': {}
            }

        # Defining the nature of the metric (higher/lower is better)
        metrics_operator = {
            '3d_iou': torch.greater,
            'degree_error': torch.less,
            'offset_error': torch.less
        }

        # The thresholds for the figure
        figure_metrics_thresholds = {
            '3d_iou': torch.linspace(0, 1, APS_NUM_OF_POINTS),
            'degree_error': torch.linspace(0, 60, APS_NUM_OF_POINTS),
            'offset_error': torch.linspace(0, 10, APS_NUM_OF_POINTS)
        }

        # The thresholds for the table
        table_metrics_thresholds = {
            '3d_iou': torch.tensor([0.25, 0.50]),
            'degree_error': torch.tensor([5, 10]),
            'offset_error': torch.tensor([5, 10])
        }

        # Load the .pth file with the tensors
        all_matches = torch.load(pth_path)

        # For each match calculate the 3D IoU, degree error, and offset error 
        for match in tqdm.tqdm(all_matches):

            for class_id in range(len(match)):

                # Catching no-instance scenario
                if 'quaternion' not in match[class_id].keys():
                    continue 

                # Obtaining essential data
                gt_q = match[class_id]['quaternion'][0]
                pred_q = match[class_id]['quaternion'][1]
                gt_RTs = match[class_id]['RT'][0]
                gt_scales = match[class_id]['scales'][0]
                pred_RTs = match[class_id]['RT'][1]
                pred_scales = match[class_id]['scales'][1]

                # Calculating the distance between the quaternions
                degree_distance = lib.gtf.torch_quat_distance(gt_q, pred_q)

                # Calculating the iou 3d for between the ground truth and predicted 
                ious_3d = lib.gtf.get_3d_ious(gt_RTs, pred_RTs, gt_scales, pred_scales)

                # Determing the offset errors
                offset_errors = lib.gtf.from_RTs_get_T_offset_errors(
                    gt_RTs,
                    pred_RTs
                )

                # Store data
                if class_id not in raw_data['degree_error'].keys():
                    raw_data['degree_error'][class_id] = [degree_distance]
                    raw_data['3d_iou'][class_id] = [ious_3d]
                    raw_data['offset_error'][class_id] = [offset_errors]
                else:
                    raw_data['degree_error'][class_id].append(degree_distance)
                    raw_data['3d_iou'][class_id].append(ious_3d)
                    raw_data['offset_error'][class_id].append(offset_errors)

        # After the loop of the matches
        for class_id in range(len(HPARAM.SELECTED_CLASSES)-1): # -1 to remove bg
            raw_data['degree_error'][class_id] = torch.cat(raw_data['degree_error'][class_id])
            raw_data['3d_iou'][class_id] = torch.cat(raw_data['3d_iou'][class_id])
            raw_data['offset_error'][class_id] = torch.cat(raw_data['offset_error'][class_id])

        # Determine the APs values for figure data
        figure_aps = lib.gtf.calculate_aps(
            raw_data,
            figure_metrics_thresholds,
            metrics_operator
        )

        # Plot the figure aps
        fig = tools.vz.plot_aps(
            figure_aps,
            titles={'3d_iou': '3D Iou AP', 'degree_error':'Rotation AP', 'offset_error': 'Translation AP'},
            x_ranges=figure_metrics_thresholds,
            cls_names=HPARAM.SELECTED_CLASSES[1:] + ['mean'],
            x_axis_labels={'3d_iou': '3D IoU %', 'degree_error': 'Rotation error/degree', 'offset_error': 'Translation error/cm'}
        )

        # Saving the plot
        fig.savefig(
            str(PATH.parent.parent / f'all_metrics_{HPARAM.VALID_SIZE}_aps.png')
        )

        # Determine the APs values for table data
        table_aps = lib.gtf.calculate_aps(
            raw_data,
            table_metrics_thresholds,
            metrics_operator
        )

        # Storing the table critical values into an excel sheet
        excel_path = PATH.parent.parent / f'{HPARAM.VALID_SIZE}_aps_values_table.xlsx'
        tools.et.save_aps_to_excel(excel_path, table_metrics_thresholds, table_aps)