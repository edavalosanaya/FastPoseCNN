# Imports
import os
import sys
import argparse
import pathlib
from pprint import pprint
import tqdm
import pandas as pd

# DEBUGGING
import pdb
#import matplotlib
#matplotlib.use('Agg')

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
import config

#-------------------------------------------------------------------------------
# Constants

#PATH = pathlib.Path(os.getenv("LOGS")) / 'good_saved_runs' / '2_object' / '21-38-XY_REGRESSION_DEBUG-CAMERA-resnet18-imagenet' / '_' / 'checkpoints' / 'last.ckpt'
#PATH = pathlib.Path(os.getenv("LOGS")) / 'good_saved_runs' / 'all_object' / '11-24-010_SYM_TEST-CAMERA-resnet18-imagenet' / '_' / 'checkpoints' / 'last.ckpt'
#PATH = pathlib.Path(os.getenv("LOGS")) / 'debugging_test_runs' / '23-40-LOG_DEBUG-CAMERA-resnet18-imagenet' / '_' / 'checkpoints' / 'n-ckpt_epoch=5.ckpt'
#PATH = pathlib.Path(os.getenv("LOGS")) / '21-02-25' / '19-57-7_56_DEBUG-CAMERA-resnet18-imagenet' / '_' / 'checkpoints' / 'n-ckpt_epoch=5.ckpt'
PATH = pathlib.Path('/home/students/edavalos/GitHub/FastPoseCNN/source_code/FastPoseCNN/logs/21-03-17/18-32-STABLE_BASE2-PoseRegressor-CAMERA-resnet18-imagenet/_/checkpoints/epoch=45-checkpoint_on=0.5219.ckpt')

HPARAM = config.EVALUATING()

COLLECT_DATA = False
DRAW = True
TOTAL_DRAW_IMAGES = 10
APS_NUM_OF_POINTS = 50

#-------------------------------------------------------------------------------
# File Main

if __name__ == '__main__':

    # Parse arguments and replace global variables if needed
    parser = argparse.ArgumentParser(description='Train with PyTorch Lightning framework')
    
    # Automatically adding all the attributes of the HPARAM to the parser
    for attr in dir(HPARAM):
        if '__' in attr or attr[0] == '_': # Private or magic attributes
            continue
        
        parser.add_argument(f'--{attr}', type=type(getattr(HPARAM, attr)), default=getattr(HPARAM, attr))

    # Updating the HPARAMs
    parser.parse_args(namespace=HPARAM)

    # Construct the json path depending on the PATH string
    pth_path = PATH.parent.parent / f'{PATH.stem}_{HPARAM.VALID_SIZE}_results.pth'

    # Constructing folder of images if not existent
    images_path = PATH.parent.parent / 'images'
    if images_path.exists() is False:
        os.mkdir(str(images_path))

    # If not debugging, then make matplotlib use the non-GUI backend to 
    # improve stability and speed, otherwise allow debugging sessions to use 
    # matplotlib figures.
    if not HPARAM.DEBUG:
        import matplotlib
        matplotlib.use('Agg')

    # Getting the intrinsics for the dataset selected
    HPARAM.NUMPY_INTRINSICS = tools.pj.constants.INTRINSICS[HPARAM.DATASET_NAME]

    # Determining if collect model's performance data
    # or visualizing the results of the model's performance
    if COLLECT_DATA:

        model = lib.pose_regressor.MODELS[HPARAM.MODEL].load_from_ckpt(
            PATH,
            HPARAM
        )

        # Put the model into evaluation mode
        #model.to('cuda') # ! Make it work with multiple GPUs
        model.eval()

        # Load the PyTorch Lightning dataset
        datamodule = tools.ds.PoseRegressionDataModule(
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

            # Determine matches between the aggreated ground truth and preds
            gt_pred_matches = lib.mg.batchwise_find_matches(
                outputs['auxilary']['agg_pred'],
                batch['agg_data']
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
                    dpi=400
                )
                pred_fig.savefig(
                    str(images_path / f'{image_counter}_pred.png'), 
                    dpi=400
                )
                poses_fig.savefig(
                    str(images_path / f'{image_counter}_poses.png'), 
                    dpi=400
                )

            # Saving the matched data (if not None )
            if gt_pred_matches:
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

        if all_matches == []:
            print("All matches made were empty!")
            sys.exit(0)

        # For each match calculate the 3D IoU, degree error, and offset error 
        for match in tqdm.tqdm(all_matches):

            # Catching no-instance scenario
            if type(match) == type(None) or 'quaternion' not in match.keys():
                continue

            # Identify all the classes present in the match
            classes = match['class_ids']

            for class_id in torch.unique(classes):

                # Identify the instances of this class
                class_instances = torch.where(classes == class_id)[0]

                # Obtaining essential data
                gt_q = match['quaternion'][0][class_instances]
                pred_q = match['quaternion'][1][class_instances]
                gt_RTs = match['RT'][0][class_instances]
                pred_RTs = match['RT'][1][class_instances]
                gt_scales = match['scales'][0][class_instances]
                pred_scales = match['scales'][1][class_instances]

                # Calculating the distance between the quaternions
                degree_distance = lib.gtf.get_quat_distance(
                    gt_q, 
                    pred_q,
                    match['symmetric_ids'][class_instances]
                )

                # Calculating the iou 3d for between the ground truth and predicted 
                ious_3d = lib.gtf.get_3d_ious(gt_RTs, pred_RTs, gt_scales, pred_scales)

                # Determing the offset errors
                offset_errors = lib.gtf.from_RTs_get_T_offset_errors(
                    gt_RTs,
                    pred_RTs
                )

                # Store data
                if int(class_id) not in raw_data['degree_error'].keys():
                    raw_data['degree_error'][int(class_id)] = [degree_distance]
                    raw_data['3d_iou'][int(class_id)] = [ious_3d]
                    raw_data['offset_error'][int(class_id)] = [offset_errors]
                else:
                    raw_data['degree_error'][int(class_id)].append(degree_distance)
                    raw_data['3d_iou'][int(class_id)].append(ious_3d)
                    raw_data['offset_error'][int(class_id)].append(offset_errors)

        # After the loop of the matches
        for class_id in range(1, len(HPARAM.SELECTED_CLASSES)): # -1 to remove bg
            raw_data['degree_error'][class_id] = torch.cat(raw_data['degree_error'][class_id])
            raw_data['3d_iou'][class_id] = torch.cat(raw_data['3d_iou'][class_id])
            raw_data['offset_error'][class_id] = torch.cat(raw_data['offset_error'][class_id])

        # Creating a list of all the plotted classes
        plot_classes = HPARAM.SELECTED_CLASSES[1:] + ['mean']

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
            cls_names=plot_classes,
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
        tools.et.save_aps_to_excel(excel_path, table_metrics_thresholds, table_aps, plot_classes)