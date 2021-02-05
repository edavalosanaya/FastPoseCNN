import os
import argparse
import pathlib

# Local imports
import tools

#-------------------------------------------------------------------------------
# Configurations

# Run hyperparameters
class DEFAULT_POSE_HPARAM(argparse.Namespace):
    
    # Experiment Identification 
    EXPERIMENT_NAME = "TESTING" # string
    CHECKPOINT = pathlib.Path(os.getenv("LOGS")) / 'good_saved_runs' / '12-49-SMALL_RUN-NOCS-resnet18-imagenet' / '_' / 'checkpoints' / 'last.ckpt' # pathlib
    DATASET_NAME = 'CAMERA' # string
    SELECTED_CLASSES = ['bg','camera','laptop'] #tools.pj.constants.NUM_CLASSES[DATASET_NAME] 

    # Run Specifications
    BATCH_SIZE = 5
    NUM_WORKERS = 0#9 # 18 # 18 # 36 total CPUs
    NUM_GPUS = 0#1# 1 # 4 total GPUs
    TRAIN_SIZE= 50#5000#100
    VALID_SIZE= 20#300#20

    # Training Specifications
    WEIGHT_DECAY = 0.0003
    LEARNING_RATE = 0.0001
    ENCODER_LEARNING_RATE = 0.0005
    NUM_EPOCHS = 2#50#2#50
    DISTRIBUTED_BACKEND = None if NUM_GPUS <= 1 else 'ddp'

    # Freezing Training Specifications
    FREEZE_ENCODER = False
    FREEZE_MASK_TRAINING = False
    FREEZE_ROTATION_TRAINING = False
    FREEZE_TRANSLATION_TRAINING = False
    FREEZE_SCALES_TRAINING = False

    # Algorithmic Training Specifications
    PERFORM_AGGREGATION = True
    PERFORM_HOUGH_VOTING = True
    PERFORM_RT_CALCULATION = True
    PERFORM_MATCHING = True

    # Architecture Parameters
    BACKBONE_ARCH = 'FPN'
    ENCODER = 'resnet18' #'resnext50_32x4d'
    ENCODER_WEIGHTS = 'imagenet'

    # Algorithmic Parameters
    
    ## Hough Voting Parameters 
    HV_NUM_OF_HYPOTHESES = 51 # Good at 50 though (preferably 2*n + 1 because of iqr)
    HV_HYPOTHESIS_IN_MASK_MULTIPLIER = 3 
    
    ### Pruning Parameters
    PRUN_METHOD = 'iqr' # options = (None, 'z-score', 'iqr')
    PRUN_OUTLIER_DROP = False
    PRUN_OUTLIER_REPLACEMENT_STYLE = 'median'

    ### Pruning Method Parameters (Z-score)
    PRUN_ZSCORE_THRESHOLD=1

    ### Pruning Method Parameters (IQR)
    IQR_MULTIPLIER=1.5