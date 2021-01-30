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
    DATASET_NAME = 'NOCS' # string
    SELECTED_CLASSES = ['bg','camera','laptop'] #tools.pj.constants.NUM_CLASSES[DATASET_NAME] 

    # Run Specifications
    BATCH_SIZE = 6
    NUM_WORKERS = 18 # 18 # 18 # 36 total CPUs
    NUM_GPUS = 1 # 4 total GPUs
    TRAIN_SIZE= 100#5000
    VALID_SIZE= 20#200

    # Training Specifications
    FREEZE_ENCODER = False
    FREEZE_MASK_DECODER = False
    WEIGHT_DECAY = 0.0003
    LEARNING_RATE = 0.0001
    ENCODER_LEARNING_RATE = 0.0005
    NUM_EPOCHS = 2
    DISTRIBUTED_BACKEND = None if NUM_GPUS <= 1 else 'ddp'

    # Architecture Parameters
    BACKBONE_ARCH = 'FPN'
    ENCODER = 'resnet18' #'resnext50_32x4d'
    ENCODER_WEIGHTS = 'imagenet'

    # Algorithmic Parameters
    
    ## Hough Voting Parameters 
    HV_NUM_OF_HYPOTHESES = 51 # Good at 50 though (preferably 2*n + 1 because of iqr)
    HV_HYPOTHESIS_IN_MASK_MULTIPLIER = 3 
    PRUN_METHOD = 'iqr' # options = (None, 'z-score', 'iqr')
    PRUN_OUTLIER_DROP = False
    PRUN_OUTLIER_REPLACEMENT_STYLE = 'median'

    ### Pruning Method Parameters (Z-score)
    PRUN_ZSCORE_THRESHOLD=1

    ### Pruning Method Parameters (IQR)
    IQR_MULTIPLIER=1.5