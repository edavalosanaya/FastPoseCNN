#!/bin/bash

: '
Avaliable parameters (for pl_segmentation_task.py)
    --DATASET_NAME (-d)
    --BATCH_SIZE (-b)
    --NUM_WORKERS (-nw)
    --NUM_GPUS (-ng)
    --NUM_EPOCHS (-e)
    --DISTRIBUTED_BACKEND (-db)
    --LEARNING_RATE (-lr)
    --ENCODER_LEARNING_RATE (-elr)
    --ENCODER (-enc)
    --ENCODER_WEIGHTS (-ew)
'

# Killing all tensoboards
pkill -9 tensorboard

# This file is to run training jobs
#python catalyst_train.py 

python pl_segmentation_task.py --NUM_EPOCHS 100 --NUM_GPUS 4 --DISTRIBUTED_BACKEND ddp --BATCH_SIZE 4
python pl_segmentation_task.py --NUM_EPOCHS 100 --NUM_GPUS 4 --DISTRIBUTED_BACKEND ddp --BATCH_SIZE 8
python pl_segmentation_task.py --NUM_EPOCHS 100 --NUM_GPUS 4 --DISTRIBUTED_BACKEND ddp --BATCH_SIZE 12
python pl_segmentation_task.py --NUM_EPOCHS 100 --NUM_GPUS 4 --DISTRIBUTED_BACKEND ddp --BATCH_SIZE 16
python pl_segmentation_task.py --NUM_EPOCHS 100 --NUM_GPUS 4 --DISTRIBUTED_BACKEND ddp --BATCH_SIZE 20

