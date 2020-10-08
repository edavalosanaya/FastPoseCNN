#!/bin/bash

: '
Avaliable parameters (for pl_train.py)
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
python catalyst_train.py 

python pl_train.py --NUM_EPOCHS 100 --NUM_GPUS 4 --DISTRIBUTED_BACKEND ddp --BATCH_SIZE 4
python pl_train.py --NUM_EPOCHS 100 --NUM_GPUS 4 --DISTRIBUTED_BACKEND dp --BATCH_SIZE 4
python pl_train.py --NUM_EPOCHS 100 --NUM_GPUS 1 --BATCH_SIZE 4

python pl_train.py --NUM_EPOCHS 100 --NUM_GPUS 4 --DISTRIBUTED_BACKEND ddp --BATCH_SIZE 1
python pl_train.py --NUM_EPOCHS 100 --NUM_GPUS 4 --DISTRIBUTED_BACKEND dp --BATCH_SIZE 1
python pl_train.py --NUM_EPOCHS 100 --NUM_GPUS 1 --BATCH_SIZE 1
