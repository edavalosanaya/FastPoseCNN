import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

import numpy as np
import cv2
import matplotlib.pyplot as plt
import skimage.io

import torch
import torch.utils
import torchvision

import segmentation_models_pytorch as smp

import albumentations as albu

import pdb

# Local Imports
import project
import dataset

#-------------------------------------------------------------------------------
# Constants

ENCODER = 'se_resnext50_32x4d'
ENCODER_WEIGHTS = 'imagenet'
ACTIVATION = 'sigmoid' # could be None for logits or 'softmax2d' for multiclass segmentation
DEVICE = 'cuda'

#-------------------------------------------------------------------------------
# Functions

def visualize(**images):
    """PLot images in one row."""
    n = len(images)
    plt.figure(figsize=(16, 5))
    for i, (name, image) in enumerate(images.items()):
        plt.subplot(1, n, i + 1)
        plt.xticks([])
        plt.yticks([])
        plt.title(' '.join(name.split('_')).title())
        plt.imshow(image)
    
    #plt.show()
    plt.savefig('output.png')

def get_training_augmentation():
    train_transform = [

        albu.HorizontalFlip(p=0.5),

        albu.ShiftScaleRotate(scale_limit=0.5, rotate_limit=0, shift_limit=0.1, p=1, border_mode=0),

        albu.PadIfNeeded(min_height=320, min_width=320, always_apply=True, border_mode=0),
        albu.RandomCrop(height=320, width=320, always_apply=True),

        albu.IAAAdditiveGaussianNoise(p=0.2),
        albu.IAAPerspective(p=0.5),

        albu.OneOf(
            [
                albu.CLAHE(p=1),
                albu.RandomBrightness(p=1),
                albu.RandomGamma(p=1),
            ],
            p=0.9,
        ),

        albu.OneOf(
            [
                albu.IAASharpen(p=1),
                albu.Blur(blur_limit=3, p=1),
                albu.MotionBlur(blur_limit=3, p=1),
            ],
            p=0.9,
        ),

        albu.OneOf(
            [
                albu.RandomContrast(p=1),
                albu.HueSaturationValue(p=1),
            ],
            p=0.9,
        ),
    ]
    return albu.Compose(train_transform)

def get_validation_augmentation():
    """Add paddings to make image shape divisible by 32"""
    test_transform = [
        albu.PadIfNeeded(384, 480),
        #albu.PadIfNeeded(320, 320),
        #albu.RandomCrop(height=320, width=320, always_apply=True)
    ]
    return albu.Compose(test_transform)

def to_tensor(x, **kwargs):
    return x.transpose(2, 0, 1).astype('float32')

def get_preprocessing(preprocessing_fn):
    """Construct preprocessing transform
    
    Args:
        preprocessing_fn (callbale): data normalization function 
            (can be specific for each pretrained neural network)
    Return:
        transform: albumentations.Compose
    
    """
    
    _transform = [
        albu.Lambda(image=preprocessing_fn),
        albu.Lambda(image=to_tensor, mask=to_tensor),
    ]
    return albu.Compose(_transform)

#-------------------------------------------------------------------------------
# Classes


#-------------------------------------------------------------------------------

# smp function
preprocessing_fn = smp.encoders.get_preprocessing_fn(ENCODER, ENCODER_WEIGHTS)

"""
# Determing the location of the datasets
x_train_dir = os.path.join(project.cfg.CAMVID_DATSET, 'train')
y_train_dir = os.path.join(project.cfg.CAMVID_DATSET, 'trainannot')

x_valid_dir = os.path.join(project.cfg.CAMVID_DATSET, 'val')
y_valid_dir = os.path.join(project.cfg.CAMVID_DATSET, 'valannot')

x_test_dir = os.path.join(project.cfg.CAMVID_DATSET, 'test')
y_test_dir = os.path.join(project.cfg.CAMVID_DATSET, 'testannot')

CLASSES = project.constants.CAMVID_CLASSES

# Create dataset # CAMVID
train_dataset = Dataset(
    x_train_dir, 
    y_train_dir, 
    augmentation=get_training_augmentation(), 
    preprocessing=get_preprocessing(preprocessing_fn),
    classes=CLASSES,
)

valid_dataset = Dataset(
    x_valid_dir, 
    y_valid_dir, 
    augmentation=get_validation_augmentation(), 
    preprocessing=get_preprocessing(preprocessing_fn),
    classes=CLASSES,
)
#"""

""" # Original VOC dataset class
crop_size = (320, 480)
train_dataset = dataset.VOCSegDataset(True, crop_size, project.cfg.VOC_DATASET)
valid_dataset = dataset.VOCSegDataset(False, crop_size, project.cfg.VOC_DATASET)
CLASSES = project.constants.VOC_CLASSES
#"""

""" # VOC
# Create dataset
CLASSES = project.constants.VOC_CLASSES

train_dataset = VOCDataset(
    project.cfg.VOC_DATASET, 
    is_train=True, 
    augmentation=get_training_augmentation(), 
    preprocessing=get_preprocessing(preprocessing_fn),
    classes=CLASSES,
)

valid_dataset = VOCDataset(
    project.cfg.VOC_DATASET, 
    is_train=False, 
    augmentation=get_validation_augmentation(), 
    preprocessing=get_preprocessing(preprocessing_fn),
    classes=CLASSES,
)
"""

# NOCS

CLASSES = project.constants.SYNSET_NAMES

"""
train_dataset = NOCSDataset(
    project.cfg.CAMERA_TRAIN_DATASET, 
    max_size=5000,
    is_train=True, 
    augmentation=get_training_augmentation(), 
    preprocessing=get_preprocessing(preprocessing_fn),
    classes=CLASSES,
)

valid_dataset = NOCSDataset(
    project.cfg.CAMERA_VALID_DATASET,
    max_size=1000,
    is_train=False, 
    augmentation=get_validation_augmentation(), 
    preprocessing=get_preprocessing(preprocessing_fn),
    classes=CLASSES,
)
"""

test_dataset = NOCSDataset(
    project.cfg.CAMERA_VALID_DATASET,
    max_size=10,
    is_train=False,
    classes=CLASSES,
)

#train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=8, shuffle=True, num_workers=12)
#valid_loader = torch.utils.data.DataLoader(valid_dataset, batch_size=1, shuffle=False, num_workers=4)


model_path = os.path.join(project.cfg.SAVED_MODEL_DIR, './best_model.pth')
model = torch.load(model_path)
model.eval()

for i in range(1):
    n = np.random.choice(len(test_dataset))
    
    image_vis = test_dataset[n][0].astype('uint8')
    image, gt_mask = test_dataset[n]
    
    gt_mask = gt_mask.squeeze()
    
    x_tensor = torch.from_numpy(image).to(DEVICE).permute(2,0,1).unsqueeze(0).float()
    pr_mask = model.predict(x_tensor)
    pr_mask = (pr_mask.squeeze().cpu().permute(1,2,0).numpy().round())

    gt_mask = np.argmax(gt_mask, axis=2)
    pr_mask = np.argmax(pr_mask, axis=2)

    visualize(
        image=image_vis, 
        ground_truth_mask=gt_mask, 
        predicted_mask=pr_mask
    )


"""
# create segmentation model with pretrained encoder
model = smp.FPN(
    encoder_name=ENCODER, 
    encoder_weights=ENCODER_WEIGHTS, 
    classes=len(CLASSES), 
    activation=ACTIVATION,
)

# Loss and Optimizer
loss = smp.utils.losses.DiceLoss()
metrics = [
    smp.utils.metrics.IoU(threshold=0.5),
]

optimizer = torch.optim.Adam([ 
    dict(params=model.parameters(), lr=0.0001),
])

# create epoch runners 
# it is a simple loop of iterating over dataloader`s samples
train_epoch = smp.utils.train.TrainEpoch(
    model, 
    loss=loss, 
    metrics=metrics, 
    optimizer=optimizer,
    device=DEVICE,
    verbose=True,
)

valid_epoch = smp.utils.train.ValidEpoch(
    model, 
    loss=loss, 
    metrics=metrics, 
    device=DEVICE,
    verbose=True,
)

# train model for 40 epochs
epoch = 20
max_score = 0

for i in range(0, epoch):
    
    print('\nEpoch: {}'.format(i))
    train_logs = train_epoch.run(train_loader)
    valid_logs = valid_epoch.run(valid_loader)
    
    # do something (save model, change lr, etc.)
    if max_score < valid_logs['iou_score']:
        max_score = valid_logs['iou_score']
        torch.save(model, os.path.join(project.cfg.SAVED_MODEL_DIR, './best_model.pth'))
        print('Model saved!')
        
    if i == 25:
        optimizer.param_groups[0]['lr'] = 1e-5
        print('Decrease decoder learning rate to 1e-5!')
"""