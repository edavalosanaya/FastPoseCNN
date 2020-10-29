import albumentations as albu
from albumentations.pytorch import ToTensor

import general as g

#-------------------------------------------------------------------------------
# Pre-processing

def get_preprocessing(preprocessing_fn):

    preprocessing_transform = albu.Compose([
        albu.Lambda(image=g.to_tensor)],
    )

    return preprocessing_transform

#-------------------------------------------------------------------------------
# Training

def get_training_augmentation():
    train_transform = albu.Compose([

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
    ],
    additional_targets = {'depth': 'mask'}
    )

    return train_transform

#-------------------------------------------------------------------------------
# Validation

def get_validation_augmentation():
    valid_transform = [
        albu.IAAAdditiveGaussianNoise(p=0.2),
    ]
    return albu.Compose(valid_transform)