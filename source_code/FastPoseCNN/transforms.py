import albumentations as albu
from albumentations.pytorch import ToTensor

#-------------------------------------------------------------------------------
# Pre-processing

def to_tensor(x, **kwargs):
    return x.transpose(2,0,1) if len(x.shape) == 3 else x

def get_preprocessing(preprocessing_fn):

    preprocessing_transform = albu.Compose([
        albu.Lambda(image=preprocessing_fn),
        albu.Lambda(image=to_tensor, mask=to_tensor)],
        additional_targets = {'depth': 'mask'}
    )

    return preprocessing_transform
    
#-------------------------------------------------------------------------------
# Train

def get_training_augmentation(height=320, width=320):
    train_transform = albu.Compose([

        albu.HorizontalFlip(p=0.5),

        albu.ShiftScaleRotate(scale_limit=0.5, rotate_limit=0, shift_limit=0.1, p=1, border_mode=0),

        albu.PadIfNeeded(min_height=height, min_width=width, always_apply=True, border_mode=0),
        albu.RandomCrop(height=height, width=width, always_apply=True),

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

def get_validation_augmentation(height=384, width=480):
    valid_transform = [
        albu.PadIfNeeded(height, width)
        #albu.PadIfNeeded(320, 320),
        #albu.RandomCrop(height=320, width=320, always_apply=True)
    ]
    return albu.Compose(valid_transform)

#-------------------------------------------------------------------------------
# Carvana Transformations

def pre_transforms(image_size=224):
    return [albu.Resize(image_size, image_size, p=1)]


def hard_transforms():
    result = [
      albu.RandomRotate90(),
      albu.Cutout(),
      albu.RandomBrightnessContrast(
          brightness_limit=0.2, contrast_limit=0.2, p=0.3
      ),
      albu.GridDistortion(p=0.3),
      albu.HueSaturationValue(p=0.3)
    ]

    return result
  

def resize_transforms(image_size=224):
    BORDER_CONSTANT = 0
    pre_size = int(image_size * 1.5)

    random_crop = albu.Compose([
      albu.SmallestMaxSize(pre_size, p=1),
      albu.RandomCrop(
          image_size, image_size, p=1
      )

    ])

    rescale = albu.Compose([albu.Resize(image_size, image_size, p=1)])

    random_crop_big = albu.Compose([
      albu.LongestMaxSize(pre_size, p=1),
      albu.RandomCrop(
          image_size, image_size, p=1
      )

    ])

    # Converts the image to a square of size image_size x image_size
    result = [
      albu.OneOf([
          random_crop,
          rescale,
          random_crop_big
      ], p=1)
    ]

    return result
  
def post_transforms():
    # we use ImageNet image normalization
    # and convert it to torch.Tensor
    return [albu.Normalize(), ToTensor()]
  
def compose(transforms_to_compose):
    # combine all augmentations into single pipeline
    result = albu.Compose([
      item for sublist in transforms_to_compose for item in sublist
    ])
    return result

train_transforms = compose([
    resize_transforms(), 
    hard_transforms(), 
    post_transforms()
])

valid_transforms = compose([pre_transforms(), post_transforms()])

show_transforms = compose([resize_transforms(), hard_transforms()])