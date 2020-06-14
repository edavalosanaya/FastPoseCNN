import numpy as np

intrinsics = np.array([[577.5, 0, 319.5], [0., 577.5, 239.5], [0., 0., 1.]]) # CAMERA intrinsics

synset_names = ['BG', #0
                'bottle', #1
                'bowl', #2
                'camera', #3
                'can',  #4
                'laptop',#5
                'mug'#6
                ]

class_map = {
    'bottle': 'bottle',
    'bowl':'bowl',
    'cup':'mug',
    'laptop': 'laptop',
}