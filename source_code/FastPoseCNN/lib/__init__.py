import os
import sys

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import loss
import gpu_tensor_funcs as gtf
from pose_regressor import PoseRegressor