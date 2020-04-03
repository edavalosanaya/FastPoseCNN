#__init__.py
import os
import sys
import importlib

sys.path.append(os.path.dirname(os.path.realpath(__file__)))
from ModelNOCS import NOCS, InferenceConfig
from ModelDenseDepth import DenseDepth

