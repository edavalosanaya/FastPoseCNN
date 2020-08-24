import os
import sys
import pathlib

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from fastposecnn import FastPoseCNN
from unet import unet
import loss_functions as loss