import os
import sys

"""
This file is for simple helper functions that are used frequency throughout
the tools folder
"""

#-------------------------------------------------------------------------------
# Functions

def enable_print(DEBUG):
    if not DEBUG:
        sys.stdout = sys.__stdout__ 

def disable_print(DEBUG):
    if not DEBUG:
        sys.stdout = open(os.devnull, 'w')