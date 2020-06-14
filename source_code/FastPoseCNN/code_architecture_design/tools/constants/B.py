import sys
import os
import pathlib

# Local imports
root = next(path for path in pathlib.Path(__file__).parents if path.name == 'my_own_network')
sys.path.append(str(root))

import tools

def function():

    print('Function from B.py')

    return None

if __name__ == "__main__":

    pass