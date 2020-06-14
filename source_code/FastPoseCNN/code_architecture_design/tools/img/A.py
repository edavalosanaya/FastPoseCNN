import sys
import os
import pathlib

# Local imports
root = next(path for path in pathlib.Path(__file__).parents if path.name == 'my_own_network')
sys.path.append(str(root))

def function():

    print('Function from A.py')

    return None

if __name__ == "__main__":

    print('1')
    import tools
    print('2')

    print('3')
    function()
    tools.constants.B.function()
    print('4')