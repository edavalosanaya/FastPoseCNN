import json

import numpy as np

#-------------------------------------------------------------------------------
# JSON Constants

#-------------------------------------------------------------------------------
# Classes

class NumpyEncoder(json.JSONEncoder):
    """ Special json encoder for numpy types """
    
    def default(self, obj):
        if isinstance(obj, (np.int_, np.intc, np.intp, np.int8,
            np.int16, np.int32, np.int64, np.uint8,
            np.uint16, np.uint32, np.uint64)):
            return int(obj)
        
        elif isinstance(obj, (np.float_, np.float16, np.float32, 
            np.float64)):
            return float(obj)
        
        elif isinstance(obj,(np.ndarray,)): #### This is the fix
            return obj.tolist()
        
        return json.JSONEncoder.default(self, obj)

#-------------------------------------------------------------------------------
# Functions

def save_to_json(file_path, data):

    """
    Saving data into a json file.
    Input: 
        file_path: (the destination file)
        data: the saved data
    Output:
        None
    """

    # catching possible pathlib.Path object
    if isinstance(file_path, str) is False:
        file_path = str(file_path)

    # checking if file_path is a json file
    assert file_path.endswith('.json'), 'Given file_path is invalid for saving into json file'

    with open(file_path, 'w') as outfile:
        json.dump(data, outfile, cls=NumpyEncoder)

def load_from_json(file_path):

    """
    Loading data from a json file.
    Input: 
        file_path: (the source file)
    Output:
        data: the loaded data
    """

    # catching possible pathlib.Path object
    if isinstance(file_path, str) is False:
        file_path = str(file_path)

    # checking if file_path is a json file
    assert file_path.endswith('.json'), 'Given file_path is invalid for saving into json file'

    with open(file_path, 'r') as infile:
        data = json.load(infile)

    return data



    

