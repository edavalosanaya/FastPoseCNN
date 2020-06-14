import sys
import os
import pathlib
import importlib.util

init_dir = pathlib.Path(__file__).parent

if str(init_dir) not in sys.path:
    sys.path.append(str(init_dir))

if init_dir == pathlib.Path.cwd():
    # If __init__.py is already in the cwd, no need to import anything
    pass

else:
    # else import the files within the __init__.py directory
    
    for file_path in [file_path for file_path in init_dir.iterdir() if file_path.with_suffix('.py') or file_path.is_dir()]:

        spec = importlib.util.spec_from_file_location(file_path.stem, str(file_path))
        module_variable = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module_variable)

        globals()[module_name] = module_variable