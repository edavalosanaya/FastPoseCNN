import sys
import os
import pathlib
import importlib.util

init_path = pathlib.Path(__file__).parent

if str(init_path) not in sys.path:
    sys.path.append(str(init_path))

modules = {'B': None}

for module_name, module_variable in modules.items():
    module_abs_path = init_path / (module_name + '.py')

    if init_path == pathlib.Path.cwd():
        continue

    else:
        spec = importlib.util.spec_from_file_location(module_name + '.py', str(module_abs_path))
        module_variable = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module_variable)

        globals()[module_name] = module_variable