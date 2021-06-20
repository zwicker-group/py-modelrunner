"""
.. codeauthor:: David Zwicker <david.zwicker@ds.mpg.de>
"""

import importlib.util
import inspect
import os.path
import sys

from model import function_model_command_line

# load the script as a module
script_path = sys.argv[1]
filename = os.path.basename(script_path)
spec = importlib.util.spec_from_file_location("model_code", script_path)
model_code = importlib.util.module_from_spec(spec)
spec.loader.exec_module(model_code)

# find all functions in the module
funcs = inspect.getmembers(model_code, inspect.isfunction)
if len(funcs) == 1:
    func = funcs[0][1]
else:
    for name, func in funcs:
        if name == "main":
            break
    else:
        names = [name for name, _ in funcs]
        raise RuntimeError("Not clear which of these functions needs to be run:", names)

# create the model from the function
mdl = function_model_command_line(func, args=sys.argv[2:], name=filename)
print(mdl())
