"""
.. codeauthor:: David Zwicker <david.zwicker@ds.mpg.de>
"""

import importlib.util
import inspect
import os.path
import sys

from job import run_function_with_cmd_args

if __name__ == "__main__":
    # get the script name from the command line
    try:
        script_path = sys.argv[1]
    except IndexError:
        print("Require job script as first argument", file=sys.stderr)
        sys.exit(1)
    
    # load the script as a module
    filename = os.path.basename(script_path)
    spec = importlib.util.spec_from_file_location("model_code", script_path)
    if spec is None:
        print(f"Could not find job script `{script_path}`")
        sys.exit(1)
    model_code = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(model_code)  # type: ignore

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
            raise RuntimeError("Found many function, but no 'main' function: ", names)

    # create the model from the function
    run_function_with_cmd_args(func, args=sys.argv[2:], name=filename)
