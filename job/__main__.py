"""
.. codeauthor:: David Zwicker <david.zwicker@ds.mpg.de>
"""

import importlib.util
import inspect
import os.path
import sys

from job import ModelBase, run_function_with_cmd_args

if __name__ == "__main__":
    # get the script name from the command line
    try:
        script_path = sys.argv[1]
    except IndexError:
        print("Require job script as first argument", file=sys.stderr)
        sys.exit(1)

    model_args = sys.argv[2:]

    # load the script as a module
    filename = os.path.basename(script_path)
    spec = importlib.util.spec_from_file_location("model_code", script_path)
    if spec is None:
        print(f"Could not find job script `{script_path}`")
        sys.exit(1)
    model_code = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(model_code)  # type: ignore

    # find all functions in the module
    candidate_funcs, candidate_classes = {}, []
    for name, member in inspect.getmembers(model_code):
        if inspect.isclass(member):
            if issubclass(member, ModelBase) and member is not ModelBase:
                candidate_classes.append(member)
        elif inspect.isfunction(member):
            candidate_funcs[name] = member

    if len(candidate_classes) == 1:
        # use this one class
        model = candidate_classes[0].from_command_line(model_args, name=filename)

    elif len(candidate_classes) > 1:
        # there are multiple model classes in this script
        names = [cls.__name__ for cls in candidate_classes]
        raise RuntimeError("Found multiple model classes: ", names)

    else:  # len(candidate_classes) == 0
        # there are no model classes => look for functions
        if len(candidate_funcs) == 0:
            raise RuntimeError("Found neither a model class nor a suitable function")
        elif len(candidate_funcs) == 1:
            # create the model from the function
            _, func = candidate_funcs.popitem()
        elif "main" in candidate_funcs:
            func = candidate_funcs["main"]
        else:
            funcs = ", ".join(candidate_funcs.keys())
            raise RuntimeError(f"Found many function, but no 'main' function: {funcs}")
        run_function_with_cmd_args(func, args=sys.argv[2:], name=filename)
