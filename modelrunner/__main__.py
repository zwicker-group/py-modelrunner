"""
Main module allowing to use the package to wrap existing code

.. codeauthor:: David Zwicker <david.zwicker@ds.mpg.de>
"""

import importlib.util
import inspect
import logging
import os.path
import sys

from modelrunner import ModelBase, run_function_with_cmd_args

if __name__ == "__main__":
    logger = logging.getLogger("modelrunner")

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
    logger.debug("Search for models in script")
    candidate_instance, candidate_classes, candidate_funcs = {}, {}, {}
    for name, member in inspect.getmembers(model_code):
        if isinstance(member, ModelBase):
            candidate_instance[name] = member
        elif inspect.isclass(member):
            if issubclass(member, ModelBase) and member is not ModelBase:
                candidate_classes[name] = member
        elif inspect.isfunction(member):
            candidate_funcs[name] = member

    if len(candidate_instance) == 1:
        # there is a single instance of a model => use this
        _, obj = candidate_instance.popitem()
        logger.info("Run model instance `%s`", obj.__class__.__name__)
        obj.run_from_command_line(model_args, name=filename)

    elif len(candidate_instance) > 1:
        # there are multiple instance => we do not know which one do use
        names = ", ".join(sorted(candidate_instance.keys()))
        raise RuntimeError(f"Found multiple model instances: {names}")

    elif len(candidate_classes) == 1:
        # there is a single class of a model => use this
        _, cls = candidate_classes.popitem()
        logger.info("Run model class `%s`", cls.__name__)
        cls.run_from_command_line(model_args, name=filename)

    elif len(candidate_classes) > 1:
        # there are multiple instance => we do not know which one do use
        names = ", ".join(sorted(candidate_classes.keys()))
        raise RuntimeError(f"Found multiple model classes: {names}")

    elif len(candidate_funcs) == 1 or "main" in candidate_funcs:
        # there is a single function of a model => use this
        if "main" in candidate_funcs:
            func = candidate_funcs["main"]
            logger.info("Run model function named `main`")
        else:
            _, func = candidate_funcs.popitem()
            logger.info("Run model function named `%s`", func.__name__)
        run_function_with_cmd_args(func, args=sys.argv[2:], name=filename)

    elif len(candidate_funcs) > 1:
        names = ", ".join(sorted(candidate_funcs.keys()))
        raise RuntimeError(f"Found many function, but no 'main' function: {name}")

    else:
        # we could not find any useful objects
        raise RuntimeError("Found neither a model class, instance, or function")
