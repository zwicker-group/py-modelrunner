"""
Main module allowing to use the package to wrap existing code

.. codeauthor:: David Zwicker <david.zwicker@ds.mpg.de>
"""

import sys

from modelrunner import Result, run_script


def run_script_from_command_line() -> Result:
    """helper function that runs a model from flags specified at the command line"""
    # get the script name from the command line
    try:
        script_path = sys.argv[1]
    except IndexError:
        print("Require job script as first argument", file=sys.stderr)
        sys.exit(1)

    return run_script(script_path, sys.argv[2:])


<<<<<<< Upstream, based on main
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
=======
if __name__ == "__main__":
    run_script_from_command_line()
>>>>>>> 60bd818 Added many tests and adjusted code
