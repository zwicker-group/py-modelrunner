"""Base class describing a model.

.. codeauthor:: David Zwicker <david.zwicker@ds.mpg.de>
"""

from __future__ import annotations

import importlib.util
import inspect
import logging
import os.path
from pathlib import Path
from typing import Callable, Sequence

from ..model import ModelBase, cleared_default_model, factory, make_model_class
from .results import Result


def run_function_with_cmd_args(
    func: Callable, args: Sequence[str] | None = None, *, name: str | None = None
) -> Result:
    """Create model from a function and obtain parameters from command line.

    Args:
        func (callable):
            The function that will be turned into a Model
        args (list of str):
            Command line arguments, typically :code:`sys.argv[1:]`
        name (str):
            Name of the program, which will be shown in the command line help

    Returns:
        :class:`ModelBase`: An instance of a subclass of ModelBase encompassing `func`
    """
    return make_model_class(func).run_from_command_line(args, name=name)


@cleared_default_model
def run_script(script_path: str | Path, model_args: Sequence[str]) -> Result:
    """Helper function that runs a model script.

    The function detects models automatically by trying several methods until one yields
    a unique model to run:

    * A model that have been marked as default by :func:`set_default`
    * A function named `main`
    * A model instance if there is exactly one (throw error if there are many)
    * A model class if there is exactly one (throw error if there are many)
    * A function if there is exactly one (throw error if there are many)

    Args:
        script_path (str):
            Path to the script that contains the model definition
        model_args (sequence):
            Additional arugments that define how the model is run

    Returns:
        :class:`~modelrunner.result.Result`: The result of the run
    """
    logger = logging.getLogger("modelrunner")

    # load the script as a module
    filename = Path(script_path).name
    spec = importlib.util.spec_from_file_location("model_code", script_path)
    if spec is None:
        raise OSError(f"Could not find job script `{script_path}`")
    model_code = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(model_code)  # type: ignore

    # check whether a default model was set
    if (
        isinstance(factory._DEFAULT_MODEL, ModelBase)
        or inspect.isclass(factory._DEFAULT_MODEL)
        and issubclass(factory._DEFAULT_MODEL, ModelBase)
    ):
        logger.info("Run marked default model object")
        return factory._DEFAULT_MODEL.run_from_command_line(model_args, name=filename)

    elif callable(factory._DEFAULT_MODEL):
        logger.info("Run marked default model function")
        return run_function_with_cmd_args(
            factory._DEFAULT_MODEL, args=model_args, name=filename
        )

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

    # run `main` function if there is one
    if "main" in candidate_funcs:
        func = candidate_funcs["main"]
        return run_function_with_cmd_args(func, args=model_args, name=filename)

    # search for instances, classes, and functions and run them if choice is unique
    if len(candidate_instance) == 1:
        # there is a single instance of a model => use this
        _, obj = candidate_instance.popitem()
        logger.info("Run model instance `%s`", obj.__class__.__name__)
        return obj.run_from_command_line(model_args, name=filename)

    elif len(candidate_instance) > 1:
        # there are multiple instance => we do not know which one do use
        names = ", ".join(sorted(candidate_instance.keys()))
        raise RuntimeError(f"Found multiple model instances: {names}")

    elif len(candidate_classes) == 1:
        # there is a single class of a model => use this
        _, cls = candidate_classes.popitem()
        logger.info("Run model class `%s`", cls.__name__)
        return cls.run_from_command_line(model_args, name=filename)

    elif len(candidate_classes) > 1:
        # there are multiple instance => we do not know which one do use
        names = ", ".join(sorted(candidate_classes.keys()))
        raise RuntimeError(f"Found multiple model classes: {names}")

    elif len(candidate_funcs) == 1:
        # there is a single function of a model => use this
        _, func = candidate_funcs.popitem()
        logger.info("Run model function named `%s`", func.__name__)
        return run_function_with_cmd_args(func, args=model_args, name=filename)

    elif len(candidate_funcs) > 1:
        # there are multiple functions and we do not know which one to run
        names = ", ".join(sorted(candidate_funcs.keys()))
        raise RuntimeError(f"Found many functions, but no 'main' function: {names}")

    else:
        # we could not find any useful objects
        raise RuntimeError("Found neither a model class, instance, or function")
