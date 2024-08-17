"""Functions for creating models and model classes from other input.

.. codeauthor:: David Zwicker <david.zwicker@ds.mpg.de>
"""

from __future__ import annotations

import functools
import inspect
from typing import TYPE_CHECKING, Any, Callable, Literal, TypeVar, get_args, get_origin

from ..storage import ModeType
from .base import ModelBase
from .parameters import NoValue, Parameter

if TYPE_CHECKING:
    from ..run.results import Result

_DEFAULT_MODEL: Callable | ModelBase | None = None
"""Stores the default model that will be used automatically."""


TModel = TypeVar("TModel", Callable, ModelBase, None)


def set_default(func_or_model: TModel) -> TModel:
    """Sets the function or model as the default model.

    The last model that received this flag will be run automatically. This only affects
    the behavior when the script is run using `modelrunner` from the command line, e.g.,
    using :code:`python -m modelrunner script.py`.

    Args:
        func_or_model (callabel or :class:`ModelBase`, optional):
            The function or model that should be called when the script is run.

    Returns:
        `func_or_model`, so the function can be used as a decorator
    """
    global _DEFAULT_MODEL
    _DEFAULT_MODEL = func_or_model
    return func_or_model


TFunc = TypeVar("TFunc", bound=Any)


def cleared_default_model(func: TFunc) -> TFunc:
    """Run the function with a cleared _DEFAULT_MODEL and restore it afterwards."""

    @functools.wraps(func)
    def inner(*args, **kwargs):
        global _DEFAULT_MODEL
        _old_model = _DEFAULT_MODEL
        _DEFAULT_MODEL = None
        try:
            return func(*args, **kwargs)
        finally:
            _DEFAULT_MODEL = _old_model

    return inner  # type: ignore


def make_model_class(func: Callable, *, default: bool = False) -> type[ModelBase]:
    """Create a model from a function by interpreting its signature.

    Args:
        func (callable):
            The function that will be turned into a Model
        default (bool):
            If True, set this model as the default one for the current script

    Returns:
        :class:`ModelBase`: A subclass of ModelBase, which encompasses `func`
    """
    # determine the parameters of the function
    provide_storage = False
    parameters_default = []
    for name, param in inspect.signature(func).parameters.items():
        if name == "storage":
            # treat this parameter specially and provide access to a storage object
            provide_storage = True
        else:
            # all remaining parameters are treated as model parameters
            if param.annotation is param.empty:
                cls = object
                choices = None
            elif get_origin(param.annotation) is Literal:
                cls = object
                choices = get_args(param.annotation)
            else:
                cls = param.annotation
                choices = None
            if param.default is param.empty:
                default_value = NoValue
            else:
                default_value = param.default

            parameter = Parameter(
                name, default_value=default_value, cls=cls, choices=choices
            )
            parameters_default.append(parameter)

    def __call__(self, *args, **kwargs):
        """Call the function preserving the original signature."""
        parameters = {}
        for i, (name, value) in enumerate(self.parameters.items()):
            if len(args) > i:
                if name in kwargs:
                    raise ValueError(f"{name} also given as positional argument")
                param_value = args[i]
            elif name in kwargs:
                param_value = kwargs[name]
            else:
                param_value = value

            if param_value is NoValue:
                raise TypeError(f"Model missing required argument: '{name}'")
            parameters[name] = param_value

        if provide_storage:
            if "storage" in self.storage:
                self._logger.info("Open storage group `storage`")
                parameters["storage"] = self.storage.open_group("storage")
            else:
                self._logger.info("Create storage group `storage`")
                parameters["storage"] = self.storage.create_group("storage")

        return func(**parameters)

    args = {
        "name": func.__name__,
        "description": func.__doc__,
        "parameters_default": parameters_default,
        "__doc__": func.__doc__,
        "__call__": __call__,
    }
    newclass = type(func.__name__, (ModelBase,), args)
    if default:
        set_default(newclass)
    return newclass


def make_model(
    func: Callable,
    parameters: dict[str, Any] | None = None,
    output: str | None = None,
    *,
    mode: ModeType = "insert",
    default: bool = False,
) -> ModelBase:
    """Create model from a function and a dictionary of parameters.

    Args:
        func (callable):
            The function that will be turned into a Model
        parameters (dict):
            Paramter values with which the model is initialized
        output (str):
            Path where the output file will be written.
        mode (str or :class:`~modelrunner.storage.access_modes.ModeType`):
            The file mode with which the storage is accessed, which determines the
            allowed operations. Common options are "read", "full", "append", and
            "truncate".
        default (bool):
            If True, set this model as the default one for the current script

    Returns:
        :class:`ModelBase`: An instance of a subclass of ModelBase encompassing `func`
    """
    model_class = make_model_class(func, default=default)
    return model_class(parameters, output=output, mode=mode)
