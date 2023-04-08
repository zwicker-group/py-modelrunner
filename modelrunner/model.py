"""
Base class describing a model.

.. codeauthor:: David Zwicker <david.zwicker@ds.mpg.de>
"""

import argparse
import functools
import importlib.util
import inspect
import json
import logging
import os.path
from abc import ABCMeta, abstractmethod
from datetime import datetime
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Dict,
    Optional,
    Sequence,
    Type,
    TypeVar,
    Union,
)

from .parameters import (
    DeprecatedParameter,
    HideParameter,
    NoValue,
    Parameter,
    Parameterized,
)
from .state import ObjectState, StateBase

if TYPE_CHECKING:
    from .results import Result  # @UnusedImport


class ModelBase(Parameterized, metaclass=ABCMeta):
    """base class for describing models"""

    name: Optional[str] = None
    """str: the name of the model"""
    description: Optional[str] = None
    """str: a longer description of the model"""
    state_cls: Type[StateBase] = ObjectState
    """type: the class that will contain the model results"""

    def __init__(
        self,
        parameters: Optional[Dict[str, Any]] = None,
        output: Optional[str] = None,
        *,
        strict: bool = False,
    ):
        """initialize the parameters of the object

        Args:
            parameters (dict):
                A dictionary of parameters to change the defaults of this model. The
                allowed parameters can be obtained from
                :meth:`~Parameterized.get_parameters` or displayed by calling
                :meth:`~Parameterized.show_parameters`.
            output (str):
                Path to write the output file
            strict (bool):
                Flag indicating whether parameters are strictly interpreted. If `True`,
                only parameters listed in `parameters_default` can be set and their type
                will be enforced.
        """
        super().__init__(parameters, strict=strict)
        self.output = output
        self._logger = logging.getLogger(self.__class__.__name__)

    @abstractmethod
    def __call__(self):
        """main method calculating the result"""
        pass

    def get_result(self, state: Optional[StateBase] = None) -> "Result":
        """get the result as a :class:`~model.Result` object

        Args:
            state:
                The result data. If omitted, the model is run to obtain results

        Returns:
            :class:`Result`: The result after the model is run
        """
        from .results import Result  # @Reimport

        if state is None:
            state_data = self()
            state = self.state_cls(state_data)
        elif isinstance(state, Result):
            return state

        info = {"time": datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
        return Result(self, state, info=info)

    def write_result(self, output: Optional[str] = None, result=None) -> None:
        """write the result to the output file

        Args:
            output (str):
                File where the output will be written to. If omitted self.output will be
                used. If self.output is also None, an error will be thrown.
            result:
                The result data. If omitted, the model is run to obtain results
        """
        if output is None:
            output = self.output
        if output is None:
            raise RuntimeError("output file needs to be specified")

        if result is None:
            result = self.get_result()
        else:
            from .results import Result  # @Reimport

            assert isinstance(result, Result)
        result.to_file(output)

    @classmethod
    def _prepare_argparser(cls, name: Optional[str] = None) -> argparse.ArgumentParser:
        """create argument parser for setting parameters of this model

        Args:
            name (str):
                Name of the program, which will be shown in the command line help

        Returns:
            :class:`~argparse.ArgumentParser`
        """
        parser = argparse.ArgumentParser(
            prog=name,
            description=cls.description,
            formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        )

        group = parser.add_argument_group()
        seen = set()
        # iterate over all parent classes
        for cls1 in cls.__mro__:
            if hasattr(cls1, "parameters_default"):
                # add all parameters of this class
                for p in cls1.parameters_default:
                    if p.name not in seen:
                        if not isinstance(p, (HideParameter, DeprecatedParameter)):
                            p._argparser_add(group)
                        seen.add(p.name)

        # add special parameters
        parser.add_argument(
            "--json",
            metavar="JSON",
            help="JSON-encoded parameter values. Overwrites other parameters.",
        )

        return parser

    @classmethod
    def from_command_line(
        cls, args: Optional[Sequence[str]] = None, name: Optional[str] = None
    ) -> "ModelBase":
        """create model from command line parameters

        Args:
            args (list):
                Sequence of strings corresponding to the command line arguments
            name (str):
                Name of the program, which will be shown in the command line help

        Returns:
            :class:`ModelBase`: An instance of this model with appropriate parameters
        """
        if args is None:
            args = []

        # read the command line arguments
        parser = cls._prepare_argparser(name)

        # add special parameters to determine the output file
        parser.add_argument(
            "-o",
            "--output",
            metavar="PATH",
            help="Path to output file. If omitted, no output file is created.",
        )

        parameters = vars(parser.parse_args(args))
        output = parameters.pop("output")
        parameters_json = parameters.pop("json")

        # update parameters with data from the json argument
        if parameters_json:
            parameters.update(json.loads(parameters_json))

        # create the model
        return cls(parameters, output=output)

    @classmethod
    def run_from_command_line(
        cls, args: Optional[Sequence[str]] = None, name: Optional[str] = None
    ) -> "Result":
        """run model using command line parameters

        Args:
            args (list):
                Sequence of strings corresponding to the command line arguments
            name (str):
                Name of the program, which will be shown in the command line help

        Returns:
            :class:`Result`: The result of running the model
        """
        # create model from command line parameters
        mdl = cls.from_command_line(args, name)

        # run the model
        result = mdl.get_result()
        if mdl.output:
            # write the results to a file
            mdl.write_result(output=None, result=result)
        else:
            # display the results on stdout
            print(result.state._to_simple_objects()["data"])

        return result

    @property
    def _state_attributes(self) -> Dict[str, Any]:
        """dict: information about the element state, which does not change in time"""
        return {
            "class": self.__class__.__name__,
            "name": self.name,
            "description": self.description,
            "parameters": self.parameters,
        }


_DEFAULT_MODEL: Union[Callable, ModelBase, None] = None
"""stores the default model that will be used automatically"""


TModel = TypeVar("TModel", Callable, ModelBase, None)


def set_default(func_or_model: TModel) -> TModel:
    """sets the function or model as the default model

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
    """run the function with a cleared _DEFAULT_MODEL and restore it afterwards"""

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


def make_model_class(func: Callable, *, default: bool = False) -> Type[ModelBase]:
    """create a model from a function by interpreting its signature

    Args:
        func (callable):
            The function that will be turned into a Model
        default (bool):
            If True, set this model as the default one for the current script

    Returns:
        :class:`ModelBase`: A subclass of ModelBase, which encompasses `func`
    """
    # determine the parameters of the function
    parameters_default = []
    for name, param in inspect.signature(func).parameters.items():
        if param.annotation is param.empty:
            cls = object
        else:
            cls = param.annotation
        if param.default is param.empty:
            default_value = NoValue
        else:
            default_value = param.default

        parameters_default.append(Parameter(name, default_value=default_value, cls=cls))

    def __call__(self, *args, **kwargs):
        """call the function preserving the original signature"""
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
    parameters: Optional[Dict[str, Any]] = None,
    *,
    default: bool = False,
) -> ModelBase:
    """create model from a function and a dictionary of parameters

    Args:
        func (callable):
            The function that will be turned into a Model
        parameters (dict):
            Paramter values with which the model is initialized
        default (bool):
            If True, set this model as the default one for the current script

    Returns:
        :class:`ModelBase`: An instance of a subclass of ModelBase encompassing `func`
    """
    model_class = make_model_class(func, default=default)
    return model_class(parameters)


def run_function_with_cmd_args(
    func: Callable, args: Optional[Sequence[str]] = None, *, name: Optional[str] = None
) -> "Result":
    """create model from a function and obtain parameters from command line

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
def run_script(script_path: str, model_args: Sequence[str]) -> "Result":
    """helper function that runs a model script

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
    global _DEFAULT_MODEL
    logger = logging.getLogger("modelrunner")

    # load the script as a module
    filename = os.path.basename(script_path)
    spec = importlib.util.spec_from_file_location("model_code", script_path)
    if spec is None:
        raise IOError(f"Could not find job script `{script_path}`")
    model_code = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(model_code)  # type: ignore

    # check whether a default model was set
    if (
        isinstance(_DEFAULT_MODEL, ModelBase)
        or inspect.isclass(_DEFAULT_MODEL)
        and issubclass(_DEFAULT_MODEL, ModelBase)
    ):
        logger.info("Run marked default model object")
        return _DEFAULT_MODEL.run_from_command_line(model_args, name=filename)

    elif callable(_DEFAULT_MODEL):
        logger.info("Run marked default model function")
        return run_function_with_cmd_args(
            _DEFAULT_MODEL, args=model_args, name=filename
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
