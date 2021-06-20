"""
.. codeauthor:: David Zwicker <david.zwicker@ds.mpg.de>
"""

import argparse
import inspect
import json
from abc import ABCMeta, abstractmethod
from typing import Any, Callable, Dict, Type

from .parameters import Parameter, Parameterized


class ModelBase(Parameterized, metaclass=ABCMeta):
    """base class for describing models"""

    name: str = None
    description: str = None

    @abstractmethod
    def __call__(self):
        """main method calculating the result"""
        pass

    def get_result(self):
        """get the result as a :class:`~model.Result` object"""
        from .results import Result

        return Result(self, self())

    @classmethod
    def _prepare_argparser(cls, name=None):
        parser = argparse.ArgumentParser(
            prog=name,
            description=cls.description,
            formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        )

        # add all model parameters
        for p in cls.parameters_default:
            p._argparser_add(parser)

        # add special parameters
        parser.add_argument(
            "-o",
            "--output",
            metavar="PATH",
            help="Path to output file. If omitted, no output file is created.",
        )
        # add special parameters
        parser.add_argument(
            "--json",
            metavar="JSON",
            help="JSON-encoded parameter values. Overwrites other parameters.",
        )

        return parser

    @classmethod
    def from_command_line(cls, args=None, name=None):
        """create model from command line parameters"""
        # read the command line arguments
        parser = cls._prepare_argparser(name)
        args = vars(parser.parse_args(args))
        output = args.pop("output")
        parameters_json = args.pop("json")

        # build parameter list
        parameters = args
        if parameters_json:
            parameters.update(json.loads(parameters_json))

        # create the model and run it
        mdl = cls(parameters)
        result = mdl.get_result()

        # store the result if requested
        if output:
            result.write_to_file(output)
        return result

    @property
    def attributes(self) -> Dict[str, Any]:
        """dict: information about the element state, which does not change in time"""
        return {
            "class": self.__class__.__name__,
            "name": self.name,
            "description": self.description,
            "parameters": self.parameters,
        }


def FunctionModelFactory(func: Callable) -> Type[ModelBase]:
    """create a model from a function by interpreting its signature"""
    # determine the parameters of the function
    parameters = []
    for name, param in inspect.signature(func).parameters.items():
        if param.default is inspect.Parameter.empty:
            default_value = None
        else:
            default_value = param.default
        if param.annotation is inspect.Parameter.empty:
            cls = object
        else:
            cls = param.annotation
        parameters.append(Parameter(name, default_value=default_value, cls=cls))

    def __call__(self):
        return func(**self.parameters)

    args = {
        "name": func.__name__,
        "description": func.__doc__,
        "parameters_default": parameters,
        "__call__": __call__,
    }
    newclass = type(func.__name__, (ModelBase,), args)
    return newclass


def get_function_model(func: Callable, parameters: Dict = None):
    """create model from a function and a dictionary of parameters"""
    model_cls = FunctionModelFactory(func)
    return model_cls(parameters)


def run_function_with_cmd_args(func: Callable, args=None, name=None):
    """create model from a function and obtain parameters from command line"""
    model_cls = FunctionModelFactory(func)
    return model_cls.from_command_line(args, name=name)
