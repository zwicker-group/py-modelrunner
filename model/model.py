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
        pass

    def get_result(self):
        from .results import ModelResult

        return ModelResult(self, self())

    @classmethod
    def _prepare_argparser(cls, name=None):
        parser = argparse.ArgumentParser(
            prog=name,
            description=cls.description,
            formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        )
        for p in cls.parameters_default:
            p._argparser_add(parser)

        # parser.add_argument(
        #     "-o",
        #     "--output",
        #     help="Output path",
        # )

        return parser

    @classmethod
    def from_command_line(cls, args=None, name=None):
        parser = cls._prepare_argparser(name)
        args = parser.parse_args(args)
        return cls(vars(args))

    @property
    def attributes(self) -> Dict[str, Any]:
        """dict: information about the element state, which does not change in time"""
        return {
            "class": self.__class__.__name__,
            "name": self.name,
            "description": self.description,
            "parameters": self.parameters,
        }

    def serialize_attribute(self, name: str, value) -> str:
        """serialize an attribute into a string

        Args:
            name (str): Name of the attribute
            value: The value of the attribute that needs to be serialized

        Returns:
            str: A string representation from which the `value` can be reconstructed
        """
        if name == "parameters":
            # serialize the individual parameters
            default_parameters = self.get_parameters(
                include_hidden=True, include_deprecated=True, sort=False
            )

            parameters = {}
            for key in self.parameters:
                serializer = json.dumps
                if key in default_parameters:
                    def_param_extra = default_parameters[key].extra
                    if "serializer" in def_param_extra:
                        serializer = def_param_extra["serializer"]
                parameters[key] = serializer(value[key])
            value = parameters

        # serialize the value using JSON
        try:
            return json.dumps(value)
        except TypeError as e:
            msg = f'Cannot serialize "{key}" of "{self.__class__.__name__}"'
            raise TypeError(msg) from e

    @classmethod
    def unserialize_attribute(cls, name: str, value_str: str) -> Any:
        """unserializes the given attribute

        Args:
            name (str): Name of the attribute
            value_str (str): Serialized value of the attribute

        Returns:
            The unserialized value
        """
        # unserialize assuming it is JSON-encoded
        value = json.loads(value_str)

        if name == "parameters":
            # unserialize the individual parameters
            default_parameters = cls.get_parameters(
                include_hidden=True, include_deprecated=True, sort=False
            )

            for key in value:
                unserializer = json.loads
                if key in default_parameters:
                    def_param_extra = default_parameters[key].extra
                    if "unserializer" in def_param_extra:
                        unserializer = def_param_extra["unserializer"]
                value[key] = unserializer(value[key])

        return value


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


def function_model_init(func: Callable, parameters: Dict = None):
    """create model from a function and a dictionary of parameters"""
    model_cls = FunctionModelFactory(func)
    return model_cls(parameters)


def function_model_command_line(func: Callable, args=None, name=None):
    """create model from a function and obtain parameters from command line"""
    model_cls = FunctionModelFactory(func)
    return model_cls.from_command_line(args, name=name)
