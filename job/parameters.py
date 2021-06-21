"""
Infrastructure for managing classes with parameters

One aim is to allow easy management of inheritance of parameters.

.. autosummary::
   :nosignatures:

   Parameter
   DeprecatedParameter
   HideParameter
   Parameterized
   
.. codeauthor:: David Zwicker <david.zwicker@ds.mpg.de>
"""

import importlib
import logging
from collections import OrderedDict
from typing import Any, Dict, Sequence, Union

import numpy as np


def import_class(identifier: str):
    """import a class or module given an identifier

    Args:
        identifier (str):
            The identifier can be a module or a class. For instance, calling the
            function with the string `identifier == 'numpy.linalg.norm'` is
            roughly equivalent to running `from numpy.linalg import norm` and
            would return a reference to `norm`.
    """
    module_path, _, class_name = identifier.rpartition(".")
    if module_path:
        module = importlib.import_module(module_path)
        return getattr(module, class_name)
    else:
        # this happens when identifier does not contain a dot
        return importlib.import_module(class_name)


class Parameter:
    """class representing a single parameter"""

    def __init__(
        self,
        name: str,
        default_value=None,
        cls=object,
        description: str = "",
        allow: Sequence = None,
        hidden: bool = False,
        extra: Dict[str, Any] = None,
    ):
        """initialize a parameter

        Args:
            name (str):
                The name of the parameter
            default_value:
                The default value
            cls:
                The type of the parameter, which is used for conversion
            description (str):
                A string describing the impact of this parameter. This
                description appears in the parameter help
            allow (sequence):
                Allowed values of this parameter
            hidden (bool):
                Whether the parameter is hidden in the description summary
            extra (dict):
                Extra arguments that are stored with the parameter
        """
        self.name = name
        self.default_value = default_value
        self.cls = cls
        self.description = description
        self.hidden = hidden
        self.extra = {} if extra is None else extra

        if cls is not object:
            # check whether the default value is of the correct type
            converted_value = cls(default_value)
            if isinstance(converted_value, np.ndarray):
                valid_default = np.allclose(converted_value, default_value)
            else:
                valid_default = converted_value == default_value
            if not valid_default:
                logging.warning(
                    "Default value `%s` does not seem to be of type `%s`",
                    name,
                    cls.__name__,
                )

    def __repr__(self):
        return (
            f'{self.__class__.__name__}(name="{self.name}", default_value='
            f"{self.default_value}, cls={self.cls.__name__}, description="
            f'"{self.description}", hidden={self.hidden})'
        )

    __str__ = __repr__

    def __getstate__(self):
        # replace the object class by its class path
        return {
            "name": str(self.name),
            "default_value": self.convert(),
            "cls": object.__module__ + "." + self.cls.__name__,
            "description": self.description,
            "hidden": self.hidden,
            "extra": self.extra,
        }

    def __setstate__(self, state):
        # restore the object from the class path
        state["cls"] = import_class(state["cls"])
        # restore the state
        self.__dict__.update(state)

    def convert(self, value=None):
        """converts a `value` into the correct type for this parameter. If
        `value` is not given, the default value is converted.

        Note that this does not make a copy of the values, which could lead to
        unexpected effects where the default value is changed by an instance.

        Args:
            value: The value to convert

        Returns:
            The converted value, which is of type `self.cls`
        """
        if value is None:
            value = self.default_value

        if self.cls is object:
            try:
                return float(value)
            except ValueError:
                return value
        else:
            try:
                return self.cls(value)
            except ValueError:
                raise ValueError(
                    f"Could not convert {value!r} to {self.cls.__name__} for parameter "
                    f"'{self.name}'"
                )

    def _argparser_add(self, parser):
        if not self.hidden:

            if self.description:
                description = self.description
            else:
                description = f"Parameter `{self.name}`"

            if self.cls is object:
                parser.add_argument(
                    "--" + self.name,
                    default=self.default_value,
                    help=description,
                )

            else:
                parser.add_argument(
                    "--" + self.name,
                    default=self.default_value,
                    type=self.cls,
                    help=description,
                )


class DeprecatedParameter(Parameter):
    """a parameter that can still be used normally but is deprecated"""

    pass


class HideParameter:
    """a helper class that allows hiding parameters of the parent classes"""

    def __init__(self, name: str):
        """
        Args:
            name (str):
                The name of the parameter
        """
        self.name = name


ParameterListType = Sequence[Union[Parameter, HideParameter]]


class Parameterized:
    """a mixin that manages the parameters of a class"""

    parameters_default: ParameterListType = []

    def __init__(self, parameters: Dict[str, Any] = None):
        """initialize the parameters of the object

        Args:
            parameters (dict):
                A dictionary of parameters to change the defaults. The allowed
                parameters can be obtained from
                :meth:`~Parameterized.get_parameters` or displayed by calling
                :meth:`~Parameterized.show_parameters`.
        """
        # set logger if this has not happened, yet
        if not hasattr(self, "_logger"):
            self._logger = logging.getLogger(self.__class__.__name__)

        # set parameters if they have not been initialized, yet
        if not hasattr(self, "parameters"):
            self.parameters = self._parse_parameters(
                parameters, include_deprecated=True
            )

    def __init_subclass__(cls, **kwargs):  # @NoSelf
        """register all subclasses to reconstruct them later"""
        # normalize the parameters_default attribute
        if hasattr(cls, "parameters_default") and isinstance(
            cls.parameters_default, dict
        ):
            # default parameters are given as a dictionary
            cls.parameters_default = [
                Parameter(*args) for args in cls.parameters_default.items()
            ]

    @classmethod
    def get_parameters(
        cls,
        include_hidden: bool = False,
        include_deprecated: bool = False,
        sort: bool = True,
    ) -> Dict[str, Parameter]:
        """return a dictionary of parameters that the class supports

        Args:
            include_hidden (bool): Include hidden parameters
            include_deprecated (bool): Include deprecated parameters
            sort (bool): Return ordered dictionary with sorted keys

        Returns:
            dict: a dictionary of instance of :class:`Parameter` with their
            names as keys.
        """
        # collect the parameters from the class hierarchy
        parameters: Dict[str, Parameter] = {}
        for cls in reversed(cls.__mro__):
            if hasattr(cls, "parameters_default"):
                for p in cls.parameters_default:
                    if isinstance(p, HideParameter):
                        if include_hidden:
                            parameters[p.name].hidden = True
                        else:
                            del parameters[p.name]

                    else:
                        parameters[p.name] = p

        # filter parameters based on hidden and deprecated flags
        def show(p):
            """helper function to decide whether parameter will be shown"""
            # show based on hidden flag?
            show1 = include_hidden or not p.hidden
            # show based on deprecated flag?
            show2 = include_deprecated or not isinstance(p, DeprecatedParameter)
            return show1 and show2

        # filter parameters based on `show`
        result = {
            name: parameter for name, parameter in parameters.items() if show(parameter)
        }

        if sort:
            result = OrderedDict(sorted(result.items()))
        return result

    @classmethod
    def _parse_parameters(
        cls,
        parameters: Dict[str, Any] = None,
        check_validity: bool = True,
        allow_hidden: bool = True,
        include_deprecated: bool = False,
    ) -> Dict[str, Any]:
        """parse parameters

        Args:
            parameters (dict):
                A dictionary of parameters that will be parsed.
            check_validity (bool):
                Determines whether a `ValueError` is raised if there are keys in
                parameters that are not in the defaults. If `False`, additional
                items are simply stored in `self.parameters`
            allow_hidden (bool):
                Allow setting hidden parameters
            include_deprecated (bool):
                Include deprecated parameters
        """
        if parameters is None:
            parameters = {}
        else:
            parameters = parameters.copy()  # do not modify the original

        # obtain all possible parameters
        param_objs = cls.get_parameters(
            include_hidden=allow_hidden, include_deprecated=include_deprecated
        )

        # initialize parameters with default ones from all parent classes
        result: Dict[str, Any] = {}
        for name, param_obj in param_objs.items():
            if not allow_hidden and param_obj.hidden:
                continue  # skip hidden parameters
            # take value from parameters or set default value
            result[name] = param_obj.convert(parameters.pop(name, None))

        # update parameters with the supplied ones
        if check_validity and parameters:
            raise ValueError(
                f"Parameters `{sorted(parameters.keys())}` were provided for an "
                f"instance but are not defined for the class `{cls.__name__}`"
            )
        else:
            result.update(parameters)  # add remaining parameters

        return result

    def get_parameter_default(self, name):
        """return the default value for the parameter with `name`

        Args:
            name (str): The parameter name
        """
        for cls in self.__class__.__mro__:
            if hasattr(cls, "parameters_default"):
                for p in cls.parameters_default:
                    if isinstance(p, Parameter) and p.name == name:
                        return p.default_value

        raise KeyError(f"Parameter `{name}` is not defined")