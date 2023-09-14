"""
.. codeauthor:: David Zwicker <david.zwicker@ds.mpg.de>
"""

from __future__ import annotations

import codecs
import inspect
import pickle
from collections import defaultdict
from importlib import import_module
from typing import Any, Callable, Dict, Optional, Sequence, Type, Union

PICKLE_PROTOCOL = pickle.HIGHEST_PROTOCOL


import numpy as np

Location = Union[None, str, Sequence[str]]
Attrs = Dict[str, Any]


def encode_binary(obj: Any, *, binary: bool = False) -> Union[str, bytes]:
    obj_bin = pickle.dumps(obj)
    if binary:
        return obj_bin
    else:
        return codecs.encode(obj_bin, "base64").decode()


def decode_binary(obj_str: Union[str, bytes]) -> Any:
    if isinstance(obj_str, str):
        obj_str = codecs.decode(obj_str.encode(), "base64")
    return pickle.loads(obj_str)


def encode_class(cls: Type) -> str:
    if cls is None:
        return "None"
    return cls.__module__ + "." + cls.__qualname__


def decode_class(class_path: Optional[str], *, guess: Optional[Type] = None) -> Type:
    if class_path is None or class_path == "None":
        return None

    # import class from a package
    try:
        module_path, class_name = class_path.rsplit(".", 1)
    except ValueError:
        raise ImportError(f"Cannot import class {class_path}")

    try:
        module = import_module(module_path)
    except ModuleNotFoundError:
        # see whether the class is already defined ...
        if guess is not None and guess.__name__ == class_name:
            return guess  # ... as the `guess`
        elif class_name in globals():
            return globals()[class_name]  # ... in the global context
        else:
            raise ModuleNotFoundError(f"Cannot load `{class_path}`")

    else:
        # load the class from the module
        try:
            return getattr(module, class_name)
        except AttributeError:
            raise ImportError(f"Module {module_path} does not define {class_name}")


class Array(np.ndarray):
    """Numpy array augmented with attributes"""

    def __new__(cls, input_array, attrs: Optional[Attrs] = None):
        obj = np.asarray(input_array).view(cls)
        obj.attrs = {} if attrs is None else attrs
        return obj

    def __array_finalize__(self, obj):
        if obj is None:  # __new__ handles instantiation
            return
        self.attrs = getattr(obj, "attrs", {})


class _StorageRegistry:
    allowed_actions = {
        "read_object",  # read object from storage
    }
    # TODO: also keep information on whether a method needs to be a classmethod or not
    # (but still allow pure functions). In fact, the registry should convert methods
    # and classmethods to callable functions to provide a unified interface

    _classes: Dict[Type, Dict[str, Callable]]

    def __init__(self):
        self._classes = defaultdict(dict)

    def register(
        self,
        action: str,
        cls: Type,
        method_or_func: Callable
        # Optional[Type] = None, func: Optional[Callable] = None
    ):
        """register an action for the given class

        Example:
            The method can either be used directly:

            .. code-block:: python

                storage_actions.register("read_object")

            or as a decorator for the factory function:

            .. code-block:: python

                @storage_actions.register("read_object")
                def _read_object_from_storage():
                    ...

        Args:
            action (str):
                The action provided by the method or function
        """
        if action not in self.allowed_actions:
            raise ValueError(f"Unknown action `{action}` ")

        if isinstance(method_or_func, classmethod):
            # extract class from decorated object
            def _call_classmethod(*args, **kwargs):
                return method_or_func(cls, *args, **kwargs)

            self._classes[cls][action] = _call_classmethod
        elif callable(method_or_func):
            self._classes[cls][action] = method_or_func
        else:
            raise TypeError("`method_or_func` must be method or function")

    def get(self, cls: Type, action: str) -> Callable:
        # look for defined operators on all parent classes (except `object`)
        classes = inspect.getmro(cls)[:-1]
        for c in classes:
            if c in self._classes and action in self._classes[c]:
                return self._classes[c][action]

        raise RuntimeError(f"No action `{action}` for `{cls.__name__}`")


storage_actions = _StorageRegistry()
