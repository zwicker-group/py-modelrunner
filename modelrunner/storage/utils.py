"""Functions and classes that are used commonly used by the storage classes.

.. codeauthor:: David Zwicker <david.zwicker@ds.mpg.de>
"""

from __future__ import annotations

import codecs
import inspect
import pickle
from collections import defaultdict
from importlib import import_module
from typing import TYPE_CHECKING, Any, Callable, Literal, Sequence, Union, overload

import numpy as np

if TYPE_CHECKING:
    from .attributes import Attrs

PICKLE_PROTOCOL = pickle.HIGHEST_PROTOCOL


Location = Union[None, str, Sequence["Location"]]


@overload
def encode_binary(obj: Any, *, binary: Literal[True]) -> bytes: ...


@overload
def encode_binary(obj: Any, *, binary: Literal[False]) -> str: ...


def encode_binary(obj: Any, *, binary: bool = False) -> str | bytes:
    """Encodes an arbitrary object as a string.

    The object can be decoded using :func:`decode_binary`.

    Args:
        obj:
            The object to encode
        binary (bool):
            Encode as a byte array if `True`. Otherwise, a unicode string is returned

    Returns:
        str or bytes: The encoded object
    """
    obj_bin = pickle.dumps(obj)
    if binary:
        return obj_bin
    else:
        return codecs.encode(obj_bin, "base64").decode()


def decode_binary(obj_str: str | bytes | np.ndarray) -> Any:
    """Decode an object encoded with :func:`encode_binary`.

    Args:
        obj_str (str or bytes):
            The string that encodes the object

    Returns:
        Any: the object
    """
    if isinstance(obj_str, np.ndarray):
        if np.issubdtype(obj_str.dtype, np.uint8):
            obj_str = obj_str.tobytes()
        else:
            raise TypeError(f"Unexpected dtype `{obj_str.dtype}`")
    elif isinstance(obj_str, str):
        obj_str = codecs.decode(obj_str.encode(), "base64")
    return pickle.loads(obj_str)


def encode_class(cls: type) -> str:
    """Encode a class such that it can be restored.

    The class can be decoded using :func:`decode_class`.

    Args:
        cls (type):
            The class

    Returns:
        str: the encoded class
    """
    if cls is None:
        return "None"
    return cls.__module__ + "." + cls.__qualname__


def decode_class(class_path: str | None, *, guess: type | None = None) -> type | None:
    """Decode a class encoded with :func:`encode_class`.

    Args:
        class_path (str):
            The string that encodes the class
        guess (type):
            A class that is used if the encoded class cannot be found and the name of
            the guess matches the encoded class.

    Returns:
        type: the class or `None` if class_path was None
    """
    if class_path is None or class_path == "None":
        return None

    # import class from a package
    try:
        module_path, class_name = class_path.rsplit(".", 1)
    except (AttributeError, ValueError) as err:
        raise ImportError(f"Cannot import class {class_path}") from err

    try:
        module = import_module(module_path)
    except ModuleNotFoundError as err:
        # see whether the class is already defined ...
        if guess is not None and guess.__name__ == class_name:
            # ... as the `guess`
            return guess
        elif class_name in globals():
            # ... in the global context
            return globals()[class_name]  # type: ignore
        else:
            raise ModuleNotFoundError(f"Cannot load `{class_path}`") from err

    else:
        # load the class from the module
        try:
            return getattr(module, class_name)  # type: ignore
        except AttributeError as err:
            raise ImportError(
                f"Module {module_path} does not define {class_name}"
            ) from err


class Array(np.ndarray):
    """Numpy array augmented with attributes."""

    def __new__(cls, input_array, attrs: Attrs | None = None):
        obj = np.asarray(input_array).view(cls)
        obj.attrs = {} if attrs is None else attrs
        return obj

    def __array_finalize__(self, obj):
        if obj is None:  # __new__ handles instantiation
            return
        self.attrs = getattr(obj, "attrs", {})


ActionType = Literal[
    "read_item",  # read an item from storage
    "write_item",  # write an item to storage
]


class _StorageRegistry:
    """Registry that stores information about how to use storage."""

    allowed_actions = set(ActionType.__args__)  # type: ignore
    """set: all actions that can be registered"""

    _hooks: dict[type, dict[str, tuple[Callable, bool]]]
    """dict: register for all defined hooks"""

    def __init__(self):
        self._hooks = defaultdict(dict)

    def register(
        self,
        action: ActionType,
        cls: type,
        method_or_func: Callable,
        *,
        inherit: bool = True,
    ) -> None:
        """Register an action for the given class.

        Example:
            The method is used like so

            .. code-block:: python

                storage_actions.register("read_item", MyObj, MyObj.read_object)

        Args:
            action (str):
                The action provided by the method or function
            cls (type):
                The class this action is associated with
            method_or_func (callable):
                The function/method that is called for the action
            inherit (bool):
                Determines whether child classes of `cls` inherit this action and will
                be able to use it.
        """
        if action not in self.allowed_actions:
            raise ValueError(f"Unknown action `{action}` ")

        if isinstance(method_or_func, classmethod):
            # extract class from decorated object
            def _call_classmethod(*args, **kwargs):
                """Helper function to call the classmethod."""
                return method_or_func(cls, *args, **kwargs)

            self._hooks[cls][action] = (_call_classmethod, inherit)
        elif callable(method_or_func):
            self._hooks[cls][action] = (method_or_func, inherit)
        else:
            raise TypeError("`method_or_func` must be method or function")

    def get(self, cls: type, action: ActionType) -> Callable:
        """Obtain an action for a given class.

        Args:
            action (str):
                The action provided by the method or function
            cls (type):
                The class this action is associated with

        Returns:
            callable: The function/method that is called for the action
        """
        # look for defined operators on all parent classes (except `object`)
        classes = inspect.getmro(cls)[:-1]
        for c in classes:
            if c in self._hooks and action in self._hooks[c]:
                func, inherit = self._hooks[c][action]
                if inherit or c is cls:
                    return func

        raise RuntimeError(f"No action `{action}` for `{cls.__name__}`")


storage_actions = _StorageRegistry()
