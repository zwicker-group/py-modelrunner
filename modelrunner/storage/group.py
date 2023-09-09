"""
.. codeauthor:: David Zwicker <david.zwicker@ds.mpg.de>
"""

from __future__ import annotations

import logging
from abc import ABCMeta, abstractmethod
from importlib import import_module
from typing import List, Iterator, Sequence, Tuple, Union, Dict, Optional, Any
import numpy as np


from .base import StorageBase, InfoDict


class ArrayWithAttrs(np.ndarray):
    """array class that is returned if no class can be loaded"""

    def __new__(cls, input_array, attrs: Optional[InfoDict] = None):
        obj = np.asarray(input_array).view(cls)
        obj.attrs = {} if attrs is None else attrs
        return obj

    def __array_finalize__(self, obj):
        if obj is None:  # __new__ handles instantiation
            return
        """we essentially need to set all our attributes that are set in __new__ here again (including their default values). 
        Otherwise numpy's view-casting and new-from-template mechanisms would break our class.
        """
        self.attrs = getattr(obj, "attrs", {})


def encode_class(cls) -> str:
    if cls is None:
        return "None"
    return cls.__module__ + "." + cls.__name__


def decode_class(class_path: Optional[str]):
    if class_path is None or class_path == "None":
        return None

    # import class from a package
    try:
        module_path, class_name = class_path.rsplit(".", 1)
    except ValueError:
        raise ImportError(f"Cannot import class {class_path}")

    module = import_module(module_path)

    try:
        return getattr(module, class_name)
    except AttributeError:
        raise ImportError(f"Module {module_path} does not define {class_name}")


class Group:
    def __init__(
        self, storage: StorageBase, path: Union[None, str, Sequence[str]] = None
    ):
        self._storage = storage
        if path is None:
            self.path = []
        elif isinstance(path, str):
            self.path = path.split("/")
        else:
            self.path = path

    def __getitem__(self, key: str) -> Any:
        """read state or trajectory from storage"""
        key = self.path + key.split("/")
        if self._storage.is_group(key):
            return Group(self._storage, key)
        else:
            attrs = self._storage._read_attrs(key)
            cls = decode_class(attrs.get("__class__"))
            if cls is None:
                # return numpy array
                data, attrs = self._storage._read_array(key)
                attrs = dict(attrs)  # make a shallow copy
                attrs.pop("__class__")
                return ArrayWithAttrs(data, attrs=attrs)
            else:
                # create object
                parent = self if len(key) == 1 else self[key[:-1]]
                return cls._from_stored(parent, attrs)

    def keys(self) -> Sequence[str]:
        """return name of all stored items"""
        return self._storage.keys(self.path)

    def __iter__(self) -> Iterator[Any]:
        """iterate over all stored items and trajectories"""
        for key in self.keys():
            yield self[key]

    def items(self) -> Iterator[Tuple[str, Any]]:
        """iterate over stored items and trajectories"""
        for key in self.keys():
            yield key, self[key]

    def create_group(self, key: str) -> Group:
        """key: relative path in current group"""
        path = self.path + key.split("/")
        self._storage.create_group(path)
        return Group(self._storage, path)

    def write_array(
        self, key: str, arr: np.ndarray, *, cls=None, attrs: Optional[InfoDict] = None
    ):
        path = self.path + key.split("/")
        attrs_ = {"__class__": encode_class(cls)}
        if attrs is not None:
            attrs_.update(attrs)
        self._storage._write_array(path, arr, attrs_)
