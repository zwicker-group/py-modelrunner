"""
.. codeauthor:: David Zwicker <david.zwicker@ds.mpg.de>
"""

from __future__ import annotations

from typing import Any, Dict, Iterator, Optional, Sequence, Tuple, Type, Union

import numpy as np
from numpy.typing import ArrayLike, DTypeLike

from .base import StorageBase
from .utils import Array, InfoDict, KeyType, decode_class, storage_actions

# TODO: Provide .attrs attribute with a descriptor protocol (implemented by the backend)


class Group:
    def __init__(
        self, storage: StorageBase, path: Union[None, str, Sequence[str]] = None
    ):
        if path is None:
            self.path = []
        elif isinstance(path, str):
            self.path = path.split("/")
        else:
            self.path = path

        if isinstance(storage, StorageBase):
            self._storage = storage
        elif isinstance(storage, Group):
            self.path = storage.path + self.path
            self._storage = storage._storage
        else:
            raise TypeError

    def _get_key(self, key: Optional[KeyType] = None):
        # TODO: use regex to check whether key is only alphanumerical and has no "/"
        if key is None:
            return self.path
        elif isinstance(key, str):
            return self.path + key.split("/")
        else:
            return self.path + key

    def __getitem__(self, key: KeyType) -> Any:
        """read state or trajectory from storage"""
        key = self._get_key(key)
        if self._storage.is_group(key):
            # just return a subgroup at this location
            return Group(self._storage, key)
        else:
            # reconstruct objected stored at this place
            return self._read_object(key)

    def keys(self) -> Sequence[str]:
        """return name of all stored items"""
        return self._storage.keys(self.path)

    def __iter__(self) -> Iterator[Any]:
        """iterate over all stored items and trajectories"""
        for key in self.keys():
            yield self[key]

    def __contains__(self, key: KeyType):
        return self._get_key(key) in self._storage

    def items(self) -> Iterator[Tuple[str, Any]]:
        """iterate over stored items and trajectories"""
        for key in self.keys():
            yield key, self[key]

    def read_attrs(
        self, key: Optional[KeyType] = None, *, copy: bool = True
    ) -> InfoDict:
        return self._storage.read_attrs(self._get_key(key), copy=copy)

    def write_attrs(
        self, key: Optional[KeyType] = None, attrs: InfoDict = None
    ) -> None:
        self._storage.write_attrs(self._get_key(key), attrs=attrs)

    @property
    def attrs(self) -> InfoDict:
        return self.read_attrs()

    def _read_object(self, key: Sequence[str]):
        attrs = self._storage.read_attrs(key, copy=False)
        cls = decode_class(attrs.get("__class__"))
        if cls is None:
            # return numpy array
            arr = self._storage._read_array(key)
            attrs = self._storage.read_attrs(key, copy=True)
            attrs.pop("__class__")
            return Array(arr, attrs=attrs)
        else:
            # create object using a registered action
            create_object = storage_actions.get(cls, "read_object")
            return create_object(self._storage, key)

    def create_group(
        self,
        key: str,
        *,
        attrs: Optional[InfoDict] = None,
        cls: Optional[Type] = None,
    ) -> Group:
        """key: relative path in current group"""
        key = self._get_key(key)
        return self._storage.create_group(key, attrs=attrs, cls=cls)

    def read_array(
        self,
        key: KeyType,
        *,
        out: Optional[np.ndarray] = None,
        index: Optional[int] = None,
        copy: bool = True,
    ) -> np.ndarray:
        return self._storage.read_array(
            self._get_key(key), out=out, index=index, copy=copy
        )

    def write_array(
        self,
        key: KeyType,
        arr: np.ndarray,
        *,
        attrs: Optional[InfoDict] = None,
        cls: Optional[Type] = None,
    ):
        key = self._get_key(key)
        self._storage.write_array(key, arr, attrs=attrs, cls=cls)

    def create_dynamic_array(
        self,
        key: KeyType,
        shape: Tuple[int, ...],
        *,
        dtype: DTypeLike = float,
        attrs: Optional[InfoDict] = None,
        cls: Optional[Type] = None,
    ):
        self._storage.create_dynamic_array(
            self._get_key(key), shape, dtype=dtype, attrs=attrs, cls=cls
        )

    def extend_dynamic_array(self, key: KeyType, data: ArrayLike):
        self._storage.extend_dynamic_array(self._get_key(key), data)

    def get_dynamic_array(self, key: KeyType) -> ArrayLike:
        return self._storage.get_dynamic_array(self._get_key(key))
