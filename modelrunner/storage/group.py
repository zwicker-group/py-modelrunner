"""
.. codeauthor:: David Zwicker <david.zwicker@ds.mpg.de>
"""

from __future__ import annotations

from typing import Any, Iterator, List, Optional, Sequence, Tuple, Type, Union

import numpy as np
from numpy.typing import ArrayLike, DTypeLike

from .base import StorageBase
from .utils import Array, Attrs, Location, decode_class, storage_actions

# TODO: Provide .attrs attribute with a descriptor protocol (implemented by the backend)


class StorageGroup:
    def __init__(
        self, storage: StorageBase, loc: Union[None, str, Sequence[str]] = None
    ):
        self.loc = []
        self.loc = self._get_loc(loc)

        if isinstance(storage, StorageBase):
            self._storage = storage
        elif isinstance(storage, StorageGroup):
            self.loc = storage.loc + self.loc
            self._storage = storage._storage
        else:
            raise TypeError

    def _get_loc(self, loc: Location):
        # TODO: use regex to check whether loc is only alphanumerical and has no "/"
        def parse_loc(loc_data) -> List[str]:
            if loc_data is None:
                return []
            elif isinstance(loc_data, str):
                return loc_data.split("/")
            else:
                return sum((parse_loc(k) for k in loc_data), start=list())

        return self.loc + parse_loc(loc)

    def __getitem__(self, loc: Location) -> Any:
        """read state or trajectory from storage"""
        loc = self._get_loc(loc)
        if self._storage.is_group(loc):
            # just return a subgroup at this location
            return StorageGroup(self._storage, loc)
        else:
            # reconstruct objected stored at this place
            return self._read_object(loc)

    def keys(self) -> Sequence[str]:
        """return name of all stored items"""
        return self._storage.keys(self.loc)

    def __iter__(self) -> Iterator[Any]:
        """iterate over all stored items and trajectories"""
        for loc in self.keys():
            yield self[loc]

    def __contains__(self, loc: Location):
        return self._get_loc(loc) in self._storage

    def items(self) -> Iterator[Tuple[str, Any]]:
        """iterate over stored items and trajectories"""
        for loc in self.keys():
            yield loc, self[loc]

    def read_attrs(self, loc: Location = None) -> Attrs:
        return self._storage.read_attrs(self._get_loc(loc))

    def write_attrs(self, loc: Location = None, attrs: Attrs = None) -> None:
        self._storage.write_attrs(self._get_loc(loc), attrs=attrs)

    @property
    def attrs(self) -> Attrs:
        return self.read_attrs()

    def _read_object(self, loc: Sequence[str]):
        attrs = self._storage.read_attrs(loc)
        cls = decode_class(attrs.pop("__class__"))
        if cls is None:
            # return numpy array
            arr = self._storage._read_array(loc)
            return Array(arr, attrs=attrs)
        else:
            # create object using a registered action
            create_object = storage_actions.get(cls, "read_object")
            return create_object(self._storage, loc)

    def create_group(
        self,
        loc: str,
        *,
        attrs: Optional[Attrs] = None,
        cls: Optional[Type] = None,
    ) -> StorageGroup:
        loc = self._get_loc(loc)
        return self._storage.create_group(loc, attrs=attrs, cls=cls)

    def read_array(
        self,
        loc: Location,
        *,
        out: Optional[np.ndarray] = None,
        index: Optional[int] = None,
        copy: bool = True,
    ) -> np.ndarray:
        return self._storage.read_array(
            self._get_loc(loc), out=out, index=index, copy=copy
        )

    def write_array(
        self,
        loc: Location,
        arr: np.ndarray,
        *,
        attrs: Optional[Attrs] = None,
        cls: Optional[Type] = None,
    ):
        loc = self._get_loc(loc)
        self._storage.write_array(loc, arr, attrs=attrs, cls=cls)

    def create_dynamic_array(
        self,
        loc: Location,
        shape: Tuple[int, ...],
        *,
        dtype: DTypeLike = float,
        attrs: Optional[Attrs] = None,
        cls: Optional[Type] = None,
    ):
        self._storage.create_dynamic_array(
            self._get_loc(loc), shape, dtype=dtype, attrs=attrs, cls=cls
        )

    def extend_dynamic_array(self, loc: Location, data: ArrayLike):
        self._storage.extend_dynamic_array(self._get_loc(loc), data)

    def get_dynamic_array(self, loc: Location) -> ArrayLike:
        return self._storage.get_dynamic_array(self._get_loc(loc))
