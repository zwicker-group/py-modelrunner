"""
.. codeauthor:: David Zwicker <david.zwicker@ds.mpg.de>
"""

from __future__ import annotations

from typing import (
    Any,
    Collection,
    Iterator,
    List,
    Optional,
    Sequence,
    Tuple,
    Type,
    Union,
)

import numpy as np
from numpy.typing import ArrayLike, DTypeLike

from .attributes import Attrs
from .base import StorageBase
from .utils import Array, Location, decode_class, storage_actions

# TODO: Provide .attrs attribute with a descriptor protocol (implemented by the backend)
# TODO: Provide a simple viewer of the tree structure (e.g. a `tree` method)


class StorageGroup:
    """refers to a group within a storage"""

    def __init__(
        self,
        storage: Union[StorageBase, StorageGroup],
        loc: Union[None, str, Sequence[str]] = None,
    ):
        """
        Args:
            storage (:class:`StorageBase` or :class:`StorageGroup`):
                The storage where the group is defined. If this is a
                :class:`StorageGroup` itself, `loc` is interpreted relative to that
                group
            loc (str or list of str):
                Denotes the location (path) of the group within the storage
        """
        self.loc = []
        self.loc = self._get_loc(loc)

        if isinstance(storage, StorageBase):
            self._storage = storage
        elif isinstance(storage, StorageGroup):
            self.loc = storage.loc + self.loc
            self._storage = storage._storage
        else:
            raise TypeError

    def __repr__(self):
        return f'StorageGroup(storage={self._storage}, loc="/{"/".join(self.loc)}")'

    def _get_loc(self, loc: Location) -> List[str]:
        """return a normalized location from various input

        Args:
            loc (str or list of str):
                location in a general formation. For instance, "/" is interpreted as a
                group separator.

        Returns:
            list of str: A list of the individual location items
        """

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

    def keys(self) -> Collection[str]:
        """return name of all stored items in this group"""
        return self._storage.keys(self.loc)

    def __iter__(self) -> Iterator[Any]:
        """iterate over all stored items in this group"""
        for loc in self.keys():
            yield self[loc]

    def __contains__(self, loc: Location):
        """check wether a particular item is contained in this group"""
        return self._get_loc(loc) in self._storage

    def items(self) -> Iterator[Tuple[str, Any]]:
        """iterate over stored items, yielding the location and item of each"""
        for loc in self.keys():
            yield loc, self[loc]

    def read_attrs(self, loc: Location = None) -> Attrs:
        """read attributes associated with a particular location

        Args:
            loc (str or list of str):
                The location in the storage where the attributes are read

        Returns:
            dict: A copy of the attributes at this location
        """
        return self._storage.read_attrs(self._get_loc(loc))

    def write_attrs(self, loc: Location = None, attrs: Optional[Attrs] = None) -> None:
        """write attributes to a particular location

        Args:
            loc (str or list of str):
                The location in the storage where the attributes are written
            attrs (dict):
                The attributes to be added to this location
        """
        self._storage.write_attrs(self._get_loc(loc), attrs=attrs)

    @property
    def attrs(self) -> Attrs:
        """dict: the attributes associated with this group"""
        return self.read_attrs()

    def _read_object(self, loc: Sequence[str]):
        """read a python object from a particular location"""
        attrs = self._storage.read_attrs(loc)
        cls = decode_class(attrs.pop("__class__", None))
        if cls is None:
            # return numpy array
            arr = self._storage._read_array(loc)
            return Array(arr, attrs=attrs)
        else:
            # create object using a registered action
            create_object = storage_actions.get(cls, "read_object")
            return create_object(self._storage, loc)

    def is_group(self, loc: Location) -> bool:
        """determine whether the location is a group

        Args:
            loc (sequence of str):
                A list of strings determining the location in the storage

        Returns:
            bool: `True` if the loation is a group
        """
        return self._storage.is_group(self._get_loc(loc))

    def create_group(
        self,
        loc: Location,
        *,
        attrs: Optional[Attrs] = None,
        cls: Optional[Type] = None,
    ) -> StorageGroup:
        """create a new group at a particular location

        Args:
            loc (str or list of str):
                The location where the group will be created
            attrs (dict, optional):
                Attributes stored with the group
            cls (type):
                A class associated with this group

        Returns:
            :class:`StorageGroup`: The reference of the new group
        """
        loc_list = self._get_loc(loc)
        return self._storage.create_group(loc_list, attrs=attrs, cls=cls)

    def read_array(
        self,
        loc: Location,
        *,
        out: Optional[np.ndarray] = None,
        index: Optional[int] = None,
        copy: bool = True,
    ) -> np.ndarray:
        """read an array from a particular location

        Args:
            loc (str or list of str):
                The location where the array is created
            out (array, optional):
                An array to which the results are written
            index (int, optional):
                An index denoting the subarray that will be read
            copy (bool):
                Determines whether a copy of the data is returned. Set this flag to
                `False` for better performance in cases where the array is not modified.

        Returns:
            :class:`~numpy.ndarray`:
                An array containing the data. Identical to `out` if specified.
        """
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
        """write an array to a particular location

        Args:
            loc (str or list of str):
                The location where the array is read
            arr (:class:`~numpy.ndarray`):
                The array that will be written
            attrs (dict, optional):
                Attributes stored with the array
            cls (type):
                A class associated with this array
        """
        loc_list = self._get_loc(loc)
        self._storage.write_array(loc_list, arr, attrs=attrs, cls=cls)

    def create_dynamic_array(
        self,
        loc: Location,
        *,
        arr: Optional[np.ndarray] = None,
        shape: Optional[Tuple[int, ...]] = None,
        dtype: DTypeLike = float,
        record_array: bool = False,
        attrs: Optional[Attrs] = None,
        cls: Optional[Type] = None,
    ):
        """creates a dynamic array of flexible size

        Args:
            loc (str or list of str):
                The location where the dynamic array is created
            shape (tuple of int):
                The shape of the individual arrays. A singular axis is prepended to the
                shape, which can then be extended subsequently.
            dtype:
                The data type of the array to be written
            record_array (bool):
                Flag indicating whether the array is of type :class:`~numpy.recarray`
            attrs (dict, optional):
                Attributes stored with the array
            cls (type):
                A class associated with this array
        """
        if arr is not None:
            if shape is not None:
                raise TypeError("Cannot set `arr` and `shape` simultanously")
            shape = arr.shape
            dtype = arr.dtype
            record_array = isinstance(arr, np.recarray)
        if shape is None:
            raise TypeError("Either `arr` or `shape` need to be specified")

        self._storage.create_dynamic_array(
            self._get_loc(loc),
            shape,
            dtype=dtype,
            record_array=record_array,
            attrs=attrs,
            cls=cls,
        )

    def extend_dynamic_array(self, loc: Location, data: ArrayLike):
        """extend a dynamic array previously created

        Args:
            loc (str or list of str):
                The location where the dynamic array is located
            arr (array):
                The array that will be appended to the dynamic array
        """
        self._storage.extend_dynamic_array(self._get_loc(loc), data)
