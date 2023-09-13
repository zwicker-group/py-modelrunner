"""
Base classes for storing data

.. codeauthor:: David Zwicker <david.zwicker@ds.mpg.de> 
"""

# TODO:
#   Rename `key` to `loc`?
#   Rename `Group` to `StorageGroup`?

from __future__ import annotations
import logging
from abc import ABCMeta, abstractmethod
from typing import TYPE_CHECKING, List, Optional, Sequence, Tuple, Type

import numcodecs
import numpy as np
from numpy.typing import ArrayLike, DTypeLike

from .attributes import decode_attrs, encode_attr
from .utils import Attrs, KeyType, encode_class
from .access import Access, AccessType, AccessError

if TYPE_CHECKING:
    from .group import Group  # @UnusedImport


class StorageBase(metaclass=ABCMeta):
    """base class for storing data"""

    extensions: List[str] = []
    default_codec = numcodecs.Pickle()

    def __init__(self, *, access: AccessType = "full"):
        """
        Args:
            mode (str):
                The file mode with which the storage is accessed. Might not be used by
                all storages.
            overwrite (bool):
                Determines whether existing data can be overwritten
        """
        self.access = Access.parse(access)
        self._logger = logging.getLogger(self.__class__.__name__)

    def close(self):
        ...

    @property
    def codec(self) -> numcodecs.abc.Codec:
        """:class:`~numcodecs.abc.Codec`: A codec used to encode binary data"""
        try:
            return self._codec
        except AttributeError:
            attrs = self._read_attrs([])
            if "__codec__" in attrs:
                self._codec = numcodecs.get_codec(attrs["__codec__"])
            else:
                self._codec = self.default_codec
                self._write_attr([], "__codec__", self._codec.get_config())
        return self._codec

    def _get_key(self, key: KeyType) -> Sequence[str]:
        """convert `key` to definite format"""

        # TODO: use regex to check whether key is only alphanumerical and has no "/"
        # FIXME: get rid of this function and move all its functionality to `Group`
        def parse_key(key_data) -> List[str]:
            if key_data is None:
                return []
            elif isinstance(key_data, str):
                return key_data.split("/")
            else:
                return sum((parse_key(k) for k in key_data), start=list())

        return parse_key(key)

    def _get_attrs(
        self, attrs: Optional[Attrs], *, cls: Optional[Type] = None
    ) -> Attrs:
        """create attributes dictionary

        Args:
            attrs (dict):
                Dictionary with arbitrary attributes
            cls (type):
                Class information that needs to be stored alongside
        """
        if attrs is None:
            attrs = {}
        else:
            attrs = dict(attrs)
        if cls is not None:
            attrs["__class__"] = encode_class(cls)
        return attrs

    @abstractmethod
    def keys(self, key: Sequence[str]) -> List[str]:
        """return all sub-items defined for at the given `key`"""
        ...

    def __contains__(self, key: Sequence[str]):
        return key[-1] in self.keys(key[:-1])

    @abstractmethod
    def is_group(self, key: Sequence[str]) -> bool:
        ...

    @abstractmethod
    def _create_group(self, key: Sequence[str]):
        ...

    def create_group(
        self,
        key: Sequence[str],
        *,
        attrs: Optional[Attrs] = None,
        cls: Optional[Type] = None,
    ) -> "Group":
        """create a new group at a particular location

        Args:
            key (list of str):
                The location marking the new group
            attrs (dict, optional):
                Attributes stored with the group
            cls (type):
                A class associated with this group

        Returns:
            :class:`Group`: The reference of the new group
        """
        from .group import Group  # @Reimport to avoid circular import

        key = self._get_key(key)

        if key in self:
            if self.access.overwrite:
                pass  # group already exists, but we can overwrite things
            else:
                # we cannot overwrite anythign
                raise AccessError(f"Group `{'/'.join(key)}` already exists")
        else:
            if not self.access.insert:
                raise AccessError(f"No right to insert group `{'/'.join(key)}`")
            self._create_group(key)

        self.write_attrs(key, self._get_attrs(attrs, cls=cls))
        return Group(self, key)

    @abstractmethod
    def _read_attrs(self, key: Sequence[str]) -> Attrs:
        ...

    def read_attrs(self, key: Sequence[str]) -> Attrs:
        """read attributes associated with a particular location

        Args:
            key (list of str):
                The location

        Returns:
            dict: A copy of the attributes at this location
        """
        if not self.access.read:
            raise AccessError("No right to read attributes")
        return decode_attrs(self._read_attrs(key))

    @abstractmethod
    def _write_attr(self, key: Sequence[str], name: str, value) -> None:
        """write a single attribute to a particular location"""
        ...

    def write_attrs(self, key: Sequence[str], attrs: Optional[Attrs]) -> None:
        """write attributes to a particular location

        Args:
            key (list of str):
                The location
            attrs (dict):
                The attributes to be added to this location
        """
        # check whether we can insert anything
        if not self.access.insert:
            raise AccessError(f"No right to insert attributes into `{'/'.join(key)}`")
        # check whether there are actually any attributes to be written
        if attrs is None or len(attrs) == 0:
            return

        if self.access.overwrite:
            current_attrs = set()  # effectively disables check below
        else:
            current_attrs = self.read_attrs(key)  # previous attrs not to be changed
        for name, value in attrs.items():
            if name in current_attrs:
                raise AccessError(f"No right to overwrite attribute {name}")
            else:
                self._write_attr(key, name, encode_attr(value))

    @abstractmethod
    def _read_array(
        self,
        key: Sequence[str],
        *,
        index: Optional[int] = None,
    ) -> np.ndarray:
        ...

    def read_array(
        self,
        key: KeyType,
        *,
        out: Optional[np.ndarray] = None,
        index: Optional[int] = None,
        copy: bool = True,
    ) -> np.ndarray:
        """read an array from a particular location

        Args:
            key (list of str):
                The location where the array is created
            out (array):
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
        if not self.access.read:
            raise AccessError("No right to read array")

        key = self._get_key(key)
        if out is not None:
            out[:] = self._read_array(key, index=index)
        elif copy:
            out = np.array(self._read_array(key, index=index), copy=True)
        else:
            out = self._read_array(key, index=index)
        return out

    @abstractmethod
    def _write_array(self, key: Sequence[str], arr: np.ndarray, attrs: Attrs) -> None:
        ...

    def write_array(
        self,
        key: KeyType,
        arr: np.ndarray,
        *,
        attrs: Optional[Attrs] = None,
        cls: Optional[Type] = None,
    ) -> None:
        """write an array to a particular location

        Args:
            key (list of str):
                The location where the array is read
            arr (array):
                The array which will be written
            attrs (dict, optional):
                Attributes stored with the array
            cls (type):
                A class associated with this array
        """
        key = self._get_key(key)

        if key in self:
            # check whether we can overwrite the existing array
            if not self.access.overwrite:
                raise RuntimeError(f"Array `{'/'.join(key)}` already exists")
        else:
            # check whether we can insert a new array
            if not self.access.insert:
                raise RuntimeError(f"No right to insert array `{'/'.join(key)}`")

        self._write_array(key, arr)
        self.write_attrs(key, self._get_attrs(attrs, cls=cls))

    @abstractmethod
    def _create_dynamic_array(
        self, key: Sequence[str], shape: Tuple[int, ...], dtype: DTypeLike
    ) -> None:
        raise NotImplementedError(f"No dynamic arrays for {self.__class__.__name__}")

    def create_dynamic_array(
        self,
        key: KeyType,
        shape: Tuple[int, ...],
        *,
        dtype: DTypeLike = float,
        attrs: Optional[Attrs] = None,
        cls: Optional[Type] = None,
    ) -> None:
        """creates a dynamic array of flexible size

        Args:
            key (list of str):
                The location where the array is created
            shape (tuple of int):
                The shape of the individual arrays. A singular axis is prepended to the
                shape, which can then be extended subsequently.
            dtype:
                The data type of the array to be written
            attrs (dict, optional):
                Attributes stored with the array
            cls (type):
                A class associated with this array
        """
        key = self._get_key(key)

        if key in self:
            # check whether we can overwrite the existing array
            if not self.access.overwrite:
                raise RuntimeError(f"Array `{'/'.join(key)}` already exists")
            # TODO: Do we need to clear this array?
        else:
            # check whether we can insert a new array
            if not self.access.insert:
                raise RuntimeError(f"No right to insert array `{'/'.join(key)}`")

        self._create_dynamic_array(key, tuple(shape), dtype=dtype)
        self.write_attrs(key, self._get_attrs(attrs, cls=cls))

    @abstractmethod
    def _extend_dynamic_array(self, key: Sequence[str], arr: ArrayLike) -> None:
        raise NotImplementedError(f"No dynamic arrays for {self.__class__.__name__}")

    def extend_dynamic_array(self, key: KeyType, arr: ArrayLike) -> None:
        """extend a dynamic array previously created

        Args:
            key (list of str):
                The location of the dynamic array
            arr (array):
                The array which will be appended to the dynamic array
        """
        if not self.access.insert and not self.access.overwrite:
            raise RuntimeError(f"Cannot add data to `{'/'.join(key)}`")
        self._extend_dynamic_array(self._get_key(key), arr)

    @abstractmethod
    def _get_dynamic_array(self, key: Sequence[str]) -> ArrayLike:
        ...

    def get_dynamic_array(self, key: KeyType) -> ArrayLike:
        # FIXME: get rid of `get_dynamic_array`
        return self._get_dynamic_array(self._get_key(key))
