"""
Base classes for storing data

.. codeauthor:: David Zwicker <david.zwicker@ds.mpg.de> 
"""

from __future__ import annotations

import logging
from abc import ABCMeta, abstractmethod
from typing import TYPE_CHECKING, Collection, List, Optional, Sequence, Tuple, Type

import numcodecs
import numpy as np
from numpy.typing import ArrayLike, DTypeLike

from .access_modes import AccessError, AccessMode, ModeType
from .attributes import Attrs, AttrsLike, decode_attrs, encode_attr
from .utils import encode_class

if TYPE_CHECKING:
    from .group import StorageGroup  # @UnusedImport


class StorageBase(metaclass=ABCMeta):
    """base class for storing data"""

    extensions: List[str] = []
    default_codec = numcodecs.Pickle()

    _codec: numcodecs.abc.Codec

    def __init__(self, *, mode: ModeType = "readonly"):
        """
        Args:
            mode (str or :class:`~modelrunner.storage.access_modes.AccessMode`):
                The file mode with which the storage is accessed. Determines allowed
                operations.
        """
        self.mode = AccessMode.parse(mode)
        self._logger = logging.getLogger(self.__class__.__name__)

    def close(self) -> None:
        """closes the storage, potentially writing data to a persistent place"""
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
    def keys(self, loc: Sequence[str]) -> Collection[str]:
        """return all sub-items defined at a given location

        Args:
            loc (sequence of str):
                A list of strings determining the location in the storage

        Returns:
            list: a list of all items defined at this location
        """
        ...

    def __contains__(self, loc: Sequence[str]):
        return loc[-1] in self.keys(loc[:-1])

    @abstractmethod
    def is_group(self, loc: Sequence[str]) -> bool:
        """determine whether the location is a group

        Args:
            loc (sequence of str):
                A list of strings determining the location in the storage

        Returns:
            bool: `True` if the loation is a group
        """
        ...

    @abstractmethod
    def _create_group(self, loc: Sequence[str]) -> None:
        ...

    def create_group(
        self,
        loc: Sequence[str],
        *,
        attrs: Optional[Attrs] = None,
        cls: Optional[Type] = None,
    ) -> "StorageGroup":
        """create a new group at a particular location

        Args:
            loc (list of str):
                The location in the storage where the group will be created
            attrs (dict, optional):
                Attributes stored with the group
            cls (type):
                A class associated with this group

        Returns:
            :class:`StorageGroup`: The reference of the new group
        """
        from .group import StorageGroup  # @Reimport to avoid circular import

        # TODO: allow creating many hierachies at once

        if loc in self:
            if self.mode.overwrite:
                pass  # group already exists, but we can overwrite things
            else:
                # we cannot overwrite anything
                raise AccessError(f"Group `{'/'.join(loc)}` already exists")
        else:
            if not self.mode.insert:
                raise AccessError(f"No right to insert group `{'/'.join(loc)}`")
            self._create_group(loc)

        self.write_attrs(loc, self._get_attrs(attrs, cls=cls))
        return StorageGroup(self, loc)

    @abstractmethod
    def _read_attrs(self, loc: Sequence[str]) -> AttrsLike:
        ...

    def read_attrs(self, loc: Sequence[str]) -> Attrs:
        """read attributes associated with a particular location

        Args:
            loc (list of str):
                The location in the storage where the attributes are read

        Returns:
            dict: A copy of the attributes at this location
        """
        if not self.mode.read:
            raise AccessError("No right to read attributes")
        return decode_attrs(self._read_attrs(loc))

    @abstractmethod
    def _write_attr(self, loc: Sequence[str], name: str, value) -> None:
        """write a single attribute to a particular location"""
        ...

    def write_attrs(self, loc: Sequence[str], attrs: Optional[Attrs]) -> None:
        """write attributes to a particular location

        Args:
            loc (list of str):
                The location in the storage where the attributes are written
            attrs (dict):
                The attributes to be added to this location
        """
        # check whether we can insert anything
        if not self.mode.set_attrs:
            raise AccessError(f"No right to set attributes of `{'/'.join(loc)}`")
        # check whether there are actually any attributes to be written
        if attrs is None or len(attrs) == 0:
            return

        for name, value in attrs.items():
            self._write_attr(loc, name, encode_attr(value))

    @abstractmethod
    def _read_array(
        self,
        loc: Sequence[str],
        *,
        index: Optional[int] = None,
    ) -> ArrayLike:
        ...

    def read_array(
        self,
        loc: Sequence[str],
        *,
        out: Optional[np.ndarray] = None,
        index: Optional[int] = None,
        copy: bool = True,
    ) -> np.ndarray:
        """read an array from a particular location

        Args:
            loc (list of str):
                The location in the storage where the array is created
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
        if not self.mode.read:
            raise AccessError("No right to read array")

        if out is not None:
            out[:] = self._read_array(loc, index=index)
        elif copy:
            out = np.array(self._read_array(loc, index=index), copy=True)
        else:
            out = np.asanyarray(self._read_array(loc, index=index))
        return out

    @abstractmethod
    def _write_array(self, loc: Sequence[str], arr: np.ndarray) -> None:
        ...

    def write_array(
        self,
        loc: Sequence[str],
        arr: np.ndarray,
        *,
        attrs: Optional[Attrs] = None,
        cls: Optional[Type] = None,
    ) -> None:
        """write an array to a particular location

        Args:
            loc (list of str):
                The location in the storage where the array is read
            arr (:class:`~numpy.ndarray`):
                The array which will be written
            attrs (dict, optional):
                Attributes stored with the array
            cls (type):
                A class associated with this array
        """
        if loc in self:
            # check whether we can overwrite the existing array
            if not self.mode.overwrite:
                raise RuntimeError(f"Array `{'/'.join(loc)}` already exists")
        else:
            # check whether we can insert a new array
            if not self.mode.insert:
                raise RuntimeError(f"No right to insert array `{'/'.join(loc)}`")

        self._write_array(loc, arr)
        self.write_attrs(loc, self._get_attrs(attrs, cls=cls))

    @abstractmethod
    def _create_dynamic_array(
        self, loc: Sequence[str], shape: Tuple[int, ...], dtype: DTypeLike
    ) -> None:
        raise NotImplementedError(f"No dynamic arrays for {self.__class__.__name__}")

    def create_dynamic_array(
        self,
        loc: Sequence[str],
        shape: Tuple[int, ...],
        *,
        dtype: DTypeLike = float,
        attrs: Optional[Attrs] = None,
        cls: Optional[Type] = None,
    ) -> None:
        """creates a dynamic array of flexible size

        Args:
            loc (list of str):
                The location in the storage where the dynamic array is created
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
        if loc in self:
            # check whether we can overwrite the existing array
            if not self.mode.overwrite:
                raise RuntimeError(f"Array `{'/'.join(loc)}` already exists")
            # TODO: Do we need to clear this array?
        else:
            # check whether we can insert a new array
            if not self.mode.insert:
                raise RuntimeError(f"No right to insert array `{'/'.join(loc)}`")

        self._create_dynamic_array(loc, tuple(shape), dtype=dtype)
        self.write_attrs(loc, self._get_attrs(attrs, cls=cls))

    @abstractmethod
    def _extend_dynamic_array(self, loc: Sequence[str], arr: ArrayLike) -> None:
        raise NotImplementedError(f"No dynamic arrays for {self.__class__.__name__}")

    def extend_dynamic_array(self, loc: Sequence[str], arr: ArrayLike) -> None:
        """extend a dynamic array previously created

        Args:
            loc (list of str):
                The location in the storage where the dynamic array is located
            arr (array):
                The array which will be appended to the dynamic array
        """
        if not self.mode.dynamic_append:
            raise RuntimeError(f"Cannot append data to dynamic array `{'/'.join(loc)}`")
        self._extend_dynamic_array(loc, arr)
