"""
Base classes for storing data

.. codeauthor:: David Zwicker <david.zwicker@ds.mpg.de> 
"""

from __future__ import annotations

import logging
from abc import ABCMeta, abstractmethod
from typing import TYPE_CHECKING, List, Optional, Sequence, Tuple, Type

import numcodecs
import numpy as np
from numpy.typing import ArrayLike, DTypeLike

from .attributes import decode_attrs, encode_attrs
from .utils import InfoDict, KeyType, encode_class

if TYPE_CHECKING:
    from .group import Group  # @UnusedImport


class StorageBase(metaclass=ABCMeta):
    """base class for storing data"""

    extensions: List[str] = []
    default_codec = numcodecs.Pickle()

    def __init__(self, *, overwrite: bool = False):
        """
        Args:
            overwrite (bool):
                Determines whether existing data can be overwritten
        """
        self.overwrite = overwrite
        self._logger = logging.getLogger(self.__class__.__name__)

    @property
    def codec(self):
        try:
            return self._codec
        except AttributeError:
            if "__codec__" in self._root.attrs:
                self._codec = numcodecs.get_codec(self._root.attrs["__codec__"])
            else:
                self._codec = self.default_codec
                # FIX this to work for all
                self._root.attrs["__codec__"] = self._codec.get_config()
        return self._codec

    def _get_key(self, key: KeyType):
        # TODO: use regex to check whether key is only alphanumerical and has no "/"
        def parse_key(key_data) -> List[str]:
            if key_data is None:
                return []
            elif isinstance(key_data, str):
                return key_data.split("/")
            else:
                return sum((parse_key(k) for k in key_data), start=list())

        return parse_key(key)

    def _get_attrs(self, attrs: Optional[InfoDict], cls: Optional[Type] = None):
        if attrs is None:
            attrs = {}
        else:
            attrs = dict(attrs)
        if cls is not None:
            attrs["__class__"] = encode_class(cls)
        return attrs

    @abstractmethod
    def keys(self, key: Sequence[str]) -> List[str]:
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
        attrs: Optional[InfoDict] = None,
        cls: Optional[Type] = None,
    ) -> "Group":
        from .group import Group  # @Reimport to avoid circular import

        key = self._get_key(key)
        self._create_group(key)
        self.write_attrs(key, self._get_attrs(attrs, cls))
        return Group(self, key)

    @abstractmethod
    def _read_attrs(self, key: Sequence[str]) -> InfoDict:
        ...

    def read_attrs(self, key: Sequence[str], *, copy: bool = True) -> InfoDict:
        # FIXME: remove `copy` argument
        return decode_attrs(self._read_attrs(key))

    @abstractmethod
    def _write_attrs(self, key: Sequence[str], attrs: InfoDict) -> None:
        ...

    def write_attrs(self, key: Sequence[str], attrs: Optional[InfoDict]) -> None:
        if attrs is not None and len(attrs) > 0:
            self._write_attrs(key, encode_attrs(attrs))

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
        key = self._get_key(key)
        if out is not None:
            out[:] = self._read_array(key, index=index)
        elif copy:
            out = np.array(self._read_array(key, index=index), copy=True)
        else:
            out = self._read_array(key, index=index)
        return out

    @abstractmethod
    def _write_array(self, key: Sequence[str], arr: np.ndarray, attrs: InfoDict):
        ...

    def write_array(
        self,
        key: KeyType,
        arr: np.ndarray,
        *,
        attrs: Optional[InfoDict] = None,
        cls: Optional[Type] = None,
    ):
        key = self._get_key(key)
        self._write_array(key, arr)
        self.write_attrs(key, self._get_attrs(attrs, cls))

    @abstractmethod
    def _create_dynamic_array(
        self, key: Sequence[str], shape: Tuple[int, ...], dtype: DTypeLike
    ):
        raise NotImplementedError(f"No dynamic arrays for {self.__class__.__name__}")

    def create_dynamic_array(
        self,
        key: KeyType,
        shape: Tuple[int, ...],
        *,
        dtype: DTypeLike = float,
        attrs: Optional[InfoDict] = None,
        cls: Optional[Type] = None,
    ):
        key = self._get_key(key)
        self._create_dynamic_array(key, shape, dtype=dtype)
        self.write_attrs(key, self._get_attrs(attrs, cls))

    @abstractmethod
    def _extend_dynamic_array(self, key: Sequence[str], data: ArrayLike):
        raise NotImplementedError(f"No dynamic arrays for {self.__class__.__name__}")

    def extend_dynamic_array(self, key: KeyType, data: ArrayLike):
        self._extend_dynamic_array(self._get_key(key), data)

    @abstractmethod
    def _get_dynamic_array(self, key: Sequence[str]) -> ArrayLike:
        ...

    def get_dynamic_array(self, key: KeyType) -> ArrayLike:
        return self._get_dynamic_array(self._get_key(key))
