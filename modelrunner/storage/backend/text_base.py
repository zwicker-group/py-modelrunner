"""
.. codeauthor:: David Zwicker <david.zwicker@ds.mpg.de>
"""

from __future__ import annotations

from abc import ABCMeta, abstractmethod
from io import StringIO
from pathlib import Path
from typing import Any, Optional, Sequence, Tuple, Union

import numpy as np
from numpy.lib.recfunctions import (
    structured_to_unstructured,
    unstructured_to_structured,
)
from numpy.typing import ArrayLike, DTypeLike

from ..access_modes import ModeType
from ..utils import decode_binary, encode_binary
from .memory import MemoryStorage
from .utils import simplify_data


class TextStorageBase(MemoryStorage, metaclass=ABCMeta):
    """base class for storage that stores data in a text file

    Note that the data is only written once the storage is closed.
    """

    encode_internal_attrs: bool = True

    def __init__(
        self,
        path: Union[str, Path],
        *,
        mode: ModeType = "readonly",
        simplify: bool = True,
        **kwargs,
    ):
        """
        Args:
            path (str or :class:`~pathlib.Path`):
                File path to the file
            mode (str or :class:`~modelrunner.storage.access_modes.AccessMode`):
                The file mode with which the storage is accessed. Determines allowed
                operations.
            simplify (bool):
                Flag indicating whether the data is stored in a simplified form
        """
        super().__init__(mode=mode)

        self.simplify = simplify
        self._path = Path(path)
        self._write_flags = kwargs
        if self.mode.file_mode in {"r", "x", "a"}:
            if self._path.exists():
                with open(self._path, mode="r") as fp:
                    self._read_data_from_fp(fp)

    def __repr__(self):
        return f'{self.__class__.__name__}("{self._path}", ' f'mode="{self.mode.name}")'

    def close(self) -> None:
        """close the file and write the data to the file"""
        if self.mode.file_mode in {"x", "a", "w"}:
            if self.simplify:
                data = simplify_data(self._data)
            else:
                data = self._data
            with open(self._path, mode="w") as fp:
                self._write_data_to_fp(fp, data)

    def to_text(self, simplify: Optional[bool] = None) -> str:
        """serialize the data and return it as a string

        Args:
            simplify (bool):
                Flag indicating whether the data is stored in a simplified form. If
                `None`, the object-level value is used.
        """
        if simplify is None:
            simplify = self.simplify
        if simplify:
            data = simplify_data(self._data)
        else:
            data = self._data
        with StringIO() as fp:
            self._write_data_to_fp(fp, data)
            return fp.getvalue()

    @abstractmethod
    def _read_data_from_fp(self, fp) -> None:
        ...

    @abstractmethod
    def _write_data_to_fp(self, fp, data) -> None:
        ...

    def _read_array(
        self, loc: Sequence[str], *, index: Optional[int] = None
    ) -> np.ndarray:
        # read the data from the location
        if index is None:
            arr = self[loc]["data"]
        else:
            arr = self[loc]["data"][index]

        if hasattr(arr, "__iter__"):  # minimal sanity check
            dtype = decode_binary(self[loc]["dtype"])
            if dtype.names is not None:
                arr = unstructured_to_structured(np.asarray(arr), dtype=dtype)
            else:
                arr = np.array(arr, dtype=dtype)
            if self[loc].get("record_array", False):
                arr = arr.view(np.recarray)
            return arr  # type: ignore
        else:
            raise RuntimeError(f"No array at `/{'/'.join(loc)}`")

    def _write_array(self, loc: Sequence[str], arr: np.ndarray) -> None:
        parent, name = self._get_parent(loc, check_write=True)

        dtype = arr.dtype  # extract dtype here since `arr` is changed later
        if dtype.names is not None:
            # structured array
            arr = structured_to_unstructured(arr)

        parent[name] = {
            "data": np.array(arr, copy=True),
            "dtype": encode_binary(dtype, binary=False),
        }
        if isinstance(arr, np.record):
            parent[name]["record_array"] = True

    def _create_dynamic_array(
        self,
        loc: Sequence[str],
        shape: Tuple[int, ...],
        dtype: DTypeLike,
        *,
        record_array: bool = False,
    ) -> None:
        parent, name = self._get_parent(loc, check_write=True)
        if name in parent:
            raise RuntimeError(f"Array `/{'/'.join(loc)}` already exists")
        parent[name] = {
            "data": [],
            "shape": shape,
            "dtype": encode_binary(np.dtype(dtype), binary=False),
        }
        if record_array:
            parent[name]["record_array"] = True

    def _extend_dynamic_array(self, loc: Sequence[str], arr: ArrayLike) -> None:
        item = self[loc]

        # check data shape that is stored at this position
        data = np.asanyarray(arr)
        stored_shape = tuple(item["shape"])
        if stored_shape != data.shape:
            raise TypeError(f"Shape mismatch ({stored_shape} != {data.shape})")

        # convert the data to the correct format
        stored_dtype = decode_binary(item["dtype"])
        if not np.issubdtype(data.dtype, stored_dtype):
            raise TypeError(f"Dtype mismatch ({data.dtype} != {stored_dtype}")
        if data.dtype.names is not None:
            # structured array
            data = structured_to_unstructured(data)

        # append the data to the dynamic array
        if data.ndim == 0:
            item["data"].append(data.item())
        else:
            item["data"].append(np.array(data, copy=True))

    def _read_object(self, loc: Sequence[str]) -> Any:
        return self.codec.decode(self[loc]["data"])

    def _write_object(self, loc: Sequence[str], obj: Any) -> None:
        parent, name = self._get_parent(loc, check_write=True)
        parent[name] = {"data": self.codec.encode(obj)}
