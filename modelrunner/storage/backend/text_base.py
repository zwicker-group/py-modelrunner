"""
.. codeauthor:: David Zwicker <david.zwicker@ds.mpg.de>
"""

from __future__ import annotations

import io
from abc import ABCMeta, abstractmethod
from io import StringIO
from pathlib import Path
from typing import Any, Sequence

import numpy as np
from numpy.lib.recfunctions import (
    structured_to_unstructured,
    unstructured_to_structured,
)
from numpy.typing import ArrayLike, DTypeLike

from ..access_modes import AccessError, ModeType
from ..utils import decode_binary, encode_binary
from .memory import MemoryStorage
from .utils import simplify_data


class TextStorageBase(MemoryStorage, metaclass=ABCMeta):
    """Base class for storage that stores data in a text file.

    Note that the data is only written once the storage is closed.
    """

    def __init__(
        self,
        path: str | Path,
        *,
        mode: ModeType = "read",
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
        self._modified = False
        if self.mode.file_mode in {"r", "x", "a"} and self._path.exists():
            if self.mode.file_mode == "x":
                raise FileExistsError(f"File `{path}` already exists")
            # read content from file
            with self._path.open() as fp:
                data = self._read_data_from_fp(fp)
            # interprete empty files correctly
            self._data = {} if data is None else data

    def __repr__(self):
        return f'{self.__class__.__name__}("{self._path}", mode="{self.mode.name}")'

    def flush(self) -> None:
        """Write (cached) data to storage."""
        if self.mode.file_mode in {"x", "a", "w"}:
            # Write the data to the writeable file. Note that we do not check the
            # self._modified flag since it might not capture all changes, e.g., when an
            # item (attribute, array, or object) was modified in place
            data = simplify_data(self._data) if self.simplify else self._data
            with self._path.open("w") as fp:
                self._write_data_to_fp(fp, data)
            self._modified = False  # reset modifications

        elif self._modified:
            # The storage was modified, but it cannot be written to the file. This
            # should not happen, but it's better to throw an explicit error
            raise AccessError("Cannot write modifications to file opened read-only")

    def close(self) -> None:
        """Close the file and write the data to the file."""
        self.flush()
        super().close()

    def to_text(self, simplify: bool | None = None) -> str:
        """Serialize the data and return it as a string.

        Args:
            simplify (bool):
                Flag indicating whether the data is stored in a simplified form. If
                `None`, the object-level value is used.
        """
        if simplify is None:
            simplify = self.simplify
        data = simplify_data(self._data) if self.simplify else self._data

        with StringIO() as fp:
            self._write_data_to_fp(fp, data)
            return fp.getvalue()

    @abstractmethod
    def _read_data_from_fp(self, fp: io.TextIOBase):
        """Read data from an open file.

        Args:
            fp (:class:`io.TextIOBase`): The opened text file
        """

    @abstractmethod
    def _write_data_to_fp(self, fp: io.TextIOBase, data) -> None:
        """Write data to an open file.

        Args:
            fp (:class:`io.TextIOBase`): The opened text file
            data: The data to write
        """

    def _write_attr(self, loc: Sequence[str], name: str, value: str) -> None:
        super()._write_attr(loc, name, value)
        self._modified = True

    def _read_array(
        self, loc: Sequence[str], *, copy: bool, index: int | None = None
    ) -> np.ndarray:
        # read the data from the location
        if index is None:
            arr = self[loc]["data"]
        else:
            arr = self[loc]["data"][index]

        if hasattr(arr, "__iter__"):  # minimal sanity check
            dtype = decode_binary(self[loc]["dtype"])
            if dtype.names is not None:
                arr = unstructured_to_structured(
                    np.asarray(arr), dtype=dtype, copy=copy
                )
            elif copy:
                arr = np.array(arr, dtype=dtype, copy=True)
            else:
                arr = np.asarray(arr, dtype=dtype)
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
        if isinstance(arr, np.recarray):
            parent[name]["record_array"] = True
        self._modified = True

    def _create_dynamic_array(
        self,
        loc: Sequence[str],
        shape: tuple[int, ...],
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
        self._modified = True

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
        self._modified = True

    def _read_object(self, loc: Sequence[str]) -> Any:
        return self.codec.decode(self[loc]["data"])

    def _write_object(self, loc: Sequence[str], obj: Any) -> None:
        parent, name = self._get_parent(loc, check_write=True)
        parent[name] = {"data": self.codec.encode(obj)}
        self._modified = True
