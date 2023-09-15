"""
Defines a class storing data in memory. 

.. codeauthor:: David Zwicker <david.zwicker@ds.mpg.de> 
"""

from __future__ import annotations

from typing import Any, Collection, Dict, Optional, Sequence, Tuple

import numpy as np
from numpy.lib.recfunctions import (
    structured_to_unstructured,
    unstructured_to_structured,
)
from numpy.typing import ArrayLike, DTypeLike

from ..access_modes import AccessError, ModeType
from ..attributes import Attrs, decode_attr, encode_attr
from ..base import StorageBase


class MemoryStorage(StorageBase):
    """store items in memory"""

    _data: Attrs

    def __init__(self, *, mode: ModeType = "insert"):
        """
        Args:
            mode (str or :class:`~modelrunner.storage.access_modes.AccessMode`):
                The file mode with which the storage is accessed. Determines allowed
                operations.
        """
        super().__init__(mode=mode)
        self._data = {}

    def clear(self) -> None:
        """truncate the storage by removing all stored data.

        Args:
            clear_data_shape (bool):
                Flag determining whether the data shape is also deleted.
        """
        self._data = {}

    def _get_parent(
        self, loc: Sequence[str], *, check_write: bool = False
    ) -> Tuple[Dict, str]:
        value = self._data
        for part in loc[:-1]:
            try:
                value = value[part]
            except KeyError:
                if isinstance(value, dict):
                    value[part] = {}
                    value = value[part]
                else:
                    raise TypeError(f"Cannot add item to `{'/'.join(loc)}`")
        if not isinstance(value, dict):
            raise TypeError(f"Cannot add item to `{'/'.join(loc)}`")

        try:
            name = loc[-1]
        except IndexError:
            raise KeyError(f"Location `{'/'.join(loc)}` has no parent")

        if check_write and not self.mode.overwrite and name in value:
            raise AccessError(f"Overwriting `{'/'.join(loc)}` disabled")
        return value, name

    def __getitem__(self, loc: Sequence[str]) -> Any:
        if loc:
            parent, name = self._get_parent(loc)
            return parent[name]
        else:
            return self._data

    def keys(self, loc: Sequence[str]) -> Collection[str]:
        if loc:
            return self[loc].keys()  # type: ignore
        else:
            return self._data.keys()

    def is_group(self, loc: Sequence[str]) -> bool:
        item = self[loc]
        if isinstance(item, dict):
            attrs = item.get("__attrs__", {})
            return "__class__" not in attrs
        else:
            return False

    def _create_group(self, loc: Sequence[str]) -> None:
        parent, name = self._get_parent(loc, check_write=True)
        parent[name] = {}

    def _read_attrs(self, loc: Sequence[str]) -> Attrs:
        res = self[loc].get("__attrs__", {})
        if isinstance(res, dict):
            return res
        else:
            raise RuntimeError(f"No attributes at {'/'.join(loc)}")

    def _write_attr(self, loc: Sequence[str], name: str, value) -> None:
        item = self[loc]
        if "__attrs__" not in item:
            item["__attrs__"] = {name: value}
        else:
            item["__attrs__"][name] = value

    def _read_array(
        self, loc: Sequence[str], *, index: Optional[int] = None
    ) -> np.ndarray:
        if index is None:
            arr = self[loc]["data"]
        else:
            arr = self[loc]["data"][index]

        if hasattr(arr, "__iter__"):  # minimal sanity check
            dtype = decode_attr(self[loc]["__dtype__"])
            if dtype.names is not None:
                arr = unstructured_to_structured(np.asarray(arr), dtype=dtype)
            else:
                arr = np.array(arr, dtype=dtype)
            if self[loc].get("__record_array__", False):
                arr = arr.view(np.recarray)
            return arr
        else:
            raise RuntimeError(f"No array at {'/'.join(loc)}")

    def _write_array(self, loc: Sequence[str], arr: np.ndarray) -> None:
        parent, name = self._get_parent(loc, check_write=True)

        dtype = arr.dtype
        if dtype.names is not None:
            # structured array
            arr = structured_to_unstructured(arr)

        parent[name] = {
            "data": np.array(arr, copy=True),
            "__dtype__": encode_attr(dtype),
            "__record_array__": True,
        }

    def _create_dynamic_array(
        self, loc: Sequence[str], shape: Tuple[int, ...], dtype: DTypeLike
    ) -> None:
        parent, name = self._get_parent(loc, check_write=True)
        parent[name] = {"data": [], "shape": shape, "dtype": np.dtype(dtype)}

    def _extend_dynamic_array(self, loc: Sequence[str], arr: ArrayLike) -> None:
        item = self[loc]
        data = np.asanyarray(arr)
        if item["shape"] != data.shape:
            raise TypeError("Shape mismatch")
        if not np.issubdtype(data.dtype, item["dtype"]):
            raise TypeError("Dtype mismatch")

        if data.ndim == 0:
            item["data"].append(data.item())
        else:
            item["data"].append(np.array(data, copy=True))
