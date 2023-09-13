"""
Defines a class storing data in memory. 

.. codeauthor:: David Zwicker <david.zwicker@ds.mpg.de> 
"""

from __future__ import annotations

from typing import Any, List, Optional, Sequence, Tuple

import numpy as np
from numpy.typing import ArrayLike, DTypeLike

from ..base import StorageBase
from ..access import AccessType
from ..utils import Attrs


class MemoryStorage(StorageBase):
    """store items in memory"""

    _data: Attrs

    def __init__(self, *, access: AccessType = "full"):
        super().__init__(access=access)
        self._data = {}

    def clear(self) -> None:
        """truncate the storage by removing all stored data.

        Args:
            clear_data_shape (bool):
                Flag determining whether the data shape is also deleted.
        """
        self._data = {}

    def _get_parent(self, key: Sequence[str], *, check_write: bool = False):
        value = self._data
        for part in key[:-1]:
            try:
                value = value[part]
            except KeyError:
                if isinstance(value, dict):
                    value[part] = {}
                    value = value[part]
                else:
                    raise TypeError(f"Cannot add item to `{'/'.join(key)}`")
        if not isinstance(value, dict):
            raise TypeError(f"Cannot add item to `{'/'.join(key)}`")

        name = key[-1]
        if check_write and not self.access.overwrite and name in value:
            raise RuntimeError(f"Overwriting `{'/'.join(key)}` disabled")
        return value, name

    def __getitem__(self, key: Sequence[str]) -> Any:
        parent, name = self._get_parent(key)
        return parent[name]

    def keys(self, key: Sequence[str]) -> List[str]:
        if key:
            return self[key].keys()
        else:
            return self._data.keys()

    def is_group(self, key: Sequence[str]) -> bool:
        item = self[key]
        if isinstance(item, dict):
            attrs = item.get("__attrs__", {})
            return "__class__" not in attrs
        else:
            return False

    def _create_group(self, key: Sequence[str]):
        parent, name = self._get_parent(key, check_write=True)
        parent[name] = {}

    def _read_attrs(self, key: Sequence[str]) -> Attrs:
        return self[key].get("__attrs__", {})

    def _write_attr(self, key: Sequence[str], name: str, value):
        item = self[key]
        if "__attrs__" not in item:
            item["__attrs__"] = {name: value}
        else:
            item["__attrs__"][name] = value

    def _read_array(
        self, key: Sequence[str], *, index: Optional[int] = None
    ) -> np.ndarray:
        if index is None:
            return self[key]["data"]
        else:
            return self[key]["data"][index]

    def _write_array(self, key: Sequence[str], arr: np.ndarray):
        parent, name = self._get_parent(key, check_write=True)
        parent[name] = {"data": np.array(arr, copy=True)}

    def _create_dynamic_array(
        self, key: Sequence[str], shape: Tuple[int, ...], dtype: DTypeLike
    ):
        parent, name = self._get_parent(key, check_write=True)
        parent[name] = {"data": [], "shape": shape, "dtype": np.dtype(dtype)}

    def _extend_dynamic_array(self, key: Sequence[str], arr: ArrayLike):
        item = self[key]
        data = np.asanyarray(arr)
        if item["shape"] != data.shape:
            raise TypeError("Shape mismatch")
        if not np.issubdtype(data.dtype, item["dtype"]):
            raise TypeError("Dtype mismatch")

        if data.ndim == 0:
            item["data"].append(data.item())
        else:
            item["data"].append(np.array(data, copy=True))

    def _get_dynamic_array(self, key: Sequence[str]) -> ArrayLike:
        return self[key]["data"]
