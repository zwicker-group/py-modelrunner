"""
Defines a class storing data in memory. 

.. codeauthor:: David Zwicker <david.zwicker@ds.mpg.de> 
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
from numpy.typing import ArrayLike, DTypeLike

from ..access_modes import ModeType
from ..base import StorageBase
from ..utils import Attrs


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

        name = loc[-1]
        if check_write and not self.mode.overwrite and name in value:
            raise RuntimeError(f"Overwriting `{'/'.join(loc)}` disabled")
        return value, name

    def __getitem__(self, loc: Sequence[str]) -> Any:
        parent, name = self._get_parent(loc)
        return parent[name]

    def keys(self, loc: Sequence[str]) -> List[str]:
        if loc:
            return self[loc].keys()
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
        return self[loc].get("__attrs__", {})

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
            return self[loc]["data"]
        else:
            return self[loc]["data"][index]

    def _write_array(self, loc: Sequence[str], arr: np.ndarray) -> None:
        parent, name = self._get_parent(loc, check_write=True)
        parent[name] = {"data": np.array(arr, copy=True)}

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
