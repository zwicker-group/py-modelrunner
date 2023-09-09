"""
Defines a class storing data in memory. 

.. codeauthor:: David Zwicker <david.zwicker@ds.mpg.de> 
"""

from __future__ import annotations

from typing import Optional, Sequence, Tuple, Any, List

import numpy as np

from ..base import StorageBase, InfoDict


class MemoryStorage(StorageBase):
    """store items in memory"""

    _data: InfoDict

    def __init__(self, *, overwrite: bool = False):
        super().__init__(overwrite=overwrite)
        self._data = {}

    def clear(self) -> None:
        """truncate the storage by removing all stored data.

        Args:
            clear_data_shape (bool):
                Flag determining whether the data shape is also deleted.
        """
        self._data = {}

    def _get_parent(
        self,
        key: Sequence[str],
        *,
        validate_key: bool = True,
        check_write: bool = False,
    ):
        # TODO: use regex to check whether key is only alphanumerical and has /

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
        if check_write and not self.overwrite and name in value:
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
            attrs = item.get("attrs", {})
            return "__class__" not in attrs
        else:
            return False

    def create_group(self, key: Sequence[str]):
        parent, name = self._get_parent(key, check_write=True)
        parent[name] = {}

    def _read_attrs(self, key: Sequence[str]) -> InfoDict:
        return self[key].get("attrs", {})

    def _read_array(self, key: Sequence[str]) -> Tuple[np.ndarray, InfoDict]:
        item = self[key]
        return item["data"], item.get("attrs", {})

    def _write_array(self, key: Sequence[str], arr: np.ndarray, attrs: InfoDict):
        parent, name = self._get_parent(key, check_write=True)
        parent[name] = {"data": arr, "attrs": attrs}
