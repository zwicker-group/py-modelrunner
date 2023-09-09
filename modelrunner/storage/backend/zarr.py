"""
Defines a class storing data on the file system using the hierarchical data format (hdf)

.. codeauthor:: David Zwicker <david.zwicker@ds.mpg.de> 
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Optional, List, Tuple, Union, Sequence

import numpy as np


import zarr
from zarr._storage.store import Store
from ..base import StorageBase, InfoDict


zarrElement = Union[zarr.Group, zarr.Array]


class ZarrStorage(StorageBase):
    """store data in an zarr file"""

    extensions = ["zarr"]

    def __init__(self, store_or_path, *, overwrite: bool = False):
        super().__init__(overwrite=overwrite)
        mode = "w" if overwrite else "x"  # zarr.ZipStore does only supports r, w, a, x

        if isinstance(store_or_path, (str, Path)):
            path = Path(store_or_path)
            if path.suffix != "":
                self._store = zarr.storage.ZipStore(path, mode=mode)
            else:
                self._store = zarr.DirectoryStore(path, mode=mode)
        elif isinstance(store_or_path, Store):
            self._store = store_or_path

        self._root = zarr.group(store=self._store, overwrite=overwrite)

    def _get_parent(
        self,
        key: Sequence[str],
        *,
        validate_key: bool = True,
        check_write: bool = False,
    ):
        path, name = key[:-1], key[-1]
        parent = self._root
        for part in path:
            try:
                parent = parent[part]
            except KeyError:
                parent = parent.create_group(part)

        if check_write and not self.overwrite and name in parent:
            raise RuntimeError(f"Overwriting `{', '.join(key)}` disabled")

        return parent, name

    def __getitem__(self, key: Sequence[str]) -> Any:
        parent, name = self._get_parent(key)
        return parent[name]

    def keys(self, key: Sequence[str]) -> List[str]:
        if key:
            return self[key].keys()
        else:
            return self._root.keys()

    def is_group(self, key: Sequence[str]) -> bool:
        return isinstance(self[key], zarr.hierarchy.Group)

    def create_group(self, key: str):
        parent, name = self._get_parent(key, check_write=True)
        return parent.create_group(name)

    def _read_attrs(self, key: Sequence[str]) -> InfoDict:
        return self[key].attrs

    def _read_array(self, key: Sequence[str]) -> Tuple[np.ndarray, InfoDict]:
        parent, name = self._get_parent(key)
        element = parent[name]
        return element, element.attrs

    def _write_array(self, key: Sequence[str], arr: np.ndarray, attrs: InfoDict):
        parent, name = self._get_parent(key, check_write=True)
        element = parent.array(name, arr)
        element.attrs.update(attrs)  # write the attributes of the state
        return element
