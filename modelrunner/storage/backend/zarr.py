"""
Defines a class storing data on the file system using the hierarchical data format (hdf)

.. codeauthor:: David Zwicker <david.zwicker@ds.mpg.de> 
"""

from __future__ import annotations

import shutil
from pathlib import Path
from typing import Any, List, Optional, Sequence, Tuple, Union

import numpy as np
import zarr
from numpy.typing import ArrayLike, DTypeLike
from zarr._storage.store import Store

from ..base import StorageBase
from ..utils import InfoDict, OpenMode

zarrElement = Union[zarr.Group, zarr.Array]


class ZarrStorage(StorageBase):
    """store data in an zarr file"""

    extensions = ["zarr"]

    def __init__(self, store_or_path, *, mode: OpenMode = "x", overwrite: bool = False):
        super().__init__(overwrite=overwrite)

        if isinstance(store_or_path, (str, Path)):
            # open zarr storage on file system
            path = Path(store_or_path)
            if path.suffix == "":
                # path seems to be a directory
                if path.is_dir() and mode == "w":
                    self._logger.info(f"Delete directory `{path}`")
                    shutil.rmtree(path)  # remove the directory to reinstate it
                if mode == "r":
                    self._logger.info(f"Directory are always opened writable")

                self._store = zarr.DirectoryStore(path)
            else:
                # path seems to be point to a file
                if path.exists():
                    if mode == "x":
                        self._logger.info('`ZipStore` uses mode="r" instead of "x"')
                        mode = "r"
                    if mode == "w":
                        self._logger.info(f"Delete file `{path}`")
                        path.unlink()
                self._store = zarr.storage.ZipStore(path, mode=mode)

        elif isinstance(store_or_path, Store):
            # open abstract zarr storage
            self._store = store_or_path

        self._root = zarr.group(store=self._store, overwrite=overwrite)

    def close(self):
        self._store.close()
        self._root = None

    def _get_parent(self, key: Sequence[str], *, check_write: bool = False):
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

    def keys(self, key: Optional[Sequence[str]] = None) -> List[str]:
        if key:
            return self[key].keys()
        else:
            return self._root.keys()

    def is_group(self, key: Sequence[str]) -> bool:
        item = self[key]
        if isinstance(item, zarr.hierarchy.Group):
            return "__class__" not in item.attrs
        else:
            return False

    def _create_group(self, key: str):
        parent, name = self._get_parent(key, check_write=True)
        return parent.create_group(name)

    def _read_attrs(self, key: Sequence[str]) -> InfoDict:
        return self[key].attrs

    def _write_attrs(self, key: Sequence[str], attrs: InfoDict) -> None:
        item = self[key]
        if not self.overwrite:
            for k in attrs.keys():
                if k in item.attrs:
                    raise KeyError(f"Cannot overwrite attribute `{k}`")
        item.attrs.update(attrs)

    def _read_array(
        self, key: Sequence[str], *, index: Optional[int] = None
    ) -> np.ndarray:
        if index is None:
            return self[key]
        else:
            return self[key][index]

    def _write_array(self, key: Sequence[str], arr: np.ndarray):
        parent, name = self._get_parent(key, check_write=True)

        if arr.dtype == object:
            return parent.array(name, arr, object_codec=self.codec)
        else:
            return parent.array(name, arr)

    def _create_dynamic_array(
        self, key: Sequence[str], shape: Tuple[int, ...], dtype: DTypeLike
    ):
        parent, name = self._get_parent(key, check_write=True)
        if dtype == object:
            return parent.zeros(
                name,
                shape=(0,) + shape,
                chunks=(1,) + shape,
                dtype=dtype,
                object_codec=self.codec,
            )

        else:
            return parent.zeros(
                name, shape=(0,) + shape, chunks=(1,) + shape, dtype=dtype
            )

    def _extend_dynamic_array(self, key: Sequence[str], data: ArrayLike):
        self[key].append([data])

    def _get_dynamic_array(self, key: Sequence[str]) -> ArrayLike:
        return self[key]
