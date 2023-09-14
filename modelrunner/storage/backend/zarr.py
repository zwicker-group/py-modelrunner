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

from ..access_modes import ModeType
from ..base import StorageBase
from ..utils import Attrs

zarrElement = Union[zarr.Group, zarr.Array]


class ZarrStorage(StorageBase):
    """storage that stores data in an zarr file"""

    extensions = ["zarr"]

    def __init__(
        self, store_or_path: Union[str, Path, Store], *, mode: ModeType = "readonly"
    ):
        """
        Args:
            store_or_path (str or :class:`~pathlib.Path` or :class:`~zarr._storage.store.Store`):
                File path to the file/folder or a :mod:`zarr` Store
            mode (str or :class:`~modelrunner.storage.access_modes.AccessMode`):
                The file mode with which the storage is accessed. Determines allowed
                operations.
        """
        super().__init__(mode=mode)

        if isinstance(store_or_path, (str, Path)):
            # open zarr storage on file system
            self._close = True
            path = Path(store_or_path)
            if path.suffix == "":
                # path seems to be a directory
                if path.is_dir() and self.mode.file_mode == "w":
                    self._logger.info(f"Delete directory `{path}`")
                    shutil.rmtree(path)  # remove the directory to reinstate it
                if self.mode.file_mode == "r":
                    self._logger.info(f"Directory are always opened writable")

                self._store = zarr.DirectoryStore(path)
            else:
                # path seems to be point to a file
                file_mode = self.mode.file_mode
                if path.exists():
                    if file_mode == "x":
                        self._logger.info('`ZipStore` uses mode="r" instead of "x"')
                        file_mode = "r"
                    if file_mode == "w":
                        self._logger.info(f"Delete file `{path}`")
                        path.unlink()
                self._store = zarr.storage.ZipStore(path, mode=file_mode)

        elif isinstance(store_or_path, Store):
            # use already opened zarr storage
            self._close = False
            self._store = store_or_path

        else:
            raise TypeError(f"Unknown store `{store_or_path}`")

        self._root = zarr.group(store=self._store, overwrite=self.mode.overwrite)

    def __repr__(self):
        return f'{self.__class__.__name__}({self._root.store}, mode="{self.mode.name}")'

    def close(self) -> None:
        if self._close:
            self._store.close()
        self._root = None

    def _get_parent(
        self, loc: Sequence[str], *, check_write: bool = False
    ) -> Tuple[zarr.Group, str]:
        path, name = loc[:-1], loc[-1]
        parent = self._root
        for part in path:
            try:
                parent = parent[part]
            except KeyError:
                parent = parent.create_group(part)

        if check_write and not self.mode.overwrite and name in parent:
            raise RuntimeError(f"Overwriting `{', '.join(loc)}` disabled")

        return parent, name

    def __getitem__(self, loc: Sequence[str]) -> Any:
        if len(loc) == 0:
            return self._root
        else:
            parent, name = self._get_parent(loc)
            return parent[name]

    def keys(self, loc: Optional[Sequence[str]] = None) -> List[str]:
        if loc:
            return self[loc].keys()
        else:
            return self._root.keys()

    def is_group(self, loc: Sequence[str]) -> bool:
        item = self[loc]
        if isinstance(item, zarr.hierarchy.Group):
            return "__class__" not in item.attrs
        else:
            return False

    def _create_group(self, loc: Sequence[str]) -> None:
        parent, name = self._get_parent(loc, check_write=True)
        parent.create_group(name)

    def _read_attrs(self, loc: Sequence[str]) -> Attrs:
        return self[loc].attrs

    def _write_attr(self, loc: Sequence[str], name: str, value) -> None:
        self[loc].attrs[name] = value

    def _read_array(
        self, loc: Sequence[str], *, index: Optional[int] = None
    ) -> np.ndarray:
        if index is None:
            return self[loc]
        else:
            return self[loc][index]

    def _write_array(self, loc: Sequence[str], arr: np.ndarray) -> None:
        parent, name = self._get_parent(loc, check_write=True)

        if arr.dtype == object:
            parent.array(name, arr, object_codec=self.codec)
        else:
            parent.array(name, arr)

    def _create_dynamic_array(
        self, loc: Sequence[str], shape: Tuple[int, ...], dtype: DTypeLike
    ) -> None:
        parent, name = self._get_parent(loc, check_write=True)
        if dtype == object:
            parent.zeros(
                name,
                shape=(0,) + shape,
                chunks=(1,) + shape,
                dtype=dtype,
                object_codec=self.codec,
            )

        else:
            parent.zeros(name, shape=(0,) + shape, chunks=(1,) + shape, dtype=dtype)

    def _extend_dynamic_array(self, loc: Sequence[str], data: ArrayLike) -> None:
        self[loc].append([data])
