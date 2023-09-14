"""
Defines a class storing data on the file system using the hierarchical data format (hdf)

.. codeauthor:: David Zwicker <david.zwicker@ds.mpg.de> 
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import h5py
import numpy as np
from numpy.typing import ArrayLike, DTypeLike

from ..access_modes import AccessError, ModeType
from ..attributes import Attrs, encode_attr
from ..base import StorageBase
from ..utils import decode_binary, encode_binary


class HDFStorage(StorageBase):
    """storage that stores data in an HDF file"""

    extensions = ["h5", "hdf", "hdf5"]

    def __init__(
        self,
        file_or_path: Union[str, Path, h5py.File],
        *,
        mode: ModeType = "readonly",
        compression: bool = True,
    ):
        """
        Args:
            file_or_path (str or :class:`~pathlib.Path` or :class:`~zarr._storage.store.Store`):
                File path to the file/folder or a :mod:`zarr` Store
            mode (str or :class:`~modelrunner.storage.access_modes.AccessMode`):
                The file mode with which the storage is accessed. Determines allowed
                operations.
            compression (bool):
                Whether to store the data in compressed form. Automatically enabled
                chunked storage.
        """
        super().__init__(mode=mode)
        self.compression = compression
        self._dynamic_array_size: Dict[str, int] = {}  # lengths of the dynamic arrays

        if isinstance(file_or_path, (str, Path)):
            # open HDF storage on file system
            self._close = True
            file_mode = self.mode.file_mode
            try:
                self._file = h5py.File(file_or_path, mode=file_mode)
            except FileExistsError:
                raise RuntimeError(f"File already exists")

        elif isinstance(file_or_path, h5py.File):
            # use opened HDF file
            self._close = False
            self._file = file_or_path

        else:
            raise TypeError(f"Unknown store `{file_or_path}`")

    def __repr__(self):
        return (
            f'{self.__class__.__name__}("{self._file.filename}", '
            f'mode="{self.mode.name}")'
        )

    def close(self) -> None:
        # TODO: Shorten dynamic arrays to correct size
        if self._close:
            self._file.close()

    def _get_hdf_path(self, loc: Sequence[str]) -> str:
        return "/" + "/".join(loc)

    def _get_parent(
        self,
        loc: Sequence[str],
        *,
        create_groups: bool = True,
        check_write: bool = False,
    ) -> Tuple[h5py.Group, str]:
        path, name = loc[:-1], loc[-1]
        if create_groups:
            # creat
            parent = self._file
            for part in path:
                try:
                    parent = parent[part]
                except KeyError:
                    if self.mode.insert:
                        parent = parent.create_group(part)
                    else:
                        raise AccessError(f"Cannot create group `{', '.join(loc)}`")
        else:
            parent = self._file[self._get_hdf_path(path)]

        if check_write and not self.mode.overwrite and name in parent:
            raise RuntimeError(f"Overwriting `{', '.join(loc)}` disabled")

        return parent, name

    def __getitem__(self, loc: Sequence[str]) -> Any:
        if len(loc) == 0:
            return self._file
        else:
            parent, name = self._get_parent(loc)
            return parent[name]

    def keys(self, loc: Optional[Sequence[str]] = None) -> List[str]:
        if loc:
            return self[loc].keys()
        else:
            return self._file.keys()

    def is_group(self, loc: Sequence[str]) -> bool:
        item = self[loc]
        if isinstance(item, h5py.Group):
            return "__class__" not in item.attrs
        else:
            return False

    def _create_group(self, loc: Sequence[str]):
        parent, name = self._get_parent(loc, check_write=True)
        return parent.create_group(name)

    def _read_attrs(self, loc: Sequence[str]) -> Attrs:
        return self[loc].attrs

    def _write_attr(self, loc: Sequence[str], name: str, value) -> None:
        self[loc].attrs[name] = value

    def _read_array(
        self, loc: Sequence[str], *, index: Optional[int] = None
    ) -> np.ndarray:
        if index is None:
            arr = self[loc]
        else:
            arr = self[loc][index]

        if self._read_attrs(loc).get("__pickled__", False):
            return decode_binary(arr[()].tobytes())
        else:
            return arr

    def _write_array(self, loc: Sequence[str], arr: np.ndarray) -> None:
        parent, name = self._get_parent(loc, check_write=True)

        if arr.dtype == object:
            arr_str = encode_binary(arr, binary=True)
            dataset = parent.create_dataset(name, data=np.void(arr_str))
            dataset.attrs["__pickled__"] = encode_attr(True)
        else:
            args = {"compression": "gzip"} if self.compression else {}
            parent.create_dataset(name, data=arr, **args)

    def _create_dynamic_array(
        self, loc: Sequence[str], shape: Tuple[int, ...], dtype: DTypeLike
    ) -> None:
        parent, name = self._get_parent(loc, check_write=True)
        if dtype == object:
            dataset = parent.create_dataset(
                name, shape=(1,) + shape, maxshape=(None,) + shape, dtype=np.void
            )
            dataset.attrs["__pickled__"] = encode_attr(True)
        else:
            args = {"compression": "gzip"} if self.compression else {}
            parent.create_dataset(
                name, shape=(1,) + shape, maxshape=(None,) + shape, dtype=dtype, **args
            )
        self._dynamic_array_size[self._get_hdf_path(loc)] = 0

    def _extend_dynamic_array(self, loc: Sequence[str], arr: ArrayLike) -> None:
        hdf_path = self._get_hdf_path(loc)
        dataset = self._file[hdf_path]
        size0 = self._dynamic_array_size[hdf_path]
        if size0 >= dataset.shape[0]:
            # need to resize array
            dataset.resize(size0 + 10, axis=0)

        if dataset.attrs.get("__pickled__", False):
            arr_str = encode_binary(arr, binary=True)
            dataset[size0] = np.void(arr_str)
        else:
            dataset[size0] = arr

        self._dynamic_array_size[hdf_path] += 1
