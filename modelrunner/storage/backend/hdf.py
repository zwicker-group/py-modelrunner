"""
Defines a class storing data on the file system using the hierarchical data format (hdf)

Requires the optional :mod:`h5py` module.

.. codeauthor:: David Zwicker <david.zwicker@ds.mpg.de> 
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Collection, Dict, Optional, Sequence, Tuple, Union

import h5py
import numpy as np
from numpy.typing import ArrayLike, DTypeLike

from ..access_modes import AccessError, ModeType
from ..attributes import AttrsLike, encode_attr
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
        # shorten dynamic arrays to correct size
        for hdf_path, size in self._dynamic_array_size.items():
            self._file[hdf_path].resize(size, axis=0)

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
        """get the parent group for a particular location

        Args:
            loc (list of str):
                The location in the storage where the group will be created
            create_groups (bool):
                Create all intermediate groups if they not already exist
            check_write (bool):
                Check whether the parent group is writable if `True`

        Returns:
            (group, str):
                A tuple consisting of the parent group and the name of the current item
        """
        try:
            path, name = loc[:-1], loc[-1]
        except IndexError:
            raise KeyError(f"Location `{'/'.join(loc)}` has no parent")

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
            raise AccessError(f"Overwriting `{', '.join(loc)}` disabled")

        return parent, name

    def __getitem__(self, loc: Sequence[str]) -> Any:
        if len(loc) == 0:
            return self._file
        else:
            parent, name = self._get_parent(loc)
            return parent[name]

    def keys(self, loc: Optional[Sequence[str]] = None) -> Collection[str]:
        if loc:
            return self[loc].keys()  # type: ignore
        else:
            return self._file.keys()  # type: ignore

    def is_group(self, loc: Sequence[str]) -> bool:
        item = self[loc]
        if isinstance(item, h5py.Group):
            return "__class__" not in item.attrs
        else:
            return False

    def _create_group(self, loc: Sequence[str]):
        parent, name = self._get_parent(loc, check_write=True)
        return parent.create_group(name)

    def _read_attrs(self, loc: Sequence[str]) -> AttrsLike:
        return self[loc].attrs  # type: ignore

    def _write_attr(self, loc: Sequence[str], name: str, value) -> None:
        self[loc].attrs[name] = value

    def _read_array(
        self, loc: Sequence[str], *, index: Optional[int] = None
    ) -> ArrayLike:
        if index is None:
            arr = self[loc]
        else:
            arr = self[loc][index]

        if self._read_attrs(loc).get("__pickled__", False):
            return decode_binary(np.asarray(arr).item())  # type: ignore
        elif isinstance(arr, (h5py.Dataset, np.ndarray, np.generic)):
            return arr  # type: ignore
        else:
            raise RuntimeError(f"Found {arr.__class__} at location `{'/'.join(loc)}`")

    def _write_array(self, loc: Sequence[str], arr: np.ndarray) -> None:
        parent, name = self._get_parent(loc, check_write=True)

        if name in parent:
            # update an existing array assuming it has the same shape. The credentials
            # for this operation need to be checked by the caller!
            dataset = parent[name]
            if dataset.attrs.get("__pickled__", None) == encode_attr(True):
                arr_str = encode_binary(arr, binary=True)
                dataset[...] = np.void(arr_str)
            else:
                dataset[...] = arr

        else:
            # create a new data set
            if arr.dtype == object:
                arr_str = encode_binary(arr, binary=True)
                dataset = parent.create_dataset(name, data=np.void(arr_str))
                dataset.attrs["__pickled__"] = encode_attr(True)
            else:
                args = {"compression": "gzip"} if self.compression else {}
                parent.create_dataset(name, data=arr, **args)

    def _create_dynamic_array(
        self,
        loc: Sequence[str],
        shape: Tuple[int, ...],
        *,
        dtype: DTypeLike,
        record_array: bool = False,
    ) -> None:
        parent, name = self._get_parent(loc, check_write=True)
        if dtype == object:
            dt = h5py.special_dtype(vlen=np.dtype("uint8"))
            try:
                dataset = parent.create_dataset(
                    name, shape=(1,) + shape, maxshape=(None,) + shape, dtype=dt
                )
            except ValueError:
                raise RuntimeError(f"Array `{'/'.join(loc)}` already exists")
            dataset.attrs["__pickled__"] = encode_attr(True)

        else:
            args = {"compression": "gzip"} if self.compression else {}
            try:
                parent.create_dataset(
                    name,
                    shape=(1,) + shape,
                    maxshape=(None,) + shape,
                    dtype=dtype,
                    **args,
                )
            except ValueError:
                raise RuntimeError(f"Array `{'/'.join(loc)}` already exists")
        self._dynamic_array_size[self._get_hdf_path(loc)] = 0

    def _extend_dynamic_array(self, loc: Sequence[str], arr: ArrayLike) -> None:
        # load the dataset
        hdf_path = self._get_hdf_path(loc)
        dataset = self._file[hdf_path]

        # determine size of the currently written data
        size = self._dynamic_array_size.get(hdf_path, None)
        if size is None:
            # we extend a dataset that has not been created by this instance. Assume
            # that it has the correct size
            size = dataset.shape[0]

        if dataset.shape[0] <= size:
            # the old data barely fits into the current size => We need to extend the
            # array to make space for an additional record. We directly extend by a bit
            # so we don't need to resize every iteration
            dataset.resize(size + 1, axis=0)

        if dataset.attrs.get("__pickled__", False):
            arr_bin = encode_binary(arr, binary=True)
            dataset[size] = np.frombuffer(arr_bin, dtype="uint8")
        else:
            dataset[size] = arr

        self._dynamic_array_size[hdf_path] = dataset.shape[0]
