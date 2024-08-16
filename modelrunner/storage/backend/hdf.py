"""Defines a class storing data on the file system using the hierarchical data format
(hdf)

Requires the optional :mod:`h5py` module.

.. codeauthor:: David Zwicker <david.zwicker@ds.mpg.de>
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Collection, Sequence

import h5py
import numpy as np
from numpy.typing import ArrayLike, DTypeLike

from ..access_modes import AccessError, ModeType
from ..attributes import AttrsLike, encode_attr
from ..base import StorageBase
from ..utils import decode_binary, encode_binary


class HDFStorage(StorageBase):
    """Storage that stores data in an HDF file."""

    extensions = ["h5", "hdf", "hdf5"]

    def __init__(
        self,
        file_or_path: str | Path | h5py.File,
        *,
        mode: ModeType = "read",
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
        self._dynamic_array_size: dict[str, int] = {}  # lengths of the dynamic arrays

        if isinstance(file_or_path, (str, Path)):
            # open HDF storage on file system
            self._close = True
            file_mode = self.mode.file_mode
            self._file = h5py.File(file_or_path, mode=file_mode)

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
        super().close()

    def _get_hdf_path(self, loc: Sequence[str]) -> str:
        return "/" + "/".join(loc)

    def _get_parent(
        self, loc: Sequence[str], *, create_groups: bool = True
    ) -> tuple[h5py.Group, str]:
        """Get the parent group for a particular location.

        Args:
            loc (list of str):
                The location in the storage where the group will be created
            create_groups (bool):
                Create all intermediate groups if they not already exist

        Returns:
            (group, str):
                A tuple consisting of the parent group and the name of the current item
        """
        try:
            path, name = loc[:-1], loc[-1]
        except IndexError as err:
            raise KeyError(f"Location `/{'/'.join(loc)}` has no parent") from err

        if create_groups:
            # creat
            parent = self._file
            for part in path:
                try:
                    parent = parent[part]
                except KeyError as err:
                    if self.mode.insert:
                        parent = parent.create_group(part)
                    else:
                        raise AccessError(
                            f"Cannot create group `/{'/'.join(loc)}`"
                        ) from err
        else:
            parent = self._file[self._get_hdf_path(path)]

        return parent, name

    def __getitem__(self, loc: Sequence[str]) -> Any:
        if len(loc) == 0:
            return self._file
        else:
            parent, name = self._get_parent(loc)
            try:
                return parent[name]
            except ValueError as e:
                raise ValueError(
                    f"Invalid location `{name}` in path `{parent.name}`"
                ) from e

    def keys(self, loc: Sequence[str] | None = None) -> Collection[str]:
        if loc:
            return self[loc].keys()  # type: ignore
        else:
            return self._file.keys()  # type: ignore

    def is_group(self, loc: Sequence[str]) -> bool:
        return isinstance(self[loc], h5py.Group)

    def _create_group(self, loc: Sequence[str]):
        parent, name = self._get_parent(loc)
        try:
            return parent.create_group(name)
        except ValueError as e:
            raise ValueError(f"Cannot create group `{name}`") from e

    def _read_attrs(self, loc: Sequence[str]) -> AttrsLike:
        return self[loc].attrs  # type: ignore

    def _write_attr(self, loc: Sequence[str], name: str, value: str) -> None:
        self[loc].attrs[name] = value

    def _read_array(
        self, loc: Sequence[str], *, copy: bool, index: int | None = None
    ) -> np.ndarray:
        if index is None:
            arr_like = self[loc]
        else:
            arr_like = self[loc][index]

        # decode potentially binary data
        attrs = self._read_attrs(loc)
        if attrs.get("__pickled__", False):
            # data has been pickled inside the array
            if np.issubdtype(arr_like.dtype, "O"):
                # array of object dtype
                arr_like = np.frompyfunc(decode_binary, nin=1, nout=1)(arr_like)
            elif np.issubdtype(arr_like.dtype, np.uint8):
                arr_like = decode_binary(arr_like)
            else:
                data = np.asarray(arr_like).item()
                arr_like = decode_binary(data)

        elif not isinstance(arr_like, (h5py.Dataset, np.ndarray, np.generic)):
            raise RuntimeError(
                f"Found {arr_like.__class__} at location `/{'/'.join(loc)}`"
            )

        # convert it into the right type
        arr = np.array(arr_like, copy=copy)
        if attrs.get("__recarray__", False):
            arr = arr.view(np.recarray)

        return arr

    def _write_array(self, loc: Sequence[str], arr: np.ndarray) -> None:
        parent, name = self._get_parent(loc)

        if name in parent:
            # update an existing array assuming it has the same shape. The credentials
            # for this operation need to be checked by the caller!
            dataset = parent[name]
            if dataset.attrs.get("__pickled__", None) == encode_attr(True):
                arr_bin = encode_binary(arr, binary=True)
                assert isinstance(arr_bin, bytes)
                dataset[...] = np.void(arr_bin)
            else:
                dataset[...] = arr

        else:
            # create a new data set
            if arr.dtype == object:
                arr_bin = encode_binary(arr, binary=True)
                assert isinstance(arr_bin, bytes)
                dataset = parent.create_dataset(name, data=np.void(arr_bin))
                dataset.attrs["__pickled__"] = True
            else:
                args = {"compression": "gzip"} if self.compression else {}
                dataset = parent.create_dataset(name, data=arr, **args)

            if isinstance(arr, np.recarray):
                dataset.attrs["__recarray__"] = True

    def _create_dynamic_array(
        self,
        loc: Sequence[str],
        shape: tuple[int, ...],
        *,
        dtype: DTypeLike,
        record_array: bool = False,
    ) -> None:
        parent, name = self._get_parent(loc)
        if np.issubdtype(dtype, "O"):
            try:
                dataset = parent.create_dataset(
                    name,
                    shape=(1,) + shape,
                    maxshape=(None,) + shape,
                    dtype=h5py.vlen_dtype(np.uint8),
                )
            except ValueError as err:
                raise RuntimeError(f"Array `{'/'.join(loc)}` already exists") from err
            dataset.attrs["__pickled__"] = encode_attr(True)

        else:
            args = {"compression": "gzip"} if self.compression else {}
            try:
                dataset = parent.create_dataset(
                    name,
                    shape=(1,) + shape,
                    maxshape=(None,) + shape,
                    dtype=dtype,
                    **args,
                )
            except ValueError as err:
                raise RuntimeError(f"Array `/{'/'.join(loc)}` already exists") from err
        self._dynamic_array_size[self._get_hdf_path(loc)] = 0

        if record_array:
            dataset.attrs["__recarray__"] = True

    def _extend_dynamic_array(self, loc: Sequence[str], arr: ArrayLike) -> None:
        # load the dataset
        hdf_path = self._get_hdf_path(loc)
        dataset = self._file[hdf_path]

        if not dataset.maxshape[0] == None:
            raise RuntimeError(f"Array `/{'/'.join(loc)}` is not resizeable")

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
            assert isinstance(arr_bin, bytes)
            dataset[size] = np.frombuffer(arr_bin, dtype=np.uint8)
        else:
            dataset[size] = arr

        self._dynamic_array_size[hdf_path] = dataset.shape[0]

    def _read_object(self, loc: Sequence[str]) -> Any:
        return decode_binary(np.asarray(self[loc]).item())

    def _write_object(self, loc: Sequence[str], obj: Any) -> None:
        parent, name = self._get_parent(loc)
        arr_str = encode_binary(obj, binary=True)
        if name in parent:
            del parent[name]  # delete old dataset
        parent.create_dataset(name, data=np.void(arr_str))
