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

    encode_internal_attrs: bool = False
    """bool: Flag determining whether flags used by this class internally are encoded as
    a string. This can be important if the data is stored to a file later"""

    _data: Attrs

    # TODO: Write arrays and objects as true python objects
    # Move current methods to TextBase. However, we still need to keep track of a way
    # to store attributes

    def __init__(self, *, mode: ModeType = "insert"):
        """
        Args:
            mode (str or :class:`~modelrunner.storage.access_modes.AccessMode`):
                The file mode with which the storage is accessed. Determines allowed
                operations.
        """
        super().__init__(mode=mode)
        self._data = {}

    def _encode_internal_attr(self, attr: Any) -> Any:
        if self.encode_internal_attrs:
            return encode_attr(attr)
        else:
            return attr

    def _decode_internal_attr(self, attr: Any) -> Any:
        if self.encode_internal_attrs:
            return decode_attr(attr)
        else:
            return attr

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
        """get the parent group for a particular location

        Args:
            loc (list of str):
                The location in the storage where the group will be created
            check_write (bool):
                Check whether the parent group is writable if `True`

        Returns:
            (group, str):
                A tuple consisting of the parent group and the name of the current item
        """
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
            # dictionaries are usually groups, unless they have the `type` entry
            return "__type__" not in item
        else:
            return False  # no group, since it's not a dictionary

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
        # check whether we actually have an array here
        if self[loc]["__type__"] not in {"array", "dynamic_array"}:
            raise RuntimeError(f"Found `{self[loc]['__type__']}` at {'/'.join(loc)}")

        # read the data from the location
        if index is None:
            arr = self[loc]["data"]
        else:
            arr = self[loc]["data"][index]

        if hasattr(arr, "__iter__"):  # minimal sanity check
            dtype = self._decode_internal_attr(self[loc]["dtype"])
            if dtype.names is not None:
                arr = unstructured_to_structured(np.asarray(arr), dtype=dtype)
            else:
                arr = np.array(arr, dtype=dtype)
            if self[loc].get("record_array", False):
                arr = arr.view(np.recarray)
            return arr  # type: ignore
        else:
            raise RuntimeError(f"No array at {'/'.join(loc)}")

    def _write_array(self, loc: Sequence[str], arr: np.ndarray) -> None:
        parent, name = self._get_parent(loc, check_write=True)

        dtype = arr.dtype  # extract dtype here since `arr` is changed later
        if dtype.names is not None:
            # structured array
            arr = structured_to_unstructured(arr)

        parent[name] = {
            "__type__": "array",
            "data": np.array(arr, copy=True),
            "dtype": self._encode_internal_attr(dtype),
        }
        if isinstance(arr, np.record):
            parent[name]["record_array"] = True

    def _create_dynamic_array(
        self,
        loc: Sequence[str],
        shape: Tuple[int, ...],
        dtype: DTypeLike,
        *,
        record_array: bool = False,
    ) -> None:
        parent, name = self._get_parent(loc, check_write=True)
        if name in parent:
            raise RuntimeError(f"Array `{'/'.join(loc)}` already exists")
        parent[name] = {
            "__type__": "dynamic_array",
            "data": [],
            "shape": shape,
            "dtype": self._encode_internal_attr(np.dtype(dtype)),
        }
        if record_array:
            parent[name]["record_array"] = True

    def _extend_dynamic_array(self, loc: Sequence[str], arr: ArrayLike) -> None:
        item = self[loc]
        # check whether we actually have an array here
        if item["__type__"] != "dynamic_array":
            raise RuntimeError(f"Found `{self[loc]['__type__']}` at {'/'.join(loc)}")

        # check data shape that is stored at this position
        data = np.asanyarray(arr)
        stored_shape = tuple(item["shape"])
        if stored_shape != data.shape:
            raise TypeError(f"Shape mismatch ({stored_shape} != {data.shape})")

        # convert the data to the correct format
        stored_dtype = self._decode_internal_attr(item["dtype"])
        if not np.issubdtype(data.dtype, stored_dtype):
            raise TypeError(f"Dtype mismatch ({data.dtype} != {stored_dtype}")
        if data.dtype.names is not None:
            # structured array
            data = structured_to_unstructured(data)

        # append the data to the dynamic array
        if data.ndim == 0:
            item["data"].append(data.item())
        else:
            item["data"].append(np.array(data, copy=True))
