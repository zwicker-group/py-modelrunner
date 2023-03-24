"""
Classes describing state of a single numpy array

.. codeauthor:: David Zwicker <david.zwicker@ds.mpg.de>
"""

from __future__ import annotations

import codecs
import pickle
from typing import Any, Dict, Optional

import numpy as np
import zarr
from numpy.lib.recfunctions import (
    structured_to_unstructured,
    unstructured_to_structured,
)

from .base import NoData, StateBase
from .io import zarrElement


class ArrayState(StateBase):
    """State characterized by a single numpy array as the payload `data`"""

    def __init__(self, data: np.ndarray):
        """
        Args:
            data: The data describing the state
        """
        self._state_data = data

    @property
    def _is_structured_array(self) -> bool:
        """bool: property determining whether this array is a structured array"""
        return self._state_data.dtype.names is not None

    @property
    def _state_attributes_store(self) -> Dict[str, Any]:
        """dict: Attributes in the form in which they will be written to storage

        This property modifies the normal `_state_attributes` and adds information
        necessary for restoring the class using :meth:`StateBase.from_data`.
        """
        attrs = super()._state_attributes_store

        if self._is_structured_array:
            # store dtype for structured arrays
            dtype_pickled = pickle.dumps(self._state_data.dtype)
            attrs["__dtype_pickled__"] = codecs.encode(dtype_pickled, "base64").decode()
        if isinstance(self._state_data, np.recarray):
            attrs["__record_array__"] = True

        return attrs

    @property
    def _state_data_store(self) -> Any:
        """determines what data is stored in this state"""
        if self._is_structured_array:
            # we store a structured array as an unstructured one
            return structured_to_unstructured(self._state_data)
        else:
            return self._state_data

    def _state_init(self, attributes: Dict[str, Any], data=NoData) -> None:
        """initialize the state with attributes and (optionally) data

        Args:
            attributes (dict): Additional (unserialized) attributes
            data: The data of the degerees of freedom of the physical system
        """
        if data is not NoData:
            # set the array if any data was given
            data = np.asarray(data)

            if "__dtype_pickled__" in attributes:
                # the prescence of this attribute signals that data is a record array
                # the attribute then contains information about the dtype, which we need
                # to reconstruct or at least check
                dtype_pickled = attributes.pop("__dtype_pickled__")
                dtype = pickle.loads(codecs.decode(dtype_pickled.encode(), "base64"))
                if data.dtype.names is None:
                    data = unstructured_to_structured(data, dtype=dtype)
                elif dtype != data.dtype:
                    raise AssertionError(f"{dtype} != {data.dtype}")

            if attributes.pop("__record_array__", False):
                # the array was a record array
                data = data.view(np.recarray)

        super()._state_init(attributes, data)

    @classmethod
    def _state_read_zarr_data(cls, zarr_element: zarr.Array, *, index=...):
        return zarr_element[index]

    def _state_update_from_zarr(self, element: zarrElement, *, index=...) -> None:
        self._state_data[:] = element[index]

    def _state_write_zarr_data(  # type: ignore
        self,
        zarr_group: zarr.Group,
        *,
        label: str = "data",
    ) -> zarr.Array:
        return zarr_group.array(label, self._state_data)

    def _state_prepare_zarr_trajectory(
        self,
        zarr_group: zarr.Group,
        attrs: Optional[Dict[str, Any]] = None,
        *,
        label: str = "data",
        **kwargs,
    ) -> zarr.Array:
        """prepare the zarr storage for this state"""
        data = self._state_data
        zarr_element = zarr_group.zeros(
            label, shape=(0,) + data.shape, chunks=(1,) + data.shape, dtype=data.dtype
        )
        self._state_write_zarr_attributes(zarr_element, attrs)

        return zarr_element

    def _state_append_to_zarr_trajectory(self, zarr_element: zarr.Array) -> None:
        """append current data to a stored element"""
        zarr_element.append([self._state_data])
