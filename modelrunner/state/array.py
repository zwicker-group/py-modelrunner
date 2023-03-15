"""
Classes describing state of a single numpy array

.. codeauthor:: David Zwicker <david.zwicker@ds.mpg.de>
"""

from __future__ import annotations

from typing import Any, Dict, Optional

import numpy as np
import zarr

from .base import StateBase
from .io import zarrElement


class ArrayState(StateBase):
    """State characterized by a single numpy array"""

    _data_attribute: str = "data"

    def __init__(self, data: np.ndarray):
        """
        Args:
            data: The data describing the state
        """
        setattr(self, self._data_attribute, data)

    @classmethod
    def from_data(cls, attributes: Dict[str, Any], data=None):
        """create instance from attributes and data

        Args:
            attributes (dict): Additional (unserialized) attributes
            data: The data of the degerees of freedom of the physical system
        """
        if data is not None:
            data = np.asarray(data)
        return super().from_data(attributes, data)

    @classmethod
    def _read_zarr_data(cls, zarr_element: zarr.Array, *, index=...):
        return zarr_element[index]

    def _update_from_zarr(self, element: zarrElement, *, index=...) -> None:
        data = getattr(self, self._data_attribute)
        data[:] = element[index]

    def _write_zarr_data(  # type: ignore
        self,
        zarr_group: zarr.Group,
        *,
        label: str = "data",
    ) -> zarr.Array:
        data = getattr(self, self._data_attribute)
        return zarr_group.array(label, data)

    def _prepare_zarr_trajectory(
        self,
        zarr_group: zarr.Group,
        attrs: Optional[Dict[str, Any]] = None,
        *,
        label: str = "data",
        **kwargs,
    ) -> zarr.Array:
        """prepare the zarr storage for this state"""
        data = getattr(self, self._data_attribute)
        zarr_element = zarr_group.zeros(
            label, shape=(0,) + data.shape, chunks=(1,) + data.shape, dtype=data.dtype
        )
        self._write_zarr_attributes(zarr_element, attrs)

        return zarr_element

    def _append_to_zarr_trajectory(self, zarr_element: zarr.Array) -> None:
        """append current data to a stored element"""
        data = getattr(self, self._data_attribute)
        zarr_element.append([data])
