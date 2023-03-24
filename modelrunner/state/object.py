"""
Classes that describe the state of a simulation as a single python object

.. codeauthor:: David Zwicker <david.zwicker@ds.mpg.de>
"""

from __future__ import annotations

from typing import Any, Dict, Optional

import numcodecs
import zarr

from .base import StateBase
from .io import zarrElement


class ObjectState(StateBase):
    """State characterized by a serializable python object as the payload `data`

    The data needs to be accessible from the :attr:`data` property of the instance.
    Additional attributes can be supplied via the :attr:`attribute` property, which will
    then be stored in files. To support reading such augmented states, the method
    :meth:`from_data` needs to be overwritten.
    """

    default_codec = numcodecs.Pickle()

    def __init__(self, data: Optional[Any] = None):
        """
        Args:
            data: The data describing the state
        """
        self._state_data = data

    @classmethod
    def _state_read_zarr_data(cls, zarr_element: zarr.Array, *, index=...):
        if zarr_element.shape == () and index is ...:
            return zarr_element[index].item()
        else:
            return zarr_element[index]

    def _state_update_from_zarr(self, element: zarrElement, *, index=...) -> None:
        if element.shape == () and index is ...:
            self._state_data = element[index].item()
        else:
            self._state_data = element[index]

    def _state_write_zarr_data(  # type: ignore
        self,
        zarr_group: zarr.Group,
        *,
        label: str = "data",
        codec: Optional[numcodecs.abc.Codec] = None,
    ) -> zarrElement:
        if codec is None:
            codec = self.default_codec
        return zarr_group.array(
            label, self._state_data, shape=(0,), dtype=object, object_codec=codec
        )

    def _state_prepare_zarr_trajectory(  # type: ignore
        self,
        zarr_group: zarr.Group,
        attrs: Optional[Dict[str, Any]] = None,
        *,
        label: str = "data",
        codec: Optional[numcodecs.abc.Codec] = None,
    ) -> zarr.Array:
        """prepare the zarr storage for this state"""
        if codec is None:
            codec = self.default_codec

        zarr_element = zarr_group.zeros(
            label,
            shape=(0,),
            chunks=(1,),
            dtype=object,
            object_codec=codec,
        )
        self._state_write_zarr_attributes(zarr_element, attrs)

        return zarr_element

    def _state_append_to_zarr_trajectory(self, zarr_element: zarr.Array) -> None:
        """append current data to a stored element"""
        zarr_element.resize(len(zarr_element) + 1)
        zarr_element[-1] = self._state_data
