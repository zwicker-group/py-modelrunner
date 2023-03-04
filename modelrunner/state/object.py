"""
Classes that describe the state of a simulation at a single point in time

Each state is defined by :attr:`attributes` and :attr:`data`. Attributes describe
general aspects about a state, which typically do not change, e.g., its `name`.
These classes define how data is read and written and they contain methods that can be
used to write multiple states of the same class to a file consecutively, e.g., to store
a trajectory. Here, it is assumed that the `attributes` do not change over time.

TODO:
- document the succession of calls for storing fields (to get a better idea of the
  available hooks)
- do the same for loading data

.. autosummary::
   :nosignatures:

   ObjectState
   ArrayState
   ArrayCollectionState
   DictState

.. codeauthor:: David Zwicker <david.zwicker@ds.mpg.de>
"""

from __future__ import annotations

from typing import Any, Dict, Optional

import numcodecs
import zarr

from .base import StateBase
from .io import zarrElement


class ObjectState(StateBase):
    """State characterized by a serializable python object

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
        self.data = data

    @classmethod
    def _read_zarr_data(cls, zarr_element: zarr.Array, *, index=...):
        if zarr_element.shape == () and index is ...:
            return zarr_element[index].item()
        else:
            return zarr_element[index]

    def _update_from_zarr(self, element: zarrElement, *, index=...) -> None:
        if element.shape == () and index is ...:
            self.data = element[index].item()
        else:
            self.data = element[index]

    def _write_zarr_data(  # type: ignore
        self,
        zarr_group: zarr.Group,
        *,
        label: str = "data",
        codec: Optional[numcodecs.abc.Codec] = None,
    ) -> zarrElement:
        if codec is None:
            codec = self.default_codec
        return zarr_group.array(
            label, self._data_store, shape=(0,), dtype=object, object_codec=codec
        )

    def _prepare_zarr_trajectory(  # type: ignore
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
        self._write_zarr_attributes(zarr_element, attrs)

        return zarr_element

    def _append_to_zarr_trajectory(self, zarr_element: zarr.Array) -> None:
        """append current data to a stored element"""
        zarr_element.resize(len(zarr_element) + 1)
        zarr_element[-1] = self._data_store
