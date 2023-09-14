"""
Classes that describe the state of a simulation as a single python object

.. codeauthor:: David Zwicker <david.zwicker@ds.mpg.de>
"""

from __future__ import annotations

from typing import Any, Optional, Sequence

import numpy as np

from ..storage import storage_actions
from .base import StateBase


class ObjectState(StateBase):
    """State characterized by a serializable python object as the payload `data`

    The data needs to be accessible from the :attr:`data` property of the instance.
    Additional attributes can be supplied via the :attr:`attribute` property, which will
    then be stored in files. To support reading such augmented states, the method
    :meth:`from_data` needs to be overwritten.
    """

    def __init__(self, data: Optional[Any] = None):
        """
        Args:
            data: The data describing the state
        """
        self._state_data = data

    @classmethod
    def _state_from_stored_data(
        cls, storage, loc: Sequence[str], index: Optional[int] = None
    ):
        obj = cls.__new__(cls)
        attrs = storage.read_attrs(loc)
        attrs.pop("__class__")
        attrs.pop("__version__", None)
        arr = storage.read_array(loc, index=index)
        if arr.size == 1:
            # revert the trick of storing the object in an object array with one item
            data = arr.item()
        elif arr.shape[0] == 1:
            # when reconstructing objects that are arrays, the trick above implies that
            # the actual data is stored in the rest of the dimensions
            data = arr[0]
        else:
            raise RuntimeError(f"Data has shape {arr.shape} instead of single item")
        obj._state_init(attrs, data)
        return obj

    def _state_update_from_stored_data(
        self, storage, loc: Sequence[str], index: Optional[int] = None
    ):
        self._state_data = storage.read_array(loc, index=index).item()

    def _state_write_to_storage(self, storage, loc: Sequence[str]):
        # store the data in a single object array
        arr = np.empty(1, dtype=object)
        arr[0] = self._state_data_store
        attrs = self._state_attributes_store
        return storage.write_array(loc, arr, attrs=attrs, cls=self.__class__)

    def _state_create_trajectory(self, storage, loc: Sequence[str]):
        """prepare the zarr storage for this state"""
        attrs = self._state_attributes_store
        storage.create_dynamic_array(
            loc, shape=(1,), dtype=object, attrs=attrs, cls=self.__class__
        )

    def _state_append_to_trajectory(self, storage, loc: Sequence[str]):
        arr = np.empty(1, dtype=object)
        arr[0] = self._state_data_store
        storage.extend_dynamic_array(loc, arr)


storage_actions.register(
    "read_object", ObjectState, ObjectState._state_from_stored_data
)
