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
    def _state_from_stored_data(cls, storage, key: str, index: Optional[int] = None):
        obj = cls.__new__(cls)
        attrs = storage.read_attrs(key)
        attrs.pop("__class__")
        attrs.pop("__version__", None)
        print("SHAPE", storage.read_array(key).shape)
        print("INDEX", index)
        arr = storage.read_array(key, index=index)
        if arr.size == 1:
            data = arr.item()
        else:
            raise RuntimeError(f"Data has shape {arr.shape} instead of single item")
        obj._state_init(attrs, data)
        return obj

    def _state_update_from_stored_data(
        self, storage, key: str, index: Optional[int] = None
    ):
        self._state_data = storage.read_array(key, index=index).item()

    def _state_write_to_storage(self, storage, key: Sequence[str]):
        # store the data in a single object array
        arr = np.empty(1, dtype=object)
        arr[0] = self._state_data_store
        attrs = self._state_attributes_store
        return storage.write_array(key, arr, attrs=attrs, cls=self.__class__)

    def _state_create_trajectory(self, storage, key: str):
        """prepare the zarr storage for this state"""
        attrs = self._state_attributes_store
        storage.create_dynamic_array(
            key, shape=(1,), dtype=object, attrs=attrs, cls=self.__class__
        )

    def _state_append_to_trajectory(self, storage, key: str):
        arr = np.empty(1, dtype=object)
        arr[0] = self._state_data_store
        storage.extend_dynamic_array(key, arr)


storage_actions.register(
    "read_object", ObjectState, ObjectState._state_from_stored_data
)
