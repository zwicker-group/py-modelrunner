"""
Classes that describe the state of a simulation as a single python object

.. codeauthor:: David Zwicker <david.zwicker@ds.mpg.de>
"""

from __future__ import annotations

from typing import Any, Optional, Sequence

import numcodecs
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

    codec = numcodecs.Pickle()

    def __init__(self, data: Optional[Any] = None):
        """
        Args:
            data: The data describing the state
        """
        self._state_data = data

    @classmethod
    def _state_from_stored_data(cls, storage, key: str, index: Optional[int] = None):
        obj = cls.__new__(cls)
        attrs = storage.read_attrs(key, copy=True)
        attrs.pop("__class__")
        attrs.pop("__version__", None)

        arr = storage.read_array(key, index=index).item()
        if "__codec__" in attrs:
            codec = numcodecs.get_codec(attrs.pop("__codec__"))
        else:
            codec = self.codec
        data = codec.decode(arr)
        obj._state_init(attrs, data)
        return obj

    def _state_update_from_stored_data(
        self, storage, key: str, index: Optional[int] = None
    ):
        attrs = storage.read_attrs(key, copy=False)
        if "__codec__" in attrs:
            codec = numcodecs.get_codec(attrs.pop("__codec__"))
        else:
            codec = self.codec
        arr = storage.read_array(key, index=index).item()
        self._state_data = codec.decode(arr)

    def _state_write_to_storage(self, storage, key: Sequence[str]):
        data = self.codec.encode(self._state_data)
        arr = np.array(data, dtype=object)

        attrs = self._state_attributes_store
        attrs["__codec__"] = self.codec.get_config()
        return storage.write_array(key, arr, attrs=attrs, cls=self.__class__)

    def _state_create_trajectory(self, storage, key: str):
        """prepare the zarr storage for this state"""
        attrs = self._state_attributes_store
        attrs["__codec__"] = self.codec.get_config()
        storage.create_dynamic_array(
            key, shape=tuple(), dtype=object, attrs=attrs, cls=self.__class__
        )

    def _state_append_to_trajectory(self, storage, key: str):
        data = self.codec.encode(self._state_data)
        arr = np.array(data, dtype=object)
        storage.extend_dynamic_array(key, arr)


storage_actions.register(
    "read_object", ObjectState, ObjectState._state_from_stored_data
)
