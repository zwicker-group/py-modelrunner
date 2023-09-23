"""
Classes that describe the state of a simulation as a single python object

.. codeauthor:: David Zwicker <david.zwicker@ds.mpg.de>
"""

from __future__ import annotations

from typing import Any, Optional

import numpy as np

from ..storage import Location, StorageGroup
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
        cls, storage: StorageGroup, loc: Location, *, index: Optional[int] = None
    ):
        """create the state from storage

        Args:
            storage (:class:`StorageGroup`):
                A storage opened with :func:`~modelrunner.storage.open_storage`
            loc (str or list of str):
                The location in the storage where the state is read
            index (int, optional):
                If the location contains a trajectory of the state, `index` must denote
                the index determining which state should be created
        """
        attrs = cls._state_get_attrs_from_storage(storage, loc, check_version=True)

        # create the state from the read data
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

        obj = cls.__new__(cls)
        obj._state_init(attrs, data)
        return obj

    def _state_update_from_stored_data(
        self, storage: StorageGroup, loc: Location, index: Optional[int] = None
    ) -> None:
        """update the state data (but not its attributes) from storage

        Args:
            storage (:class:`StorageGroup`):
                A storage opened with :func:`~modelrunner.storage.open_storage`
            loc (str or list of str):
                The location in the storage where the state is read
            index (int, optional):
                If the location contains a trajectory of the state, `index` must denote
                the index determining which state should be created
        """
        self._state_data = storage.read_array(loc, index=index).item()

    def _state_write_to_storage(self, storage: StorageGroup, loc: Location) -> None:
        """write the state to storage

        Args:
            storage (:class:`StorageGroup`):
                A storage opened with :func:`~modelrunner.storage.open_storage`
            loc (str or list of str):
                The location in the storage where the state is written
        """
        # store the data in a single object array
        arr = np.empty(1, dtype=object)
        arr[0] = self._state_data_store
        attrs = self._state_attributes_store
        storage.write_array(loc, arr, attrs=attrs, cls=self.__class__)

    def _state_create_trajectory(self, storage: StorageGroup, loc: Location) -> None:
        """prepare a trajectory of the current state

        Args:
            storage (:class:`StorageGroup`):
                A storage opened with :func:`~modelrunner.storage.open_storage`
            loc (str or list of str):
                The location in the storage where the trajectory is written
        """
        attrs = self._state_attributes_store
        storage.create_dynamic_array(
            loc, shape=(1,), dtype=object, attrs=attrs, cls=self.__class__
        )

    def _state_append_to_trajectory(self, storage: StorageGroup, loc: Location) -> None:
        """append the current state to a prepared trajectory

        Args:
            storage (:class:`StorageGroup`):
                A storage opened with :func:`~modelrunner.storage.open_storage`
            loc (str or list of str):
                The location in the storage where the trajectory is written
        """
        arr = np.empty(1, dtype=object)
        arr[0] = self._state_data_store
        storage.extend_dynamic_array(loc, arr)
