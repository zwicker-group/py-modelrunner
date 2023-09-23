"""
Classes describing state of a single numpy array

.. codeauthor:: David Zwicker <david.zwicker@ds.mpg.de>
"""

from __future__ import annotations

import codecs
import pickle
from typing import Any, Dict, Optional

import numpy as np
from numpy.lib.recfunctions import (
    structured_to_unstructured,
    unstructured_to_structured,
)

from ..storage import Location, StorageGroup
from .base import NoData, StateBase


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
        data = storage.read_array(loc, index=index)
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
        storage.read_array(loc, index=index, out=self._state_data)

    def _state_write_to_storage(self, storage: StorageGroup, loc: Location) -> None:
        """write the state to storage

        Args:
            storage (:class:`StorageGroup`):
                A storage opened with :func:`~modelrunner.storage.open_storage`
            loc (str or list of str):
                The location in the storage where the state is written
        """
        storage.write_array(
            loc,
            self._state_data_store,
            attrs=self._state_attributes_store,
            cls=self.__class__,
        )

    def _state_create_trajectory(self, storage: StorageGroup, loc: Location) -> None:
        """prepare a trajectory of the current state

        Args:
            storage (:class:`StorageGroup`):
                A storage opened with :func:`~modelrunner.storage.open_storage`
            loc (str or list of str):
                The location in the storage where the trajectory is written
        """
        data = self._state_data
        storage.create_dynamic_array(
            loc,
            shape=data.shape,
            dtype=data.dtype,
            attrs=self._state_attributes_store,
            cls=self.__class__,
        )

    def _state_append_to_trajectory(self, storage: StorageGroup, loc: Location) -> None:
        """append the current state to a prepared trajectory

        Args:
            storage (:class:`StorageGroup`):
                A storage opened with :func:`~modelrunner.storage.open_storage`
            loc (str or list of str):
                The location in the storage where the trajectory is written
        """
        storage.extend_dynamic_array(loc, self._state_data)
