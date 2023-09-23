"""
Class that describe a state of multiple numpy arrays

.. codeauthor:: David Zwicker <david.zwicker@ds.mpg.de>
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np

from ..storage import Location, StorageGroup
from .base import StateBase


class ArrayCollectionState(StateBase):
    """State characterized by a multiple numpy array as the payload `data`

    The arrays are given as a tuple of :class:`~numpy.ndarray`. Optionally, a sequence
    of labels can be supplied to refer to the arrays by convenient names. If the labels
    are omitted, default labels using a string of the respective index of the array in
    the sequence are generated.
    """

    def __init__(
        self,
        data: Optional[Tuple[np.ndarray, ...]] = None,
        *,
        labels: Optional[Sequence[str]] = None,
    ):
        """
        Args:
            data: The arrays describing this collection
            labels (sequence of str): Optional list of names for all arrays
        """
        if data is None:
            self._state_data = tuple()
        else:
            self._state_data = tuple(data)

        self.labels = labels  # type: ignore

    @property
    def labels(self) -> List[str]:
        """list: the label assigned to each array"""
        labels = getattr(self, "_labels", None)
        if labels is None:
            return [str(i) for i in range(len(self))]
        else:
            return list(labels)

    @labels.setter
    def labels(self, labels: Sequence[str]) -> None:
        """set the labels assigned to each array"""
        num_arrays = len(self)
        if labels is None:
            self._labels = [str(i) for i in range(num_arrays)]
        else:
            assert num_arrays == len(labels) == len(set(labels))
            self._labels = list(labels)  # type: ignore

    @property
    def _state_attributes(self) -> Dict[str, Any]:
        """dict: Additional attributes, which are required to restore the state"""
        attributes = super()._state_attributes
        attributes["labels"] = self.labels
        return attributes

    @_state_attributes.setter
    def _state_attributes(self, attributes: Dict[str, Any]) -> None:
        """dict: Additional attributes, which are required to restore the state"""
        self.labels = attributes.pop("labels", None)
        super(ArrayCollectionState, ArrayCollectionState)._state_attributes.__set__(self, attributes)  # type: ignore

    @classmethod
    def from_data(cls, attributes: Dict[str, Any], data=None):
        """create instance from attributes and data

        Args:
            attributes (dict): Additional (unserialized) attributes
            data: The data of the degerees of freedom of the physical system
        """
        if data is not None:
            data = tuple(np.asarray(subdata) for subdata in data)
        return super().from_data(attributes, data)

    def __len__(self) -> int:
        return len(self._state_data)

    def __getitem__(self, index: Union[int, str]) -> np.ndarray:
        if isinstance(index, str):
            return self._state_data[self.labels.index(index)]  # type: ignore
        elif isinstance(index, int):
            return self._state_data[index]  # type: ignore
        else:
            raise TypeError

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
        group = StorageGroup(storage, loc)
        data = tuple(group.read_array(label, index=index) for label in attrs["labels"])
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
        group = StorageGroup(storage, loc)
        for label, data_arr in zip(self.labels, self._state_data):
            group.read_array(label, index=index, out=data_arr)

    def _state_write_to_storage(self, storage: StorageGroup, loc: Location) -> None:
        group = storage.create_group(
            loc, cls=self.__class__, attrs=self._state_attributes_store
        )
        """write the state to storage

        Args:
            storage (:class:`StorageGroup`):
                A storage opened with :func:`~modelrunner.storage.open_storage`
            loc (str or list of str):
                The location in the storage where the state is written
        """

        for sublabel, substate in zip(self.labels, self._state_data_store):
            group.write_array(sublabel, substate)

    def _state_create_trajectory(self, storage: StorageGroup, loc: Location) -> None:
        """prepare a trajectory of the current state

        Args:
            storage (:class:`StorageGroup`):
                A storage opened with :func:`~modelrunner.storage.open_storage`
            loc (str or list of str):
                The location in the storage where the trajectory is written
        """
        group = storage.create_group(
            loc, cls=self.__class__, attrs=self._state_attributes_store
        )

        for sublabel, subdata in zip(self.labels, self._state_data_store):
            group.create_dynamic_array(
                sublabel, shape=subdata.shape, dtype=subdata.dtype
            )

    def _state_append_to_trajectory(self, storage: StorageGroup, loc: Location) -> None:
        """append the current state to a prepared trajectory

        Args:
            storage (:class:`StorageGroup`):
                A storage opened with :func:`~modelrunner.storage.open_storage`
            loc (str or list of str):
                The location in the storage where the trajectory is written
        """
        group = StorageGroup(storage, loc)
        for label, subdata in zip(self.labels, self._state_data_store):
            group.extend_dynamic_array(label, subdata)
