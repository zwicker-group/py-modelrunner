"""
Classes that describe the state of a simulation using a dictionary of states

.. codeauthor:: David Zwicker <david.zwicker@ds.mpg.de>
"""

from __future__ import annotations

import itertools
from typing import Any, Dict, Literal, Optional, Sequence, Union

from ..storage import Location, StorageGroup
from .base import StateBase


class DictState(StateBase):
    """State characterized by a dictionary of states to allow for nested statesß"""

    data: Dict[str, StateBase]

    def __init__(
        self, data: Optional[Union[Dict[str, StateBase], Sequence[StateBase]]] = None
    ):
        """
        Args:
            data (dict or tuple):
                A dictionary of instances of :class:`StateBase` stored in this state.
                If instead a sequence is given, we generated ascending keys starting at
                zero automatically.
        """
        if data is None:
            self._state_data = {}  # no data
        elif isinstance(data, dict):
            self._state_data = data
        else:
            # data given in some other from -> we assume it's a sequence
            self._state_data = {str(i): v for i, v in enumerate(data)}

    @property
    def _state_attributes(self) -> Dict[str, Any]:
        """dict: Additional attributes, which are required to restore the state"""
        attributes = super()._state_attributes
        attributes["_keys"] = list(self._state_data.keys())
        return attributes

    @_state_attributes.setter
    def _state_attributes(self, attributes: Dict[str, Any]) -> None:
        """set the attributes of the state"""
        attributes.pop("_keys", None)  # remove auxillary information if present
        super(DictState, DictState)._state_attributes.__set__(self, attributes)  # type: ignore

    @classmethod
    def from_data(cls, attributes: Dict[str, Any], data=None):
        """create instance from attributes and data

        Args:
            attributes (dict): Additional (unserialized) attributes
            data: The data of the degerees of freedom of the physical system
        """
        if data is None or not isinstance(data, (dict, tuple, list)):
            raise TypeError("`data` must be a dictionary or sequence")
        keys = attributes.pop("_keys")
        if not isinstance(data, dict) and keys:
            data = {k: v for k, v in zip(keys, data)}
        return super().from_data(attributes, data)

    def copy(self, method: Literal["clean", "shallow", "data"], data=None):
        """create a copy of the state

        Args:
            method (str):
                Determines whether a `clean`, `shallow`, or `data` copy is performed.
                See :meth:`~modelrunner.state.base.StateBase.copy` for details.
            data:
                Data to be used instead of the one in the current state. This data is
                used as is and not copied!

        Returns:
            A copy of the current state object
        """
        if method == "data":
            # This special copy mode needs to be implemented in a very special way for
            # `DictState` since only the data needs to be deep-copied, while all other
            # attributes shall receive shallow copies. This particularly also needs to
            # hold for the substates stored in `_state_data`.
            obj = self.__class__.__new__(self.__class__)
            obj.__dict__ = self.__dict__.copy()
            if data is None:
                obj._state_data = {
                    k: v.copy(method="data") for k, v in self._state_data.items()
                }
            else:
                obj._state_data = data

        else:
            obj = super().copy(method=method, data=data)
        return obj

    def __len__(self) -> int:
        return len(self._state_data)

    def __getitem__(self, index: Union[int, str]) -> StateBase:
        if isinstance(index, str):
            return self._state_data[index]  # type: ignore
        elif isinstance(index, int):
            return next(itertools.islice(self._state_data.values(), index, None))  # type: ignore
        else:
            raise TypeError()

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
        data = {
            label: StateBase._state_from_stored_data(group, label, index=index)
            for label in attrs["_keys"]
        }
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
        for loc, substate in self._state_data.items():
            substate._state_update_from_stored_data(group, loc, index=index)

    def _state_write_to_storage(self, storage: StorageGroup, loc: Location) -> None:
        """write the state to storage

        Args:
            storage (:class:`StorageGroup`):
                A storage opened with :func:`~modelrunner.storage.open_storage`
            loc (str or list of str):
                The location in the storage where the state is written
        """
        group = storage.create_group(
            loc, cls=self.__class__, attrs=self._state_attributes_store
        )

        for label, substate in self._state_data_store.items():
            substate._state_write_to_storage(group, label)

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

        for label, substate in self._state_data_store.items():
            substate._state_create_trajectory(group, label)

    def _state_append_to_trajectory(self, storage: StorageGroup, loc: Location) -> None:
        """append the current state to a prepared trajectory

        Args:
            storage (:class:`StorageGroup`):
                A storage opened with :func:`~modelrunner.storage.open_storage`
            loc (str or list of str):
                The location in the storage where the trajectory is written
        """
        group = StorageGroup(storage, loc)
        for label, substate in self._state_data_store.items():
            substate._state_append_to_trajectory(group, label)
