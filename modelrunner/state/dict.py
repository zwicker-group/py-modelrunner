"""
Classes that describe the state of a simulation using a dictionary of states

.. codeauthor:: David Zwicker <david.zwicker@ds.mpg.de>
"""

from __future__ import annotations

import itertools
from typing import Any, Dict, Optional, Sequence, Union

import zarr

from .base import StateBase
from .io import zarrElement


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
        attributes["__keys__"] = list(self._state_data.keys())
        return attributes

    @_state_attributes.setter
    def _state_attributes(self, attributes: Dict[str, Any]) -> None:
        """set the attributes of the state"""
        attributes.pop("__keys__")  # remove auxillary information
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
        if not isinstance(data, dict) and "__keys__" in attributes:
            data = {k: v for k, v in zip(attributes.pop("__keys__"), data)}
        return super().from_data(attributes, data)

    def copy(self, method: str, data=None):
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
    def _state_read_zarr_data(
        cls, zarr_element: zarr.Array, *, index=...
    ) -> Dict[str, StateBase]:
        return {
            label: StateBase._from_zarr(zarr_element[label], index=index)
            for label in zarr_element.attrs["__keys__"]
        }

    def _state_update_from_zarr(self, element: zarrElement, *, index=...) -> None:
        for key, substate in self._state_data.items():
            substate._state_update_from_zarr(element[key], index=index)

    def _state_write_zarr_data(
        self, zarr_group: zarr.Group, *, label: str = "data", **kwargs
    ) -> zarr.Group:
        zarr_subgroup = zarr_group.create_group(label)
        for label, substate in self._state_data_store.items():
            substate._write_zarr(zarr_subgroup, label=label, **kwargs)
        return zarr_subgroup

    def _state_prepare_zarr_trajectory(
        self,
        zarr_group: zarr.Group,
        attrs: Optional[Dict[str, Any]] = None,
        *,
        label: str = "data",
        **kwargs,
    ) -> zarr.Group:
        """prepare the zarr storage for this state"""
        zarr_subgroup = zarr_group.create_group(label)
        for label, substate in self._state_data_store.items():
            substate._state_prepare_zarr_trajectory(
                zarr_subgroup, label=label, **kwargs
            )

        self._state_write_zarr_attributes(zarr_subgroup, attrs)
        return zarr_subgroup

    def _state_append_to_zarr_trajectory(self, zarr_element: zarr.Group) -> None:
        """append current data to a stored element"""
        for label, substate in self._state_data_store.items():
            substate._state_append_to_zarr_trajectory(zarr_element[label])

    @classmethod
    def _from_simple_objects(
        cls, content, *, state_cls: Optional[StateBase] = None
    ) -> StateBase:
        """create state from JSON data

        Args:
            content: The data loaded from json
        """
        if state_cls is None:
            return super()._from_simple_objects(content)

        data = {}
        for label, substate in content["data"].items():
            data[label] = StateBase._from_simple_objects(substate)
        return state_cls.from_data(content["attributes"], data)

    def _to_simple_objects(self):
        """return object data suitable for encoding as JSON"""
        data = {
            label: substate._to_simple_objects()
            for label, substate in self._state_data_store.items()
        }
        return {"attributes": self._state_attributes_store, "data": data}
