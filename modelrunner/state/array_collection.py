"""
Class that describe a state of multiple numpy arrays

.. codeauthor:: David Zwicker <david.zwicker@ds.mpg.de>
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np

from ..storage import Group, storage_actions
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

    # @property
    # def _state_data(self) -> Any:
    #     """determines what data is stored in this state
    #
    #     This property can be used to determine what is stored as `data` and in which
    #     form.
    #     """
    #     try:
    #         return getattr(self, self._state_data_attr_name)
    #     except AttributeError:
    #         # this can happen if the `data` attribute is not defined
    #         raise AttributeError("`_state_data` should be defined by subclass")
    #
    # @_state_data.setter
    # def _state_data(self, data) -> None:
    #     """set the data of the class"""
    #     try:
    #         setattr(self, self._state_data_attr_name, data)  # try setting data directly
    #     except AttributeError:
    #         # this can happen if `data` is a read-only attribute, i.e., if the data
    #         # attribute is managed by the child class
    #         raise AttributeError("`_state_data` should be defined by subclass")

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
    def _state_from_stored_data(cls, storage, key: str, index: Optional[int] = None):
        attrs = storage.read_attrs(key)
        attrs.pop("__class__")

        group = Group(storage, key)
        data = tuple(group.read_array(label, index=index) for label in attrs["labels"])

        return cls.from_data(attrs, data)

    def _state_update_from_stored_data(
        self, storage, key: str, index: Optional[int] = None
    ):
        group = Group(storage, key)
        for label, data_arr in zip(self.labels, self._state_data):
            group.read_array(label, index=index, out=data_arr)

    def _state_write_to_storage(self, storage, key: Sequence[str]):
        group = storage.create_group(
            key, cls=self.__class__, attrs=self._state_attributes_store
        )

        for sublabel, substate in zip(self.labels, self._state_data_store):
            group.write_array(sublabel, substate)

    def _state_create_trajectory(self, storage, key: str):
        """prepare the zarr storage for this state"""
        group = storage.create_group(
            key, cls=self.__class__, attrs=self._state_attributes_store
        )

        for sublabel, subdata in zip(self.labels, self._state_data_store):
            group.create_dynamic_array(
                sublabel, shape=subdata.shape, dtype=subdata.dtype
            )

    def _state_append_to_trajectory(self, storage, key: str):
        group = Group(storage, key)
        for label, subdata in zip(self.labels, self._state_data_store):
            group.extend_dynamic_array(label, subdata)


storage_actions.register(
    "read_object", ArrayCollectionState, ArrayCollectionState._state_from_stored_data
)
