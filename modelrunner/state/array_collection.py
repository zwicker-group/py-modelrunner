"""
Class that describe a state of multiple numpy arrays

.. codeauthor:: David Zwicker <david.zwicker@ds.mpg.de>
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import zarr

from .base import StateBase
from .io import zarrElement


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
    def _state_read_zarr_data(
        cls, zarr_element: zarr.Array, *, index=...
    ) -> ArrayCollectionState:
        data = tuple(
            zarr_element[label][index] for label in zarr_element.attrs["labels"]
        )
        return cls.from_data(zarr_element.attrs.asdict(), data)  # type: ignore

    def _state_update_from_zarr(self, element: zarrElement, *, index=...) -> None:
        for label, data_arr in zip(self.labels, self._state_data):
            data_arr[:] = element[label][index]

    def _state_write_zarr_data(
        self, zarr_group: zarr.Group, *, label: str = "data", **kwargs
    ) -> zarr.Group:
        zarr_subgroup = zarr_group.create_group(label)
        for sublabel, substate in zip(self.labels, self._state_data_store):
            zarr_subgroup.array(sublabel, substate)
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
        for sublabel, subdata in zip(self.labels, self._state_data_store):
            zarr_subgroup.zeros(
                sublabel,
                shape=(0,) + subdata.shape,
                chunks=(1,) + subdata.shape,
                dtype=subdata.dtype,
            )

        self._state_write_zarr_attributes(zarr_subgroup, attrs)
        return zarr_subgroup

    def _state_append_to_zarr_trajectory(self, zarr_element: zarr.Group) -> None:
        """append current data to a stored element"""
        for label, subdata in zip(self.labels, self._state_data_store):
            zarr_element[label].append([subdata])

    @classmethod
    def _from_simple_objects(
        cls, content, *, state_cls: Optional[StateBase] = None
    ) -> StateBase:
        """create state from text data

        Args:
            content: The data loaded from text
        """
        if state_cls is None:
            return super()._from_simple_objects(content)

        data = tuple(
            np.array(content["data"][label])
            for label in content["attributes"]["labels"]
        )
        return state_cls.from_data(content["attributes"], data)

    def _to_simple_objects(self):
        """return object data suitable for encoding as JSON"""
        data = self._state_data_store
        data_simple = {label: substate for label, substate in zip(self.labels, data)}
        return {"attributes": self._state_attributes_store, "data": data_simple}
