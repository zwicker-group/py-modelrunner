"""
Classes that describe the state of a simulation at a single point in time

Each state is defined by :attr:`attributes` and :attr:`data`. Attributes describe
general aspects about a state, which typically do not change, e.g., its `name`.
These classes define how data is read and written and they contain methods that can be
used to write multiple states of the same class to a file consecutively, e.g., to store
a trajectory. Here, it is assumed that the `attributes` do not change over time.

TODO:
- document the succession of calls for storing fields (to get a better idea of the
  available hooks)
- do the same for loading data

.. autosummary::
   :nosignatures:

   ObjectState
   ArrayState
   ArrayCollectionState
   DictState

.. codeauthor:: David Zwicker <david.zwicker@ds.mpg.de>
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Sequence, Tuple, Union

import numpy as np
import zarr

from .base import StateBase
from .io import zarrElement


class ArrayCollectionState(StateBase):
    """State characterized by a multiple numpy array

    The arrays are given as a tuple of :class:`~numpy.ndarray`. Optionally, a sequence
    of labels can be supplied to refer to the arrays by convenient names. If the labels
    are omitted, default labels using a string of the respective index of the array in
    the sequence are generated.
    """

    data: Tuple[np.ndarray, ...]

    def __init__(
        self,
        data: Optional[Tuple[np.ndarray, ...]] = None,
        *,
        labels: Optional[Sequence[str]] = None,
    ):
        """
        Args:
            data: The data describing the state
        """
        if data is None:
            self.data = tuple()
        else:
            self.data = tuple(data)

        if labels is None:
            self._labels = tuple(str(i) for i in range(len(self.data)))
        else:
            assert len(self.data) == len(labels) == len(set(labels))
            self._labels = tuple(labels)

    def __eq__(self, other):
        return (
            self.__class__ == other.__class__
            and len(self.data) == len(other.data)
            and all(np.array_equal(s, o) for s, o in zip(self.data, other.data))
        )

    @property
    def labels(self) -> Sequence[str]:
        """list: the label assigned to each array"""
        labels = getattr(self, "_labels", None)
        if labels is None:
            return [str(i) for i in range(len(self.data))]
        else:
            return list(labels)

    @property
    def attributes(self) -> Dict[str, Any]:
        """dict: Additional attributes, which are required to restore the state"""
        attributes = super().attributes
        attributes["labels"] = self.labels
        return attributes

    @classmethod
    def from_data(cls, attributes: Dict[str, Any], data=None):
        """create instance from attributes and data

        Args:
            attributes (dict): Additional (unserialized) attributes
            data: The data of the degerees of freedom of the physical system
        """
        if data is not None:
            data = tuple(np.asarray(subdata) for subdata in data)
        labels = attributes.get("labels")

        # create a new object without calling __init__, which might be overwriten by
        # the subclass and not follow our interface
        obj = cls.__new__(cls)
        if data is not None:
            obj.data = data
        if labels is not None:
            obj._labels = labels
        return obj

    def __getitem__(self, index: Union[int, str]) -> np.ndarray:
        if isinstance(index, str):
            return self.data[self.labels.index(index)]
        elif isinstance(index, int):
            return self.data[index]
        else:
            raise TypeError()

    @classmethod
    def _read_zarr_data(
        cls, zarr_element: zarr.Array, *, index=...
    ) -> ArrayCollectionState:
        data = tuple(
            zarr_element[label][index] for label in zarr_element.attrs["labels"]
        )
        return cls.from_data(zarr_element.attrs.asdict(), data)  # type: ignore

    def _update_from_zarr(self, element: zarrElement, *, index=...) -> None:
        for label, data_arr in zip(self.labels, self.data):
            data_arr[:] = element[label][index]

    def _write_zarr_data(
        self, zarr_group: zarr.Group, *, label: str = "data", **kwargs
    ) -> zarr.Group:
        zarr_subgroup = zarr_group.create_group(label)
        for sublabel, substate in zip(self.labels, self._data_store):
            zarr_subgroup.array(sublabel, substate)
        return zarr_subgroup

    def _prepare_zarr_trajectory(
        self,
        zarr_group: zarr.Group,
        attrs: Optional[Dict[str, Any]] = None,
        *,
        label: str = "data",
        **kwargs,
    ) -> zarr.Group:
        """prepare the zarr storage for this state"""
        zarr_subgroup = zarr_group.create_group(label)
        for sublabel, subdata in zip(self.labels, self._data_store):
            zarr_subgroup.zeros(
                sublabel,
                shape=(0,) + subdata.shape,
                chunks=(1,) + subdata.shape,
                dtype=subdata.dtype,
            )

        self._write_zarr_attributes(zarr_subgroup, attrs)
        return zarr_subgroup

    def _append_to_zarr_trajectory(self, zarr_element: zarr.Group) -> None:
        """append current data to a stored element"""
        for label, subdata in zip(self.labels, self._data_store):
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
        data = {
            label: substate for label, substate in zip(self.labels, self._data_store)
        }
        return {"attributes": self._attributes_store, "data": data}
