"""
Classes that describe the state of a simulation using a dictionary of states

.. codeauthor:: David Zwicker <david.zwicker@ds.mpg.de>
"""

from __future__ import annotations

import itertools
from typing import Any, Dict, Optional, Tuple, Union

import zarr

from .base import StateBase
from .io import zarrElement


class DictState(StateBase):
    """State characterized by a dictionary of states"""

    data: Dict[str, StateBase]

    def __init__(
        self, data: Optional[Union[Dict[str, StateBase], Tuple[StateBase]]] = None
    ):
        if data is None:
            data = {}
        elif not isinstance(data, dict):
            data = {str(i): v for i, v in enumerate(data)}

        try:
            setattr(self, self._data_attribute, data)
        except AttributeError:
            # this can happen if `data` is a read-only attribute, i.e., if the data
            # attribute is managed by the child class
            pass

    @property
    def attributes(self) -> Dict[str, Any]:
        """dict: Additional attributes, which are required to restore the state"""
        attributes = super().attributes
        attributes["__keys__"] = list(getattr(self, self._data_attribute).keys())
        return attributes

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
            data = {k: v for k, v in zip(attributes["__keys__"], data)}

        # create a new object without calling __init__, which might be overwriten by
        # the subclass and not follow our interface
        obj = cls.__new__(cls)
        if data is not None:
            setattr(obj, obj._data_attribute, data)
        return obj

    def __len__(self) -> int:
        return len(getattr(self, self._data_attribute))

    def __getitem__(self, index: Union[int, str]) -> StateBase:
        data = getattr(self, self._data_attribute)
        if isinstance(index, str):
            return data[index]  # type: ignore
        elif isinstance(index, int):
            return next(itertools.islice(data.values(), index, None))  # type: ignore
        else:
            raise TypeError()

    @classmethod
    def _read_zarr_data(
        cls, zarr_element: zarr.Array, *, index=...
    ) -> Dict[str, StateBase]:
        return {
            label: StateBase._from_zarr(zarr_element[label], index=index)
            for label in zarr_element.attrs["__keys__"]
        }

    def _update_from_zarr(self, element: zarrElement, *, index=...) -> None:
        data = getattr(self, self._data_attribute)
        for key, substate in data.items():
            substate._update_from_zarr(element[key], index=index)

    def _write_zarr_data(
        self, zarr_group: zarr.Group, *, label: str = "data", **kwargs
    ) -> zarr.Group:
        zarr_subgroup = zarr_group.create_group(label)
        for label, substate in self._data_store.items():
            substate._write_zarr(zarr_subgroup, label=label, **kwargs)
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
        for label, substate in self._data_store.items():
            substate._prepare_zarr_trajectory(zarr_subgroup, label=label, **kwargs)

        self._write_zarr_attributes(zarr_subgroup, attrs)
        return zarr_subgroup

    def _append_to_zarr_trajectory(self, zarr_element: zarr.Group) -> None:
        """append current data to a stored element"""
        for label, substate in self._data_store.items():
            substate._append_to_zarr_trajectory(zarr_element[label])

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
            for label, substate in self._data_store.items()
        }
        return {"attributes": self._attributes_store, "data": data}
