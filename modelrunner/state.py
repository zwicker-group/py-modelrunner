"""
Classes that describe the state of a simulation at a single point in time

.. autosummary::
   :nosignatures:

   ObjectState
   ArrayState
   ArrayCollectionState
   DictState

.. codeauthor:: David Zwicker <david.zwicker@ds.mpg.de>
"""

from __future__ import annotations

import copy
import itertools
import warnings
from typing import Any, Dict, Sequence, Tuple, Union

import numcodecs
import numpy as np
import zarr

from .io import IOBase, zarrElement


def _equals(left: Any, right: Any) -> bool:
    """checks whether two objects are equal, also supporting :class:~numpy.ndarray`

    Args:
        left: one object
        right: other object

    Returns:
        bool: Whether the two objects are equal
    """
    if left.__class__ is not right.__class__:
        return False

    if isinstance(left, str):
        return bool(left == right)

    if isinstance(left, np.ndarray):
        return np.array_equal(left, right)

    if isinstance(left, dict):
        return left.keys() == right.keys() and all(
            _equals(left[key], right[key]) for key in left
        )

    if isinstance(left, StateBase):
        return left.attributes == right.attributes and _equals(left.data, right.data)

    if hasattr(left, "__iter__"):
        return len(left) == len(right) and all(
            _equals(l, r) for l, r in zip(left, right)
        )

    return bool(left == right)


class StateBase(IOBase):
    """Base class for specifying degrees of freedom of a simulation

    A state contains values of all degrees of freedom of a physical system (stored in
    the `data` field) and potentially some additional information (stored in the
    `attributes` field). The `data` is mutable and often a numpy array or a collection
    of numpy arrays. Conversely, the `attributes` are a dictionary with immutable
    values.
    """

    data: Any
    _state_classes: Dict[str, StateBase] = {}

    def __init__(self, data: Any = None):
        """
        Args:
            data: The data describing the state
        """
        self.data = data

    @property
    def attributes(self) -> Dict[str, Any]:
        """dict: Additional attributes, which are required to restore the state"""
        return {"__class__": self.__class__.__name__}

    def __init_subclass__(cls, **kwargs):  # @NoSelf
        """register all subclassess to reconstruct them later"""
        super().__init_subclass__(**kwargs)
        cls._state_classes[cls.__name__] = cls

    def __eq__(self, other):
        return _equals(self.data, other.data)

    @classmethod
    def from_state(cls, attributes: Dict[str, Any], data=None) -> StateBase:
        """create any state class from attributes and data

        Args:
            attributes (dict): Additional attributes
            data: The data of the degerees of freedom of the physical system
        """
        if cls.__name__ == "StateBase":
            # use the base class as a point to load arbitrary subclasses
            if attributes["__class__"] == "StateBase":
                raise RuntimeError("Cannot create StateBase instances")
            state_cls = cls._state_classes[attributes["__class__"]]
            return state_cls.from_state(attributes, data)

        elif cls.__name__ == attributes["__class__"]:
            # simple version instantiating the current class with the given data
            if len(attributes) > 1:
                warnings.warn(
                    f"Unused attributes, but {cls.__name__} did not implemented custom "
                    "from_state"
                )
            return cls(data)

        else:
            raise ValueError(f"Incompatible state class {attributes['class']}")

    def copy(self):
        return self.__class__.from_state(self.attributes, copy.deepcopy(self.data))

    def _write_zarr_attributes(
        self, element: zarrElement, attrs: Dict[str, Any] = None
    ) -> zarrElement:
        """prepare the zarr element for this state"""
        # write the attributes of the state
        element.attrs.update(self.attributes)

        # write additional attributes provided as argument
        if attrs is not None:
            if "__class__" in attrs:
                raise ValueError("`class` attribute is reserved")
            element.attrs.update(attrs)

        return element

    def _write_zarr_data(self, zarr_group: zarr.Group, **kwargs) -> zarrElement:
        raise NotImplementedError

    def _write_zarr(
        self, zarr_group: zarr.Group, attrs: Dict[str, Any] = None, **kwargs
    ):
        element = self._write_zarr_data(zarr_group, **kwargs)
        self._write_zarr_attributes(element, attrs)

    @classmethod
    def _from_zarr(cls, zarr_element: zarrElement, *, index=...) -> StateBase:
        """create instance of correct subclass from data stored in zarr"""
        attributes = zarr_element.attrs.asdict()
        class_name = attributes["__class__"]
        state_cls = cls._state_classes[class_name]
        data = state_cls._read_zarr_data(zarr_element, index=index)
        return state_cls.from_state(attributes, data)

    @classmethod
    def _read_zarr_data(cls, zarr_element: zarrElement, *, index=...) -> Any:
        """read data stored in zarr element"""
        raise NotImplementedError

    def _update_from_zarr(self, element: zarrElement, *, index=...) -> None:
        raise NotImplementedError

    def _prepare_zarr_trajectory(
        self, zarr_group: zarr.Group, attrs: Dict[str, Any] = None, **kwargs
    ) -> zarrElement:
        """prepare the zarr element for this state"""
        raise NotImplementedError

    def _append_to_zarr_trajectory(self, zarr_element: zarrElement) -> None:
        """append current data to the prepared zarr element"""
        raise NotImplementedError

    @classmethod
    def _from_text_data(cls, content, *, state_cls: StateBase = None) -> StateBase:
        """create state from text data

        Args:
            content: The loaded data
        """
        if state_cls is None:
            state_cls = cls._state_classes[content["attributes"]["__class__"]]
            return state_cls._from_text_data(content, state_cls=state_cls)
        else:
            return state_cls.from_state(content["attributes"], content["data"])

    def _to_text_data(self):
        """return object data suitable for encoding as text"""
        return {"attributes": self.attributes, "data": self.data}


class ObjectState(StateBase):
    """State characterized by a serializable python object"""

    default_codec = numcodecs.Pickle()

    @classmethod
    def _read_zarr_data(cls, zarr_element: zarr.Array, *, index=...):
        if zarr_element.shape == () and index is ...:
            return zarr_element[index].item()
        else:
            return zarr_element[index]

    def _update_from_zarr(self, element: zarrElement, *, index=...) -> None:
        self.data = element[index].item()

    def _write_zarr_data(  # type: ignore
        self,
        zarr_group: zarr.Group,
        *,
        label: str = "data",
        codec: numcodecs.abc.Codec = None,
    ):
        if codec is None:
            codec = self.default_codec
        return zarr_group.array(
            label, self.data, shape=(0,), dtype=object, object_codec=codec
        )

    def _prepare_zarr_trajectory(  # type: ignore
        self,
        zarr_group: zarr.Group,
        attrs: Dict[str, Any] = None,
        *,
        label: str = "data",
        codec: numcodecs.abc.Codec = None,
    ) -> zarr.Array:
        """prepare the zarr storage for this state"""
        if codec is None:
            codec = self.default_codec

        zarr_element = zarr_group.zeros(
            label,
            shape=(0,),
            chunks=(1,),
            dtype=object,
            object_codec=codec,
        )
        self._write_zarr_attributes(zarr_element, attrs)

        return zarr_element

    def _append_to_zarr_trajectory(self, zarr_element: zarr.Array) -> None:
        """append current data to a stored element"""
        zarr_element.resize(len(zarr_element) + 1)
        zarr_element[-1] = self.data


class ArrayState(StateBase):
    """State characterized by a single numpy array"""

    data: np.ndarray

    def __eq__(self, other):
        return np.array_equal(self.data, other.data)

    @classmethod
    def from_state(cls, attributes: Dict[str, Any], data=None):
        if data is not None:
            data = np.asarray(data)
        return super().from_state(attributes, data)

    @classmethod
    def _read_zarr_data(cls, zarr_element: zarr.Array, *, index=...):
        return zarr_element[index]

    def _update_from_zarr(self, element: zarrElement, *, index=...) -> None:
        self.data = element[index]

    def _write_zarr_data(  # type: ignore
        self,
        zarr_group: zarr.Group,
        *,
        label: str = "data",
    ) -> zarr.Array:
        return zarr_group.array(label, self.data)

    def _prepare_zarr_trajectory(
        self,
        zarr_group: zarr.Group,
        attrs: Dict[str, Any] = None,
        *,
        label: str = "data",
        **kwargs,
    ) -> zarr.Array:
        """prepare the zarr storage for this state"""
        zarr_element = zarr_group.zeros(
            label,
            shape=(0,) + self.data.shape,
            chunks=(1,) + self.data.shape,
            dtype=self.data.dtype,
        )
        self._write_zarr_attributes(zarr_element, attrs)

        return zarr_element

    def _append_to_zarr_trajectory(self, zarr_element: zarr.Array) -> None:
        """append current data to a stored element"""
        zarr_element.append([self.data])

    @classmethod
    def _from_text_data(cls, content, *, state_cls: StateBase = None) -> StateBase:
        """create state from text data

        Args:
            content: The loaded data
        """
        if state_cls is None:
            return super()._from_text_data(content)

        return cls(np.asarray(content["data"]))


class ArrayCollectionState(StateBase):
    """State characterized by a multiple numpy array"""

    data: Tuple[np.ndarray, ...]

    def __init__(
        self, data: Tuple[np.ndarray, ...] = None, *, labels: Sequence[str] = None
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
            self.labels = tuple(str(i) for i in range(len(self.data)))
        else:
            assert len(self.data) == len(labels) == len(set(labels))
            self.labels = tuple(labels)

    def __eq__(self, other):
        return len(self.data) == len(other.data) and all(
            np.array_equal(s, o) for s, o in zip(self.data, other.data)
        )

    @property
    def attributes(self) -> Dict[str, Any]:
        """dict: Additional attributes, which are required to restore the state"""
        attributes = super().attributes
        attributes["labels"] = self.labels
        return attributes

    @classmethod
    def from_state(cls, attributes: Dict[str, Any], data=None):
        if data is not None:
            data = tuple(np.asarray(subdata) for subdata in data)
        labels = attributes.pop("labels")
        return cls(data, labels=labels)

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
        return cls(
            tuple(zarr_element[label][index] for label in zarr_element.attrs["labels"]),
            labels=zarr_element.attrs["labels"],
        )

    def _update_from_zarr(self, element: zarrElement, *, index=...) -> None:
        for label, data_arr in zip(self.labels, self.data):
            data_arr[:] = element[label][index]

    def _write_zarr_data(
        self, zarr_group: zarr.Group, *, label: str = "data", **kwargs
    ):
        zarr_subgroup = zarr_group.create_group(label)
        for sublabel, substate in zip(self.labels, self.data):
            zarr_subgroup.array(sublabel, substate)
        return zarr_subgroup

    def _prepare_zarr_trajectory(
        self,
        zarr_group: zarr.Group,
        attrs: Dict[str, Any] = None,
        *,
        label: str = "data",
        **kwargs,
    ) -> zarr.Group:
        """prepare the zarr storage for this state"""
        zarr_subgroup = zarr_group.create_group(label)
        for sublabel, subdata in zip(self.labels, self.data):
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
        for label, subdata in zip(self.labels, self.data):
            zarr_element[label].append([subdata])

    @classmethod
    def _from_text_data(cls, content, *, state_cls: StateBase = None) -> StateBase:
        """create state from text data

        Args:
            content: The data loaded from text
        """
        if state_cls is None:
            return super()._from_text_data(content)

        data = tuple(
            np.array(content["data"][label])
            for label in content["attributes"]["labels"]
        )
        return state_cls.from_state(content["attributes"], data)

    def _to_text_data(self):
        """return object data suitable for encoding as JSON"""
        data = {label: substate for label, substate in zip(self.labels, self.data)}
        return {"attributes": self.attributes, "data": data}


class DictState(StateBase):
    """State characterized by a dictionary of data"""

    data: Dict[str, StateBase]

    def __init__(self, data: Union[Dict[str, StateBase], Tuple[StateBase]] = None):
        if data is None:
            data = {}
        elif not isinstance(data, dict):
            data = {str(i): v for i, v in enumerate(data)}
        super().__init__(data)

    @property
    def attributes(self) -> Dict[str, Any]:
        """dict: Additional attributes, which are required to restore the state"""
        attributes = super().attributes
        attributes["__keys__"] = list(self.data.keys())
        return attributes

    @classmethod
    def from_state(cls, attributes: Dict[str, Any], data=None):
        if data is None or not isinstance(data, (dict, tuple, list)):
            raise TypeError("`data` must be a dictionary or sequence")
        assert cls.__name__ == attributes["__class__"]
        if not isinstance(data, dict) and "__keys__" in attributes:
            data = {k: v for k, v in zip(attributes["__keys__"], data)}
        return cls(data)

    def copy(self):
        return self.__class__({k: v.copy() for k, v in self.data.items()})

    def __getitem__(self, index: Union[int, str]) -> StateBase:
        if isinstance(index, str):
            return self.data[index]
        elif isinstance(index, int):
            return next(itertools.islice(self.data.values(), index, None))
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
        for key, substate in self.data.items():
            substate._update_from_zarr(element[key], index=index)

    def _write_zarr_data(
        self, zarr_group: zarr.Group, *, label: str = "data", **kwargs
    ):
        zarr_subgroup = zarr_group.create_group(label)
        for label, substate in self.data.items():
            substate._write_zarr(zarr_subgroup, label=label, **kwargs)
        return zarr_subgroup

    def _prepare_zarr_trajectory(
        self,
        zarr_group: zarr.Group,
        attrs: Dict[str, Any] = None,
        *,
        label: str = "data",
        **kwargs,
    ) -> zarr.Group:
        """prepare the zarr storage for this state"""
        zarr_subgroup = zarr_group.create_group(label)
        for label, substate in self.data.items():
            substate._prepare_zarr_trajectory(zarr_subgroup, label=label, **kwargs)

        self._write_zarr_attributes(zarr_subgroup, attrs)
        return zarr_subgroup

    def _append_to_zarr_trajectory(self, zarr_element: zarr.Group) -> None:
        """append current data to a stored element"""
        for label, substate in self.data.items():
            substate._append_to_zarr_trajectory(zarr_element[label])

    @classmethod
    def _from_text_data(cls, content, *, state_cls: StateBase = None) -> StateBase:
        """create state from JSON data

        Args:
            content: The data loaded from json
        """
        if state_cls is None:
            return super()._from_text_data(content)

        data = {}
        for label, substate in content["data"].items():
            data[label] = StateBase._from_text_data(substate)
        return state_cls.from_state(content["attributes"], data)

    def _to_text_data(self):
        """return object data suitable for encoding as JSON"""
        data = {
            label: substate._to_text_data() for label, substate in self.data.items()
        }
        return {"attributes": self.attributes, "data": data}


def make_state(data: Any) -> StateBase:
    """turn any data into a :class:`StateBase`"""
    if isinstance(data, StateBase):
        return data
    elif isinstance(data, np.ndarray):
        return ArrayState(data)
    return ObjectState(data)
