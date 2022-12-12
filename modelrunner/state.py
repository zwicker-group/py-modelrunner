"""
Classes that describe the state of a simulation at a single point in time

Each state is defined by :attr:`attributes` and :attr:`data`. Attributes describe
general aspects about a state, which typically do not change, e.g., its `name`.
These classes define how data is read and written and they contain methods that can be
used to write multiple states of the same class to a file consecutively, e.g., to store
a trajectory. Here, it is assumed that the `attributes` do not change over time.

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
from typing import Any, Dict, Optional, Sequence, Tuple, Union

import numcodecs
import numpy as np
import zarr

from .io import IOBase, simplify_data, zarrElement


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
        return left._attributes_store == right._attributes_store and _equals(
            left._data_store, right._data_store
        )

    if hasattr(left, "__iter__"):
        return len(left) == len(right) and all(
            _equals(l, r) for l, r in zip(left, right)
        )

    return bool(left == right)


class StateBase(IOBase):
    """Base class for specifying the state of a simulation

    A state contains values of all degrees of freedom of a physical system (stored in
    the `data` field) and potentially some additional information (stored in the
    `attributes` field). The `data` is mutable and often a numpy array or a collection
    of numpy arrays. Conversely, the `attributes` are a dictionary with immutable
    values. To allow flexible storage, we define the attributes `_attributes_store` and
    `_data_store`, which by default return `attributes` and `data` directly, but may be
    overwritten to process the data before storage (e.g., by additional serialization).
    """

    _format_version = 1
    """int: number indicating the version of the file format"""

    _state_classes: Dict[str, StateBase] = {}
    """dict: class-level list of all subclasses of StateBase"""

    def __init_subclass__(cls, **kwargs):  # @NoSelf
        """register all subclasses to reconstruct them later"""
        # register the subclasses
        super().__init_subclass__(**kwargs)
        if cls is not StateBase:
            cls._state_classes[cls.__name__] = cls

    @property
    def attributes(self) -> Dict[str, Any]:
        """dict: Additional attributes, which are required to restore the state"""
        return {}

    @property
    def _attributes_store(self) -> Dict[str, Any]:
        """dict: Attributes in the form in which they will be written to storage"""
        attrs = simplify_data(self.attributes)
        attrs["__class__"] = self.__class__.__name__
        attrs["__version__"] = self._format_version
        return attrs  # type: ignore

    @classmethod
    def _read_attributes(cls, attributes: Dict[str, Any]) -> Dict[str, Any]:
        """recreating stored attributes

        This classmethod can be overwritten if attributes have been serialized by the
        :meth:`_attributes_store` property.
        """
        return attributes

    @property
    def _data_store(self) -> Any:
        """attribute that determines what data is stored in this state"""
        if hasattr(self, "data"):
            return self.data
        else:
            return None

    def __eq__(self, other):
        return _equals(self, other)

    @classmethod
    def from_state(cls, attributes: Dict[str, Any], data=None) -> StateBase:
        """create instance of any state class from attributes and data

        Args:
            attributes (dict): Additional (unserialized) attributes
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
            num_attrs = 1  # __class__ was stored in attributes
            if "__version__" in attributes:
                format_version = attributes["__version__"]
                num_attrs += 1
            else:
                format_version = 0
            if format_version != cls._format_version:
                warnings.warn(
                    f"File format version mismatch "
                    f"({format_version} != {cls._format_version})"
                )
            if len(attributes) > num_attrs:  # additional attributes given
                warnings.warn(
                    f"Unused attributes, but {cls.__name__} did not implemented custom "
                    "from_state method"
                )

            # create a new object without calling __init__, which might be overwriten by
            # the subclass and not follow our interface
            obj = cls.__new__(cls)
            if data is not None:
                obj.data = data  # type: ignore
            return obj

        else:
            raise ValueError(f"Incompatible state class {attributes['class']}")

    def copy(self, data=None):
        if data is None:
            data = copy.deepcopy(self.data)
        return self.__class__.from_state(copy.deepcopy(self.attributes), data)

    def _write_zarr_attributes(
        self, element: zarrElement, attrs: Optional[Dict[str, Any]] = None
    ) -> zarrElement:
        """prepare the zarr element for this state"""
        # write the attributes of the state
        element.attrs.update(simplify_data(self._attributes_store))

        # write additional attributes provided as argument
        if attrs is not None:
            element.attrs.update(simplify_data(attrs))

        return element

    def _write_zarr_data(
        self, zarr_group: zarr.Group, *, name: str = "data", **kwargs
    ) -> zarrElement:
        raise NotImplementedError

    def _write_zarr(
        self, zarr_group: zarr.Group, attrs: Optional[Dict[str, Any]] = None, **kwargs
    ) -> zarrElement:
        element = self._write_zarr_data(zarr_group, **kwargs)
        self._write_zarr_attributes(element, attrs)
        return element

    @classmethod
    def _from_zarr(cls, zarr_element: zarrElement, *, index=...) -> StateBase:
        """create instance of correct subclass from data stored in zarr"""
        attributes = cls._read_attributes(zarr_element.attrs.asdict())
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
        self, zarr_group: zarr.Group, attrs: Optional[Dict[str, Any]] = None, **kwargs
    ) -> zarrElement:
        """prepare the zarr element for this state"""
        raise NotImplementedError

    def _append_to_zarr_trajectory(self, zarr_element: zarrElement) -> None:
        """append current data to the prepared zarr element"""
        raise NotImplementedError

    @classmethod
    def _from_simple_objects(
        cls, content, *, state_cls: Optional[StateBase] = None
    ) -> StateBase:
        """create state from text data

        Args:
            content: The loaded data
        """
        if state_cls is None:
            # general branch that determines the state class to use to load the object
            state_cls = cls._state_classes[content["attributes"]["__class__"]]
            return state_cls._from_simple_objects(content, state_cls=state_cls)
        else:
            # specific (basic) implementation that just reads the state
            attributes = cls._read_attributes(content["attributes"])
            return state_cls.from_state(attributes, content["data"])

    def _to_simple_objects(self):
        """return object data suitable for encoding as text"""
        return {"attributes": self._attributes_store, "data": self._data_store}


class ObjectState(StateBase):
    """State characterized by a serializable python object

    The data needs to be accessible form the :attr:`data` property of the instance.
    Additional attributes can be supplied via the :attr:`attribute` property, which will
    then be stored in files. To support reading such augmented states, the method
    :meth:`from_state` needs to be overwritten.
    """

    default_codec = numcodecs.Pickle()

    def __init__(self, data: Optional[Any] = None):
        """
        Args:
            data: The data describing the state
        """
        self.data = data

    @classmethod
    def _read_zarr_data(cls, zarr_element: zarr.Array, *, index=...):
        if zarr_element.shape == () and index is ...:
            return zarr_element[index].item()
        else:
            return zarr_element[index]

    def _update_from_zarr(self, element: zarrElement, *, index=...) -> None:
        if element.shape == () and index is ...:
            self.data = element[index].item()
        else:
            self.data = element[index]

    def _write_zarr_data(  # type: ignore
        self,
        zarr_group: zarr.Group,
        *,
        label: str = "data",
        codec: Optional[numcodecs.abc.Codec] = None,
    ) -> zarrElement:
        if codec is None:
            codec = self.default_codec
        return zarr_group.array(
            label, self._data_store, shape=(0,), dtype=object, object_codec=codec
        )

    def _prepare_zarr_trajectory(  # type: ignore
        self,
        zarr_group: zarr.Group,
        attrs: Optional[Dict[str, Any]] = None,
        *,
        label: str = "data",
        codec: Optional[numcodecs.abc.Codec] = None,
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
        zarr_element[-1] = self._data_store


class ArrayState(StateBase):
    """State characterized by a single numpy array"""

    def __init__(self, data: Optional[np.ndarray] = None):
        """
        Args:
            data: The data describing the state
        """
        self.data = data

    @classmethod
    def from_state(cls, attributes: Dict[str, Any], data=None):
        """create instance from attributes and data

        Args:
            attributes (dict): Additional (unserialized) attributes
            data: The data of the degerees of freedom of the physical system
        """
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
        return zarr_group.array(label, self._data_store)

    def _prepare_zarr_trajectory(
        self,
        zarr_group: zarr.Group,
        attrs: Optional[Dict[str, Any]] = None,
        *,
        label: str = "data",
        **kwargs,
    ) -> zarr.Array:
        """prepare the zarr storage for this state"""
        zarr_element = zarr_group.zeros(
            label,
            shape=(0,) + self._data_store.shape,
            chunks=(1,) + self._data_store.shape,
            dtype=self._data_store.dtype,
        )
        self._write_zarr_attributes(zarr_element, attrs)

        return zarr_element

    def _append_to_zarr_trajectory(self, zarr_element: zarr.Array) -> None:
        """append current data to a stored element"""
        zarr_element.append([self._data_store])


class ArrayCollectionState(StateBase):
    """State characterized by a multiple numpy array"""

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
    def from_state(cls, attributes: Dict[str, Any], data=None):
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
        return cls.from_state(zarr_element.attrs.asdict(), data)  # type: ignore

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
        return state_cls.from_state(content["attributes"], data)

    def _to_simple_objects(self):
        """return object data suitable for encoding as JSON"""
        data = {
            label: substate for label, substate in zip(self.labels, self._data_store)
        }
        return {"attributes": self._attributes_store, "data": data}


class DictState(StateBase):
    """State characterized by a dictionary of states"""

    data: Dict[str, StateBase]

    def __init__(
        self, data: Optional[Union[Dict[str, StateBase], Tuple[StateBase]]] = None
    ):
        if data is None:
            self.data = {}
        elif not isinstance(data, dict):
            self.data = {str(i): v for i, v in enumerate(data)}
        else:
            self.data = data

    @property
    def attributes(self) -> Dict[str, Any]:
        """dict: Additional attributes, which are required to restore the state"""
        attributes = super().attributes
        attributes["__keys__"] = list(self.data.keys())
        return attributes

    @classmethod
    def from_state(cls, attributes: Dict[str, Any], data=None):
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
            obj.data = data
        return obj

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
        return state_cls.from_state(content["attributes"], data)

    def _to_simple_objects(self):
        """return object data suitable for encoding as JSON"""
        data = {
            label: substate._to_simple_objects()
            for label, substate in self._data_store.items()
        }
        return {"attributes": self._attributes_store, "data": data}


def make_state(data: Any) -> StateBase:
    """turn any data into a :class:`StateBase`"""
    if isinstance(data, StateBase):
        return data
    elif isinstance(data, np.ndarray):
        return ArrayState(data)
    return ObjectState(data)