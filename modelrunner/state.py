"""
<<<<<<< Upstream, based on main
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
from typing import Any, Dict, Optional, Sequence, Tuple, Union

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

    @property
    def attributes(self) -> Dict[str, Any]:
        """dict: Additional attributes, which are required to restore the state"""
        return {}

    @property
    def _attributes_store(self) -> Dict[str, Any]:
        """dict: Attributes in the form in which they will be written to storage"""
        attrs = self.attributes.copy()
        attrs["__class__"] = self.__class__.__name__
        attrs["__version__"] = self._format_version
        return attrs

    @property
    def _data_store(self) -> Any:
        """attribute that determines what data is stored in this state"""
        if hasattr(self, "data"):
            return self.data
        else:
            return None

    def __init_subclass__(cls, **kwargs):  # @NoSelf
        """register all subclassess to reconstruct them later"""
        super().__init_subclass__(**kwargs)
        cls._state_classes[cls.__name__] = cls

    def __eq__(self, other):
        return _equals(self, other)

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
            format_version = attributes.get("__version__", 0)
            if format_version != cls._format_version:
                warnings.warn(
                    f"File format version mismatch ({format_version} != "
                    f"{cls._format_version})"
                )
            if len(attributes) > 2:  # __class__ and __version__ expected
                warnings.warn(
                    f"Unused attributes, but {cls.__name__} did not implemented custom "
                    "from_state"
                )
            return cls() if data is None else cls(data)  # type: ignore

        else:
            raise ValueError(f"Incompatible state class {attributes['class']}")

    def copy(self):
        return self.__class__.from_state(self.attributes, copy.deepcopy(self.data))

    def _write_zarr_attributes(
        self, element: zarrElement, attrs: Optional[Dict[str, Any]] = None
    ) -> zarrElement:
        """prepare the zarr element for this state"""
        # write the attributes of the state
        element.attrs.update(self._attributes_store)

        # write additional attributes provided as argument
        if attrs is not None:
            element.attrs.update(attrs)

        return element

    def _write_zarr_data(self, zarr_group: zarr.Group, **kwargs) -> zarrElement:
        raise NotImplementedError

    def _write_zarr(
        self, zarr_group: zarr.Group, attrs: Optional[Dict[str, Any]] = None, **kwargs
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
            state_cls = cls._state_classes[content["attributes"]["__class__"]]
            return state_cls._from_simple_objects(content, state_cls=state_cls)
        else:
            return state_cls.from_state(content["attributes"], content["data"])

    def _to_simple_objects(self):
        """return object data suitable for encoding as text"""
        return {"attributes": self._attributes_store, "data": self._data_store}


class ObjectState(StateBase):
    """State characterized by a serializable python object"""

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
    ):
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

    @classmethod
    def _from_simple_objects(
        cls, content, *, state_cls: Optional[StateBase] = None
    ) -> StateBase:
        """create state from text data

        Args:
            content: The loaded data
        """
        if state_cls is None:
            return super()._from_simple_objects(content)

        return cls(np.asarray(content["data"]))


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
            self.labels = tuple(str(i) for i in range(len(self.data)))
        else:
            assert len(self.data) == len(labels) == len(set(labels))
            self.labels = tuple(labels)

    def __eq__(self, other):
        return (
            self.__class__ == other.__class__
            and len(self.data) == len(other.data)
            and all(np.array_equal(s, o) for s, o in zip(self.data, other.data))
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
    """State characterized by a dictionary of data"""

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
=======
Classes that describe the state of a simulation at each point in time

.. codeauthor:: David Zwicker <david.zwicker@ds.mpg.de>
"""

from __future__ import annotations

from typing import Any, Dict, Iterator, Tuple, Union

import numpy as np
import zarr

# States = attributes + data (shape: data_shape)
# Trajectory = attributes + time information + data (shape: n_times x data_shape)
# Result = model + state

# Examples:
# FieldBase: State
# SphericalDroplet, Emulsion: ArrayState (or State?)
# DropletTrack: Trajectory

zarrElement = Union[zarr.Group, zarr.Array]


class StateBase:
    """Base class for specifying degrees of freedom of a simulation"""

    _subclasses: Dict[str, StateBase] = {}  # type: ignore

    def __init_subclass__(cls, **kwargs):  # @NoSelf
        """register all subclassess to reconstruct them later"""
        super().__init_subclass__(**kwargs)
        cls._subclasses[cls.__name__] = cls

    @property
    def attributes(self) -> Dict[str, Any]:
        """dict: information about the element state, which does not change in time"""
        return {"class": self.__class__.__name__}

    @classmethod
    def _load_state(cls, zarr_element: zarrElement, *, index=...) -> StateBase:
        """create instance of correct subclass from data stored in zarr"""
        class_name = zarr_element.attrs["class"]
        state_cls = cls._subclasses[class_name]
        return state_cls._read(zarr_element, index=index)

    @classmethod
    def from_file(cls, store) -> StateBase:
        root = zarr.open_group(store, mode="r")
        return cls._read(root["data"])

    @classmethod
    def _read(cls, zarr_element: zarrElement, *, index=...) -> StateBase:
        """create instance of this class from data stored in zarr"""
        raise NotImplementedError

    def _update(self, element: zarrElement, *, index=...) -> None:
        raise NotImplementedError

    def _write_attributes(
        self, element: zarrElement, attrs: Dict[str, Any] = None
    ) -> zarrElement:
        """prepare the zarr element for this state"""
        # write additional attributes
        attributes = self.attributes
        if attrs is not None:
            if "class" in attrs:
                raise ValueError("`class` attribute is reserved")
            attributes.update(attrs)

        for k, v in attributes.items():
            element.attrs[k] = v
        return element

    def _write(self, element: zarrElement, attrs: Dict[str, Any] = None) -> zarrElement:
        raise NotImplementedError

    def to_file(self, store, *, attrs: Dict[str, Any] = None, overwrite: bool = False):
        """write this state to a file"""
        with zarr.group(store=store, overwrite=overwrite) as group:
            self._write(group, attrs)

    def _prepare_trajectory(
        self, element: zarrElement, attrs: Dict[str, Any] = None
    ) -> zarrElement:
        """prepare the zarr element for this state"""
        raise NotImplementedError

    def _append_to_trajectory(self, element: zarrElement) -> None:
        """append current data to the prepared zarr element"""
        raise NotImplementedError


class ArrayState(StateBase):
    """Data characterized by a single numpy array"""

    def __init__(self, data: np.ndarray):
        self.data = data

    @classmethod
    def _read(cls, zarr_element: zarr.Array, *, index=...) -> ArrayState:
        return cls(zarr_element[index])

    def _update(self, element: zarrElement, *, index=...) -> None:
        self.data = element[index]

    def _write(
        self,
        zarr_group: zarr.Group,
        attrs: Dict[str, Any] = None,
        *,
        label: str = "data",
    ):
        zarr_element = zarr_group.array(label, self.data)
        super()._write_attributes(zarr_element, attrs)
        return zarr_element

    def _prepare_trajectory(
        self,
        zarr_group: zarr.Group,
        attrs: Dict[str, Any] = None,
        *,
        label: str = "data",
    ) -> zarr.Array:
        """prepare the zarr storage for this state"""
        zarr_element = zarr_group.zeros(
            label,
            shape=(0,) + self.data.shape,
            chunks=(1,) + self.data.shape,
            dtype=self.data.dtype,
        )
        self._write_attributes(zarr_element, attrs)

        return zarr_element

    def _append_to_trajectory(self, zarr_element: zarr.Array) -> None:
        """append current data to a stored element"""
        zarr_element.append([self.data])


class StateCollection(StateBase):
    """Data characterized by a collection of data"""

    def __init__(self, data: Union[Dict[str, StateBase], Tuple[StateBase]]):

        if isinstance(data, dict):
            self.keys = list(data.keys())
            self.data = list(data.values())

        else:
            self.keys = [str(i) for i in range(len(data))]
            self.data = data

    def __getitem__(self, index: Union[int, str]) -> StateBase:
        if isinstance(index, str):
            return self.data[self.keys.index(index)]
        else:
            return self.data[index]

    @property
    def attributes(self) -> Dict[str, Any]:
        """dict: information about the element state, which does not change in time"""
        attributes = super().attributes
        attributes["keys"] = self.keys
        return attributes

    @classmethod
    def _read(cls, zarr_element: zarr.Array, *, index=...) -> ArrayState:
        keys = zarr_element.attrs["keys"]
        data = {
            label: StateBase._load_state(zarr_element[label], index=index)
            for label in keys
        }
        return cls(data)

    def _update(self, element: zarrElement, *, index=...) -> None:
        for key in self.keys:
            self.data._update(element[key], index=index)

    def _write(
        self,
        zarr_group: zarr.Group,
        attrs: Dict[str, Any] = None,
        *,
        label: str = "data",
    ):
        zarr_subgroup = zarr_group.create_group(label)
        self._write_attributes(zarr_subgroup, attrs)
        for label, data in zip(self.keys, self.data):
            data._write(zarr_subgroup, label=label)
        return zarr_subgroup

    def _prepare_trajectory(
        self,
        zarr_group: zarr.Group,
        attrs: Dict[str, Any] = None,
        *,
        label: str = "data",
    ) -> zarr.Group:
        """prepare the zarr storage for this state"""
        zarr_subgroup = zarr_group.create_group(label)
        for label, data in zip(self.keys, self.data):
            data._prepare_trajectory(zarr_subgroup, label=label)

        self._write_attributes(zarr_subgroup, attrs)

        return zarr_subgroup

    def _append_to_trajectory(self, zarr_group: zarr.Group) -> None:
        """append current data to a stored element"""
        for label, data in zip(self.keys, self.data):
            data._append_to_trajectory(zarr_group[label])


class TrajectoryWriter:
    """

    Example:
        # explicit use
        writer = trajectory_writer("test.zarr")
        writer.append(data0)
        writer.append(data1)
        writer.close()

        # context manager
        with trajectory_writer("test.zarr") as write:
            for t, data in simulation:
                write(data, t)
    """

    def __init__(self, store, *, attrs: Dict[str, Any] = None, overwrite: bool = False):
        self._attrs = attrs
        self._root = zarr.group(store, overwrite=overwrite)
        self.times = self._root.zeros("times", shape=(0,), chunks=(1,))

    def append(self, data: StateBase, time: float = None) -> None:
        if "data" not in self._root:
            data._prepare_trajectory(self._root, attrs=self._attrs, label="data")

        if time is None:
            time = 0 if len(self.times) == 0 else self.times[-1] + 1

        self.times.append([time])
        data._append_to_trajectory(self._root["data"])

    def close(self):
        self._root.store.close()

    def __enter__(self):
        return self.append

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()


class Trajectory:
    def __init__(self, store, ret_copy: bool = True):
        self.ret_copy = ret_copy
        self._root = zarr.open_group(store, mode="r")
        self.times = np.array(self._root["times"])
        self._state = None

    def __len__(self) -> int:
        return len(self.times)

    def _get_state(self, t_index: int) -> StateBase:
        """return the data object corresponding to the given time index

        Load the data given an index, i.e., the data at time `self.times[t_index]`.

        Args:
            t_index (int):
                The index of the data to load

        Returns:
            :class:`~StateBase`: The requested state
        """
        if t_index < 0:
            t_index += len(self)

        if not 0 <= t_index < len(self):
            raise IndexError("Time index out of range")

        if self.ret_copy or self._state is None:
            # create the state with the data of the given index
            self._state = StateBase._load_state(self._root["data"], index=t_index)

        else:
            # update the state with the data of the given index
            self._state._update(self._root["data"], index=t_index)
        return self._state

    def __getitem__(self, key: Union[int, slice]) -> Union[StateBase, Trajectory]:
        """return field at given index or a list of fields for a slice"""
        if isinstance(key, int):
            return self._get_state(key)
        # elif isinstance(key, slice):
        #     # TODO: implement slicing of zarr
        #     return [self._get_field(i) for i in range(*key.indices(len(self)))]
        else:
            raise TypeError("Unknown key type")

    def __iter__(self) -> Iterator[StateBase]:
        """iterate over all stored fields"""
        for i in range(len(self)):
            yield self[i]  # type: ignore
>>>>>>> a84780e First version of a general state class
