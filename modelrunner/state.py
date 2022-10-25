"""
Classes that describe the state of a simulation at each point in time

.. codeauthor:: David Zwicker <david.zwicker@ds.mpg.de>
"""

from __future__ import annotations

import itertools
from typing import Any, Dict, Iterator, Tuple, Union

import numcodecs
import numpy as np
import zarr

# States = attributes + data (shape: data_shape)
#     attributes are a dictionary with immutable values
#     data is mutable and often a numpy array or a collection of numpy arrays
# Trajectory = attributes + time information + data (shape: n_times x data_shape)
# Result = model + state

# Examples:
# Automated model wrapping: ObjectState
# FieldBase: ArrayState
# SphericalDroplet, Emulsion: ArrayState
# DropletTrack: Trajectory
# sim.State: DictState

zarrElement = Union[zarr.Group, zarr.Array]


def rich_comparison(left: Any, right: Any) -> bool:
    """checks whether two objects are equal, also supporting :class:~numpy.ndarray`

    Args:
        left: one object
        right: other object

    Returns:
    bool: Whether the two objects are equal
    """
    if left.__class__ is not right.__class__:
        return False
    if isinstance(left, np.ndarray):
        return np.array_equal(left, right)
    elif isinstance(left, dict):
        if left.keys() != right.keys():
            return False
        return rich_comparison(left.values(), right.values())
    elif hasattr(left, "__iter__"):
        return any(rich_comparison(l, r) for l, r in zip(left, right))
    else:
        return left == right


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

    def __eq__(self, other):
        if self.attributes != other.attributes:
            return False
        if hasattr(self, "data"):
            return rich_comparison(self.data, other.data)
        return True

    @classmethod
    def _load_state(cls, zarr_element: zarrElement, *, index=...) -> StateBase:
        """create instance of correct subclass from data stored in zarr"""
        class_name = zarr_element.attrs["class"]
        state_cls = cls._subclasses[class_name]
        return state_cls._read(zarr_element, index=index)

    @classmethod
    def from_state(cls, attributes: Dict[str, Any], data=None):
        if cls.__name__ == "StateBase":
            state_cls = cls._subclasses[attributes["class"]]
            return state_cls.from_state(attributes, data)
        elif cls.__name__ != attributes["class"]:
            raise ValueError(f"Incompatible state class {attributes['class']}")
        return cls()

    @classmethod
    def from_file(cls, store) -> StateBase:
        root = zarr.open_group(store, mode="r")
        return cls._read(root["data"])

    def copy(self):
        if hasattr(self, "data"):
            return self.__class__.from_state(self.attributes, self.data.copy())
        else:
            return self.__class__.from_state(self.attributes)

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

    def _write(
        self, element: zarrElement, attrs: Dict[str, Any] = None, **kwargs
    ) -> zarrElement:
        raise NotImplementedError

    def to_file(
        self, store, *, attrs: Dict[str, Any] = None, overwrite: bool = False, **kwargs
    ):
        """write this state to a file"""
        with zarr.group(store=store, overwrite=overwrite) as group:
            self._write(group, attrs, **kwargs)

    def _prepare_trajectory(
        self, element: zarrElement, attrs: Dict[str, Any] = None, **kwargs
    ) -> zarrElement:
        """prepare the zarr element for this state"""
        raise NotImplementedError

    def _append_to_trajectory(self, element: zarrElement) -> None:
        """append current data to the prepared zarr element"""
        raise NotImplementedError


class ObjectState(StateBase):
    """State characterized by a serializable python object"""

    default_codec = numcodecs.Pickle()

    def __init__(self, data: Any):
        self.data = data

    @classmethod
    def from_state(cls, attributes: Dict[str, Any], data: np.ndarray):
        assert cls.__name__ == attributes["class"]
        return cls(data)

    @classmethod
    def _read(cls, zarr_element: zarr.Array, *, index=...) -> ArrayState:
        if zarr_element.shape == () and index is ...:
            return cls(zarr_element[index].item())
        else:
            return cls(zarr_element[index])

    def _update(self, element: zarrElement, *, index=...) -> None:
        self.data = element[index].item()

    def _write(
        self,
        zarr_group: zarr.Group,
        attrs: Dict[str, Any] = None,
        *,
        label: str = "data",
        codec: numcodecs.abc.Codec = None,
    ):
        if codec is None:
            codec = self.default_codec
        zarr_element = zarr_group.array(
            label, self.data, shape=(0,), dtype=object, object_codec=codec
        )
        super()._write_attributes(zarr_element, attrs)
        return zarr_element

    def _prepare_trajectory(
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
        self._write_attributes(zarr_element, attrs)

        return zarr_element

    def _append_to_trajectory(self, zarr_element: zarr.Array) -> None:
        """append current data to a stored element"""
        zarr_element.resize(len(zarr_element) + 1)
        zarr_element[-1] = self.data


class ArrayState(StateBase):
    """State characterized by a single numpy array"""

    def __init__(self, data: np.ndarray):
        self.data = data

    def __eq__(self, other):
        return np.array_equal(self.data, other.data)

    @classmethod
    def from_state(cls, attributes: Dict[str, Any], data: np.ndarray):
        assert cls.__name__ == attributes["class"]
        return cls(data)

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
        **kwargs,
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
        **kwargs,
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


class DictState(StateBase):
    """State characterized by a dictionary of data"""

    def __init__(self, data: Union[Dict[str, StateBase], Tuple[StateBase]]):

        if isinstance(data, dict):
            self.data = data
        else:
            self.data = {str(i): v for i, v in enumerate(data)}

    @classmethod
    def from_state(
        cls,
        attributes: Dict[str, Any],
        data: Union[Dict[str, StateBase], Tuple[StateBase]],
    ):
        assert cls.__name__ == attributes["class"]
        if not isinstance(data, dict) and "keys" in attributes:
            data = {k: v for k, v in zip(attributes["keys"], data)}
        return cls(data)

    @property
    def attributes(self) -> Dict[str, Any]:
        """dict: information about the element state, which does not change in time"""
        attributes = super().attributes
        attributes["keys"] = list(self.data.keys())
        return attributes

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
    def _read(cls, zarr_element: zarr.Array, *, index=...) -> ArrayState:
        keys = zarr_element.attrs["keys"]
        data = {
            label: StateBase._load_state(zarr_element[label], index=index)
            for label in keys
        }
        return cls(data)

    def _update(self, element: zarrElement, *, index=...) -> None:
        for key, substate in self.data.items():
            substate._update(element[key], index=index)

    def _write(
        self,
        zarr_group: zarr.Group,
        attrs: Dict[str, Any] = None,
        *,
        label: str = "data",
        **kwargs,
    ):
        zarr_subgroup = zarr_group.create_group(label)
        self._write_attributes(zarr_subgroup, attrs)
        for label, substate in self.data.items():
            substate._write(zarr_subgroup, label=label, **kwargs)
        return zarr_subgroup

    def _prepare_trajectory(
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
            substate._prepare_trajectory(zarr_subgroup, label=label, **kwargs)

        self._write_attributes(zarr_subgroup, attrs)

        return zarr_subgroup

    def _append_to_trajectory(self, zarr_group: zarr.Group) -> None:
        """append current data to a stored element"""
        for label, substate in self.data.items():
            substate._append_to_trajectory(zarr_group[label])


class TrajectoryWriter:
    """allows writing trajectories of states

    Example:

        .. code-block:: python

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
    """Collection of states with identical type for successive time points"""

    def __init__(self, store, ret_copy: bool = True):
        self.ret_copy = ret_copy
        self._root = zarr.open_group(store, mode="r")
        self.times = np.array(self._root["times"])
        self._state = None

        # check temporal ordering
        assert np.all(np.diff(self.times) > 0)

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
        else:
            raise TypeError("Unknown key type")

    def __iter__(self) -> Iterator[StateBase]:
        """iterate over all stored fields"""
        for i in range(len(self)):
            yield self[i]  # type: ignore
