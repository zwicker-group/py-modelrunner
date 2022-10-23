"""
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
