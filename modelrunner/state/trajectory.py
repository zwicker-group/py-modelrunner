"""
Classes that describe the state of a simulation over time

.. autosummary::
   :nosignatures:

   TrajectoryWriter
   Trajectory

.. codeauthor:: David Zwicker <david.zwicker@ds.mpg.de>
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Iterator, Optional, Type

import numpy as np

from modelrunner.storage.access_modes import ModeType

from ..storage.group import StorageGroup
from ..storage.tools import open_storage
from ..storage.utils import Location, decode_class, encode_class, storage_actions
from .base import StateBase


class TrajectoryWriter:
    """writes trajectories of states into a storage

    Stored data can then be read using :class:`Trajectory`.

    Example:

        .. code-block:: python

            # write data using context manager
            with TrajectoryWriter("test.zarr") as writer:
                for t, data in simulation:
                    writer.append(data, t)

            # append to same file using explicit class interface
            writer = TrajectoryWriter("test.zarr", mode="append")
            writer.append(data0)
            writer.append(data1)
            writer.close()

            # read data
            trajectory = Trajectory("test.zarr")
            assert trajectory[-1] == data1
    """

    def __init__(
        self,
        storage,
        loc: Location = "trajectory",
        *,
        attrs: Optional[Dict[str, Any]] = None,
        mode: Optional[ModeType] = None,
    ):
        """
        Args:
            store (MutableMapping or string):
                Store or path to directory in file system or name of zip file.
            loc (str or list of str):
                The location in the storage where the trajectory data is written.
            attrs (dict):
                Additional attributes stored in the trajectory. The attributes of the
                state are also stored in any case.
            mode (str or :class:`~modelrunner.storage.access_modes.AccessMode`):
                The file mode with which the storage is accessed. Determines allowed
                operations. The meaning of the special (default) value `None` depends on
                whether the file given by `store` already exists. If yes, a RuntimeError
                is raised, otherwise the choice corresponds to `mode="full"` and thus
                creates a new trajectory. If the file exists, use `mode="truncate"` to
                overwrite file or `mode="append"` to insert new data into the file.
        """
        # create the root group where we store all the data
        if mode is None:
            if isinstance(storage, (str, Path)) and Path(storage).exists():
                raise RuntimeError(
                    'Storage already exists. Use `mode="truncate"` to overwrite file '
                    'or `mode="append"` to insert new data into the file.'
                )
            mode = "full"

        storage = open_storage(storage, mode=mode)

        if storage._storage.mode.insert:
            self._trajectory = storage.create_group(loc, cls=Trajectory)
        else:
            self._trajectory = StorageGroup(storage, loc)

        # make sure we don't overwrite data
        if (
            "times" in self._trajectory
            or "data" in self._trajectory
            and not storage._storage.mode.dynamic_append
        ):
            raise IOError("Storage already contains data and we cannot append")

        if attrs is not None:
            self._trajectory.write_attrs(attrs=attrs)

    def append(self, data: StateBase, time: Optional[float] = None) -> None:
        """append data to the trajectory

        Args:
            data (`StateBase`):
                The state to append to the trajectory
            time (float, optional):
                The associated time point. If omitted, the last time point is
                incremented by one.
        """
        if "data" not in self._trajectory:
            data._state_create_trajectory(self._trajectory, "data")
            self._trajectory.create_dynamic_array("time", shape=tuple(), dtype=float)
            self._trajectory.write_attrs(
                None, {"state_class": encode_class(data.__class__)}
            )
            if time is None:
                time = 0.0
        else:
            if time is None:
                time = float(self._trajectory.read_array("time", index=-1)) + 1.0

        data._state_append_to_trajectory(self._trajectory, "data")
        self._trajectory.extend_dynamic_array("time", time)

    def close(self):
        self._trajectory._storage.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()


class Trajectory:
    """Reads trajectories of states written with :class:`TrajectoryWriter`

    The class permits direct access to indivdual states using the square bracket
    notation. It is also possible to iterate over all states.

    Attributes:
        times (:class:`~numpy.ndarray`): Time points at which data is available
    """

    _state_cls: Type[StateBase]

    def __init__(
        self,
        storage: StorageGroup,
        loc: Location = "trajectory",
        *,
        ret_copy: bool = True,
    ):
        """
        Args:
            storage (MutableMapping or string):
                Store or path to directory in file system or name of zip file.
            loc (str or list of str):
                The location in the storage where the trajectory data is read.
            ret_copy (bool):
                If True, copies of states are returned, e.g., when iterating. If the
                returned state is not modified, this flag can be set to False to
                accelerate the processing. In any case, the state data is always loaded
                from the store, so this setting only affects attributes.
        """
        self.ret_copy = ret_copy

        # open the storage
        storage = open_storage(storage, mode="readonly")
        self._trajectory = StorageGroup(storage, loc)

        # read some intial data from storage
        self.times = self._trajectory.read_array("time")
        self._state: Optional[StateBase] = None
        state_cls = decode_class(self._trajectory.attrs["state_class"])
        if state_cls is None:
            raise RuntimeError("State class could not be determined")
        else:
            self._state_cls = state_cls

        # check temporal ordering
        if np.any(np.diff(self.times) < 0):
            raise ValueError(f"Times are not monotonously increasing: {self.times}")

    def close(self) -> None:
        """close the openend storage"""
        self._trajectory._storage.close()

    @property
    def _state_attributes(self) -> Dict[str, Any]:
        """dict: information about the trajectory"""
        attrs = self._trajectory.attrs
        attrs.pop("__class__", None)
        return attrs

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
            self._state = self._state_cls._state_from_stored_data(
                self._trajectory, "data", index=t_index
            )

        else:
            # update the state with the data of the given index
            self._state._state_update_from_stored_data(
                self._trajectory, "data", index=t_index
            )

        assert self._state is not None
        return self._state

    def __getitem__(self, key: int) -> StateBase:
        """return field at given index or a list of fields for a slice"""
        if isinstance(key, int):
            return self._get_state(key)
        else:
            raise TypeError("Unknown key type")

    def __iter__(self) -> Iterator[StateBase]:
        """iterate over all stored fields"""
        for i in range(len(self)):
            yield self[i]


storage_actions.register("read_item", Trajectory, Trajectory)
