"""
Classes that describe the state of a simulation over time

.. autosummary::
   :nosignatures:

   TrajectoryWriter
   Trajectory

.. codeauthor:: David Zwicker <david.zwicker@ds.mpg.de>
"""

from __future__ import annotations

from typing import Any, Dict, Iterator, Optional

import numpy as np

from ..storage.group import Group
from ..storage.tools import open_storage
from ..storage.utils import KeyType, encode_class, storage_actions
from ..storage.access import AccessType
from .base import StateBase, _get_state_cls_from_storage


class TrajectoryWriter:
    """writes trajectories of states using :mod:`zarr`

    Stored data can then be read using :class:`Trajectory`.

    Example:

        .. code-block:: python

            # write data using context manager
            with TrajectoryWriter("test.zarr", key="trajectory") as writer:
                for t, data in simulation:
                    writer.append(data, t)

            # write using explicit class interface
            writer = TrajectoryWriter("test.zarr", overwrite=True)
            writer.append(data0)
            writer.append(data1)
            writer.close()

            # read data
            trajectory = Trajectory("test.zarr")
            assert trajectory[0] == data0
    """

    def __init__(
        self,
        storage,
        key: KeyType = "trajectory",
        *,
        attrs: Optional[Dict[str, Any]] = None,
        access: AccessType = "insert",
    ):
        """
        Args:
            store (MutableMapping or string):
                Store or path to directory in file system or name of zip file.
            attrs (dict):
                Additional attributes stored in the trajectory. The attributes of the
                state are also stored in any case.
            overwrite (bool):
                If True, delete all pre-existing data in store.
        """
        # create the root group where we store all the data
        storage = open_storage(storage, access=access)
        self._group = storage.create_group(key)
        self._group.write_attrs(None, {"__class__": encode_class(Trajectory)})

        # make sure we don't overwrite data
        if "times" in self._group or "data" in self._group:
            raise IOError("Storage already contains data")

        if attrs is not None:
            self._group.write_attrs(attrs=attrs)

    def append(self, data: StateBase, time: Optional[float] = None) -> None:
        if "data" not in self._group:
            data._state_create_trajectory(self._group, "data")
            self._group.create_dynamic_array("time", shape=tuple(), dtype=float)
            if time is None:
                time = 0.0
        else:
            if time is None:
                time = float(self._group.read_array("time", index=-1)) + 1.0

        data._state_append_to_trajectory(self._group, "data")
        self._group.extend_dynamic_array("time", time)

    def close(self):
        self._group._storage.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()


class Trajectory:
    """Reads trajectories of states written with :class:`TrajectoryWriter`

    The class permits direct access to indivdual states using the square bracket
    notation. It is also possible to directly iterate over all states.

    Attributes:
        times (:class:`~numpy.ndarray`): Time points at which data is available
    """

    def __init__(self, storage, key: KeyType = "trajectory", *, ret_copy: bool = True):
        """
        Args:
            storage (MutableMapping or string):
                Store or path to directory in file system or name of zip file.
            ret_copy (bool):
                If True, copies of states are returned, e.g., when iterating. If the
                returned state is not modified, this flag can be set to False to
                accelerate the processing. In any case, the state data is always loaded
                from the store, so this setting only affects attributes.
        """
        self.ret_copy = ret_copy

        # open the storage
        storage = open_storage(storage, access="readonly")
        self._storage = Group(storage, key)

        # read some intial data from storage
        self.times = self._storage.read_array("time")
        self._state: Optional[StateBase] = None
        self._state_cls = _get_state_cls_from_storage(self._storage, "data")

        # check temporal ordering
        assert np.all(np.diff(self.times) > 0)

    @property
    def _state_attributes(self) -> Dict[str, Any]:
        """dict: information about the trajectory"""
        attrs = self._storage.attrs
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
                self._storage, "data", index=t_index
            )

        else:
            # update the state with the data of the given index
            self._state._state_update_from_stored_data(
                self._storage, "data", index=t_index
            )

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
            yield self[i]  # type: ignore


storage_actions.register("read_object", Trajectory, Trajectory)
