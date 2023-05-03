"""
Classes that describe the state of a simulation over time

.. autosummary::
   :nosignatures:

   TrajectoryWriter
   Trajectory

.. codeauthor:: David Zwicker <david.zwicker@ds.mpg.de>
"""

from __future__ import annotations

from typing import Any, Dict, Iterator, Optional, Union

import numpy as np
import zarr

from .base import StateBase
from .io import normalize_zarr_store


class TrajectoryWriter:
    """writes trajectories of states using :mod:`zarr`

    Stored data can then be read using :class:`Trajectory`.

    Example:

        .. code-block:: python

            # write data using context manager
            with TrajectoryWriter("test.zarr") as write:
                for t, data in simulation:
                    write(data, t)

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
        self, store, *, attrs: Optional[Dict[str, Any]] = None, overwrite: bool = False
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
        self._root = zarr.group(
            normalize_zarr_store(store, mode="w" if overwrite else "a"),
            overwrite=overwrite,
        )

        # make sure we don't overwrite data
        if "times" in self._root or "data" in self._root:
            raise IOError("zarr Store already contains data")

        if attrs is not None:
            self._root.attrs.put(attrs)
        self.times = self._root.zeros("times", shape=(0,), chunks=(1,))

    def append(self, data: StateBase, time: Optional[float] = None) -> None:
        if "data" not in self._root:
            data._state_prepare_zarr_trajectory(self._root, label="data")

        if time is None:
            time = 0 if len(self.times) == 0 else self.times[-1] + 1

        self.times.append([time])
        data._state_append_to_zarr_trajectory(self._root["data"])

    def close(self):
        self._root.store.close()

    def __enter__(self):
        return self.append

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()


class Trajectory:
    """Reads trajectories of states written with :class:`TrajectoryWriter`

    The class permits direct access to indivdual states using the square bracket
    notation. It is also possible to directly iterate over all states.

    Attributes:
        times (:class:`~numpy.ndarray`): Time points at which data is available
    """

    def __init__(self, store, *, ret_copy: bool = True):
        """
        Args:
            store (MutableMapping or string):
                Store or path to directory in file system or name of zip file.
            ret_copy (bool):
                If True, copies of states are returned, e.g., when iterating. If the
                returned state is not modified, this flag can be set to False to
                accelerate the processing. In any case, the state data is always loaded
                from the store, so this setting only affects attributes.
        """
        self.ret_copy = ret_copy
        self._root = zarr.open_group(normalize_zarr_store(store), mode="r")
        self.times = np.array(self._root["times"])
        self._state: Optional[StateBase] = None

        # check temporal ordering
        assert np.all(np.diff(self.times) > 0)

    @property
    def _state_attributes(self) -> Dict[str, Any]:
        """dict: information about the trajectory"""
        return self._root.attrs.asdict()  # type: ignore

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
            self._state = StateBase._from_zarr(self._root["data"], index=t_index)

        else:
            # update the state with the data of the given index
            self._state._state_update_from_zarr(self._root["data"], index=t_index)

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
