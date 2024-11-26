"""Classes that describe time-dependent data, i.e., trajectories.

.. autosummary::
   :nosignatures:

   TrajectoryWriter
   Trajectory

.. codeauthor:: David Zwicker <david.zwicker@ds.mpg.de>
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Iterator, Literal

import numpy as np

from .access_modes import AccessError, ModeType
from .base import StorageBase
from .group import StorageGroup
from .tools import StorageID, open_storage
from .utils import Location, storage_actions


class TrajectoryWriter:
    """Writes trajectories into a storage.

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

    _item_type: Literal["array", "object"]

    def __init__(
        self,
        storage: StorageID,
        loc: Location = "trajectory",
        *,
        attrs: dict[str, Any] | None = None,
        mode: ModeType | None = None,
    ):
        """
        Args:
            store:
                Store or path to directory in file system or name of zip file.
            loc (str or list of str):
                The location in the storage where the trajectory data is written.
            attrs (dict):
                Additional attributes stored in the trajectory.
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

        self._storage = open_storage(storage, mode=mode)

        if self._storage.mode.insert:
            self._trajectory = self._storage.create_group(loc, cls=Trajectory)
        elif self._storage.mode.dynamic_append:
            self._trajectory = StorageGroup(self._storage, loc)
        else:
            raise AccessError("Cannot insert data. Open storage with write access")

        # make sure we don't overwrite data
        if "times" in self._trajectory or "data" in self._trajectory:
            if not self._storage.mode.dynamic_append:
                raise OSError("Storage already contains data and we cannot append")
            self._item_type = self._trajectory.attrs["item_type"]

        if attrs is not None:
            self._trajectory.write_attrs(attrs=attrs)

    @property
    def times(self) -> np.ndarray:
        """:class:`~numpy.ndarray`: Time points written so far."""
        return self._trajectory.read_array("time")

    def append(self, data: Any, time: float | None = None) -> None:
        """Append data to the trajectory.

        Args:
            data:
                The data to append to the trajectory
            time (float, optional):
                The associated time point. If omitted, the last time point is
                incremented by one.
        """
        if "data" not in self._trajectory:
            # initialize new trajectory
            if isinstance(data, np.ndarray):
                dtype = data.dtype
                shape: tuple[int, ...] = data.shape
                self._item_type = "array"
            else:
                dtype = object
                shape = ()
                self._item_type = "object"
            self._trajectory.create_dynamic_array("data", shape=shape, dtype=dtype)
            self._trajectory.create_dynamic_array("time", shape=(), dtype=float)
            self._trajectory.write_attrs(None, {"item_type": self._item_type})
            if time is None:
                time = 0.0
        else:
            if time is None:
                time = float(self._trajectory.read_array("time", index=-1)) + 1.0

        if self._item_type == "array":
            self._trajectory.extend_dynamic_array("data", data)
        elif self._item_type == "object":
            arr: np.ndarray = np.empty((), dtype=object)
            arr[...] = data
            self._trajectory.extend_dynamic_array("data", arr)
        else:
            raise NotImplementedError
        self._trajectory.extend_dynamic_array("time", time)

    def close(self):
        self._storage.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()


class Trajectory:
    """Reads trajectories written with :class:`TrajectoryWriter`

    The class permits direct access to indivdual items in the trajcetory using the
    square bracket notation. It is also possible to iterate over all items.

    Attributes:
        times (:class:`~numpy.ndarray`): Time points at which data is available
    """

    _item_type: Literal["array", "object"]

    def __init__(self, storage: StorageGroup, loc: Location = "trajectory"):
        """
        Args:
            storage (MutableMapping or string):
                Store or path to directory in file system or name of zip file.
            loc (str or list of str):
                The location in the storage where the trajectory data is read.
        """
        # open the storage
        self._storage = open_storage(storage, mode="read")
        self._loc = self._storage._get_loc(loc)
        # self._storage = storage#StorageGroup(storage, loc)

        # read some intial data from storage
        self.attrs = self._storage.read_attrs(self._loc)
        self._item_type = self.attrs["item_type"]
        self.times = self._storage.read_array(self._loc + ["time"])

        # check temporal ordering
        if np.any(np.diff(self.times) < 0):
            raise ValueError(f"Times are not monotonously increasing: {self.times}")

    def close(self) -> None:
        """Close the openend storage."""
        self._storage.close()

    def __len__(self) -> int:
        return len(self.times)

    def _get_item(self, t_index: int) -> Any:
        """Return the data object corresponding to the given time index.

        Load the data given an index, i.e., the data at time `self.times[t_index]`.

        Args:
            t_index (int):
                The index of the data to load

        Returns:
            The requested item
        """
        if t_index < 0:
            t_index += len(self)

        if not 0 <= t_index < len(self):
            raise IndexError("Time index out of range")

        res = self._storage.read_array(self._loc + ["data"], index=t_index)
        if self._item_type == "array":
            return res
        elif self._item_type == "object":
            return res.item()
        else:
            raise NotImplementedError

    def __getitem__(self, key: int) -> Any:
        """Return field at given index or a list of fields for a slice."""
        if isinstance(key, int):
            return self._get_item(key)
        else:
            raise TypeError("Unknown key type")

    def __iter__(self) -> Iterator[Any]:
        """Iterate over all stored fields."""
        for i in range(len(self)):
            yield self[i]


storage_actions.register("read_item", Trajectory, Trajectory)
