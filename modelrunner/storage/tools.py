"""Functions that provide convenience on top of the storage classes.

.. codeauthor:: David Zwicker <david.zwicker@ds.mpg.de>
"""

from __future__ import annotations

from pathlib import Path
from typing import Union

from .access_modes import AccessMode
from .backend import AVAILABLE_STORAGE, MemoryStorage
from .base import StorageBase
from .group import StorageGroup
from .utils import Location

StorageID = Union[None, str, Path, StorageGroup, StorageBase]


class open_storage(StorageGroup):
    """Open a storage and return the root :class:`StorageGroup`

    Example:
        This can be either used like a function

        .. code-block:: python

            storage = open_storage(...)
            # use the storage
            storage.close()

        or as a context manager

        .. code-block:: python

            with open_storage(...) as storage:
                # use the storage
    """

    def __init__(
        self,
        storage: StorageID = None,
        *,
        loc: Location = None,
        **kwargs,
    ):
        r"""
        Args:
            storage:
                The path to a file or directory or a :class:`StorageBase` instance. The
                special value `None` creates a
                :class:`~modelrunner.storage.backend.memory.MemoryStorage`
            loc (str or list of str):
                Denotes the location that will be opened within the storage. The default
                `None` opens the root group of the storage.
            mode (str or :class:`~modelrunner.storage.access_modes.ModeType`):
                The file mode with which the storage is accessed, which determines the
                allowed operations. Common options are "read", "full", "append", and
                "truncate".
            **kwargs:
                All other arguments are passed on to the storage class
        """
        store_obj: StorageBase | None = None

        if isinstance(storage, StorageBase):
            # storage is of type `StorageBase`
            self._close = False
            store_obj = storage

        elif isinstance(storage, StorageGroup):
            # storage is a group and we open a sub-group instead
            self._close = False
            store_obj = storage._storage
            loc = storage.loc + [loc]

        elif storage is None:
            self._close = True
            store_obj = MemoryStorage(**kwargs)

        elif isinstance(storage, (str, Path)):
            # guess format from path extension
            self._close = True
            path = Path(storage)
            if path.suffix == "":
                # path seems to be a directory
                from .backend.zarr import ZarrStorage

                store_obj = ZarrStorage(path, **kwargs)

            else:
                # path seems to be a file
                extension = path.suffix.lower()
                for storage_cls in AVAILABLE_STORAGE:
                    for ext in storage_cls.extensions:
                        if extension == "." + ext:
                            store_obj = storage_cls(path, **kwargs)  # type: ignore
                            break
                    if store_obj is not None:
                        break
                if store_obj is None:
                    raise TypeError(f"Unsupported store with extension `{extension}`")

        elif (
            storage.__class__.__name__ == "File"
            and storage.__class__.__module__.split(".", 1)[0] == "h5py"
        ):
            # looks like an opened h5py file
            from .backend.hdf import HDFStorage

            self._close = False
            store_obj = HDFStorage(storage, **kwargs)

        elif (
            storage.__class__.__name__ == "Group"
            and storage.__class__.__module__.split(".", 1)[0] == "zarr"
        ):
            # looks like an opened zarr group
            import zarr

            from .backend.zarr import ZarrStorage

            if isinstance(storage, zarr.Group):
                self._close = False
                store_obj = ZarrStorage(storage._store, **kwargs)
                loc = [storage.path] + [loc]

        if store_obj is None:
            raise TypeError(f"Unsupported store type {storage.__class__.__name__}")
        assert isinstance(store_obj, StorageBase)

        super().__init__(store_obj, loc=loc)
        self._closed = False

    def close(self) -> None:
        """Close the storage (and flush all data to persistent storage if necessary)"""
        if self._close:
            self._storage.close()
        else:
            self._storage.flush()
        self._closed = True

    @property
    def closed(self) -> bool:
        """bool: determines whether the storage group has been closed"""
        return self._closed

    @property
    def mode(self) -> AccessMode:
        """:class:`~modelrunner.storage.access_modes.AccessMode`: access mode."""
        return self._storage.mode

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
