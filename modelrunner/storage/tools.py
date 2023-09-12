"""
.. codeauthor:: David Zwicker <david.zwicker@ds.mpg.de>
"""

from contextlib import contextmanager
from pathlib import Path
from typing import Union

from .backend import AVAILABLE_STORAGE, MemoryStorage
from .base import StorageBase
from .group import Group

StorageID = Union[None, str, Path, Group, StorageBase]


def _open_storage(storage: StorageID = None, **kwargs) -> StorageBase:
    """guess the format of a given store

    Args:
        store (str or :class:`zarr.Store`):
            Path or instance describing the storage, which is either a file path or
            a :class:`zarr.Storage`.
        fmt (str):
            Explicit file format. Determined from `store` if omitted.

    Returns:
        :class:`StorageBase`: The storage
    """
    if isinstance(storage, StorageBase):
        return storage

    elif isinstance(storage, Group):
        return storage._storage

    elif storage is None:
        return MemoryStorage(**kwargs)

    elif isinstance(storage, (str, Path)):
        # guess format from path extension
        path = Path(storage)
        if path.suffix == "":
            # path seems to be a directory
            from .backend.zarr import ZarrStorage

            return ZarrStorage(path, **kwargs)

        else:
            # path seems to be a file
            extension = path.suffix.lower()
            for storage_cls in AVAILABLE_STORAGE:
                for ext in storage_cls.extensions:
                    if extension == "." + ext:
                        return storage_cls(path, **kwargs)

            raise TypeError(f"Unsupported store with extension `{extension}`")

    raise TypeError(f"Unsupported store type {storage.__class__.__name__}")


def open_storage(storage: StorageID = None, **kwargs) -> Group:
    return Group(_open_storage(storage, **kwargs))


@contextmanager
def opened_storage(storage: StorageID = None, **kwargs):
    group = open_storage(storage, **kwargs)
    yield group
    group._storage.close()
