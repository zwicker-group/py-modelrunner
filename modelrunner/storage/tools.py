"""
.. codeauthor:: David Zwicker <david.zwicker@ds.mpg.de>
"""

from pathlib import Path
from typing import Union
from .base import StorageBase
from .group import Group
from .backend import AVAILABLE_STORAGE, MemoryStorage

StorageID = Union[None, str, Path, StorageBase]


def _open_storage(storage: StorageID = None, **kwargs) -> StorageBase:
    """guess the format of a given store

    Args:
        store (str or :class:`zarr.Store`):
            Path or instance describing the storage, which is either a file path or
            a :class:`zarr.Storage`.
        fmt (str):
            Explicit file format. Determined from `store` if omitted.

    Returns:
        str: The store format
    """
    if isinstance(storage, StorageBase):
        return storage

    elif storage is None:
        return MemoryStorage(**kwargs)

    elif isinstance(storage, (str, Path)):
        # guess format from path extension
        storage = Path(storage)
        if storage.is_dir():
            from .backend.zarr import ZarrStorage

            return ZarrStorage(storage, **kwargs)

        extension = storage.suffix.lower()
        for storage_cls in AVAILABLE_STORAGE:
            for ext in storage_cls.extensions:
                if extension == "." + ext:
                    return storage_cls(storage, **kwargs)

        raise TypeError(f"Unsupported store with extension {extension}")

    raise TypeError(f"Unsupported store type {storage.__class__.__name__}")


def open_storage(storage: StorageID = None, **kwargs) -> Group:
    return Group(_open_storage(storage))
