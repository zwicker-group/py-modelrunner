"""
.. codeauthor:: David Zwicker <david.zwicker@ds.mpg.de>
"""

from pathlib import Path
from typing import Union

from .backend import AVAILABLE_STORAGE, MemoryStorage
from .base import ModeType, StorageBase
from .group import StorageGroup

StorageID = Union[None, str, Path, StorageGroup, StorageBase]


def _open_storage(
    storage: StorageID = None, *, mode: ModeType = "insert", **kwargs
) -> StorageBase:
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

    elif isinstance(storage, StorageGroup):
        return storage._storage

    elif storage is None:
        return MemoryStorage(mode=mode, **kwargs)

    elif isinstance(storage, (str, Path)):
        # guess format from path extension
        path = Path(storage)
        if path.suffix == "":
            # path seems to be a directory
            from .backend.zarr import ZarrStorage

            return ZarrStorage(path, mode=mode, **kwargs)

        else:
            # path seems to be a file
            extension = path.suffix.lower()
            for storage_cls in AVAILABLE_STORAGE:
                for ext in storage_cls.extensions:
                    if extension == "." + ext:
                        return storage_cls(path, mode=mode, **kwargs)

            raise TypeError(f"Unsupported store with extension `{extension}`")

    raise TypeError(f"Unsupported store type {storage.__class__.__name__}")


class open_storage(StorageGroup):
    def __init__(
        self, storage: StorageID = None, *, mode: ModeType = "insert", **kwargs
    ):
        super().__init__(_open_storage(storage, mode=mode, **kwargs))

    def close(self):
        self._storage.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
