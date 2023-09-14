"""
.. codeauthor:: David Zwicker <david.zwicker@ds.mpg.de>
"""

from pathlib import Path
from typing import Union

from .backend import AVAILABLE_STORAGE, MemoryStorage
from .base import ModeType, StorageBase
from .group import StorageGroup

StorageID = Union[None, str, Path, StorageGroup, StorageBase]


# def _open_storage(
#     storage: StorageID = None, *, mode: ModeType = "readonly", **kwargs
# ) -> StorageBase:
#     """guess the format of a given store
#
#     Args:
#         store (str or :class:`zarr.Store`):
#             Path or instance describing the storage, which is either a file path or
#             a :class:`zarr.Storage`.
#         fmt (str):
#             Explicit file format. Determined from `store` if omitted.
#
#     Returns:
#         :class:`StorageBase`: The storage
#     """


class open_storage(StorageGroup):
    def __init__(
        self, storage: StorageID = None, *, mode: ModeType = "readonly", **kwargs
    ):
        store_obj: StorageBase = None
        if isinstance(storage, StorageBase):
            self._close = False
            store_obj: StorageBase = storage

        elif isinstance(storage, StorageGroup):
            self._close = False
            store_obj = storage._storage

        elif storage is None:
            self._close = True
            store_obj = MemoryStorage(mode=mode, **kwargs)

        elif isinstance(storage, (str, Path)):
            # guess format from path extension
            self._close = True
            path = Path(storage)
            if path.suffix == "":
                # path seems to be a directory
                from .backend.zarr import ZarrStorage

                store_obj = ZarrStorage(path, mode=mode, **kwargs)

            else:
                # path seems to be a file
                extension = path.suffix.lower()
                for storage_cls in AVAILABLE_STORAGE:
                    for ext in storage_cls.extensions:
                        if extension == "." + ext:
                            store_obj = storage_cls(path, mode=mode, **kwargs)
                            break
                    if store_obj is not None:
                        break
                if store_obj is None:
                    raise TypeError(f"Unsupported store with extension `{extension}`")

        else:
            raise TypeError(f"Unsupported store type {storage.__class__.__name__}")

        super().__init__(store_obj)

    def close(self):
        if self._close:
            self._storage.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.close:
            self.close()
