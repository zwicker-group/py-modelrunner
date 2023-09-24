"""
.. codeauthor:: David Zwicker <david.zwicker@ds.mpg.de>
"""

from pathlib import Path
from typing import Optional, Sequence, Union

from .backend import AVAILABLE_STORAGE, MemoryStorage
from .base import StorageBase
from .group import StorageGroup

StorageID = Union[None, str, Path, StorageGroup, StorageBase]


class open_storage(StorageGroup):
    """open a storage and return the root :class:`StorageGroup`

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
        loc: Union[None, str, Sequence[str]] = None,
        **kwargs,
    ):
        """
        Args:
            storage:
                The path to a file or directory or a :class:`StorageBase` instance
            loc (str or list of str):
                Denotes the location that will be opened within the storage. The default
                `None` opens the root group of the storage.
            mode (str or :class:`~modelrunner.storage.access_modes.AccessMode`):
                The file mode with which the storage is accessed, which determines the
                allowed operations. Common options are "readonly", "full", "append", and
                "truncate".
        """
        store_obj: Optional[StorageBase] = None
        if isinstance(storage, StorageBase):
            self._close = False
            store_obj = storage

        elif isinstance(storage, StorageGroup):
            self._close = False
            store_obj = storage._storage

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

        else:
            raise TypeError(f"Unsupported store type {storage.__class__.__name__}")

        super().__init__(store_obj, loc=loc)

    def close(self) -> None:
        """close the storage (and flush all data to persistent storage if necessary)"""
        if self._close:
            self._storage.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.close:
            self.close()
