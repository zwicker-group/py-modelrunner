"""
.. codeauthor:: David Zwicker <david.zwicker@ds.mpg.de>
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Optional, Union

from ..model import ModelBase
from ..results import Result

if TYPE_CHECKING:
    from zarr.storage import BaseStore  # @UnusedImport


Store = Union[str, Path, "BaseStore"]


def guess_format(path: Path) -> str:
    """guess the format of a given store

    Args:
        path (str or :class:`~pathlib.Path`):
            Path pointing to a file

    Returns:
        str: The store format
    """
    # guess format from path extension
    ext = Path(path).suffix.lower()
    if ext == ".json":
        return "json"
    elif ext in {".yml", ".yaml"}:
        return "yaml"
    elif ext in {".h5", ".hdf", ".hdf5"}:
        return "hdf"
    else:
        return "zarr"  # fallback to the default storage method based on zarr


def normalize_zarr_store(store: Store, mode: str = "a") -> Store:
    """determine best file format for zarr storage

    In particular, we use a :class:`~zarr.storage.ZipStore` when a path looking like a
    file is given.

    Args:
        store: User-provided store
        mode (str): The mode with which the file will be opened

    Returns:
    """
    import zarr

    if isinstance(store, (str, Path)):
        store = Path(store)
        if store.suffix != "":
            store = zarr.storage.ZipStore(store, mode=mode)
    return store


def result_check_load_old_version(
    path: Path,
    loc: str = "result",
    *,
    model: Optional[ModelBase] = None,
) -> Optional[Result]:
    # check for compatibility
    fmt = guess_format(path)
    if fmt == "json":
        import json

        with open(path, mode="r") as fp:
            format_version = json.load(fp).get("__version__", None)

    elif fmt == "yaml":
        import yaml

        with open(path, mode="r") as fp:
            format_version = yaml.safe_load(fp).get("__version__", None)

    elif fmt == "hdf":
        import h5py

        with h5py.File(path, mode="r") as root:
            format_version = root.attrs.get("__version__", None)

    elif fmt == "zarr":
        import zarr

        store = normalize_zarr_store(path, mode="r")
        with zarr.open_group(store, mode="r") as root:
            format_version = root.attrs.get("__version__", None)

    else:
        return None

    if format_version in {0, None}:
        # load result written with format version 0
        from modelrunner.compatibility.version0 import result_from_file_v0

        return result_from_file_v0(path, label=loc, model=model)

    elif format_version == 1:
        # load result written with format version 1
        from modelrunner.compatibility.version1 import result_from_file_v1

        return result_from_file_v1(path, label=loc, model=model)

    return None
