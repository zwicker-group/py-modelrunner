"""Contains code necessary for deciding which format version was used to write a file.

.. codeauthor:: David Zwicker <david.zwicker@ds.mpg.de>
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import TYPE_CHECKING, Any, Mapping, Union

from ...model import ModelBase
from ..results import Result

if TYPE_CHECKING:
    from zarr.storage import BaseStore


Store = Union[str, Path, "BaseStore"]


def guess_format(path: Path) -> str:
    """Guess the format of a given store.

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


def normalize_zarr_store(store: Store, mode: str = "a") -> Store | None:
    """Determine best file format for zarr storage.

    In particular, we use a :class:`~zarr.storage.ZipStore` when a path looking like a
    file is given.

    Args:
        store: User-provided store
        mode (str): The mode with which the file will be opened

    Returns:
    """
    import zipfile

    import zarr

    if isinstance(store, (str, Path)):
        store = Path(store)
        if store.is_file():
            try:
                store = zarr.storage.ZipStore(store, mode=mode)
            except zipfile.BadZipfile:
                return None
        else:
            return None
    return store


def _find_version(data: Mapping[str, Any], label: str) -> int | None:
    """Try finding version information in different places in `data`

    Args:
        data (dict):
            A mapping that contains attribute information
        label (str):
            The label of the data that should be loaded

    Returns:
        int: The format version or None if it could not be found
    """

    def read_version(item) -> str | None:
        """Try reading attribute from a particular item."""
        if hasattr(item, "attrs"):
            return read_version(item.attrs)
        elif "__version__" in item:
            return item["__version__"]  # type: ignore
        elif "format_version" in item:
            return item["format_version"]  # type: ignore
        elif "__attrs__" in item:
            return read_version(item["__attrs__"])
        elif "attributes" in item:
            return read_version(item["attributes"])
        else:
            return None

    format_version = read_version(data)
    if format_version is None and label in data:
        format_version = read_version(data[label])
    if format_version is None and "state" in data:
        format_version = read_version(data["state"])
    if format_version is None and "result" in data:
        format_version = read_version(data["result"])

    if isinstance(format_version, str):
        return json.loads(format_version)  # type: ignore
    else:
        return format_version


def _get_format_version(path: Path, label: str) -> int | None:
    """Determine format version of the file in `path`

    Args:
        path (str or :class:`~pathlib.Path`):
            The path to the resource to be loaded
        label (str):
            Label of the item to be loaded
    """
    format_version = None
    # check for compatibility
    fmt = guess_format(path)
    if fmt == "json":
        with path.open() as fp:
            format_version = _find_version(json.load(fp), label)

    elif fmt == "yaml":
        import yaml

        with path.open() as fp:
            format_version = _find_version(yaml.safe_load(fp), label)

    elif fmt == "hdf":
        import h5py

        with h5py.File(path, mode="r") as root:
            format_version = _find_version(root, label)

    elif fmt == "zarr":
        import zarr

        store = normalize_zarr_store(path, mode="r")
        if store is None:
            raise RuntimeError
        with zarr.open_group(store, mode="r") as root:
            format_version = _find_version(root, label)
            if format_version is None and label != "data":
                format_version = _find_version(root, "data")
                if format_version is not None:
                    label = "data"

    else:
        raise RuntimeError
    return format_version


def result_check_load_old_version(
    path: Path, loc: str | None, *, model: ModelBase | None = None
) -> Result | None:
    """Check whether the resource can be loaded with an older version of the package.

    Args:
        path (str or :class:`~pathlib.Path`):
            The path to the resource to be loaded
        loc (str):
            Label, key, or location of the item to be loaded
        model (:class:`~modelrunner.model.ModelBase`, optional):
            Optional model that was used to write this result

    Returns:
        :class:`~modelrunner.result.Result`:
            The loaded result or `None` if we cannot load it with the old versions
    """
    label = "data" if loc is None else loc
    try:
        format_version = _get_format_version(path, label)
    except RuntimeError:
        return None  # could not determine format version

    if format_version in {0, None}:
        # load result written with format version 0
        from .version0 import result_from_file_v0

        logger = logging.getLogger("modelrunner.compatiblity")
        logger.info("Load data with format version 0")
        return result_from_file_v0(path, model=model)

    elif format_version == 1:
        # load result written with format version 1
        from .version1 import result_from_file_v1

        logger = logging.getLogger("modelrunner.compatiblity")
        logger.info("Load data with format version 1")
        return result_from_file_v1(path, label=label, model=model)

    elif format_version == 2:
        # load result written with format version 1
        from .version2 import result_from_file_v2

        logger = logging.getLogger("modelrunner.compatiblity")
        logger.info("Load data with format version 2")
        return result_from_file_v2(path, label=label, model=model)

    elif not isinstance(format_version, int):
        raise RuntimeError(f"Unsupported format version {format_version}")

    return None
