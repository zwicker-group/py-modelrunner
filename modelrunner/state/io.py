"""
Module defining a base class that handles writing and reading of objects

The base class itself implements little logic, but provides general methods that are
used by concrete classes. However, it defines the interface methods `from_file` and
`to_file`, which allow reading and writing data, respectively.

.. codeauthor:: David Zwicker <david.zwicker@ds.mpg.de>
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional, Union

import numpy as np
import zarr
from zarr.storage import BaseStore

from ..parameters import NoValueType


class NumpyEncoder(json.JSONEncoder):
    """helper class for encoding python data in JSON"""

    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, np.generic):
            return obj.item()
        if isinstance(obj, NoValueType):
            return None
        return json.JSONEncoder.default(self, obj)


def simplify_data(data):
    """simplify data (e.g. for writing to json or yaml)

    This function for instance turns sets and numpy arrays into lists.
    """
    if isinstance(data, dict):
        data = {key: simplify_data(value) for key, value in data.items()}

    elif isinstance(data, (tuple, list)):
        data = [simplify_data(item) for item in data]

    elif isinstance(data, (set, frozenset)):
        data = sorted([simplify_data(item) for item in data])

    elif isinstance(data, np.ndarray):
        if np.isscalar(data):
            data = data.item()
        elif data.size <= 100:
            # for less than ~100 items a list is actually more efficient to store
            data = data.tolist()

    elif isinstance(data, np.number):
        data = data.item()

    return data


def contains_array(data) -> bool:
    """checks whether data contains a numpy array"""
    if isinstance(data, np.ndarray):
        return True
    elif isinstance(data, dict):
        return any(contains_array(d) for d in data.values())
    elif isinstance(data, str):
        return False
    elif hasattr(data, "__iter__"):
        return any(contains_array(d) for d in data)
    else:
        return False


def write_hdf_dataset(node, data, name: str) -> None:
    """writes data to an HDF node

    Args:
        node: the HDF node
        data: the data to be written
        name (str): name of the data in case a new dataset or group is created
    """
    if data is None:
        return

    if isinstance(data, np.ndarray):
        node.create_dataset(name, data=data)

    else:
        if not contains_array(data):
            # write everything as JSON encoded string
            if isinstance(data, dict):
                group = node.create_group(name)
                for key, value in data.items():
                    group.attrs[key] = json.dumps(value, cls=NumpyEncoder)
            else:
                node.attrs[name] = json.dumps(data, cls=NumpyEncoder)

        elif isinstance(data, dict):
            group = node.create_group(name)
            for key, value in data.items():
                write_hdf_dataset(group, value, key)

        else:
            group = node.create_group(name)
            for n, value in enumerate(data):
                write_hdf_dataset(group, value, str(n))


def read_hdf_data(node):
    """read structured data written with :func:`write_hdf_dataset` from an HDF node"""
    import h5py

    if isinstance(node, h5py.Dataset):
        return np.array(node)
    else:
        # this must be a group
        data = {key: json.loads(value) for key, value in node.attrs.items()}
        for key, value in node.items():
            data[key] = read_hdf_data(value)
        return data


Store = Union[str, Path, BaseStore]


def normalize_zarr_store(store: Store, mode: str = "a") -> Store:
    """determine best file format for zarr storage

    In particular, we use a :class:`~zarr.storage.ZipStore` when a path looking like a
    file is given.

    Args:
        store: User-provided store
        mode (str): The mode with which the file will be opened

    Returns:
    """
    if isinstance(store, (str, Path)):
        store = Path(store)
        if store.suffix != "":
            store = zarr.storage.ZipStore(store, mode=mode)
    return store


zarrElement = Union[zarr.Group, zarr.Array]


class IOBase:
    """base class for handling structured input and output

    Subclasses need to define :meth:`_from_simple_objects` and
    :meth:`_to_simple_objects` methods to support the interface that store data in text
    files. However, the default method for storing data is using the :mod:`zarr`
    package, where the details are defined in the method :meth:`_write_zarr`, which
    needs to be implemented by subclasses.
    """

    @classmethod
    def _from_simple_objects(cls, content) -> IOBase:
        """create object from content given as simple python objects"""
        raise NotImplementedError(f"{cls.__name__}: no text reading")

    def _to_simple_objects(self):
        """convert object to representation using simple python objects"""
        raise NotImplementedError(f"{self.__class__.__name__}: no text writing")

    @classmethod
    def _from_hdf(cls, hdf_element) -> IOBase:
        """create object from a node in an HDF file"""
        raise NotImplementedError(f"{cls.__name__}: no HDF reading")

    def _write_hdf(self, hdf_element):
        """write object to the node of an HDF file"""
        raise NotImplementedError(f"{self.__class__.__name__}: no HDF writing")

    @classmethod
    def _from_zarr(cls, zarr_element: zarrElement) -> IOBase:
        """create object from a node in an zarr file"""
        raise NotImplementedError(f"{cls.__name__}: no zarr reading")

    def _write_zarr(
        self, zarr_group: zarr.Group, *, label: str = "data", **kwargs
    ) -> zarrElement:
        """write object to the node of an zarr file

        The implementation needs to add and return an element with the name given by
        the arugment `label` to the `zarr_group`, which contains all the data. This zarr
        element can either be an array or a group to store additional data and it needs
        to be returned by the method. Attributes, which allow identifying the written
        element, need to be written as attributes into the element, so that it is
        possible to completely restore the object by calling
        :code:`cls._from_zarr(zarr_group[label])`.
        """
        raise NotImplementedError(f"{self.__class__.__name__}: no zarr writing")

    @staticmethod
    def _guess_format(store: Store, fmt: Optional[str] = None) -> str:
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
        if isinstance(fmt, str):
            return fmt

        # guess format from path extension
        if isinstance(store, (str, Path)):
            ext = Path(store).suffix.lower()
            if ext == ".json":
                return "json"
            elif ext in {".yml", ".yaml"}:
                return "yaml"
            elif ext in {".h5", ".hdf", ".hdf5"}:
                return "hdf"
            else:
                return "zarr"  # fallback to the default storage method based on zarr

        elif isinstance(store, BaseStore):
            return "zarr"

        else:
            raise TypeError(f"Unsupported store type {store.__class__.__name__}")

    @classmethod
    def from_file(
        cls, store: Store, *, fmt: Optional[str] = None, label: str = "data", **kwargs
    ):
        """load object from a file

        Args:
            store (str or :class:`zarr.Store`):
                Path or instance describing the storage, which is either a file path or
                a :class:`zarr.Storage`.
            fmt (str):
                Explicit file format. Determined from `store` if omitted.
            label (str):
                Name of the node in which the data was stored. This applies to some
                hierarchical storage formats.
        """
        fmt = cls._guess_format(store, fmt)
        if fmt == "json":
            with open(store, mode="r") as fp:
                content = json.load(fp)
            return cls._from_simple_objects(content, **kwargs)

        elif fmt == "yaml":
            import yaml

            with open(store, mode="r") as fp:
                content = yaml.safe_load(fp)
            return cls._from_simple_objects(content, **kwargs)

        elif fmt == "hdf":
            import h5py

            with h5py.File(store, mode="r") as root:
                return cls._from_hdf(root, **kwargs)

        elif fmt == "zarr":
            store = normalize_zarr_store(store, mode="r")
            root = zarr.open_group(store, mode="r")
            return cls._from_zarr(root[label], **kwargs)

        elif hasattr(cls, f"_from_{fmt}"):
            with open(store, mode="r") as fp:
                return getattr(cls, f"_from_{fmt}")(fp)

        else:
            raise NotImplementedError(f"Format `{fmt}` not implemented")

    def to_file(
        self,
        store: Store,
        *,
        fmt: Optional[str] = None,
        overwrite: bool = False,
        **kwargs,
    ) -> None:
        """write this object to a file

        Args:
            store (str or :class:`zarr.Store`):
                Where to write the data to
            fmt (str):
                File format (guessed from extension of `store` if None)
            overwrite (bool):
                If True, overwrites files even if they already exist
            **kwargs:
                Additional arguments are passed on to the method that implements the
                writing of the specific format (_write_**).
        """
        fmt = self._guess_format(store, fmt)
        mode = "w" if overwrite else "x"  # zarr.ZipStore does only supports r, w, a, x

        if fmt == "json":
            content = simplify_data(self._to_simple_objects())
            kwargs.setdefault("cls", NumpyEncoder)
            with open(store, mode=mode) as fp:
                json.dump(content, fp, **kwargs)

        elif fmt == "yaml":
            import yaml

            content = simplify_data(self._to_simple_objects())
            kwargs.setdefault("sort_keys", False)
            with open(store, mode=mode) as fp:
                yaml.dump(content, fp, **kwargs)

        elif fmt == "hdf":
            import h5py

            with h5py.File(store, mode=mode) as root:
                self._write_hdf(root, **kwargs)

        elif fmt == "zarr":
            store = normalize_zarr_store(store, mode=mode)
            with zarr.group(store=store, overwrite=overwrite) as root:
                self._write_zarr(root, **kwargs)

        elif hasattr(self, f"_write_{fmt}"):
            with open(store, mode=mode) as fp:
                getattr(self, f"_write_{fmt}")(fp)

        else:
            raise NotImplementedError(f"Format `{fmt}` not implemented")
