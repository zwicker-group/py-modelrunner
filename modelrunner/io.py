"""
<<<<<<< HEAD
Module defining a base class that handles writing and reading of objects (mostly to the 
hard disk). The base class itself implements little logic, but provides general methods
that are used by concrete classes.

.. codeauthor:: David Zwicker <david.zwicker@ds.mpg.de>
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional, Union

import numpy as np
import zarr
from zarr.storage import BaseStore

from .parameters import NoValueType


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
    """simplify data (e.g. for writing to yaml)"""
    if isinstance(data, dict):
        data = {key: simplify_data(value) for key, value in data.items()}

    elif isinstance(data, (tuple, list)):
        data = [simplify_data(item) for item in data]

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
zarrElement = Union[zarr.Group, zarr.Array]


class IOBase:
    """base class for handling structured input and output

    Subclasses need to define :meth:`_from_simple_objects` and :meth:`_to_simple_objects` methods
    to support the interface that store data in text files. However, the default method
    for storing data is using the :mod:`zarr` package, where the details are defined in
    the method :meth:`_write_zarr`, which needs to be implemented by subclasses.
    """

    @classmethod
    def _from_simple_objects(cls, content) -> IOBase:
        raise NotImplementedError(f"{cls.__name__}: no text reading")

    def _to_simple_objects(self):
        raise NotImplementedError(f"{self.__class__.__name__}: no text writing")

    @classmethod
    def _from_hdf(cls, hdf_element) -> IOBase:
        raise NotImplementedError(f"{cls.__name__}: no HDF reading")

    def _write_hdf(self, hdf_element):
        raise NotImplementedError(f"{self.__class__.__name__}: no HDF writing")

    @classmethod
    def _from_zarr(cls, zarr_element: zarrElement) -> IOBase:
        raise NotImplementedError(f"{cls.__name__}: no zarr reading")

    def _write_zarr(self, zarr_group: zarr.Group, **kwargs):
        raise NotImplementedError(f"{self.__class__.__name__}: no zarr writing")

    @staticmethod
    def _guess_format(store: Store, fmt: Optional[str] = None):
        """guess the format"""
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

        # fallback to the default storage method based on zarr
        return "zarr"

    @classmethod
    def from_file(cls, store: Store, *, fmt: Optional[str] = None, **kwargs):
        """load state from a file

        Args:
            store (str or :class:`zarr.Store`):
                Where to read the data from
            fmt (str):
                File format (guessed from extension of filename if None)
        """
        fmt = cls._guess_format(store, fmt)
        if fmt == "json":
            with open(store, "r") as fp:
                content = json.load(fp)
            return cls._from_simple_objects(content, **kwargs)

        elif fmt == "yaml":
            import yaml

            with open(store, "r") as fp:
                content = yaml.safe_load(fp)
            return cls._from_simple_objects(content, **kwargs)

        elif fmt == "hdf":
            import h5py

            with h5py.File(store, "r") as root:
                return cls._from_hdf(root, **kwargs)

        elif fmt == "zarr":
            # fallback to the default storage method based on zarr
            root = zarr.open_group(store, mode="r")
            return cls._from_zarr(root["data"], **kwargs)

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
        """write this state to a file

        Args:
            store (str or :class:`zarr.Store`):
                Where to write the data to
            fmt (str):
                File format (guessed from extension of filename if None)
            overwrite (bool):
                If True, overwrites files even if they already exist
        """
        fmt = self._guess_format(store, fmt)
        if fmt == "json":
            content = simplify_data(self._to_simple_objects())
            kwargs.setdefault("cls", NumpyEncoder)
            with open(store, "w" if overwrite else "x") as fp:
                json.dump(content, fp, **kwargs)

        elif fmt == "yaml":
            import yaml

            content = simplify_data(self._to_simple_objects())
=======
.. codeauthor:: David Zwicker <david.zwicker@ds.mpg.de>
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Union

import numpy as np
import zarr
from zarr.storage import BaseStore

from .parameters import NoValueType


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


def prepare_yaml(data):
    """prepare data for writing to yaml"""
    if isinstance(data, dict):
        data = data.copy()  # shallow copy of result
        for key, value in data.items():
            if isinstance(value, dict):
                data[key] = prepare_yaml(value)
            elif isinstance(value, tuple):
                data[key] = list(value)
            elif isinstance(value, list) and value and not np.isscalar(value[0]):
                data[key] = [prepare_yaml(a) for a in value]
            elif isinstance(value, np.ndarray) and value.size <= 100:
                # for less than ~100 items a list is actually more efficient to store
                data[key] = value.tolist()

    elif isinstance(data, np.ndarray) and data.size <= 100:
        # for less than ~100 items a list is actually more efficient to store
        data = data.tolist()

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
zarrElement = Union[zarr.Group, zarr.Array]


class IOBase:
    """base class for handling structured input and output

    Subclasses need to define :meth:`_from_simple_objects` and :meth:`_to_simple_objects` methods
    to support the interface that store data in text files. However, the default method
    for storing data is using the :mod:`zarr` package, where the details are defined in
    the method :meth:`_write_zarr`, which needs to be implemented by subclasses.
    """

    @classmethod
    def _from_simple_objects(cls, content) -> IOBase:
        raise NotImplementedError(f"{cls.__name__}: no text reading")

    def _to_simple_objects(self):
        raise NotImplementedError(f"{self.__class__.__name__}: no text writing")

    @classmethod
    def _from_hdf(cls, hdf_element) -> IOBase:
        raise NotImplementedError(f"{cls.__name__}: no HDF reading")

    def _write_hdf(self, hdf_element):
        raise NotImplementedError(f"{self.__class__.__name__}: no HDF writing")

    @classmethod
    def _from_zarr(cls, zarr_element: zarrElement) -> IOBase:
        raise NotImplementedError(f"{cls.__name__}: no zarr reading")

    def _write_zarr(self, zarr_group: zarr.Group, **kwargs):
        raise NotImplementedError(f"{self.__class__.__name__}: no zarr writing")

    @staticmethod
    def _guess_format(store: Store, fmt: str = None):
        """guess the format"""
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

        # fallback to the default storage method based on zarr
        return "zarr"

    @classmethod
    def from_file(cls, store: Store, *, fmt: str = None, **kwargs):
        """load state from a file

        Args:
            store (str or :class:`zarr.Store`):
                Where to read the data from
            fmt (str):
                File format (guessed from extension of filename if None)
        """
        fmt = cls._guess_format(store, fmt)
        if fmt == "json":
            with open(store, "r") as fp:
                content = json.load(fp)
            return cls._from_simple_objects(content, **kwargs)

        elif fmt == "yaml":
            import yaml

            with open(store, "r") as fp:
                content = yaml.safe_load(fp)
            return cls._from_simple_objects(content, **kwargs)

        elif fmt == "hdf":
            import h5py

            with h5py.File(store, "r") as root:
                return cls._from_hdf(root, **kwargs)

        elif fmt == "zarr":
            # fallback to the default storage method based on zarr
            root = zarr.open_group(store, mode="r")
            return cls._from_zarr(root["data"], **kwargs)

        else:
            raise NotImplementedError(f"Format `{fmt}` not implemented")

    def to_file(
        self, store: Store, *, fmt: str = None, overwrite: bool = False, **kwargs
    ) -> None:
        """write this state to a file

        Args:
            store (str or :class:`zarr.Store`):
                Where to write the data to
            fmt (str):
                File format (guessed from extension of filename if None)
            overwrite (bool):
                If True, overwrites files even if they already exist
        """
        fmt = self._guess_format(store, fmt)
        if fmt == "json":
            content = self._to_simple_objects()
            kwargs.setdefault("cls", NumpyEncoder)
            with open(store, "w" if overwrite else "x") as fp:
                json.dump(content, fp, **kwargs)

        elif fmt == "yaml":
            import yaml

            content = prepare_yaml(self._to_simple_objects())
>>>>>>> branch 'state' of https://github.com/zwicker-group/py-modelrunner.git
            kwargs.setdefault("sort_keys", False)
            with open(store, "w" if overwrite else "x") as fp:
                yaml.dump(content, fp, **kwargs)

        elif fmt == "hdf":
            import h5py

            with h5py.File(store, "w" if overwrite else "x") as root:
                self._write_hdf(root, **kwargs)

        elif fmt == "zarr":
            # fallback to the default storage method based on zarr
            with zarr.group(store=store, overwrite=overwrite) as group:
                self._write_zarr(group, **kwargs)

        else:
            raise NotImplementedError(f"Format `{fmt}` not implemented")
