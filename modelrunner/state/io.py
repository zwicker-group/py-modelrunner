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

from ..parameters import NoValueType
if TYPE_CHECKING:
    from ..storage import StorageID





class IOBase:
    """base class for handling structured input and output

    Subclasses need to define :meth:`_from_simple_objects` and
    :meth:`_to_simple_objects` methods to support the interface that store data in text
    files. However, the default method for storing data is using the :mod:`zarr`
    package, where the details are defined in the method :meth:`_write_zarr`, which
    needs to be implemented by subclasses.
    """

    # @classmethod
    # def _from_simple_objects(cls, content) -> IOBase:
    #     """create object from content given as simple python objects"""
    #     raise NotImplementedError(f"{cls.__name__}: no text reading")
    #
    # def _to_simple_objects(self):
    #     """convert object to representation using simple python objects"""
    #     raise NotImplementedError(f"{self.__class__.__name__}: no text writing")
    #
    # @classmethod
    # def _from_hdf(cls, hdf_element) -> IOBase:
    #     """create object from a node in an HDF file"""
    #     raise NotImplementedError(f"{cls.__name__}: no HDF reading")
    #
    # def _write_hdf(self, hdf_element):
    #     """write object to the node of an HDF file"""
    #     raise NotImplementedError(f"{self.__class__.__name__}: no HDF writing")
    #
    # @classmethod
    # def _from_zarr(cls, zarr_element: zarrElement) -> IOBase:
    #     """create object from a node in an zarr file"""
    #     raise NotImplementedError(f"{cls.__name__}: no zarr reading")
    #
    # def _write_zarr(
    #     self, zarr_group: zarr.Group, *, label: str = "data", **kwargs
    # ) -> zarrElement:
    #     """write object to the node of an zarr file
    #
    #     The implementation needs to add and return an element with the name given by
    #     the arugment `label` to the `zarr_group`, which contains all the data. This zarr
    #     element can either be an array or a group to store additional data and it needs
    #     to be returned by the method. Attributes, which allow identifying the written
    #     element, need to be written as attributes into the element, so that it is
    #     possible to completely restore the object by calling
    #     :code:`cls._from_zarr(zarr_group[label])`.
    #     """
    #     raise NotImplementedError(f"{self.__class__.__name__}: no zarr writing")

    @classmethod
    def from_file(
        cls, storage: StorageID, *, fmt: Optional[str] = None, label: str = "data", **kwargs
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

    def to_file(self, storage: StorageID, *, overwrite: bool = False, **kwargs) -> None:
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
        fmt = self._guess_format(StorageID, fmt)
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
