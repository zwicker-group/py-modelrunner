"""
Contains code necessary for loading results from format version 1

.. codeauthor:: David Zwicker <david.zwicker@ds.mpg.de>
"""

from __future__ import annotations

import json
import warnings
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, Optional, Type, Union

import numpy as np

from ..model import ModelBase
from ..results import Result
from ..state import ArrayCollectionState, ArrayState, DictState, ObjectState, StateBase
from .triage import guess_format, normalize_zarr_store
from .version0 import read_hdf_data

if TYPE_CHECKING:
    import zarr

    zarrElement = Union[zarr.Group, zarr.Array]


class NoData:
    """helper class that marks data omission"""

    ...


def _StateBase_from_data(
    cls: Type[StateBase], attributes: Dict[str, Any], data=NoData
) -> StateBase:
    """create state from data and attributes"""
    # copy attributes since they are modified in this function
    attributes = attributes.copy()
    cls_name = attributes["__class__"]
    assert cls_name == cls.__name__

    format_version = attributes.pop("__version__", 0)
    if format_version != 1:
        warnings.warn(f"File format version mismatch ({format_version} != 1)")

    # create a new object without calling __init__, which might be overwriten by
    # the subclass and not follow our interface
    obj = cls.__new__(cls)
    obj._state_init(attributes, data)
    return obj


def _StateBase_from_simple_objects(content: Dict[str, Any]) -> StateBase:
    """create state from text data

    Args:
        content: The loaded data
    """
    cls_name = content["attributes"]["__class__"]
    if cls_name == "ArrayState":
        return _StateBase_from_data(ArrayState, content["attributes"], content["data"])

    elif cls_name == "ArrayCollectionState":
        col_data = tuple(
            np.array(content["data"][label])
            for label in content["attributes"]["labels"]
        )
        return _StateBase_from_data(
            ArrayCollectionState, content["attributes"], col_data
        )

    elif cls_name == "DictState":
        dict_data: Dict[str, StateBase] = {}
        for label, substate in content["data"].items():
            dict_data[label] = _StateBase_from_simple_objects(substate)

        return _StateBase_from_data(DictState, content["attributes"], dict_data)

    elif cls_name == "ObjectState":
        return _StateBase_from_data(ObjectState, content["attributes"], content["data"])

    else:
        raise TypeError(f"Do not know how to load `{cls_name}`")


def _StateBase_from_zarr(zarr_element: "zarrElement", *, index=...) -> StateBase:
    """create instance of correct subclass from data stored in zarr"""
    # determine the class that knows how to read this data
    cls_name = zarr_element.attrs["__class__"]
    attributes = zarr_element.attrs.asdict()

    # create object form zarr data
    if cls_name == "ArrayState":
        data = zarr_element[index]
        return _StateBase_from_data(ArrayState, attributes, data)

    elif cls_name == "ArrayCollectionState":
        data = data = tuple(
            zarr_element[label][index] for label in zarr_element.attrs["labels"]
        )
        return _StateBase_from_data(ArrayCollectionState, attributes, data)

    elif cls_name == "DictState":
        data = {
            label: _StateBase_from_zarr(zarr_element[label], index=index)
            for label in zarr_element.attrs["__keys__"]
        }
        return _StateBase_from_data(DictState, attributes, data)

    elif cls_name == "ObjectState":
        if zarr_element.shape == () and index is ...:
            data = zarr_element[index].item()
        else:
            data = zarr_element[index]
        return _StateBase_from_data(ObjectState, attributes, data)

    else:
        raise TypeError(f"Do not know how to load `{cls_name}`")


def _Result_from_simple_objects(
    content: Dict[str, Any], model: Optional[ModelBase] = None
) -> Result:
    """read result from simple object (like loaded from a JSON file) using version 1

    Args:
        content (dict):
            Data from which the result is restored
        model (:class:`ModelBase`):
            Model associated with the result
    """
    format_version = content.pop("__version__", None)
    if format_version != 1:
        raise RuntimeError(f"Cannot read format version {format_version}")

    return Result.from_data(
        model_data=content.get("model", {}),
        result=_StateBase_from_simple_objects(content["state"]),
        info=content.get("info"),
    )


def _Result_from_hdf(hdf_element, model: Optional[ModelBase] = None) -> Result:
    """read result from a HDf file

    Args:
        hdf_element: The path to the file
        model (:class:`ModelBase`): The model from which the result was obtained
    """
    attributes = {key: json.loads(value) for key, value in hdf_element.attrs.items()}
    # extract version information from attributes
    format_version = attributes.pop("__version__", None)
    if format_version != Result._format_version:
        raise RuntimeError(f"Cannot read format version {format_version}")
    info = attributes.pop("__info__", {})  # load additional info

    # the remaining attributes correspond to the model
    model_data = attributes

    # load state
    state_attributes = read_hdf_data(hdf_element["state"])
    state_data = read_hdf_data(hdf_element["data"])
    state = StateBase.from_data(state_attributes, state_data)

    return Result.from_data(model_data=model_data, result=state, model=model, info=info)


def _Result_from_zarr(
    zarr_element: "zarrElement", *, index=..., model: Optional[ModelBase] = None
) -> Result:
    """create result from data stored in zarr"""
    attributes = {key: json.loads(value) for key, value in zarr_element.attrs.items()}
    # extract version information from attributes
    format_version = attributes.pop("__version__", None)
    if format_version != 1:
        raise RuntimeError(f"Cannot read format version {format_version}")
    info = attributes.pop("__info__", {})  # load additional info

    # the remaining attributes correspond to the model
    model_data = attributes

    # load state
    state = _StateBase_from_zarr(zarr_element["state"])

    return Result.from_data(model_data=model_data, result=state, model=model, info=info)


def result_from_file_v1(store: Path, *, label: str = "data", **kwargs) -> Result:
    """load object from a file using format version 1

    Args:
        store (Path):
            Path or instance describing the storage, which is either a file path or
            a :class:`zarr.Storage`.
        fmt (str):
            Explicit file format. Determined from `store` if omitted.
        label (str):
            Name of the node in which the data was stored. This applies to some
            hierarchical storage formats.
    """
    fmt = guess_format(store)
    if fmt == "json":
        with open(store, mode="r") as fp:
            content = json.load(fp)
        return _Result_from_simple_objects(content, **kwargs)

    elif fmt == "yaml":
        import yaml

        with open(store, mode="r") as fp:
            content = yaml.safe_load(fp)
        return _Result_from_simple_objects(content, **kwargs)

    elif fmt == "hdf":
        import h5py

        with h5py.File(store, mode="r") as root:
            return _Result_from_hdf(root, **kwargs)

    elif fmt == "zarr":
        import zarr  # @Reimport

        zarr_store = normalize_zarr_store(store, mode="r")
        root = zarr.open_group(zarr_store, mode="r")
        return _Result_from_zarr(root[label], **kwargs)

    else:
        raise NotImplementedError(f"Format `{fmt}` not implemented")


__all__ = ["result_from_file"]
