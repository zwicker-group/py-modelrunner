"""
Contains code necessary for loading results from previous version

.. codeauthor:: David Zwicker <david.zwicker@ds.mpg.de>
"""

from __future__ import annotations

import json
import warnings
from typing import Any, Dict, Mapping, Optional, Type

import numpy as np

from ..model import ModelBase
from ..results import Result
from ..state import ArrayCollectionState, ArrayState, DictState, ObjectState, StateBase
from .triage import Store, guess_format, normalize_zarr_store


class NoData:
    """helper class that marks data omission"""

    ...


def _StateBase_from_data(cls: Type[StateBase], attributes: Dict[str, Any], data=NoData):
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
        data = tuple(np.asarray(subdata) for subdata in content["data"])
        return _StateBase_from_data(ArrayCollectionState, content["attributes"], data)

    elif cls_name == "DictState":
        attributes, data = content["attributes"], content["data"]
        if not isinstance(data, dict) and "__keys__" in attributes:
            data = {k: v for k, v in zip(attributes.pop("__keys__"), data)}
        return _StateBase_from_data(DictState, attributes, data)

    elif cls_name == "ObjectState":
        return _StateBase_from_data(ObjectState, content["attributes"], content["data"])

    else:
        raise TypeError(f"Do not know how to load `{cls_name}`")


def _Result_from_simple_objects(
    content: Mapping[str, Any], model: Optional[ModelBase] = None
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
        state=_StateBase_from_simple_objects(content["state"]),
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
    if format_version != Result._state_format_version:
        raise RuntimeError(f"Cannot read format version {format_version}")
    info = attributes.pop("__info__", {})  # load additional info

    # the remaining attributes correspond to the model
    model_data = attributes

    # load state
    state_attributes = read_hdf_data(hdf_element["state"])
    state_data = read_hdf_data(hdf_element["data"])
    state = StateBase.from_data(state_attributes, state_data)

    return Result.from_data(model_data=model_data, state=state, model=model, info=info)


@classmethod
def _Result_from_zarr(
    zarr_element: zarrElement, *, index=..., model: Optional[ModelBase] = None
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
    state = StateBase._from_zarr(zarr_element["state"])

    return Result.from_data(model_data=model_data, state=state, model=model, info=info)


def result_from_file_v1(store: Store, *, label: str = "data", **kwargs):
    """load object from a file using format version 1

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
        import zarr

        store = normalize_zarr_store(store, mode="r")
        root = zarr.open_group(store, mode="r")
        return _Result_from_zarr(root[label], **kwargs)

    else:
        raise NotImplementedError(f"Format `{fmt}` not implemented")


__all__ = ["result_from_file"]
