"""Contains code necessary for loading results from format version 1.

.. codeauthor:: David Zwicker <david.zwicker@ds.mpg.de>
"""

from __future__ import annotations

import json
import warnings
from abc import ABCMeta
from pathlib import Path
from typing import TYPE_CHECKING, Any, TypeVar, Union

import numpy as np

from ...model import ModelBase
from ..results import Result
from .triage import guess_format, normalize_zarr_store

if TYPE_CHECKING:
    import zarr

    zarrElement = Union[zarr.Group, zarr.Array]


class NoData:
    """Helper class that marks data omission."""


TState = TypeVar("TState", bound="StateBase")


class StateBase:
    """Base class for specifying the state of a simulation.

    A state contains values of all degrees of freedom of a physical system (called the
    `data`) and some additional information (called `attributes`). The `data` is mutable
    and often a numpy array or a collection of numpy arrays. Conversely, the
    `attributes` are stroed in a dictionary with immutable values. To allow flexible
    storage, we define the properties `_state_data` and `_state_attributes`, which by
    default return `attributes` and `data` directly, but may be overwritten to process
    the data before storage (e.g., by additional serialization).
    """

    _state_format_version = 2
    """int: number indicating the version of the file format"""

    _state_classes: dict[str, type[StateBase]] = {}
    """dict: class-level list of all subclasses of StateBase"""

    def _state_init(self, attributes: dict[str, Any], data=NoData) -> None:
        """Initialize the state with attributes and (optionally) data.

        Args:
            attributes (dict): Additional (unserialized) attributes
            data: The data of the degerees of freedom of the physical system
        """
        if data is not NoData:
            self._state_data = data
        if attributes:
            self._state_attributes = attributes

    @classmethod
    def from_data(cls: type[TState], attributes: dict[str, Any], data=NoData) -> TState:
        """Create instance of any state class from attributes and data.

        Args:
            attributes (dict): Additional (unserialized) attributes
            data: The data of the degerees of freedom of the physical system

        Returns:
            The object containing the given attributes and data
        """
        # copy attributes since they are modified in this function
        attributes = attributes.copy()
        cls_name = attributes["__class__"]
        if cls_name != cls.__name__:
            raise RuntimeError(f"Expected `{cls.__name__}` but data had `{cls_name}`")

        format_version = attributes.pop("__version__", 0)
        if format_version != 1:
            warnings.warn(f"File format version mismatch ({format_version} != 1)")

        # create a new object without calling __init__, which might be overwriten by
        # the subclass and not follow our interface
        obj = cls.__new__(cls)
        obj._state_init(attributes, data)
        return obj

    @classmethod
    def _from_simple_objects(cls, content: dict[str, Any]) -> StateBase:
        """Create state from text data.

        Args:
            content: The loaded data
        """
        cls_name = content["attributes"]["__class__"]
        if cls_name == "ArrayState":
            return ArrayState.from_data(content["attributes"], content["data"])

        elif cls_name == "ArrayCollectionState":
            col_data = tuple(
                np.array(content["data"][label])
                for label in content["attributes"]["labels"]
            )
            return ArrayCollectionState.from_data(content["attributes"], col_data)

        elif cls_name == "DictState":
            dict_data: dict[str, StateBase] = {}
            for label, substate in content["data"].items():
                dict_data[label] = cls._from_simple_objects(substate)

            return DictState.from_data(content["attributes"], dict_data)

        elif cls_name == "ObjectState":
            return ObjectState.from_data(content["attributes"], content["data"])

        else:
            raise TypeError(f"Do not know how to load `{cls_name}`")

    @classmethod
    def _from_zarr(cls, zarr_element: zarrElement, *, index=...) -> StateBase:
        """Create instance of correct subclass from data stored in zarr."""
        # determine the class that knows how to read this data
        cls_name = zarr_element.attrs["__class__"]
        attributes = zarr_element.attrs.asdict()

        # create object form zarr data
        if cls_name == "ArrayState":
            data = zarr_element[index]
            return ArrayState.from_data(attributes, data)

        elif cls_name == "ArrayCollectionState":
            data = data = tuple(
                zarr_element[label][index] for label in zarr_element.attrs["labels"]
            )
            return ArrayCollectionState.from_data(attributes, data)

        elif cls_name == "DictState":
            data = {
                label: cls._from_zarr(zarr_element[label], index=index)
                for label in zarr_element.attrs["__keys__"]
            }
            return DictState.from_data(attributes, data)

        elif cls_name == "ObjectState":
            if zarr_element.shape == () and index is ...:
                data = zarr_element[index].item()
            else:
                data = zarr_element[index]
            return ObjectState.from_data(attributes, data)

        else:
            raise TypeError(f"Do not know how to load `{cls_name}`")


class ArrayState(StateBase): ...


class ArrayCollectionState(StateBase): ...


class DictState(StateBase): ...


class ObjectState(StateBase): ...


def _Result_from_simple_objects(
    content: dict[str, Any], model: ModelBase | None = None
) -> Result:
    """Read result from simple object (like loaded from a JSON file) using version 1.

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
        result=StateBase._from_simple_objects(content["state"]),
        info=content.get("info"),
    )


def _Result_from_zarr(
    zarr_element: zarrElement, *, index=..., model: ModelBase | None = None
) -> Result:
    """Create result from data stored in zarr."""
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

    return Result.from_data(model_data=model_data, result=state, model=model, info=info)


def result_from_file_v1(store: Path, *, label: str = "data", **kwargs) -> Result:
    """Load object from a file using format version 1.

    Args:
        store (Path):
            Path of file to read
        fmt (str):
            Explicit file format. Determined from `store` if omitted.
        label (str):
            Name of the node in which the data was stored. This applies to some
            hierarchical storage formats.
    """
    fmt = guess_format(store)
    if fmt == "json":
        with store.open() as fp:
            content = json.load(fp)
        return _Result_from_simple_objects(content, **kwargs)

    elif fmt == "yaml":
        import yaml

        with store.open() as fp:
            content = yaml.safe_load(fp)
        return _Result_from_simple_objects(content, **kwargs)

    elif fmt == "zarr":
        import zarr

        zarr_store = normalize_zarr_store(store, mode="r")
        root = zarr.open_group(zarr_store, mode="r")
        return _Result_from_zarr(root[label], **kwargs)

    else:
        raise NotImplementedError(f"Format `{fmt}` not implemented")
