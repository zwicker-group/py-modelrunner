"""Contains code necessary for loading results from format version 0.

.. codeauthor:: David Zwicker <david.zwicker@ds.mpg.de>
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from ...model import ModelBase
from ..results import Result
from .triage import guess_format


def read_hdf_data(node):
    """Read structured data written with :func:`write_hdf_dataset` from an HDF node."""
    import h5py

    if isinstance(node, h5py.Dataset):
        return np.array(node)
    else:
        # this must be a group
        data = {key: json.loads(value) for key, value in node.attrs.items()}
        for key, value in node.items():
            data[key] = read_hdf_data(value)
        return data


def _Result_from_simple_objects(
    content: Mapping[str, Any], model: ModelBase | None = None
) -> Result:
    """Read result from simple object (like loaded from a JSON file) using version 0.

    Args:
        content (dict):
            Data from which the result is restored
        model (:class:`ModelBase`):
            Model associated with the result

    Returns:
        :class:`Result`: the restored result
    """

    return Result.from_data(
        model_data=content.get("model", {}),
        result=content.get("result"),
        model=model,
        info=content.get("info", {}),
    )


def _Result_from_hdf(hdf_element, model: ModelBase | None = None) -> Result:
    """Old reader for backward compatible reading."""
    model_data = {key: json.loads(value) for key, value in hdf_element.attrs.items()}
    if "result" in hdf_element:
        result = read_hdf_data(hdf_element["result"])
    else:
        result = model_data.pop("result")
    # check for other nodes, which might not be read

    info = model_data.pop("__info__") if "__info__" in model_data else {}

    return Result.from_data(
        model_data=model_data, result=result, model=model, info=info
    )


def result_from_file_v0(path: Path, **kwargs) -> Result:
    """Load object from a file using format version 1.

    Args:
        store (str or :class:`zarr.Store`):
            Path or instance describing the storage, which is either a file path or
            a :class:`zarr.Storage`.
    """
    fmt = guess_format(path)
    if fmt == "json":
        with path.open() as fp:
            content = json.load(fp)
        return _Result_from_simple_objects(content, **kwargs)

    elif fmt == "yaml":
        import yaml

        with path.open() as fp:
            content = yaml.safe_load(fp)
        return _Result_from_simple_objects(content, **kwargs)

    elif fmt == "hdf":
        import h5py

        with h5py.File(path, mode="r") as root:
            return _Result_from_hdf(root, **kwargs)

    else:
        raise NotImplementedError(f"Format `{fmt}` not implemented")
