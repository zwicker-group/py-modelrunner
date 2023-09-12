"""
.. codeauthor:: David Zwicker <david.zwicker@ds.mpg.de>
"""

# Dtype needs to be properly pickled
# Encoding:
#     dtype_pickled = pickle.dumps(self._state_data.dtype)
#     codecs.encode(dtype_pickled, "base64").decode()
# Decoding:
#     dtype = pickle.loads(codecs.decode(dtype_pickled.encode(), "base64"))
#     if data.dtype.names is None:
#         data = unstructured_to_structured(data, dtype=dtype)

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional, Union

import numpy as np

from ..base import StorageBase
from ..parameters import NoValueType


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


class TextBasedStorage(StorageBase):
    ...


class JSONStorage(TextBasedStorage):
    ...


class YAMLStorage(TextBasedStorage):
    ...
