"""
.. codeauthor:: David Zwicker <david.zwicker@ds.mpg.de>
"""

from __future__ import annotations

import numpy as np


def simplify_data(data):
    """Simplify data (e.g. for writing to json or yaml)

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
        elif np.issubdtype(data.dtype, "O") and data.size == 1:
            data = [simplify_data(data.item())]
        elif data.size <= 100:
            # for less than ~100 items a list is actually more efficient to store
            data = data.tolist()

    elif isinstance(data, np.number):
        data = data.item()

    return data
