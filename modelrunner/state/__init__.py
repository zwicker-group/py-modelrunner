"""
Classes that describe the state of a simulation at a single point in time

Each state is defined by :attr:`attributes` and :attr:`data`. Attributes describe
general aspects about a state, which typically do not change, e.g., its `name`.
These classes define how data is read and written and they contain methods that can be
used to write multiple states of the same class to a file consecutively, e.g., to store
a trajectory. Here, it is assumed that the `attributes` do not change over time.

TODO:
- document the succession of calls for storing fields (to get a better idea of the
  available hooks)
- do the same for loading data

.. autosummary::
   :nosignatures:

   ~object.ObjectState
   ~array.ArrayState
   ~array_collection.ArrayCollectionState
   ~dict.DictState

.. codeauthor:: David Zwicker <david.zwicker@ds.mpg.de>
"""

from typing import Any, Optional

import numpy as np

from .array import ArrayState
from .array_collection import ArrayCollectionState
from .base import NoData, StateBase
from .dict import DictState
from .io import Store, simplify_data
from .object import ObjectState


def make_state(data: Any) -> StateBase:
    """turn any data into a :class:`StateBase`"""
    if isinstance(data, StateBase):
        return data
    elif isinstance(data, np.ndarray):
        return ArrayState(data)
    return ObjectState(data)


def load_state(
    store: Store, *, fmt: Optional[str] = None, label: str = "data", **kwargs
) -> StateBase:
    """load state from a file

    Args:
        store (str or :class:`zarr.Store`):
            Path or instance describing the storage, which is either a file path or
            a :class:`zarr.Storage`.
        fmt (str):
            Explicit file format. Determined from `store` if omitted.
        label (str):
            Name of the node in which the data was stored. This applies to some
            hierarchical storage formats.

    Returns:
        :class:`StateBase`: The state loaded from a file
    """
    return StateBase.from_file(store, fmt=fmt, label=label)  # type: ignore
