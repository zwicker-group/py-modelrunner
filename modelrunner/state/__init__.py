from typing import Any, Optional

import numpy as np

from .array import ArrayState
from .array_collection import ArrayCollectionState
from .base import StateBase
from .dict import DictState
from .io import Store
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
