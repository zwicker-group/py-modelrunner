"""
Classes that describe the state of a simulation at a single point in time

Each state is defined by :attr:`attributes` and :attr:`data`. Attributes describe
general aspects about a state, which typically do not change, e.g., its `name`.
These classes define how data is read and written and they contain methods that can be
used to write multiple states of the same class to a file consecutively, e.g., to store
a trajectory. Here, it is assumed that the `attributes` do not change over time.

All state classes can be sub-classed to adjust to specialized needs. This will often be
necessary if some attributes cannot be serialized automatically or if the data requires
some modifications before storing. To facilitate control over how data is written and
read, we provide the :attr:`~modelrunner.state.base.StateBase._state_attributes_store`
and :attr:`~modelrunner.state.base.StateBase._state_data_store` attributes which should
return respective attributes and data in a form that can be stored directly. When the
object will be restored during reading, the
:meth:`~modelrunner.state.base.StateBase._state_init` method is used to set the
properties of an object.

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
    return StateBase.from_file(store, fmt=fmt, label=label, **kwargs)  # type: ignore
