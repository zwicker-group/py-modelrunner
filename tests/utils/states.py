"""
.. codeauthor:: David Zwicker <david.zwicker@ds.mpg.de>
"""

import numpy as np

from modelrunner.state import (
    ArrayCollectionState,
    ArrayState,
    DictState,
    ObjectState,
    StateBase,
)

EXTENSIONS = ["json", "yaml", "zarr"]


def get_states():
    """generate multiple states"""
    a = np.arange(5)
    b = np.random.random(size=3)
    o = {"list": [1, 2], "bool": True}
    res = [
        ObjectState(o.copy()),
        ArrayState(a.copy()),
        ArrayCollectionState((a.copy(), b.copy()), labels=["a", "b"]),
        DictState({"o": ObjectState(o.copy()), "a": ArrayState(a.copy())}),
    ]

    if "DerivedObject" not in StateBase._state_classes:
        # make sure the derived object is only defined once

        class DerivedObject(ObjectState):
            _state_data_attribute = "values"

            def __init__(self, values):
                self.values = values

    res.append(StateBase._state_classes["DerivedObject"](o.copy()))

    if "DerivedArray" not in StateBase._state_classes:
        # make sure the derived object is only defined once

        class DerivedArray(ArrayState):
            _state_data_attribute = "array"

            def __init__(self, array):
                self.array = array

    res.append(StateBase._state_classes["DerivedArray"](a.copy()))

    return res
