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
    # define basic payload
    a = np.arange(5)
    b = np.random.random(size=3)
    da = np.ones((1,), dtype=[("x", int), ("y", float)])
    ra = np.recarray((1,), dtype=[("x", int), ("y", float)])
    ra[:] = (1, 2)
    o = {"list": [1, 2], "bool": True}

    # define basic state classes
    obj_state = ObjectState(o.copy())
    arr_state = ArrayState(a.copy())
    res = [
        obj_state.copy(),
        arr_state.copy(),
        ArrayState(da.copy()),
        ArrayState(ra.copy()),
        ArrayCollectionState((a.copy(), b.copy()), labels=["a", "b"]),
        DictState({"o": obj_state.copy(), "a": arr_state.copy()}),
    ]

    # add custom state classes
    if "DerivedObject" not in StateBase._state_classes:
        # make sure the derived object is only defined once

        class DerivedObject(ObjectState):
            _state_data_attr_name = "values"

            def __init__(self, values):
                self.values = values

    res.append(StateBase._state_classes["DerivedObject"](o.copy()))

    if "DerivedArray" not in StateBase._state_classes:
        # make sure the derived object is only defined once

        class DerivedArray(ArrayState):
            _state_data_attr_name = "array"

            def __init__(self, array):
                self.array = array

    res.append(StateBase._state_classes["DerivedArray"](a.copy()))

    if "DerivedDict" not in StateBase._state_classes:
        # make sure the derived object is only defined once

        class DerivedDict(DictState):
            _state_data_attr_name = "states"

            def __init__(self, states):
                self.states = {k: v.copy() for k, v in states.items()}

    res.append(
        StateBase._state_classes["DerivedDict"]({"o": obj_state, "a": arr_state})
    )

    return res
