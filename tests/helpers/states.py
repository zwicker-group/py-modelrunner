"""
.. codeauthor:: David Zwicker <david.zwicker@ds.mpg.de>
"""

from typing import Any

import numpy as np

from modelrunner.state import (
    ArrayCollectionState,
    ArrayState,
    DictState,
    ObjectState,
    StateBase,
)


def assert_data_equals(left: Any, right: Any) -> bool:
    """checks whether two objects are equal, also supporting :class:~numpy.ndarray`

    Args:
        left: one object
        right: other object

    Returns:
        bool: Whether the two objects are equal
    """
    if isinstance(left, set) or isinstance(right, set):
        # One of the operands is a set, while the other is ordered. This needs to be the
        # first check since otherwise there might be an ordered comparison, which can
        # fail undeterministically.
        assert set(left) == set(right)

    elif type(left) is type(right):
        # typical cases where both operands are of equal type
        if isinstance(left, StateBase):
            assert left._state_attributes == right._state_attributes
            assert_data_equals(left._state_data, right._state_data)

        elif isinstance(left, str):
            assert left == right

        elif isinstance(left, dict):
            assert left.keys() == right.keys()
            for key in left:
                assert_data_equals(left[key], right[key])

        elif hasattr(left, "__iter__"):
            assert len(left) == len(right)
            for l, r in zip(left, right):
                assert_data_equals(l, r)

        else:
            assert left == right

    elif isinstance(left, np.ndarray) or isinstance(right, np.ndarray):
        # one of the operands numpy array, while the other might be a list
        assert np.array_equal(np.asarray(left), np.asarray(right))

    elif isinstance(left, np.number) or isinstance(right, np.number):
        # one of the operands numpy number, while the other might be a normal number
        assert left == right

    else:
        # cause failure
        assert type(left) == type(right), f"{type(left)} != {type(right)}"


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
        obj_state.copy("clean"),
        ObjectState(a.copy()),
        arr_state.copy("clean"),
        ArrayState(da.copy()),
        ArrayState(ra.copy()),
        ArrayCollectionState((a.copy(), b.copy()), labels=["a", "b"]),
        DictState({"o": obj_state.copy("clean"), "a": arr_state.copy("clean")}),
    ]

    return res
