"""
.. codeauthor:: David Zwicker <david.zwicker@ds.mpg.de>
"""

from typing import Any, Sequence

import numpy as np


def homogeneous_shape(arr: Sequence) -> bool:
    """Test whether sequence items have all the same length."""
    try:
        return len({len(a) for a in arr}) == 1
    except TypeError:
        # happens if one of the items does not have __len__
        return True


def assert_data_equals(left: Any, right: Any, *, fuzzy: bool = False) -> bool:
    """Checks whether two objects are equal, also supporting :class:~numpy.ndarray`

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

    elif isinstance(left, right.__class__) or isinstance(right, left.__class__):
        # typical cases where both operands are of equal type
        if "State" in left.__class__.__name__:
            assert left._state_attributes == right._state_attributes
            assert_data_equals(left._state_data, right._state_data, fuzzy=fuzzy)

        elif isinstance(left, str):
            assert left == right

        elif isinstance(left, dict):
            assert left.keys() == right.keys()
            for key in left:
                assert_data_equals(left[key], right[key], fuzzy=fuzzy)

        elif hasattr(left, "__iter__"):
            assert len(left) == len(right)
            for l, r in zip(left, right):
                assert_data_equals(l, r, fuzzy=fuzzy)

        else:
            assert left == right

    elif fuzzy:
        if isinstance(left, np.ndarray) or isinstance(right, np.ndarray):
            # one of the operands numpy array, while the other might be a list
            if homogeneous_shape(left) and homogeneous_shape(right):
                assert np.array_equal(np.asarray(left), np.asarray(right))
            else:
                assert len(left) == len(right)
                for l, r in zip(left, right):
                    assert_data_equals(l, r, fuzzy=fuzzy)

        elif isinstance(left, np.number) or isinstance(right, np.number):
            # one of the operands numpy number, while the other might be a normal number
            assert left == right

        else:
            assert left == right

    else:
        # strict comparison of types required
        assert type(left) == type(right), f"{type(left)} != {type(right)}"
