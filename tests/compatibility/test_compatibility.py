"""
.. codeauthor:: David Zwicker <david.zwicker@ds.mpg.de>
"""

import pickle
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from modelrunner.results import Result, StateBase
from modelrunner.storage.backend.utils import simplify_data

CWD = Path(__file__).parent.resolve()
assert CWD.is_dir()

POSSIBLE_EXTENSIONS = {".yaml", ".json", ".hdf", ".zarr"}


def get_compatibility_files():
    """find all files that need to be checked for compatibility"""
    for path in CWD.glob("**/*.*"):
        if path.suffix in POSSIBLE_EXTENSIONS:
            yield path


def assert_data_equals(left: Any, right: Any) -> bool:
    """checks whether two objects are equal, also supporting :class:~numpy.ndarray`

    Args:
        left: one object
        right: other object

    Returns:
        bool: Whether the two objects are equal
    # treat numpy array first, since only one of the sides might have been cast to a
    """
    if type(left) is type(right):
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
                print(l, r)
                assert_data_equals(l, r)

        else:
            assert left == right

    elif isinstance(left, set) or isinstance(right, set):
        # one of the operands is a set, while the other is ordered
        assert set(left) == set(right)

    elif isinstance(left, np.ndarray) or isinstance(right, np.ndarray):
        # one of the operands numpy array, while the other one is a list
        print("L", left)
        print("R", right)
        assert np.array_equal(np.asarray(left), np.asarray(right))

    else:
        assert type(left) == type(right)  # cause failure


@pytest.mark.parametrize("path", get_compatibility_files())
def test_reading_compatibility(path):
    """test reading old files"""
    result = Result.from_file(path)

    with open(path.with_suffix(".pkl"), "rb") as fp:
        data = pickle.load(fp)

    if isinstance(data, StateBase):
        # newer data
        print("RESULT", result.state.__class__)
        print("DATA", data.__class__)
        assert_data_equals(result.state, data)
        # assert simplify_data(result.state._to_simple_objects()) == simplify_data(
        #     data._to_simple_objects()
        # )
    else:
        # older formats
        for key in data:
            assert_data_equals(
                simplify_data(result.data[key]), simplify_data(data[key])
            )
