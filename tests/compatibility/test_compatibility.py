"""
.. codeauthor:: David Zwicker <david.zwicker@ds.mpg.de>
"""

import pickle
from pathlib import Path

import pytest

from modelrunner.results import Result, StateBase
from modelrunner.state.base import _equals, simplify_data

CWD = Path(__file__).parent.resolve()
assert CWD.is_dir()

POSSIBLE_EXTENSIONS = {".yaml", ".json", ".hdf", ".zarr"}


def get_compatibility_files():
    """find all files that need to be checked for compatibility"""
    for path in CWD.glob("**/*.*"):
        if path.suffix in POSSIBLE_EXTENSIONS:
            yield path


@pytest.mark.parametrize("path", get_compatibility_files())
def test_reading_compatibility(path):
    """test reading old files"""
    result = Result.from_file(path)

    with open(path.with_suffix(".pkl"), "rb") as fp:
        data = pickle.load(fp)

    if isinstance(data, StateBase):
        # newer data
        assert simplify_data(result.state._to_simple_objects()) == simplify_data(
            data._to_simple_objects()
        )
    else:
        # older formats
        for key in data:
            assert _equals(simplify_data(result.data[key]), simplify_data(data[key]))
