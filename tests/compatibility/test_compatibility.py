"""
.. codeauthor:: David Zwicker <david.zwicker@ds.mpg.de>
"""

import pickle
from pathlib import Path

import pytest

from modelrunner.results import Result
from modelrunner.state import _equals, simplify_data

CWD = Path(__file__).parent.resolve()
assert CWD.is_dir()


def get_compatibility_files():
    """find all files that need to be checked for compatibility"""
    for path in CWD.glob("**/*.*"):
        if "__pycache__" in str(path):
            continue
        if path.suffix in {".pkl", ".py"} or path.name == ".DS_Store":
            continue
        yield path


@pytest.mark.parametrize("path", get_compatibility_files())
def test_reading_compatibility(path):
    """test reading old files"""
    result = Result.from_file(path)

    with open(path.with_suffix(".pkl"), "rb") as fp:
        data = pickle.load(fp)

    for key in data:
        assert _equals(simplify_data(result.data[key]), simplify_data(data[key]))
