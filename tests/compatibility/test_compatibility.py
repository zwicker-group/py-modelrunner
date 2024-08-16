"""
.. codeauthor:: David Zwicker <david.zwicker@ds.mpg.de>
"""

import pickle
from pathlib import Path

import pytest

from helpers import assert_data_equals
from modelrunner import Result

CWD = Path(__file__).parent.resolve()
assert CWD.is_dir()

POSSIBLE_EXTENSIONS = {".yaml", ".json", ".hdf", ".zarr", ".zip"}


def get_compatibility_files(version=None):
    """Find all files that need to be checked for compatibility."""
    for path in CWD.glob("**/*.*"):
        if path.suffix in POSSIBLE_EXTENSIONS:
            if path.parts[-2].startswith("_"):
                pytest.skip("Skip compatibility files starting with underscore")
            if version is None or path.parts[-2] == str(version):
                yield path


@pytest.mark.parametrize("path", get_compatibility_files())
def test_reading_compatibility(path):
    """Test reading old files."""
    result = Result.from_file(path)

    with path.with_suffix(".pkl").open("rb") as fp:
        try:
            data = pickle.load(fp)
        except ModuleNotFoundError:
            assert result.result is not None  # just test whether something was loaded
        else:
            assert_data_equals(result.result, data, fuzzy=True)
