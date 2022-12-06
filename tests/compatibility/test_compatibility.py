"""
.. codeauthor:: David Zwicker <david.zwicker@ds.mpg.de>
"""

import glob
from pathlib import Path

import pytest

from modelrunner.results import Result

CWD = Path(__file__).parent.resolve()


@pytest.mark.parametrize("path", glob.glob(str(CWD / "?/*.*")))
def test_reading_compatibility(path):
    """test reading old files"""
    Result.from_file(path)
