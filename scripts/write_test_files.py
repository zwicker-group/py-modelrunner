#!/usr/bin/env python
"""
.. codeauthor:: David Zwicker <david.zwicker@ds.mpg.de>
"""

import sys
from pathlib import Path

import numpy as np

# locate the module
PACKAGE = "modelrunner"  # name of the package that needs to be tested
PACKAGE_PATH = Path(__file__).resolve().parents[1]  # base path of the package
assert (PACKAGE_PATH / PACKAGE).is_dir()
sys.path.insert(0, str(PACKAGE_PATH))

from modelrunner import Result

# locate the storage for the compatibility files
FORMAT_VERSION = 1
STORAGE_PATH = PACKAGE_PATH / "tests" / "compatibility" / str(FORMAT_VERSION)
assert STORAGE_PATH.is_dir()


def main():
    """main function"""
    # prepare test result
    data = {
        "number": -1,
        "string": "test",
        "list_1d": [0, 1, 2],
        "list_2d": [[0, 1], [2, 3, 4]],
        "array": np.arange(5),
    }
    result = Result.from_data({"name": "model"}, data)

    # write data
    for extension in [".hdf", ".yaml", ".json"]:
        path = STORAGE_PATH / ("result" + extension)
        result.to_file(path)


if __name__ == "__main__":
    main()
