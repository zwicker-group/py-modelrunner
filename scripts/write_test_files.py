#!/usr/bin/env python
"""
.. codeauthor:: David Zwicker <david.zwicker@ds.mpg.de>
"""

import importlib
from pathlib import Path

import numpy as np

FORMAT_VERSION = 0
PACKAGE = "modelrunner"  # name of the package that needs to be tested
PACKAGE_PATH = Path(__file__).resolve().parents[1]  # base path of the package
STORAGE_PATH = PACKAGE_PATH / "tests" / "compatibility" / str(FORMAT_VERSION)


def main():
    """main function"""
    # load the modelrunner function
    path = PACKAGE_PATH / PACKAGE / "__init__.py"
    spec = importlib.util.spec_from_file_location(PACKAGE, path)
    mr = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mr)

    # prepare test result
    data = {
        "number": -1,
        "string": "test",
        "list_1d": [0, 1, 2],
        "list_2d": [[0, 1], [2, 3, 4]],
        "array": np.arange(5),
    }
    result = mr.Result.from_data({"name": "model"}, data)

    # write data
    for extension in [".hdf", ".yaml", ".json"]:
        path = STORAGE_PATH / ("result" + extension)
        result.write_to_file(path)


if __name__ == "__main__":
    main()
