#!/usr/bin/env python
"""
.. codeauthor:: David Zwicker <david.zwicker@ds.mpg.de>
"""

import pickle
import sys
from pathlib import Path

import numpy as np

# locate the module
PACKAGE = "modelrunner"  # name of the package that needs to be tested
PACKAGE_PATH = Path(__file__).resolve().parents[1]  # base path of the package
assert (PACKAGE_PATH / PACKAGE).is_dir()
sys.path.insert(0, str(PACKAGE_PATH))

from modelrunner import ArrayCollectionState, ArrayState, DictState, ObjectState, Result

# locate the storage for the compatibility files
FORMAT_VERSION = 1
STORAGE_PATH = PACKAGE_PATH / "tests" / "compatibility" / str(FORMAT_VERSION)
assert STORAGE_PATH.is_dir()


DATASETS = {
    "object": ObjectState(
        {
            "number": -1,
            "string": "test",
            "list_1d": [0, 1, 2],
            "list_2d": [[0, 1], [2, 3, 4]],
            "array": np.arange(5),
        }
    ),
    "array": ArrayState(np.arange(3)),
    "array_col": ArrayCollectionState([np.arange(2), np.arange(3)], labels="ab"),
    "dict": DictState({"a": ObjectState({"a", "b"}), "b": ArrayState(np.arange(3))}),
}
EXTENSIONS = [".yaml", ".json", ".zarr"]


def create_files(name, data, extensions=EXTENSIONS):
    """create example file

    Args:
        name (str): The name of the dataset
        data: The data contained in the result
        extensions: The extensions defining the file formats being used
    """
    # prepare test result
    result = Result.from_data({"name": "model"}, data)

    # write data
    for extension in extensions:
        path = STORAGE_PATH / (name + extension)
        try:
            result.to_file(path)
        except FileExistsError:
            pass

    # write the exact data to check later
    with open(STORAGE_PATH / (name + ".pkl"), "wb") as fp:
        pickle.dump(data, fp)


def main():
    """main function"""
    for key, value in DATASETS.items():
        create_files(key, value)


if __name__ == "__main__":
    main()
