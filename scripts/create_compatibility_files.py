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

import contextlib

from modelrunner import Result, make_model_class

# locate the storage for the compatibility files
FORMAT_VERSION = "3"
STORAGE_PATH = PACKAGE_PATH / "tests" / "compatibility" / str(FORMAT_VERSION)
assert STORAGE_PATH.is_dir()


DATASETS = {
    "dict": {
        "number": -1,
        "string": "test",
        "list_1d": [0, 1, 2],
        "list_2d": [[0, 1], [2, 3, 4]],
        "array": np.arange(5),
    },
    "array": np.arange(3),
    "array_col": [np.arange(2), np.arange(3)],
    "nested": {"a": {"a", "b"}, "b": np.arange(3)},
}
EXTENSIONS = [".yaml", ".json", ".hdf", ".zip"]


def create_result_files(name, data, extensions=EXTENSIONS):
    """Create example file storing a simple result.

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
        with contextlib.suppress(FileExistsError):
            result.to_file(path)

    # write the exact data to check later
    with (STORAGE_PATH / f"{name}.pkl").open("wb") as fp:
        pickle.dump(data, fp)


def create_model_result_files(name, data, extensions=EXTENSIONS):
    """Create example file storing a trajectory.

    Args:
        name (str): The name of the dataset
        data: The data contained in the result
        extensions: The extensions defining the file formats being used
    """

    @make_model_class
    def MyModel(storage):
        storage["custom"] = data
        return data

    for extension in extensions:
        model = MyModel(output=STORAGE_PATH / f"model_{name}{extension}")
        model.write_result()

    # write the exact data to check later
    with (STORAGE_PATH / f"model_{name}.pkl").open("wb") as fp:
        pickle.dump(data, fp)


def main():
    """Main function."""
    for key, value in DATASETS.items():
        create_result_files(key, value)
        create_model_result_files(key, value)


if __name__ == "__main__":
    main()
