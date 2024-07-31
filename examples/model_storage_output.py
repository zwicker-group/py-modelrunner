#!/usr/bin/env python3
"""This example shows defining a custom model that stores additional data.

.. codeauthor:: David Zwicker <david.zwicker@ds.mpg.de>
"""

import tempfile

from modelrunner import make_model, open_storage


def multiply(a, b=2, storage=None):
    storage["data"] = {"additional": "information"}
    return a * b


with tempfile.NamedTemporaryFile(suffix=".yaml") as fp:
    # create an instance of the model defined by the function
    model = make_model(multiply, {"a": 3}, output=fp.name)
    # run the instance and store the data
    model.write_result()

    # read the file and check whether all the data is there
    with open_storage(fp.name) as storage:
        print("Stored data:", storage["storage/data"])
        print("Model result:", storage["result"])
