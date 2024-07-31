#!/usr/bin/env python3
"""This example shows defining a custom model class by subclassing.

.. codeauthor:: David Zwicker <david.zwicker@ds.mpg.de>
"""

import numpy as np

from modelrunner import ModelBase


class MyModel(ModelBase):
    parameters_default = {"a": 1, "b": 2}

    def __call__(self):
        self.storage.write_array("arr", np.arange(4))
        self.storage.write_attrs("arr", {"key": "value"})  # write extra information
        return self.parameters["a"] * self.parameters["b"]


# create an instance of the model
model = MyModel({"a": 3}, output="test.yaml")
# run the instance
print(model())
