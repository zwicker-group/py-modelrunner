#!/usr/bin/env python3
"""
This example shows defining a custom model class by subclassing.

.. codeauthor:: David Zwicker <david.zwicker@ds.mpg.de>
"""

from modelrunner import ModelBase


class MyModel(ModelBase):

    parameters_default = {"a": 1, "b": 2}

    def __call__(self):
        return self.parameters["a"] * self.parameters["b"]


# create an instance of the model
model = MyModel({"a": 3})
# run the instance
print(model())
