#!/usr/bin/env python3
"""
.. codeauthor:: David Zwicker <david.zwicker@ds.mpg.de>
"""

from model import ModelBase


class MyModel(ModelBase):

    parameters_default = {"a": 1, "b": 2}

    def __call__(self):
        return self.parameters["a"] * self.parameters["b"]


mdl = MyModel({"a": 3})
print(mdl())
