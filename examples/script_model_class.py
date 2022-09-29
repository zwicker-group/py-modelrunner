#!/usr/bin/env python3 -m modelrunner
"""
This example displays a minimal script containing a model class. 

.. codeauthor:: David Zwicker <david.zwicker@ds.mpg.de>
"""

from modelrunner import ModelBase


class MyModel(ModelBase):

    parameters_default = {"a": 1, "b": 2}

    def __call__(self):
        print(self.parameters["a"] * self.parameters["b"])
