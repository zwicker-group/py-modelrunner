#!/usr/bin/env python3
"""
This example displays a minimal script containing two model classes 

.. codeauthor:: David Zwicker <david.zwicker@ds.mpg.de>
"""

import sys

from modelrunner import ModelBase


class MyModel(ModelBase):

    parameters_default = {"a": 1, "b": 2}

    def __call__(self):
        return self.parameters["a"] * self.parameters["b"]


class MyDerivedModel(MyModel):

    parameters_default = {"c": 3}

    def __call__(self):
        print(super().__call__() + self.parameters["c"])


if __name__ == "__main__":
    MyDerivedModel.run_from_command_line(sys.argv[1:])
