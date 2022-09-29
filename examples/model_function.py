#!/usr/bin/env python3
"""
This example shows defining a custom model class using a function.

.. codeauthor:: David Zwicker <david.zwicker@ds.mpg.de>
"""

from modelrunner import make_model


def multiply(a, b=2):
    return a * b


# create an instance of the model defined by the function
model = make_model(multiply, {"a": 3})
# run the instance
print(model())
