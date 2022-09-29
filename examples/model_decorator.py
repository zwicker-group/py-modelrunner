#!/usr/bin/env python3
"""
This example shows defining a custom model class using a decorator on a function.

.. codeauthor:: David Zwicker <david.zwicker@ds.mpg.de>
"""

import modelrunner


@modelrunner.make_model
def multiply(a, b=2):
    return a * b


# use the model instance
print(multiply.parameters)
print(multiply(a=3))
