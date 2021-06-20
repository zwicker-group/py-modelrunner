#!/usr/bin/env python3
"""
.. codeauthor:: David Zwicker <david.zwicker@ds.mpg.de>
"""

from model import get_function_model


def multiply(a, b=2):
    return a * b


mdl = get_function_model(multiply, {"a": 3})
print(mdl())
