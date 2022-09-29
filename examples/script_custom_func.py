#!/usr/bin/env python3 -m modelrunner
"""
This example shows how a function is turned into a model using decorators.

.. codeauthor:: David Zwicker <david.zwicker@ds.mpg.de>
"""

import modelrunner


def do_not_calculate(a=1, b=2):
    """This function should not be run"""
    raise RuntimeError("This must not run")


@modelrunner.make_model
def calculate(a=1, b=2):
    """This function has been marked as a model"""
    print(a * b)
