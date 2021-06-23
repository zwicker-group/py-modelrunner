#!/usr/bin/env python3 -m job
"""
This example shows how a function is turned into a model using decorators.

.. codeauthor:: David Zwicker <david.zwicker@ds.mpg.de>
"""

import job


def do_not_calculate(a=1, b=2):
    """This function should not be run"""
    raise RuntimeError("This must not run")


@job.make_model
def calculate(a=1, b=2):
    """Multiply two numbers"""
    print(a * b)
