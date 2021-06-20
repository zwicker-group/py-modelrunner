#!/usr/bin/env python3
"""
The script expects the output file as the first argument

.. codeauthor:: David Zwicker <david.zwicker@ds.mpg.de>
"""

import sys

from model.cluster import submit_job


def multiply(a: float = 1, b: float = 2):
    """Multiply two numbers"""
    return a * b


if __name__ == "__main__":
    submit_job(__file__, sys.argv[1])
