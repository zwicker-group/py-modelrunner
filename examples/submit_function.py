#!/usr/bin/env python3
"""
.. codeauthor:: David Zwicker <david.zwicker@ds.mpg.de>
"""

from job.run import submit_job


def multiply(a: float = 1, b: float = 2):
    """Multiply two numbers"""
    return a * b


if __name__ == "__main__":
    submit_job(__file__, output="data.hdf5", method="local")
