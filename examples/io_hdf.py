#!/usr/bin/env python3
"""
This example shows reading and writing data using HDF files.

.. codeauthor:: David Zwicker <david.zwicker@ds.mpg.de>
"""

import numpy as np

from modelrunner import Result, run_function_with_cmd_args


def number_range(start: float = 1, length: int = 3):
    """create an ascending list of numbers"""
    return start + np.arange(length)


if __name__ == "__main__":
    # write result to file
    result = run_function_with_cmd_args(number_range)
    result.write_to_hdf("test.hdf")

    # write result from file
    read = Result.from_hdf("test.hdf")
    print(read.parameters, "–– start + [0..length-1] =", read.result)
