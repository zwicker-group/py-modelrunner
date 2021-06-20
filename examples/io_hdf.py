#!/usr/bin/env python3
"""
.. codeauthor:: David Zwicker <david.zwicker@ds.mpg.de>
"""

import numpy as np
from model import Result, run_function_with_cmd_args


def number_range(start: float = 1, length: int = 3):
    """create an ascending list of numbers"""
    return start + np.arange(length)


if __name__ == "__main__":
    result = run_function_with_cmd_args(number_range)
    result.write_to_hdf("test.hdf")

    read = Result.from_hdf("test.hdf")
    print(read.parameters, "–– start + [0..length-1] =", read.result)
