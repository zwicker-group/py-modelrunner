#!/usr/bin/env python3
"""
.. codeauthor:: David Zwicker <david.zwicker@ds.mpg.de>
"""

import numpy as np
from model import ModelResult, function_model_command_line


def number_range(start: float = 1, length: int = 3):
    """create an ascending list of numbers"""
    return start + np.arange(length)


if __name__ == "__main__":
    mdl = function_model_command_line(number_range)
    mdl.get_result().write_to_hdf("test.hdf")

    read = ModelResult.from_hdf("test.hdf")
    print(read.parameters, "–– start + [0..length-1] =", read.results)
