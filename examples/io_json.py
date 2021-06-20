#!/usr/bin/env python3
"""
.. codeauthor:: David Zwicker <david.zwicker@ds.mpg.de>
"""

from model import Result, run_function_with_cmd_args


def multiply(a: float = 1, b: float = 2):
    """Multiply two numbers"""
    return a * b


if __name__ == "__main__":
    result = run_function_with_cmd_args(multiply)
    result.write_to_json("test.json")

    read = Result.from_json("test.json")
    print(read.parameters, "–– a * b =", read.result)
