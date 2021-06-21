#!/usr/bin/env python3
"""
.. codeauthor:: David Zwicker <david.zwicker@ds.mpg.de>
"""

from job import run_function_with_cmd_args


def multiply(a: float = 1, b: float = 2):
    """Multiply two numbers"""
    return a * b


if __name__ == "__main__":
    result = run_function_with_cmd_args(multiply)
    print(result.result)
