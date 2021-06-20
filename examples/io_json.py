#!/usr/bin/env python3
"""
.. codeauthor:: David Zwicker <david.zwicker@ds.mpg.de>
"""

from model import ModelResult, function_model_command_line


def multiply(a: float = 1, b: float = 2):
    """Multiply two numbers"""
    return a * b


if __name__ == "__main__":
    mdl = function_model_command_line(multiply)
    mdl.get_result().write_to_json("test.json")

    read = ModelResult.from_json("test.json")
    print(read.parameters, "–– a * b =", read.results)
