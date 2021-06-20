#!/usr/bin/env python3
"""
The script expects the output file as the first argument

.. codeauthor:: David Zwicker <david.zwicker@ds.mpg.de>
"""

from pathlib import Path

from model.cluster import submit_job


def multiply(a: float = 1, b: float = 2):
    """Multiply two numbers"""
    return a * b


if __name__ == "__main__":
    output_folder = Path("data")
    for a in [1, 2]:
        for b in [4, 5]:
            name = f"job_a_{a}_b_{b}"
            output = output_folder / f"{name}.json"
            submit_job(__file__, output=output, name=name, parameters={"a": a, "b": b})
