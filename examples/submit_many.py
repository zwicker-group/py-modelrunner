#!/usr/bin/env python3
"""
The example `submit_many.py` must be ran to generate the data for this example.

.. codeauthor:: David Zwicker <david.zwicker@ds.mpg.de>
"""

from pathlib import Path

from model.hpc import submit_job


def main(a: float = 1, b: float = 2):
    """Multiply two numbers"""
    return a * b


if __name__ == "__main__":
    output_folder = Path("data")
    for a in [1, 2]:
        for b in [4, 5]:
            name = f"job_a_{a}_b_{b}"
            submit_job(
                __file__,  # submit this file as a job module
                output=output_folder / f"{name}.json",
                name=name,
                parameters={"a": a, "b": b},
                method="local",  # run job locally
            )
