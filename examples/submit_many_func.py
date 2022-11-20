#!/usr/bin/env python3
"""
This example shows how to submit the same model with multiple parameters.

Note that the method `local` just runs the script locally, thus not really queuing. To
actually queue a job on a high performance computing cluster, replace the `method`
argument by something more suitable; see the documentation. 

.. codeauthor:: David Zwicker <david.zwicker@ds.mpg.de>
"""


from modelrunner import make_model, submit_jobs


@make_model
def main(a: float = 1, b: float = 2):
    """Multiply two numbers"""
    return a * b


if __name__ == "__main__":
    submit_jobs(
        __file__,  # submit this file as a job module
        output_folder="data",
        parameters={"a": [1, 2], "b": [4, 5]},
        method="foreground",  # run job locally
    )
