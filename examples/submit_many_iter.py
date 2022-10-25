#!/usr/bin/env python3
"""
This example shows how to submit the same model with multiple parameters.

Note that the method `local` just runs the script locally, thus not really queuing. To
actually queue a job on a high performance computing cluster, replace the `method`
argument by something more suitable; see the documentation. 

.. codeauthor:: David Zwicker <david.zwicker@ds.mpg.de>
"""


from modelrunner import make_model, submit_job


@make_model
def main(a: float = 1, b: float = 2):
    """Multiply two numbers"""
    return a * b


if __name__ == "__main__":
    for a in [1, 2]:
        for b in [4, 5]:
            name = f"job_a_{a}_b_{b}"
            submit_job(
                __file__,  # submit this file as a job module
                output=f"data/{name}.json",
                name=name,
                parameters={"a": a, "b": b},
                method="foreground",  # run job locally
            )
