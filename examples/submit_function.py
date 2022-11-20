#!/usr/bin/env python3
"""
This example shows how to submit a model to a queuing system.

Note that the method `local` just runs the script locally, thus not really queuing. To
actually queue a job on a high performance computing cluster, replace the `method`
argument by something more suitable; see the documentation. 

.. codeauthor:: David Zwicker <david.zwicker@ds.mpg.de>
"""

from modelrunner import make_model, submit_job


@make_model
def multiply(a: float = 1, b: float = 2):
    """Multiply two numbers"""
    return a * b


if __name__ == "__main__":
    submit_job(__file__, output="data.json", method="foreground")
