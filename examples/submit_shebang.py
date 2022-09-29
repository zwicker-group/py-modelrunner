#!/usr/bin/env python3 -m modelrunner.run --output data.hdf5 --method local
"""
This example shows how to submit a model to a queuing system using the magic line above.

Note that the method `local` just runs the script locally using a background process.
The script will thus start the job and return to the prompt immediately, thus not really
queuing. To actually queue a job on a high performance computing cluster, replace the
`method` argument by something more suitable; see the documentation. 

.. codeauthor:: David Zwicker <david.zwicker@ds.mpg.de>
"""


def multiply(a: float = 1, b: float = 2):
    """Multiply two numbers"""
    return a * b
