#!/usr/bin/env python3
"""
This example shows how a collection of results can be read.

.. codeauthor:: David Zwicker <david.zwicker@ds.mpg.de>
"""

from job import ResultCollection

if __name__ == "__main__":
    rc = ResultCollection.from_folder("data")
    print(rc.dataframe)
