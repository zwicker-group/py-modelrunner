"""
.. codeauthor:: David Zwicker <david.zwicker@ds.mpg.de>
"""

import numpy as np

from modelrunner.io import simplify_data


def test_simplify_data():
    """test the simplify_data function"""
    assert simplify_data(1) == 1
    assert simplify_data(1.5) == 1.5
    assert simplify_data("1") == "1"
    assert simplify_data((1, "1")) == [1, "1"]
    assert simplify_data([1, "1"]) == [1, "1"]
    assert simplify_data([1, (1, "1")]) == [1, [1, "1"]]
    assert simplify_data(np.arange(3)) == [0, 1, 2]
    assert simplify_data(np.array([1])) == [1]
    assert simplify_data(np.array(1)).__class__ is int
    assert simplify_data(np.int32(2)).__class__ is int
    assert simplify_data(np.float32(2.5)).__class__ is float
