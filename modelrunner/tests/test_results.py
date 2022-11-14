"""
.. codeauthor:: David Zwicker <david.zwicker@ds.mpg.de>
"""

import numpy as np
import pytest

from ..results import Result, ResultCollection, simplify_data


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


@pytest.mark.parametrize("extension", [".hdf", ".yaml", ".json"])
def test_result_serialization(extension, tmp_path):
    """test reading and writing results"""
    # prepare test result
    data = {
        "number": -1,
        "string": "test",
        "list_1d": [0, 1, 2],
        "list_2d": [[0, 1], [2, 3, 4]],
        "array": np.arange(5),
    }
    result = Result.from_data({"name": "model"}, data)

    # write data
    path = tmp_path / ("test" + extension)
    result.write_to_file(path)

    # read data
    read = Result.from_file(path)
    assert read.model.name == "model"
    np.testing.assert_equal(read.result, result.result)


def test_result_collections():
    """test some aspects of result collections"""
    p1 = {"a": 1, "b": [0, 1], "c": "c"}
    r1 = Result.from_data({"name": "1", "parameters": p1}, p1)
    p2 = {"a": 2, "b": [0, 1], "c": "c"}
    r2 = Result.from_data({"name": "2", "parameters": p2}, p2)
    p3 = {"a": 1, "b": [0, 1, 2], "c": "c"}
    r3 = Result.from_data({"name": "3", "parameters": p3}, p3)
    rc = ResultCollection([r1, r2, r3])
    assert len(rc) == 3

    assert rc.constant_parameters == {"c": "c"}
    assert rc.varying_parameters == {"a": [1, 2], "b": [(0, 1), (0, 1, 2)]}

    assert rc.get(a=1).model.name == "1"
    assert rc.get(b=[0, 1, 2]).model.name == "3"
    with pytest.raises(ValueError):
        rc.get(a=4)

    assert len(rc.filtered(a=1)) == 2
    assert len(rc.filtered(a=2)) == 1
    assert len(rc.filtered(a=3)) == 0

    assert len(rc.filtered(b=[0, 1])) == 2
    assert len(rc.filtered(b=[0, 1, 2])) == 1

    rc2 = rc.sorted("a")
    assert rc2[0].parameters["a"] == 1
    assert rc2[1].parameters["a"] == 1
    assert rc2[2].parameters["a"] == 2

    rc = ResultCollection([r1, r1])
    assert len(rc) == 2
    rc2 = rc.remove_duplicates()
    assert len(rc2) == 1


def test_collection_groupby():
    """test grouping of result collections"""
    p1 = {"a": 1, "b": (0, 1), "c": "c"}
    r1 = Result.from_data({"name": "1", "parameters": p1}, p1)
    p2 = {"a": 2, "b": (0, 1), "c": "c"}
    r2 = Result.from_data({"name": "2", "parameters": p2}, p2)
    p3 = {"a": 1, "b": (0, 1, 2), "c": "c"}
    r3 = Result.from_data({"name": "3", "parameters": p3}, p3)
    rc = ResultCollection([r1, r2, r3])

    with pytest.raises(KeyError):
        list(rc.groupby("nan"))

    assert len(list(rc.groupby("c"))) == 1

    for p, r in rc.groupby("a"):
        if p == {"a": 1}:
            assert len(r) == 2
        elif p == {"a": 2}:
            assert len(r) == 1
        else:
            raise AssertionError

    groups = list(r for p, r in rc.groupby("a", "b"))
    assert len(groups) == 3
