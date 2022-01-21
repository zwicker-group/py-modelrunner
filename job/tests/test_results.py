"""
.. codeauthor:: David Zwicker <david.zwicker@ds.mpg.de>
"""

import numpy as np
import pytest

from ..results import Result


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
