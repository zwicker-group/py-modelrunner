"""
.. codeauthor:: David Zwicker <david.zwicker@ds.mpg.de>
"""

import numpy as np
import pytest

from modelrunner.storage.utils import decode_binary, encode_binary


@pytest.mark.parametrize("obj", [[True, 1], np.arange(5)])
@pytest.mark.parametrize("binary", [True, False])
def test_object_encoding(obj, binary):
    """test encoding and decoding"""
    obj2 = decode_binary(encode_binary(obj, binary=binary))
    if isinstance(obj, np.ndarray):
        np.testing.assert_array_equal(obj, obj2)
    else:
        assert obj == obj2
