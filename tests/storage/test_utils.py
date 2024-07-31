"""
.. codeauthor:: David Zwicker <david.zwicker@ds.mpg.de>
"""

import numpy as np
import pytest

from modelrunner.storage import MemoryStorage
from modelrunner.storage.utils import (
    decode_binary,
    decode_class,
    encode_binary,
    encode_class,
)


@pytest.mark.parametrize("obj", [[True, 1], np.arange(5)])
@pytest.mark.parametrize("binary", [True, False])
def test_object_encoding(obj, binary):
    """Test encoding and decoding."""
    obj2 = decode_binary(encode_binary(obj, binary=binary))
    if isinstance(obj, np.ndarray):
        np.testing.assert_array_equal(obj, obj2)
    else:
        assert obj == obj2


def test_decode_class():
    """Test decode class."""
    cls_name = encode_class(MemoryStorage)
    assert "MemoryStorage" in cls_name
    cls_loaded = decode_class(cls_name)
    assert cls_loaded is MemoryStorage

    with pytest.raises(ImportError):
        decode_class(1)

    with pytest.raises(ImportError):
        decode_class("modelrunner.NonSenseClass")

    with pytest.raises(ModuleNotFoundError):
        decode_class("nonsense.class.that.does.not.exist")
