"""
.. codeauthor:: David Zwicker <david.zwicker@ds.mpg.de>
"""

import numpy as np
import pytest

from modelrunner.state import (
    ArrayCollectionState,
    DictState,
    ObjectState,
    StateBase,
    _equals,
)
from utils.states import EXTENSIONS, get_states


@pytest.mark.parametrize("state", get_states())
@pytest.mark.parametrize("ext", EXTENSIONS)
def test_state_io(state, ext, tmp_path):
    """test simple state IO"""
    path = tmp_path / ("file." + ext)

    state.to_file(path)
    with pytest.raises(FileExistsError):
        state.to_file(path, overwrite=False)
    state.to_file(path, overwrite=True)

    read = StateBase.from_file(path)
    assert state == read


@pytest.mark.parametrize("state_cls", [DictState, ObjectState, ArrayCollectionState])
@pytest.mark.parametrize("ext", EXTENSIONS)
def test_empty_state_io(state_cls, ext, tmp_path):
    """test simple state IO"""
    state = state_cls()
    path = tmp_path / ("file." + ext)
    state.to_file(path)
    state2 = StateBase.from_file(path)
    assert state == state2


@pytest.mark.parametrize("ext", EXTENSIONS)
def test_array_collections(ext, tmp_path):
    """test some specific behaviors of the ArrayCollectionState"""
    a = np.arange(5)
    b = np.random.random(size=3)
    state = ArrayCollectionState((a, b), labels=["a", "b"])

    path = tmp_path / ("file." + ext)
    state.to_file(path)
    state2 = StateBase.from_file(path)
    assert _equals(state.data, state2.data)
    assert state.labels == state2.labels


def test_attribute_packing():
    """test whether attribute packing is properly called"""

    class MyState(ObjectState):
        @property
        def attributes(self):
            return {"a": 1, "b": 2}

        def _pack_attribute(self, name, value):
            if name == "a":
                return "PACKED"
            else:
                return super()._pack_attribute(name, value)

        @classmethod
        def _unpack_attribute(cls, name, value):
            if name == "a":
                return value.lower()
            else:
                return super()._unpack_attribute(name, value)

    state = MyState({"a"})
    assert state.attributes == {"a": 1, "b": 2}
    # test whether the packed attributes contain everything on the right
    assert state._attributes_store.items() >= {"a": "PACKED", "b": 2}.items()
