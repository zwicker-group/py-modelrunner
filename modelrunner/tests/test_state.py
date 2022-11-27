"""
.. codeauthor:: David Zwicker <david.zwicker@ds.mpg.de>
"""

import numpy as np
import pytest

from modelrunner.state import (
    ArrayCollectionState,
    ArrayState,
    DictState,
    ObjectState,
    StateBase,
    _equals,
)

EXTENSIONS = ["json", "yaml", "zarr"]


def get_states():
    """generate multiple states"""
    a = np.arange(5)
    b = np.random.random(size=3)
    o = {"list": [1, 2], "bool": True}
    return [
        ObjectState(o),
        ArrayState(a),
        ArrayCollectionState((a, b), labels=["a", "b"]),
        DictState({"o": ObjectState(o), "a": ArrayState(a)}),
    ]


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
