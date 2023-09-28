"""
.. codeauthor:: David Zwicker <david.zwicker@ds.mpg.de>
"""

import pickle

import numpy as np
import pytest

from helpers import assert_data_equals, get_states, storage_extensions
from modelrunner.state import ArrayCollectionState, DictState, ObjectState, StateBase
from modelrunner.state.base import _equals

STORAGE_EXT = storage_extensions(incl_folder=True, dot=False)


@pytest.mark.parametrize("state", get_states())
@pytest.mark.parametrize("method", ["shallow", "data", "clean"])
def test_state_basic(state, method):
    """test basic properties of states"""
    assert state.__class__.__name__ in StateBase._state_classes
    state._extra_info = "test"

    # copy everything by copying __dict__
    s_c = state.copy(method)
    assert state == s_c
    assert state is not s_c

    if method == "clean":
        assert not hasattr(s_c, "_extra_info")
    else:
        assert s_c._extra_info == "test"

    if method == "shallow":
        assert state._state_data is s_c._state_data
    else:
        assert state._state_data is not s_c._state_data


@pytest.mark.parametrize("state", get_states())
def test_state_pickle(state):
    """test basic properties of states"""
    s4 = pickle.loads(pickle.dumps(state))
    assert state is not s4
    assert state == s4


@pytest.mark.parametrize("state", get_states())
@pytest.mark.parametrize("ext", STORAGE_EXT)
def test_state_io(state, ext, tmp_path):
    """test simple state IO"""
    path = tmp_path / ("file." + ext)

    state.to_file(path)
    with pytest.raises(RuntimeError):
        state.to_file(path, mode="insert")
    if ext == "json":
        state.to_file(path, mode="truncate", simplify=False)  # truncate file
    else:
        state.to_file(path, mode="truncate")  # truncate file

    read = StateBase.from_file(path)
    if True or ext == "json":
        assert state.__class__ == read.__class__
        assert state._state_attributes == read._state_attributes
        assert_data_equals(state._state_data, read._state_data)
    else:
        assert state == read


@pytest.mark.parametrize("state_cls", [DictState, ObjectState, ArrayCollectionState])
@pytest.mark.parametrize("ext", STORAGE_EXT)
def test_empty_state_io(state_cls, ext, tmp_path):
    """test simple state IO"""
    state = state_cls()
    path = tmp_path / ("file." + ext)
    state.to_file(path)
    state2 = StateBase.from_file(path)
    assert state == state2


@pytest.mark.parametrize("ext", STORAGE_EXT)
def test_array_collections(ext, tmp_path):
    """test some specific behaviors of the ArrayCollectionState"""
    a = np.arange(5)
    b = np.random.random(size=3)
    state = ArrayCollectionState((a, b), labels=["a", "b"])

    path = tmp_path / ("file." + ext)
    state.to_file(path)
    state2 = StateBase.from_file(path)
    assert _equals(state._state_data, state2._state_data)
    assert state.labels == state2.labels
