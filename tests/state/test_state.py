"""
.. codeauthor:: David Zwicker <david.zwicker@ds.mpg.de>
"""

import copy

import numpy as np
import pytest

from modelrunner.state import ArrayCollectionState, DictState, ObjectState, StateBase
from modelrunner.state.base import _equals
from utils.states import EXTENSIONS, get_states


@pytest.mark.parametrize("state", get_states())
def test_state_basic(state):
    """test basic properties of states"""
    assert state.__class__.__name__ in StateBase._state_classes

    s2 = state.copy()
    assert state is not s2
    assert state == s2

    s3 = copy.copy(state)
    assert state is not s3
    assert state == s3


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
    assert _equals(state._state_data, state2._state_data)
    assert state.labels == state2.labels
