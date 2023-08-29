"""
.. codeauthor:: David Zwicker <david.zwicker@ds.mpg.de>
"""

import copy
import pickle

import numpy as np
import pytest

from modelrunner.state import (
    ArrayCollectionState,
    ArrayState,
    DictState,
    ObjectState,
    StateBase,
)
from modelrunner.state.base import _equals
from utils.states import EXTENSIONS, get_states


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


@pytest.mark.parametrize("state", get_states(add_derived=False))
def test_state_pickle(state):
    """test basic properties of states"""
    s4 = pickle.loads(pickle.dumps(state))
    assert state is not s4
    assert state == s4


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


@pytest.mark.parametrize(
    "state_cls", [ArrayState, DictState, ObjectState, ArrayCollectionState]
)
def test_state_attributes_implicit(state_cls, tmp_path):
    """test simple state IO with implicitely defined attributes"""

    class MyState(state_cls):
        _state_attributes_attr_name = "attrs"

        def __init__(self, value):
            self.attrs = {"value": value}
            if state_cls is ArrayState:
                self.data = np.arange(5)
            elif state_cls is DictState:
                self.data = {"a": ArrayState(np.arange(5))}
            elif state_cls is ArrayCollectionState:
                self.data = (np.arange(5),)
            else:
                self.data = None

    state = MyState("hello")
    state2 = state.copy("clean")
    assert state == state2
    assert state2.attrs["value"] == "hello"

    for ext in EXTENSIONS:
        path = tmp_path / ("file." + ext)
        state.to_file(path)
        state2 = StateBase.from_file(path)
        assert state == state2
        assert state2.attrs["value"] == "hello"

    nested = MyState("inner")
    state = MyState(nested)
    state2 = state.copy("shallow")
    assert state.attrs["value"] is state2.attrs["value"] is nested
    assert state.data is None or (state.data is state2.data)

    state2 = state.copy("data")
    assert state.attrs["value"] is state2.attrs["value"] is nested
    assert state.data is None or (state.data is not state2.data)

    state2 = state.copy("clean")
    assert state.attrs["value"] is not state2.attrs["value"]
    assert state.data is None or (state.data is not state2.data)


@pytest.mark.parametrize(
    "state_cls", [ArrayState, DictState, ObjectState, ArrayCollectionState]
)
def test_state_attributes_explicit(state_cls, tmp_path):
    """test simple state IO with explicitely defined attributes"""

    class MyState(state_cls):
        def __init__(self, value):
            self.value = value
            if state_cls is ArrayState:
                self.data = np.arange(5)
            elif state_cls is DictState:
                self.data = {"a": ArrayState(np.arange(5))}
            elif state_cls is ArrayCollectionState:
                self.data = (np.arange(5),)
            else:
                self.data = None

        @property
        def _state_attributes(self):
            attrs = super()._state_attributes
            attrs["value"] = self.value
            return attrs

        @_state_attributes.setter
        def _state_attributes(self, attrs):
            self.value = attrs["value"]

    state = MyState("hello")
    state2 = state.copy(method="clean")
    assert state == state2
    assert state2.value == "hello"

    for ext in EXTENSIONS:
        path = tmp_path / ("file." + ext)
        state.to_file(path)
        state2 = StateBase.from_file(path)
        assert state == state2
        assert state2.value == "hello"


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
