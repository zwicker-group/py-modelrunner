"""
.. codeauthor:: David Zwicker <david.zwicker@ds.mpg.de>
"""

import numpy as np
import pytest

from modelrunner.state import ArrayState, DictState, ObjectState


@pytest.mark.parametrize("method", ["shallow", "data", "clean"])
def test_dict_state_copy(method):
    """test basic properties of states"""
    # define dict state classes
    arr_state = ArrayState(np.arange(5))
    arr_state._extra = "array"
    obj_state = ObjectState({"list": [1, 2], "bool": True})
    state = DictState({"a": arr_state, "o": obj_state})
    state._extra = "state"

    # copy everything by copying __dict__
    s_c = state.copy(method)
    assert state == s_c
    assert state is not s_c

    if method == "clean":
        assert not hasattr(s_c, "_extra")
        assert not hasattr(s_c["a"], "_extra")
    else:
        assert s_c._extra == "state"
        assert s_c["a"]._extra == "array"

    if method == "shallow":
        assert state._state_data is s_c._state_data
        assert state["a"] is s_c["a"]
    else:
        assert state._state_data is not s_c._state_data
        assert state["a"] is not s_c["a"]
        assert state["a"].data is not s_c["a"].data
