"""
.. codeauthor:: David Zwicker <david.zwicker@ds.mpg.de>
"""

import numpy as np
import pytest

from modelrunner.state import ArrayState, DictState, ObjectState, StateBase

EXTENSIONS = ["json", "yaml", "zarr"]


def get_states():
    """generate multiple states"""
    a = np.arange(5)
    o = {"list": [1, 2], "bool": True}
    return [
        ArrayState(a),
        ObjectState(o),
        DictState({"o": ObjectState(o), "a": ArrayState(a)}),
    ]


@pytest.mark.parametrize("state", get_states())
@pytest.mark.parametrize("ext", EXTENSIONS)
def test_state_io(state, ext, tmp_path):
    """test simple state IO"""
    path = tmp_path / ("file." + ext)
    state.to_file(path)
    state2 = StateBase.from_file(path)
    print(f"{state.data=}")
    print(f"{state2.data=}")
    assert state == state2
