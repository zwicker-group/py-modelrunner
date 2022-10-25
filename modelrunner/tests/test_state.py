"""
.. codeauthor:: David Zwicker <david.zwicker@ds.mpg.de>
"""

import numpy as np
import pytest

<<<<<<< Upstream, based on main
from modelrunner.state import ArrayCollectionState, ArrayState, DictState, ObjectState
from modelrunner.trajectory import Trajectory, TrajectoryWriter

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
def test_trajectory(state, ext, tmp_path):
    """test simple state IO"""
    path = tmp_path / ("file." + ext)

    with TrajectoryWriter(path, attrs={"test": "yes"}) as write:
        write(state, 1)
        write(state)

    for ret_copy in [True, False]:
        traj = Trajectory(path, ret_copy=ret_copy)
        assert len(traj) == 2
        np.testing.assert_allclose(traj.times, [1, 2])
        assert traj[1] == state
        assert traj[-1] == state
        assert traj.attributes == {"test": "yes"}

        for s in traj:
            assert s == state
=======
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
>>>>>>> 4ebae4d Added first tests and fixed some bugs
