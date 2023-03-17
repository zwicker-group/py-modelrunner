"""
.. codeauthor:: David Zwicker <david.zwicker@ds.mpg.de>
"""

import numpy as np
import pytest

from modelrunner.state.trajectory import Trajectory, TrajectoryWriter
from utils.states import EXTENSIONS, get_states


@pytest.mark.parametrize("state", get_states())
@pytest.mark.parametrize("ext", EXTENSIONS)
def test_trajectory(state, ext, tmp_path):
    """test simple trajecotry writing"""
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
        assert traj._state_attributes == {"test": "yes"}

        for s in traj:
            assert s == state
