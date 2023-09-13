"""
.. codeauthor:: David Zwicker <david.zwicker@ds.mpg.de>
"""

import shutil

import numpy as np
import pytest

from modelrunner.state import ArrayState, Trajectory, TrajectoryWriter
from utils.states import EXTENSIONS, get_states


def remove_file_or_folder(path):
    """removes file or folder `path` with all subfiles"""
    if path.is_dir():
        shutil.rmtree(path, ignore_errors=True)
    else:
        path.unlink()


@pytest.mark.parametrize("state", get_states())
@pytest.mark.parametrize("ext", ["", ".zarr"])
def test_trajectory_basic(state, ext, tmp_path):
    """test simple trajecotry writing"""
    path = tmp_path / ("file" + ext)

    # write first batch of data
    with TrajectoryWriter(path, attrs={"test": "yes"}) as writer:
        writer.append(state, 1)
        writer.append(state)

    # check that we can't accidentally overwrite
    with pytest.raises(RuntimeError):
        with TrajectoryWriter(path) as writer:
            ...

    # check first batch of data
    traj = Trajectory(path, ret_copy=False)
    assert len(traj) == 2
    np.testing.assert_allclose(traj.times, [1, 2])
    assert traj[1] == state
    assert traj[-1] == state
    assert traj[1] is traj[-1]
    assert traj._state_attributes == {"test": "yes"}

    for s in traj:
        assert s == state

    if ext == ".zarr":
        # zarr does not support deletion of data, so we choose a new path
        path2 = tmp_path / ("file2" + ext)
    else:
        path2 = path

    # write second batch of data
    writer = TrajectoryWriter(path2, attrs={"test": "no"}, access="full")
    writer.append(state, 2)
    writer.append(state)
    writer.close()

    # check second batch of data
    traj = Trajectory(path2, ret_copy=True)
    assert len(traj) == 2
    np.testing.assert_allclose(traj.times, [2, 3])
    assert traj[1] == state
    assert traj[-1] == state
    assert traj[1] is not traj[-1]
    assert traj._state_attributes == {"test": "no"}

    for s in traj:
        assert s == state

    remove_file_or_folder(path)
    if ext == ".zarr":
        remove_file_or_folder(path2)


@pytest.mark.parametrize("ext", ["", ".zarr"])
def test_trajectory_multiple_reads(ext, tmp_path):
    """test simultaneous reading of trajecotries"""
    path = tmp_path / ("file" + ext)
    state = ArrayState(np.arange(5))

    # write some data
    with TrajectoryWriter(path, attrs={"test": "yes"}) as writer:
        writer.append(state, 1)
        writer.append(state)

    # read the data
    t1 = Trajectory(path, ret_copy=False)
    t2 = Trajectory(path, ret_copy=False)
    assert len(t1) == len(t2) == 2
    np.testing.assert_allclose(t1.times, [1, 2])
    assert t1[0] == t1[1] == state
    assert t2[0] == t2[1] == state

    # try modifying the read data
    state1 = t1[0]
    state1.data[:] = 0  # this should only modify the temporary copy
    assert state1 != state
    assert t1[0] == t1[0] == state

    remove_file_or_folder(path)


@pytest.mark.parametrize("ext", ["", ".zarr"])
def test_trajectory_overwriting(ext, tmp_path):
    """test whether zarr zip files can be overwritten"""
    path = tmp_path / ("file" + ext)
    state = ArrayState(np.arange(5))

    # write some data
    with TrajectoryWriter(path, attrs={"test": "yes"}) as writer:
        writer.append(state, 1)
        writer.append(state)

    # try writing data without overwrite
    with pytest.raises(RuntimeError):
        with TrajectoryWriter(path, access="insert") as writer:
            writer.append(state)

    # try writing data with overwrite
    with TrajectoryWriter(path, access="full") as writer:
        writer.append(state)

    remove_file_or_folder(path)
