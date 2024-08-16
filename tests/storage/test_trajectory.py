"""
.. codeauthor:: David Zwicker <david.zwicker@ds.mpg.de>
"""

import shutil

import numpy as np
import pytest

from helpers import assert_data_equals, storage_extensions
from modelrunner.storage import Trajectory, TrajectoryWriter, open_storage

STORAGE_EXT = storage_extensions(incl_folder=True, dot=True, exclude=[".yaml", ".json"])
STORAGE_OBJECTS = [
    {"n": -1, "s": "t", "l1": [0, 1, 2], "l2": [[0, 1], [4]], "a": np.arange(5)},
    np.arange(3),
    [np.arange(2), np.arange(3)],
    {"a": {"a", "b"}, "b": np.arange(3)},
]


def remove_file_or_folder(path):
    """Removes file or folder `path` with all subfiles."""
    if path.is_dir():
        shutil.rmtree(path, ignore_errors=True)
    else:
        path.unlink()


@pytest.mark.parametrize("obj", STORAGE_OBJECTS)
@pytest.mark.parametrize("ext", STORAGE_EXT)
def test_trajectory_basic(obj, ext, tmp_path):
    """Test simple trajecotry writing."""
    path = tmp_path / ("file" + ext)

    # write first batch of data
    with TrajectoryWriter(path, attrs={"test": "yes"}) as writer:
        writer.append(obj, 1)
        writer.append(obj)

    # check that we can't accidentally overwrite
    with pytest.raises(RuntimeError), TrajectoryWriter(path) as writer:
        ...

    # check first batch of data
    traj = Trajectory(path)
    assert len(traj) == 2
    np.testing.assert_allclose(traj.times, [1, 2])
    assert_data_equals(traj[1], obj)
    assert_data_equals(traj[-1], obj)

    for s in traj:
        assert_data_equals(s, obj)
    traj.close()  # close file from reading, so it can be written again

    # write second batch of data
    writer = TrajectoryWriter(path, mode="append", attrs={"test": "no"})
    writer.append(obj, 5)
    writer.append(obj)
    writer.close()

    # check second batch of data
    traj = Trajectory(path)
    assert len(traj) == 4
    np.testing.assert_allclose(traj.times, [1, 2, 5, 6])
    assert_data_equals(traj[1], obj)
    assert_data_equals(traj[-1], obj)
    assert traj[1] is not traj[-1]
    assert traj.attrs["test"] == "no"

    for s in traj:
        assert_data_equals(s, obj)
    traj.close()  # close file from reading, so it can be written again

    if ext == ".zarr":
        # zarr does not support deletion of data, so we choose a new path
        path2 = tmp_path / ("file2" + ext)
    else:
        path2 = path

    # overwrite trajectory with third batch of data
    # write second batch of data
    with TrajectoryWriter(path2, mode="truncate") as writer:
        writer.append(obj, 2)
        writer.append(obj)

    # check third batch of data
    traj = Trajectory(path2)
    assert len(traj) == 2
    np.testing.assert_allclose(traj.times, [2, 3])
    assert_data_equals(traj[1], obj)
    assert_data_equals(traj[-1], obj)

    # clean up
    remove_file_or_folder(path)
    if ext == ".zarr":
        remove_file_or_folder(path2)


@pytest.mark.parametrize("obj", STORAGE_OBJECTS)
@pytest.mark.parametrize("ext", STORAGE_EXT)
def test_trajectory_writer_open_storage(obj, ext, tmp_path):
    """Test simple trajecotry writing in an externally opened storage."""
    path = tmp_path / ("file" + ext)
    storage = open_storage(path, mode="insert")
    assert not storage.closed

    # write first batch of data
    with TrajectoryWriter(storage, attrs={"test": "yes"}) as writer:
        writer.append(obj, 1)
        writer.append(obj)
    assert not storage.closed

    # check that we can't accidentally overwrite
    with pytest.raises(RuntimeError), TrajectoryWriter(storage) as writer:
        ...

    # check first batch of data
    traj = Trajectory(storage)
    assert len(traj) == 2
    np.testing.assert_allclose(traj.times, [1, 2])
    assert_data_equals(traj[1], obj)
    assert_data_equals(traj[-1], obj)

    for s in traj:
        assert_data_equals(s, obj)
    traj.close()  # close file from reading, so it can be written again
    assert not storage.closed
    storage.close()
    assert storage.closed

    # check that we can't write to a closed storage
    with pytest.raises(RuntimeError), TrajectoryWriter(storage) as writer:
        ...

    # write second batch of data
    storage = open_storage(path, mode="append")
    writer = TrajectoryWriter(storage, mode="append", attrs={"test": "no"})
    writer.append(obj, 5)
    writer.append(obj)
    writer.close()
    assert not storage.closed

    # check second batch of data
    traj = Trajectory(storage)
    assert len(traj) == 4
    np.testing.assert_allclose(traj.times, [1, 2, 5, 6])
    assert_data_equals(traj[1], obj)
    assert_data_equals(traj[-1], obj)
    assert traj[1] is not traj[-1]
    assert traj.attrs["test"] == "no"

    for s in traj:
        assert_data_equals(s, obj)
    traj.close()  # close file from reading, so it can be written again
    assert not storage.closed

    remove_file_or_folder(path)


@pytest.mark.parametrize("ext", STORAGE_EXT)
def test_trajectory_multiple_reads(ext, tmp_path):
    """Test simultaneous reading of trajecotries."""
    path = tmp_path / ("file" + ext)
    obj = np.arange(5)

    # write some data
    with TrajectoryWriter(path, attrs={"test": "yes"}) as writer:
        writer.append(obj, 1)
        writer.append(obj)

    # read the data
    t1 = Trajectory(path)
    t2 = Trajectory(path)
    assert len(t1) == len(t2) == 2
    np.testing.assert_allclose(t1.times, [1, 2])
    np.testing.assert_array_equal(t1[0], obj)
    np.testing.assert_array_equal(t1[1], obj)
    np.testing.assert_array_equal(t2[0], obj)
    np.testing.assert_array_equal(t2[1], obj)

    remove_file_or_folder(path)


@pytest.mark.parametrize("ext", STORAGE_EXT)
def test_trajectory_overwriting(ext, tmp_path):
    """Test whether zarr zip files can be overwritten."""
    path = tmp_path / ("file" + ext)
    obj = np.arange(5)

    # write some data
    with TrajectoryWriter(path, attrs={"test": "yes"}) as writer:
        writer.append(obj, 1)
        writer.append(obj)

    # try writing data without overwrite
    with pytest.raises(RuntimeError), TrajectoryWriter(path, mode="insert") as writer:
        writer.append(obj)

    # try writing data with overwrite
    with TrajectoryWriter(path, mode="truncate") as writer:
        writer.append(obj)

    remove_file_or_folder(path)


@pytest.mark.parametrize("ext", STORAGE_EXT)
def test_trajectory_reading_while_writing(ext, tmp_path):
    """Test reading while writing trajecotries."""
    path = tmp_path / ("file" + ext)
    obj = np.arange(5)

    # write some data
    with TrajectoryWriter(path, attrs={"test": "yes"}) as writer:
        writer.append(obj, 1)
        np.testing.assert_allclose(writer.times, [1])
        writer.append(obj)
        np.testing.assert_allclose(writer.times, [1, 2])

    remove_file_or_folder(path)
