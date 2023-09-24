"""
.. codeauthor:: David Zwicker <david.zwicker@ds.mpg.de>
"""

import numpy as np
import pytest

from helpers import storage_extensions
from modelrunner.storage import AccessError, open_storage

ARRAY_EXAMPLES = [
    np.arange(4).astype(np.uint8),
    np.array([{"a": 1}], dtype=object),
    np.array([(1.0, 2), (3.0, 4)], dtype=[("x", "f8"), ("y", "i8")]).view(np.recarray),
]
STORAGE_EXT = storage_extensions(incl_folder=True, dot=False)


@pytest.mark.parametrize("arr", ARRAY_EXAMPLES)
@pytest.mark.parametrize("ext", STORAGE_EXT)
def test_storage_persistence(arr, ext, tmp_path):
    """test generic properties of storages"""
    # write to storage
    with open_storage(tmp_path / f"file.{ext}", mode="truncate") as storage:
        storage.create_group("empty")
        storage.write_array("group/test/arr", arr)
        storage.create_dynamic_array("dyn", arr=arr)
        storage.extend_dynamic_array("dyn", arr)

    # read from storage
    with open_storage(tmp_path / f"file.{ext}", mode="readonly") as storage:
        assert storage.is_group("empty") and len(storage["empty"].keys()) == 0
        np.testing.assert_array_equal(storage.read_array("group/test/arr"), arr)
        np.testing.assert_array_equal(storage.read_array("dyn", index=0), arr)


@pytest.mark.parametrize("ext", STORAGE_EXT)
def test_storage_readonly(ext, tmp_path):
    """test readonly mode"""
    # create empty file
    with open_storage(tmp_path / f"file.{ext}", mode="truncate"):
        ...

    with open_storage(tmp_path / f"file.{ext}", mode="readonly") as storage:
        with pytest.raises(AccessError):
            storage.write_attrs(None, {"a": 1})
        with pytest.raises(AccessError):
            storage.create_group("a")
        with pytest.raises(AccessError):
            storage.write_array("arr", np.arange(5))
        with pytest.raises(AccessError):
            storage.create_dynamic_array("dyn", arr=np.arange(5))
        with pytest.raises(AccessError):
            storage.extend_dynamic_array("dyn", np.arange(5))


@pytest.mark.parametrize("ext", STORAGE_EXT)
def test_storage_insert(ext, tmp_path):
    """test insert mode"""
    with open_storage(tmp_path / f"file.{ext}", mode="truncate") as storage:
        storage.create_group("a")
        storage.write_array("arr1", np.arange(3))
        storage.create_dynamic_array("dyn", arr=np.arange(2))

    with open_storage(tmp_path / f"file.{ext}", mode="insert") as storage:
        storage.write_attrs(None, {"a": 1})
        storage.create_group("b")
        storage.write_array("arr2", np.arange(5))
        with pytest.raises(AccessError):
            storage.write_array("arr2", np.zeros(5))
        storage.extend_dynamic_array("dyn", np.arange(2))

        # test reading
        assert storage.is_group("a")
        np.testing.assert_array_equal(storage.read_array("arr1"), np.arange(3))

    with open_storage(tmp_path / f"file.{ext}", mode="readonly") as storage:
        assert storage.is_group("a")
        assert storage.is_group("b")
        np.testing.assert_array_equal(storage.read_array("arr1"), np.arange(3))
        np.testing.assert_array_equal(storage.read_array("arr2"), np.arange(5))
        np.testing.assert_array_equal(storage.read_array("dyn", index=0), np.arange(2))


@pytest.mark.parametrize("ext", STORAGE_EXT)
def test_storage_overwrite(ext, tmp_path):
    """test overwrite mode"""
    with open_storage(tmp_path / f"file.{ext}", mode="truncate") as storage:
        storage.write_array("arr", np.arange(5))
        storage.create_dynamic_array("dyn", arr=np.arange(2, dtype=float))
        storage.extend_dynamic_array("dyn", np.arange(2, dtype=float))

    with open_storage(tmp_path / f"file.{ext}", mode="overwrite") as storage:
        storage.write_attrs(None, {"a": 1})
        with pytest.raises(AccessError):
            storage.create_group("a")

        np.testing.assert_array_equal(storage.read_array("arr"), np.arange(5))
        storage.write_array("arr", np.zeros(5))
        with pytest.raises(AccessError):
            storage.write_array("b", np.arange(5))

        with pytest.raises(RuntimeError):
            storage.create_dynamic_array("dyn", arr=np.arange(3))
        storage.extend_dynamic_array("dyn", np.ones(2, dtype=float))

    with open_storage(tmp_path / f"file.{ext}", mode="readonly") as storage:
        np.testing.assert_array_equal(storage.read_array("arr"), np.zeros(5))
        np.testing.assert_array_equal(storage.read_array("dyn", index=1), np.ones(2))


@pytest.mark.parametrize("ext", STORAGE_EXT)
def test_storage_full(ext, tmp_path):
    """test full mode"""
    with open_storage(tmp_path / f"file.{ext}", mode="truncate") as storage:
        storage.write_array("arr", np.arange(5))
        storage.create_dynamic_array("dyn", arr=np.arange(2, dtype=float))
        storage.extend_dynamic_array("dyn", np.full(2, 1.0))

    with open_storage(tmp_path / f"file.{ext}", mode="full") as storage:
        storage.write_attrs(None, {"a": 1})
        storage.create_group("a")

        np.testing.assert_array_equal(storage.read_array("arr"), np.arange(5))
        storage.write_array("arr", np.zeros(5))
        storage.write_array("b", np.arange(5))
        storage.extend_dynamic_array("dyn", np.full(2, 2.0))

    with open_storage(tmp_path / f"file.{ext}", mode="readonly") as storage:
        np.testing.assert_array_equal(storage.read_array("arr"), np.zeros(5))
        np.testing.assert_array_equal(storage.read_array("dyn", index=0), np.ones(2))


@pytest.mark.parametrize("ext", STORAGE_EXT)
def test_storage_truncate(ext, tmp_path):
    """test truncate mode"""
    with open_storage(tmp_path / f"file.{ext}", mode="truncate") as storage:
        storage.create_group("a")
        storage.write_attrs(None, {"a": 1})
        storage.write_array("arr", np.arange(5))
        storage.create_dynamic_array("dyn", arr=np.arange(2, dtype=float))
        storage.extend_dynamic_array("dyn", np.full(2, 2.0))

    with open_storage(tmp_path / f"file.{ext}", mode="truncate") as storage:
        storage.create_group("c/b/a")
        storage.write_attrs(None, {"test": 1})
        storage.write_array("arr2", np.arange(5))
        storage.create_dynamic_array("dyn", arr=np.arange(2, dtype=float))
        storage.extend_dynamic_array("dyn", np.full(2, 1.0))

    with open_storage(tmp_path / f"file.{ext}", mode="readonly") as storage:
        assert "a" not in storage
        assert storage.read_attrs() == {"test": 1}
        assert "arr" not in storage

        assert "c/b/a" in storage
        np.testing.assert_array_equal(storage.read_array("arr2"), np.arange(5))
        np.testing.assert_array_equal(storage.read_array("dyn", index=0), np.ones(2))


@pytest.mark.parametrize("ext", STORAGE_EXT)
def test_appending_to_fixed_array(ext, tmp_path):
    """test appending an array to a non-dynamic array"""
    with open_storage(tmp_path / f"file.{ext}", mode="truncate") as storage:
        storage.write_array("a1", np.arange(4))
        with pytest.raises(RuntimeError):
            storage.extend_dynamic_array("a1", np.zeros(4))

        storage.write_array("a2", np.zeros((2, 4)))
        with pytest.raises(RuntimeError):
            storage.extend_dynamic_array("a2", np.zeros(4))
