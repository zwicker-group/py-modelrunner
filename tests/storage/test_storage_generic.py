"""
.. codeauthor:: David Zwicker <david.zwicker@ds.mpg.de>
"""

import numpy as np
import pytest

from helpers import STORAGE_EXT, STORAGE_OBJECTS, assert_data_equals
from modelrunner.storage import AccessError, open_storage

OBJ = {"a": 1, "b": np.arange(5)}
ARRAY_EXAMPLES = [
    np.arange(4).astype(np.uint8),
    np.array([{"a": 1}], dtype=object),
    np.array([(1.0, 2), (3.0, 4)], dtype=[("x", "f8"), ("y", "i8")]).view(np.recarray),
]


STORAGE_CLASSES = {
    "json": "JSONStorage",
    "yaml": "YAMLStorage",
    "hdf": "HDFStorage",
    "zarr": "ZarrStorage",
    "zip": "ZarrStorage",
    "": "ZarrStorage",
    "sqldb": "ZarrStorage",
}


@pytest.mark.parametrize("ext", STORAGE_EXT)
def test_storage_class(ext, tmp_path):
    """Test whether open_storage uses the correct Storage class."""
    storage = open_storage(tmp_path / f"file{ext}", mode="truncate")
    assert storage._storage.__class__.__name__ == STORAGE_CLASSES[ext[1:]]

    storage["item"] = 1
    storage["a/b/c"] = 2
    assert storage.is_group("a/b")
    assert storage["a/b"]["c"] == 2

    with pytest.raises(TypeError):
        storage["item/test"] = 1  # cannot make group when there is an item
    with pytest.raises(TypeError):
        storage.create_group("item/test")  # cannot make group when there is an item


@pytest.mark.parametrize("arr", ARRAY_EXAMPLES)
@pytest.mark.parametrize("ext", STORAGE_EXT)
def test_storage_persistence(arr, ext, tmp_path):
    """Test generic properties of storages."""
    # write to storage
    with open_storage(tmp_path / f"file{ext}", mode="truncate") as storage:
        storage.create_group("empty")
        storage.write_array("group/test/arr", arr, attrs={"array": True})
        storage.create_dynamic_array("dyn", arr=arr)
        storage.extend_dynamic_array("dyn", arr)
        storage.extend_dynamic_array("dyn", arr)
        storage.write_object("obj", OBJ)

        assert isinstance(str(storage), str)  # test whether __repr__ works somewhat
        assert not storage.closed

    # read from storage
    with open_storage(tmp_path / f"file{ext}", mode="read") as storage:
        assert storage.is_group("empty")
        assert len(storage["empty"].keys()) == 0

        arr_read = storage.read_array("group/test/arr")
        assert arr.__class__ is arr_read.__class__
        np.testing.assert_array_equal(arr_read, arr)
        assert storage.read_attrs("group/test/arr") == {"array": True}

        out = storage.read_array("dyn", index=0)
        np.testing.assert_array_equal(out, arr)
        assert out.__class__ is arr.__class__
        out = storage.read_array("dyn", index=1, out=np.empty_like(arr))
        np.testing.assert_array_equal(out, arr)
        assert out.__class__ is arr.__class__
        assert_data_equals(storage.read_object("obj"), OBJ)

        with pytest.raises(RuntimeError):
            storage.read_object("group/test/arr")

        assert len(list(storage.keys())) == 4
        assert len(list(storage.items())) == 4
        assert len(list(storage)) == 4
        assert not storage.closed


@pytest.mark.parametrize("ext", STORAGE_EXT)
def test_storage_read(ext, tmp_path):
    """Test read mode."""
    # create empty file
    with open_storage(tmp_path / f"file{ext}", mode="truncate"):
        ...

    with open_storage(tmp_path / f"file{ext}", mode="read") as storage:
        with pytest.raises(AccessError):
            storage.write_attrs(None, {"a": 1})
        with pytest.raises(AccessError):
            storage.create_group("a")
        with pytest.raises(AccessError):
            storage._storage.ensure_group(["a"])
        with pytest.raises(AccessError):
            storage.write_array("arr", np.arange(5))
        with pytest.raises(AccessError):
            storage.create_dynamic_array("dyn", arr=np.arange(5))
        with pytest.raises(AccessError):
            storage.extend_dynamic_array("dyn", np.arange(5))
        with pytest.raises(AccessError):
            storage.write_object("obj", OBJ)


@pytest.mark.parametrize("ext", STORAGE_EXT)
def test_storage_exclusive(ext, tmp_path):
    """Test exclusive mode."""
    with open_storage(tmp_path / f"file{ext}", mode="exclusive") as storage:
        storage.write_array("arr", np.arange(5))
        storage.create_dynamic_array("dyn", arr=np.arange(2, dtype=float))
        storage.extend_dynamic_array("dyn", np.full(2, 1.0))
        storage.extend_dynamic_array("dyn", np.full(2, 2.0))
        storage.write_object("obj", OBJ)

        storage.write_attrs(None, {"a": 1})
        storage.create_group("a")
        storage.write_array("b", np.arange(5))

        np.testing.assert_array_equal(storage.read_array("arr"), np.arange(5))

    with open_storage(tmp_path / f"file{ext}", mode="read") as storage:
        np.testing.assert_array_equal(storage.read_array("arr"), np.arange(5))
        np.testing.assert_array_equal(storage.read_array("dyn", index=0), np.ones(2))
        assert_data_equals(storage.read_object("obj"), OBJ)

    with pytest.raises(FileExistsError):
        open_storage(tmp_path / f"file{ext}", mode="exclusive")


@pytest.mark.parametrize("ext", STORAGE_EXT)
def test_storage_insert(ext, tmp_path):
    """Test insert mode."""
    with open_storage(tmp_path / f"file{ext}", mode="truncate") as storage:
        storage.create_group("a")
        storage.write_array("arr1", np.arange(3))
        storage.create_dynamic_array("dyn", arr=np.arange(2))
        storage.write_object("obj", OBJ)

    with open_storage(tmp_path / f"file{ext}", mode="insert") as storage:
        storage.write_attrs(None, {"a": 1})
        storage.create_group("b")
        storage.write_array("arr2", np.arange(5))
        with pytest.raises((AccessError, RuntimeError)):
            storage.write_array("arr2", np.zeros(5))
        storage.extend_dynamic_array("dyn", np.arange(2))
        with pytest.raises((AccessError, RuntimeError)):
            storage.write_object("obj", OBJ)

        # test reading
        assert storage.is_group("a")
        np.testing.assert_array_equal(storage.read_array("arr1"), np.arange(3))

    with open_storage(tmp_path / f"file{ext}", mode="read") as storage:
        assert storage.is_group("a")
        assert storage.is_group("b")
        np.testing.assert_array_equal(storage.read_array("arr1"), np.arange(3))
        np.testing.assert_array_equal(storage.read_array("arr2"), np.arange(5))
        np.testing.assert_array_equal(storage.read_array("dyn", index=0), np.arange(2))
        assert_data_equals(storage.read_object("obj"), OBJ)


@pytest.mark.parametrize("ext", STORAGE_EXT)
def test_storage_overwrite(ext, tmp_path):
    """Test overwrite mode."""
    with open_storage(tmp_path / f"file{ext}", mode="truncate") as storage:
        storage.write_array("arr", np.arange(5))
        storage.create_dynamic_array("dyn", arr=np.arange(2, dtype=float))
        storage.extend_dynamic_array("dyn", np.arange(2, dtype=float))
        storage.write_object("obj", {})

    with open_storage(tmp_path / f"file{ext}", mode="overwrite") as storage:
        storage.write_attrs(None, {"a": 1})
        with pytest.raises(AccessError):
            storage.create_group("a")

        np.testing.assert_array_equal(storage.read_array("arr"), np.arange(5))
        if storage._storage.can_update:
            storage.write_array("arr", np.zeros(5))
        with pytest.raises(AccessError):
            storage.write_array("b", np.arange(5))

        with pytest.raises(RuntimeError):
            storage.create_dynamic_array("dyn", arr=np.arange(3))
        storage.extend_dynamic_array("dyn", np.ones(2, dtype=float))

        if storage._storage.can_update:
            storage.write_object("obj", OBJ)

    with open_storage(tmp_path / f"file{ext}", mode="read") as storage:
        if storage._storage.can_update:
            np.testing.assert_array_equal(storage.read_array("arr"), np.zeros(5))
            assert_data_equals(storage.read_object("obj"), OBJ)
        else:
            np.testing.assert_array_equal(storage.read_array("arr"), np.arange(5))
            assert_data_equals(storage.read_object("obj"), {})
        np.testing.assert_array_equal(storage.read_array("dyn", index=1), np.ones(2))


@pytest.mark.parametrize("ext", STORAGE_EXT)
def test_storage_full(ext, tmp_path):
    """Test full mode."""
    with open_storage(tmp_path / f"file{ext}", mode="truncate") as storage:
        storage.write_array("arr", np.arange(5))
        storage.create_dynamic_array("dyn", arr=np.arange(2, dtype=float))
        storage.extend_dynamic_array("dyn", np.full(2, 1.0))
        storage.write_object("obj", {})

    with open_storage(tmp_path / f"file{ext}", mode="full") as storage:
        storage.write_attrs(None, {"a": 1})
        storage.create_group("a")

        np.testing.assert_array_equal(storage.read_array("arr"), np.arange(5))
        if storage._storage.can_update:
            storage.write_array("arr", np.zeros(5))
        storage.write_array("b", np.arange(5))
        storage.extend_dynamic_array("dyn", np.full(2, 2.0))
        if storage._storage.can_update:
            storage.write_object("obj", OBJ)

    with open_storage(tmp_path / f"file{ext}", mode="read") as storage:
        if storage._storage.can_update:
            np.testing.assert_array_equal(storage.read_array("arr"), np.zeros(5))
            assert_data_equals(storage.read_object("obj"), OBJ)
        else:
            np.testing.assert_array_equal(storage.read_array("arr"), np.arange(5))
            assert_data_equals(storage.read_object("obj"), {})
        np.testing.assert_array_equal(storage.read_array("dyn", index=0), np.ones(2))


@pytest.mark.parametrize("ext", STORAGE_EXT)
def test_storage_truncate(ext, tmp_path):
    """Test truncate mode."""
    with open_storage(tmp_path / f"file{ext}", mode="truncate") as storage:
        storage.create_group("a")
        storage.write_attrs(None, {"a": 1})
        storage.write_array("arr", np.arange(5))
        storage.create_dynamic_array("dyn", arr=np.arange(2, dtype=float))
        storage.extend_dynamic_array("dyn", np.full(2, 2.0))
        storage.write_object("obj", {})

    with open_storage(tmp_path / f"file{ext}", mode="truncate") as storage:
        storage.create_group("c/b/a")
        storage.write_attrs(None, {"test": 1})
        storage.write_array("arr2", np.arange(5))
        storage.create_dynamic_array("dyn", arr=np.arange(2, dtype=float))
        storage.extend_dynamic_array("dyn", np.full(2, 1.0))
        storage.write_object("obj", OBJ)

    with open_storage(tmp_path / f"file{ext}", mode="read") as storage:
        assert "a" not in storage
        assert storage.read_attrs() == {"test": 1}
        assert "arr" not in storage

        assert "c/b/a" in storage
        np.testing.assert_array_equal(storage.read_array("arr2"), np.arange(5))
        np.testing.assert_array_equal(storage.read_array("dyn", index=0), np.ones(2))
        assert_data_equals(storage.read_object("obj"), OBJ)


@pytest.mark.parametrize("ext", STORAGE_EXT)
def test_appending_to_fixed_array(ext, tmp_path):
    """Test appending an array to a non-dynamic array."""
    with open_storage(tmp_path / f"file{ext}", mode="truncate") as storage:
        storage.write_array("a1", np.arange(4))
        with pytest.raises(RuntimeError):
            storage.extend_dynamic_array("a1", np.zeros(4))

        storage.write_array("a2", np.zeros((2, 4)))
        with pytest.raises(RuntimeError):
            storage.extend_dynamic_array("a2", np.zeros(4))


@pytest.mark.parametrize("obj", STORAGE_OBJECTS)
@pytest.mark.parametrize("ext", STORAGE_EXT)
def test_arbitrary_objects(obj, ext, tmp_path):
    """Test appending arbitrary objects."""
    with open_storage(tmp_path / f"file{ext}", mode="truncate") as storage:
        storage["obj"] = obj

    with open_storage(tmp_path / f"file{ext}", mode="read") as storage:
        assert_data_equals(storage["obj"], obj)


@pytest.mark.parametrize("ext", STORAGE_EXT)
def test_storage_copy_array(ext, tmp_path):
    """Test whether storages make a copy, i.e., that not only a view is stored."""
    obj = np.arange(5)
    with open_storage(tmp_path / f"file{ext}", mode="truncate") as storage:
        storage["obj"] = obj
        assert storage["obj"] is not obj
        np.testing.assert_array_equal(storage["obj"], obj)
        obj[0] = -42
        assert not np.array_equal(storage["obj"], obj)


@pytest.mark.parametrize("ext", STORAGE_EXT)
def test_storage_copy_dict(ext, tmp_path):
    """Test whether storages make a copy, i.e., that not only a view is stored."""
    obj = {"a": 1, "b": 2}
    with open_storage(tmp_path / f"file{ext}", mode="truncate") as storage:
        storage["obj"] = obj
        assert storage["obj"] is not obj
        assert storage["obj"] == obj
        obj["c"] = 1
        assert storage["obj"] != obj


@pytest.mark.parametrize("ext", STORAGE_EXT)
def test_storage_close(ext, tmp_path):
    """Test closing of storages."""
    writer = open_storage(tmp_path / f"file{ext}", mode="truncate")
    assert not writer.closed
    writer["obj"] = 1
    writer.close()
    assert writer.closed

    with pytest.raises(RuntimeError):
        writer["obj"]
    with pytest.raises(RuntimeError):
        assert "obj" in writer

    reader = open_storage(tmp_path / f"file{ext}", mode="read")
    assert not reader.closed
    assert reader["obj"] == 1
    reader.close()
    assert reader.closed
