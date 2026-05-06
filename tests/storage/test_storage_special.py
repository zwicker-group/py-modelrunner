"""
.. codeauthor:: David Zwicker <david.zwicker@ds.mpg.de>
"""

import json

import numpy as np
import pytest

from helpers import STORAGE_OBJECTS, assert_data_equals, module_available
from modelrunner.storage import MemoryStorage, open_storage


@pytest.mark.parametrize("obj", STORAGE_OBJECTS)
def test_memory_storage(obj):
    """Test MemoryStorage."""
    with open_storage(MemoryStorage()) as storage:
        storage["a"] = obj
        assert_data_equals(storage["a"], obj)

        storage.create_dynamic_array("dyn", arr=np.arange(2, dtype=float))
        storage.extend_dynamic_array("dyn", np.ones(2))
        np.testing.assert_array_equal(storage.read_array("dyn", index=0), np.ones(2))


@pytest.mark.skipif(not module_available("h5py"), reason="requires `h5py` module")
def test_hdf_storage(tmp_path):
    """Test HDFStorage."""
    import h5py

    with h5py.File(tmp_path / "storage.hdf", "w") as root:
        with open_storage(root, mode="insert") as storage:
            storage["obj"] = {"info": True}
        root.attrs["test"] = 5

    with h5py.File(tmp_path / "storage.hdf", "r") as root:
        with open_storage(root, mode="read") as storage:
            assert storage["obj"] == {"info": True}
        assert root.attrs["test"] == 5


@pytest.mark.skipif(not module_available("zarr"), reason="requires `zarr` module")
def test_zarr_storage(tmp_path):
    """Test ZarrStorage."""
    import zarr

    if hasattr(zarr, "open_group"):
        open_group = zarr.open_group
    else:  # pragma: no cover
        import zarr.convenience

        open_group = zarr.convenience.open_group

    if hasattr(zarr.storage, "LocalStore"):
        store = zarr.storage.LocalStore(tmp_path / "storage.zarr")
    else:
        store = zarr.storage.DirectoryStore(str(tmp_path / "storage.zarr"))

    root = open_group(store=store, mode="w")
    with open_storage(root, mode="insert") as storage:
        storage["obj"] = {"info": True}
    root.attrs["test"] = 5

    root = open_group(store=store, mode="r")
    with open_storage(root, mode="read") as storage:
        assert storage["obj"] == {"info": True}
    assert root.attrs["test"] == 5


def test_json_storage(tmp_path):
    """Test JSONStorage."""
    with open_storage(tmp_path / "test.json", mode="truncate") as storage:
        storage["obj"] = {"info": True}
        json_txt = storage._storage.to_text()

    with (tmp_path / "test.json").open() as fp:
        assert json.load(fp) == json.loads(json_txt)


@pytest.mark.skipif(not module_available("yaml"), reason="requires `yaml` module")
def test_yaml_storage(tmp_path):
    """Test YAMLStorage."""
    import yaml

    with open_storage(tmp_path / "test.yaml", mode="truncate") as storage:
        storage["obj"] = {"info": True}
        json_txt = storage._storage.to_text()

    with (tmp_path / "test.yaml").open() as fp:
        assert yaml.safe_load(fp) == yaml.safe_load(json_txt)
