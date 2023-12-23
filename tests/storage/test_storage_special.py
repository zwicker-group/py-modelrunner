"""
.. codeauthor:: David Zwicker <david.zwicker@ds.mpg.de>
"""

import json

import numpy as np
import pytest
from pde.tools.misc import skipUnlessModule

from helpers import STORAGE_OBJECTS, assert_data_equals
from modelrunner.storage import MemoryStorage, open_storage


@pytest.mark.parametrize("obj", STORAGE_OBJECTS)
def test_memory_storage(obj):
    """test MemoryStorage"""
    with open_storage(MemoryStorage()) as storage:
        storage["a"] = obj
        assert_data_equals(storage["a"], obj)

        storage.create_dynamic_array("dyn", arr=np.arange(2, dtype=float))
        storage.extend_dynamic_array("dyn", np.ones(2))
        np.testing.assert_array_equal(storage.read_array("dyn", index=0), np.ones(2))


@skipUnlessModule("h5py")
def test_hdf_storage(tmp_path):
    """test HDFStorage"""
    import h5py

    with h5py.File(tmp_path / "storage.hdf", "w") as root:
        with open_storage(root, mode="insert") as storage:
            storage["obj"] = {"info": True}
        root.attrs["test"] = 5

    with h5py.File(tmp_path / "storage.hdf", "r") as root:
        with open_storage(root, mode="read") as storage:
            assert storage["obj"] == {"info": True}
        assert root.attrs["test"] == 5


@skipUnlessModule("zarr")
def test_zarr_storage(tmp_path):
    """test ZarrStorage"""
    import zarr

    with zarr.open(tmp_path / "storage.zarr", "w") as root:
        with open_storage(root, mode="insert") as storage:
            storage["obj"] = {"info": True}
        root.attrs["test"] = 5

    with zarr.open(tmp_path / "storage.zarr", "r") as root:
        with open_storage(root, mode="read") as storage:
            assert storage["obj"] == {"info": True}
        assert root.attrs["test"] == 5


def test_json_storage(tmp_path):
    """test JSONStorage"""
    with open_storage(tmp_path / "test.json", mode="truncate") as storage:
        storage["obj"] = {"info": True}
        json_txt = storage._storage.to_text()

    with open(tmp_path / "test.json") as fp:
        assert json.load(fp) == json.loads(json_txt)


@skipUnlessModule("yaml")
def test_yaml_storage(tmp_path):
    """test YAMLStorage"""
    import yaml

    with open_storage(tmp_path / "test.yaml", mode="truncate") as storage:
        storage["obj"] = {"info": True}
        json_txt = storage._storage.to_text()

    with open(tmp_path / "test.yaml") as fp:
        assert yaml.safe_load(fp) == yaml.safe_load(json_txt)
