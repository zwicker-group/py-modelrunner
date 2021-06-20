"""
.. codeauthor:: David Zwicker <david.zwicker@ds.mpg.de>
"""

import inspect
import json
import os.path
import warnings
from typing import Any, Dict

import numpy as np

from .model import ModelBase


def contains_array(data) -> bool:
    if isinstance(data, np.ndarray):
        return True
    elif isinstance(data, dict):
        return any(contains_array(d) for d in data.values())
    elif isinstance(data, str):
        return False
    elif hasattr(data, "__iter__"):
        return any(contains_array(d) for d in data)
    else:
        return False


def write_hdf_dataset(node, data, name):
    if data is None:
        return

    if isinstance(data, np.ndarray):
        node.create_dataset(name, data=data)

    else:
        group = node.create_group(name)

        if not contains_array(data):
            # write everything as JSON encoded string
            if isinstance(data, dict):
                for key, value in data.items():
                    group.attrs[key] = json.dumps(value)
            else:
                group.attrs[name] = json.dumps(data)

        elif isinstance(data, dict):
            for key, value in data.items():
                write_hdf_dataset(group, value, key)

        else:
            for n, value in enumerate(data):
                write_hdf_dataset(group, value, str(n))


def read_hdf_data(node):
    import h5py

    if isinstance(node, h5py.Dataset):
        return np.array(node)
    else:
        # this must be a group
        data = {key: json.loads(value) for key, value in node.attrs.items()}
        for key, value in node.items():
            data[key] = read_hdf_data(value)
        return data


class MockModel(ModelBase):
    """helper class to store parameter values when the original model is not present"""

    def __init__(self, parameters: Dict[str, Any] = None):
        """
        Args:
            parameters (dict): A dictionary of parameters
        """
        self.parameters = self._parse_parameters(parameters, check_validity=False)

    def __call__(self):
        raise RuntimeError("MockModel cannot be called")


class Result:
    """describes a model (with parameters) together with its result"""

    def __init__(self, model: ModelBase, result=None):
        self.model = model
        self.result = result

    @classmethod
    def from_data(cls, model_data, result, model: ModelBase = None) -> "Result":
        if model is None:
            model_cls = MockModel
        else:
            model_cls = model if inspect.isclass(model) else model.__class__

        if not model_data:
            warnings.warn("Model data not found")
        obj = model_cls(model_data.get("parameters", {}))
        obj.name = model_data.get("name")
        obj.description = model_data.get("description")

        return cls(obj, result)

    @property
    def parameters(self) -> Dict[str, Any]:
        return self.model.parameters

    def write_to_file(self, path):
        ext = os.path.splitext(path)[1].lower()
        if ext == ".json":
            self.write_to_json(path)
        elif ext in {".h", ".h4", ".h5", ".hdf", ".hdf4", ".hdf5"}:
            self.write_to_hdf(path)
        else:
            raise ValueError(f"Unknown file format `{ext}`")

    @classmethod
    def from_json(cls, path, model: ModelBase = None):
        """load result from a JSON file"""
        with open(path, "r") as fp:
            data = json.load(fp)

        return cls.from_data(
            model_data=data.get("model", {}), result=data.get("result"), model=model
        )

    def write_to_json(self, path) -> "Result":
        """write result to JSON file"""
        with open(path, "w") as fp:
            json.dump({"model": self.model.attributes, "result": self.result}, fp)

    @classmethod
    def from_hdf(cls, path, model: ModelBase = None):
        """load result from HDF file"""
        import h5py

        with h5py.File(path, "r") as fp:
            model_data = {key: json.loads(value) for key, value in fp.attrs.items()}
            if "result" in fp:
                result = read_hdf_data(fp["result"])
            else:
                result = model_data.pop("result")
            # check for other nodes, which might not be read

        return cls.from_data(model_data=model_data, result=result, model=model)

    def write_to_hdf(self, path) -> "Result":
        """write result to HDF file"""
        import h5py

        with h5py.File(path, "w") as fp:
            # write attributes
            for key, value in self.model.attributes.items():
                fp.attrs[key] = json.dumps(value)

            # write the actual data
            write_hdf_dataset(fp, self.result, "result")
