"""
.. codeauthor:: David Zwicker <david.zwicker@ds.mpg.de>
"""

import inspect
import json
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


class ModelResult:
    """describes a model (with parameters) together with its result"""

    def __init__(self, model: ModelBase, results=None):
        self.model = model
        self.results = results

    @classmethod
    def from_data(cls, model_data, results, model: ModelBase = None) -> "ModelResult":
        if model is None:
            model_cls = MockModel
        else:
            model_cls = model if inspect.isclass(model) else model.__class__

        if not model_data:
            warnings.warn("Model data not found")
        obj = model_cls(model_data.get("parameters", {}))
        obj.name = model_data.get("name")
        obj.description = model_data.get("description")

        return cls(obj, results)

    @property
    def parameters(self) -> Dict[str, Any]:
        return self.model.parameters

    @classmethod
    def from_json(cls, path, model: ModelBase = None):
        """load result from a JSON file"""
        with open(path, "r") as fp:
            data = json.load(fp)

        return cls.from_data(
            model_data=data.get("model", {}), results=data.get("results"), model=model
        )

    def write_to_json(self, path) -> "ModelResult":
        """write result to JSON file"""
        with open(path, "w") as fp:
            json.dump({"model": self.model.attributes, "results": self.results}, fp)

    @classmethod
    def from_hdf(cls, path, model: ModelBase = None):
        """load result from HDF file"""
        import h5py

        with h5py.File(path, "r") as fp:
            model_data = {key: json.loads(value) for key, value in fp.attrs.items()}
            if "results" in fp:
                results = read_hdf_data(fp["results"])
            else:
                results = model_data.pop("results")
            # check for other nodes, which might not be read

        return cls.from_data(model_data=model_data, results=results, model=model)

    def write_to_hdf(self, path) -> "ModelResult":
        """write result to HDF file"""
        import h5py

        with h5py.File(path, "w") as fp:
            # write attributes
            for key, value in self.model.attributes.items():
                fp.attrs[key] = json.dumps(value)

            # write the actual data
            write_hdf_dataset(fp, self.results, "results")
