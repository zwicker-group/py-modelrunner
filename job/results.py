"""
.. codeauthor:: David Zwicker <david.zwicker@ds.mpg.de>
"""

import inspect
import json
import os.path
import warnings
from pathlib import Path
from typing import Any, Dict, Type

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

    def __init__(self, model: ModelBase, result=None, name: str = None):
        self.model = model
        self.result = result
        self.name = name

    @classmethod
    def from_data(
        cls, model_data, result, model: ModelBase = None, name: str = None
    ) -> "Result":
        if model is None:
            model_cls: Type[ModelBase] = MockModel
        else:
            model_cls = model if inspect.isclass(model) else model.__class__  # type: ignore

        if not model_data:
            warnings.warn("Model data not found")
        obj = model_cls(model_data.get("parameters", {}))
        obj.name = model_data.get("name")
        obj.description = model_data.get("description")

        return cls(obj, result, name)

    @property
    def parameters(self) -> Dict[str, Any]:
        return self.model.parameters

    @classmethod
    def from_file(cls, path, model: ModelBase = None):
        ext = os.path.splitext(path)[1].lower()
        if ext == ".json":
            return cls.from_json(path, model)
        elif ext in {".h", ".h4", ".h5", ".hdf", ".hdf4", ".hdf5"}:
            return cls.from_hdf(path, model)
        else:
            raise ValueError(f"Unknown file format `{ext}`")

    def write_to_file(self, path):
        ext = os.path.splitext(path)[1].lower()
        if ext == ".json":
            self.write_to_json(path)
        elif ext in {".h", ".h4", ".h5", ".hdf", ".hdf4", ".hdf5"}:
            self.write_to_hdf(path)
        else:
            raise ValueError(f"Unknown file format `{ext}`")

    @classmethod
    def from_json(cls, path, model: ModelBase = None) -> "Result":
        """load result from a JSON file"""

        with open(path, "r") as fp:
            data = json.load(fp)

        return cls.from_data(
            model_data=data.get("model", {}),
            result=data.get("result"),
            model=model,
            name=Path(path).with_suffix("").stem,
        )

    def write_to_json(self, path) -> None:
        """write result to JSON file"""
        with open(path, "w") as fp:
            json.dump({"model": self.model.attributes, "result": self.result}, fp)

    @classmethod
    def from_hdf(cls, path, model: ModelBase = None) -> "Result":
        """load result from HDF file"""
        import h5py

        with h5py.File(path, "r") as fp:
            model_data = {key: json.loads(value) for key, value in fp.attrs.items()}
            if "result" in fp:
                result = read_hdf_data(fp["result"])
            else:
                result = model_data.pop("result")
            # check for other nodes, which might not be read

        return cls.from_data(
            model_data=model_data,
            result=result,
            model=model,
            name=Path(path).with_suffix("").stem,
        )

    def write_to_hdf(self, path) -> None:
        """write result to HDF file"""
        import h5py

        with h5py.File(path, "w") as fp:
            # write attributes
            for key, value in self.model.attributes.items():
                fp.attrs[key] = json.dumps(value)

            # write the actual data
            write_hdf_dataset(fp, self.result, "result")


class ResultCollection(list):
    """represents a collection of results"""

    @classmethod
    def from_folder(cls, folder, pattern="*.*", model: ModelBase = None):
        """create results collection from a folder"""
        folder = Path(folder)
        assert folder.is_dir()

        results = [
            Result.from_file(path, model)
            for path in folder.glob(pattern)
            if path.is_file()
        ]
        return cls(results)

    @property
    def homogeneous_model(self) -> bool:
        """flag determining whether all results are from the same model"""
        if len(self) < 2:
            return True
        name = self[0].model.name
        keys = self[0].model.parameters.keys()
        for result in self:
            if result.model.name != name or result.model.parameters.keys() != keys:
                return False
        return True

    @property
    def dataframe(self):
        """create a pandas Dataframe summarizing the data"""
        import pandas as pd

        assert self.homogeneous_model

        def get_data(result):
            """helper function to extract the data"""
            data = {"name": result.name}
            data.update(result.parameters)
            if np.isscalar(result.result):
                data["result"] = result.result
            elif isinstance(result.result, dict):
                for key, value in result.result:
                    if np.isscalar(value):
                        data[key] = value
            else:
                raise RuntimeError("Do not know how to interpret result")
            return data

        return pd.DataFrame([get_data(result) for result in self])
