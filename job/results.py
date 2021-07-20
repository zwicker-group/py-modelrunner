"""
.. codeauthor:: David Zwicker <david.zwicker@ds.mpg.de>
"""

import collections
import inspect
import json
import os.path
import warnings
from pathlib import Path
from typing import Any, Dict, List, Set, Type

import numpy as np

from .model import ModelBase


def contains_array(data) -> bool:
    """checks whether data contains a numpy array"""
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


def json_encoder(data):
    """helper function for encoding python data in JSON"""
    if isinstance(data, np.generic):
        return data.item()


def write_hdf_dataset(node, data, name: str) -> None:
    """writes data to an HDF node

    Args:
        node: the HDF node
        data: the data to be written
        name (str): name of the data in case a new dataset or group is created
    """
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
                    group.attrs[key] = json.dumps(value, default=json_encoder)
            else:
                group.attrs[name] = json.dumps(data, default=json_encoder)

        elif isinstance(data, dict):
            for key, value in data.items():
                write_hdf_dataset(group, value, key)

        else:
            for n, value in enumerate(data):
                write_hdf_dataset(group, value, str(n))


def read_hdf_data(node):
    """read structured data written with :func:`write_hdf_dataset` from an HDF node"""
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
        raise RuntimeError(f"{self.__class__.__name__} cannot be called")

    def __repr__(self):
        return f"{self.__class__.__name__}({self.parameters})"


class Result:
    """describes a model (with parameters) together with its result"""

    def __init__(self, model: ModelBase, result, info: Dict[str, Any] = None):
        """
        Args:
            model (:class:`ModelBase`): The model from which the result was obtained
            result: The actual result
            info (dict): Additional information for this result
        """
        self.model = model
        self.result = result
        self.info = info

    @classmethod
    def from_data(
        cls,
        model_data: Dict[str, Any],
        result,
        model: ModelBase = None,
        info: Dict[str, Any] = None,
    ) -> "Result":
        """create result from data

        Args:
            model_data (dict): The data identifying the model
            result: The actual result
            model (:class:`ModelBase`): The model from which the result was obtained
            info (dict): Additional information for this result
        """
        if model is None:
            model_cls: Type[ModelBase] = MockModel
        else:
            model_cls = model if inspect.isclass(model) else model.__class__  # type: ignore

        if not model_data:
            warnings.warn("Model data not found")
        obj = model_cls(model_data.get("parameters", {}))
        obj.name = model_data.get("name")
        obj.description = model_data.get("description")

        return cls(obj, result, info)

    @property
    def parameters(self) -> Dict[str, Any]:
        return self.model.parameters

    @classmethod
    def from_file(cls, path, model: ModelBase = None):
        """read result from file

        Args:
            path (str or :class:`~pathlib.Path`): The path to the file
            model (:class:`ModelBase`): The model from which the result was obtained
        """
        ext = os.path.splitext(path)[1].lower()
        if ext == ".json":
            return cls.from_json(path, model)
        elif ext in {".yml", ".yaml"}:
            return cls.from_yaml(path, model)
        elif ext in {".h5", ".hdf", ".hdf5"}:
            return cls.from_hdf(path, model)
        else:
            raise ValueError(f"Unknown file format `{ext}`")

    def write_to_file(self, path):
        """write result to a file

        Args:
            path (str or :class:`~pathlib.Path`): The path to the file
        """
        ext = os.path.splitext(path)[1].lower()
        if ext == ".json":
            self.write_to_json(path)
        elif ext in {".yml", ".yaml"}:
            self.write_to_yaml(path)
        elif ext in {".h5", ".hdf", ".hdf5"}:
            self.write_to_hdf(path)
        else:
            raise ValueError(f"Unknown file format `{ext}`")

    @classmethod
    def from_json(cls, path, model: ModelBase = None) -> "Result":
        """read result from a JSON file

        Args:
            path (str or :class:`~pathlib.Path`): The path to the file
            model (:class:`ModelBase`): The model from which the result was obtained
        """
        with open(path, "r") as fp:
            data = json.load(fp)

        info = data.get("info", {})
        info.setdefault("name", Path(path).with_suffix("").stem)

        return cls.from_data(
            model_data=data.get("model", {}),
            result=data.get("result"),
            model=model,
            info=info,
        )

    def write_to_json(self, path) -> None:
        """write result to JSON file

        Args:
            path (str or :class:`~pathlib.Path`): The path to the file
        """
        data = {"model": self.model.attributes, "result": self.result}
        if self.info:
            data["info"] = self.info

        with open(path, "w") as fp:
            json.dump(data, fp, default=json_encoder)

    @classmethod
    def from_yaml(cls, path, model: ModelBase = None) -> "Result":
        """read result from a YAML file

        Args:
            path (str or :class:`~pathlib.Path`): The path to the file
            model (:class:`ModelBase`): The model from which the result was obtained
        """
        import yaml

        with open(path, "r") as fp:
            data = yaml.safe_load(fp)

        info = data.get("info", {})
        info.setdefault("name", Path(path).with_suffix("").stem)

        return cls.from_data(
            model_data=data.get("model", {}),
            result=data.get("result"),
            model=model,
            info=info,
        )

    def write_to_yaml(self, path) -> None:
        """write result to YAML file

        Args:
            path (str or :class:`~pathlib.Path`): The path to the file
        """
        import yaml

        data = {"model": self.model.attributes, "result": self.result}
        if self.info:
            data["info"] = self.info

        with open(path, "w") as fp:
            yaml.dump(data, fp)

    @classmethod
    def from_hdf(cls, path, model: ModelBase = None) -> "Result":
        """read result from a HDf file

        Args:
            path (str or :class:`~pathlib.Path`): The path to the file
            model (:class:`ModelBase`): The model from which the result was obtained
        """
        import h5py

        with h5py.File(path, "r") as fp:
            model_data = {key: json.loads(value) for key, value in fp.attrs.items()}
            if "result" in fp:
                result = read_hdf_data(fp["result"])
            else:
                result = model_data.pop("result")
            # check for other nodes, which might not be read

        info = model_data.pop("__info__") if "__info__" in model_data else {}
        info.setdefault("name", Path(path).with_suffix("").stem)

        return cls.from_data(
            model_data=model_data, result=result, model=model, info=info
        )

    def write_to_hdf(self, path) -> None:
        """write result to HDF file

        Args:
            path (str or :class:`~pathlib.Path`): The path to the file
        """
        import h5py

        with h5py.File(path, "w") as fp:
            # write attributes
            for key, value in self.model.attributes.items():
                fp.attrs[key] = json.dumps(value, default=json_encoder)

            if self.info:
                fp.attrs["__info__"] = json.dumps(self.info, default=json_encoder)

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
    def same_model(self) -> bool:
        """bool: flag determining whether all results are from the same model"""
        if len(self) < 2:
            return True
        model_cls = self[0].model.__class__
        keys = self[0].model.parameters.keys()

        return all(
            res.model.__class__ == model_cls and res.model.parameters.keys() == keys
            for res in self
        )

    @property
    def parameters(self) -> Dict[str, Set[Any]]:
        """dict: the parameter values in this result collection"""
        params = collections.defaultdict(set)
        for result in self:
            for k, v in result.model.parameters.items():
                params[k].add(v)
        return dict(params)

    @property
    def constant_parameters(self) -> Dict[str, Any]:
        """dict: the parameters that are constant in this result collection"""
        return {
            k: next(iter(v))  # get the single item from the set
            for k, v in self.parameters.items()
            if len(v) == 1
        }

    @property
    def varying_parameters(self) -> Dict[str, List[Any]]:
        """dict: the parameters that vary in this result collection"""
        return {k: sorted(v) for k, v in self.parameters.items() if len(v) > 1}

    def filtered(self, **kwargs) -> "ResultCollection":
        r"""return a subset of the results

        Args:
            **kwargs: Specify parameter values of results that are retained

        Returns:
            :class:`ResultColelction`: The filtered collection
        """
        return self.__class__(
            item
            for item in self
            if all(item.parameters[k] == v for k, v in kwargs.items())
        )

    def sorted(self, *args) -> "ResultCollection":
        r"""return a sorted version of the results

        Args:
            *args: Specify parameters according to which the results are sorted

        Returns:
            :class:`ResultColelction`: The filtered collection
        """

        def sort_func(item):
            """helper function for ordering the results"""
            return [item.parameters[name] for name in args]

        return self.__class__(sorted(self, key=sort_func))

    @property
    def dataframe(self):
        """create a pandas Dataframe summarizing the data"""
        import pandas as pd

        assert self.same_model

        def get_data(result):
            """helper function to extract the data"""
            data = result.parameters.copy()
            if result.info.get("name"):
                data.setdefault("name", result.info["name"])
            if np.isscalar(result.result):
                data["result"] = result.result
            elif isinstance(result.result, dict):
                for key, value in result.result.items():
                    if np.isscalar(value):
                        data[key] = value
            else:
                raise RuntimeError("Do not know how to interpret result")
            return data

        return pd.DataFrame([get_data(result) for result in self])
