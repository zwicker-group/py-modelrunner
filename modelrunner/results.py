"""
Classes that describe the final result of a simulation.

.. codeauthor:: David Zwicker <david.zwicker@ds.mpg.de>
"""

from __future__ import annotations

import collections
import inspect
import itertools
import json
import logging
import os.path
import warnings
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Set, Tuple, Type, Union

import numpy as np
from tqdm.auto import tqdm

from .model import ModelBase
from .parameters import NoValueType


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


class NumpyEncoder(json.JSONEncoder):
    """helper class for encoding python data in JSON"""

    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, np.generic):
            return obj.item()
        if isinstance(obj, NoValueType):
            return None
        return json.JSONEncoder.default(self, obj)


def simplify_data(data):
    """simplify data (e.g. for writing to yaml)"""
    if isinstance(data, dict):
        data = {key: simplify_data(value) for key, value in data.items()}

    elif isinstance(data, (tuple, list)):
        data = [simplify_data(item) for item in data]

    elif isinstance(data, np.ndarray):
        if np.isscalar(data):
            data = data.item()
        elif data.size <= 100:
            # for less than ~100 items a list is actually more efficient to store
            data = data.tolist()

    elif isinstance(data, np.number):
        data = data.tolist()

    return data


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
        if not contains_array(data):
            # write everything as JSON encoded string
            if isinstance(data, dict):
                group = node.create_group(name)
                for key, value in data.items():
                    group.attrs[key] = json.dumps(value, cls=NumpyEncoder)
            else:
                node.attrs[name] = json.dumps(data, cls=NumpyEncoder)

        elif isinstance(data, dict):
            group = node.create_group(name)
            for key, value in data.items():
                write_hdf_dataset(group, value, key)

        else:
            group = node.create_group(name)
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

    def __init__(self, parameters: Optional[Dict[str, Any]] = None):
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

    def __init__(
        self, model: ModelBase, result_data, info: Optional[Dict[str, Any]] = None
    ):
        """
        Args:
            model (:class:`ModelBase`): The model from which the result was obtained
            result_data: The actual result data
            info (dict): Additional information for this result
        """
        self.model = model
        self.info = info
        if isinstance(result_data, Result):
            self.result: Any = result_data.result
        else:
            self.result = result_data

    @classmethod
    def from_data(
        cls,
        model_data: Dict[str, Any],
        result_data,
        model: Optional[ModelBase] = None,
        info: Optional[Dict[str, Any]] = None,
    ) -> Result:
        """create result from data

        Args:
            model_data (dict): The data identifying the model
            result_data: The actual result data
            model (:class:`ModelBase`): The model from which the result was obtained
            info (dict): Additional information for this result

        Returns:
            :class:`Result`: The result object
        """
        if model is None:
            model_cls: Type[ModelBase] = MockModel
        else:
            model_cls = model if inspect.isclass(model) else model.__class__

        if not model_data:
            warnings.warn("Model data not found")
        obj = model_cls(model_data.get("parameters", {}))
        obj.name = model_data.get("name")
        obj.description = model_data.get("description")

        return cls(obj, result_data, info)

    @property
    def parameters(self) -> Dict[str, Any]:
        return self.model.parameters

    @classmethod
    def from_file(cls, path, model: Optional[ModelBase] = None):
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
            raise ValueError(f"Unknown file format of `{path}`")

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
    def from_json(cls, path, model: Optional[ModelBase] = None) -> Result:
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
            result_data=data.get("result"),
            model=model,
            info=info,
        )

    def write_to_json(self, path) -> None:
        """write result to JSON file

        Args:
            path (str or :class:`~pathlib.Path`): The path to the file
        """
        data = {
            "model": simplify_data(self.model.attributes),
            "result": simplify_data(self.result),
        }
        if self.info:
            data["info"] = self.info

        with open(path, "w") as fp:
            json.dump(data, fp, cls=NumpyEncoder)

    @classmethod
    def from_yaml(cls, path, model: Optional[ModelBase] = None) -> Result:
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
            result_data=data.get("result"),
            model=model,
            info=info,
        )

    def write_to_yaml(self, path) -> None:
        """write result to YAML file

        Args:
            path (str or :class:`~pathlib.Path`): The path to the file
        """
        import yaml

        # compile all data
        data = {
            "model": simplify_data(self.model.attributes),
            "result": simplify_data(self.result),
        }
        if self.info:
            data["info"] = self.info

        with open(path, "w") as fp:
            yaml.dump(data, fp)

    @classmethod
    def from_hdf(cls, path, model: Optional[ModelBase] = None) -> Result:
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
            model_data=model_data, result_data=result, model=model, info=info
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
                fp.attrs[key] = json.dumps(simplify_data(value), cls=NumpyEncoder)

            if self.info:
                fp.attrs["__info__"] = json.dumps(
                    simplify_data(self.info), cls=NumpyEncoder
                )

            # write the actual data
            write_hdf_dataset(fp, self.result, "result")


class ResultCollection(List[Result]):
    """represents a collection of results"""

    @classmethod
    def from_folder(
        cls,
        folder: Union[str, Path],
        pattern: str = "*.*",
        model: Optional[ModelBase] = None,
        *,
        strict: bool = False,
        progress: bool = False,
    ):
        """create results collection from a folder

        args:
            folder (str):
                Path to the folder that is scanned
            pattern (str):
                Filename pattern that is used to detect result files
            model (:class:`~modelrunner.model.ModelBase`):
                Base class from which models are initialized
            strict (bool):
                Whether to raise an exception or just emit a warning when a file cannot
                be read
            progress (bool):
                Flag indicating whether a progress bar is shown
        """
        logger = logging.getLogger(cls.__name__)

        folder = Path(folder)
        assert folder.is_dir()

        # iterate over all files and load them as a Result
        results = []
        for path in tqdm(list(folder.glob(pattern)), disable=not progress):
            if path.is_file():
                try:
                    result = Result.from_file(path, model)
                except Exception as err:
                    if strict:
                        err.args = (str(err) + f"\nError reading file `{path}`",)
                        raise
                    else:
                        logger.warning(f"Error reading file `{path}`")
                else:
                    results.append(result)

        # raise a warning if now results were detected
        if not results:
            if pattern == "*.*":
                logger.warning("Did not find any files")
            else:
                logger.warning(
                    f"Did not find any files. Is pattern `{pattern}` too restrictive?"
                )

        return cls(results)

    def __repr__(self):
        return f"{self.__class__.__name__}(<{len(self)} Results>)"

    __str__ = __repr__

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
        """dict: the parameter values in this result collection

        Note that parameters that are lists in the individual models are turned into
        tuples, so they can be handled efficiently, e.g., in sets.
        """
        params = collections.defaultdict(set)

        for result in self:
            for k, v in result.model.parameters.items():
                if isinstance(v, list):
                    v = tuple(v)  # work around to make lists hashable
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

    def get(self, **kwargs) -> Result:
        """return a single result with the given parameters

        Warning:
            If there are multiple results compatible with the specified parameters, only
            the first one is returned.

        Args:
            **kwargs: Specify parameter values of result that is returned

        Returns:
            :class:`Result`: A single result from the collection
        """
        # return the first result that matches the requirements
        for item in self:
            if all(item.parameters[k] == v for k, v in kwargs.items()):
                return item
        raise ValueError("Result not contained in collection")

    def filtered(self, **kwargs) -> ResultCollection:
        r"""return a subset of the results

        Args:
            **kwargs: Specify parameter values of results that are retained

        Returns:
            :class:`ResultColelction`: The filtered collection
        """
        # return a filtered result collection
        return self.__class__(
            item
            for item in self
            if all(item.parameters[k] == v for k, v in kwargs.items())
        )

    def groupby(self, *args) -> Iterator[Tuple[Dict[str, List[Any]], ResultCollection]]:
        r"""group results according to the given variables

        Args:
            *args: Specify parameters according to which the results are sorted

        Returns:
            generator that allows iterating over the groups. Each iteration returns a
            dictionary with the current parameters and the associated
            :class:`ResultCollection`.
        """
        group_values = [self.parameters[name] for name in args]

        for group_value in itertools.product(*group_values):
            group_parameters = dict(zip(args, group_value))
            subset = self.filtered(**group_parameters)
            if len(subset) > 0:
                yield group_parameters, subset

    def sorted(self, *args, reverse: bool = False) -> ResultCollection:
        r"""return a sorted version of the results

        Args:
            *args: Specify parameters according to which the results are sorted
            reverse (bool): If True, sort in descending order

        Returns:
            :class:`ResultColelction`: The filtered collection
        """

        def sort_func(item):
            """helper function for ordering the results"""
            return [item.parameters[name] for name in args]

        return self.__class__(sorted(self, key=sort_func, reverse=reverse))

    def remove_duplicates(self) -> ResultCollection:
        """remove duplicates in the result collection"""
        #  we cannot use a set for `seen`, since parameters might not always be hashable
        unique_results, seen = [], []
        for result in self:
            if result.parameters not in seen:
                unique_results.append(result)
                seen.append(result.parameters)
        return self.__class__(unique_results)

    @property
    def dataframe(self):
        """create a pandas dataframe summarizing the data"""
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
                    elif isinstance(value, (list, tuple, np.ndarray)):
                        data[key] = np.asarray(value)
            else:
                raise RuntimeError("Do not know how to interpret result")
            return data

        return pd.DataFrame([get_data(result) for result in self])
