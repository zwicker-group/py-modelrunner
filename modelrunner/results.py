"""
.. codeauthor:: David Zwicker <david.zwicker@ds.mpg.de>
"""

from __future__ import annotations

import collections
import inspect
import itertools
import json
import logging
import warnings
from pathlib import Path
from typing import Any, Dict, Iterator, List, Set, Tuple, Type, Union

import numpy as np
from tqdm.auto import tqdm

from .io import IOBase, NumpyEncoder, read_hdf_data, write_hdf_dataset
from .model import ModelBase
from .state import StateBase, make_state


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


class Result(IOBase):
    """describes a model (with parameters) together with its result"""

    def __init__(self, model: ModelBase, state: StateBase, info: Dict[str, Any] = None):
        """
        Args:
            model (:class:`ModelBase`):
                The model from which the result was obtained
            state (:class:`StateBase`):
                The actual result, saved as a state
            info (dict):
                Additional information for this result
        """
        assert isinstance(model, ModelBase)
        assert isinstance(state, StateBase)
        self.model = model
        self.info = info
        self.state = state

    @property
    def data(self):
        """direct access to the underlying state data"""
        return self.state.data

    @classmethod
    def from_data(
        cls,
        model_data: Dict[str, Any],
        state: StateBase,
        model: ModelBase = None,
        info: Dict[str, Any] = None,
    ) -> Result:
        """create result from data

        Args:
            model_data (dict):
                The data identifying the model
            state:
                The actual result data
            model (:class:`ModelBase`):
                The model from which the result was obtained
            info (dict):
                Additional information for this result
        """
        if model is None:
            model_cls: Type[ModelBase] = MockModel
        else:
            model_cls = model if inspect.isclass(model) else model.__class__

        if not model_data:
            warnings.warn("Model data not found")
        model = model_cls(model_data.get("parameters", {}))
        model.name = model_data.get("name")
        model.description = model_data.get("description")

        return cls(model, make_state(state), info)

    @property
    def parameters(self) -> Dict[str, Any]:
        return self.model.parameters

    @classmethod
    def _from_text_data(cls, content, model: ModelBase = None) -> Result:
        """read result from a JSON file

        Args:
            path (str or :class:`~pathlib.Path`): The path to the file
            model (:class:`ModelBase`): The model from which the result was obtained
        """
        return cls.from_data(
            model_data=content.get("model", {}),
            state=StateBase._from_text_data(content["state"]),
            model=model,
            info=content.get("info"),
        )

    def _to_text_data(self):
        """write result to JSON file"""
        content = {
            "model": self.model.attributes,
            "state": self.state._to_text_data(),
        }
        if self.info:
            content["info"] = self.info
        return content

    @classmethod
    def _from_hdf(cls, hdf_element, model: ModelBase = None) -> Result:
        """read result from a HDf file

        Args:
            hdf_element: The path to the file
            model (:class:`ModelBase`): The model from which the result was obtained
        """
        model_data = {
            key: json.loads(value) for key, value in hdf_element.attrs.items()
        }
        attributes = read_hdf_data(hdf_element["state"])
        data = read_hdf_data(hdf_element["data"])
        # else:
        #     state = model_data.pop("result")
        # # check for other nodes, which might not be read

        # load state
        state = StateBase.from_state(attributes, data)

        # load additional info
        info = model_data.pop("__info__") if "__info__" in model_data else {}

        return cls.from_data(model_data=model_data, state=state, model=model, info=info)

    def _write_hdf(self, root) -> None:
        """write result to HDF file

        Args:
            path (str or :class:`~pathlib.Path`): The path to the file
        """
        # write attributes
        for key, value in self.model.attributes.items():
            root.attrs[key] = json.dumps(value, cls=NumpyEncoder)

        if self.info:
            root.attrs["__info__"] = json.dumps(self.info, cls=NumpyEncoder)

        # write the actual data
        write_hdf_dataset(root, self.state.attributes, "state")
        write_hdf_dataset(root, self.state.data, "data")


class ResultCollection(List[Result]):
    """represents a collection of results"""

    @classmethod
    def from_folder(
        cls,
        folder: Union[str, Path],
        pattern: str = "*.*",
        model: ModelBase = None,
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
                    result = Result.from_file(path, model=model)
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
            if np.isscalar(result.state.data):
                data["result"] = result.state.data
            elif isinstance(result.state.data, dict):
                for key, value in result.state.data.items():
                    if np.isscalar(value):
                        data[key] = value
                    elif isinstance(value, (list, tuple, np.ndarray)):
                        data[key] = np.asarray(value)
            else:
                raise RuntimeError("Do not know how to interpret result")
            return data

        return pd.DataFrame([get_data(result) for result in self])
