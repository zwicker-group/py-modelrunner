"""
Classes that describe the final result of a model simulation.

.. codeauthor:: David Zwicker <david.zwicker@ds.mpg.de>
"""

from __future__ import annotations

import collections
import inspect
import itertools
import logging
import warnings
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Set, Tuple, Type, Union

import numpy as np
from tqdm.auto import tqdm

from .model import ModelBase
from .storage import StorageID, open_storage, storage_actions
from .storage.access_modes import ModeType
from .storage.attributes import Attrs


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

    _format_version = 2
    """int: number indicating the version of the file format"""

    def __init__(
        self, model: ModelBase, result: Any, info: Optional[Dict[str, Any]] = None
    ):
        """
        Args:
            model (:class:`ModelBase`):
                The model from which the result was obtained
            result:
                The actual result
            info (dict):
                Additional information for this result
        """
        if not isinstance(model, ModelBase):
            raise TypeError("The model should be of type `ModelBase`")
        self.model = model
        self.result = result
        self.info: Attrs = {} if info is None else info

    @property
    def data(self):
        """direct access to the underlying state data"""
        return self.result

    @classmethod
    def from_data(
        cls,
        model_data: Dict[str, Any],
        result,
        model: Optional[ModelBase] = None,
        info: Optional[Dict[str, Any]] = None,
    ) -> Result:
        """create result from data

        Args:
            model_data (dict):
                The data identifying the model
            result:
                The actual result data
            model (:class:`ModelBase`):
                The model from which the result was obtained
            info (dict):
                Additional information for this result

        Returns:
            :class:`Result`: The result object
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

        return cls(model, result, info)

    @property
    def parameters(self) -> Dict[str, Any]:
        return self.model.parameters

    @classmethod
    def from_file(
        cls,
        storage: StorageID,
        loc: str = "result",
        *,
        model: Optional[ModelBase] = None,
    ):
        """load object from a file

        Args:
            store (str or :class:`zarr.Store`):
                Path or instance describing the storage, which is either a file path or
                a :class:`zarr.Storage`.
            key (str):
                Name of the node in which the data was stored. This applies to some
                hierarchical storage formats.
        """
        if isinstance(storage, (str, Path)):
            # check whether the file was written with an old format version
            from .compatibility.triage import result_check_load_old_version

            result = result_check_load_old_version(Path(storage), loc=loc, model=model)
            if result is not None:
                return result  # Result created from old version

        # assume that file was written with latest format version
        with open_storage(storage, mode="read") as storage_obj:
            attrs = storage_obj.read_attrs(loc)
            format_version = attrs.pop("format_version", None)
            if format_version == cls._format_version:
                # current version of storing results
                return cls.from_data(
                    model_data=attrs.get("model", {}),
                    result=storage_obj.read_item(loc, use_class=False),
                    model=model,
                    info=attrs.pop("info", {}),  # load additional info,
                )

            else:
                raise RuntimeError(f"Cannot read format version {format_version}")

    def to_file(
        self, storage: StorageID, loc: str = "result", *, mode: ModeType = "insert"
    ) -> None:
        """write this object to a file

        Args:
            storage (:class:`StorageBase` or :class:`StorageGroup`):
                The storage where the group is defined. If this is a
                :class:`StorageGroup` itself, `loc` is interpreted relative to that
                group
            loc (str or list of str):
                Denotes the location (path) of the group within the storage
            mode (str or :class:`~modelrunner.storage.access_modes.ModeType`):
                The file mode with which the storage is accessed, which determines the
                allowed operations. Common options are "read", "full", "append", and
                "truncate".
        """
        with open_storage(storage, mode=mode) as storage_obj:
            # collect attributes from the result
            attrs: Attrs = {
                "model": dict(self.model._state_attributes),
                "format_version": self._format_version,
            }
            if self.info:
                attrs["info"] = self.info
            # write the actual data
            storage_obj.write_object(loc, self.result, attrs=attrs, cls=self.__class__)


storage_actions.register("read_item", Result, Result.from_file)


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
                    result = Result.from_file(path, model=model)
                except Exception as err:
                    if strict:
                        err.args = (str(err) + f"\nError reading file `{path}`",)
                        raise
                    else:
                        logger.warning(f"Error reading file `{path}`")
                else:
                    results.append(result)

        # raise a warning if no results were detected
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

    def __add__(self, other: ResultCollection) -> ResultCollection:  # type: ignore
        if isinstance(other, ResultCollection):
            return ResultCollection(super().__add__(other))

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
        # deprecated on 2023-10-19
        warnings.warn("Property `dataframe` deprecated; use method `as_dataframe`")
        return self.as_dataframe()

    def as_dataframe(self, *, enforce_same_model: bool = True):
        """create a pandas dataframe summarizing the data

        Args:
            enforce_same_model (bool):
                If True, forces all model results to derive from the same model
        """
        import pandas as pd

        if enforce_same_model and not self.same_model:
            raise RuntimeError("Results are not from the same model")

        def get_data(result):
            """helper function to extract the data"""
            df_data = result.parameters.copy()

            # try obtaining the name of the result
            if result.info.get("name"):
                df_data.setdefault("name", result.info["name"])
            elif hasattr(result.model, "name"):
                df_data.setdefault("name", result.model.name)

            # try interpreting the result data in a format understood by pandas
            data = result.result
            if np.isscalar(data):
                df_data["result"] = data
            elif isinstance(data, dict):
                for key, value in data.items():
                    if np.isscalar(value):
                        df_data[key] = value
                    elif isinstance(value, (list, tuple, np.ndarray)):
                        df_data[key] = np.asarray(value)
            else:
                raise RuntimeError("Do not know how to interpret result")
            return df_data

        return pd.DataFrame([get_data(result) for result in self])
