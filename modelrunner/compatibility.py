"""
Contains code necessary for loading results from previous version

.. codeauthor:: David Zwicker <david.zwicker@ds.mpg.de>
"""


from __future__ import annotations

from typing import TYPE_CHECKING, Any, Dict, Iterator, List, Optional

from .model import ModelBase
from .state import StateBase, make_state
from .state.io import (
    NumpyEncoder,
    read_hdf_data,
    simplify_data,
    write_hdf_dataset,
    zarrElement,
)

if TYPE_CHECKING:
    from .results import Result  # @UnusedImport


def result_from_simple_objects_version0(
    content, model: Optional[ModelBase] = None
) -> "Result":
    """old reader for backward compatible reading"""
    from .results import Result  # @Reimport

    return Result.from_data(
        model_data=content.get("model", {}),
        state=content.get("result"),
        model=model,
        info=content.get("info", {}),
    )

    @classmethod
    def _from_simple_objects(cls, content, model: Optional[ModelBase] = None) -> Result:
        """read result from a JSON file

        Args:
            path (str or :class:`~pathlib.Path`): The path to the file
            model (:class:`ModelBase`): The model from which the result was obtained
        """
        format_version = content.pop("__version__", None)
        if format_version is None:
            return cls._from_simple_objects_version0(content, model)
        elif format_version != cls._state_format_version:
            raise RuntimeError(f"Cannot read format version {format_version}")

        return cls.from_data(
            model_data=content.get("model", {}),
            state=StateBase._from_simple_objects(content["state"]),
            info=content.get("info"),
        )

    def _to_simple_objects(self) -> Any:
        """convert result to simple objects

        Args:
            path (str or :class:`~pathlib.Path`): The path to the file
        """
        content = {
            "__version__": self._state_format_version,
            "model": simplify_data(self.model._state_attributes),
            "state": self.state._to_simple_objects(),
        }
        if self.info:
            content["info"] = self.info
        return content

    @classmethod
    def _from_hdf_version0(
        cls, hdf_element, model: Optional[ModelBase] = None
    ) -> Result:
        """old reader for backward compatible reading"""
        model_data = {
            key: json.loads(value) for key, value in hdf_element.attrs.items()
        }
        if "result" in hdf_element:
            result = read_hdf_data(hdf_element["result"])
        else:
            result = model_data.pop("result")
        # check for other nodes, which might not be read

        info = model_data.pop("__info__") if "__info__" in model_data else {}

        return cls.from_data(
            model_data=model_data, state=result, model=model, info=info
        )

    @classmethod
    def _from_hdf(cls, hdf_element, model: Optional[ModelBase] = None) -> Result:
        """read result from a HDf file

        Args:
            hdf_element: The path to the file
            model (:class:`ModelBase`): The model from which the result was obtained
        """
        attributes = {
            key: json.loads(value) for key, value in hdf_element.attrs.items()
        }
        # extract version information from attributes
        format_version = attributes.pop("__version__", None)
        if format_version is None:
            return cls._from_hdf_version0(hdf_element, model)
        elif format_version != cls._state_format_version:
            raise RuntimeError(f"Cannot read format version {format_version}")
        info = attributes.pop("__info__", {})  # load additional info

        # the remaining attributes correspond to the model
        model_data = attributes

        # load state
        state_attributes = read_hdf_data(hdf_element["state"])
        state_data = read_hdf_data(hdf_element["data"])
        state = StateBase.from_data(state_attributes, state_data)

        return cls.from_data(model_data=model_data, state=state, model=model, info=info)

    def _write_hdf(self, root) -> None:
        """write result to HDF file

        Args:
            path (str or :class:`~pathlib.Path`): The path to the file
        """
        warnings.warn(
            "The HDF format is deprecated. Use `zarr` instead", DeprecationWarning
        )

        # write attributes
        for key, value in self.model._state_attributes.items():
            root.attrs[key] = json.dumps(value, cls=NumpyEncoder)
        if self.info:
            root.attrs["__info__"] = json.dumps(self.info, cls=NumpyEncoder)
        root.attrs["__version__"] = json.dumps(self._state_format_version)

        # write the actual data
        write_hdf_dataset(root, self.state._state_attributes_store, "state")
        write_hdf_dataset(root, self.state._state_data_store, "data")

    @classmethod
    def _from_zarr(
        cls, zarr_element: zarrElement, *, index=..., model: Optional[ModelBase] = None
    ) -> Result:
        """create result from data stored in zarr"""
        attributes = {
            key: json.loads(value) for key, value in zarr_element.attrs.items()
        }
        # extract version information from attributes
        format_version = attributes.pop("__version__", None)
        if format_version != cls._state_format_version:
            raise RuntimeError(f"Cannot read format version {format_version}")
        info = attributes.pop("__info__", {})  # load additional info

        # the remaining attributes correspond to the model
        model_data = attributes

        # load state
        state = StateBase._from_zarr(zarr_element["state"])

        return cls.from_data(model_data=model_data, state=state, model=model, info=info)

    def _write_zarr(
        self, zarr_group: zarr.Group, *, label: str = "data", **kwargs
    ) -> zarrElement:
        """write the entire Result object to a `zarr` file"""
        # create a zarr group to store all data
        result_group = zarr_group.create_group(label)

        # collect attributes from
        attributes = {}
        for key, value in self.model._state_attributes.items():
            attributes[key] = json.dumps(value, cls=NumpyEncoder)
        if self.info:
            attributes["__info__"] = json.dumps(self.info, cls=NumpyEncoder)
        attributes["__version__"] = json.dumps(self._state_format_version)

        # write the actual data
        self.state._write_zarr(result_group, label="state")
        result_group.attrs.update(attributes)
        return result_group
