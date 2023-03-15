"""
Base classes that describe the state of a simulation at a single point in time

.. codeauthor:: David Zwicker <david.zwicker@ds.mpg.de>
"""

from __future__ import annotations

import copy
import warnings
from typing import Any, Dict, Optional

import numpy as np
import zarr

from .io import IOBase, simplify_data, zarrElement


class DoNotStore(Exception):
    """helper exception to signal that an attribute should not be stored"""

    ...


def _equals(left: Any, right: Any) -> bool:
    """checks whether two objects are equal, also supporting :class:~numpy.ndarray`

    Args:
        left: one object
        right: other object

    Returns:
        bool: Whether the two objects are equal
    """
    if left.__class__ is not right.__class__:
        return False

    if isinstance(left, str):
        return bool(left == right)

    if isinstance(left, np.ndarray):
        return np.array_equal(left, right)

    if isinstance(left, dict):
        return left.keys() == right.keys() and all(
            _equals(left[key], right[key]) for key in left
        )

    if isinstance(left, StateBase):
        return left._attributes_store == right._attributes_store and _equals(
            left._data_store, right._data_store
        )

    if hasattr(left, "__iter__"):
        return len(left) == len(right) and all(
            _equals(l, r) for l, r in zip(left, right)
        )

    return bool(left == right)


class StateBase(IOBase):
    """Base class for specifying the state of a simulation

    A state contains values of all degrees of freedom of a physical system (stored in
    the `data` field) and potentially some additional information (stored in the
    `attributes` field). The `data` is mutable and often a numpy array or a collection
    of numpy arrays. Conversely, the `attributes` are a dictionary with immutable
    values. To allow flexible storage, we define the attributes `_attributes_store` and
    `_data_store`, which by default return `attributes` and `data` directly, but may be
    overwritten to process the data before storage (e.g., by additional serialization).
    """

    _format_version = 1
    """int: number indicating the version of the file format"""

    _state_classes: Dict[str, StateBase] = {}
    """dict: class-level list of all subclasses of StateBase"""

    _data_attribute: str = "data"
    """str: name of the attribute where the data is stored"""

    def __init_subclass__(cls, **kwargs):  # @NoSelf
        """register all subclasses to reconstruct them later"""
        # register the subclasses
        super().__init_subclass__(**kwargs)
        if cls is not StateBase:
            cls._state_classes[cls.__name__] = cls

    @property
    def attributes(self) -> Dict[str, Any]:
        """dict: Additional attributes, which are required to restore the state"""
        return {}

    def _pack_attribute(self, name: str, value) -> Any:
        """convert an attribute into a form that can be stored

        If this function raises :class:`DoNotStore`, the attribute will not be stored

        Args:
            name (str): Name of the attribute
            value: The value of the attribute

        Returns:
            A simplified form of the attribute that can be restored
        """
        return simplify_data(value)

    @classmethod
    def _unpack_attribute(cls, name: str, value) -> Any:
        """convert an attribute from a form that was stored

        Args:
            name (str): Name of the attribute
            value: The value of the attribute

        Returns:
            A restored form of the attribute
        """
        return value

    @property
    def _attributes_store(self) -> Dict[str, Any]:
        """dict: Attributes in the form in which they will be written to storage"""
        # pack all attributes for storage
        attrs = {}
        for name, value in self.attributes.items():
            try:
                attrs[name] = self._pack_attribute(name, value)
            except DoNotStore:
                pass

        # add some additional information
        attrs["__class__"] = self.__class__.__name__
        attrs["__version__"] = self._format_version
        return attrs

    @classmethod
    def _read_attributes(cls, attributes: Dict[str, Any]) -> Dict[str, Any]:
        """recreating stored attributes

        This classmethod can be overwritten if attributes have been serialized by the
        :meth:`_attributes_store` property.
        """
        return {
            name: cls._unpack_attribute(name, value)
            for name, value in attributes.items()
        }

    @property
    def _data_store(self) -> Any:
        """attribute that determines what data is stored in this state"""
        try:
            return getattr(self, self._data_attribute)
        except AttributeError:
            return None

    def __eq__(self, other):
        return _equals(self, other)

    @classmethod
    def from_data(cls, attributes: Dict[str, Any], data=None) -> StateBase:
        """create instance of any state class from attributes and data

        Args:
            attributes (dict): Additional (unserialized) attributes
            data: The data of the degerees of freedom of the physical system
        """
        if cls.__name__ == "StateBase":
            # use the base class as a point to load arbitrary subclasses
            if attributes["__class__"] == "StateBase":
                raise RuntimeError("Cannot create StateBase instances")
            state_cls = cls._state_classes[attributes["__class__"]]
            return state_cls.from_data(attributes, data)

        elif cls.__name__ == attributes["__class__"]:
            # simple version instantiating the current class with the given data
            num_attrs = 1  # __class__ was stored in attributes
            if "__version__" in attributes:
                format_version = attributes["__version__"]
                num_attrs += 1
            else:
                format_version = 0
            if format_version != cls._format_version:
                warnings.warn(
                    f"File format version mismatch "
                    f"({format_version} != {cls._format_version})"
                )
            if len(attributes) > num_attrs:  # additional attributes given
                warnings.warn(
                    f"Unused attributes, but {cls.__name__} did not implemented custom "
                    "from_data method"
                )

            # create a new object without calling __init__, which might be overwriten by
            # the subclass and not follow our interface
            obj = cls.__new__(cls)
            if data is not None:
                setattr(obj, obj._data_attribute, data)
            return obj

        else:
            raise ValueError(f"Incompatible state class {attributes['class']}")

    def copy(self, data=None):
        if data is None:
            data = copy.deepcopy(getattr(self, self._data_attribute))
        return self.__class__.from_data(copy.deepcopy(self.attributes), data)

    def _write_zarr_attributes(
        self, element: zarrElement, attrs: Optional[Dict[str, Any]] = None
    ) -> zarrElement:
        """prepare the zarr element for this state"""
        # write the attributes of the state
        element.attrs.update(simplify_data(self._attributes_store))

        # write additional attributes provided as argument
        if attrs is not None:
            element.attrs.update(simplify_data(attrs))

        return element

    def _write_zarr_data(
        self, zarr_group: zarr.Group, *, name: str = "data", **kwargs
    ) -> zarrElement:
        raise NotImplementedError

    def _write_zarr(
        self, zarr_group: zarr.Group, attrs: Optional[Dict[str, Any]] = None, **kwargs
    ) -> zarrElement:
        element = self._write_zarr_data(zarr_group, **kwargs)
        self._write_zarr_attributes(element, attrs)
        return element

    @classmethod
    def _from_zarr(cls, zarr_element: zarrElement, *, index=...) -> StateBase:
        """create instance of correct subclass from data stored in zarr"""
        # determine the class that knows how to read this data
        class_name = zarr_element.attrs["__class__"]
        state_cls = cls._state_classes[class_name]

        # read the attributes and the data using this class
        attributes = state_cls._read_attributes(zarr_element.attrs.asdict())
        data = state_cls._read_zarr_data(zarr_element, index=index)

        # create an instance of this class
        return state_cls.from_data(attributes, data)

    @classmethod
    def _read_zarr_data(cls, zarr_element: zarrElement, *, index=...) -> Any:
        """read data stored in zarr element"""
        raise NotImplementedError

    def _update_from_zarr(self, element: zarrElement, *, index=...) -> None:
        raise NotImplementedError

    def _prepare_zarr_trajectory(
        self, zarr_group: zarr.Group, attrs: Optional[Dict[str, Any]] = None, **kwargs
    ) -> zarrElement:
        """prepare the zarr element for this state"""
        raise NotImplementedError

    def _append_to_zarr_trajectory(self, zarr_element: zarrElement) -> None:
        """append current data to the prepared zarr element"""
        raise NotImplementedError

    @classmethod
    def _from_simple_objects(
        cls, content, *, state_cls: Optional[StateBase] = None
    ) -> StateBase:
        """create state from text data

        Args:
            content: The loaded data
        """
        if state_cls is None:
            # general branch that determines the state class to use to load the object
            state_cls = cls._state_classes[content["attributes"]["__class__"]]
            return state_cls._from_simple_objects(content, state_cls=state_cls)
        else:
            # specific (basic) implementation that just reads the state
            attributes = cls._read_attributes(content["attributes"])
            return state_cls.from_data(attributes, content["data"])

    def _to_simple_objects(self):
        """return object data suitable for encoding as text"""
        return {"attributes": self._attributes_store, "data": self._data_store}
