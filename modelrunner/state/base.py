"""
Base classes that describe the state of a simulation at a single point in time

.. codeauthor:: David Zwicker <david.zwicker@ds.mpg.de>
"""


from __future__ import annotations

import copy
import warnings
from abc import ABCMeta, abstractmethod
from typing import Any, Dict, Optional, Type, TypeVar

import numpy as np
import zarr

from .io import IOBase, simplify_data, zarrElement


class NoData:
    """helper class that marks data omission"""

    ...


def _equals(left: Any, right: Any) -> bool:
    """checks whether two objects are equal, also supporting :class:~numpy.ndarray`

    Args:
        left: one object
        right: other object

    Returns:
        bool: Whether the two objects are equal
    """
    if type(left) is not type(right):
        return False

    if isinstance(left, str):
        return bool(left == right)

    if isinstance(left, np.ndarray):
        return np.array_equal(left, right)

    if isinstance(left, dict):
        return left.keys() == right.keys() and all(
            _equals(left[key], right[key]) for key in left
        )

    if hasattr(left, "__iter__"):
        return len(left) == len(right) and all(
            _equals(l, r) for l, r in zip(left, right)
        )

    return bool(left == right)


TState = TypeVar("TState", bound="StateBase")


class StateBase(IOBase, metaclass=ABCMeta):
    """Base class for specifying the state of a simulation

    A state contains values of all degrees of freedom of a physical system (called the
    `data`) and some additional information (called `attributes`). The `data` is mutable
    and often a numpy array or a collection of numpy arrays. Conversely, the
    `attributes` are stroed in a dictionary with immutable values. To allow flexible
    storage, we define the properties `_state_data` and `_state_attributes`, which by
    default return `attributes` and `data` directly, but may be overwritten to process
    the data before storage (e.g., by additional serialization).

    .. automethod:: StateBase._state_init
    .. autoproperty:: StateBase._state_attributes
    .. autoproperty:: StateBase._state_attributes_store
    .. autoproperty:: StateBase._state_data
    .. autoproperty:: StateBase._state_data_store
    """

    _state_format_version = 1
    """int: number indicating the version of the file format"""

    _state_classes: Dict[str, StateBase] = {}
    """dict: class-level list of all subclasses of StateBase"""

    _state_attributes_attr_name: Optional[str] = None
    """str: name of the class attribute holding the state attributes. This is only used
    if the subclass does not overwrite the `_state_attributes` attribute. If the value
    is `None`, no attributes are stored with the state."""

    _state_data_attr_name: str = "data"
    """str: name of the attribute where the data is stored. This is only used if the
    subclass does not overwrite the `_state_data` attribute."""

    @abstractmethod
    def __init__(self, data: Optional[Any] = None):
        """
        Args:
            data: The data describing the state
        """
        ...

    def __init_subclass__(cls, **kwargs):  # @NoSelf
        """register all subclasses to reconstruct them later"""
        # register the subclasses
        super().__init_subclass__(**kwargs)
        if cls is not StateBase:
            if cls.__name__ in StateBase._state_classes:
                warnings.warn(f"Redefining class {cls.__name__}")
            StateBase._state_classes[cls.__name__] = cls

    def _state_init(self, attributes: Dict[str, Any], data=NoData) -> None:
        """initialize the state with attributes and (optionally) data

        Args:
            attributes (dict): Additional (unserialized) attributes
            data: The data of the degerees of freedom of the physical system
        """
        if data is not NoData:
            self._state_data = data
        if attributes:
            self._state_attributes = attributes

    @property
    def _state_attributes(self) -> Dict[str, Any]:
        """dict: Additional attributes, which are required to restore the state"""
        if self._state_attributes_attr_name is None:
            return {}

        # try getting the attributes from the default name
        try:
            return getattr(self, self._state_attributes_attr_name)  # type: ignore
        except AttributeError:
            # this can happen if the attribute is not defined
            raise AttributeError("`_state_attributes` should be defined by subclass")

    @_state_attributes.setter
    def _state_attributes(self, attributes: Dict[str, Any]) -> None:
        """set the attributes of the state"""
        if self._state_attributes_attr_name is None:
            # attribute name was not specified
            if attributes:
                raise ValueError("`_state_attributes_attr_name` not set")

        else:
            # attribute name was specified
            try:
                # try setting attributes directly
                setattr(self, self._state_attributes_attr_name, attributes)
            except AttributeError:
                # this can happen if `data` is a read-only attribute, i.e., if the data
                # attribute is managed by the child class
                raise AttributeError("subclass should define `_state_attributes`")

    @property
    def _state_attributes_store(self) -> Dict[str, Any]:
        """dict: Attributes in the form in which they will be written to storage

        This property modifies the normal `_state_attributes` and adds information
        necessary for restoring the class using :meth:`StateBase.from_data`.
        """
        # make a copy since we add additional fields below
        attrs = self._state_attributes.copy()

        # add some additional information
        attrs["__class__"] = self.__class__.__name__
        attrs["__version__"] = self._state_format_version
        return attrs

    @property
    def _state_data(self) -> Any:
        """determines what data is stored in this state

        This property can be used to determine what is stored as `data` and in which
        form.
        """
        try:
            return getattr(self, self._state_data_attr_name)
        except AttributeError:
            # this can happen if the `data` attribute is not defined
            raise AttributeError("subclass should define `_state_data`")

    @_state_data.setter
    def _state_data(self, data) -> None:
        """set the data of the class"""
        try:
            setattr(self, self._state_data_attr_name, data)  # try setting data directly
        except AttributeError:
            # this can happen if `data` is a read-only attribute, i.e., if the data
            # attribute is managed by the child class
            raise AttributeError("`_state_data` should be defined by subclass")

    @property
    def _state_data_store(self) -> Any:
        """form of the data stored in this state which will be written to storage

        This property modifies the normal `_state_data` and adds information
        necessary for restoring the class using :meth:`StateBase.from_data`.
        """
        return self._state_data

    def __eq__(self, other) -> bool:
        if self.__class__ is not other.__class__:
            return False
        if self._state_attributes != other._state_attributes:
            return False
        return _equals(self._state_data, other._state_data)

    def __getstate__(self) -> Dict[str, Any]:
        """return a representation of the current state

        Note that this representation might contain views into actual data
        """
        attrs = self._state_attributes_store
        # remove private attributes used for persistent storage
        attrs.pop("__class__")
        attrs.pop("__version__")
        return {"attributes": attrs, "data": self._state_data_store}

    def __setstate__(self, dictdata: Dict[str, Any]):
        """set all properties of the object from a stored representation"""
        self._state_init(dictdata.get("attributes", {}), dictdata.get("data", NoData))

    @classmethod
    def from_data(cls: Type[TState], attributes: Dict[str, Any], data=NoData) -> TState:
        """create instance of any state class from attributes and data

        Args:
            attributes (dict): Additional (unserialized) attributes
            data: The data of the degerees of freedom of the physical system

        Returns:
            The object containing the given attributes and data
        """
        # copy attributes since they are modified in this function
        attributes = attributes.copy()
        cls_name = attributes.pop("__class__", None)

        if cls_name is None or cls.__name__ == cls_name:
            # attributes contain right class name or no class information at all
            # => instantiate current class with given data
            format_version = attributes.pop("__version__", 0)
            if format_version != cls._state_format_version:
                warnings.warn(
                    f"File format version mismatch "
                    f"({format_version} != {cls._state_format_version})"
                )

            # create a new object without calling __init__, which might be overwriten by
            # the subclass and not follow our interface
            obj = cls.__new__(cls)
            obj._state_init(attributes, data)
            return obj

        elif cls is StateBase:
            # use the base class as a point to load arbitrary subclasses
            if cls_name == "StateBase":
                raise RuntimeError("Cannot create StateBase instances")
            state_cls = cls._state_classes[cls_name]
            return state_cls.from_data(attributes, data)  # type: ignore

        else:
            raise ValueError(f"Incompatible state class {cls_name}")

    def copy(self: TState, method: str, data=None) -> TState:
        """create a copy of the state

        There are several methods of copying the state:

        `clean`:
            Makes a copy of the state by gathering its contents using
            :meth:`~StateBase.__getstate__`, makeing a copy of only the actual data and
            then instantiating a new state class, using :meth:`~StateBase.__setstate__`
            to restore the state. Since a new object is created, all data not captured
            by `__getstate__` (like internal caches) are lost!
        `shallow`:
            Performs a shallow copy of all attributes of the class. This is simply
            copying the entire :attr:`__dict__`
        `data`:
            Like `shallow`, but additionally makes a deep copy of the state data (stored
            in the :attr:`_state_data`, which typically is aliased by :attr:`data`).

        Args:
            method (str):
                Determines whether a `clean`, `shallow`, or `data` copy is performed.
                See description above for details.
            data:
                Data to be used instead of the one in the current state. This data is
                used as is and not copied!

        Returns:
            A copy of the current state object
        """
        # create a new object of the same class without any attributes
        obj = self.__class__.__new__(self.__class__)

        if method == "clean":
            # make clean copy by re-initializing state with copy of relevant attributes
            state = copy.deepcopy(self.__getstate__())  # copy current state
            if data is not None:
                state["data"] = data
            # use __setstate__ to set data on new object
            obj.__setstate__(state)

        elif method == "shallow":
            # (shallow) copy of all attributes of current state, including `data`
            obj.__dict__ = self.__dict__.copy()
            if data is not None:
                obj._state_data = data

        elif method == "data":
            # (shallow) copy of all attributes of current state, except `data`, which is
            # copied using a deep-copy
            obj.__dict__ = self.__dict__.copy()
            if data is None:
                obj._state_data = copy.deepcopy(self._state_data)
            else:
                obj._state_data = data

        else:
            raise ValueError(f"Unknown copy method {method}")
        return obj

    def _state_write_zarr_attributes(
        self, element: zarrElement, attrs: Optional[Dict[str, Any]] = None
    ) -> zarrElement:
        """prepare the zarr element for this state"""
        # write the attributes of the state
        element.attrs.update(simplify_data(self._state_attributes_store))

        # write additional attributes provided as argument
        if attrs is not None:
            element.attrs.update(simplify_data(attrs))

        return element

    def _state_write_zarr_data(
        self, zarr_group: zarr.Group, *, name: str = "data", **kwargs
    ) -> zarrElement:
        raise NotImplementedError

    def _write_zarr(
        self, zarr_group: zarr.Group, attrs: Optional[Dict[str, Any]] = None, **kwargs
    ) -> zarrElement:
        """writes the state to a zarr storage

        Args:
            zarr_group (:class:`zarr.Group`): Group into which the data is written
            attrs (dict): Additional attributes that are stored

        Returns:
            :class:`zarr.Group` or :class:`zarr.Array`: The written zarr element
        """
        element = self._state_write_zarr_data(zarr_group, **kwargs)
        self._state_write_zarr_attributes(element, attrs)
        return element

    @classmethod
    def _from_zarr(cls, zarr_element: zarrElement, *, index=...) -> StateBase:
        """create instance of correct subclass from data stored in zarr"""
        # determine the class that knows how to read this data
        class_name = zarr_element.attrs["__class__"]
        state_cls = cls._state_classes[class_name]

        # read the attributes and the data using this class
        attributes = zarr_element.attrs.asdict()
        data = state_cls._state_read_zarr_data(zarr_element, index=index)

        # create an instance of this class
        return state_cls.from_data(attributes, data)

    @classmethod
    def _state_read_zarr_data(cls, zarr_element: zarrElement, *, index=...) -> Any:
        """read data stored in zarr element"""
        raise NotImplementedError

    def _state_update_from_zarr(self, element: zarrElement, *, index=...) -> None:
        """update the current state from an zarr element"""
        raise NotImplementedError

    def _state_prepare_zarr_trajectory(
        self, zarr_group: zarr.Group, attrs: Optional[Dict[str, Any]] = None, **kwargs
    ) -> zarrElement:
        """prepare the zarr element for this state"""
        raise NotImplementedError

    def _state_append_to_zarr_trajectory(self, zarr_element: zarrElement) -> None:
        """append current data to the prepared zarr element"""
        raise NotImplementedError

    @classmethod
    def _from_simple_objects(
        cls, content: Dict[str, Any], *, state_cls: Optional[StateBase] = None
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
            return state_cls.from_data(content["attributes"], content["data"])

    def _to_simple_objects(self) -> Dict[str, Any]:
        """return object data suitable for encoding as text"""
        return {
            "attributes": self._state_attributes_store,
            "data": self._state_data_store,
        }
