"""
Base classes that describe the state of a simulation at a single point in time

.. codeauthor:: David Zwicker <david.zwicker@ds.mpg.de>
"""


from __future__ import annotations

import copy
import warnings
from abc import ABCMeta, abstractmethod
from typing import TYPE_CHECKING, Any, Dict, Optional, Type, TypeVar

import numpy as np

from ..storage import Location, StorageGroup, open_storage, storage_actions
from ..storage.access_modes import ModeType
from ..storage.attributes import Attrs, attrs_remove_dunderscore
from ..storage.utils import decode_class

if TYPE_CHECKING:
    from ..storage import StorageID


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


class StateBase(metaclass=ABCMeta):
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

    _state_format_version = 2
    """int: number indicating the version of the file format"""

    _state_classes: Dict[str, Type[StateBase]] = {}
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
            storage_actions.register("read_object", cls, cls._state_from_stored_data)

    def _state_init(self, attributes: Dict[str, Any], data=NoData) -> None:
        """initialize the state with attributes and (optionally) data

        Args:
            attributes (dict): Additional (unserialized) attributes
            data: The data of the degerees of freedom of the physical system
        """
        if data is not NoData:
            self._state_data = data
        attributes = attrs_remove_dunderscore(attributes)
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

    @classmethod
    def _state_get_attrs_from_storage(
        cls, storage: StorageGroup, loc: Location, *, check_version: bool = True
    ) -> Attrs:
        """read attributes from storage and optionally check format version

        Args:
            storage (str or :class:`~modelrunner.storage.StorageBase`):
                A storage opened with :func:`~modelrunner.storage.open_storage`
            loc (str or list of str):
                Name of the location where the data will be stored.
            check_version (bool):
                A number indicating whether the format version should be checked

        Raises:
            `RuntimeError`: If format version is specified, but not matched

        Returns:
            dict: Attributes without the `__class__` and `__version__` item
        """
        # read relevant attributes of the state
        attrs = storage.read_attrs(loc)
        attrs.pop("__class__", None)  # remove this information

        # check whether the data can be read
        version = attrs.pop("__version__", None)
        if check_version is not None and version != cls._state_format_version:
            raise RuntimeError(f"Cannot read format version {version}")

        return attrs

    @classmethod
    def _state_from_stored_data(
        cls, storage: StorageGroup, loc: Location, *, index: Optional[int] = None
    ):
        """create the state from storage

        Args:
            storage (:class:`StorageGroup`):
                A storage opened with :func:`~modelrunner.storage.open_storage`
            loc (str or list of str):
                The location in the storage where the state is read
            index (int, optional):
                If the location contains a trajectory of the state, `index` must denote
                the index determining which state should be created
        """
        # determine the class to reconstruct the data from attribute
        state_cls = _get_state_cls_from_storage(storage, loc)
        if state_cls == StateBase:
            raise NotImplementedError(f"Cannot read `{cls.__name__}`")
        else:
            return state_cls._state_from_stored_data(storage, loc, index=index)

    def _state_update_from_stored_data(
        self, storage: StorageGroup, loc: Location, index: Optional[int] = None
    ) -> None:
        """update the state data (but not its attributes) from storage

        Args:
            storage (:class:`StorageGroup`):
                A storage opened with :func:`~modelrunner.storage.open_storage`
            loc (str or list of str):
                The location in the storage where the state is read
            index (int, optional):
                If the location contains a trajectory of the state, `index` must denote
                the index determining which state should be created
        """
        raise NotImplementedError(f"Cannot update `{self.__class__.__name__}`")

    def _state_write_to_storage(self, storage: StorageGroup, loc: Location) -> None:
        """write the state to storage

        Args:
            storage (:class:`StorageGroup`):
                A storage opened with :func:`~modelrunner.storage.open_storage`
            loc (str or list of str):
                The location in the storage where the state is written
        """
        raise NotImplementedError(f"Cannot write `{self.__class__.__name__}`")

    def _state_create_trajectory(self, storage: StorageGroup, loc: Location) -> None:
        """prepare a trajectory of the current state

        Args:
            storage (:class:`StorageGroup`):
                A storage opened with :func:`~modelrunner.storage.open_storage`
            loc (str or list of str):
                The location in the storage where the trajectory is written
        """
        raise NotImplementedError(
            f"Cannot create trajectory for `{self.__class__.__name__}`"
        )

    def _state_append_to_trajectory(self, storage: StorageGroup, loc: Location) -> None:
        """append the current state to a prepared trajectory

        Args:
            storage (:class:`StorageGroup`):
                A storage opened with :func:`~modelrunner.storage.open_storage`
            loc (str or list of str):
                The location in the storage where the trajectory is written
        """
        raise NotImplementedError(
            f"Cannot extend trajectory for `{self.__class__.__name__}`"
        )

    @classmethod
    def from_file(cls, storage: StorageID, loc: Location = "state", **kwargs):
        r"""load object from a file

        Args:
            storage (str or :class:`~modelrunner.storage.StorageBase`):
                Path or instance describing the storage. The simplest choice is a path
                to a file, where the data is written in a format deduced from the file
                extension.
            loc (str or list of str):
                Name of the location where the data was stored.
            **kwargs:
                Arguments passed to :func:`~modelrunner.storage.open_storage`
        """
        kwargs.setdefault("mode", "readonly")
        with open_storage(storage, **kwargs) as opened_storage:
            return cls._state_from_stored_data(opened_storage, loc)

    def to_file(
        self,
        storage: StorageID,
        loc: Location = "state",
        *,
        mode: ModeType = "insert",
        **kwargs,
    ) -> None:
        """write this object to a file

        Args:
            storage (str or :class:`~modelrunner.storage.StorageBase`):
                Path or instance describing the storage. The simplest choice is a path
                to a file, where the data is written in a format deduced from the file
                extension.
            loc (str or list of str):
                Name of the location where the data will be stored.
            mode (str or :class:`~modelrunner.storage.access_modes.AccessMode`):
                The file mode with which the storage is accessed, which determines the
                allowed operations. Common options are "readonly", "full", "append", and
                "truncate".
            **kwargs:
                Arguments passed to :func:`~modelrunner.storage.open_storage`
        """
        with open_storage(storage, mode=mode, **kwargs) as opened_storage:
            self._state_write_to_storage(opened_storage, loc=loc)


def _get_state_cls_from_storage(
    storage: StorageGroup, loc: Location
) -> Type[StateBase]:
    """obtain class of state stored in a particular location

    Args:
        storage (str or :class:`~modelrunner.storage.StorageBase`):
            A storage opened with :func:`~modelrunner.storage.open_storage`
        loc (str or list of str):
            Name of the location where the data will be stored.

    Returns:
        A subclass of :class:`StateBase`
    """
    stored_cls = storage.read_attrs(loc).get("__class__", None)
    _, class_name = stored_cls.rsplit(".", 1)
    if class_name in StateBase._state_classes:
        return StateBase._state_classes[class_name]
    else:
        cls = decode_class(stored_cls)
        if cls is None:
            raise RuntimeError(f"Could not decode class `{stored_cls}`")
        return cls
