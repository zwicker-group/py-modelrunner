"""
.. codeauthor:: David Zwicker <david.zwicker@ds.mpg.de>
"""

from __future__ import annotations

from typing import Any, Collection, Iterator

import numpy as np
from numpy.typing import ArrayLike, DTypeLike

from .attributes import Attrs
from .base import StorageBase
from .utils import Array, Location, decode_class, encode_class, storage_actions


class StorageGroup:
    """Refers to a group within a storage."""

    def __init__(self, storage: StorageBase | StorageGroup, loc: Location = None):
        """
        Args:
            storage (:class:`StorageBase` or :class:`StorageGroup`):
                The storage where the group is defined. If this is a
                :class:`StorageGroup` itself, `loc` is interpreted relative to that
                group
            loc (str or list of str):
                Denotes the location (path) of the group within the storage
        """
        self.loc = []  # initialize empty location, since `loc` is relative to root

        # set the underlying storage and the relative location
        if isinstance(storage, StorageBase):
            self._storage = storage
            self.loc = self._get_loc(loc)
        elif isinstance(storage, StorageGroup):
            self._storage = storage._storage
            self.loc = self._get_loc([storage.loc, loc])
        else:
            raise TypeError(f"Cannot interpret `storage` of type `{storage.__class__}`")

        # do some sanity checks
        assert isinstance(self._storage, StorageBase)
        if self._storage.closed:
            raise RuntimeError("Storage is closed")
        if self.loc not in self._storage:
            raise RuntimeError(
                f'"/{"/".join(self.loc)}" is not in storage. Available root items are: '
                f"{list(self._storage.keys(loc=[]))}"
            )
        if not self.is_group():
            raise RuntimeError(f'"/{"/".join(self.loc)}" is not a group')

    def __repr__(self):
        return f'StorageGroup(storage={self._storage}, loc="/{"/".join(self.loc)}")'

    @property
    def parent(self) -> StorageGroup:
        """:class:`StorageGroup`: Parent group.

        Raises:
            RuntimeError: If current group is root group
        """
        if self.loc:
            return StorageGroup(self._storage, loc=self.loc[:-1])
        else:
            raise RuntimeError("Root group has no parent")

    def tree(self) -> None:
        """Print the hierarchical storage as a tree structure."""
        vertic = "│  "
        cross = "├──"
        corner = "└──"
        space = "   "

        def print_tree(loc: list[str], header: str = ""):
            """Recursive function printing information about one group."""
            group = StorageGroup(self._storage, loc)
            for i, key in enumerate(sorted(group.keys())):
                last = i == len(group) - 1
                if self._storage.is_group(loc + [key]):
                    cls = self._storage._read_attrs(loc).get("__class__")
                    if cls is None:
                        # item is a sub group
                        print(header + (corner if last else cross) + key)
                        print_tree(
                            loc + [key], header=header + (space if last else vertic)
                        )
                    else:
                        # item contains information to restore a certain class
                        print(header + (corner if last else cross) + f"{key} ({cls})")
                else:
                    # item is a simple, scalar item
                    print(header + (corner if last else cross) + key)

        if self.loc:
            print("/" + "/".join(self.loc))
        print_tree(self.loc)

    def _get_loc(self, loc: Location) -> list[str]:
        """Return a normalized location from various input.

        Args:
            loc (str or list of str):
                location in a general formation. For instance, "/" is interpreted as a
                group separator.

        Returns:
            list of str: A list of the individual location items
        """
        if self._storage.closed:
            raise RuntimeError("Storage is closed")

        # TODO: use regex to check whether loc is only alphanumerical and has no "/"
        def parse_loc(loc_data) -> list[str]:
            if loc_data is None or loc_data == "":
                return []
            elif isinstance(loc_data, str):
                return loc_data.strip("/").split("/")
            else:
                return sum((parse_loc(k) for k in loc_data), start=[])

        return self.loc + parse_loc(loc)

    def __getitem__(self, loc: Location) -> Any:
        """Read state or trajectory from storage."""
        loc_list = self._get_loc(loc)
        if self._storage.is_group(loc_list):  # storage is a group  # noqa: SIM102
            if "__class__" not in self._storage._read_attrs(loc_list):
                # group does not contain class information => just return a subgroup
                return StorageGroup(self._storage, loc_list)
        # reconstruct objected stored at this place
        return self.read_item(loc, use_class=True)

    def get(self, loc: Location, default: Any = None) -> Any:
        try:
            return self[loc]
        except KeyError:
            return default

    def __setitem__(self, loc: Location, obj: Any) -> None:
        self.write_item(loc, obj)

    def keys(self) -> Collection[str]:
        """Return name of all stored items in this group."""
        return self._storage.keys(self.loc)

    def __len__(self) -> int:
        return len(self.keys())

    def __iter__(self) -> Iterator[Any]:
        """Iterate over all stored items in this group."""
        for loc in self.keys():
            yield self[loc]

    def __contains__(self, loc: Location):
        """Check wether a particular item is contained in this group."""
        return self._get_loc(loc) in self._storage

    def items(self) -> Iterator[tuple[str, Any]]:
        """Iterate over stored items, yielding the location and item of each."""
        for loc in self.keys():
            yield loc, self[loc]

    def read_attrs(self, loc: Location = None) -> Attrs:
        """Read attributes associated with a particular location.

        Args:
            loc (str or list of str):
                The location in the storage where the attributes are read

        Returns:
            dict: A copy of the attributes at this location
        """
        return self._storage.read_attrs(self._get_loc(loc))

    def write_attrs(self, loc: Location = None, attrs: Attrs | None = None) -> None:
        """Write attributes to a particular location.

        Args:
            loc (str or list of str):
                The location in the storage where the attributes are written
            attrs (dict):
                The attributes to be added to this location
        """
        self._storage.write_attrs(self._get_loc(loc), attrs=attrs)

    @property
    def attrs(self) -> Attrs:
        """dict: the attributes associated with this group"""
        return self.read_attrs()

    def get_class(self, loc: Location = None) -> type | None:
        """Get the class associated with a particular location.

        Class information can be written using the `cls` attribute of `write_array`,
        `write_object`, and similar functions.

        Args:
            loc (str or list of str):
                The location where the class information is read from

        Retruns: the class associated with the lcoation
        """
        loc_list = self._get_loc(loc)
        attrs = self._storage._read_attrs(loc_list)
        return decode_class(attrs.get("__class__"))

    def read_item(self, loc: Location, *, use_class: bool = True) -> Any:
        """Read an item from a particular location.

        Args:
            loc (str or list of str):
                The location where the item is read from
            use_class (bool):
                If `True`, looks for class information in the attributes and evokes a
                potentially registered hook to instantiate the associated object. If
                `False`, only the current data or object is returned.

        Returns:
            The reconstructed python object
        """
        loc_list = self._get_loc(loc)
        if use_class:
            cls = self.get_class(loc)
            if cls is not None:
                # create object using a registered action
                read_item = storage_actions.get(cls, "read_item")
                return read_item(self._storage, loc_list)

        # read the item using the generic classes
        obj_type = self._storage._read_attrs(loc_list).get("__type__")
        if obj_type in {"array", "dynamic_array"}:
            arr = self._storage.read_array(loc_list)
            return Array(arr, attrs=self._storage.read_attrs(loc_list))
        elif obj_type == "object":
            return self._storage.read_object(loc_list)
        else:
            raise RuntimeError(f"Cannot read objects of type `{obj_type}`")

    def write_item(
        self,
        loc: Location,
        item: Any,
        *,
        attrs: Attrs | None = None,
        use_class: bool = True,
    ) -> None:
        """Write an item to a particular location.

        Args:
            loc (sequence of str):
                The location where the item is written to
            item:
                The item that will be written
            attrs (dict, optional):
                Attributes stored with the object
            use_class (bool):
                If `True`, looks for class information in the attributes and evokes a
                potentially registered hook to instantiate the associated object. If
                `False`, only the current data or object is returned.
        """
        # try writing the object using the class definition
        if use_class:
            try:
                write_item = storage_actions.get(item.__class__, "write_item")
            except RuntimeError:
                pass  # fall back to the generic writing
            else:
                loc_list = self._get_loc(loc)
                write_item(self._storage, loc_list, item)
                self._storage._write_attr(
                    loc_list, "__class__", encode_class(item.__class__)
                )
                return

        # write the object using generic writers
        if isinstance(item, np.ndarray):
            self.write_array(loc, item, attrs=attrs)
        else:
            self.write_object(loc, item, attrs=attrs)

    def is_group(self, loc: Location = None) -> bool:
        """Determine whether the location is a group.

        Args:
            loc (sequence of str):
                A list of strings determining the location in the storage

        Returns:
            bool: `True` if the loation is a group
        """
        return self._storage.is_group(self._get_loc(loc))

    def open_group(self, loc: Location) -> StorageGroup:
        """Open an existing group at a particular location.

        Args:
            loc (str or list of str):
                The location where the group will be opened

        Returns:
            :class:`StorageGroup`: The reference to the group
        """
        loc_list = self._get_loc(loc)
        if not self._storage.is_group(loc_list):
            raise TypeError(f"`/{'/'.join(loc_list)}` is not a group")
        return StorageGroup(self._storage, loc_list)

    def create_group(
        self,
        loc: Location,
        *,
        attrs: Attrs | None = None,
        cls: type | None = None,
    ) -> StorageGroup:
        """Create a new group at a particular location.

        Args:
            loc (str or list of str):
                The location where the group will be created
            attrs (dict, optional):
                Attributes stored with the group
            cls (type):
                A class associated with this group

        Returns:
            :class:`StorageGroup`: The reference of the new group
        """
        loc_list = self._get_loc(loc)
        return self._storage.create_group(loc_list, attrs=attrs, cls=cls)

    def read_array(
        self,
        loc: Location,
        *,
        out: np.ndarray | None = None,
        index: int | None = None,
    ) -> np.ndarray:
        """Read an array from a particular location.

        Args:
            loc (str or list of str):
                The location where the array is created
            out (array, optional):
                An array to which the results are written
            index (int, optional):
                An index denoting the subarray that will be read

        Returns:
            :class:`~numpy.ndarray`:
                An array containing the data. Identical to `out` if specified.
        """
        loc_list = self._get_loc(loc)
        return self._storage.read_array(loc_list, out=out, index=index)

    def write_array(
        self,
        loc: Location,
        arr: np.ndarray,
        *,
        attrs: Attrs | None = None,
        cls: type | None = None,
    ):
        """Write an array to a particular location.

        Args:
            loc (str or list of str):
                The location where the array is read
            arr (:class:`~numpy.ndarray`):
                The array that will be written
            attrs (dict, optional):
                Attributes stored with the array
            cls (type):
                A class associated with this array
        """
        loc_list = self._get_loc(loc)
        self._storage.write_array(loc_list, arr, attrs=attrs, cls=cls)

    def create_dynamic_array(
        self,
        loc: Location,
        *,
        arr: np.ndarray | None = None,
        shape: tuple[int, ...] | None = None,
        dtype: DTypeLike = float,
        record_array: bool = False,
        attrs: Attrs | None = None,
        cls: type | None = None,
    ):
        """Creates a dynamic array of flexible size.

        Args:
            loc (str or list of str):
                The location where the dynamic array is created
            shape (tuple of int):
                The shape of the individual arrays. A singular axis is prepended to the
                shape, which can then be extended subsequently.
            dtype:
                The data type of the array to be written
            record_array (bool):
                Flag indicating whether the array is of type :class:`~numpy.recarray`
            attrs (dict, optional):
                Attributes stored with the array
            cls (type):
                A class associated with this array
        """
        if arr is not None:
            if shape is not None:
                raise TypeError("Cannot set `arr` and `shape` simultanously")
            shape = arr.shape
            dtype = arr.dtype
            record_array = isinstance(arr, np.recarray)
        if shape is None:
            raise TypeError("Either `arr` or `shape` need to be specified")

        self._storage.create_dynamic_array(
            self._get_loc(loc),
            shape,
            dtype=dtype,
            record_array=record_array,
            attrs=attrs,
            cls=cls,
        )

    def extend_dynamic_array(self, loc: Location, data: ArrayLike):
        """Extend a dynamic array previously created.

        Args:
            loc (str or list of str):
                The location where the dynamic array is located
            arr (array):
                The array that will be appended to the dynamic array
        """
        self._storage.extend_dynamic_array(self._get_loc(loc), data)

    def read_object(self, loc: Location) -> Any:
        """Read an object from a particular location.

        Args:
            loc (str or list of str):
                The location where the object is created

        Returns:
            The object that has been read from the storage
        """
        return self._storage.read_object(self._get_loc(loc))

    def write_object(
        self,
        loc: Location,
        obj: Any,
        *,
        attrs: Attrs | None = None,
        cls: type | None = None,
    ):
        """Write an object to a particular location.

        Args:
            loc (str or list of str):
                The location where the object is read
            obj:
                The object that will be written
            attrs (dict, optional):
                Attributes stored with the object
            cls (type):
                A class associated with this object
        """
        loc_list = self._get_loc(loc)
        self._storage.write_object(loc_list, obj, attrs=attrs, cls=cls)
