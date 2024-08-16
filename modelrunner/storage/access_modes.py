"""
.. codeauthor:: David Zwicker <david.zwicker@ds.mpg.de>
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import ClassVar, Literal, Union

FileMode = Literal[
    "r",  # open as readable
    "x",  # open as extensible (read and write)
    "a",  # open to append (read and write)
    "w",  # open as writeable and truncate
]


@dataclass(frozen=True, repr=False)
class AccessMode:
    """Determines access modes for storages."""

    name: str  # identifier of the access role; can be used instead of the object
    description: str  # human-readable description of the mode
    file_mode: FileMode  # how to open files
    read: bool = False  # allow reading data
    set_attrs: bool = False  # allows setting attributes
    insert: bool = False  # allow inserting new items
    overwrite: bool = False  # allow overwriting existing data
    dynamic_append: bool = False  # allow appending to dynamic arrays

    _defined: ClassVar[dict[str, AccessMode]] = {}  # all defined access modes

    def __repr__(self):
        return f"AccessMode(name={self.name})"

    def __post_init__(self):
        # register the access mode, so we can construct it from its name
        if self.name is self._defined:
            warnings.warn(f"Overwriting access mode `{self.name}`")
        self._defined[self.name] = self

    @classmethod
    def parse(cls, obj_or_name: str | AccessMode) -> AccessMode:
        """Gets access mode from various formats.

        Args:
            obj_or_name (str or :class:`AccessMode`):
                An :class:`AccessMode` object or the name of a registered access mode

        Returns:
            :class:`AccessMode`: the access mode object
        """
        if isinstance(obj_or_name, AccessMode):
            return obj_or_name
        elif isinstance(obj_or_name, str):
            if obj_or_name == "closed":
                raise ValueError("Cannot use `closed` access mode.")
            try:
                return cls._defined[obj_or_name]
            except KeyError as err:
                raise ValueError(
                    f"Access mode '{obj_or_name}' not in {list(cls._defined.keys())}"
                ) from err
        else:
            raise TypeError(f"Unsupported type '{obj_or_name}'")


# define default access modes
_access_closed = AccessMode(
    name="closed", description="Does not allow anything", file_mode="r", read=False
)
access_read = AccessMode(
    name="read",
    description="Only allows reading",
    file_mode="r",
    read=True,
)
access_exclusive = AccessMode(
    name="exclusive",
    description="Creates new file, allowing read and write access",
    file_mode="x",
    read=True,
    set_attrs=True,
    insert=True,
    overwrite=True,
    dynamic_append=True,
)
access_insert = AccessMode(
    name="insert",
    description="Allows inserting new items, but not changing existing items",
    file_mode="a",
    read=True,
    set_attrs=True,
    insert=True,
    overwrite=False,
    dynamic_append=True,
)
access_overwrite = AccessMode(
    name="overwrite",
    description="Allows changing existing items, but not inserting new items",
    file_mode="a",
    read=True,
    set_attrs=True,
    insert=False,
    overwrite=True,
    dynamic_append=True,
)
access_full = AccessMode(
    name="full",
    description="Allows changing existing items and inserting new items",
    file_mode="a",
    read=True,
    set_attrs=True,
    insert=True,
    overwrite=True,
    dynamic_append=True,
)
access_append = AccessMode(
    name="append",
    description="Only allows appending to already existing dynamic arrays",
    file_mode="a",
    read=True,
    set_attrs=True,
    insert=False,
    overwrite=False,
    dynamic_append=True,
)
access_truncate = AccessMode(
    name="truncate",
    description="Removes all old items and allows inserting and changing new items",
    file_mode="w",
    read=True,
    set_attrs=True,
    insert=True,
    overwrite=True,
    dynamic_append=True,
)

ModeType = Union[str, AccessMode]


class AccessError(RuntimeError):
    """An error indicating that an access credential was not present."""
