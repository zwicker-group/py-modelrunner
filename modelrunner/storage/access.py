"""
.. codeauthor:: David Zwicker <david.zwicker@ds.mpg.de>
"""
from __future__ import annotations

from typing import Literal, Union, Dict
from dataclasses import dataclass


# TODO: Rename `access` to `mode`?

@dataclass(frozen=True)
class Access:
    """Determines access modes for storages"""

    name: str
    description: str
    file_mode: Literal["r", "x", "w"]
    read: bool  # allow reading data
    overwrite: bool  # allow overwriting existing data
    insert: bool  # allow inserting new items

    def __post_init__(self):
        _access_objects[self.name] = self

    @classmethod
    def parse(cls, obj_or_name: Union[str, Access]) -> Access:
        if isinstance(obj_or_name, Access):
            return obj_or_name
        elif isinstance(obj_or_name, str):
            return _access_objects[obj_or_name]
        else:
            raise TypeError(f"Unsupported type `{obj_or_name}`")


_access_objects: Dict[str, Access] = {}


AccessType = Union[str, Access]


access_readonly = Access(
    name="readonly",
    description="Only allows reading",
    file_mode="r",
    read=True,
    overwrite=False,
    insert=False,
)
access_insert = Access(
    name="insert",
    description="Allows inserting new items, but not changing existing items",
    file_mode="x",
    read=True,
    overwrite=False,
    insert=True,
)
access_overwrite = Access(
    name="overwrite",
    description="Allows changing existing items, but not inserting new items",
    file_mode="x",
    read=True,
    overwrite=True,
    insert=False,
)
access_full = Access(
    name="full",
    description="Allows changing existing items and inserting new items",
    file_mode="x",
    read=True,
    overwrite=True,
    insert=True,
)
access_truncate = Access(
    name="truncate",
    description="Removes all old items and allows inserting and changing new items",
    file_mode="w",
    read=True,
    overwrite=True,
    insert=True,
)


class AccessError(RuntimeError):
    ...
