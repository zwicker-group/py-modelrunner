"""Contains code necessary for loading results from format version 2.

.. codeauthor:: David Zwicker <david.zwicker@ds.mpg.de>
"""

from __future__ import annotations

from pathlib import Path

from ...storage import open_storage
from ..results import Result


def result_from_file_v2(store: Path, *, loc: str = "result", **kwargs) -> Result:
    """Load object from a file using format version 1.

    Args:
        store (Path):
            Path of file to read
        fmt (str):
            Explicit file format. Determined from `store` if omitted.
        label (str):
            Name of the node in which the data was stored. This applies to some
            hierarchical storage formats.
    """
    # assume that file was written with latest format version
    with open_storage(store, mode="read") as storage_obj:
        attrs = storage_obj.read_attrs(loc)
        format_version = attrs.pop("format_version", None)
        if format_version == 2:
            # current version of storing results
            if "data" in storage_obj:
                data_storage = open_storage(storage_obj, loc="data", mode="read")
            else:
                data_storage = None
            return Result.from_data(
                model_data=attrs.get("model", {}),
                result=storage_obj.read_item(loc, use_class=False),
                storage=data_storage,
                info=attrs.pop("info", {}),  # load additional info,
            )

        else:
            raise RuntimeError(f"Cannot read format version {format_version}")
