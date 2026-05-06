"""Defines a class storing data in various storages.

Requires the optional :mod:`zarr` module.

.. codeauthor:: David Zwicker <david.zwicker@ds.mpg.de>
"""

from __future__ import annotations

import zarr

zarr_version = int(zarr.__version__.split(".", 1)[0])

if zarr_version == 2:
    # import classes from paths of zarr version 2
    from ._zarr2 import ZarrStorage

elif zarr_version == 3:
    # import classes from paths of zarr version 3
    from ._zarr3 import ZarrStorage
