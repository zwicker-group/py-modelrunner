"""
Helper functions for dealing with storage

.. codeauthor:: David Zwicker <david.zwicker@ds.mpg.de>
"""

import importlib
from typing import Any, Callable, List, Optional, Sequence, TypeVar, Union

import numpy as np
import pytest

TFunc = TypeVar("TFunc", bound=Callable[..., Any])


def module_available(module_name: str) -> bool:
    """check whether a python module is available

    Args:
        module_name (str):
            The name of the module

    Returns:
        `True` if the module can be imported and `False` otherwise
    """
    try:
        importlib.import_module(module_name)
    except ImportError:
        return False
    else:
        return True


def skipUnlessModule(
    module_names: Union[Sequence[str], str]
) -> Callable[[TFunc], TFunc]:
    """decorator that skips a test when a module is not available

    Args:
        module_names (str):
            The name of the required module(s)

    Returns:
        A function, so this can be used as a decorator
    """
    if isinstance(module_names, str):
        module_names = [module_names]

    for module_name in module_names:
        if not module_available(module_name):
            # return decorator skipping test
            return pytest.skip(f"requires {module_name}")

    # return no-op decorator if all modules are available
    def wrapper(f: TFunc) -> TFunc:
        return f

    return wrapper


def storage_extensions(
    incl_folder: bool = True, dot: bool = False, *, exclude: Optional[List] = None
):
    """determine the extensions for storage objects

    Args:
        incl_folder (bool):
            Indicates whether extensions that typically designate folders are included
        dot (bool):
            Indicates whether the returned extensions are prepended with a dot (`.`)
        exclude (list, optional):
            Extensions (without dots) that should be explicitely excluded

    Returns:
        sequence: sorted list of extensions
    """
    exts = ["json"]
    if module_available("yaml"):
        exts.append("yaml")
    if module_available("h5py"):
        exts.append("hdf")
    if module_available("zarr"):
        exts.append("zip")
        if incl_folder:
            exts.extend(["", "zarr"])
        if module_available("sqlite3"):
            exts.append("sqldb")

    if dot:
        exts = ["." + s for s in exts]
    if exclude is not None:
        exts = set(exts) - set(exclude)
    return sorted(exts)


STORAGE_OBJECTS = [
    {"n": -1, "s": "t", "l1": [0, 1, 2], "l2": [[0, 1], [4]], "a": np.arange(5)},
    np.arange(3),
    [np.arange(2), np.arange(3)],
    {"a": {"a", "b"}, "b": np.arange(3)},
]
STORAGE_EXT = storage_extensions(incl_folder=True, dot=True)
