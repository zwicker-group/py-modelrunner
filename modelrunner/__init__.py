"""
.. codeauthor:: David Zwicker <david.zwicker@ds.mpg.de>
"""

# determine the package version
try:
    # try reading version of the automatically generated module
    from ._version import __version__
except ImportError:
    # determine version automatically from CVS information
    from importlib.metadata import PackageNotFoundError, version

    try:
        __version__ = version("modelrunner")
    except PackageNotFoundError:
        # package is not installed, so we cannot determine any version
        __version__ = "unknown"
    del PackageNotFoundError, version  # clean name space


from .model import *
from .run import *
from .storage import open_storage
