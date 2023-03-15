"""

.. inheritance-diagram::
        state.io.IOBase
        state.base.StateBase
        state.array.ArrayState
        state.array_collection.ArrayCollectionState
        state.object.ObjectState
        state.dict.DictState
        results.Result
   :parts: 1

.. codeauthor:: David Zwicker <david.zwicker@ds.mpg.de>
"""


# determine the package version
try:
    # try reading version of the automatically generated module
    from ._version import __version__  # type: ignore
except ImportError:
    # determine version automatically from CVS information
    from importlib.metadata import PackageNotFoundError, version

    try:
        __version__ = version("modelrunner")
    except PackageNotFoundError:
        # package is not installed, so we cannot determine any version
        __version__ = "unknown"
    del version, PackageNotFoundError  # clean name space


from .model import (
    ModelBase,
    make_model,
    make_model_class,
    run_function_with_cmd_args,
    run_script,
    set_default,
)
from .parameters import HideParameter, Parameter, Parameterized
from .results import Result, ResultCollection
from .run import submit_job, submit_jobs
from .state import (
    ArrayCollectionState,
    ArrayState,
    DictState,
    ObjectState,
    load_state,
    make_state,
)
