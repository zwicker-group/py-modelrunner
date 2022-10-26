"""

.. inheritance-diagram::
        io.IOBase
        state.StateBase
        state.ArrayState
        state.ArrayCollectionState
        state.ObjectState
        state.DictState
        results.Result
   :parts: 1

.. codeauthor:: David Zwicker <david.zwicker@ds.mpg.de>
"""

from .model import (
    ModelBase,
    make_model,
    make_model_class,
    run_function_with_cmd_args,
    run_script,
    set_default,
)
from .parameters import Parameter
from .results import Result, ResultCollection
from .run import submit_job, submit_jobs
from .state import ArrayState, DictState, ObjectState
from .version import __version__
