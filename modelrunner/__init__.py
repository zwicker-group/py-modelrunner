"""

.. inheritance-diagram::
<<<<<<< Upstream, based on main
<<<<<<< Upstream, based on main
        io.IOBase
        state.StateBase
        state.ArrayState
        state.ArrayCollectionState
<<<<<<< Upstream, based on main
=======
        _io.IOBase
=======
        io.IOBase
>>>>>>> 58f9ab8 Renamed _io to io
        state.StateBase
        state.ArrayState
>>>>>>> 5b3d6ac More restructuring
=======
>>>>>>> 140ae3e Added ArrayCollectionState
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
<<<<<<< Upstream, based on main
    set_default,
=======
>>>>>>> 45f8d0c Added many tests and adjusted code
)
from .parameters import Parameter
from .results import Result, ResultCollection
from .run import submit_job, submit_jobs
from .state import ArrayState, DictState, ObjectState
from .version import __version__
