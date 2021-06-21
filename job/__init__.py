"""
.. codeauthor:: David Zwicker <david.zwicker@ds.mpg.de>
"""

from .model import (
    FunctionModelFactory,
    ModelBase,
    get_function_model,
    run_function_with_cmd_args,
)
from .parameters import Parameter
from .results import Result, ResultCollection
