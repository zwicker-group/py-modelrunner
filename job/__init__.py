"""
.. codeauthor:: David Zwicker <david.zwicker@ds.mpg.de>
"""

from .model import ModelBase, get_function_model, make_model, run_function_with_cmd_args
from .parameters import Parameter
from .results import Result, ResultCollection
from .run import submit_job
