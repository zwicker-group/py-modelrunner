"""Defines classes and function used to run models defined using
:mod:`~modelrunner.model`

.. autosummary::
   :nosignatures:

   ~modelrunner.run.job.submit_job
   ~modelrunner.run.job.submit_jobs
   ~modelrunner.run.launch.run_function_with_cmd_args
   ~modelrunner.run.launch.run_script
   ~modelrunner.run.results.Result
   ~modelrunner.run.results.ResultCollection

.. codeauthor:: David Zwicker <david.zwicker@ds.mpg.de>
"""

from .job import submit_job, submit_jobs
from .launch import run_function_with_cmd_args, run_script
from .results import Result, ResultCollection
