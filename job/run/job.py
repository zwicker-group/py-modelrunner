"""
.. codeauthor:: David Zwicker <david.zwicker@ds.mpg.de>
"""

import errno
import json
import os
import pipes
import subprocess as sp
from pathlib import Path
from typing import Any, Dict, Tuple, Union


def escape_string(obj) -> str:
    """escape a string for the command line"""
    return pipes.quote(str(obj))


def ensure_directory_exists(folder):
    """creates a folder if it not already exists"""
    if folder == "":
        return
    try:
        os.makedirs(folder)
    except OSError as err:
        if err.errno != errno.EEXIST:
            raise


def submit_job(
    script: Union[str, Path],
    output: Union[str, Path],
    name: str = "job",
    parameters: Union[str, Dict[str, Any]] = None,
    logfolder: Union[str, Path] = "logs",
    method: str = "qsub",
    template: Union[str, Path] = None,
    overwrite_files: bool = False,
) -> Tuple[str, str]:
    """submit a script to the cluster queue

    Args:
        script (str of :class:`~pathlib.Path`): Path to the script file
        output (str of :class:`~pathlib.Path`): Path to the output file
        name (str): Name of the job
        parameters: Parameters for the script
        logfolder (str of :class:`~pathlib.Path`): Path to the logging folder
        method (str): Submission method. Currently `qsub` and `local` are supported
        template (str of :class:`~pathlib.Path`): Template file for submission script
        overwrite_files (bool): Determines whether output files are overwritten\
        
    Returns:
        tuple: The result `(stdout, stderr)` of the submission call
    """
    if template is None:
        template_path = Path(__file__).parent / "templates" / (method + ".template")
    else:
        template_path = Path(template)
    with open(template_path, "r") as fp:
        script_template = fp.read()

    # prepare submission script
    ensure_directory_exists(logfolder)

    script_args = {
        "LOG_FOLDER": logfolder,
        "JOB_NAME": name,
        "MODEL_FILE": escape_string(script),
    }

    job_args = []
    if parameters is not None:
        if isinstance(parameters, dict):
            parameters = json.dumps(parameters)
        elif not isinstance(parameters, str):
            raise TypeError("Parameters need to be given as a string or a dict")
        job_args.append(f"--json {escape_string(parameters)}")

    if output:
        output = Path(output)
        if output.is_file() and not overwrite_files:
            raise RuntimeError(f"Output file `{output}` already exists")
        script_args["OUTPUT_FOLDER"] = pipes.quote(str(output.parent))
        job_args.append(f"--output {escape_string(output)}")
    else:
        script_args["OUTPUT_FOLDER"] = "."
    script_args["JOB_ARGS"] = " ".join(job_args)
    script = script_template.format(**script_args)

    # submit job to queue
    if method == "qsub":
        proc = sp.Popen(
            ["qsub"],
            stdin=sp.PIPE,
            stdout=sp.PIPE,
            stderr=sp.PIPE,
            universal_newlines=True,
        )

    elif method == "local":
        proc = sp.Popen(
            ["bash"],
            stdin=sp.PIPE,
            stdout=sp.PIPE,
            stderr=sp.PIPE,
            universal_newlines=True,
        )

    else:
        raise ValueError(f"Unknown submit method `{method}`")

    return proc.communicate(script)
