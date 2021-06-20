"""
.. codeauthor:: David Zwicker <david.zwicker@ds.mpg.de>
"""

import errno
import json
import os
import pipes
import subprocess as sp
from pathlib import Path
from typing import Any, Dict, Union


def escape_string(obj) -> str:
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
    script,
    output,
    name="job",
    parameters: Union[str, Dict[str, Any]] = None,
    logfolder="logs",
    method="qsub",
):
    """submit a script to the cluster queue"""
    template_path = Path(__file__).parent / "templates" / (method + ".template")
    with open(template_path, "r") as fp:
        script_template = fp.read()

    # prepare submission script
    ensure_directory_exists(logfolder)

    script_args = {
        "LOG_FOLDER": logfolder,
        "JOB_NAME": name,
        "MODEL_FILE": escape_string(script),
    }

    if parameters is None:
        job_args = ""
    else:
        if isinstance(parameters, dict):
            parameters = json.dumps(parameters)
        elif not isinstance(parameters, str):
            raise TypeError("Parameters need to be given as a string or a dict")
        job_args = f"--json {escape_string(parameters)}"

    if output:
        output = Path(output)
        script_args["OUTPUT_FOLDER"] = pipes.quote(str(output.parent))
        job_args += f" --output {escape_string(output)}"
    else:
        script_args["OUTPUT_FOLDER"] = "."
    script_args["JOB_ARGS"] = job_args
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
