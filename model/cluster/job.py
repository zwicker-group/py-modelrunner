"""
.. codeauthor:: David Zwicker <david.zwicker@ds.mpg.de>
"""

import errno
import os
import pipes
import subprocess as sp
from pathlib import Path


def ensure_directory_exists(folder):
    """creates a folder if it not already exists"""
    if folder == "":
        return
    try:
        os.makedirs(folder)
    except OSError as err:
        if err.errno != errno.EEXIST:
            raise


def submit_job(script, output, name, parameters, logfolder="logs", template="qsub"):
    """submit a script to the cluster queue"""
    template_path = Path(__file__).parent / "templates" / (template + ".template")
    with open(template_path, "r") as fp:
        script_template = fp.read()

    # prepare submission script
    output = Path(output)
    script_args = {
        "LOG_FOLDER": logfolder,
        "JOB_NAME": name,
        "OUTPUT_FOLDER": pipes.quote(output.parent),
        "MODEL_FILE": pipes.quote(script),
        "OUTPUT": pipes.quote(output),
        "PARAMETERS": pipes.quote(parameters),
    }
    script = script_template.format(**script_args)

    # submit job to queue
    ensure_directory_exists(logfolder)
    proc = sp.Popen(
        ["qsub"], stdin=sp.PIPE, stdout=sp.PIPE, stderr=sp.PIPE, universal_newlines=True
    )
    return proc.communicate(script)
