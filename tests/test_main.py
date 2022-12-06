"""
.. codeauthor:: David Zwicker <david.zwicker@ds.mpg.de>
"""

import os
import subprocess as sp
import sys
from pathlib import Path

PACKAGEPATH = Path(__file__).parents[2].resolve()


def test_empty_main():
    """run a script (with potential arguments) and collect stdout"""
    # prepare environment
    env = os.environ.copy()
    env["PYTHONPATH"] = str(PACKAGEPATH) + ":" + env.get("PYTHONPATH", "")

    # run example in temporary folder since it might create data files
    cmd_args = (sys.executable, "-m", "modelrunner")
    proc = sp.Popen(cmd_args, env=env, stdout=sp.PIPE, stderr=sp.PIPE)
    _, errs = proc.communicate(timeout=30)
    assert b"Require job script as first argument" in errs
