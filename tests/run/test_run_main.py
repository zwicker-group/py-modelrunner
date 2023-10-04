"""
.. codeauthor:: David Zwicker <david.zwicker@ds.mpg.de>
"""

import os
import subprocess as sp
import sys
from pathlib import Path

PACKAGEPATH = Path(__file__).parents[1].resolve()
assert PACKAGEPATH.is_dir()


def test_run_main_help():
    """test the __main__ module help system"""
    # prepare environment
    env = os.environ.copy()
    env["PYTHONPATH"] = str(PACKAGEPATH) + ":" + env.get("PYTHONPATH", "")

    # run example in temporary folder since it might create data files
    cmd_args = (sys.executable, "-m", "modelrunner.run", "--help")
    proc = sp.Popen(cmd_args, env=env, stdout=sp.PIPE, stderr=sp.PIPE)
    outs, errs = proc.communicate(timeout=30)

    assert "parameters" in str(outs)
    assert errs == b""
