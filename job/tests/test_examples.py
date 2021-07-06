"""
.. codeauthor:: David Zwicker <david.zwicker@ds.mpg.de>
"""


import glob
import os
import subprocess as sp
import sys
from pathlib import Path
from typing import List  # @UnusedImport

import pytest

PACKAGEPATH = Path(__file__).parents[2].resolve()
EXAMPLE_PATH = PACKAGEPATH / "examples"


@pytest.mark.no_cover
@pytest.mark.skipif(sys.platform == "win32", reason="Assumes unix setup")
@pytest.mark.parametrize("path", glob.glob(str(EXAMPLE_PATH / "*.py")))
def test_examples(path, tmp_path):
    """runs an example script given by path"""
    # prepare environment
    env = os.environ.copy()
    env["PYTHONPATH"] = str(PACKAGEPATH) + ":" + env.get("PYTHONPATH", "")

    # run example in temporary folder since it might create data files
    proc = sp.Popen(
        [sys.executable, path], cwd=tmp_path, env=env, stdout=sp.PIPE, stderr=sp.PIPE
    )
    outs, errs = proc.communicate(timeout=30)

    msg = "Script `%s` failed with following output:" % path
    if outs:
        msg = "%s\nSTDOUT:\n%s" % (msg, outs)
    if errs:
        msg = "%s\nSTDERR:\n%s" % (msg, errs)

    assert proc.returncode <= 0, msg
