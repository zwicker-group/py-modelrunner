"""
.. codeauthor:: David Zwicker <david.zwicker@ds.mpg.de>
"""

import os
import subprocess as sp
import sys
from pathlib import Path

PACKAGEPATH = Path(__file__).parents[1].resolve()
assert PACKAGEPATH.is_dir()
SCRIPT_PATH = Path(__file__).parent / "scripts"
assert SCRIPT_PATH.is_dir()


def test_run_main_help():
    """Test the __main__ module help system."""
    # prepare environment
    env = os.environ.copy()
    env["PYTHONPATH"] = str(PACKAGEPATH) + ":" + env.get("PYTHONPATH", "")

    # run example in temporary folder since it might create data files
    cmd_args = (sys.executable, "-m", "modelrunner.run", "--help")
    proc = sp.Popen(cmd_args, env=env, stdout=sp.PIPE, stderr=sp.PIPE)
    outs, errs = proc.communicate(timeout=30)

    assert "parameters" in str(outs)
    assert errs == b""


def test_run_main_without_log():
    """Test running script with the __main__ module system."""
    # prepare environment
    env = os.environ.copy()
    env["PYTHONPATH"] = str(PACKAGEPATH) + ":" + env.get("PYTHONPATH", "")

    # run example in temporary folder since it might create data files
    cmd_args = (
        sys.executable,
        "-m",
        "modelrunner.run",
        SCRIPT_PATH / "print.py",
        "--parameters",
        '{"a": 2, "b": 3}',
        "--method",
        "foreground",
    )
    proc = sp.Popen(cmd_args, env=env, stdout=sp.PIPE, stderr=sp.PIPE)
    outs, errs = proc.communicate(timeout=30)

    assert outs.strip() == b"5.0"
    assert errs.strip() == b""


def test_run_main_with_log(tmp_path):
    """Test running script with the __main__ module system."""
    # prepare environment
    env = os.environ.copy()
    env["PYTHONPATH"] = str(PACKAGEPATH) + ":" + env.get("PYTHONPATH", "")

    # run example in temporary folder since it might create data files
    cmd_args = (
        sys.executable,
        "-m",
        "modelrunner.run",
        SCRIPT_PATH / "print.py",
        "--parameters",
        '{"a": 2, "b": 3}',
        "--method",
        "foreground",
        "--log_folder",
        tmp_path,
    )
    proc = sp.Popen(cmd_args, env=env, stdout=sp.PIPE, stderr=sp.PIPE)
    outs, errs = proc.communicate(timeout=30)

    assert outs.strip() == errs.strip() == b""
    assert (tmp_path / "job.out.txt").open().read().strip() == "5.0"
    assert (tmp_path / "job.err.txt").open().read().strip() == ""
