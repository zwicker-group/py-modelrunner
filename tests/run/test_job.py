"""
.. codeauthor:: David Zwicker <david.zwicker@ds.mpg.de>
"""

from pathlib import Path

import numpy as np
import pytest

from modelrunner import Result, ResultCollection
from modelrunner.run.job import submit_job, submit_jobs

SCRIPT_PATH = Path(__file__).parent / "scripts"
assert SCRIPT_PATH.is_dir()


def test_submit_job(tmp_path, capsys):
    """Test some basic usage of the submit_job function."""

    def run(**p):
        """Helper submitting job locally."""
        output = tmp_path / "output.json"
        submit_job(
            SCRIPT_PATH / "function.py",
            output,
            parameters=p,
            log_folder=tmp_path,
            method="foreground",
            overwrite_strategy="silent_overwrite",
        )
        return Result.from_file(output)

    assert run().result["a"] == 1
    assert run(a=2).result["a"] == 2
    assert run(b=[1, 2, 3]).result["b"] == [1, 2, 3]
    std = capsys.readouterr()
    assert std.out == std.err == ""


@pytest.mark.parametrize("method", ["foreground", "background"])
def test_submit_job_fail(method):
    """Test some basic usage of the submit_job function."""
    outs, errs = submit_job(SCRIPT_PATH / "fail.py", method=method)
    assert outs == ""
    assert "Traceback" in errs


@pytest.mark.parametrize("method", ["foreground", "background"])
def test_submit_job_stdout(tmp_path, method):
    """Test logging to stdout for the submit_job function."""

    output = tmp_path / "output.json"
    outs, errs = submit_job(
        SCRIPT_PATH / "print.py",
        output,
        method=method,
        overwrite_strategy="silent_overwrite",
    )

    assert errs == ""
    assert outs == "3.0\n"
    assert Result.from_file(output).result is None


def test_submit_job_no_output():
    """Test logging to stdout for the submit_job function."""
    outs, errs = submit_job(
        SCRIPT_PATH / "print.py",
        method="foreground",
        overwrite_strategy="silent_overwrite",
    )
    assert errs == ""
    assert outs == "3.0\n"


def test_submit_jobs(tmp_path):
    """Test some edge cases of the submit_jobs function."""

    def run(parameters, **kwargs):
        """Helper submitting job locally."""
        num_jobs = submit_jobs(
            SCRIPT_PATH / "function.py",
            tmp_path,
            parameters=parameters.copy(),
            log_folder=tmp_path,
            method="foreground",
            overwrite_strategy="silent_overwrite",
            **kwargs,
        )

        # read result
        col = ResultCollection.from_folder(tmp_path).as_dataframe()
        assert len(col) == num_jobs

        # delete temporary files
        for path in tmp_path.iterdir():
            path.unlink()

        return col

    df = run({"a": (1, 2)})
    print(df)
    res = df["a"]
    # the order of the results might not be deterministic => sort result
    np.testing.assert_allclose(np.sort(res), [1, 2])

    res = run({"b": [[1, 2], [3, 4]]})["b"]
    # the order of the results might not be deterministic => test both variants
    test1 = np.allclose(res[0], [1, 2]) and np.allclose(res[1], [3, 4])
    test2 = np.allclose(res[0], [3, 4]) and np.allclose(res[1], [1, 2])
    assert test1 or test2

    res = run({"b": [1, 2]}, list_params=["b"])
    # the order of the results might not be deterministic => test both variants
    assert len(res) == 1
    print(res)
    assert res["a"][0] == 1
    np.testing.assert_allclose(res["b"][0], [1, 2])


def test_submit_job_no_modelrunner(tmp_path):
    """Test some basic usage of the submit_job function."""

    def run(**p):
        """Helper submitting job locally."""
        submit_job(
            SCRIPT_PATH / "script.py",
            parameters=p,
            log_folder=tmp_path,
            method="foreground",
            use_modelrunner=False,
            overwrite_strategy="silent_overwrite",
        )
        out = (tmp_path / "job.out.txt").open().read()
        err = (tmp_path / "job.err.txt").open().read()
        return out, err

    assert run() == ("", "")
    assert run(a=1) == ('--json{"a": 1}', "")


def test_submit_job_own_template(tmp_path):
    """Test the submit_job function with a custom template."""
    outs, errs = submit_job(
        SCRIPT_PATH / "print.py",
        method="foreground",
        parameters={"a": 5, "b": 10},  # b is not used by template
        template=SCRIPT_PATH / "custom.jinja",
        overwrite_strategy="silent_overwrite",
    )

    assert errs == ""
    assert outs == "7.0\n"  # a + 2 (default value of b)
