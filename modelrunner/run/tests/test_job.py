"""
.. codeauthor:: David Zwicker <david.zwicker@ds.mpg.de>
"""

from pathlib import Path

import numpy as np

from ... import Result, ResultCollection
from ..job import submit_job, submit_jobs

SCRIPT_PATH = Path(__file__).parent / "scripts"


def test_submit_job(tmp_path):
    """test some basic usage of the submit_job function"""

    def run(**p):
        """helper submitting job locally"""
        output = tmp_path / "output.yaml"
        submit_job(
            SCRIPT_PATH / "function.py",
            output,
            parameters=p,
            log_folder=tmp_path,
            method="foreground",
            overwrite_strategy="silent_overwrite",
        )
        return Result.from_file(output).result

    assert run()["a"] == 1
    assert run(a=2)["a"] == 2
    assert run(b=[1, 2, 3])["b"] == [1, 2, 3]


def test_submit_jobs(tmp_path):
    """test some edge cases of the submit_jobs function"""

    def run(parameters, **kwargs):
        """helper submitting job locally"""
        submit_jobs(
            SCRIPT_PATH / "function.py",
            tmp_path,
            parameters=parameters.copy(),
            log_folder=tmp_path,
            method="foreground",
            overwrite_strategy="silent_overwrite",
            **kwargs,
        )

        # read result
        col = ResultCollection.from_folder(tmp_path).dataframe

        # delete temporary files
        for path in tmp_path.iterdir():
            path.unlink()

        return col

    res = run({"a": (1, 2)})["a"]
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
    """test some basic usage of the submit_job function"""

    def run(**p):
        """helper submitting job locally"""
        submit_job(
            SCRIPT_PATH / "script.py",
            parameters=p,
            log_folder=tmp_path,
            method="foreground",
            use_modelrunner=False,
            overwrite_strategy="silent_overwrite",
        )
        out = open(tmp_path / "job.out.txt").read()
        err = open(tmp_path / "job.err.txt").read()
        return out, err

    assert run() == ("", "")
    assert run(a=1) == ('--json{"a": 1}', "")
