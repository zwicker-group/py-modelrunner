"""
.. codeauthor:: David Zwicker <david.zwicker@ds.mpg.de>
"""

import platform
from pathlib import Path

import pytest

import modelrunner as mr

SCRIPT_PATH = Path(__file__).parent / "scripts"
assert SCRIPT_PATH.is_dir()


@pytest.mark.skipif(
    platform.system() == "Windows", reason="submit_jobs has issues on windows"
)
def test_modelrunner_storage_many(tmp_path):
    """Test how modelrunner storage can be used."""
    num_jobs = mr.submit_jobs(
        SCRIPT_PATH / "function.py",
        output_folder=tmp_path,
        parameters={"a": 3.0, "b": [1.5, 2.5]},
        log_folder=tmp_path,
        method="foreground",
        overwrite_strategy="error",
    )
    assert num_jobs == 2

    # read storage manually
    for path in tmp_path.iterdir():
        if path.is_file() and not path.name.endswith("txt"):
            with mr.open_storage(path, mode="read") as storage:
                assert "result" in storage
                assert storage["result"]["a"] == 3.0
                assert storage["result"]["b"] in [1.5, 2.5]

    # read result using ResultCollection
    results = mr.ResultCollection.from_folder(tmp_path)
    assert len(results) == num_jobs
    for result in results:
        assert result.result["a"] == 3.0
        assert result.result["b"] in [1.5, 2.5]

    # delete temporary files
    for path in tmp_path.iterdir():
        path.unlink()
