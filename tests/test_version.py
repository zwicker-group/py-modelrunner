"""
.. codeauthor:: David Zwicker <david.zwicker@ds.mpg.de>
"""

import modelrunner


def test_version():
    assert isinstance(modelrunner.__version__, str)
