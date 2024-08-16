"""
.. codeauthor:: David Zwicker <david.zwicker@ds.mpg.de>
"""

import pytest

from modelrunner import Parameter
from modelrunner.config import Config


def test_config():
    """Test configuration system."""
    conf1 = Config([Parameter("a", 1)])
    conf2 = conf1.copy()

    for c in [conf1, conf2]:
        assert c["a"] == 1
        assert c.mode == "update"
        assert len(c._default) == 1

        assert "a" in c
        assert any(k == "a" for k in c)
        assert any(k == "a" and v > 0 for k, v in c.items())
        assert "a" in c.to_dict()
        assert isinstance(repr(c), str)

    assert conf1 is not conf2
    conf2["a"] = 2
    assert conf1["a"] == 1
    assert conf2["a"] == 2


def test_config_modes():
    """Test configuration system running in different modes."""
    c = Config([Parameter("a", 1)], mode="insert")
    assert c["a"] > 0
    c["a"] = 0
    assert c["a"] == 0
    c["new_value"] = "value"
    assert c["new_value"] == "value"
    del c["new_value"]
    with pytest.raises(KeyError):
        c["new_value"]
    with pytest.raises(KeyError):
        c["undefined"]

    c = Config([Parameter("a", 1)], mode="update")
    assert c["a"] > 0
    c["a"] = 0

    with pytest.raises(KeyError):
        c["new_value"] = "value"
    with pytest.raises(RuntimeError):
        del c["a"]
    with pytest.raises(KeyError):
        c["undefined"]

    c = Config([Parameter("a", 1)], mode="locked")
    assert c["a"] > 0
    with pytest.raises(RuntimeError):
        c["a"] = 0
    with pytest.raises(RuntimeError):
        c["new_value"] = "value"
    with pytest.raises(RuntimeError):
        del c["a"]
    with pytest.raises(KeyError):
        c["undefined"]

    c = Config([Parameter("a", 1)], mode="undefined")
    assert c["a"] > 0
    with pytest.raises(ValueError):
        c["a"] = 0
    with pytest.raises(RuntimeError):
        del c["a"]


def test_config_contexts():
    """Test context manager temporarily changing configuration."""
    c = Config([Parameter("a", 10)])
    assert c["a"] == 10

    with c({"a": 0}):
        assert c["a"] == 0
        with c({"a": 1}):
            assert c["a"] == 1
        assert c["a"] == 0
    assert c["a"] == 10

    with c(a=0):
        assert c["a"] == 0
        with c(a=1):
            assert c["a"] == 1
        assert c["a"] == 0
    assert c["a"] == 10


def test_config_io(tmp_path):
    """Test configuration system."""
    path = tmp_path / "config.yaml"
    conf1 = Config([Parameter("a", 1)])
    conf2 = conf1.copy()
    conf1["a"] = 3
    conf1.save(path)

    assert conf2["a"] == 1
    conf2.load(path)
    assert conf2["a"] == 3
