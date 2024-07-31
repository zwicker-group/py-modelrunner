"""
.. codeauthor:: David Zwicker <david.zwicker@ds.mpg.de>
"""

from modelrunner.storage import MemoryStorage, StorageGroup


def test_group_location():
    """Test location managment."""
    group = StorageGroup(MemoryStorage())
    assert group._get_loc(None) == []
    assert group._get_loc("a") == ["a"]
    assert group._get_loc("abc") == ["abc"]
    assert group._get_loc("/a") == ["a"]
    assert group._get_loc("a/") == ["a"]
    assert group._get_loc("/a/") == ["a"]
    assert group._get_loc("a/b") == ["a", "b"]
    assert group._get_loc("/a/b/") == ["a", "b"]
    assert group._get_loc(["a", "b"]) == ["a", "b"]
    assert group._get_loc(["a", ["b"]]) == ["a", "b"]
    assert group._get_loc(["a", ["b/c"]]) == ["a", "b", "c"]
    assert group._get_loc([["a/b"], "c"]) == ["a", "b", "c"]
    assert group._get_loc([["a/b"], "", [], "c", None, [None]]) == ["a", "b", "c"]


def test_group_tree():
    """Test printing a tree."""
    group = StorageGroup(MemoryStorage())
    group["a/b/c"] = 1
    group["a/d"] = 2
    group["f/d"] = 2
    group.tree()
    group["a"].tree()
