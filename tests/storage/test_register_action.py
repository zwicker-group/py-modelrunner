"""
.. codeauthor:: David Zwicker <david.zwicker@ds.mpg.de>
"""

import pytest

from helpers import STORAGE_EXT, STORAGE_OBJECTS, assert_data_equals
from modelrunner.storage import StorageGroup, open_storage, storage_actions


class A:
    def __init__(self, value):
        self.value = value
        self.loaded = False

    def save(self, storage, loc):
        storage.write_object(loc, self.value, cls=self.__class__)

    @classmethod
    def load(cls, storage, loc):
        obj = cls(storage.read_object(loc))
        obj.loaded = True
        return obj


def save_A(storage, loc, obj):
    obj.save(storage, loc)


storage_actions.register("read_item", A, A.load, inherit=True)
storage_actions.register("write_item", A, save_A, inherit=True)


class A1(A): ...


class A2(A):
    @classmethod
    def load(cls, storage, loc):
        obj = cls(storage.read_object(loc))
        obj.loaded = "yes"
        return obj


storage_actions.register("read_item", A2, A2.load, inherit=False)


class B:
    def __init__(self, a, b):
        self.a = a
        self.b = b
        self.loaded = False

    def save(self, storage, loc):
        group = storage.create_group(loc, cls=self.__class__)
        self.a.save(group, "a")
        group.write_object("b", self.b)

    @classmethod
    def load(cls, storage, loc):
        group = StorageGroup(storage, loc)
        obj = cls(a=group["a"], b=group.read_object("b"))
        obj.loaded = True
        return obj


storage_actions.register("read_item", B, B.load, inherit=False)
storage_actions.register("write_item", B, lambda s, l, o: o.save(s, l), inherit=False)


class B1(B): ...


@pytest.mark.parametrize("ext", STORAGE_EXT)
def test_register_class_action(ext, tmp_path):
    """Test loading custom classes."""

    a = A({"a": 1})
    assert not a.loaded

    with open_storage(tmp_path / f"file{ext}", mode="truncate") as store:
        store["a"] = a

    with open_storage(tmp_path / f"file{ext}", mode="read") as store:
        b = store["a"]

    assert b.loaded
    assert b.value == {"a": 1}


@pytest.mark.parametrize("ext", STORAGE_EXT)
def test_register_class_action_nested(tmp_path, ext):
    """Test loading custom nested classes."""

    a = A({"a": 1})
    b = B(a, "str")
    assert not a.loaded
    assert not b.loaded

    with open_storage(tmp_path / f"file{ext}", mode="truncate") as store:
        store["data"] = b

    with open_storage(tmp_path / f"file{ext}", mode="read") as store:
        res = store["data"]

    assert res.loaded
    assert res.a.loaded
    assert res.a.value == {"a": 1}
    assert res.b == "str"


@pytest.mark.parametrize("ext", STORAGE_EXT)
def test_register_class_action_subclass(ext, tmp_path):
    """Test loading custom classes and subclasses."""

    a = A({"a": 1})
    a1 = A1({"b": 2})  # has not defined hooks itself
    a2 = A2({"c": 3})  # has own loading hook
    assert not a.loaded
    assert not a1.loaded
    assert not a2.loaded

    with open_storage(tmp_path / f"file{ext}", mode="truncate") as store:
        store["a"] = a
        store["a1"] = a1
        store["a2"] = a2

    with open_storage(tmp_path / f"file{ext}", mode="read") as store:
        b = store["a"]
        b1 = store["a1"]
        b2 = store["a2"]

    assert b.loaded
    assert b1.loaded
    assert b2.loaded == "yes"
    assert b.value == {"a": 1}
    assert b1.value == {"b": 2}
    assert b2.value == {"c": 3}


@pytest.mark.parametrize("ext", STORAGE_EXT)
def test_register_class_action_no_inhert(ext, tmp_path):
    """Test loading custom classes and subclasses."""

    a = B(A({"a": 1}), 2)  # does not use hooks that inherit
    a1 = B1(A({"a": 2}), 4)  # has not defined hooks itself
    assert not a.loaded
    assert not a1.loaded

    with open_storage(tmp_path / f"file{ext}", mode="truncate") as store:
        store["b"] = a
        store["b1"] = a1

    with open_storage(tmp_path / f"file{ext}", mode="read") as store:
        b = store["b"]
        b1 = store["b1"]

    assert b.loaded  # B uses the hook class
    assert not b1.loaded  # B2 does not yse the hook classes
    assert b.a.value == {"a": 1}
    assert b.b == 2
    assert b1.a.value == {"a": 2}
    assert b1.b == 4
