"""
.. codeauthor:: David Zwicker <david.zwicker@ds.mpg.de>
"""

import itertools
import logging
import math
import pickle

import numpy as np
import pytest

from modelrunner.model.parameters import (
    DeprecatedParameter,
    HideParameter,
    Parameter,
    Parameterized,
    auto_type,
    get_all_parameters,
)


def test_autotype():
    """Test automatic type conversion."""
    assert auto_type(1) == 1
    assert isinstance(auto_type(1), int)
    assert isinstance(auto_type(1.0), int)
    assert auto_type("1") == 1
    assert auto_type(1.5) == 1.5
    assert auto_type("1.5") == 1.5
    assert isinstance(auto_type("1.0"), float)
    assert auto_type("asdf") == "asdf"
    assert np.isnan(auto_type(math.nan))
    assert auto_type(math.inf) == math.inf
    assert auto_type(-math.inf) == -math.inf


def test_parameters():
    """Test mixing Parameterized."""

    param = Parameter("a", 1, int, "help", extra={"b": 3})
    assert isinstance(str(param), str)

    p_string = pickle.dumps(param)
    param_new = pickle.loads(p_string)
    assert param.__dict__ == param_new.__dict__
    assert param is not param_new
    assert param_new.extra["b"] == 3

    class Test1(Parameterized):
        parameters_default = [param]

    t = Test1()
    assert t.parameters["a"] == 1
    assert t.get_parameter_default("a") == 1
    assert Test1.get_parameter_default("a") == 1

    t = Test1(parameters={"a": 2})
    assert t.parameters["a"] == 2
    assert t.get_parameter_default("a") == 1
    assert Test1.get_parameter_default("a") == 1

    with pytest.raises(ValueError):
        t = Test1(parameters={"b": 3})
    t = Test1()
    ps = t._parse_parameters({"b": 3}, check_validity=False)
    assert ps["a"] == 1
    assert ps["b"] == 3

    class Test2(Test1):
        # also test conversion of default parameters
        parameters_default = [Parameter("b", "2", int, "help")]

    t = Test2()
    assert t.parameters["a"] == 1
    assert t.parameters["b"] == 2

    t = Test2(parameters={"a": 10, "b": 20})
    assert t.parameters["a"] == 10
    assert t.parameters["b"] == 20
    assert t.get_parameter_default("a") == 1
    assert t.get_parameter_default("b") == 2
    assert Test2.get_parameter_default("a") == 1
    assert Test2.get_parameter_default("b") == 2
    with pytest.raises(KeyError):
        t.get_parameter_default("c")

    class Test3(Test2):
        # test overwriting defaults
        parameters_default = [Parameter("a", 3), Parameter("c", 4)]

    t = Test3()
    assert t.parameters["a"] == 3
    assert t.get_parameter_default("a") == 3
    assert Test3.get_parameter_default("a") == 3
    assert set(t.parameters.keys()) == {"a", "b", "c"}

    # test get_all_parameters function after having used Parameters
    p1 = get_all_parameters()
    for key in ["value", "description"]:
        p2 = get_all_parameters(key)
        assert set(p1) == p2.keys()

    class Test4(Test3):
        # test default parameters given as dictionarys
        parameters_default = {"a": 30, "b": 40}

    t = Test4()
    assert t.parameters["a"] == 30
    assert t.get_parameter_default("a") == 30
    assert Test4.get_parameter_default("a") == 30
    assert set(t.parameters.keys()) == {"a", "b", "c"}
    assert t.parameters["b"] == 40
    assert t.parameters["c"] == 4


def test_parameters_simple():
    """Test adding parameters using a simple dictionary."""

    class TestSimple(Parameterized):
        parameters_default = {"a": 1}

    assert TestSimple().parameters["a"] == 1
    assert TestSimple({"a": 2}).parameters["a"] == 2
    assert TestSimple({"a": "2"}).parameters["a"] == 2
    assert TestSimple({"a": "two"}).parameters["a"] == "two"


def test_parameter_help(capsys):
    """Test how parameters are shown."""

    class TestHelp1(Parameterized):
        parameters_default = [DeprecatedParameter("a", 1, int, "random string")]

    class TestHelp2(TestHelp1):
        parameters_default = [Parameter("b", 2, int, "another word")]

    t = TestHelp2()
    for flags in itertools.combinations_with_replacement([True, False], 3):
        TestHelp2.show_parameters(*flags)
        o1, e1 = capsys.readouterr()
        t.show_parameters(*flags)
        o2, e2 = capsys.readouterr()
        assert o1 == o2
        assert e1 == e2 == ""


def test_parameter_required():
    """Test required parameter."""

    class TestRequired(Parameterized):
        parameters_default = [Parameter("a", required=True)]

    assert TestRequired({"a": 2}).parameters["a"] == 2
    with pytest.raises(ValueError):
        TestRequired()


def test_parameter_choices():
    """Test parameter with explicit choices."""

    class TestChoices(Parameterized):
        parameters_default = [Parameter("a", choices={1, 2, 3})]

    assert TestChoices().parameters["a"] is None
    assert TestChoices({"a": 2}).parameters["a"] == 2
    with pytest.raises(ValueError):
        TestChoices({"a": 0})
    with pytest.raises(ValueError):
        TestChoices({"a": 4})

    class TestChoicesRequired(Parameterized):
        parameters_default = [Parameter("a", required=True, choices={1, 2, 3})]

    assert TestChoicesRequired({"a": 2}).parameters["a"] == 2
    with pytest.raises(ValueError):
        TestChoicesRequired({"a": 0})
    with pytest.raises(ValueError):
        TestChoicesRequired({"a": 4})
    with pytest.raises(ValueError):
        TestChoicesRequired()

    with pytest.raises(ValueError):

        class TestChoicesInconsistent(Parameterized):
            parameters_default = [Parameter("a", 4, choices={1, 2, 3})]


def test_hidden_parameter():
    """Test how hidden parameters are handled."""

    class TestHidden1(Parameterized):
        parameters_default = [Parameter("a", 1), Parameter("b", 2)]

    assert TestHidden1().parameters == {"a": 1, "b": 2}

    class TestHidden2(TestHidden1):
        parameters_default = [HideParameter("b")]

    class TestHidden2a(Parameterized):
        parameters_default = [Parameter("a", 1), Parameter("b", 2, hidden=True)]

    for t_class in [TestHidden2, TestHidden2a]:
        assert "b" not in t_class.get_parameters()
        assert len(t_class.get_parameters()) == 1
        assert len(t_class.get_parameters(include_hidden=True)) == 2
        t2 = t_class()
        assert t2.parameters == {"a": 1, "b": 2}
        assert t2.get_parameter_default("b") == 2
        with pytest.raises(ValueError):
            t2._parse_parameters({"b": 2}, check_validity=True, allow_hidden=False)

    class TestHidden3(TestHidden1):
        parameters_default = [Parameter("b", 3)]

    t3 = TestHidden3()
    assert t3.parameters == {"a": 1, "b": 3}
    assert t3.get_parameter_default("b") == 3


def test_convert_default_values(caplog):
    """Test how default values are handled."""
    caplog.set_level(logging.WARNING)

    class TestConvert1(Parameterized):
        parameters_default = [Parameter("a", 1, float)]

    t1 = TestConvert1()
    assert "Default value" not in caplog.text
    assert isinstance(t1.parameters["a"], float)

    class TestConvert2(Parameterized):
        parameters_default = [Parameter("a", np.arange(3), np.array)]

    t2 = TestConvert2()
    np.testing.assert_equal(t2.parameters["a"], np.arange(3))

    class TestConvert3(Parameterized):
        parameters_default = [Parameter("a", [0, 1, 2], np.array)]

    t3 = TestConvert3()
    np.testing.assert_equal(t3.parameters["a"], np.arange(3))

    caplog.clear()

    class TestConvert4(Parameterized):
        parameters_default = [Parameter("a", 1, str)]

    t4 = TestConvert4()
    assert len(caplog.records) == 1
    assert "Default value" in caplog.text
    np.testing.assert_equal(t4.parameters["a"], "1")

    caplog.clear()

    class TestConvert5(Parameterized):
        parameters_default = [Parameter("a", math.nan, float)]

    t5 = TestConvert5()
    assert len(caplog.records) == 0
    assert t5.parameters["a"] is math.nan


def test_parameters_default_full():
    """Test the _parameters_default_full property."""
    ps = [
        Parameter("a", 1),  # 0
        Parameter("b", 2),  # 1
        HideParameter("b"),  # 2
        Parameter("a", 3),  # 3
        Parameter("b", 4, hidden=True),  # 4
        Parameter("a", 5),  # 5
    ]
    ps1_hidden = Parameter("b", 2, hidden=True)

    class TestDefault1(Parameterized):
        parameters_default = [ps[0], ps[1]]

    assert TestDefault1()._parameters_default_full == [ps[0], ps[1]]

    class TestDefault2(TestDefault1):
        parameters_default = [ps[2]]

    assert TestDefault2()._parameters_default_full == [ps[0], ps1_hidden]

    class TestDefault3(TestDefault1):
        parameters_default = [ps[3], ps[4]]

    assert TestDefault3()._parameters_default_full == [ps[3], ps[4]]

    class TestDefault4(TestDefault3):
        parameters_default = [ps[2]]

    assert TestDefault4()._parameters_default_full == [ps[3], ps[4]]

    class TestDefault5(TestDefault3):
        parameters_default = [ps[5]]

    assert TestDefault5()._parameters_default_full == [ps[5], ps[4]]
