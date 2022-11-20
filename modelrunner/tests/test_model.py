"""
.. codeauthor:: David Zwicker <david.zwicker@ds.mpg.de>
"""

import os
import subprocess as sp
import sys
from pathlib import Path
from typing import List  # @UnusedImport

import pytest

from ..model import ModelBase, make_model, make_model_class
from ..parameters import DeprecatedParameter, HideParameter, NoValue, Parameter

PACKAGEPATH = Path(__file__).parents[2].resolve()
SCRIPT_PATH = Path(__file__).parent / "scripts"


def run_script(script, *args):
    """run a script (with potential arguments) and collect stdout"""
    # prepare environment
    env = os.environ.copy()
    env["PYTHONPATH"] = str(PACKAGEPATH) + ":" + env.get("PYTHONPATH", "")

    # run example in temporary folder since it might create data files
    path = SCRIPT_PATH / script
    cmd_args = (sys.executable, "-m", "modelrunner", path) + args
    proc = sp.Popen(cmd_args, env=env, stdout=sp.PIPE, stderr=sp.PIPE)
    outs, errs = proc.communicate(timeout=30)

    if errs != b"":
        print(errs)
        assert False
    return outs.strip()


def test_empty_script():
    """test the empty.py script"""
    with pytest.raises(AssertionError):
        run_script("empty.py")


def test_function_script():
    """test the function.py script"""
    assert float(run_script("function.py")) == 2
    assert float(run_script("function.py", "--a", "3")) == 6
    assert float(run_script("function.py", "--a", "3", "--b", "4")) == 12


def test_function_main_script():
    """test the function_main.py script"""
    assert float(run_script("function_main.py")) == 2
    assert float(run_script("function_main.py", "--a", "3")) == 6
    assert float(run_script("function_main.py", "--a", "3", "--b", "4")) == 12


def test_function_marked_script():
    """test the function_main.py script"""
    assert float(run_script("function_marked.py")) == 3
    assert float(run_script("function_marked.py", "--a", "3")) == 5
    assert float(run_script("function_marked.py", "--a", "3", "--b", "4")) == 7


def test_make_model_script():
    """test the make_model.py script"""
    assert run_script("make_model.py") == b"2"
    assert run_script("make_model.py", "--a", "3") == b"6"
    assert run_script("make_model.py", "--a", "3", "--b", "4") == b"12"


def test_make_model_class_script():
    """test the make_model_class.py script"""
    assert run_script("make_model_class.py") == b"2"
    assert run_script("make_model_class.py", "--a", "3") == b"6"
    assert run_script("make_model_class.py", "--a", "3", "--b", "4") == b"12"


def test_make_model_marked_script():
    """test the function_main.py script"""
    assert float(run_script("make_model_marked.py")) == 3
    assert float(run_script("make_model_marked.py", "--a", "3")) == 5
    assert float(run_script("make_model_marked.py", "--a", "3", "--b", "4")) == 7


def test_required_arguments_model():
    """test required arguments"""

    @make_model
    def f1(a=1):
        return a

    assert f1.parameters == {"a": 1}
    assert f1() == 1

    @make_model
    def f2(a):
        return a

    assert f2.parameters == {"a": NoValue}
    with pytest.raises(TypeError):
        f2()

    @make_model
    def f3(a=None):
        return a

    assert f3.parameters == {"a": None}
    assert f3() is None


def test_required_arguments_model_class():
    """test required arguments"""

    @make_model_class
    def f1(a=1):
        return a

    assert f1().parameters == {"a": 1}
    assert f1()() == 1

    @make_model_class
    def f2(a):
        return a

    assert f2().parameters == {"a": NoValue}
    with pytest.raises(TypeError):
        f2()()

    @make_model_class
    def f3(a=None):
        return a

    assert f3().parameters == {"a": None}
    assert f3()() is None


def test_make_model():
    """test the make_model decorator"""

    @make_model
    def f(a=2):
        return a**2

    assert f.parameters == {"a": 2}

    assert f() == 4
    assert f(3) == 9
    assert f(a=4) == 16
    assert f.get_result().result == 4

    @make_model
    def g(a, b=2):
        return a * b

    assert g.parameters == {"a": NoValue, "b": 2}

    assert g(3) == 6
    assert g(a=3) == 6
    assert g(3, 3) == 9
    assert g(a=3, b=3) == 9
    assert g(3, b=3) == 9

    with pytest.raises(TypeError):
        g()


def test_make_model_class():
    """test the make_model_class function"""

    def f(a=2):
        return a**2

    model = make_model_class(f)

    assert model()() == 4
    assert model({"a": 3})() == 9
    assert model({"a": 4}).get_result().result == 16


def test_argparse_boolean_arguments():
    """test boolean parameters"""

    @make_model
    def f0(flag: bool):
        return flag

    with pytest.raises(SystemExit):
        f0.run_from_command_line()
    assert f0.run_from_command_line(["--flag"]).result
    assert not f0.run_from_command_line(["--no-flag"]).result

    @make_model
    def f1(flag: bool = False):
        return flag

    assert not f1.run_from_command_line().result
    assert f1.run_from_command_line(["--flag"]).result

    @make_model
    def f2(flag: bool = True):
        return flag

    assert f2.run_from_command_line().result
    assert not f2.run_from_command_line(["--no-flag"]).result


def test_argparse_list_arguments():
    """test list parameters"""

    @make_model
    def f0(flag: list):
        return flag

    with pytest.raises(TypeError):
        assert f0.run_from_command_line()
    assert f0.run_from_command_line(["--flag"]).result == []
    assert f0.run_from_command_line(["--flag", "0"]).result == ["0"]
    assert f0.run_from_command_line(["--flag", "0", "1"]).result == ["0", "1"]

    @make_model
    def f1(flag: list = [0, 1]):
        return flag

    assert f1.run_from_command_line().result == [0, 1]
    assert f1.run_from_command_line(["--flag"]).result == []
    assert f1.run_from_command_line(["--flag", "0"]).result == ["0"]
    assert f1.run_from_command_line(["--flag", "0", "1"]).result == ["0", "1"]


def test_model_class_inheritence():
    """test whether inheritence works as intended"""

    class A(ModelBase):
        parameters_default = [
            Parameter("a", 1),
            DeprecatedParameter("b", 2),
            Parameter("c", 3),
        ]

        def __call__(self):
            return self.parameters["a"] + self.parameters["c"]

    class B(A):
        parameters_default = [HideParameter("a"), Parameter("c", 4), Parameter("d", 5)]

        def __call__(self):
            return super().__call__() + self.parameters["d"]

    assert A().parameters == {"a": 1, "b": 2, "c": 3}
    assert A()() == 4
    assert A.run_from_command_line(["--a", "2"]).result == 5
    with pytest.raises(SystemExit):
        A.run_from_command_line(["--b", "2"])

    assert B().parameters == {"a": 1, "b": 2, "c": 4, "d": 5}
    assert B()() == 10
    with pytest.raises(SystemExit):
        B.run_from_command_line(["--a", "2"])
    with pytest.raises(SystemExit):
        B.run_from_command_line(["--b", "2"])
    assert B.run_from_command_line(["--c", "2"]).result == 8
    assert B.run_from_command_line(["--d", "6"]).result == 11
