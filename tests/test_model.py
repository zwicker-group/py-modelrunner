"""
.. codeauthor:: David Zwicker <david.zwicker@ds.mpg.de>
"""

from pathlib import Path
from typing import Literal

import pytest

from modelrunner.model import ModelBase, make_model, make_model_class, run_script
from modelrunner.parameters import (
    DeprecatedParameter,
    HideParameter,
    NoValue,
    Parameter,
)
from modelrunner.storage import open_storage

PACKAGEPATH = Path(__file__).parents[2].resolve()
SCRIPT_PATH = Path(__file__).parent / "scripts"
assert SCRIPT_PATH.is_dir()


def run(script, *args):
    """run a script (with potential arguments) and collect stdout"""
    result = run_script(SCRIPT_PATH / script, args)
    return result.data


def test_empty_script():
    """test the empty.py script"""
    with pytest.raises(RuntimeError):
        run("empty.py")


def test_function_script():
    """test the function.py script"""
    assert float(run("function.py")) == 2
    assert float(run("function.py", "--a", "3")) == 6
    assert float(run("function.py", "--a", "3", "--b", "4")) == 12


def test_function_main_script():
    """test the function_main.py script"""
    assert float(run("function_main.py")) == 2
    assert float(run("function_main.py", "--a", "3")) == 6
    assert float(run("function_main.py", "--a", "3", "--b", "4")) == 12


def test_function_marked_script():
    """test the function_main.py script"""
    assert float(run("function_marked.py")) == 3
    assert float(run("function_marked.py", "--a", "3")) == 5
    assert float(run("function_marked.py", "--a", "3", "--b", "4")) == 7


def test_make_model_script():
    """test the make_model.py script"""
    assert run("make_model.py") == 2
    assert run("make_model.py", "--a", "3") == 6
    assert run("make_model.py", "--a", "3", "--b", "4") == 12


def test_make_model_class_script():
    """test the make_model_class.py script"""
    assert run("make_model_class.py") == 2
    assert run("make_model_class.py", "--a", "3") == 6
    assert run("make_model_class.py", "--a", "3", "--b", "4") == 12


def test_make_model_marked_script():
    """test the function_main.py script"""
    assert float(run("make_model_marked.py")) == 3
    assert float(run("make_model_marked.py", "--a", "3")) == 5
    assert float(run("make_model_marked.py", "--a", "3", "--b", "4")) == 7


def test_required_arguments_model():
    """test required arguments"""

    @make_model
    def req_args_1(a=1):
        return a

    assert req_args_1.parameters == {"a": 1}
    assert req_args_1() == 1

    @make_model
    def req_args_2(a):
        return a

    assert req_args_2.parameters == {"a": NoValue}
    with pytest.raises(TypeError):
        req_args_2()

    @make_model
    def req_args_3(a=None):
        return a

    assert req_args_3.parameters == {"a": None}
    assert req_args_3() is None


def test_required_arguments_model_class_decorator():
    """test required arguments"""

    @make_model_class
    def required_arg_1(a=1):
        return a

    assert required_arg_1().parameters == {"a": 1}
    assert required_arg_1()() == 1

    @make_model_class
    def required_arg_2(a):
        return a

    assert required_arg_2().parameters == {"a": NoValue}
    with pytest.raises(TypeError):
        required_arg_2()()

    @make_model_class
    def required_arg_3(a=None):
        return a

    assert required_arg_3().parameters == {"a": None}
    assert required_arg_3()() is None


def test_required_arguments_model_class():
    """test required arguments"""

    class A(ModelBase):
        parameters_default = [Parameter("a", required=True), Parameter("b", 2)]

        def __call__(self):
            return self.parameters["a"] + self.parameters["b"]

    assert A({"a": 3})() == 5
    assert A.run_from_command_line(["--a", "3"]).data == 5
    with pytest.raises(SystemExit):
        A.run_from_command_line([])


def test_choices_arguments_model_class():
    """test arguments with choices"""

    class A(ModelBase):
        parameters_default = [
            Parameter("a", 1, cls=int, choices={1, 2, 3}),
            Parameter("b", 2),
        ]

        def __call__(self):
            return self.parameters["a"] + self.parameters["b"]

    assert A()() == 3
    assert A({"a": 3})() == 5
    with pytest.raises(ValueError):
        A({"a": 4})
    assert A.run_from_command_line(["--a", "3"]).data == 5
    with pytest.raises(SystemExit):
        A.run_from_command_line(["--a", "4"])


def test_make_model():
    """test the make_model decorator"""

    @make_model
    def model1(a=2):
        return a**2

    assert model1.parameters == {"a": 2}

    assert model1() == 4
    assert model1(3) == 9
    assert model1(a=4) == 16
    assert model1.get_result().data == 4

    @make_model
    def model2(a, b=2):
        return a * b

    assert model2.parameters == {"a": NoValue, "b": 2}

    assert model2(3) == 6
    assert model2(a=3) == 6
    assert model2(3, 3) == 9
    assert model2(a=3, b=3) == 9
    assert model2(3, b=3) == 9

    with pytest.raises(TypeError):
        model2()


def test_make_model_class():
    """test the make_model_class function"""

    def model_func(a=2):
        return a**2

    model = make_model_class(model_func)

    assert model()() == 4
    assert model({"a": 3})() == 9
    assert model({"a": 4}).get_result().data == 16


def test_make_model_class_literal_args():
    """test the make_model_class function"""

    def model_func(a: Literal["a", 2] = 2):
        return a * 2

    model = make_model_class(model_func)
    assert model.parameters_default[0].choices == ("a", 2)

    assert model()() == 4
    assert model({"a": "a"})() == "aa"
    with pytest.raises(ValueError):
        model({"a": 3})

    assert model.run_from_command_line(["--a", "a"]).data == "aa"
    with pytest.raises(SystemExit):
        model.run_from_command_line(["--a", "b"])


def test_argparse_boolean_arguments():
    """test boolean parameters"""

    @make_model
    def parse_bool_0(flag: bool):
        return flag

    with pytest.raises(SystemExit):
        parse_bool_0.run_from_command_line()
    assert parse_bool_0.run_from_command_line(["--flag"]).data
    assert not parse_bool_0.run_from_command_line(["--no-flag"]).data

    @make_model
    def parse_bool_1(flag: bool = False):
        return flag

    assert not parse_bool_1.run_from_command_line().data
    assert parse_bool_1.run_from_command_line(["--flag"]).data

    @make_model
    def parse_bool_2(flag: bool = True):
        return flag

    assert parse_bool_2.run_from_command_line().data
    assert not parse_bool_2.run_from_command_line(["--no-flag"]).data


def test_argparse_list_arguments():
    """test list parameters"""

    @make_model
    def parse_list_0(flag: list):
        return flag

    with pytest.raises(TypeError):
        assert parse_list_0.run_from_command_line()
    assert parse_list_0.run_from_command_line(["--flag"]).data == []
    assert parse_list_0.run_from_command_line(["--flag", "0"]).data == ["0"]
    assert parse_list_0.run_from_command_line(["--flag", "0", "1"]).data == ["0", "1"]

    @make_model
    def parse_list_1(flag: list = [0, 1]):
        return flag

    assert parse_list_1.run_from_command_line().data == [0, 1]
    assert parse_list_1.run_from_command_line(["--flag"]).data == []
    assert parse_list_1.run_from_command_line(["--flag", "0"]).data == ["0"]
    assert parse_list_1.run_from_command_line(["--flag", "0", "1"]).data == ["0", "1"]


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
    assert A.run_from_command_line(["--a", "2"]).data == 5
    with pytest.raises(SystemExit):
        A.run_from_command_line(["--b", "2"])

    assert B().parameters == {"a": 1, "b": 2, "c": 4, "d": 5}
    assert B()() == 10
    with pytest.raises(SystemExit):
        B.run_from_command_line(["--a", "2"])
    with pytest.raises(SystemExit):
        B.run_from_command_line(["--b", "2"])
    assert B.run_from_command_line(["--c", "2"]).data == 8
    assert B.run_from_command_line(["--d", "6"]).data == 11


def test_model_output(tmp_path):
    """test whether model output works as intended"""

    class A(ModelBase):
        parameters_default = [Parameter("a", 1)]

        def __call__(self, b):
            self.storage["info"] = {"args": b}
            return self.parameters["a"] * b

    a = A(output=tmp_path / "model.json")
    assert a(5) == 5
    a.close()

    with open_storage(tmp_path / "model.json") as storage:
        assert storage["data/info"] == {"args": 5}


@pytest.mark.parametrize("kwarg", [True, False])
def test_model_storage(kwarg, tmp_path):
    """test storage argument in model"""

    if kwarg:

        @make_model_class
        def model_with_output(a=3, storage=None):
            storage["saved"] = {"A": "B"}
            return a + 2

    else:

        @make_model_class
        def model_with_output(storage, a=3):
            storage["saved"] = {"A": "B"}
            return a + 2

    path = tmp_path / "output1.yaml"
    m = model_with_output(output=path)
    m.write_result()

    with open_storage(path) as storage:
        assert storage["data/saved"] == {"A": "B"}
        assert storage["result"].data == 5
