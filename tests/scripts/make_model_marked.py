from modelrunner import make_model, set_default


@set_default
@make_model
def first(a: float = 1, b: float = 2):
    """Add two numbers"""
    print(a + b)


@make_model
def second(a: float = 1, b: float = 2):
    """Multiply two numbers"""
    print(a * b)
