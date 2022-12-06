from modelrunner import set_default


@set_default
def first(a: float = 1, b: float = 2):
    """Add two numbers"""
    print(a + b)


def second(a: float = 1, b: float = 2):
    """Multiply two numbers"""
    print(a * b)
