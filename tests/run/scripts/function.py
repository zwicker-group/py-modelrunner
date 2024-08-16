def test_func(a: float = 1, b: list = None):
    """Function returning arguments."""
    if b is None:
        b = [0, 1]
    return {"a": a, "b": b}
