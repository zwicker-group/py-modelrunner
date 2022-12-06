import modelrunner


def do_not_calculate(a=1, b=2):
    """This function should not be run"""
    raise RuntimeError("This must not run")


@modelrunner.make_model_class
def calculate(a=1, b=2):
    """This function has been marked as a model"""
    print(a * b)
