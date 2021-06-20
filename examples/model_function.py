"""
.. codeauthor:: David Zwicker <david.zwicker@ds.mpg.de>
"""

from model import function_model_init


def multiply(a, b=2):
    return a * b


mdl = function_model_init(multiply, {"a": 3})
print(mdl())
