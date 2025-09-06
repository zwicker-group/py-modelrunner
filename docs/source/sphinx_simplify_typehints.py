"""Simple sphinx plug-in that simplifies  type information in function signatures."""

import collections
import re

from modelrunner import Parameterized
from modelrunner.model.parameters import sphinx_display_parameters

# simple (literal) replacement rules
REPLACEMENTS = collections.OrderedDict(
    [
        # Remove some unnecessary information in favor of a more compact style
        ("Dict[KT, VT]", "dict"),
        ("Dict[str, Any]", "dict"),
        ("Optional[str]", "str"),
        ("Optional[float]", "float"),
        ("Optional[int]", "int"),
        ("Optional[dict]", "dict"),
        ("Optional[Dict[str, Any]]", "dict"),
    ]
)


# replacement rules based on regular expressions
REPLACEMENTS_REGEX = {
    # remove full package path and only leave the module/class identifier
    r"modelrunner\.(\w+\.)*": "",
}


def process_signature(
    app, what: str, name: str, obj, options, signature, return_annotation
):
    """Process signature by applying replacement rules."""
    if signature is not None:
        for key, value in REPLACEMENTS.items():
            signature = signature.replace(key, value)
        for key, value in REPLACEMENTS_REGEX.items():
            signature = re.sub(key, value, signature)
    return signature, return_annotation


def setup(app):
    """Set up hooks for this sphinx plugin."""
    app.connect("autodoc-process-signature", process_signature)
    app.connect("autodoc-process-docstring", sphinx_display_parameters)
