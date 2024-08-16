"""Simple sphinx plug-in that simplifies  type information in function signatures."""

import collections
import re

from modelrunner import Parameterized

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


def sphinx_display_parameters(app, what, name, obj, options, lines):
    """Helper function to display parameters in sphinx documentation.

    Example:
        This function should be connected to the 'autodoc-process-docstring'
        event like so:

            app.connect('autodoc-process-docstring', sphinx_display_parameters)
    """
    if (
        what == "class"
        and issubclass(obj, Parameterized)
        and any(":param parameters:" in line for line in lines)
    ):
        # parse parameters
        parameters = obj.get_parameters(sort=False)
        if parameters:
            lines.append(".. admonition::")
            lines.append(f"   Parameters of {obj.__name__}:")
            lines.append("   ")
            for p in parameters.values():
                lines.append(f"   {p.name}")
                text = p.description.splitlines()
                text.append(f"(Default value: :code:`{p.default_value!r}`)")
                text = ["     " + t for t in text]
                lines.extend(text)
                lines.append("")
            lines.append("")


def setup(app):
    """Set up hooks for this sphinx plugin."""
    app.connect("autodoc-process-signature", process_signature)
    app.connect("autodoc-process-docstring", sphinx_display_parameters)
