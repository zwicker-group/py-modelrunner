"""
.. codeauthor:: David Zwicker <david.zwicker@ds.mpg.de>
"""

import json

from .utils import Attrs, decode_binary, encode_binary


class AttrsEncoder(json.JSONEncoder):
    """helper class for encoding python data in JSON"""

    def default(self, obj):
        try:
            return json.JSONEncoder.default(self, obj)
        except TypeError:
            return {"__pickled__": encode_binary(obj)}


def _decode_pickled(dct):
    if "__pickled__" in dct:
        return decode_binary(dct["__pickled__"])
    return dct


def encode_attrs(attrs: Attrs) -> Attrs:
    return {k: json.dumps(v, cls=AttrsEncoder) for k, v in attrs.items()}


def encode_attr(value):
    return json.dumps(value, cls=AttrsEncoder)


def decode_attrs(attrs: Attrs) -> Attrs:
    return {k: json.loads(v, object_hook=_decode_pickled) for k, v in attrs.items()}


def attrs_remove_dunderscore(attrs: Attrs) -> Attrs:
    return {k: v for k, v in attrs.items() if not k.startswith("__")}


__all__ = ["encode_attrs", "decode_attrs", "attrs_remove_dunderscore"]
