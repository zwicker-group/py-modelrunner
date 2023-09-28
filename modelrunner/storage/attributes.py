"""
.. codeauthor:: David Zwicker <david.zwicker@ds.mpg.de>
"""

import json
from typing import Any, Dict, Mapping

from .utils import decode_binary, encode_binary

Attrs = Dict[str, Any]
AttrsLike = Mapping[str, Any]


class AttrsEncoder(json.JSONEncoder):
    """helper class for encoding python data in JSON"""

    def default(self, obj):
        try:
            return json.JSONEncoder.default(self, obj)
        except TypeError:
            return {"__pickled__": encode_binary(obj)}


def _decode_pickled(dct: AttrsLike):
    if "__pickled__" in dct:
        return decode_binary(dct["__pickled__"])
    return dct


def encode_attr(value) -> str:
    return json.dumps(value, cls=AttrsEncoder)


def encode_attrs(attrs: AttrsLike) -> Attrs:
    return {k: encode_attr(v) for k, v in attrs.items()}


def decode_attr(value: str) -> Any:
    return json.loads(value, object_hook=_decode_pickled)


def decode_attrs(attrs: AttrsLike) -> Attrs:
    return {k: decode_attr(v) for k, v in attrs.items()}


__all__ = ["encode_attrs", "decode_attrs"]
