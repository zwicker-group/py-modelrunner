"""
.. codeauthor:: David Zwicker <david.zwicker@ds.mpg.de>
"""

from __future__ import annotations

import json
from typing import Any, Dict, Mapping

from .utils import decode_binary, encode_binary

Attrs = Dict[str, Any]
AttrsLike = Mapping[str, Any]


class AttrsEncoder(json.JSONEncoder):
    """Helper class for encoding python data in JSON."""

    def default(self, obj):
        if isinstance(obj, dict) and "__pickled__" in obj:
            raise ValueError("Cannot encode dictionary with key `__pickled__`")
        try:
            return json.JSONEncoder.default(self, obj)
        except TypeError:
            return {"__pickled__": encode_binary(obj)}


def _decode_pickled(dct: AttrsLike) -> AttrsLike:
    """Decode pickled data.

    Args:
        dct (dict): The encoded attributes dictionary

    Returns:
        dict: Decoded attributes dictionary
    """
    if "__pickled__" in dct:
        return decode_binary(dct["__pickled__"])  # type: ignore
    return dct


def encode_attr(value: Any) -> str:
    """Encode an attribute using JSON.

    Args:
        value: The value to be encoded

    Returns:
        str: The encoded attribute
    """
    return json.dumps(value, cls=AttrsEncoder)


def encode_attrs(attrs: AttrsLike) -> Attrs:
    """Encode many attributes.

    Args:
        attrs (dict): The attributes dictionary

    Returns:
        dict: The encoded attributes
    """
    return {k: encode_attr(v) for k, v in attrs.items()}


def decode_attr(value: str) -> Any:
    """Decode an attribute.

    Args:
        value (str): The encoded attribute

    Returns:
        The decoded attribute
    """
    return json.loads(value, object_hook=_decode_pickled)


def decode_attrs(attrs: AttrsLike) -> Attrs:
    """Decode many attributes.

    Args:
        attrs (dict): The attributes dictionary

    Returns:
        dict: The decoded attributes
    """
    return {k: decode_attr(v) for k, v in attrs.items()}


__all__ = ["encode_attrs", "decode_attrs"]
