"""
.. codeauthor:: David Zwicker <david.zwicker@ds.mpg.de>
"""

import codecs
import json
import pickle

from .utils import Attrs

PICKLE_PROTOCOL = pickle.HIGHEST_PROTOCOL


class AttrsEncoder(json.JSONEncoder):
    """helper class for encoding python data in JSON"""

    def default(self, obj):
        try:
            return json.JSONEncoder.default(self, obj)
        except TypeError:
            data_str = codecs.encode(pickle.dumps(obj), "base64").decode()
            return {"__pickled__": data_str}
            # return json.dumps(obj_enc)
            # return json.JSONEncoder.default(self, obj_enc)


def _decode_pickled(dct):
    if "__pickled__" in dct:
        data = codecs.decode(dct["__pickled__"].encode(), "base64")
        return pickle.loads(data)
    return dct


def encode_attrs(attrs: Attrs) -> Attrs:
    return {k: json.dumps(v, cls=AttrsEncoder) for k, v in attrs.items()}

def encode_attr(value):
    return json.dumps(value, cls=AttrsEncoder)


def decode_attrs(attrs: Attrs) -> Attrs:
    return {k: json.loads(v, object_hook=_decode_pickled) for k, v in attrs.items()}


def remove_dunderscore_attrs(attrs: Attrs) -> Attrs:
    return {k: v for k, v in attrs.items() if not k.startswith("__")}


__all__ = ["encode_attrs", "decode_attrs", "remove_dunderscore_attrs"]
