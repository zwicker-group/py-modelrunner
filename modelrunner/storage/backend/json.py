"""Defines a class storing data in memory and writing it to a file in JSON format.

.. codeauthor:: David Zwicker <david.zwicker@ds.mpg.de>
"""

from __future__ import annotations

import json

from ..attributes import AttrsEncoder, _decode_pickled
from .text_base import TextStorageBase


class JSONStorage(TextStorageBase):
    """Storage that stores data in a JSON text file.

    Note that the data is only written once the storage is closed.
    """

    extensions = ["json"]

    def _read_data_from_fp(self, fp):
        return json.load(fp, object_hook=_decode_pickled)

    def _write_data_to_fp(self, fp, data) -> None:
        self._write_flags.setdefault("cls", AttrsEncoder)
        json.dump(data, fp, **self._write_flags)
