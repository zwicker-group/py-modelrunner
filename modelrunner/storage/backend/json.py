"""
.. codeauthor:: David Zwicker <david.zwicker@ds.mpg.de>
"""

from __future__ import annotations

import json

from ..attributes import AttrsEncoder, _decode_pickled
from .text_base import TextStorageBase


class JSONStorage(TextStorageBase):
    extensions = ["json"]

    def _read_data_from_file(self) -> None:
        with open(self._path, mode="r") as fp:
            self._data = json.load(fp, object_hook=_decode_pickled)

    def _write_data_to_fp(self, fp, data) -> None:
        self._write_flags.setdefault("cls", AttrsEncoder)
        json.dump(data, fp, **self._write_flags)
