"""
.. codeauthor:: David Zwicker <david.zwicker@ds.mpg.de>
"""

from __future__ import annotations

import yaml

from .text_base import TextStorageBase


class YAMLStorage(TextStorageBase):
    extensions = ["yaml", "yml"]

    def _read_data_from_fp(self, fp) -> None:
        try:
            self._data = yaml.safe_load(fp)
        except yaml.constructor.ConstructorError:
            raise RuntimeError("Some data cannot be reconstructed from YAML")

    def _write_data_to_fp(self, fp, data) -> None:
        self._write_flags.setdefault("sort_keys", False)
        try:
            yaml.dump(data, fp, **self._write_flags)
        except yaml.constructor.ConstructorError:
            raise RuntimeError("Some data cannot be represented by YAML")
