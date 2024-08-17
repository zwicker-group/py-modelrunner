"""Defines a class storing data in memory and writing it to a file in YAML format.

Requires the optional :mod:`yaml` module.

.. codeauthor:: David Zwicker <david.zwicker@ds.mpg.de>
"""

from __future__ import annotations

import yaml

from .text_base import TextStorageBase


class YAMLStorage(TextStorageBase):
    """Storage that stores data in a YAML text file.

    Note that the data is only written once the storage is closed.
    """

    extensions = ["yaml", "yml"]
    encode_internal_attrs = True

    def _read_data_from_fp(self, fp):
        try:
            return yaml.safe_load(fp)
        except yaml.constructor.ConstructorError as err:
            raise RuntimeError("Some data cannot be reconstructed from YAML") from err

    def _write_data_to_fp(self, fp, data) -> None:
        self._write_flags.setdefault("sort_keys", False)
        try:
            yaml.dump(data, fp, **self._write_flags)
        except yaml.constructor.ConstructorError as err:
            raise RuntimeError("Some data cannot be represented by YAML") from err
