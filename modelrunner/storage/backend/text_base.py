"""
.. codeauthor:: David Zwicker <david.zwicker@ds.mpg.de>
"""

from __future__ import annotations

from abc import ABCMeta, abstractmethod
from io import StringIO
from pathlib import Path
from typing import Optional, Union

from .memory import MemoryStorage
from ..access import AccessType
from .utils import simplify_data


class TextStorageBase(MemoryStorage, metaclass=ABCMeta):
    def __init__(
        self,
        path: Union[str, Path],
        *,
        access: AccessType = "full",
        simplify: bool = True,
        **kwargs,
    ):
        super().__init__(access=access)

        self.simplify = simplify
        self._path = Path(path)
        self._write_flags = kwargs
        if self.access.file_mode in {"r", "x", "a"}:
            if self._path.exists():
                self._read_data_from_file()

    def close(self):
        if self.access.file_mode in {"w", "x", "a"}:
            if self.simplify:
                data = simplify_data(self._data)
            else:
                data = self._data
            with open(self._path, mode="w") as fp:
                self._write_data_to_fp(fp, data)

    def to_text(self, simplify: Optional[bool] = None) -> str:
        if simplify is None:
            simplify = self.simplify
        if simplify:
            data = simplify_data(self._data)
        else:
            data = self._data
        with StringIO() as fp:
            self._write_data_to_fp(fp, data)
            return fp.getvalue()

    @abstractmethod
    def _read_data_from_file(self) -> None:
        ...

    @abstractmethod
    def _write_data_to_fp(self, fp, data) -> None:
        ...

    # @abstractmethod
    # def _write_data_to_file(self) -> None:
    #     ...


#
# class YAMLStorage(TextBasedStorage):
#     extensions = ["yaml", "yml"]
#
#     def _read_data_from_file(self) -> None:
#         import yaml
#
#         with open(self._path, mode="r") as fp:
#             self._data = yaml.safe_load(fp)
#
#     def _write_data_to_file(self, data) -> None:
#         import yaml
#
#         self._write_flags.setdefault("sort_keys", False)
#         with open(self._path, mode="w") as fp:
#             yaml.dump(data, fp, **self._write_flags)
