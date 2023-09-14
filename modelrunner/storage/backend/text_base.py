"""
.. codeauthor:: David Zwicker <david.zwicker@ds.mpg.de>
"""

from __future__ import annotations

from abc import ABCMeta, abstractmethod
from io import StringIO
from pathlib import Path
from typing import Optional, Union

from ..access_modes import ModeType
from .memory import MemoryStorage
from .utils import simplify_data


class TextStorageBase(MemoryStorage, metaclass=ABCMeta):
    def __init__(
        self,
        path: Union[str, Path],
        *,
        mode: ModeType = "readonly",
        simplify: bool = True,
        **kwargs,
    ):
        """
        Args:
            path (str or :class:`~pathlib.Path`):
                File path to the file
            mode (str or :class:`~modelrunner.storage.access_modes.AccessMode`):
                The file mode with which the storage is accessed. Determines allowed
                operations.
            simplify (bool):
                Flag indicating whether the data is stored in a simplified form
        """
        super().__init__(mode=mode)

        self.simplify = simplify
        self._path = Path(path)
        self._write_flags = kwargs
        if self.mode.file_mode in {"r", "x", "a"}:
            if self._path.exists():
                with open(self._path, mode="r") as fp:
                    self._read_data_from_fp(fp)

    def close(self):
        if self.mode.file_mode in {"x", "a", "w"}:
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
    def _read_data_from_fp(self, fp) -> None:
        ...

    @abstractmethod
    def _write_data_to_fp(self, fp, data) -> None:
        ...
