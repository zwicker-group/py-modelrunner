"""
Base classes for storing data

.. codeauthor:: David Zwicker <david.zwicker@ds.mpg.de> 
"""

from __future__ import annotations

import logging
from abc import ABCMeta, abstractmethod
from importlib import import_module
from typing import List, Iterator, Sequence, Tuple, Union, Dict, Optional, Any
import numpy as np

InfoDict = Dict[str, Any]


class StorageBase(metaclass=ABCMeta):
    """base class for storing time series of discretized fields

    These classes store time series of :class:`~pde.fields.base.FieldBase`, i.e., they
    store the values of the fields at particular time points. Iterating of the storage
    will return the fields in order and individual time points can also be accessed.
    """

    extensions: List[str] = []

    def __init__(self, *, overwrite: bool = False):
        """
        Args:
            overwrite (bool):
                Determines whether existing data can be overwritten
        """
        self.overwrite = overwrite
        self._logger = logging.getLogger(self.__class__.__name__)

    def __getitem__(self, key: str):
        from .group import Group

        return Group(self)[key]

    @abstractmethod
    def keys(self, key: Sequence[str]) -> List[str]:
        ...

    @abstractmethod
    def _read_attrs(self, key: Sequence[str]) -> InfoDict:
        ...

    @abstractmethod
    def is_group(self, key: Sequence[str]) -> bool:
        ...

    @abstractmethod
    def create_group(self, key: Sequence[str]):
        ...

    @abstractmethod
    def _read_array(self, key: Sequence[str]) -> Tuple[np.ndarray, InfoDict]:
        ...

    @abstractmethod
    def _write_array(self, key: Sequence[str], arr: np.ndarray, attrs: InfoDict):
        ...
