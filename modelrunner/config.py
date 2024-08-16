"""Handles configuration variables.

.. autosummary::
   :nosignatures:

   Config

.. codeauthor:: David Zwicker <david.zwicker@ds.mpg.de>
"""

from __future__ import annotations

import collections
import contextlib
from pathlib import Path
from typing import Any, Sequence

from .model.parameters import DeprecatedParameter, Parameter


class Config(collections.UserDict):
    """Class handling the package configuration."""

    def __init__(
        self,
        default: Sequence[Parameter] | None = None,
        mode: str = "update",
        *,
        check_validity: bool = True,
        include_deprecated: bool = True,
    ):
        """
        Args:
            default (sequence of :class:`~modelrunner.parameters.Parameter`, optional):
                Default configuration values. The default configuration also defines
                what parameters can typically be defined and it provides additional
                information for these parameters.
            mode (str):
                Defines the mode in which the configuration is used. Possible values are

                * `insert`: any new configuration key can be inserted
                * `update`: only the values defined by `default` can be updated
                * `locked`: no values can be changed

                Note that the items specified by `items` will always be inserted,
                independent of the `mode`.
            check_validity (bool):
                Determines whether a `ValueError` is raised if there are keys in
                parameters that are not in the defaults. If `False`, additional items
                are simply stored in `self.parameters`
            include_deprecated (bool):
                Include deprecated parameters
        """
        if default is None:
            default = []
        self._default = default

        # initialize parameters with default ones
        self.mode = "insert"  # temporarily allow inserting items
        super().__init__()
        for param_obj in default:
            if include_deprecated or not isinstance(param_obj, DeprecatedParameter):
                self[param_obj.name] = param_obj.convert(strict=check_validity)

        # set the mode for future additions
        self.mode = mode

    def load(self, path: str | Path):
        """Load configuration from yaml file."""
        import yaml

        with Path(path).open() as fp:
            self.update(yaml.safe_load(fp))

    def save(self, path: str | Path):
        """Save configuration to yaml file."""
        import yaml

        with Path(path).open("w") as fp:
            yaml.dump(self.to_dict(), fp)

    def __getitem__(self, key: str):
        """Retrieve item `key`"""
        parameter = self.data[key]
        if isinstance(parameter, Parameter):
            return parameter.convert()
        else:
            return parameter

    def __setitem__(self, key: str, value):
        """Update item `key` with `value`"""
        if self.mode == "insert":
            self.data[key] = value

        elif self.mode == "update":
            if key not in self:
                raise KeyError(
                    f"{key} is not present, but config is in `{self.mode}` mode"
                )
            self.data[key] = value

        elif self.mode == "locked":
            raise RuntimeError("Configuration is locked")

        else:
            raise ValueError(f"Unsupported configuration mode `{self.mode}`")

    def __delitem__(self, key: str):
        """Removes item `key`"""
        if self.mode == "insert":
            del self.data[key]
        else:
            raise RuntimeError("Configuration is not in `insert` mode")

    def copy(self) -> Config:
        """Return a copy of the configuration."""
        obj = self.__class__(self._default, self.mode)
        obj.update(self)
        return obj

    def to_dict(self) -> dict[str, Any]:
        """Convert the configuration to a simple dictionary.

        Returns:
            dict: A representation of the configuration in a normal :class:`dict`.
        """
        return dict(self.items())

    def __repr__(self) -> str:
        """Represent the configuration as a string."""
        return f"{self.__class__.__name__}({repr(self.to_dict())})"

    @contextlib.contextmanager
    def __call__(self, values: dict[str, Any] | None = None, **kwargs):
        """Context manager temporarily changing the configuration.

        Args:
            values (dict): New configuration parameters
            **kwargs: New configuration parameters
        """
        data_initial = self.to_dict()  # save old configuration
        # set new configuration
        if values is not None:
            self.update(values)
        self.update(kwargs)
        yield  # return to caller
        # restore old configuration
        self.update(data_initial)
