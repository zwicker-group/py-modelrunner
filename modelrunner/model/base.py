"""Base class describing a model.

.. codeauthor:: David Zwicker <david.zwicker@ds.mpg.de>
"""

from __future__ import annotations

import argparse
import json
import logging
from abc import ABCMeta, abstractmethod
from datetime import datetime
from typing import TYPE_CHECKING, Any, Sequence

from ..storage import ModeType, StorageGroup, open_storage
from ..utils import is_serial_or_mpi_root
from .parameters import DeprecatedParameter, HideParameter, Parameterized

if TYPE_CHECKING:
    from ..run.results import Result

_base_logger = logging.getLogger(__name__.rsplit(".", 1)[0])
""":class:`logging.Logger`: Base logger for models."""


class ModelBase(Parameterized, metaclass=ABCMeta):
    """Base class for describing models."""

    name: str | None = None
    """str: the name of the model"""
    description: str | None = None
    """str: a longer description of the model"""
    _logger: logging.Logger  # logger instance to output information

    def __init__(
        self,
        parameters: dict[str, Any] | None = None,
        output: str | None = None,
        *,
        mode: ModeType = "insert",
        strict: bool = False,
    ):
        """Initialize the parameters of the object.

        Args:
            parameters (dict):
                A dictionary of parameters to change the defaults of this model. The
                allowed parameters can be obtained from
                :meth:`~Parameterized.get_parameters` or displayed by calling
                :meth:`~Parameterized.show_parameters`.
            output (str):
                Path where the output file will be written. The output will be written
                using :mod:`~modelrunner.storage` and might contain two groups: `result`
                to which the final result of the model is written, and `data`, which
                can contain extra information that is written using
                :attr:`~ModelBase.storage`.
            mode (str or :class:`~modelrunner.storage.access_modes.ModeType`):
                The file mode with which the storage is accessed, which determines the
                allowed operations. Common options are "read", "full", "append", and
                "truncate".
            strict (bool):
                Flag indicating whether parameters are strictly interpreted. If `True`,
                only parameters listed in `parameters_default` can be set and their type
                will be enforced.
        """
        super().__init__(parameters, strict=strict)
        self.output = output  # TODO: also allow already opened storages
        self.mode = mode
        self._storage: open_storage | None = None

    def __init_subclass__(cls, **kwargs):
        """Initialize class-level attributes of subclasses."""
        super().__init_subclass__(**kwargs)
        # create logger for this specific field class
        cls._logger = _base_logger.getChild(cls.__qualname__)

    @property
    def storage(self) -> StorageGroup:
        """:class:`StorageGroup`: Storage to which data can be written."""
        if self._storage is None:
            if self.output is None:
                raise RuntimeError("Output file needs to be specified")
            self._storage = open_storage(self.output, mode=self.mode)
        return self._storage

    def close(self) -> None:
        """Close any opened storages."""
        if self._storage is not None:
            self._storage.close()
        self._storage = None

    @abstractmethod
    def __call__(self):
        """Main method calculating the result.

        Needs to be specified by sub-class
        """

    def get_result(self, data: Any = None) -> Result:
        """Get the result as a :class:`~model.Result` object.

        Args:
            data:
                The result data. If omitted, the model is run to obtain results

        Returns:
            :class:`Result`: The result after the model is run
        """
        from ..run.results import Result

        if data is None:
            data = self()

        info = {"time": datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
        return Result(self, data, info=info)

    def write_result(self, result: Result | None = None) -> None:
        """Write the result to the output file.

        Args:
            result:
                The result data. If omitted, the model is run to obtain results
        """
        from ..run.results import Result

        if self.output is None:
            raise RuntimeError("Output file needs to be specified")

        if result is None:
            result = self.get_result()
        elif not isinstance(result, Result):
            raise TypeError(f"result has type {result.__class__} instead of `Result`")

        if self._storage is not None:
            # reuse the opened storage
            result.to_file(self._storage)
        else:
            result.to_file(self.output, mode=self.mode)

    @classmethod
    def _prepare_argparser(cls, name: str | None = None) -> argparse.ArgumentParser:
        """Create argument parser for setting parameters of this model.

        Args:
            name (str):
                Name of the program, which will be shown in the command line help

        Returns:
            :class:`~argparse.ArgumentParser`
        """
        parser = argparse.ArgumentParser(
            prog=name,
            description=cls.description,
            formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        )

        group = parser.add_argument_group()
        seen = set()
        # iterate over all parent classes
        for cls1 in cls.__mro__:
            if hasattr(cls1, "parameters_default"):
                # add all parameters of this class
                for p in cls1.parameters_default:
                    if p.name not in seen:
                        if not isinstance(p, (HideParameter, DeprecatedParameter)):
                            p._argparser_add(group)
                        seen.add(p.name)

        # add special parameters
        parser.add_argument(
            "--json",
            metavar="JSON",
            help="JSON-encoded parameter values. Overwrites other parameters.",
        )

        return parser

    @classmethod
    def from_command_line(
        cls, args: Sequence[str] | None = None, name: str | None = None
    ) -> ModelBase:
        """Create model from command line parameters.

        Args:
            args (list):
                Sequence of strings corresponding to the command line arguments
            name (str):
                Name of the program, which will be shown in the command line help

        Returns:
            :class:`ModelBase`: An instance of this model with appropriate parameters
        """
        if args is None:
            args = []

        # read the command line arguments
        parser = cls._prepare_argparser(name)

        # add special parameters to determine the output file
        parser.add_argument(
            "-o",
            "--output",
            metavar="PATH",
            help="Path to output file. If omitted, no output file is created.",
        )

        parameters = vars(parser.parse_args(args))
        output = parameters.pop("output")
        parameters_json = parameters.pop("json")

        # update parameters with data from the json argument
        if parameters_json:
            parameters.update(json.loads(parameters_json))

        # create the model
        return cls(parameters, output=output)

    @classmethod
    def run_from_command_line(
        cls, args: Sequence[str] | None = None, name: str | None = None
    ) -> Result:
        """Run model using command line parameters.

        Args:
            args (list):
                Sequence of strings corresponding to the command line arguments
            name (str):
                Name of the program, which will be shown in the command line help

        Returns:
            :class:`Result`: The result of running the model
        """
        # create model from command line parameters
        mdl = cls.from_command_line(args, name)

        # run the model
        result = mdl.get_result()

        # write the results
        if mdl.output and is_serial_or_mpi_root():
            # Write the results to a file if `output` is specified and if we are on the
            # root node of an MPI run (or a serial program). The second check is a
            # safe-guard against writing data on sub-nodes during an MPI program.
            mdl.write_result(result=result)

        # close the output file
        mdl.close()

        return result

    @property
    def _state_attributes(self) -> dict[str, Any]:
        """dict: information about the element state, which does not change in time"""
        return {
            "class": self.__class__.__name__,
            "name": self.name,
            "description": self.description,
            "parameters": self.parameters,
        }
