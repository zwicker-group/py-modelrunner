"""
Base class describing a model

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
from .parameters import DeprecatedParameter, HideParameter, Parameterized

if TYPE_CHECKING:
    from ..run.results import Result  # @UnusedImport


class ModelBase(Parameterized, metaclass=ABCMeta):
    """base class for describing models"""

    name: str | None = None
    """str: the name of the model"""
    description: str | None = None
    """str: a longer description of the model"""

    def __init__(
        self,
        parameters: dict[str, Any] | None = None,
        output: str | None = None,
        *,
        mode: ModeType = "insert",
        strict: bool = False,
    ):
        """initialize the parameters of the object

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
        self._logger = logging.getLogger(self.__class__.__name__)

    @property
    def storage(self) -> StorageGroup:
        """:class:`StorageGroup`: Storage to which data can be written"""
        if self._storage is None:
            if self.output is None:
                raise RuntimeError("Output file needs to be specified")
            self._storage = open_storage(self.output, mode=self.mode)
        return self._storage

    def close(self) -> None:
        """close any opened storages"""
        if self._storage is not None:
            self._storage.close()
        self._storage = None

    @abstractmethod
    def __call__(self):
        """main method calculating the result. Needs to be specified by sub-class"""

    def get_result(self, data: Any = None) -> Result:
        """get the result as a :class:`~model.Result` object

        Args:
            data:
                The result data. If omitted, the model is run to obtain results

        Returns:
            :class:`Result`: The result after the model is run
        """
        from ..run.results import Result  # @Reimport

        if data is None:
            data = self()

        info = {"time": datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
        return Result(self, data, info=info)

    def write_result(self, result: Result | None = None) -> None:
        """write the result to the output file

        Args:
            result:
                The result data. If omitted, the model is run to obtain results
        """
        from ..run.results import Result  # @Reimport

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
        """create argument parser for setting parameters of this model

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
        """create model from command line parameters

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
        """run model using command line parameters

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
        if mdl.output:
            # write the results to a file
            mdl.write_result(result=result)
        # else:
        #     # display the results on stdout
        #     storage = MemoryStorage()
        #     result.to_file(storage)

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
