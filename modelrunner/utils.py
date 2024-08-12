"""Miscellaneous utility methods.

.. autosummary::
   :nosignatures:

   hybridmethod
   import_class

.. codeauthor:: David Zwicker <david.zwicker@ds.mpg.de>
"""

from __future__ import annotations

import importlib


class hybridmethod:
    """Decorator to use a method both as a classmethod and an instance method.

    Note:
        The decorator can be used like so:

        .. code-block:: python

            @hybridmethod
            def method(cls, ...): ...

            @method.instancemethod
            def method(self, ...): ...

    Adapted from https://stackoverflow.com/a/28238047
    """

    def __init__(self, fclass, finstance=None, doc=None):
        self.fclass = fclass
        self.finstance = finstance
        self.__doc__ = doc or fclass.__doc__
        # support use on abstract base classes
        self.__isabstractmethod__ = bool(getattr(fclass, "__isabstractmethod__", False))

    def classmethod(self, fclass):
        return type(self)(fclass, self.finstance, None)

    def instancemethod(self, finstance):
        return type(self)(self.fclass, finstance, self.__doc__)

    def __get__(self, instance, cls):
        if instance is None or self.finstance is None:
            # either bound to the class, or no instance method available
            return self.fclass.__get__(cls, None)
        return self.finstance.__get__(instance, cls)


def import_class(identifier: str):
    """Import a class or module given an identifier.

    Args:
        identifier (str):
            The identifier can be a module or a class. For instance, calling the
            function with the string `identifier == 'numpy.linalg.norm'` is
            roughly equivalent to running `from numpy.linalg import norm` and
            would return a reference to `norm`.
    """
    module_path, _, class_name = identifier.rpartition(".")
    if module_path:
        module = importlib.import_module(module_path)
        return getattr(module, class_name)
    else:
        # this happens when identifier does not contain a dot
        return importlib.import_module(class_name)


def is_serial_or_mpi_root() -> bool:
    """Function checking whether the current program is serial or an MPI root node."""
    try:
        from mpi4py import MPI
    except ImportError:
        return True  # assume we are in a serial run
    else:
        return MPI.COMM_WORLD.rank == 0  # check whether we are the root node
