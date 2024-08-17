"""Defines stroages, which contain store objects in a hierarchical format.

.. autosummary::
   :nosignatures:

   ~modelrunner.storage.tools.open_storage
   ~modelrunner.storage.trajectory.Trajectory
   ~modelrunner.storage.trajectory.TrajectoryWriter

.. codeauthor:: David Zwicker <david.zwicker@ds.mpg.de>
"""

from .access_modes import AccessError, ModeType
from .attributes import Attrs
from .backend import *
from .base import StorageBase
from .group import StorageGroup
from .tools import StorageID, open_storage
from .trajectory import Trajectory, TrajectoryWriter
from .utils import Location, storage_actions
