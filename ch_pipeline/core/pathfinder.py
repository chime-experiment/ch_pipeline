"""
Pathfinder Telescope

.. deprecated:: pass1
    Use :mod:`ch_pipeline.core.telescope` instead.
"""
import warnings

# pylint: disable=wildcard-import,unused-wildcard-import
from .telescope import *  # noqa: F401,F403,W401

warnings.warn(
    "`ch_pipeline.core.pathfinder module deprecated. "
    "Use `ch_pipeline.core.telescope` instead"
)
