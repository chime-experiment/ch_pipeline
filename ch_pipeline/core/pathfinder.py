"""Pathfinder Telescope.

.. deprecated:: pass1
    Use :mod:`ch_pipeline.core.telescope` instead.
"""

import warnings

from .telescope import *  # noqa

warnings.warn(
    "`ch_pipeline.core.pathfinder module deprecated. "
    "Use `ch_pipeline.core.telescope` instead"
)
