# === Start Python 2/3 compatibility
from __future__ import absolute_import, division, print_function, unicode_literals
from future.builtins import *  # noqa  pylint: disable=W0401, W0614
from future.builtins.disabled import *  # noqa  pylint: disable=W0401, W0614

# === End Python 2/3 compatibility

import warnings
from .telescope import *

warnings.warn(
    "`ch_pipeline.core.pathfinder module deprecated. "
    "Use `ch_pipeline.core.telescope` instead"
)