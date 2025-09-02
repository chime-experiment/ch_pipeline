"""
Submodules
==========

.. autosummary::
    :toctree: _autosummary

    analysis
    core
    processing
    synthesis

"""

from importlib.metadata import version, PackageNotFoundError

try:
    __version__ = version("ch_pipeline")
except PackageNotFoundError:
    # package is not installed
    pass
del version, PackageNotFoundError
