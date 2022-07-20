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


def __getattr__(name: str):
    """Custom handling for __version__ global variable."""
    if name == "__version__":
        # No need to re-get __version__ if it already exists
        v = globals().get(name, None)
        from ._version import get_versions

        return v if v else get_versions()["version"]

    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
