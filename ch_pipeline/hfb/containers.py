"""HFB-related containers
"""
import numpy as np

from caput import memh5

from ..core.containers import RawContainer, FreqContainer


class HFBData(RawContainer, FreqContainer):
    """A container for HFB data.

    This attempts to wrap the HFB archive format.

    .. note:: This does not yet support distributed loading of HDF5 archive
       files.
    """

    _axes = ("subfreq", "beam")

    _dataset_spec = {
        "hfb": {
            "axes": ["freq", "subfreq", "beam", "time"],
            "dtype": np.float32,
            "initialise": True,
            "distributed": True,
            "distributed_axis": "freq",
        },
        "flags/hfb_weight": {
            "axes": ["freq", "subfreq", "beam", "time"],
            "dtype": np.float32,
            "initialise": True,
            "distributed": True,
            "distributed_axis": "freq",
        },
        "flags/dataset_id": {
            "axes": ["freq", "time"],
            "dtype": "U32",
            "initialise": True,
            "distributed": False,
        },
        "flags/frac_lost": {
            "axes": ["freq", "time"],
            "dtype": np.float32,
            "initialise": False,
            "distributed": False,
        },
    }

    @property
    def hfb(self) -> memh5.MemDataset:
        """The main hfb dataset."""
        return self.datasets["hfb"]

    @property
    def weight(self) -> memh5.MemDataset:
        """The inverse variance weight dataset."""
        return self["flags/hfb_weight"]
