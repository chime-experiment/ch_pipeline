"""HFB containers
"""

from caput.cache import cached_property
from typing import Union

import numpy as np

from caput import tod
from caput import memh5

from ch_util import andata

from ..core.containers import ContainerBase, RawContainer, FreqContainer

from draco.core.containers import TODContainer, SiderealContainer


class HFBContainer(ContainerBase):
    """A base class for all HFB containers.

    Like :class:`ContainerBase`, but with some properties specific to HFB data.
    """

    @property
    def hfb(self) -> memh5.MemDataset:
        """Convenience access to the main hfb dataset."""
        if "hfb" in self.datasets:
            return self.datasets["hfb"]
        else:
            raise KeyError("Dataset 'hfb' not initialised.")

    @property
    def weight(self) -> memh5.MemDataset:
        """The inverse variance weight dataset."""
        if "weight" in self:
            weight = self["weight"]
        elif "hfb_weight" in self:
            weight = self["hfb_weight"]
        elif "flags" in self and "hfb_weight" in self["flags"]:
            weight = self["flags/hfb_weight"]
        else:
            raise KeyError("Cannot find weight dataset.")
        return weight

    @property
    def nsample(self) -> memh5.MemDataset:
        """Get the nsample dataset (number of non-zero samples) if it exists."""
        if "nsample" in self.datasets:
            return self.datasets["nsample"]
        else:
            raise KeyError("Dataset 'nsample' not initialised.")


class HFBBeamContainer(HFBContainer):
    """A pipeline container for HFB data with a beam axis.

    This works like a normal :class:`HFBContainer` container, but already has a beam
    axis defined, and specific properties for dealing with beams.
    """

    _axes = ("beam",)

    @property
    def beam(self) -> np.ndarray:
        """The beam indices associated with each entry of the beam axis."""
        return self.index_map["beam"]

    @cached_property
    def beam_ew(self) -> np.ndarray:
        """The unique EW-beam indices (i.e., from 0 to 3) in the beam axis."""
        return np.unique(self.beam // 256)

    @cached_property
    def beam_ns(self) -> np.ndarray:
        """The unique NS-beam indices (i.e., from 0 to 256) in the beam axis."""
        return np.unique(self.beam % 256)


class HFBData(RawContainer, FreqContainer, HFBBeamContainer):
    """A container for HFB data.

    This attempts to wrap the HFB archive format.

    .. note:: This does not yet support distributed loading of HDF5 archive
       files.
    """

    _axes = ("subfreq",)

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
        "nsample": {
            "axes": ["freq", "subfreq", "beam", "time"],
            "dtype": np.uint16,
            "initialise": False,
            "distributed": True,
        },
    }


class HFBReader(tod.Reader):
    """A reader for HFB type data."""

    data_class = HFBData

    _freq_sel = None

    @property
    def freq_sel(self) -> Union[int, list, slice]:
        """Get the current frequency selection.

        Returns
        -------
        freq_sel
            A frequency selection.
        """

        return self._freq_sel

    @freq_sel.setter
    def freq_sel(self, value: Union[int, list, slice]):
        """Set a frequency selection.

        Parameters
        ----------
        value
            Any type accepted by h5py is valid.
        """
        self._freq_sel = andata._ensure_1D_selection(value)

    _beam_sel = None

    @property
    def beam_sel(self):
        """Get the current beam selection.

        Returns
        -------
        beam_sel
            The current beam selection.
        """

        return self._beam_sel

    @beam_sel.setter
    def beam_sel(self, value):
        """Set a beam selection.

        Parameters
        ----------
        value
            Any type accepted by h5py is valid.
        """
        self._beam_sel = andata._ensure_1D_selection(value)

    def read(self, out_group=None):
        """Read the selected data.

        Parameters
        ----------
        out_group : `h5py.Group`, hdf5 filename or `memh5.Group`
            Underlying hdf5 like container that will store the data for the
            BaseData instance.

        Returns
        -------
        data : :class:`TOData`
            Data read from :attr:`~Reader.files` based on the selections made
            by user.

        """
        kwargs = {}

        if self._freq_sel is not None:
            kwargs["freq_sel"] = self._freq_sel

        if self._beam_sel is not None:
            kwargs["beam_sel"] = self._beam_sel

        kwargs["ondisk"] = False

        return self.data_class.from_mult_files(
            self.files,
            data_group=out_group,
            start=self.time_sel[0],
            stop=self.time_sel[1],
            datasets=self.dataset_sel,
            **kwargs,
        )


class HFBRFIMask(TODContainer, FreqContainer):
    """Container for holding a mask that indicates HFB data that is free of
    RFI events.

    The `sens` dataset (if initialized) holds the sensitivity metric data.
    """

    _axes = ("subfreq",)

    _dataset_spec = {
        "mask": {
            "axes": ["freq", "subfreq", "time"],
            "dtype": bool,
            "initialise": True,
            "distributed": True,
            "distributed_axis": "freq",
        },
        "sens": {
            "axes": ["freq", "subfreq", "time"],
            "dtype": np.float32,
            "initialise": False,
            "distributed": True,
            "distributed_axis": "freq",
        },
    }

    @property
    def mask(self):
        return self.datasets["mask"]

    @property
    def sens(self):
        return self.datasets["sens"]


class HFBTimeAverage(FreqContainer, HFBBeamContainer):
    """Container for holding average data for flattening sub-frequency band shape."""

    _axes = ("subfreq",)

    _dataset_spec = {
        "hfb": {
            "axes": ["freq", "subfreq", "beam"],
            "dtype": np.float32,
            "initialise": True,
            "distributed": True,
            "distributed_axis": "freq",
        },
        "weight": {
            "axes": ["freq", "subfreq", "beam"],
            "dtype": np.float32,
            "initialise": True,
            "distributed": True,
            "distributed_axis": "freq",
        },
        "nsample": {
            "axes": ["freq", "subfreq", "beam"],
            "dtype": np.uint16,
            "initialise": False,
            "distributed": True,
        },
    }


class HFBHighResContainer(FreqContainer, HFBContainer):
    """Base class for HFB containers with high-resolution frequency data."""


class HFBHighResData(TODContainer, HFBHighResContainer, HFBBeamContainer):
    """Container for holding high-resolution frequency data."""

    _dataset_spec = {
        "hfb": {
            "axes": ["freq", "beam", "time"],
            "dtype": np.float32,
            "initialise": True,
            "distributed": True,
            "distributed_axis": "freq",
        },
        "weight": {
            "axes": ["freq", "beam", "time"],
            "dtype": np.float32,
            "initialise": True,
            "distributed": True,
            "distributed_axis": "freq",
        },
        "nsample": {
            "axes": ["freq", "beam", "time"],
            "dtype": np.uint16,
            "initialise": False,
            "distributed": True,
        },
    }


class HFBHighResTimeAverage(HFBHighResContainer, HFBBeamContainer):
    """Container for holding time-averaged high-resolution frequency data."""

    _dataset_spec = {
        "hfb": {
            "axes": ["freq", "beam"],
            "dtype": np.float32,
            "initialise": True,
            "distributed": True,
            "distributed_axis": "freq",
        },
        "weight": {
            "axes": ["freq", "beam"],
            "dtype": np.float32,
            "initialise": True,
            "distributed": True,
            "distributed_axis": "freq",
        },
        "nsample": {
            "axes": ["freq", "beam"],
            "dtype": np.uint16,
            "initialise": False,
            "distributed": True,
        },
    }


class HFBHighResSpectrum(HFBHighResContainer):
    """Container for holding high-resolution frequency spectrum."""

    _dataset_spec = {
        "hfb": {
            "axes": ["freq"],
            "dtype": np.float32,
            "initialise": True,
            "distributed": False,
        },
        "weight": {
            "axes": ["freq"],
            "dtype": np.float32,
            "initialise": True,
            "distributed": False,
        },
        "nsample": {
            "axes": ["freq"],
            "dtype": np.uint16,
            "initialise": False,
            "distributed": False,
        },
    }


class HFBRingMapBase(SiderealContainer, HFBContainer):
    """Base class for HFB ringmaps.

    These containers include axes to mark the EW and NS beam indices, as well as
    RA (inherited from :class:`SiderealContainer`) and el = sin(zenith angle) axes.

    The el axis corresponds to the sin(za) of the reference angles for the NS beams.
    The true el for a given bit of data also depends on the frequency and can be
    computed from the NS beam index and frequency using the synthetic beam model.
    """

    _axes = ("beam_ew", "beam_ns", "el")

    @property
    def beam_ew(self) -> np.ndarray:
        """The (unique) EW beam indices (i.e., from 0 to 3) of the beam_ew axis."""
        return self.index_map["beam_ew"]

    @property
    def beam_ns(self) -> np.ndarray:
        """The (unique) NS beam indices (i.e., from 0 to 256) of the beam_ns axis."""
        return self.index_map["beam_ns"]

    @property
    def el(self) -> np.ndarray:
        """The el = sin(zenith angle) associated with each sample of the el axis.

        The zenith angle used is the reference angle for the NS beam in question.
        The true el of a data sample can be computed from the NS beam index and
        the sample's frequency using the synthetic beam model.
        """
        return self.index_map["el"]

    @property
    def ra(self) -> np.ndarray:
        """The RA in degrees associated with each sample of the RA axis.

        This is valid for EW beam index 1. For other EW beams, there is an
        offset in RA that depends on the EW and NS beam index.
        """
        return self.index_map["ra"]


class HFBRingMap(FreqContainer, HFBRingMapBase):
    """Container for holding HFB ringmap data."""

    _axes = ("subfreq",)

    _dataset_spec = {
        "hfb": {
            "axes": ["freq", "subfreq", "beam_ew", "el", "ra"],
            "dtype": np.float32,
            "initialise": True,
            "distributed": True,
            "distributed_axis": "freq",
        },
        "weight": {
            "axes": ["freq", "subfreq", "beam_ew", "el", "ra"],
            "dtype": np.float32,
            "initialise": True,
            "distributed": True,
            "distributed_axis": "freq",
        },
        "nsample": {
            "axes": ["freq", "subfreq", "beam_ew", "el", "ra"],
            "dtype": np.uint16,
            "initialise": False,
            "distributed": True,
            "distributed_axis": "freq",
        },
    }


class HFBHighResRingMap(HFBRingMapBase, HFBHighResContainer):
    """Container for holding high-resolution frequency ringmap data.

    With respect to :class:`HFBRingMap`, the (combined) frequency axis is moved
    to the back, and the distributed axis is changed to the el = sin(za) axis.
    This is because further downstream in the pipeline, we will look for features
    along the frequency axis.
    """

    _dataset_spec = {
        "hfb": {
            "axes": ["beam_ew", "el", "ra", "freq"],
            "dtype": np.float32,
            "initialise": True,
            "distributed": True,
            "distributed_axis": "el",
        },
        "weight": {
            "axes": ["beam_ew", "el", "ra", "freq"],
            "dtype": np.float32,
            "initialise": True,
            "distributed": True,
            "distributed_axis": "el",
        },
        "nsample": {
            "axes": ["beam_ew", "el", "ra", "freq"],
            "dtype": np.uint16,
            "initialise": False,
            "distributed": True,
            "distributed_axis": "el",
        },
    }


class HFBSearchResult(HFBRingMapBase, HFBHighResContainer):
    """Container for holding results of blind search."""

    _dataset_spec = {
        "max_snr": {
            "axes": ["beam_ew", "el", "ra", "freq"],
            "dtype": np.float32,
            "initialise": True,
            "distributed": True,
            "distributed_axis": "el",
        },
        "best_width": {
            "axes": ["beam_ew", "el", "ra", "freq"],
            "dtype": np.float32,
            "initialise": True,
            "distributed": True,
            "distributed_axis": "el",
        },
    }

    @property
    def max_snr(self):
        """Get the dataset of maximum SNR over template widths."""
        return self.datasets["max_snr"]

    @property
    def best_width(self):
        """Get the dataset of the template width corresponding to the maximum SNR."""
        return self.datasets["best_width"]
