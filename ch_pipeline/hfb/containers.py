"""HFB containers."""

from functools import cached_property
from typing import ClassVar

import numpy as np
from caput import memh5, tod
from caput.containers import (
    COMPRESSION,
    COMPRESSION_OPTS,
    DataWeightContainer,
)
from ch_util import andata
from draco.core.containers import (
    SiderealContainer,
    TODContainer,
)

from ..core.containers import FreqContainer, RawContainer


class HFBContainer(DataWeightContainer):
    """A base class for all HFB containers.

    Like :class:`ContainerBase`, but with some properties specific to HFB data.
    """

    _data_dset_name = "hfb"
    _weight_dset_name = None  # Leave as None as this could potentially change location

    @property
    def hfb(self) -> memh5.MemDataset:
        """Convenience access to the main hfb dataset."""
        if "hfb" in self.datasets:
            return self.datasets["hfb"]

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
        """The unique NS-beam indices (i.e., from 0 to 255) in the beam axis."""
        return np.unique(self.beam % 256)


class HFBData(RawContainer, FreqContainer, HFBBeamContainer):
    """A container for HFB data.

    This attempts to wrap the HFB archive format.

    .. note:: This does not yet support distributed loading of HDF5 archive
       files.
    """

    _axes = ("subfreq",)

    _dataset_spec: ClassVar = {
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

    @classmethod
    def from_file(cls, *args, **kwargs) -> "HFBData":
        """Load an HFB file.

        This overrides the default implementation to forcibly hint that the datasets
        should be distributed.
        """
        # If hints exist, then don't modify them. This allows hints=False to override this override!
        if "hints" not in kwargs:
            hints = {}

            # Try and extract the hint information from the dataset spec
            # TODO: move this into base classes
            for dname, dspec in cls._class_dataset_spec().items():
                if not dspec.get("distributed", False):
                    continue

                hspec = {"distributed": True}

                if "distributed_axis" in dspec:
                    ax = dspec["distributed_axis"]

                    if isinstance(ax, str) and ax in dspec["axes"]:
                        ax = dspec["axes"].index(ax)

                    hspec["axis"] = ax

                dname = dname if dname[0] == "/" else "/" + dname
                hints[dname] = hspec
                kwargs["hints"] = hints

        return super().from_file(*args, **kwargs)


class HFBReader(tod.Reader):
    """A reader for HFB type data."""

    data_class = HFBData

    _freq_sel = None

    @property
    def freq_sel(self) -> int | list | slice:
        """Get the current frequency selection.

        Returns
        -------
        freq_sel
            A frequency selection.
        """
        return self._freq_sel

    @freq_sel.setter
    def freq_sel(self, value: int | list | slice):
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
    """Container for holding a mask indicating HFB data free of RFI events.

    The `sens` dataset (if initialized) holds the sensitivity metric data.
    """

    _axes = ("subfreq",)

    _dataset_spec: ClassVar = {
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
        """Get the mask dataset."""
        return self.datasets["mask"]

    @property
    def sens(self):
        """Get the sensitivity metric dataset."""
        return self.datasets["sens"]


class HFBTimeAverage(FreqContainer, HFBBeamContainer):
    """Container for holding average data for flattening sub-frequency band shape."""

    _axes = ("subfreq",)

    _dataset_spec: ClassVar = {
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

    _dataset_spec: ClassVar = {
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

    _dataset_spec: ClassVar = {
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

    _dataset_spec: ClassVar = {
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

    This container includes an axis to mark the indices of the NS beams, as well as
    RA (inherited from :class:`SiderealContainer`) and el = sin(zenith angle) axes.

    The el axis corresponds to the sin(za) of the reference angles for the NS beams.
    The true el for a given bit of data also depends on the frequency and can be
    computed from the NS beam index and frequency using the synthetic beam model.
    """

    _axes = ("beam_ns", "el")

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


class HFBBeamRingMap(HFBRingMapBase):
    """Base class for HFB ringmaps that have separate EW beams.

    This container includes an axis to mark the indices of the EW beams.
    """

    _axes = ("beam_ew",)

    @property
    def beam_ew(self):
        """The (unique) EW beam indices (i.e., from 0 to 3) of the beam_ew axis."""
        return self.index_map["beam_ew"]


class HFBRingMap(FreqContainer, HFBBeamRingMap):
    """Container for holding HFB ringmap data."""

    _axes = ("subfreq",)

    _dataset_spec: ClassVar = {
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


class HFBHighResRingMap(HFBBeamRingMap, HFBHighResContainer):
    """Container for holding high-resolution frequency ringmap data.

    With respect to :class:`HFBRingMap`, the (combined) frequency axis is moved
    to the back, and the distributed axis is changed to the el = sin(za) axis.
    This is because further downstream in the pipeline, we will look for features
    along the frequency axis.
    """

    _dataset_spec: ClassVar = {
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


class HFBHighResBeamAvgRingMap(HFBRingMapBase, HFBHighResContainer):
    """Container for holding EW-beam-averaged high-resolution frequency ringmap data."""

    _dataset_spec: ClassVar = {
        "hfb": {
            "axes": ["el", "ra", "freq"],
            "dtype": np.float32,
            "initialise": True,
            "distributed": True,
            "distributed_axis": "el",
        },
        "weight": {
            "axes": ["el", "ra", "freq"],
            "dtype": np.float32,
            "initialise": True,
            "distributed": True,
            "distributed_axis": "el",
        },
        "nsample": {
            "axes": ["el", "ra", "freq"],
            "dtype": np.uint16,
            "initialise": False,
            "distributed": True,
            "distributed_axis": "el",
        },
    }


class HFBSearchResult(HFBRingMapBase, HFBHighResContainer):
    """Container for holding results of blind search."""

    _axes = ("width",)

    _dataset_spec: ClassVar = {
        "ln_lambda": {
            "axes": ["width", "beam_ew", "el", "ra", "freq"],
            "dtype": np.float32,
            "initialise": True,
            "distributed": True,
            "distributed_axis": "el",
        },
        "amplitude": {
            "axes": ["width", "beam_ew", "el", "ra", "freq"],
            "dtype": np.float32,
            "initialise": True,
            "distributed": True,
            "distributed_axis": "el",
        },
    }

    @property
    def ln_lambda(self):
        """The log-likelihood-ratio dataset."""
        return self.datasets["ln_lambda"]

    @property
    def amplitude(self):
        """The absorption-feature-amplitude dataset."""
        return self.datasets["amplitude"]


class HFBDirectionalRFIMask(FreqContainer, TODContainer):
    """Container for an RFI mask based on HFB data.

    This RFI mask takes advantage of the high-frequency resolution of HFB data.
    It can also store the fraction of subfrequency channels (out of 128)
    that detected RFI events.

    """

    _axes = ("beam_ns",)

    _dataset_spec: ClassVar = {
        "mask": {
            "axes": ["freq", "beam_ns", "time"],
            "dtype": bool,
            "initialise": True,
            "distributed": True,
            "distributed_axis": "freq",
        },
        "frac_rfi": {
            "axes": ["freq", "beam_ns", "time"],
            "dtype": np.float32,
            "initialise": False,
            "distributed": True,
            "distributed_axis": "freq",
            "compression": COMPRESSION,
            "compression_opts": COMPRESSION_OPTS,
            "chunks": (64, 128, 512),
            "truncate": True,
        },
    }

    @property
    def mask(self):
        """Return the north-south beam-dependent mask."""
        return self.datasets["mask"]

    @property
    def frac_rfi(self):
        """Return the north-south beam-dependent rfi frac dataset."""
        return self.datasets["frac_rfi"]

    @property
    def beam_ns(self):
        """Return the north-south beam index map."""
        return self.index_map["beam_ns"]
