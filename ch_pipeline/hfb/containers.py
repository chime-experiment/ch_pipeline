"""HFB containers."""

from functools import cached_property
from typing import ClassVar

import numpy as np
from caput import memh5, tod
from ch_util import andata
from draco.core.containers import (
    COMPRESSION,
    COMPRESSION_OPTS,
    DataWeightContainer,
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


class HFBCompressed(RawContainer, FreqContainer, HFBBeamContainer):
    """A container for HFB data with compressed weights."""

    _axes = ("subfreq",)

    _dataset_spec = {
        "hfb": {
            "axes": ["freq", "subfreq", "beam", "time"],
            "dtype": np.float32,
            "initialise": True,
            "distributed": True,
            "distributed_axis": "freq",
        },
        "weight_subf": {
            "axes": ["freq", "subfreq", "time"],
            "dtype": np.float32,
            "initialise": True,
            "distributed": True,
            "distributed_axis": "freq",
        },
        "weight_beam": {
            "axes": ["freq", "beam", "time"],
            "dtype": np.float32,
            "initialise": True,
            "distributed": True,
            "distributed_axis": "freq",
        },
        "weight_norm": {
            "axes": ["freq", "time"],
            "dtype": np.float32,
            "initialise": True,
            "distributed": True,
            "distributed_axis": "freq",
        },
        "nsample": {
            "axes": ["freq", "subfreq", "beam", "time"],
            "dtype": np.uint16,
            "initialise": False,
            "distributed": True,
        },
    }

    @property
    def weight_subf(self) -> np.ndarray:
        """Weight vector in subfrequency axis."""
        return self.datasets["weight_subf"]

    @property
    def weight_beam(self) -> np.ndarray:
        """Weight vector in beam axis."""
        return self.datasets["weight_beam"]

    @property
    def weight_norm(self) -> np.ndarray:
        """Weight normalization."""
        return self.datasets["weight_norm"]


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


class HFBDirectionalRFIMaskBitmap(FreqContainer, TODContainer):
    """Container for HFB directional RFI masks.

    Each 32-bit unsigned integer stores four separate 8-bit data segments,
    corresponding to the number of HFB subfrequency channels detecting RFI
    under different values of estimated standard deviation used for detection.
    """

    _axes = ("beam_ns",)

    _dataset_spec: ClassVar = {
        "subfreq_rfi": {
            "axes": ["freq", "beam_ns", "time"],
            "dtype": np.uint32,
            "initialise": True,
            "distributed": True,
            "distributed_axis": "freq",
            "compression": COMPRESSION,
            "compression_opts": COMPRESSION_OPTS,
            "chunks": (64, 128, 512),
            "truncate": False,
        },
    }

    def __init__(self, *args, std_key: list[float] = [], **kwargs):
        """Sets up the bitmap attribute in the packed 32-bit representation."""
        super().__init__(*args, **kwargs)

        # If std_key is provided, validat and store mapping for decoding individual 8-bit RFI segments
        if std_key:
            if len(std_key) != 4:
                raise ValueError(
                    f"Exactly four std values must be provided for packing into 32 bits, but got '{len(std_key)}'."
                )
        self.attrs["bitmap"] = {std: i for i, std in enumerate(std_key)}

    @property
    def bitmap(self):
        """Return the bitmap, the std-to-byte-offset mapping used in the packed 32-bit representation.

        The bitmap is a dictionary mapping each std_key value to an integer in [0, 3],
        indicating which 8-bit segment (out of four) stores the corresponding RFI data.
        """
        return self.attrs["bitmap"]

    @property
    def beam_ns(self):
        """Return the north-south beam index map."""
        return self.index_map["beam_ns"]

    @property
    def subfreq_rfi(self):
        """Return the packed 32-bit unsigned integer subfrequency RFI masks.

        Each 32-bit unsigned integer encodes four independent 8-bit RFI counts, one for each std_key.
        The lowest-order byte (bits 0-7) corresponds to the first std_key,
        the next byte (bits 8-15) to the second, and so on, up to bits 24-31.

        Note: On little-endian systems, this byte order matches the memory layout.
        On big-endian systems, the order in memory will differ, but bit-level encoding remains consistent.
        """
        """Return the packed 32-bit unsigned integer subfrequency RFI masks."""
        return self.datasets["subfreq_rfi"]

    @property
    def mask(self):
        """Disables the property for this container."""
        raise AttributeError(
            "The 'mask' property is not available in HFBDirectionalRFIMaskBitmap. "
            "Use 'get_mask(std_key, subfreq_threshold)' to extract a specific 8-bit mask."
        )

    @property
    def frac_rfi(self):
        """Disables the property for this container."""
        raise AttributeError(
            "The 'frac_rfi' property is not available in HFBDirectionalRFIMaskBitmap. "
            "Use 'get_frac_rfi(std_type)' to extract a specific 8-bit mask."
        )

    def get_subfreq_rfi(self, std_key: float) -> np.ndarray:
        """Extract the 8-bit RFI data for a given std value."""
        if not self.attrs["bitmap"]:
            raise AttributeError(
                "'bitmap' has not been set in attrs. It must be defined to unpack RFI data."
            )

        offset = self.bitmap.get(std_key)

        if offset is None:
            raise KeyError(
                f"Invalid std_key '{std_key}'. Must be one of {list(self.bitmap.keys())}."
            )

        return ((self.subfreq_rfi[:] >> (8 * offset)) & 0xFF).astype(np.uint8)

    def set_subfreq_rfi(self, std_key: float, values: np.ndarray) -> None:
        """Set the 8-bit RFI data for a given beam type."""
        if not self.attrs["bitmap"]:
            raise AttributeError(
                "'bitmap' has not been set in attrs. It must be defined to unpack RFI data."
            )

        offset = self.bitmap.get(std_key)

        if offset is None:
            raise KeyError(
                f"Invalid std_key '{std_key}'. Must be one of {list(self.bitmap.keys())}."
            )
        if np.any((values < 0) | (values > 128)):
            raise ValueError("Values must be in range 0 to 128.")

        # Clear the target byte
        self.subfreq_rfi[:] &= np.uint32(~(0xFF << (8 * offset)) & 0xFFFFFFFF)
        # Set the new values in the correct byte position
        self.subfreq_rfi[:] |= (values.astype(np.uint32) & 0xFF) << (8 * offset)

    def get_mask(self, std_key: float, subfreq_threshold: int) -> np.ndarray:
        """Get a boolean RFI mask for a given std value and subfrequency RFI threshold."""
        return self.get_subfreq_rfi(std_key) >= subfreq_threshold

    def get_frac_rfi(self, std_key: float) -> np.ndarray:
        """Get the fraction of HFB subfrequency channels detecting RFI for a given std value."""
        return self.get_subfreq_rfi(std_key) / 128
