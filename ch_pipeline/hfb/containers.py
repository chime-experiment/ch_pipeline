"""HFB containers."""

import re
from functools import cached_property
from typing import ClassVar

import numpy as np
from caput import memdata
from caput.containers import tod
from ch_util import andata
from draco.core.containers import (
    COMPRESSION,
    COMPRESSION_OPTS,
    DataWeightContainer,
    SiderealContainer,
    SourceCatalog,
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
    def hfb(self) -> memdata.MemDataset:
        """Convenience access to the main hfb dataset."""
        if "hfb" in self.datasets:
            return self.datasets["hfb"]

        raise KeyError("Dataset 'hfb' not initialised.")

    @property
    def weight(self) -> memdata.MemDataset:
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
    def nsample(self) -> memdata.MemDataset:
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


class HFBReader(tod.TODReader):
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
        out_group : `h5py.Group`, hdf5 filename or `memdata.Group`
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
    under different values of significance used for detection.
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

    def __init__(self, *args, sigma_key: list[float] | None = None, **kwargs):
        """Sets up the bitmap attribute in the packed 32-bit representation."""
        super().__init__(*args, **kwargs)

        # If sigma_key is provided
        if sigma_key is not None:

            # Sort in increasing order
            sigma_sorted = sorted(float(sigma) for sigma in sigma_key)

            # De-duplicate
            sigma_unique: list[float] = []
            duplicates: list[float] = []
            for sigma in sigma_sorted:
                if any(np.isclose(s, sigma, rtol=0, atol=0.001) for s in sigma_unique):
                    duplicates.append(sigma)
                else:
                    sigma_unique.append(sigma)

            # If duplicates exist, print them and proceed with unique values only
            if duplicates:
                self.log.warning(
                    f"Duplicate sigma_key values provided (within atol=0.001): {duplicates}. "
                    f"Using unique values only: {sigma_unique}."
                )

            # Validate after de-duplication
            if len(sigma_unique) == 0:
                raise ValueError(
                    "sigma_key must contain at least one unique value (within atol=0.001)."
                )
            if len(sigma_unique) > 4:
                raise ValueError(
                    f"sigma_key must contain at most 4 unique values, but got {len(sigma_unique)}."
                )
            if any(sigma <= 0 for sigma in sigma_unique):
                raise ValueError("All sigma_key values must be strictly positive.")

            # Store mapping for decoding individual 8-bit RFI segments
            self.attrs["bitmap"] = {
                float(sigma): i for i, sigma in enumerate(sigma_key)
            }

    @property
    def bitmap(self):
        """Return the bitmap, the sigma-to-byte-offset mapping used in the packed 32-bit representation.

        The bitmap is a dictionary mapping each sigma_key value to an integer in [0, 3],
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

        Each 32-bit unsigned integer encodes four independent 8-bit RFI counts, one for each sigma_key.
        The lowest-order byte (bits 0-7) corresponds to the first sigma_key,
        the next byte (bits 8-15) to the second, and so on, up to bits 24-31.

        Note: On little-endian systems, this byte order matches the memory layout.
        On big-endian systems, the order in memory will differ, but bit-level encoding remains consistent.
        """
        return self.datasets["subfreq_rfi"]

    @property
    def mask(self):
        """Disables the property for this container."""
        raise AttributeError(
            "The 'mask' property is not available in HFBDirectionalRFIMaskBitmap. "
            "Use 'get_mask(sigma_key, subfreq_threshold)' to extract a specific 8-bit mask."
        )

    @property
    def frac_rfi(self):
        """Disables the property for this container."""
        raise AttributeError(
            "The 'frac_rfi' property is not available in HFBDirectionalRFIMaskBitmap. "
            "Use 'get_frac_rfi(sigma_type)' to extract a specific 8-bit mask."
        )

    def get_subfreq_rfi(self, sigma_key: float) -> np.ndarray:
        """Extract the 8-bit RFI data for a given sigma value."""
        if not self.attrs["bitmap"]:
            raise AttributeError(
                "'bitmap' has not been set in attrs. It must be defined to unpack RFI data."
            )

        offset = None
        for k, v in self.bitmap.items():
            if np.isclose(float(k), float(sigma_key), rtol=0, atol=0.001):
                offset = v
                break
        else:
            raise KeyError(
                f"Invalid sigma_key {sigma_key}. Must be one of {list(self.bitmap.keys())}."
            )

        return ((self.subfreq_rfi[:] >> (8 * offset)) & 0xFF).astype(np.uint8)

    def set_subfreq_rfi(self, sigma_key: float, values: np.ndarray) -> None:
        """Set the 8-bit RFI data for a given beam type."""
        if not self.attrs["bitmap"]:
            raise AttributeError(
                "'bitmap' has not been set in attrs. It must be defined to unpack RFI data."
            )

        offset = None
        for k, v in self.bitmap.items():
            if np.isclose(float(k), float(sigma_key), rtol=0, atol=0.001):
                offset = v
                break
        else:
            raise KeyError(
                f"Invalid sigma_key {sigma_key}. Must be one of {list(self.bitmap.keys())}."
            )

        if np.any((values < 0) | (values > 128)):
            raise ValueError("Values must be in range 0 to 128.")

        # Clear the target byte
        self.subfreq_rfi[:] &= np.uint32(~(0xFF << (8 * offset)) & 0xFFFFFFFF)
        # Set the new values in the correct byte position
        self.subfreq_rfi[:] |= (values.astype(np.uint32) & 0xFF) << (8 * offset)

    def get_mask(
        self,
        sigma_key: float,
        subfreq_threshold: int,
        *,
        remove_persistent_beamns_frac: float | None = None,
    ) -> np.ndarray:
        """Return a boolean RFI mask for a given sigma value and subfrequency threshold.

        If desired, beam_ns rows that are persistently flagged across time are removed.
        These are interpreted as instrumental offsets or calibration effects rather
        than physical RFI.

        Parameters
        ----------
        sigma_key : float
            Sigma value identifying which 8-bit RFI bitmap to extract.
            Must match one of the values provided (within atol=0.001).
        subfreq_threshold : int
            Minimum number of flagged HFB subfrequency channels required for a
            time-frequency-beam sample to be considered RFI-contaminated.
        remove_persistent_beamns_frac : float or None, optional
            If provided, beam_ns rows that are flagged for more than this fraction
            of time samples are treated as persistent instrumental contamination
            or calibration offset rather than RFI, and removed from the mask.
            Must be in the range [0, 1]. If None, no persistent beam_ns removal is applied.

        Returns
        -------
        mask : np.ndarray
            Boolean array of shape ``(freq, beam_ns, time)`` indicating RFI-flagged
            samples.
        """
        mask = (
            self.get_subfreq_rfi(sigma_key) >= subfreq_threshold
        )  # mask shape: (freq, beam_ns, time)

        if remove_persistent_beamns_frac is not None:
            if not (0.0 <= remove_persistent_beamns_frac <= 1.0):
                raise ValueError("remove_persistent_beamns_frac must be in [0, 1].")

            ntime = mask.shape[-1]
            persistent = (
                mask.sum(axis=-1) > remove_persistent_beamns_frac * ntime
            )  # persistent shape: (freq, beam_ns)

            mask &= ~persistent[..., np.newaxis]

        return mask

    def get_frac_rfi(self, sigma_key: float) -> np.ndarray:
        """Get the fraction of HFB subfrequency channels detecting RFI for a given sigma value."""
        return self.get_subfreq_rfi(sigma_key) / 128


class AbsorberCatalogue(SourceCatalog):
    """A catalogue of absorbers (known and candidate).

    Required per-entry values are: 'ra' (degrees), 'dec' (degrees), 'freq' (MHz),
    and 'status'. The 'amplitude' is optional: when unknown it is stored as
    NaN, and when provided it must be a finite float.

    Status values
    -------------
    Allowed values are "confirmed", "false_positive", "control", and
    candidates. A candidate records the S/N of its detection in the status
    itself, e.g. "candidate_snr5" or "candidate_snr7" (plain "candidate" is
    also allowed if the S/N is unknown). "control" marks bright continuum
    test sources (e.g. Cyg A) or narrowband RFI used to validate the pipeline.
    """

    STATUS_VALUES = ("confirmed", "candidate", "false_positive", "control")

    _STATUS_RE = re.compile(
        r"^(confirmed|false_positive|control|candidate(_snr\d+(\.\d+)?)?)$"
    )

    _table_spec: ClassVar = {
        "absorber": {
            "columns": [
                ["freq", np.float64],
                ["amplitude", np.float64],
                ["status", "<U24"],
            ],
            "axis": "object_id",
        },
    }

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # Record the allowed status values in the container attributes
        self.attrs["status_values"] = list(self.STATUS_VALUES)

    @staticmethod
    def _snr_of(status) -> float:
        """S/N recorded in a status ('candidate_snr5' -> 5.0; NaN if none)."""
        s = str(status)
        if "_snr" in s:
            try:
                return float(s.split("_snr", 1)[1])
            except ValueError:
                return np.nan
        return np.nan

    @classmethod
    def status_match(cls, status, include) -> np.ndarray:
        """Boolean mask of 'status' entries matching the 'include' list.

        Rules per entry of 'include':
        - "confirmed" / "false_positive" : exact match.
        - "candidate" : matches all candidates.
        - "candidate_snrX" : matches candidates with S/N >= X
          (candidates without a recorded S/N are not matched).

        Parameters
        ----------
        status : str or array_like of str
            Status values to test.
        include : str or list of str
            Status values to match against.

        Returns
        -------
        mask : np.ndarray of bool
            True for entries that match.
        """
        if isinstance(include, str):
            include = [include]

        status = np.atleast_1d(np.asarray(status, dtype=str))
        is_candidate = np.char.startswith(status, "candidate")
        snr = np.array([cls._snr_of(s) for s in status])

        mask = np.zeros(status.size, dtype=bool)
        for inc in include:
            inc = str(inc)
            if inc == "candidate":
                mask |= is_candidate
            elif inc.startswith("candidate_snr"):
                threshold = float(inc.split("_snr", 1)[1])
                mask |= is_candidate & (snr >= threshold)
            else:
                mask |= status == inc

        return mask

    def validate(self):
        """Check that all entries hold valid values."""
        names = self.index_map["object_id"]

        # Required columns:
        ra = self["position"]["ra"][:]
        dec = self["position"]["dec"][:]
        freq = self["absorber"]["freq"][:]
        for field, values, ok in [
            ("ra", ra, (ra >= 0.0) & (ra <= 360.0)),
            ("dec", dec, (dec >= -90.0) & (dec <= 90.0)),
            ("freq", freq, (freq > 0.0) & np.isfinite(freq)),
        ]:
            bad = ~ok
            if bad.any():
                raise ValueError(
                    f"Required column '{field}' has missing or out-of-range "
                    f"values for entries {names[bad].tolist()}: "
                    f"{values[bad].tolist()}. "
                    "Expected 0 <= ra < 360, -90 <= dec <= 90, freq > 0 (MHz)."
                )

        # Status: "confirmed", "false_positive", "candidate" or "candidate_snrX"
        status = self["absorber"]["status"][:]
        bad = np.array(
            [self._STATUS_RE.match(str(s)) is None for s in status], dtype=bool
        )
        if bad.any():
            raise ValueError(
                f"Invalid status values {np.unique(status[bad]).tolist()} for "
                f"entries {names[bad].tolist()}. "
                f"Allowed: {self.STATUS_VALUES}, where candidates may record "
                "their detection S/N as 'candidate_snrX' (e.g. 'candidate_snr5')."
            )

        # Amplitude (optional):
        amplitude = self["absorber"]["amplitude"][:]
        bad = np.isinf(amplitude)
        if bad.any():
            raise ValueError(
                f"Column 'amplitude' has non-finite (infinite) values for "
                f"entries: {names[bad].tolist()}. Amplitude must be a finite "
                "float, or NaN if unknown."
            )

    @property
    def id(self) -> np.ndarray:
        """The names/IDs of the absorbers."""
        return self.index_map["object_id"]

    @property
    def ra(self) -> np.ndarray:
        """Right ascension of each absorber."""
        return self["position"]["ra"]

    @property
    def dec(self) -> np.ndarray:
        """Declination of each absorber."""
        return self["position"]["dec"]

    @property
    def freq(self) -> np.ndarray:
        """Observed frequency (MHz) of the absorption feature."""
        return self["absorber"]["freq"]

    @property
    def amplitude(self) -> np.ndarray:
        """Estimated amplitude of the absorption feature (NaN if unknown)."""
        return self["absorber"]["amplitude"]

    @property
    def status(self) -> np.ndarray:
        """Status of each absorber (confirmed/candidate[_snrX]/false_positive)."""
        return self["absorber"]["status"]

    @property
    def snr(self) -> np.ndarray:
        """Detection S/N of each entry, parsed from the status (NaN if none)."""
        return np.array([self._snr_of(s) for s in self["absorber"]["status"][:]])
