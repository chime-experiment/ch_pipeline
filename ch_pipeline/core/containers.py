"""Parallel data containers.

Containers for holding various types of CHIME specific analysis data in a
distributed fashion. The module `draco.core.containers` contains general data
containers which are imported into this module.

Containers
==========
- :py:class:`FrequencyMap`
- :py:class:`RFIMask`
- :py:class:`CorrInputMask`
- :py:class:`CorrInputTest`
- :py:class:`CorrInputMonitor`
- :py:class:`SiderealDayFlag`
- :py:class:`TransitFitParams`
- :py:class:`PointSourceTransit`
- :py:class:`SourceModel`
- :py:class:`SunTransit`
- :py:class:`RingMap`
- :py:class:`Photometry`
- :py:class:`MFilteredHybridVisStream`
- :py:class:`TransientMatchedFilter`
- :py:class:`TransientMatchedFilterTrials`
- :py:class:`TransientCandidateMap`
"""

# Ignore missing docstrings in container @property methods.
# These are generally just convenience methods for datasets
# ruff: noqa: D102

import posixpath
from typing import ClassVar

import numpy as np
from caput import memdata
from caput.containers import Container, ContainerPrototype, TableSpec
from caput.util import typeutils
from ch_util import andata
from draco.core.containers import (
    COMPRESSION,
    COMPRESSION_OPTS,
    DataWeightContainer,
    FormedBeam,
    FreqContainer,
    HybridVisStream,
    SiderealContainer,
    SourceCatalog,
    SpectroscopicCatalog,
    StaticGainData,
    TimeStream,
    TODContainer,
)


class FrequencyMapSingle(FreqContainer):
    """Map between frequency bin and FPGA [crate, slot, link].

    This container holds a map between the frequency bins and stream ids.
    A stream id is encoded based on the shuffle, crate, slot, and link values
    as follows:

    stream_id = shuffle*2**12 + crate*2**8 + slot*2**4 + link

    The value of shuffle is always 3.
    """

    _axes = ("level",)

    _dataset_spec: ClassVar = {
        "stream": {
            "axes": ["freq", "level"],
            "dtype": np.int32,
            "initialise": True,
            "distributed": False,
        },
        "stream_id": {
            "axes": ["freq"],
            "dtype": np.int64,
            "initialise": True,
            "distributed": False,
        },
    }

    def __init__(self, *args, **kwargs):
        if "level" not in kwargs:
            kwargs["level"] = np.array(["shuffle", "crate", "slot", "link"], dtype=str)

        super().__init__(*args, **kwargs)

    @property
    def shuffle(self):
        return self.stream[:, 0]

    @property
    def crate(self):
        return self.stream[:, 1]

    @property
    def slot(self):
        return self.stream[:, 2]

    @property
    def link(self):
        return self.stream[:, 3]

    @property
    def stream(self):
        return self.datasets["stream"]

    @property
    def stream_id(self):
        return self.datasets["stream_id"]

    @property
    def level(self):
        return self.index_map["level"]


class FrequencyMap(FreqContainer, TODContainer):
    """Map between frequency bin and FPGA stream (GPU node) as a function of time."""

    _axes = ("level",)

    _dataset_spec: ClassVar = {
        "stream": {
            "axes": ["time", "freq", "level"],
            "dtype": int,
            "initialise": True,
            "distributed": False,
        },
        "node": {
            "axes": ["time", "freq"],
            "dtype": "<U5",
            "initialise": True,
            "distributed": False,
        },
    }

    @property
    def level(self):
        return self.index_map["level"]

    @property
    def stream(self):
        return self.datasets["stream"]

    @property
    def node(self):
        return self.datasets["node"]


class RFIMask(ContainerPrototype):
    """Container for holding a mask that indicates data that is free of RFI events."""

    _axes = ("freq", "input", "time")

    _dataset_spec: ClassVar = {
        "mask": {
            "axes": ["freq", "input", "time"],
            "dtype": bool,
            "initialise": True,
            "distributed": True,
            "distributed_axis": "freq",
        },
        "auto": {
            "axes": ["freq", "input", "time"],
            "dtype": np.float32,
            "initialise": False,
            "distributed": True,
            "distributed_axis": "freq",
        },
        "ndev": {
            "axes": ["freq", "input", "time"],
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
    def auto(self):
        return self.datasets["auto"]

    @property
    def ndev(self):
        return self.datasets["ndev"]


class CorrInputMask(ContainerPrototype):
    """Container for holding mask indicating good correlator inputs."""

    _axes = ("input",)

    _dataset_spec: ClassVar = {
        "input_mask": {
            "axes": ["input"],
            "dtype": bool,
            "initialise": True,
            "distributed": False,
        }
    }

    @property
    def input_mask(self):
        return self.datasets["input_mask"]

    @property
    def input(self):
        return self.index_map["input"]


class CorrInputTest(ContainerPrototype):
    """Container for holding results of tests for good correlator inputs."""

    _axes = ("freq", "input", "test")

    _dataset_spec: ClassVar = {
        "input_mask": {
            "axes": ["input"],
            "dtype": bool,
            "initialise": True,
            "distributed": False,
        },
        "passed_test": {
            "axes": ["freq", "input", "test"],
            "dtype": bool,
            "initialise": False,
            "distributed": False,
        },
    }

    def __init__(self, *args, **kwargs):
        if "test" not in kwargs:
            kwargs["test"] = np.array(
                ["is_chime", "not_known_bad", "digital_gain", "radiometer", "sky_fit"]
            )

        super().__init__(*args, **kwargs)

    @property
    def input_mask(self):
        return self.datasets["input_mask"]

    @property
    def passed_test(self):
        return self.datasets["passed_test"]

    @property
    def freq(self):
        return self.index_map["freq"]

    @property
    def input(self):
        return self.index_map["input"]

    @property
    def test(self):
        return self.index_map["test"]


class CorrInputMonitor(ContainerPrototype):
    """Container for holding results of good correlator inputs monitor."""

    _axes = ("freq", "input", "coord")

    _dataset_spec: ClassVar = {
        "input_mask": {
            "axes": ["input"],
            "dtype": bool,
            "initialise": True,
            "distributed": False,
        },
        "input_powered": {
            "axes": ["input"],
            "dtype": bool,
            "initialise": True,
            "distributed": False,
        },
        "freq_mask": {
            "axes": ["freq"],
            "dtype": bool,
            "initialise": True,
            "distributed": False,
        },
        "freq_powered": {
            "axes": ["freq"],
            "dtype": bool,
            "initialise": True,
            "distributed": False,
        },
        "position": {
            "axes": ["input", "coord"],
            "dtype": float,
            "initialise": False,
            "distributed": False,
        },
        "expected_position": {
            "axes": ["input", "coord"],
            "dtype": float,
            "initialise": False,
            "distributed": False,
        },
    }

    def __init__(self, *args, **kwargs):
        if "coord" not in kwargs:
            kwargs["coord"] = np.array(["east_west", "north_south"])

        super().__init__(*args, **kwargs)

    @property
    def input_mask(self):
        return self.datasets["input_mask"]

    @property
    def input_powered(self):
        return self.datasets["input_powered"]

    @property
    def position(self):
        return self.datasets["position"]

    @property
    def expected_position(self):
        return self.datasets["expected_position"]

    @property
    def freq_mask(self):
        return self.datasets["freq_mask"]

    @property
    def freq_powered(self):
        return self.datasets["freq_powered"]

    @property
    def freq(self):
        return self.index_map["freq"]

    @property
    def input(self):
        return self.index_map["input"]

    @property
    def coord(self):
        return self.index_map["coord"]


class SiderealDayFlag(ContainerPrototype):
    """Container for holding flag that indicates good chime sidereal days."""

    _axes = ("csd",)

    _dataset_spec: ClassVar = {
        "csd_flag": {
            "axes": ["csd"],
            "dtype": bool,
            "initialise": True,
            "distributed": False,
        }
    }

    @property
    def csd_flag(self):
        return self.datasets["csd_flag"]

    @property
    def csd(self):
        return self.index_map["csd"]


class TransitFitParams(ContainerPrototype):
    """Parallel container for holding the results of fitting a model to a point source transit."""

    _axes = ("freq", "input", "param", "component")

    _dataset_spec: ClassVar = {
        "parameter": {
            "axes": ["freq", "input", "param"],
            "dtype": np.float32,
            "initialise": True,
            "distributed": True,
            "distributed_axis": "freq",
        },
        "parameter_cov": {
            "axes": ["freq", "input", "param", "param"],
            "dtype": np.float32,
            "initialise": True,
            "distributed": True,
            "distributed_axis": "freq",
        },
        "chisq": {
            "axes": ["freq", "input", "component"],
            "dtype": np.float32,
            "initialise": False,
            "distributed": True,
            "distributed_axis": "freq",
        },
        "ndof": {
            "axes": ["freq", "input", "component"],
            "dtype": int,
            "initialise": False,
            "distributed": True,
            "distributed_axis": "freq",
        },
    }

    @property
    def parameter(self):
        return self.datasets["parameter"]

    @property
    def parameter_cov(self):
        return self.datasets["parameter_cov"]

    @property
    def chisq(self):
        return self.datasets["chisq"]

    @property
    def ndof(self):
        return self.datasets["ndof"]

    @property
    def freq(self):
        return self.index_map["freq"]["centre"]

    @property
    def input(self):
        return self.index_map["input"]

    @property
    def param(self):
        return self.index_map["param"]

    @property
    def component(self):
        return self.index_map["component"]


class PointSourceTransit(StaticGainData):
    """Parallel container for holding sidereal calibration.

    Contains the response of each feed to the source inferred from the eigendecomposition,
    the largest eigenvalues, the parameters of a model fit to the response, and the gain
    obtained by evaluating that model at transit.
    """

    _axes = (
        "freq",
        "input",
        "ra",
        "pol_x",
        "pol_y",
        "param",
        "param_cov1",
        "param_cov2",
    )

    _dataset_spec: ClassVar = {
        "gain": {
            "axes": ["freq", "input"],
            "dtype": np.complex128,
            "initialise": True,
            "distributed": True,
            "distributed_axis": "freq",
        },
        "weight": {
            "axes": ["freq"],
            "dtype": np.float64,
            "initialise": False,
            "distributed": True,
            "distributed_axis": "freq",
        },
        "evalue_x": {
            "axes": ["freq", "pol_x", "ra"],
            "dtype": np.float64,
            "initialise": True,
            "distributed": True,
            "distributed_axis": "freq",
        },
        "evalue_y": {
            "axes": ["freq", "pol_y", "ra"],
            "dtype": np.float64,
            "initialise": True,
            "distributed": True,
            "distributed_axis": "freq",
        },
        "response": {
            "axes": ["freq", "input", "ra"],
            "dtype": np.complex128,
            "initialise": True,
            "distributed": True,
            "distributed_axis": "freq",
        },
        "response_error": {
            "axes": ["freq", "input", "ra"],
            "dtype": np.float64,
            "initialise": True,
            "distributed": True,
            "distributed_axis": "freq",
        },
        "flag": {
            "axes": ["freq", "input", "ra"],
            "dtype": bool,
            "initialise": True,
            "distributed": True,
            "distributed_axis": "freq",
        },
        "parameter": {
            "axes": ["freq", "input", "param"],
            "dtype": np.float64,
            "initialise": True,
            "distributed": True,
            "distributed_axis": "freq",
        },
        "parameter_cov": {
            "axes": ["freq", "input", "param_cov1", "param_cov2"],
            "dtype": np.float64,
            "initialise": True,
            "distributed": True,
            "distributed_axis": "freq",
        },
    }

    def __init__(self, *args, **kwargs):
        kwargs["param"] = np.array(
            ["peak_amplitude", "centroid", "fwhm", "phase_intercept", "phase_slope"]
        )
        kwargs["param_cov1"] = np.array(
            ["peak_amplitude", "centroid", "fwhm", "phase_intercept", "phase_slope"]
        )
        kwargs["param_cov2"] = np.array(
            ["peak_amplitude", "centroid", "fwhm", "phase_intercept", "phase_slope"]
        )

        super().__init__(*args, **kwargs)

    @property
    def gain(self):
        return self.datasets["gain"]

    @property
    def weight(self):
        return self.datasets["weight"]

    @property
    def evalue_x(self):
        return self.datasets["evalue_x"]

    @property
    def evalue_y(self):
        return self.datasets["evalue_y"]

    @property
    def response(self):
        return self.datasets["response"]

    @property
    def response_error(self):
        return self.datasets["response_error"]

    @property
    def flag(self):
        return self.datasets["flag"]

    @property
    def parameter(self):
        return self.datasets["parameter"]

    @property
    def parameter_cov(self):
        return self.datasets["parameter_cov"]

    @property
    def freq(self):
        return self.index_map["freq"]

    @property
    def input(self):
        return self.index_map["input"]

    @property
    def param(self):
        return self.index_map["param"]

    @property
    def param_cov1(self):
        return self.index_map["param_cov1"]

    @property
    def param_cov2(self):
        return self.index_map["param_cov2"]


class SourceModel(FreqContainer):
    """Container for holding model for visiblities.

    Model consists of the sum of the signal from
    multiple (possibly extended) sources.
    """

    _axes = ("pol", "time", "param", "source")

    _dataset_spec: ClassVar = {
        "amplitude": {
            "axes": ["freq", "pol", "time", "source"],
            "dtype": np.complex64,
            "initialise": False,
            "distributed": True,
            "distributed_axis": "freq",
        },
        "coeff": {
            "axes": ["freq", "pol", "param"],
            "dtype": np.complex64,
            "initialise": False,
            "distributed": True,
            "distributed_axis": "freq",
        },
    }

    @property
    def amplitude(self):
        return self.datasets["amplitude"]

    @property
    def coeff(self):
        return self.datasets["coeff"]

    @property
    def param(self):
        return self.index_map["param"]

    @property
    def source(self):
        return self.index_map["source"]

    @property
    def source_index(self):
        src = list(self.source)
        return np.array([src.index(par) for par in self.param["source"]])


class SunTransit(ContainerPrototype):
    """Parallel container for holding the results of a fit to a point source transit."""

    _axes = (
        "freq",
        "input",
        "time",
        "pol",
        "eigen",
        "good_input1",
        "good_input2",
        "udegree",
        "vdegree",
        "coord",
        "param",
    )

    _dataset_spec: ClassVar = {
        "coord": {
            "axes": ["time", "coord"],
            "dtype": np.float64,
            "initialise": True,
            "distributed": False,
        },
        "evalue1": {
            "axes": ["freq", "good_input1", "time"],
            "dtype": np.float64,
            "initialise": True,
            "distributed": True,
            "distributed_axis": "freq",
        },
        "evalue2": {
            "axes": ["freq", "good_input2", "time"],
            "dtype": np.float64,
            "initialise": False,
            "distributed": True,
            "distributed_axis": "freq",
        },
        "response": {
            "axes": ["freq", "input", "time", "eigen"],
            "dtype": np.complex128,
            "initialise": True,
            "distributed": True,
            "distributed_axis": "freq",
        },
        "response_error": {
            "axes": ["freq", "input", "time", "eigen"],
            "dtype": np.float64,
            "initialise": True,
            "distributed": True,
            "distributed_axis": "freq",
        },
        "coeff": {
            "axes": ["freq", "pol", "time", "udegree", "vdegree"],
            "dtype": np.complex128,
            "initialise": False,
            "distributed": True,
            "distributed_axis": "freq",
        },
        "is_sun": {
            "axes": ["freq", "pol", "time"],
            "dtype": np.float64,
            "initialise": True,
            "distributed": True,
            "distributed_axis": "freq",
        },
        "flag": {
            "axes": ["freq", "input", "time"],
            "dtype": bool,
            "initialise": False,
            "distributed": True,
            "distributed_axis": "freq",
        },
        "parameter": {
            "axes": ["freq", "input", "param"],
            "dtype": np.float64,
            "initialise": False,
            "distributed": True,
            "distributed_axis": "freq",
        },
        "parameter_cov": {
            "axes": ["freq", "input", "param", "param"],
            "dtype": np.float64,
            "initialise": False,
            "distributed": True,
            "distributed_axis": "freq",
        },
    }

    def __init__(self, *args, **kwargs):
        kwargs["param"] = np.array(
            [
                "peak_amplitude",
                "centroid",
                "fwhm",
                "phase_intercept",
                "phase_slope",
                "phase_quad",
                "phase_cube",
                "phase_quart",
                "phase_quint",
            ]
        )
        kwargs["coord"] = np.array(["ha", "dec", "alt", "az"])

        super().__init__(*args, **kwargs)

    @property
    def coord(self):
        return self.datasets["coord"]

    @property
    def evalue1(self):
        return self.datasets["evalue1"]

    @property
    def evalue2(self):
        return self.datasets["evalue2"]

    @property
    def evalue_x(self):
        return self.datasets["evalue1"]

    @property
    def evalue_y(self):
        return self.datasets["evalue2"]

    @property
    def response(self):
        return self.datasets["response"]

    @property
    def response_error(self):
        return self.datasets["response_error"]

    @property
    def coeff(self):
        return self.datasets["coeff"]

    @property
    def is_sun(self):
        return self.datasets["is_sun"]

    @property
    def flag(self):
        return self.datasets["flag"]

    @property
    def parameter(self):
        return self.datasets["parameter"]

    @property
    def parameter_cov(self):
        return self.datasets["parameter_cov"]

    @property
    def freq(self):
        return self.index_map["freq"]

    @property
    def input(self):
        return self.index_map["input"]

    @property
    def time(self):
        return self.index_map["time"]

    @property
    def param(self):
        return self.index_map["param"]

    @property
    def ha(self):
        ind = list(self.index_map["coord"]).index("ha")
        return self.datasets["coord"][:, ind]

    @property
    def dec(self):
        ind = list(self.index_map["coord"]).index("dec")
        return self.datasets["coord"][:, ind]

    @property
    def alt(self):
        ind = list(self.index_map["coord"]).index("alt")
        return self.datasets["coord"][:, ind]

    @property
    def az(self):
        ind = list(self.index_map["coord"]).index("az")
        return self.datasets["coord"][:, ind]


class FormedBeamTime(FormedBeam, TODContainer):
    """Container for formed beams with a time axis."""

    _dataset_spec: ClassVar = {
        "beam": {
            "axes": ["object_id", "pol", "freq", "time"],
            "dtype": np.complex128,
            "initialise": True,
            "distributed": True,
            "distributed_axis": "freq",
        },
        "weight": {
            "axes": ["object_id", "pol", "freq", "time"],
            "dtype": np.float64,
            "initialise": True,
            "distributed": True,
            "distributed_axis": "freq",
        },
    }


class RingMap(ContainerPrototype):
    """Container for holding multifrequency ring maps.

    The maps are packed in format `[freq, pol, ra, EW beam, el]` where
    the polarisations are Stokes I, Q, U and V.

    Parameters
    ----------
    nside : int
        The nside of the Healpix maps.
    polarisation : bool, optional
        If `True` all Stokes parameters are stored, if `False` only Stokes I is
        stored.
    """

    _axes = ("freq", "pol", "ra", "beam", "el")

    _dataset_spec: ClassVar = {
        "map": {
            "axes": ["beam", "pol", "freq", "ra", "el"],
            "dtype": np.float64,
            "initialise": True,
            "distributed": True,
            "distributed_axis": "freq",
        },
        "dirty_beam": {
            "axes": ["beam", "pol", "freq", "ra", "el"],
            "dtype": np.float64,
            "initialise": False,
            "distributed": True,
            "distributed_axis": "freq",
        },
        "rms": {
            "axes": ["pol", "freq", "ra"],
            "dtype": np.float64,
            "initialise": False,
            "distributed": True,
            "distributed_axis": "freq",
        },
    }

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @property
    def freq(self):
        return self.index_map["freq"]

    @property
    def pol(self):
        return self.index_map["pol"]

    @property
    def ra(self):
        return self.index_map["ra"]

    @property
    def el(self):
        return self.index_map["el"]

    @property
    def map(self):
        return self.datasets["map"]

    @property
    def rms(self):
        return self.datasets["rms"]

    @property
    def dirty_beam(self):
        return self.datasets["dirty_beam"]


class Photometry(ContainerPrototype):
    """Parallel container for holding photometry extracted from a map."""

    _axes = ("freq", "pol", "param", "source")

    _dataset_spec: ClassVar = {
        "parameter": {
            "axes": ["freq", "pol", "param", "source"],
            "dtype": np.float64,
            "initialise": True,
            "distributed": True,
            "distributed_axis": "freq",
        },
        "parameter_cov": {
            "axes": ["freq", "pol", "param", "param", "source"],
            "dtype": np.float64,
            "initialise": True,
            "distributed": True,
            "distributed_axis": "freq",
        },
        "rms": {
            "axes": ["freq", "pol", "source"],
            "dtype": np.float64,
            "initialise": True,
            "distributed": True,
            "distributed_axis": "freq",
        },
    }

    @property
    def parameter(self):
        return self.datasets["parameter"]

    @property
    def parameter_cov(self):
        return self.datasets["parameter_cov"]

    @property
    def rms(self):
        return self.datasets["rms"]

    @property
    def freq(self):
        return self.index_map["freq"]

    @property
    def param(self):
        return self.index_map["param"]

    @property
    def source(self):
        return self.index_map["source"]


class SpectralLineCatalog(SpectroscopicCatalog):
    """A container for spectral line sources."""

    _table_spec: ClassVar = {
        "frequency": {
            "columns": [["freq", np.float64], ["freq_error", np.float64]],
            "axis": "object_id",
        },
        "flux": {
            "columns": [["line_depth", np.float64], ["continuum", np.float64]],
            "axis": "object_id",
        },
        "significance": {
            "columns": [["signal_to_noise", np.float64], ["npixels", int]],
            "axis": "object_id",
        },
    }


class MFilteredHybridVisStream(HybridVisStream):
    """Hybrid beamformed visibilities after an m-mode filter.

    For a filter ``y -> (I - P) y`` along right ascension, this also holds the
    elements of the filter near the diagonal, ``P[t, t + lag]``, and the inverse
    variance of the visibilities before filtering.
    """

    _axes = ("lag",)

    _dataset_spec: ClassVar = {
        "ra_response": {
            "axes": ["pol", "freq", "ew", "ra", "lag"],
            "dtype": np.complex64,
            "initialise": False,
            "distributed": True,
            "distributed_axis": "freq",
            "chunks": (1, 32, 1, 2048, 64),
            "compression": COMPRESSION,
            "compression_opts": COMPRESSION_OPTS,
        },
        "noise_weight": {
            "axes": ["pol", "freq", "ew", "ra"],
            "dtype": np.float32,
            "initialise": False,
            "distributed": True,
            "distributed_axis": "freq",
        },
        "noise_scale": {
            "axes": ["pol", "freq", "ew", "el"],
            "dtype": np.float32,
            "initialise": False,
            "distributed": True,
            "distributed_axis": "freq",
        },
    }

    @property
    def ra_response(self):
        return self.datasets["ra_response"]

    @property
    def noise_weight(self):
        return self.datasets["noise_weight"]

    @property
    def noise_scale(self):
        return self.datasets["noise_scale"]


class HybridVisScintillationFit(SiderealContainer):
    """Fit of the scintillation of the foregrounds in m-mode filtered visibilities.

    At each polarisation, east-west baseline, elevation and time, the filtered
    visibilities were fit with ``sum_j a_j phi_j(freq) f(freq)``, where ``f`` is the
    leave-one-out estimate of the foreground and ``phi_j`` are spectral basis
    functions (the ``param`` axis; see the attributes).  ``amplitude`` holds the
    ``a_j`` and ``template_power`` the weighted power of the first template,
    ``sum w |phi_0 f|^2``.  ``prior_sigma`` (if present) is the width of the prior on
    the first amplitude at each sample, so that ``prior_sigma**2 * template_power``
    is the scintillation power of the foreground relative to the noise.
    """

    _axes = ("pol", "ew", "el", "param")

    _dataset_spec: ClassVar = {
        "amplitude": {
            "axes": ["pol", "ew", "el", "ra", "param"],
            "dtype": np.complex64,
            "initialise": True,
            "distributed": True,
            "distributed_axis": "el",
        },
        "template_power": {
            "axes": ["pol", "ew", "el", "ra"],
            "dtype": np.float32,
            "initialise": True,
            "distributed": True,
            "distributed_axis": "el",
        },
        "prior_sigma": {
            "axes": ["pol", "ew", "el", "ra"],
            "dtype": np.float32,
            "initialise": False,
            "distributed": True,
            "distributed_axis": "el",
        },
    }

    @property
    def amplitude(self):
        return self.datasets["amplitude"]

    @property
    def template_power(self):
        return self.datasets["template_power"]


class HybridVisNoiseScale(FreqContainer):
    """Ratio of the measured to the expected noise variance of hybrid visibilities.

    The ``scale`` dataset multiplies the noise variance implied by the weights, as a
    function of polarisation, frequency, east-west baseline and elevation.
    ``nsample`` is the number of samples it was estimated from.
    """

    _axes = ("pol", "ew", "el")

    _dataset_spec: ClassVar = {
        "scale": {
            "axes": ["pol", "freq", "ew", "el"],
            "dtype": np.float32,
            "initialise": True,
            "distributed": True,
            "distributed_axis": "freq",
        },
        "nsample": {
            "axes": ["pol", "freq", "ew", "el"],
            "dtype": np.int32,
            "initialise": True,
            "distributed": True,
            "distributed_axis": "freq",
        },
    }

    @property
    def scale(self):
        return self.datasets["scale"]

    @property
    def nsample(self):
        return self.datasets["nsample"]


class HybridVisInbandNoise(FreqContainer, SiderealContainer):
    """Local scale of the noise of hybrid visibilities, from the scatter in frequency.

    ``scale`` multiplies the noise variance implied by the weights and the static
    noise scale, for each polarisation, band of channels (the ``freq`` axis holds
    the centre and width of each band), east-west baseline, elevation and sample.
    ``dof`` is the number of degrees of freedom of the estimate in each band before
    smoothing (the same at every elevation).  ``common`` is the factor of the scale
    shared by all elevations at each sample, and ``flag`` marks the samples of a
    band where it was too large to use.

    The scale is stored without chunking or compression, like the group maps.
    """

    _axes = ("pol", "ew", "el")

    _dataset_spec: ClassVar = {
        "scale": {
            "axes": ["pol", "freq", "ew", "el", "ra"],
            "dtype": np.float32,
            "initialise": True,
            "distributed": True,
            "distributed_axis": "el",
        },
        "dof": {
            "axes": ["pol", "freq", "ew", "ra"],
            "dtype": np.float32,
            "initialise": True,
            "distributed": False,
        },
        "common": {
            "axes": ["pol", "freq", "ew", "ra"],
            "dtype": np.float32,
            "initialise": True,
            "distributed": False,
        },
        "flag": {
            "axes": ["freq", "ra"],
            "dtype": bool,
            "initialise": True,
            "distributed": False,
        },
    }

    @property
    def scale(self):
        return self.datasets["scale"]

    @property
    def dof(self):
        return self.datasets["dof"]


class HybridVisSkyNoise(FreqContainer):
    """Fit of the noise of m-mode filtered visibilities that scales with the sky.

    The model for the filtered visibility of channel ``c`` and polarisation ``p`` at
    a given elevation and time, with unfiltered visibility ``S_cp``, is

        r_cp = n_cp + kappa_coh e S_cp + kappa_inc e_cp S_cp

    where ``n_cp`` is the noise expected from the weights and noise scale (scaled by
    ``alpha``), ``e`` is a unit fluctuation common to every channel in a band (with
    correlation ``rho_pol`` between polarisations), and ``e_cp`` is independent
    between channels.  ``kappa2_tot = kappa_coh**2 + kappa_inc**2``.

    The ``freq`` axis holds the centre and width of each band.  Datasets ending in
    ``_block`` are fits within blocks of right ascension (starting at the RA given by
    the ``block`` axis), and those ending in ``_source`` are fits within windows
    around the transit of each source.  Both have the ``alpha`` coefficients fixed
    at the values for the whole day.  ``count``, ``excess`` and ``model`` give, in
    bins of the modelled sky term of the band sum (``qbin`` is the upper edge, in
    units of the noise of the band sum), the number of samples and the mean
    measured and modelled power of the band sum in the same units.

    Datasets ending in ``_cell`` are fits within cells of RA block and telescope-y
    band (the ``yband`` axis holds the lower edge of each band, and the attribute
    ``yband_edges`` all the edges), present if the fit was asked for them.

    ``rho_band`` is the correlation of ``e`` between two bands (the ``band`` axis
    holds the centres of the bands of the ``freq`` axis), which is one on the
    diagonal.

    Datasets ending in ``_weight`` are inverse variances; coefficients that could
    not be fitted (for example in bands without data) are zero with zero weight.
    ``alpha_single``, ``alpha_band`` and ``alpha_pol`` are valid where the weights
    of ``kappa2_tot``, ``kappa2_coh`` and ``rho_pol`` are non-zero.
    """

    _axes = ("ew", "block", "source", "qbin", "band", "yband")

    _dataset_spec: ClassVar = {
        **{
            name: {
                "axes": ["freq", "ew"],
                "dtype": np.float64,
                "initialise": True,
                "distributed": False,
            }
            for name in (
                "alpha_single",
                "kappa2_tot",
                "kappa2_tot_weight",
                "alpha_band",
                "kappa2_coh",
                "kappa2_coh_weight",
                "alpha_pol",
                "rho_pol",
                "rho_pol_weight",
                "nsample",
            )
        },
        **{
            name: {
                "axes": ["freq", "ew", "block"],
                "dtype": np.float64,
                "initialise": True,
                "distributed": False,
            }
            for name in (
                "kappa2_tot_block",
                "kappa2_tot_block_weight",
                "kappa2_coh_block",
                "kappa2_coh_block_weight",
            )
        },
        **{
            name: {
                "axes": ["freq", "ew", "source"],
                "dtype": np.float64,
                "initialise": True,
                "distributed": False,
            }
            for name in (
                "kappa2_tot_source",
                "kappa2_tot_source_weight",
                "kappa2_coh_source",
                "kappa2_coh_source_weight",
            )
        },
        **{
            name: {
                "axes": ["freq", "ew", "qbin"],
                "dtype": np.float64,
                "initialise": True,
                "distributed": False,
            }
            for name in ("count", "excess", "model")
        },
        **{
            name: {
                "axes": ["freq", "ew", "block", "yband"],
                "dtype": np.float64,
                "initialise": False,
                "distributed": False,
            }
            for name in (
                "kappa2_tot_cell",
                "kappa2_tot_cell_weight",
                "kappa2_coh_cell",
                "kappa2_coh_cell_weight",
            )
        },
        **{
            name: {
                "axes": ["freq", "band", "ew"],
                "dtype": np.float64,
                "initialise": False,
                "distributed": False,
            }
            for name in ("rho_band", "rho_band_weight")
        },
        "kappa2_self": {
            "axes": ["freq"],
            "dtype": np.float64,
            "initialise": True,
            "distributed": False,
        },
    }


class HybridVisSourceCutouts(FreqContainer):
    """Hybrid beamformed visibilities in windows around the transits of sources.

    Each source has a window of ``ra_offset`` samples centred on the sample nearest
    its transit, and ``el_offset`` elevations centred on the elevation nearest its
    transit.  The ``ra_index`` and ``el_index`` datasets give the indices of the
    window into the original stream, with -1 where the window extends past the end of
    a stream that does not cover a full day.  The weights there are zero.  Stored
    without compression: the windows are small, and parallel writes of compressed
    datasets with small chunks can stall.
    """

    _axes = ("source", "pol", "ew", "el_offset", "ra_offset")

    _dataset_spec: ClassVar = {
        "vis": {
            "axes": ["source", "pol", "freq", "ew", "el_offset", "ra_offset"],
            "dtype": np.complex64,
            "initialise": True,
            "distributed": True,
            "distributed_axis": "freq",
        },
        "vis_weight": {
            "axes": ["source", "pol", "freq", "ew", "ra_offset"],
            "dtype": np.float32,
            "initialise": True,
            "distributed": True,
            "distributed_axis": "freq",
        },
        "ra": {
            "axes": ["source", "ra_offset"],
            "dtype": np.float64,
            "initialise": True,
            "distributed": False,
        },
        "el": {
            "axes": ["source", "el_offset"],
            "dtype": np.float64,
            "initialise": True,
            "distributed": False,
        },
        "ra_index": {
            "axes": ["source", "ra_offset"],
            "dtype": np.int64,
            "initialise": True,
            "distributed": False,
        },
        "el_index": {
            "axes": ["source", "el_offset"],
            "dtype": np.int64,
            "initialise": True,
            "distributed": False,
        },
        "source_ra": {
            "axes": ["source"],
            "dtype": np.float64,
            "initialise": True,
            "distributed": False,
        },
        "source_dec": {
            "axes": ["source"],
            "dtype": np.float64,
            "initialise": True,
            "distributed": False,
        },
    }

    @property
    def vis(self):
        return self.datasets["vis"]

    @property
    def weight(self):
        return self.datasets["vis_weight"]


class TransientMatchedFilter(FreqContainer, SiderealContainer, DataWeightContainer):
    """Transient matched filter for each band.

    The ``freq`` axis holds the centre and width of each band.  The ``ra`` axis is
    the start of the transient, ``x`` is its telescope-x at the start, and ``ha``
    and ``dec`` give the hour angle and declination of each ``(el, x)``.
    """

    _axes = ("pol", "duration", "el", "x")

    _dataset_spec: ClassVar = {
        "amplitude": {
            "axes": ["pol", "duration", "freq", "el", "ra", "x"],
            "dtype": np.float32,
            "initialise": True,
            "distributed": True,
            "distributed_axis": "el",
            "chunks": (1, 1, 8, 32, 1024, 32),
            "compression": COMPRESSION,
            "compression_opts": COMPRESSION_OPTS,
        },
        "weight": {
            "axes": ["pol", "duration", "freq", "el", "ra", "x"],
            "dtype": np.float32,
            "initialise": True,
            "distributed": True,
            "distributed_axis": "el",
            "chunks": (1, 1, 8, 32, 1024, 32),
            "compression": COMPRESSION,
            "compression_opts": COMPRESSION_OPTS,
        },
        "amplitude_imag": {
            "axes": ["pol", "duration", "freq", "el", "ra", "x"],
            "dtype": np.float32,
            "initialise": False,
            "distributed": True,
            "distributed_axis": "el",
            "chunks": (1, 1, 8, 32, 1024, 32),
            "compression": COMPRESSION,
            "compression_opts": COMPRESSION_OPTS,
        },
        "ha": {
            "axes": ["el", "x"],
            "dtype": np.float64,
            "initialise": True,
            "distributed": False,
        },
        "dec": {
            "axes": ["el", "x"],
            "dtype": np.float64,
            "initialise": True,
            "distributed": False,
        },
    }

    _data_dset_name = "amplitude"
    _weight_dset_name = "weight"

    @property
    def amplitude(self):
        return self.datasets["amplitude"]

    @property
    def amplitude_imag(self):
        return self.datasets["amplitude_imag"]

    @property
    def pol(self):
        return self.index_map["pol"]

    @property
    def el(self):
        return self.index_map["el"]

    @property
    def x(self):
        return self.index_map["x"]


class TransientMatchedFilterTrials(SiderealContainer, DataWeightContainer):
    """Transient matched filter for each duration and spectral index.

    The amplitude is the flux density at the frequency in the ``reference_freq``
    attribute.  The ``ra`` axis is the start of the transient, ``x`` is its
    telescope-x at the start, and ``ha`` and ``dec`` give the hour angle and
    declination of each ``(el, x)``.
    """

    _axes = ("pol", "duration", "spectral_index", "el", "x")

    _dataset_spec: ClassVar = {
        "amplitude": {
            "axes": ["pol", "duration", "spectral_index", "el", "ra", "x"],
            "dtype": np.float32,
            "initialise": True,
            "distributed": True,
            "distributed_axis": "el",
            "chunks": (1, 1, 1, 32, 1024, 32),
            "compression": COMPRESSION,
            "compression_opts": COMPRESSION_OPTS,
        },
        "weight": {
            "axes": ["pol", "duration", "spectral_index", "el", "ra", "x"],
            "dtype": np.float32,
            "initialise": True,
            "distributed": True,
            "distributed_axis": "el",
            "chunks": (1, 1, 1, 32, 1024, 32),
            "compression": COMPRESSION,
            "compression_opts": COMPRESSION_OPTS,
        },
        "amplitude_imag": {
            "axes": ["pol", "duration", "spectral_index", "el", "ra", "x"],
            "dtype": np.float32,
            "initialise": False,
            "distributed": True,
            "distributed_axis": "el",
            "chunks": (1, 1, 1, 32, 1024, 32),
            "compression": COMPRESSION,
            "compression_opts": COMPRESSION_OPTS,
        },
        "ha": {
            "axes": ["el", "x"],
            "dtype": np.float64,
            "initialise": True,
            "distributed": False,
        },
        "dec": {
            "axes": ["el", "x"],
            "dtype": np.float64,
            "initialise": True,
            "distributed": False,
        },
    }

    _data_dset_name = "amplitude"
    _weight_dset_name = "weight"

    @property
    def amplitude(self):
        return self.datasets["amplitude"]

    @property
    def amplitude_imag(self):
        return self.datasets["amplitude_imag"]

    @property
    def x(self):
        return self.index_map["x"]


class TransientBandGroupMap(FreqContainer, SiderealContainer):
    """Single-sample matched filter summed over groups of bands.

    For each group of bands, the S/N of the sum over the bands in the group, at every
    elevation and start time, either at the best telescope-x and spectral index or
    at a single telescope-x (the ``x_reduce`` attribute is "max" or that x).  The
    ``freq`` axis holds the centre and width of the bands, and ``band_used`` which
    of them are in each group.  ``x_index`` (only for "max") gives the best
    telescope-x, and ``spectral_index`` (only present if more than one was searched)
    the best spectral index.  ``ha`` and ``dec`` give the hour angle and declination of each
    ``(el, x)``.

    The maps are stored without chunking or compression.  Writing compressed
    datasets in parallel needs collective writes of every chunk, which is very
    slow for many small chunks shared between processes, and S/N values do not
    compress much anyway.
    """

    _axes = ("pol", "group", "el", "x")

    _dataset_spec: ClassVar = {
        "snr": {
            "axes": ["pol", "group", "el", "ra"],
            "dtype": np.float32,
            "initialise": True,
            "distributed": True,
            "distributed_axis": "el",
        },
        "x_index": {
            "axes": ["pol", "group", "el", "ra"],
            "dtype": np.int16,
            "initialise": False,
            "distributed": True,
            "distributed_axis": "el",
        },
        "spectral_index": {
            "axes": ["pol", "group", "el", "ra"],
            "dtype": np.float32,
            "initialise": False,
            "distributed": True,
            "distributed_axis": "el",
        },
        "band_used": {
            "axes": ["group", "freq"],
            "dtype": bool,
            "initialise": True,
            "distributed": False,
        },
        "ha": {
            "axes": ["el", "x"],
            "dtype": np.float64,
            "initialise": True,
            "distributed": False,
        },
        "dec": {
            "axes": ["el", "x"],
            "dtype": np.float64,
            "initialise": True,
            "distributed": False,
        },
    }

    @property
    def snr(self):
        return self.datasets["snr"]

    @property
    def group(self):
        return self.index_map["group"]


class TransientCandidateMap(SiderealContainer, DataWeightContainer):
    """Best transient trial at each elevation and start time."""

    _axes = ("pol", "el", "x")

    _dataset_spec: ClassVar = {
        "snr": {
            "axes": ["pol", "el", "ra"],
            "dtype": np.float32,
            "initialise": True,
            "distributed": True,
            "distributed_axis": "el",
        },
        "amplitude": {
            "axes": ["pol", "el", "ra"],
            "dtype": np.float32,
            "initialise": True,
            "distributed": True,
            "distributed_axis": "el",
        },
        "weight": {
            "axes": ["pol", "el", "ra"],
            "dtype": np.float32,
            "initialise": True,
            "distributed": True,
            "distributed_axis": "el",
        },
        "duration": {
            "axes": ["pol", "el", "ra"],
            "dtype": np.int32,
            "initialise": True,
            "distributed": True,
            "distributed_axis": "el",
        },
        "spectral_index": {
            "axes": ["pol", "el", "ra"],
            "dtype": np.float32,
            "initialise": True,
            "distributed": True,
            "distributed_axis": "el",
        },
        "x_index": {
            "axes": ["pol", "el", "ra"],
            "dtype": np.int16,
            "initialise": True,
            "distributed": True,
            "distributed_axis": "el",
        },
        "amplitude_imag": {
            "axes": ["pol", "el", "ra"],
            "dtype": np.float32,
            "initialise": False,
            "distributed": True,
            "distributed_axis": "el",
        },
        "ha": {
            "axes": ["el", "x"],
            "dtype": np.float64,
            "initialise": True,
            "distributed": False,
        },
        "dec": {
            "axes": ["el", "x"],
            "dtype": np.float64,
            "initialise": True,
            "distributed": False,
        },
    }

    _data_dset_name = "snr"
    _weight_dset_name = "weight"

    @property
    def snr(self):
        return self.datasets["snr"]

    @property
    def amplitude(self):
        return self.datasets["amplitude"]

    @property
    def x(self):
        return self.index_map["x"]


class FluxModelCatalog(SourceCatalog):
    """A catalogue of sources with a model of their spectra.

    ``position`` holds the ``ra`` and ``dec`` (degrees, ICRS unless the
    ``coords`` attribute of ``position`` says otherwise).  The ``flux_model``
    table holds the parameters of fluxcat's ``CurvedPowerLaw`` for each source,
    ``S(nu) = p0 exp(sum_k p_k ln(nu / freq_pivot)^k)`` in Jy with ``freq_pivot``
    in MHz (unused parameters zero), which
    :class:`ch_pipeline.synthesis.inject.FluxCatalog` evaluates in place of a
    fluxcat collection.
    """

    _table_spec: ClassVar = {
        "flux_model": {
            "columns": [
                ["freq_pivot", np.float64],
                ["p0", np.float64],
                ["p1", np.float64],
                ["p2", np.float64],
                ["p3", np.float64],
                ["p4", np.float64],
                ["p5", np.float64],
            ],
            "axis": "object_id",
        }
    }


class LPTCatalog(SourceCatalog):
    """A catalogue of simulated long-period transients (intrinsic properties only).

    ``position`` holds the ICRS ``ra`` and ``dec`` (degrees).  The ``lpt`` table
    holds for each source (see
    :class:`ch_pipeline.synthesis.inject_lpt.GenerateLPTCatalog`): the peak ``flux``
    (Jy at the ``reference_freq`` attribute) with its pulse-to-pulse scatter
    (``flux_scatter``, sigma of ln flux) and ``null_fraction``; ``period`` (s) and
    ``epoch`` (unix time of a pulse at the solar-system barycentre), or period 0
    for one pulse a day at ``t_offset_beam`` beam widths from transit; ``duration``
    (s); ``spectral_index`` and ``spectral_curvature``; ``frac_lin``, ``frac_circ``
    and ``pol_angle`` (degrees, from the X feed at transit); ``rm`` (rad / m^2);
    the scintillation (``scint_index``, ``scint_bw`` in MHz); ``distance`` (kpc, or
    NaN) and the random ``seed`` of the source's pulses.  Where each pulse appears on
    a given day is in the :class:`LPTInjectionLog` of that day.
    """

    _table_spec: ClassVar = {
        "lpt": {
            "columns": [
                ["flux", np.float64],
                ["flux_scatter", np.float64],
                ["null_fraction", np.float64],
                ["period", np.float64],
                ["epoch", np.float64],
                ["t_offset_beam", np.float64],
                ["duration", np.float64],
                ["spectral_index", np.float64],
                ["spectral_curvature", np.float64],
                ["frac_lin", np.float64],
                ["frac_circ", np.float64],
                ["pol_angle", np.float64],
                ["rm", np.float64],
                ["scint_index", np.float64],
                ["scint_bw", np.float64],
                ["distance", np.float64],
                ["seed", np.int64],
            ],
            "axis": "object_id",
        }
    }


_LPT_PULSE_COLUMNS = [
    ["source", np.int32],
    ["pulse", np.int64],
    ["injected", bool],
    ["nulled", bool],
    ["in_day", bool],
    ["unix_center", np.float64],
    ["t_offset", np.float64],
    ["ha_center", np.float64],
    ["flux", np.float64],
    ["duration", np.float64],
    ["spectral_index", np.float64],
    ["frac_lin", np.float64],
    ["frac_circ", np.float64],
    ["pol_angle", np.float64],
    ["rm", np.float64],
    ["ra", np.float64],
    ["dec", np.float64],
    ["ra_cirs", np.float64],
    ["dec_cirs", np.float64],
    ["y", np.float64],
    ["x_start", np.float64],
    ["ra_start", np.float64],
    ["beam_mean", np.float64],
    ["scint_seed", np.int64],
]


class LPTInjectionLog(TableSpec):
    """The pulses of simulated LPTs that fell in the main lobe on one day.

    One row per pulse (``pulse`` axis): the ``source`` (row of the
    :class:`LPTCatalog`) and ``pulse`` number (the LSD for one pulse a day), whether
    it was ``injected`` (not ``nulled``, on samples covered by the beam model),
    whether its centre is ``in_day`` (meridian RA 0-360 of the LSD, where the
    candidate maps are), the time of its centre (topocentric unix), its offset from
    the nearest upper transit (s) and hour angle (degrees), its peak ``flux`` (after
    the pulse-to-pulse scatter), duration, spectral index, polarisation
    fractions, angle (degrees, from the X feed at transit and lambda = 0) and RM, the ICRS and CIRS position, telescope-y at the centre, telescope-x
    and meridian RA (degrees) at the start of the pulse, the XX primary beam
    averaged over the pulse and the band (``beam_mean``), and the seed of its
    scintillation.
    """

    _table_spec: ClassVar = {"pulse": {"columns": _LPT_PULSE_COLUMNS, "axis": "pulse"}}

    @property
    def pulse(self):
        return self.datasets["pulse"]


class LPTInjectionResults(TableSpec):
    """The injected pulses of a day and the candidates that found them.

    The columns of the :class:`LPTInjectionLog`, then whether a candidate was found
    within the match window (``found``, ``candidate`` its row in the candidate
    table or -1, ``n_match`` the number in the window), and that candidate's
    properties and sifting result.  ``cand_imag`` is the candidate's
    ``amplitude_imag`` (for the Stokes search: U at RM 0 for V, the best RM for L).
    The ``pol`` attribute names the polarisation of the candidates (I, V, L, co).
    """

    _table_spec: ClassVar = {
        "injection": {
            "columns": _LPT_PULSE_COLUMNS
            + [
                ["found", bool],
                ["candidate", np.int32],
                ["n_match", np.int16],
                ["snr", np.float32],
                ["amplitude", np.float32],
                ["cand_duration", np.float32],
                ["cand_spectral_index", np.float32],
                ["cand_y", np.float64],
                ["cand_x", np.float64],
                ["cand_ra", np.float64],
                ["keep", bool],
                ["flags", np.uint32],
                ["ns_scale", np.float32],
                ["ns_dchi2", np.float32],
                ["cand_imag", np.float32],
            ],
            "axis": "pulse",
        }
    }

    @property
    def injection(self):
        return self.datasets["injection"]


class TransientCandidateCutouts(FreqContainer):
    """The per-band matched filter in windows around candidates.

    For each candidate (``candidate_index`` is its row in the candidate table), the
    amplitude (Jy) and weight of each band and polarisation (``pol``, e.g. I, Q, U
    and V of the Stokes search) in a window of ``el_offset`` elevations and
    ``ra_offset`` start times centred on the candidate, at its telescope-x
    (``x_index``), for ``duration`` samples (1 for the cutouts of
    :class:`ExtractCandidateCutouts`; each candidate's own for
    :class:`CandidateBandSpectra`).  ``el_index`` and ``ra_index`` give the
    indices into the maps, -1 outside them, where the weight is zero; ``el`` and
    ``ra`` continue the grid there.  Distributed over candidates.
    """

    _axes = ("candidate", "pol", "el_offset", "ra_offset")

    _dataset_spec: ClassVar = {
        "amplitude": {
            "axes": ["candidate", "pol", "freq", "el_offset", "ra_offset"],
            "dtype": np.float32,
            "initialise": True,
            "distributed": True,
            "distributed_axis": "candidate",
        },
        "weight": {
            "axes": ["candidate", "pol", "freq", "el_offset", "ra_offset"],
            "dtype": np.float32,
            "initialise": True,
            "distributed": True,
            "distributed_axis": "candidate",
        },
        "candidate_index": {
            "axes": ["candidate"],
            "dtype": np.int64,
            "initialise": True,
            "distributed": False,
        },
        "x_index": {
            "axes": ["candidate"],
            "dtype": np.int64,
            "initialise": True,
            "distributed": False,
        },
        "duration": {
            "axes": ["candidate"],
            "dtype": np.int64,
            "initialise": True,
            "distributed": False,
        },
        "el_index": {
            "axes": ["candidate", "el_offset"],
            "dtype": np.int64,
            "initialise": True,
            "distributed": False,
        },
        "ra_index": {
            "axes": ["candidate", "ra_offset"],
            "dtype": np.int64,
            "initialise": True,
            "distributed": False,
        },
        "el": {
            "axes": ["candidate", "el_offset"],
            "dtype": np.float64,
            "initialise": True,
            "distributed": False,
        },
        "ra": {
            "axes": ["candidate", "ra_offset"],
            "dtype": np.float64,
            "initialise": True,
            "distributed": False,
        },
    }

    @property
    def amplitude(self):
        return self.datasets["amplitude"]

    @property
    def weight(self):
        return self.datasets["weight"]



class TransientCandidateTrialCutouts(ContainerPrototype):
    """The calibrated trials of the matched filter in windows around candidates.

    For each candidate (``candidate_index`` is its row in the candidate table), the
    amplitude and weight (and the imaginary part, e.g. the best RM of L) of each
    polarisation of the trials (I, V and L of the Stokes search) and every duration,
    at the candidate's telescope-x (``x_index``) and spectral index
    (``spectral_index``, per polarisation: L is searched at fewer), in a window of
    ``el_offset`` elevations and ``ra_offset`` start times centred on the candidate.
    Unlike the best-trial maps, each duration is a single trial, with no tail to
    earlier starts.  ``el_index`` and ``ra_index`` give the indices into the maps, -1
    outside them, where the weight is zero; ``el`` and ``ra`` continue the grid
    there.  Distributed over candidates.
    """

    _axes = ("candidate", "pol", "duration", "el_offset", "ra_offset")

    _dataset_spec: ClassVar = {
        "amplitude": {
            "axes": ["candidate", "pol", "duration", "el_offset", "ra_offset"],
            "dtype": np.float32,
            "initialise": True,
            "distributed": True,
            "distributed_axis": "candidate",
        },
        "weight": {
            "axes": ["candidate", "pol", "duration", "el_offset", "ra_offset"],
            "dtype": np.float32,
            "initialise": True,
            "distributed": True,
            "distributed_axis": "candidate",
        },
        "amplitude_imag": {
            "axes": ["candidate", "pol", "duration", "el_offset", "ra_offset"],
            "dtype": np.float32,
            "initialise": False,
            "distributed": True,
            "distributed_axis": "candidate",
        },
        "candidate_index": {
            "axes": ["candidate"],
            "dtype": np.int64,
            "initialise": True,
            "distributed": False,
        },
        "x_index": {
            "axes": ["candidate"],
            "dtype": np.int64,
            "initialise": True,
            "distributed": False,
        },
        "spectral_index": {
            "axes": ["candidate", "pol"],
            "dtype": np.float64,
            "initialise": True,
            "distributed": False,
        },
        "el_index": {
            "axes": ["candidate", "el_offset"],
            "dtype": np.int64,
            "initialise": True,
            "distributed": False,
        },
        "ra_index": {
            "axes": ["candidate", "ra_offset"],
            "dtype": np.int64,
            "initialise": True,
            "distributed": False,
        },
        "el": {
            "axes": ["candidate", "el_offset"],
            "dtype": np.float64,
            "initialise": True,
            "distributed": False,
        },
        "ra": {
            "axes": ["candidate", "ra_offset"],
            "dtype": np.float64,
            "initialise": True,
            "distributed": False,
        },
    }

    @property
    def amplitude(self):
        return self.datasets["amplitude"]

    @property
    def weight(self):
        return self.datasets["weight"]

class TransientCandidateTable(TableSpec):
    """Candidates found in the maps of the transient search, one row per candidate.

    The ``candidate`` table holds the position of each candidate and the properties
    measured by :class:`ch_pipeline.analysis.transients.GroupTransientCandidates`
    (see there for their definitions).  ``group_snr`` is the S/N of each group of
    bands at the candidate (at the best telescope-x for each group).  If the
    per-band matched filter was available,
    :class:`ch_pipeline.analysis.transients.AddCandidateSpectra` adds ``spectrum``
    and ``spectrum_weight``, the amplitude (Jy) and its inverse variance in each band
    at the candidate's telescope-x, and
    :class:`ch_pipeline.analysis.transients.AddNSProfile` the ``ns_*`` columns, a fit
    of the north-south synthesized beam to the profile across telescope-y.  The
    ``sift`` table holds the result of
    :class:`ch_pipeline.analysis.transients.SiftTransientCandidates`: whether each
    candidate was kept, and a bit mask of the tests it failed (the bits are named in
    the ``sift_flags`` attribute).  The ``fidelity`` table holds the comparison with
    the negative search (:class:`ch_pipeline.analysis.transients.AddNegativeFidelity`;
    in a table of negative peaks from
    :class:`ch_pipeline.analysis.transients.TransientNegativePeaks`, only ``bowl``),
    the ``significance`` table the S/N calibrated against the local tail of the
    negative search (:class:`ch_pipeline.analysis.transients.CalibrateSignificance`),
    and the ``window`` table statistics of the single samples within each candidate's
    duration (:class:`ch_pipeline.analysis.transients.AddWindowStats`).
    """

    _axes = ("group", "freq")

    _table_spec: ClassVar = {
        "candidate": {
            "columns": [
                # Position on the map and the best trial
                ["el_index", np.int32],
                ["ra_index", np.int32],
                ["x_index", np.int16],
                ["y", np.float64],
                ["x", np.float64],
                ["ra", np.float64],
                ["unix", np.float64],
                ["snr", np.float32],
                ["amplitude", np.float32],
                ["duration", np.float32],
                ["spectral_index", np.float32],
                # The imaginary part of the best trial, if saved (for the Stokes
                # search: U at RM 0 for V, the best RM for L)
                ["amplitude_imag", np.float32],
                # Position on the sky (ICRS) and hour angle
                ["ra_icrs", np.float64],
                ["dec_icrs", np.float64],
                ["ha", np.float32],
                # Shape at half the peak S/N, and of the island above grow_snr
                ["npix", np.int32],
                ["fwhm_y", np.float32],
                ["fwhm_t", np.float32],
                ["x_spread", np.float32],
                ["island_npix", np.int32],
                ["island_dy", np.float32],
                ["island_dt", np.float32],
                ["island_slope", np.float32],
                ["island_corr", np.float32],
                ["ext_npix", np.int32],
                ["ext_dy", np.float32],
                ["ext_dt", np.float32],
                # Other peaks along the row
                ["row_npeak", np.int16],
                ["row_max_ratio", np.float32],
                ["row_spacing", np.float32],
                ["row_spacing_cv", np.float32],
                # Other rows at the same time
                ["col_nrows", np.int16],
                ["col_nrows_bg", np.int16],
                ["col_ncand", np.int16],
                # Spectrum across the groups of bands
                ["group_nsig", np.int16],
                ["group_frac_max", np.float32],
                ["group_best", np.int16],
                # Spectrum across the bands (AddCandidateSpectra)
                ["band_nsig", np.int16],
                ["band_frac_max", np.float32],
                ["band_best_freq", np.float32],
                ["band_alpha", np.float32],
                ["band_chi2", np.float32],
                ["band_fit_frac", np.float32],
                # North-south profile across telescope-y (AddNSProfile)
                ["ns_nband", np.int16],
                ["ns_dof", np.int32],
                ["ns_snr", np.float32],
                ["ns_dy", np.float32],
                ["ns_chi2", np.float32],
                ["ns_scale", np.float32],
                ["ns_scale_lo", np.float32],
                ["ns_scale_hi", np.float32],
                ["ns_dchi2", np.float32],
                # Noise
                ["weight_ratio", np.float32],
                # The Sun and bright sources
                ["sun_ha", np.float32],
                ["sun_alt", np.float32],
                ["track_name", "S12"],
                ["track_dy", np.float32],
                ["track_ha", np.float32],
                ["lobe_name", "S12"],
                ["lobe_ha", np.float32],
                ["lobe_freq", np.float32],
                # Catalogue
                ["cat_name", "S32"],
                ["cat_flux", np.float32],
                ["cat_dra", np.float32],
                ["cat_ddec", np.float32],
                ["cat_nmatch", np.int16],
                ["cat_near_name", "S32"],
                ["cat_near_flux", np.float32],
                ["cat_near_sep", np.float32],
                # Known pulsars
                ["psr_name", "S24"],
                ["psr_sep", np.float32],
                ["psr_flux", np.float32],
            ],
            "axis": "candidate",
        },
        "sift": {
            "columns": [
                ["keep", bool],
                ["flags", np.uint32],
                # The counts used by the row_repeat and crowded tests
                ["row_npeak", np.int16],
                ["crowd_ratio", np.float32],
                ["crowd_n", np.int16],
            ],
            "axis": "candidate",
            "initialise": False,
        },
        "fidelity": {
            "columns": [
                # Negative and positive peaks around the candidate at least as
                # bright, the fidelity 1 - neg_n / pos_n, and the brightest
                # negative peak there
                ["neg_n", np.int32],
                ["pos_n", np.int32],
                ["fidelity", np.float32],
                ["neg_max", np.float32],
                # Negative peaks only: on the filter's negative lobe of a brighter
                # positive peak (not used as noise)
                ["bowl", bool],
            ],
            "axis": "candidate",
            "initialise": False,
        },
        "significance": {
            "columns": [
                # Sky-to-thermal variance ratio at the candidate, the local level of
                # the negative tail over the model, the expected number of noise
                # peaks a day at least as bright, and the S/N with that rate in the
                # reference class (CalibrateSignificance)
                ["sky_q", np.float32],
                ["local_ratio", np.float32],
                ["far", np.float32],
                ["snr_cal", np.float32],
            ],
            "axis": "candidate",
            "initialise": False,
        },
        "window": {
            "columns": [
                # The single samples along the candidate's track over its duration
                # (AddWindowStats): their number, the fraction above zero, the
                # self-normalised t = mean / (std / sqrt(n)), and the share of the
                # sum from the largest two
                ["win_n", np.int16],
                ["win_fpos", np.float32],
                ["win_t", np.float32],
                ["win_top2", np.float32],
            ],
            "axis": "candidate",
            "initialise": False,
        },
    }

    _dataset_spec: ClassVar = {
        "group_snr": {
            "axes": ["candidate", "group"],
            "dtype": np.float32,
            "initialise": True,
            "distributed": False,
        },
        "spectrum": {
            "axes": ["candidate", "freq"],
            "dtype": np.float32,
            "initialise": False,
            "distributed": False,
        },
        "spectrum_weight": {
            "axes": ["candidate", "freq"],
            "dtype": np.float32,
            "initialise": False,
            "distributed": False,
        },
    }

    @property
    def candidate(self):
        return self.datasets["candidate"]

    @property
    def group(self):
        return self.index_map["group"]



class PolarisationCalibration(TableSpec):
    """The X-Y phase and polarisation of calibrator pulsars on one day.

    From :class:`ch_pipeline.analysis.transients.FitPolarisationCalibrators`: one
    row of ``calibrator`` per pulsar and transit (upper or lower) with its fitted
    RM, linear fraction and angle, and the X-Y phase (delay and offset, both the
    total and the residual after the phase applied by the matched filter), the
    amplitude check ``|A|`` and the constant term ``B / A`` (``i v`` plus leakage),
    and the quality of the fit.  The per-band datasets hold the fractional Q and C
    (``U + iV`` after the applied phase) from the regression on I, with their
    noise; ``band_used`` marks the bands of the fit.
    """

    _axes = ("freq",)

    _table_spec: ClassVar = {
        "calibrator": {
            "columns": [
                ["name", "S16"],
                ["transit", "S8"],
                ["ok", bool],
                ["ra", np.float64],
                ["dec", np.float64],
                ["y", np.float32],
                ["x_mean", np.float32],
                ["nsamp", np.int32],
                ["nband", np.int16],
                ["frac_noise", np.float32],
                ["rm_cat", np.float32],
                ["rm", np.float32],
                ["rm_abs", np.float32],
                ["rm_dchi2", np.float32],
                ["rot_deg", np.float32],
                ["pfrac", np.float32],
                ["chi_deg", np.float32],
                ["delay_ns", np.float32],
                ["offset_deg", np.float32],
                ["delay_resid_ns", np.float32],
                ["offset_resid_deg", np.float32],
                ["abs_a", np.float32],
                ["b_re", np.float32],
                ["b_im", np.float32],
                ["phase_rms_deg", np.float32],
                ["chi2", np.float32],
                ["dof", np.int32],
            ],
            "axis": "calibrator",
        }
    }

    _dataset_spec: ClassVar = {
        "q": {
            "axes": ["calibrator", "freq"],
            "dtype": np.float32,
            "initialise": True,
            "distributed": False,
        },
        "q_err": {
            "axes": ["calibrator", "freq"],
            "dtype": np.float32,
            "initialise": True,
            "distributed": False,
        },
        "c": {
            "axes": ["calibrator", "freq"],
            "dtype": np.complex64,
            "initialise": True,
            "distributed": False,
        },
        "c_err": {
            "axes": ["calibrator", "freq"],
            "dtype": np.float32,
            "initialise": True,
            "distributed": False,
        },
        "band_used": {
            "axes": ["calibrator", "freq"],
            "dtype": bool,
            "initialise": True,
            "distributed": False,
        },
    }

    @property
    def calibrator(self):
        return self.datasets["calibrator"]

class RawContainer(TODContainer):
    """Base class for using raw CHIME data via the draco containers API.

    This modifies the set of allowed group/dataset names to allow access to the
    flags group.
    """

    _allowed_groups: ClassVar[str] = ["flags"]

    def group_name_allowed(self, name: str) -> bool:
        """Allow access to the flags group."""
        # Strip leading and trailing "/"
        name = name.strip("/")

        return name in self._allowed_groups

    def dataset_name_allowed(self, name: str) -> bool:
        """Allow access to datasets under both / and /flags."""
        parent_name, name = posixpath.split(name)
        return parent_name == "/" or self.group_name_allowed(parent_name)

    @classmethod
    def from_acq_h5(
        cls,
        acq_files: str | list[str],
        start: int | None = None,
        stop: int | None = None,
        **kwargs,
    ) -> "RawContainer":
        """Load from an HDF5 file on disk.

        This is a thin wrapper around `from_file` to support the
        `andata.BaseData.from_acq_h5` API. The main difference is supporting
        the `start` and `stop` parameters. All other parameters are passed
        straight into `from_file`.

        .. note:: This does not support loading in a distributed manner as the archive
            files don't have the correct hints set, but that should be added into this
            routine by using the information given in the `_dataset_spec` of derived
            classes.

        Parameters
        ----------
        acq_files
            Path or glob to files (or list of).
        start, stop
            Indices into the full set of files to select.
        **kwargs
            Additional keyword arguments to pass to `from_file`

        Returns
        -------
        cont
            A single container instance.
        """
        if start is None and stop is None:
            pass
        elif start is not None and stop is not None:
            kwargs["time_sel"] = slice(start, stop)
        else:
            raise ValueError(
                f"Got mixed types for start ({type(start)}) and stop ({type(stop)})"
            )

        return cls.from_file(acq_files, **kwargs)


class CHIMETimeStream(TimeStream, RawContainer):
    """A container for CHIME visibility data.

    This provides a close analog of the CorrData container but implemented via the draco
    containers API.
    """

    # Add in the extra datasets contained within the CHIME corrdata time streams
    _dataset_spec: ClassVar = {
        "flags/dataset_id": {
            "axes": ["freq", "time"],
            "dtype": "S32",
            "initialise": False,
            "distributed": True,
        },
        "flags/frac_lost": {
            "axes": ["freq", "time"],
            "dtype": np.float32,
            "initialise": False,
            "distributed": True,
        },
    }

    @classmethod
    def from_corrdata(cls, data: andata.CorrData) -> "CHIMETimeStream":
        """Turn a CorrData container into a CHIMETimeStream.

        This makes a shallow copy of the input `data`. The internals of `data` will be
        destroyed during this process and so the container should be discarded.

        Parameters
        ----------
        data
            A CorrData instance to clone. This instance should be discarded after this
            call.

        Returns
        -------
        newdata
            The correlator data as a CHIMETimeStream object.
        """
        # Copy over all the internal data store
        newdata = cls(data_group=data, distributed=data.distributed, comm=data.comm)

        # Manipulate location of weights as this is the one dataset that has a different
        # location in CHIMETimeStream compared to CorrData
        storage = newdata._storage_root._get_storage()

        if "/flags/vis_weight" in newdata:
            storage["vis_weight"] = storage["flags"].pop("vis_weight")
            storage["vis_weight"]._name = "/vis_weight"

        if "/flags/inputs" in newdata:
            storage["input_flags"] = storage["flags"].pop("inputs")
            storage["input_flags"]._name = "/input_flags"

        # Remove any datasets/flags which shouldn't be present in this container.
        # If any other dataset is needed, then a CorrData container should be used
        # or a new dataset_spec entry should be added to this container. At present,
        # this should only affect the "frac_rfi" flag.
        contains = set(newdata.datasets) | {
            "flags/" + name for name in newdata["flags"]
        }

        for name in contains:
            if name not in newdata.dataset_spec:
                del newdata[name]
                continue

        return newdata

    @property
    def frac_lost(self):
        """Get the input flags dataset."""
        return self["flags/frac_lost"]

    @property
    def dataset_id(self):
        """Get the dataset_id dataset in Unicode."""
        dsid = typeutils.ensure_unicode(self["flags/dataset_id"][:])
        dsid.flags.writeable = False

        return dsid

    @property
    def flags(self):
        """Get the flags group."""
        flags_group = dict(self["flags"])

        # Alias the groups back in for maximum compatibility with CHIME Andata
        # based code
        if "vis_weight" in self:
            flags_group["vis_weight"] = self["vis_weight"]

        if "input_flags" in self:
            flags_group["input_flags"] = self["input_flags"]

        return flags_group


def make_empty_corrdata(
    freq=None,
    input=None,
    time=None,
    axes_from=None,
    attrs_from=None,
    distributed=True,
    distributed_axis=0,
    comm=None,
):
    """Make an empty CorrData (i.e. timestream) container.

    Parameters
    ----------
    freq : np.ndarray, optional
        Frequency map to use.
    input : np.ndarray, optional
        Input map.
    time : np.ndarray, optional
        Time map.
    axes_from : BasicCont, optional
        Another container to copy any unspecified axes from.
    attrs_from : BasicCont, optional
        Another container to copy any unspecified attributes from.
    distributed : boolean, optional
        Whether to create the container in distributed mode.
    distributed_axis : int, optional
        Axis to distribute over.
    comm : MPI.Comm, optional
        MPI communicator to distribute over.

    Returns
    -------
    data : andata.CorrData
    """
    # Setup frequency axis
    if freq is None:
        if axes_from is not None and "freq" in axes_from.index_map:
            freq = axes_from.index_map["freq"]
        else:
            raise RuntimeError("No frequency axis defined.")

    # Setup input axis
    if input is None:
        if axes_from is not None and "input" in axes_from.index_map:
            input = axes_from.index_map["input"]
        else:
            raise RuntimeError("No input axis defined.")

    # Setup time axis
    if time is None:
        if axes_from is not None and "time" in axes_from.index_map:
            time = axes_from.index_map["time"]
        else:
            raise RuntimeError("No time axis defined.")

    # Create CorrData object and setup axies
    from ch_util import andata

    # Initialise distributed container
    data = andata.CorrData.__new__(andata.CorrData)
    Container.__init__(data, distributed=True, comm=comm)

    # Copy over attributes
    if attrs_from is not None:
        memdata.copyattrs(attrs_from.attrs, data.attrs)

    # Create index map
    data.create_index_map("freq", freq)
    data.create_index_map("input", input)
    data.create_index_map("time", time)

    # Construct and create product map
    if axes_from is not None and "prod" in axes_from.index_map:
        prodmap = axes_from.index_map["prod"]
    else:
        nfeed = len(input)
        prodmap = np.array([[fi, fj] for fi in range(nfeed) for fj in range(fi, nfeed)])
    data.create_index_map("prod", prodmap)

    # Construct and create stack map
    if axes_from is not None and "stack" in axes_from.index_map:
        stackmap = axes_from.index_map["stack"]
        vis_shape = (data.nfreq, len(stackmap), data.ntime)
        vis_axis = np.array(["freq", "stack", "time"])
    else:
        stackmap = np.empty_like(prodmap, dtype=[("prod", "<u4"), ("conjugate", "u1")])
        stackmap["prod"][:] = np.arange(len(prodmap))
        stackmap["conjugate"] = 0
        vis_shape = (data.nfreq, data.nprod, data.ntime)
        vis_axis = np.array(["freq", "prod", "time"])
    data.create_index_map("stack", stackmap)

    # Construct and create reverse map stack
    if axes_from is not None and "stack" in axes_from.reverse_map:
        reverse_map_stack = axes_from.reverse_map["stack"]
        data.create_reverse_map("stack", reverse_map_stack)

    # Determine datatype for weights
    if (
        (axes_from is not None)
        and hasattr(axes_from, "flags")
        and ("vis_weight" in axes_from.flags)
    ):
        weight_dtype = axes_from.flags["vis_weight"].dtype
    else:
        weight_dtype = np.float32

    # Create empty datasets, and add axis attributes to them
    dset = data.create_dataset(
        "vis",
        shape=vis_shape,
        dtype=np.complex64,
        distributed=distributed,
        distributed_axis=distributed_axis,
    )
    dset.attrs["axis"] = vis_axis
    dset[:] = 0.0

    dset = data.create_flag(
        "vis_weight",
        shape=vis_shape,
        dtype=weight_dtype,
        distributed=distributed,
        distributed_axis=distributed_axis,
    )
    dset.attrs["axis"] = vis_axis
    dset[:] = 0.0

    dset = data.create_flag(
        "inputs",
        shape=(data.ninput, data.ntime),
        dtype=np.float32,
        distributed=False,
        distributed_axis=None,
    )
    dset.attrs["axis"] = np.array(["input", "time"])
    dset[:] = 0.0

    dset = data.create_dataset(
        "gain",
        shape=(data.nfreq, data.ninput, data.ntime),
        dtype=np.complex64,
        distributed=distributed,
        distributed_axis=distributed_axis,
    )
    dset.attrs["axis"] = np.array(["freq", "input", "time"])
    dset[:] = 0.0

    return data
