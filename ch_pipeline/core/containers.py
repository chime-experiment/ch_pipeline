"""Parallel data containers

Containers for holding various types of CHIME specific analysis data in a
distributed fashion. The module `draco.core.containers` contains general data
containers which are imported into this module.

Containers
==========
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

Tasks
=====
- :py:class:`MonkeyPatchContainers`
"""
import posixpath
from typing import List, Optional, Union
from draco.core.task import MPILoggedTask

import numpy as np

from caput import memh5, tod
from ch_util import andata

from draco.core.containers import (
    ContainerBase,
    StaticGainData,
    TODContainer,
    FreqContainer,
    TimeStream,
)


class RFIMask(ContainerBase):
    """Container for holding a mask that indicates
    data that is free of RFI events.
    """

    _axes = ("freq", "input", "time")

    _dataset_spec = {
        "mask": {
            "axes": ["freq", "input", "time"],
            "dtype": np.bool,
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


class CorrInputMask(ContainerBase):
    """Container for holding mask indicating good correlator inputs."""

    _axes = ("input",)

    _dataset_spec = {
        "input_mask": {
            "axes": ["input"],
            "dtype": np.bool,
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


class CorrInputTest(ContainerBase):
    """Container for holding results of tests for good correlator inputs."""

    _axes = ("freq", "input", "test")

    _dataset_spec = {
        "input_mask": {
            "axes": ["input"],
            "dtype": np.bool,
            "initialise": True,
            "distributed": False,
        },
        "passed_test": {
            "axes": ["freq", "input", "test"],
            "dtype": np.bool,
            "initialise": False,
            "distributed": False,
        },
    }

    def __init__(self, *args, **kwargs):

        if "test" not in kwargs:
            kwargs["test"] = np.array(
                ["is_chime", "not_known_bad", "digital_gain", "radiometer", "sky_fit"]
            )

        super(CorrInputTest, self).__init__(*args, **kwargs)

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


class CorrInputMonitor(ContainerBase):
    """Container for holding results of good correlator inputs monitor."""

    _axes = ("freq", "input", "coord")

    _dataset_spec = {
        "input_mask": {
            "axes": ["input"],
            "dtype": np.bool,
            "initialise": True,
            "distributed": False,
        },
        "input_powered": {
            "axes": ["input"],
            "dtype": np.bool,
            "initialise": True,
            "distributed": False,
        },
        "freq_mask": {
            "axes": ["freq"],
            "dtype": np.bool,
            "initialise": True,
            "distributed": False,
        },
        "freq_powered": {
            "axes": ["freq"],
            "dtype": np.bool,
            "initialise": True,
            "distributed": False,
        },
        "position": {
            "axes": ["input", "coord"],
            "dtype": np.float,
            "initialise": False,
            "distributed": False,
        },
        "expected_position": {
            "axes": ["input", "coord"],
            "dtype": np.float,
            "initialise": False,
            "distributed": False,
        },
    }

    def __init__(self, *args, **kwargs):

        if "coord" not in kwargs:
            kwargs["coord"] = np.array(["east_west", "north_south"])

        super(CorrInputMonitor, self).__init__(*args, **kwargs)

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


class SiderealDayFlag(ContainerBase):
    """Container for holding flag that indicates
    good chime sidereal days.
    """

    _axes = ("csd",)

    _dataset_spec = {
        "csd_flag": {
            "axes": ["csd"],
            "dtype": np.bool,
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


class TransitFitParams(ContainerBase):
    """Parallel container for holding the results of fitting a model to a point source transit."""

    _axes = ("freq", "input", "param", "component")

    _dataset_spec = {
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
            "dtype": np.int,
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

    _dataset_spec = {
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
            "dtype": np.bool,
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

        super(PointSourceTransit, self).__init__(*args, **kwargs)

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

    _dataset_spec = {
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


class SunTransit(ContainerBase):
    """Parallel container for holding the results of a fit to a point source transit."""

    _axes = ("freq", "input", "time", "pol_x", "pol_y", "coord", "param")

    _dataset_spec = {
        "coord": {
            "axes": ["time", "coord"],
            "dtype": np.float64,
            "initialise": True,
            "distributed": False,
        },
        "evalue_x": {
            "axes": ["freq", "pol_x", "time"],
            "dtype": np.float64,
            "initialise": True,
            "distributed": True,
            "distributed_axis": "freq",
        },
        "evalue_y": {
            "axes": ["freq", "pol_y", "time"],
            "dtype": np.float64,
            "initialise": True,
            "distributed": True,
            "distributed_axis": "freq",
        },
        "response": {
            "axes": ["freq", "input", "time"],
            "dtype": np.complex128,
            "initialise": True,
            "distributed": True,
            "distributed_axis": "freq",
        },
        "response_error": {
            "axes": ["freq", "input", "time"],
            "dtype": np.float64,
            "initialise": True,
            "distributed": True,
            "distributed_axis": "freq",
        },
        "flag": {
            "axes": ["freq", "input", "time"],
            "dtype": np.bool,
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
            ["peak_amplitude", "centroid", "fwhm", "phase_intercept", "phase_slope"]
        )
        kwargs["coord"] = np.array(["ha", "dec", "alt", "az"])

        super(SunTransit, self).__init__(*args, **kwargs)

    @property
    def coord(self):
        return self.datasets["coord"]

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


class RingMap(ContainerBase):
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

    _dataset_spec = {
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

        super(RingMap, self).__init__(*args, **kwargs)

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


class Photometry(ContainerBase):
    """Parallel container for holding photometry extracted from a map."""

    _axes = ("freq", "pol", "param", "source")

    _dataset_spec = {
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


class RawContainer(TODContainer):
    """Base class for using raw CHIME data via the draco containers API.

    This modifies the set of allowed group/dataset names to allow access to the
    flags group.
    """

    _allowed_groups = ["flags"]

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
        acq_files: Union[str, List[str]],
        start: Optional[int] = None,
        stop: Optional[int] = None,
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
        return self.datasets["flags/hfb_weight"]


class CHIMETimeStream(TimeStream, RawContainer):
    """A container for CHIME visibility data.

    This provides a close analog of the CorrData container but implemented via the draco
    containers API.
    """

    # Add in the extra datasets contained within the CHIME corrdata time streams
    _dataset_spec = {
        "flags/dataset_id": {
            "axes": ["freq", "time"],
            "dtype": "U32",
            "initialise": False,
            "distributed": False,
        },
        "flags/frac_lost": {
            "axes": ["freq", "time"],
            "dtype": np.float32,
            "initialise": False,
            "distributed": False,
        },
        "flags/inputs": {
            "axes": ["input", "time"],
            "dtype": np.float32,
            "initialise": False,
            "distributed": False,
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
        if "/flags/vis_weight" in newdata:
            storage = newdata._storage_root._get_storage()
            storage["vis_weight"] = storage["flags"].pop("vis_weight")
            storage["vis_weight"]._name = "/vis_weight"

        return newdata


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
    memh5.BasicCont.__init__(data, distributed=True, comm=comm)

    # Copy over attributes
    if attrs_from is not None:
        memh5.copyattrs(attrs_from.attrs, data.attrs)

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


class MonkeyPatchContainers(MPILoggedTask):
    """Patch draco to use CHIME timestream containers.

    This task does nothing but perform a monkey patch on `draco.core.containers`

    .. deprecated::
        This monkey patching scheme is now deprecated. Try to use just use the usual
        draco infrastructure instead, that is using `draco.core.containers.empty_like`
        to create a new container and `draco.core.containers.TimeStream` as a generic
        timestream class and `ch_pipeline.core.containers.CHIMETimeStream` as a slightly
        more CHIME specific one.
    """

    def __init__(self):

        self.log.warning("Deprecated. Try and stop using this monkey patching scheme.")

        import ch_pipeline.core.containers as ccontainers
        import draco.core.containers as dcontainers

        # Replace the routine for making an empty timestream. This needs to be replaced
        # in both draco and ch_pipeline because of the ways the imports work

        def empty_timestream_patch(*args, **kwargs):
            self.log.warning(
                "This patching is deprecated. Try using `CHIMETimeStream` instead."
            )
            return ccontainers.make_empty_corrdata(*args, **kwargs)

        empty_timestream_patch.__doc__ = ccontainers.make_empty_corrdata.__doc__

        dcontainers.empty_timestream = empty_timestream_patch
        ccontainers.empty_timestream = empty_timestream_patch

        # Save a reference to the original routine
        _make_empty_like = dcontainers.empty_like

        # A new routine which wraps the old empty_like, but can additionally handle
        # andata.CorrData types
        def empty_like_patch(obj, **kwargs):
            """Create an empty container like `obj`.

            Parameters
            ----------
            obj : ContainerBase or CorrData
                Container to base this one off.
            kwargs : optional
                Optional definitions of specific axes we want to override. Works in the
                same way as the `ContainerBase` constructor, though `axes_from=obj` and
                `attrs_from=obj` are implied.

            Returns
            -------
            newobj : container.ContainerBase or CorrData
                New data container.
            """

            from ch_util import andata

            self.log.warning(
                "This patching is deprecated. Try using `CHIMETimeStream` instead."
            )

            if isinstance(obj, andata.CorrData):
                return dcontainers.empty_timestream(
                    axes_from=obj, attrs_from=obj, **kwargs
                )
            else:
                return _make_empty_like(obj, **kwargs)

        # Replace the empty_like routine
        dcontainers.empty_like = empty_like_patch
