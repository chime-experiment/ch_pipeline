"""HFB containers
"""

import warnings

from typing import Union

import numpy as np

from caput import tod
from caput import memh5
from caput import mpiutil
from caput import fileformats

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


class HFBData(RawContainer, FreqContainer, HFBContainer):
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
        "nsample": {
            "axes": ["freq", "subfreq", "beam", "time"],
            "dtype": np.uint16,
            "initialise": False,
            "distributed": True,
        },
    }

    @classmethod
    def from_file(
        cls,
        filename,
        distributed=False,
        hints=True,
        comm=None,
        convert_dataset_strings=False,
        convert_attribute_strings=True,
        file_format=None,
        **kwargs,
    ):
        """Create a new instance by copying from a file group.

        Any keyword arguments are passed on to the constructor for `h5py.File` or
        `zarr.File`.

        Parameters
        ----------
        filename : string
            Name of file to load.
        distributed : boolean, optional
            Whether to load file in distributed mode.
        hints : boolean, optional
            If in distributed mode use hints to determine whether datasets are
            distributed or not.
        comm : MPI.Comm, optional
            MPI communicator to distributed over. If :obj:`None` use
            :obj:`MPI.COMM_WORLD`.
        convert_attribute_strings : bool, optional
            Try and convert attribute string types to unicode. Default is `True`.
        convert_dataset_strings : bool, optional
            Try and convert dataset string types to unicode. Default is `False`.
        file_format : `fileformats.FileFormat`, optional
            File format to use. Default is `None`, i.e. guess from the name.
        **kwargs : dict
            Arbitrary keyword arguments.

        Returns
        -------
        group : HFBData
            Container with HFB data.
        """
        if comm is None:
            comm = mpiutil.world

        if file_format is None:
            file_format = fileformats.guess_file_format(filename)

        if comm is None:
            if distributed:
                warnings.warn(
                    "Cannot load file in distributed mode when there is no MPI"
                    "communicator!!"
                )
            distributed = False

        # Parse _dataset_spec into hints
        hints = {}
        for key, ds_spec in cls._dataset_spec.items():
            hints_dict = {}
            hints_dict["distributed"] = ds_spec["distributed"]
            if "distributed_axis" in ds_spec:
                distributed_axis = ds_spec["distributed_axis"]
                hints_dict["axis"] = ds_spec["axes"].index(distributed_axis)
            ds_path = "/" + key
            hints[ds_path] = hints_dict

        # Look for *_sel parameters in kwargs, collect and remove them from kwargs
        sel_args = {}
        for a in list(kwargs):
            if a.endswith("_sel"):
                sel_args[a[:-4]] = kwargs.pop(a)

        # Map selections to datasets
        sel = cls._make_selections(sel_args)

        if not distributed or not hints:
            kwargs["mode"] = "r"
            with file_format.open(filename, **kwargs) as f:
                self = cls(distributed=distributed, comm=comm)
                memh5.deep_group_copy(
                    f,
                    self,
                    selections=sel,
                    convert_attribute_strings=convert_attribute_strings,
                    convert_dataset_strings=convert_dataset_strings,
                    file_format=file_format,
                )
        else:
            self = memh5._distributed_group_from_file(
                filename,
                comm=comm,
                hints=hints,
                selections=sel,
                convert_attribute_strings=convert_attribute_strings,
                convert_dataset_strings=convert_dataset_strings,
                file_format=file_format,
            )

        return self


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


class HFBTimeAverage(FreqContainer, HFBContainer):
    """Container for holding average data for flattening sub-frequency band shape."""

    _axes = ("subfreq", "beam")

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


class HFBHighResData(TODContainer, HFBHighResContainer):
    """Container for holding high-resolution frequency data."""

    _axes = ("beam",)

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


class HFBHighResTimeAverage(HFBHighResContainer):
    """Container for holding time-averaged high-resolution frequency data."""

    _axes = ("beam",)

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
    """Base class for HFB ringmaps."""

    _axes = ("beam", "el")

    @property
    def el(self):
        """The elevation in degrees associated with each sample of the elevation axis."""
        return self.index_map["el"]


class HFBRingMap(FreqContainer, HFBRingMapBase):
    """Container for holding HFB ringmap data."""

    _axes = ("subfreq",)

    _dataset_spec = {
        "hfb": {
            "axes": ["freq", "subfreq", "beam", "el", "ra"],
            "dtype": np.float32,
            "initialise": True,
            "distributed": True,
            "distributed_axis": "freq",
        },
        "weight": {
            "axes": ["freq", "subfreq", "beam", "el", "ra"],
            "dtype": np.float32,
            "initialise": True,
            "distributed": True,
            "distributed_axis": "freq",
        },
        "nsample": {
            "axes": ["freq", "subfreq", "beam", "el", "ra"],
            "dtype": np.uint16,
            "initialise": False,
            "distributed": True,
            "distributed_axis": "freq",
        },
    }


class HFBHighResRingMap(HFBRingMapBase, HFBHighResContainer):
    """Container for holding high-resolution frequency ringmap data.

    With respect to :class:`HFBRingMap`, the (combined) frequency axis is moved
    to the back, and the distributed axis is changed to the elevation axis.
    This is because further downstream in the pipeline, we will look for features
    along the frequency axis.
    """

    _dataset_spec = {
        "hfb": {
            "axes": ["beam", "el", "ra", "freq"],
            "dtype": np.float32,
            "initialise": True,
            "distributed": True,
            "distributed_axis": "el",
        },
        "weight": {
            "axes": ["beam", "el", "ra", "freq"],
            "dtype": np.float32,
            "initialise": True,
            "distributed": True,
            "distributed_axis": "el",
        },
        "nsample": {
            "axes": ["beam", "el", "ra", "freq"],
            "dtype": np.uint16,
            "initialise": False,
            "distributed": True,
            "distributed_axis": "el",
        },
    }
