"""
==============================================================
Parallel data containers (:mod:`~ch_pipeline.core.containers`)
==============================================================

.. currentmodule:: ch_pipeline.core.containers

Containers for holding various types of analysis data in a dsitributed fashion.

Containers
==========

.. autosummary::
    :toctree: generated/

    TimeStream
    MaskedTimeStream
    SiderealStream
"""

import numpy as np

from caput import memh5


class ContainerBase(memh5.BasicCont):
    """A base class for pipeline containers.

    This class is designed to do much of the work of setting up pipeline
    containers. It should be derived from, and two variables set `_axes` and
    `_dataset_spec`.

    The variable `_axes` should be a tuple containing the names of axes that
    datasets in this container will use.

    The variable `_dataset_spec` should define the datasets. It's a dictionary
    with the name of the dataset as key. Each entry should be another
    dictionary, the entry 'axes' is mandatory and should be a list of the axes
    the dataset has (these should correspond to entries in `_axes`), as is
    `dtype` which should be a datatype understood by numpy. Other possible
    entries are:

    - `initialise` : if set to `True` the dataset will be created as the container is initialised.

    - `distributed` : the dataset will be distributed if the entry is `True`, if
      `False` it won't be, and if not set it will be distributed if the
      container is set to be.

    - `distributed_axis` : the axis to distribute over. Should be a name given in the `axes` entry.

    Parameters
    ----------
    axes_from : `memh5.BasicCont`, optional
        Another container to copy axis definitions from.
    kwargs : dict
        Should contain entries for all other axes.
    """

    _axes = ()

    _dataset_spec = {}

    def __init__(self, axes_from=None, attrs_from=None, *args, **kwargs):

        dist = kwargs['distributed'] if 'distributed' in kwargs else True
        comm = kwargs['comm'] if 'comm' in kwargs and dist else None

        # Run base initialiser
        memh5.BasicCont.__init__(self, distributed=dist, comm=comm)

        # Copy over attributes
        if attrs_from is not None:
            memh5.copyattrs(attrs_from.attrs, self.attrs)

        # Create axis entries
        for axis in self._axes:

            axis_map = None

            # Check if axis is specified in initialiser
            if axis in kwargs:

                # If axis is an integer, turn into an arange as a default definition
                if isinstance(kwargs[axis], int):
                    axis_map = np.arange(kwargs[axis])
                else:
                    axis_map = kwargs[axis]

            # If not set in the arguments copy from another object if set
            elif axes_from is not None and axis in axes_from.index_map:
                axis_map = axes_from.index_map[axis]

            # Set the index_map[axis] if we have a definition, otherwise throw an error
            if axis_map is not None:
                self.create_index_map(axis, axis_map)
            else:
                raise RuntimeError('No definition of axis %s supplied.' % axis)

        # Iterate over datasets and initialise any that specify it
        for name, spec in self._dataset_spec.items():
            if 'initialise' in spec and spec['initialise']:
                self.add_dataset(name)

    def add_dataset(self, name):
        """Create an empty dataset.

        The dataset must be defined in the specification for the container.

        Parameters
        ----------
        name : string
            Name of the dataset to create.

        Returns
        -------
        dset : `memh5.MemDataset`
        """

        # Dataset must be specified
        if name not in self._dataset_spec:
            raise RuntimeError('Dataset name not known.')

        dspec = self._dataset_spec[name]

        # Fetch dataset properties
        axes = dspec['axes']
        dtype = dspec['dtype']

        # Get distribution properties
        dist = dspec['distributed'] if 'distributed' in dspec else self._data._distributed
        shape = ()

        # Check that all the specified axes are defined, and fetch their lengths
        for axis in axes:
            if axis not in self.index_map:
                if isinstance(axis, int):
                    l = axis
                else:
                    raise RuntimeError('Axis not defined in index_map')
            else:
                l = len(self.index_map[axis])

            shape += (l,)

        # Fetch distributed axis, and turn into axis index
        dist_axis = dspec['distributed_axis'] if 'distributed_axis' in dspec else axes[0]
        dist_axis = list(axes).index(dist_axis)

        # Create dataset
        dset = self.create_dataset(name, shape=shape, dtype=dtype, distributed=dist,
                                   distributed_axis=dist_axis)

        dset.attrs['axis'] = np.array(axes)

        return dset

    @property
    def datasets(self):
        """Return the datasets in this container.

        Do not try to add a new dataset by assigning to an item of this
        property. Use `create_dataset` instead.

        Returns
        -------
        datasets : read only dictionary
            Entries are :mod:`caput.memh5` datasets.

        """
        out = {}
        for name, value in self._data.iteritems():
            if not memh5.is_group(value):
                out[name] = value
        return memh5.ro_dict(out)


class Map(ContainerBase):
    """Container for holding multifrequency sky maps.

    The maps are packed in format `[freq, pol, pixel]` where the polarisations
    are Stokes I, Q, U and V, and the pixel dimension stores a Healpix map.

    Parameters
    ----------
    nside : int
        The nside of the Healpix maps.
    polarisation : bool, optional
        If `True` all Stokes parameters are stored, if `False` only Stokes I is
        stored.
    """

    _axes = ('freq', 'pol', 'pixel')

    _dataset_spec = {
        'map': {
            'axes': ['freq', 'pol', 'pixel'],
            'dtype': np.float64,
            'initialise': True,
            'distributed': True,
            'distributed_axis': 'freq'
        }
    }

    def __init__(self, nside=None, polarisation=True, *args, **kwargs):

        # Set up axes from passed arguments
        if nside is not None:
            kwargs['pixel'] = 12 * nside**2

        kwargs['pol'] = np.array(['I', 'Q', 'U', 'V']) if polarisation else np.array(['I'])

        super(Map, self).__init__(*args, **kwargs)

    @property
    def freq(self):
        return self.index_map['freq']

    @property
    def map(self):
        return self.datasets['map']


class SiderealStream(ContainerBase):
    """A container for holding a visibility dataset in sidereal time.

    Parameters
    ----------
    ra : int
        The number of points to divide the RA axis up into.
    """

    _axes = ('freq', 'prod', 'input', 'ra')

    _dataset_spec = {
        'vis': {
            'axes': ['freq', 'prod', 'ra'],
            'dtype': np.complex64,
            'initialise': True,
            'distributed': True,
            'distributed_axis': 'freq'
        },

        'vis_weight': {
            'axes': ['freq', 'prod', 'ra'],
            'dtype': np.float32,
            'initialise': True,
            'distributed': True,
            'distributed_axis': 'freq'
        },

        'gain': {
            'axes': ['freq', 'input', 'ra'],
            'dtype': np.complex64,
            'initialise': False,
            'distributed': True,
            'distributed_axis': 'freq'
        }
    }

    def __init__(self, ra=None, *args, **kwargs):

        # Set up axes passed ra langth
        if ra is not None:
            if isinstance(ra, int):
                ra = np.linspace(0.0, 360.0, ra, endpoint=False)
            kwargs['ra'] = ra

        # Resolve product map
        prod = None
        if 'prod' in kwargs:
            prod = kwargs['prod']
        elif ('axes_from' in kwargs) and ('prod' in kwargs['axes_from'].index_map):
            prod = kwargs['axes_from'].index_map['prod']

        # Resolve input map
        inputs = None
        if 'input' in kwargs:
            inputs = kwargs['input']
        elif ('axes_from' in kwargs) and ('input' in kwargs['axes_from'].index_map):
            inputs = kwargs['axes_from'].index_map['input']

        # Automatically construct product map from inputs if not given
        if prod is None and inputs is not None:
            nfeed = inputs if isinstance(inputs, int) else len(inputs)
            kwargs['prod'] = np.array([[fi, fj] for fi in range(nfeed) for fj in range(fi, nfeed)])

        # Initialize from parent class
        super(SiderealStream, self).__init__(*args, **kwargs)

    @property
    def vis(self):
        return self.datasets['vis']

    @property
    def gain(self):
        return self.datasets['gain']

    @property
    def weight(self):
        return self.datasets['vis_weight']

    @property
    def ra(self):
        return self.index_map['ra']

    @property
    def freq(self):
        return self.index_map['freq']

    @property
    def input(self):
        return self.index_map['input']


class MModes(ContainerBase):
    """Parallel container for holding m-mode data.

    Parameters
    ----------
    mmax : integer
        Number of samples in RA.
    nfreq : integer
        Number of frequencies.
    ncorr : integer
        Number of correlation products.
    comm : MPI.Comm
        MPI communicator to distribute over.

    Attributes
    ----------
    vis : mpidataset.MPIArray
        Visibility array.
    weight : mpidataset.MPIArray
        Array of weights for each point.
    """

    _axes = ('m', 'msign', 'freq', 'prod', 'input')

    _dataset_spec = {
        'vis': {
            'axes': ['m', 'msign', 'freq', 'prod'],
            'dtype': np.complex128,
            'initialise': True,
            'distributed': True,
            'distributed_axis': 'm'
        },

        'vis_weight': {
            'axes': ['m', 'msign', 'freq', 'prod'],
            'dtype': np.float64,
            'initialise': True,
            'distributed': True,
            'distributed_axis': 'm'
        },
    }

    @property
    def vis(self):
        return self.datasets['vis']

    @property
    def weight(self):
        return self.datasets['vis_weight']

    @property
    def ra(self):
        return self.index_map['ra']

    @property
    def freq(self):
        return self.index_map['freq']

    @property
    def input(self):
        return self.index_map['input']

    def __init__(self, mmax=None, *args, **kwargs):

        # Set up axes from passed arguments
        if mmax is not None:
            kwargs['m'] = mmax + 1

        # Ensure the sign axis is set correctly
        kwargs['msign'] = np.array(['+', '-'])

        super(MModes, self).__init__(*args, **kwargs)


class GainData(ContainerBase):
    """Parallel container for holding gain data.
    """

    _axes = ('freq', 'input', 'time', 'good_input')

    _dataset_spec = {
        'gain': {
            'axes': ['freq', 'input', 'time'],
            'dtype': np.complex128,
            'initialise': True,
            'distributed': True,
            'distributed_axis': 'freq'
        },
        'weight': {
            'axes': ['freq'],
            'dtype': np.float64,
            'initialise': False,
            'distributed': True,
            'distributed_axis': 'freq'
            },
        'evalue': {
            'axes': ['freq', 'good_input', 'time'],
            'dtype': np.float64,
            'initialise': True,
            'distributed': True,
            'distributed_axis': 'freq'
            },     
        'illumination_med': {
            'axes': ['freq', 'input'],
            'dtype': np.float64,
            'initialise': True,
            'distributed': True,
            'distributed_axis': 'freq'
            },        
        'gain_error_med': {
            'axes': ['freq', 'input'],
            'dtype': np.float64,
            'initialise': True,
            'distributed': True,
            'distributed_axis': 'freq'
            }      
    }

    @property
    def gain(self):
        return self.datasets['gain']

    @property
    def evalue(self):
        return self.datasets['evalue']

    @property
    def illumination_med(self):
        return self.datasets['illumination_med']

    @property
    def gain_error_med(self):
        return self.datasets['gain_error_med']

    @property
    def weight(self):
        return self.datasets['weight']

    @property
    def time(self):
        try:
            return self.index_map['time'][:]['ctime']
        # Need to check for both types as different numpy versions return
        # different exceptions.
        except (IndexError, ValueError):
            return self.index_map['time'][:]

    @property
    def freq(self):
        return self.index_map['freq']

    @property
    def input(self):
        return self.index_map['input']

    @property
    def good_input(self):
        return self.index_map['good_input']


class CorrInputMask(ContainerBase):
    """Container for holding mask indicating good correlator inputs.
    """

    _axes = ('input', )

    _dataset_spec = {
        'input_mask': {
            'axes': ['input'],
            'dtype': np.bool,
            'initialise': True,
            'distributed': False,
            }
        }

    @property
    def input_mask(self):
        return self.datasets['input_mask']

    @property
    def input(self):
        return self.index_map['input']



class CorrInputTest(ContainerBase):
    """Container for holding results of tests for good correlator inputs.
    """

    _axes = ('freq', 'input', 'test')

    _dataset_spec = {
        'input_mask': {
            'axes': ['input'],
            'dtype': np.bool,
            'initialise': True,
            'distributed': False,
            },
        'passed_test': {
            'axes': ['freq', 'input', 'test'],
            'dtype': np.bool,
            'initialise': False,
            'distributed': False,
            }
        }

    def __init__(self, *args, **kwargs):

        if 'test' not in kwargs:
            kwargs['test'] = np.array(['is_chime', 'not_known_bad', 'digital_gain', 'radiometer', 'sky_fit'])

        super(CorrInputTest, self).__init__(*args, **kwargs)

    @property
    def input_mask(self):
        return self.datasets['input_mask']

    @property
    def passed_test(self):
        return self.datasets['passed_test']

    @property
    def freq(self):
        return self.index_map['freq']

    @property
    def input(self):
        return self.index_map['input']

    @property
    def test(self):
        return self.index_map['test']


class CorrInputMonitor(ContainerBase):
    """Container for holding results of good correlator inputs monitor.
    """

    _axes = ('freq', 'input', 'coord')

    _dataset_spec = {
        'input_mask': {
            'axes': ['input'],
            'dtype': np.bool,
            'initialise': True,
            'distributed': False,
            },
        'input_powered': {
            'axes': ['input'],
            'dtype': np.bool,
            'initialise': True,
            'distributed': False,
            },
        'freq_mask': {
            'axes': ['freq'],
            'dtype': np.bool,
            'initialise': True,
            'distributed': False,
            },
        'freq_powered': {
            'axes': ['freq'],
            'dtype': np.bool,
            'initialise': True,
            'distributed': False,
            },
        'position': {
            'axes': ['input', 'coord'],
            'dtype': np.float,
            'initialise': False,
            'distributed': False,
            },
        'expected_position': {
            'axes': ['input', 'coord'],
            'dtype': np.float,
            'initialise': False,
            'distributed': False,
            }
        }

    def __init__(self, *args, **kwargs):

        if 'coord' not in kwargs:
            kwargs['coord'] = np.array(['east_west', 'north_south'])

        super(CorrInputMonitor, self).__init__(*args, **kwargs)

    @property
    def input_mask(self):
        return self.datasets['input_mask']

    @property
    def input_powered(self):
        return self.datasets['input_powered']

    @property
    def position(self):
        return self.datasets['position']

    @property
    def expected_position(self):
        return self.datasets['expected_position']

    @property
    def freq_mask(self):
        return self.datasets['freq_mask']

    @property
    def freq_powered(self):
        return self.datasets['freq_powered']

    @property
    def freq(self):
        return self.index_map['freq']

    @property
    def input(self):
        return self.index_map['input']

    @property
    def coord(self):
        return self.index_map['coord']


class SiderealDayFlag(ContainerBase):
    """Container for holding flag that indicates
       good chime sidereal days.
    """

    _axes = ('csd', )

    _dataset_spec = {
        'csd_flag': {
            'axes': ['csd'],
            'dtype': np.bool,
            'initialise': True,
            'distributed': False,
            }
        }

    @property
    def csd_flag(self):
        return self.datasets['csd_flag']

    @property
    def csd(self):
        return self.index_map['csd']



class StaticGainData(ContainerBase):
    """Parallel container for holding static gain data (i.e. non time varying).
    """

    _axes = ('freq', 'input')

    _dataset_spec = {
        'gain': {
            'axes': ['freq', 'input'],
            'dtype': np.complex128,
            'initialise': True,
            'distributed': True,
            'distributed_axis': 'freq'
        },
        'weight': {
            'axes': ['freq'],
            'dtype': np.float64,
            'initialise': False,
            'distributed': True,
            'distributed_axis': 'freq'
        }
    }

    @property
    def gain(self):
        return self.datasets['gain']

    @property
    def weight(self):
        return self.datasets['weight']

    @property
    def freq(self):
        return self.index_map['freq']

    @property
    def input(self):
        return self.index_map['input']


class PointSourceTransit(ContainerBase):
    """Parallel container for holding the results of a fit to a point source transit.
    """

    _axes = ('freq', 'input', 'ra', 'pol_x', 'pol_y',
             'param', 'param_cov1', 'param_cov2')

    _dataset_spec = {
        'gain': {
            'axes': ['freq', 'input'],
            'dtype': np.complex128,
            'initialise': True,
            'distributed': True,
            'distributed_axis': 'freq'
            },
        'weight': {
            'axes': ['freq'],
            'dtype': np.float64,
            'initialise': False,
            'distributed': True,
            'distributed_axis': 'freq'
            },
        'evalue_x': {
            'axes': ['freq', 'pol_x', 'ra'],
            'dtype': np.float64,
            'initialise': True,
            'distributed': True,
            'distributed_axis': 'freq'
            },
        'evalue_y': {
            'axes': ['freq', 'pol_y', 'ra'],
            'dtype': np.float64,
            'initialise': True,
            'distributed': True,
            'distributed_axis': 'freq'
            },
        'response': {
            'axes': ['freq', 'input', 'ra'],
            'dtype': np.complex128,
            'initialise': True,
            'distributed': True,
            'distributed_axis': 'freq'
            },
        'response_error': {
            'axes': ['freq', 'input', 'ra'],
            'dtype': np.float64,
            'initialise': True,
            'distributed': True,
            'distributed_axis': 'freq'
            },
        'flag': {
            'axes': ['freq', 'input', 'ra'],
            'dtype': np.bool,
            'initialise': True,
            'distributed': True,
            'distributed_axis': 'freq'
            },
        'parameter': {
            'axes': ['freq', 'input', 'param'],
            'dtype': np.float64,
            'initialise': True,
            'distributed': True,
            'distributed_axis': 'freq'
            },
        'parameter_cov': {
            'axes': ['freq', 'input', 'param_cov1', 'param_cov2'],
            'dtype': np.float64,
            'initialise': True,
            'distributed': True,
            'distributed_axis': 'freq'
            }
        }

    def __init__(self, *args, **kwargs):

        kwargs['param'] = np.array(['peak_amplitude', 'centroid', 'fwhm', 'phase_intercept', 'phase_slope'])
        kwargs['param_cov1'] = np.array(['peak_amplitude', 'centroid', 'fwhm', 'phase_intercept', 'phase_slope'])
        kwargs['param_cov2'] = np.array(['peak_amplitude', 'centroid', 'fwhm', 'phase_intercept', 'phase_slope'])

        super(PointSourceTransit, self).__init__(*args, **kwargs)

    @property
    def gain(self):
        return self.datasets['gain']

    @property
    def weight(self):
        return self.datasets['weight']

    @property
    def evalue_x(self):
        return self.datasets['evalue_x']

    @property
    def evalue_y(self):
        return self.datasets['evalue_y']

    @property
    def response(self):
        return self.datasets['response']

    @property
    def response_error(self):
        return self.datasets['response_error']

    @property
    def flag(self):
        return self.datasets['flag']

    @property
    def parameter(self):
        return self.datasets['parameter']

    @property
    def parameter_cov(self):
        return self.datasets['parameter_cov']

    @property
    def freq(self):
        return self.index_map['freq']

    @property
    def input(self):
        return self.index_map['input']

    @property
    def param(self):
        return self.index_map['param']

    @property
    def param_cov1(self):
        return self.index_map['param_cov1']

    @property
    def param_cov2(self):
        return self.index_map['param_cov2']


class SunTransit(ContainerBase):
    """Parallel container for holding the results of a fit to a point source transit.
    """

    _axes = ('freq', 'input', 'time', 'pol_x', 'pol_y', 'coord', 'param')

    _dataset_spec = {
        'coord': {
            'axes': ['time', 'coord'],
            'dtype': np.float64,
            'initialise': True,
            'distributed': False,
            },
        'evalue_x': {
            'axes': ['freq', 'pol_x', 'time'],
            'dtype': np.float64,
            'initialise': True,
            'distributed': True,
            'distributed_axis': 'freq'
            },
        'evalue_y': {
            'axes': ['freq', 'pol_y', 'time'],
            'dtype': np.float64,
            'initialise': True,
            'distributed': True,
            'distributed_axis': 'freq'
            },
        'response': {
            'axes': ['freq', 'input', 'time'],
            'dtype': np.complex128,
            'initialise': True,
            'distributed': True,
            'distributed_axis': 'freq'
            },
        'response_error': {
            'axes': ['freq', 'input', 'time'],
            'dtype': np.float64,
            'initialise': True,
            'distributed': True,
            'distributed_axis': 'freq'
            },
        'flag': {
            'axes': ['freq', 'input', 'time'],
            'dtype': np.bool,
            'initialise': False,
            'distributed': True,
            'distributed_axis': 'freq'
            },
        'parameter': {
            'axes': ['freq', 'input', 'param'],
            'dtype': np.float64,
            'initialise': False,
            'distributed': True,
            'distributed_axis': 'freq'
            },
        'parameter_cov': {
            'axes': ['freq', 'input', 'param', 'param'],
            'dtype': np.float64,
            'initialise': False,
            'distributed': True,
            'distributed_axis': 'freq'
            }
        }

    def __init__(self, *args, **kwargs):

        kwargs['param'] = np.array(['peak_amplitude', 'centroid', 'fwhm', 'phase_intercept', 'phase_slope'])
        kwargs['coord'] = np.array(['ha', 'dec', 'alt', 'az'])

        super(SunTransit, self).__init__(*args, **kwargs)

    @property
    def coord(self):
        return self.datasets['coord']

    @property
    def evalue_x(self):
        return self.datasets['evalue_x']

    @property
    def evalue_y(self):
        return self.datasets['evalue_y']

    @property
    def response(self):
        return self.datasets['response']

    @property
    def response_error(self):
        return self.datasets['response_error']

    @property
    def flag(self):
        return self.datasets['flag']

    @property
    def parameter(self):
        return self.datasets['parameter']

    @property
    def parameter_cov(self):
        return self.datasets['parameter_cov']

    @property
    def freq(self):
        return self.index_map['freq']

    @property
    def input(self):
        return self.index_map['input']

    @property
    def time(self):
        return self.index_map['time']

    @property
    def param(self):
        return self.index_map['param']

    @property
    def ha(self):
        ind = list(self.index_map['coord']).index('ha')
        return self.datasets['coord'][:, ind]

    @property
    def dec(self):
        ind = list(self.index_map['coord']).index('dec')
        return self.datasets['coord'][:, ind]

    @property
    def alt(self):
        ind = list(self.index_map['coord']).index('alt')
        return self.datasets['coord'][:, ind]

    @property
    def az(self):
        ind = list(self.index_map['coord']).index('az')
        return self.datasets['coord'][:, ind]


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

    _axes = ('freq', 'pol', 'ra', 'beam', 'el')

    _dataset_spec = {
        'map': {
            'axes': ['freq', 'pol', 'ra', 'beam', 'el'],
            'dtype': np.float64,
            'initialise': True,
            'distributed': True,
            'distributed_axis': 'freq'
            },
        'dirty_beam': {
            'axes': ['freq', 'pol', 'ra', 'beam', 'el'],
            'dtype': np.float64,
            'initialise': False,
            'distributed': True,
            'distributed_axis': 'freq'
            },
        'rms': {
            'axes': ['freq', 'pol', 'ra'],
            'dtype': np.float64,
            'initialise': False,
            'distributed': True,
            'distributed_axis': 'freq'
            }
    }

    def __init__(self, *args, **kwargs):

        super(RingMap, self).__init__(*args, **kwargs)

    @property
    def freq(self):
        return self.index_map['freq']

    @property
    def pol(self):
        return self.index_map['pol']

    @property
    def ra(self):
        return self.index_map['ra']

    @property
    def el(self):
        return self.index_map['el']

    @property
    def map(self):
        return self.datasets['map']

    @property
    def rms(self):
        return self.datasets['rms']

    @property
    def dirty_beam(self):
        return self.datasets['dirty_beam']


class Photometry(ContainerBase):
    """Parallel container for holding photometry extracted from a map.
    """

    _axes = ('freq', 'pol', 'param', 'source')

    _dataset_spec = {
        'parameter': {
            'axes': ['freq', 'pol', 'param', 'source'],
            'dtype': np.float64,
            'initialise': True,
            'distributed': True,
            'distributed_axis': 'freq'
            },
        'parameter_cov': {
            'axes': ['freq', 'pol', 'param', 'param', 'source'],
            'dtype': np.float64,
            'initialise': True,
            'distributed': True,
            'distributed_axis': 'freq'
            },
        'rms': {
            'axes': ['freq', 'pol', 'source'],
            'dtype': np.float64,
            'initialise': True,
            'distributed': True,
            'distributed_axis': 'freq'
            }
    }

    @property
    def parameter(self):
        return self.datasets['parameter']

    @property
    def parameter_cov(self):
        return self.datasets['parameter_cov']

    @property
    def rms(self):
        return self.datasets['rms']

    @property
    def freq(self):
        return self.index_map['freq']

    @property
    def param(self):
        return self.index_map['param']

    @property
    def source(self):
        return self.index_map['source']


def make_empty_corrdata(freq=None, input=None, time=None,
                        axes_from=None, attrs_from=None,
                        distributed=True, distributed_axis=0, comm=None):
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
        if axes_from is not None and 'freq' in axes_from.index_map:
            freq = axes_from.index_map['freq']
        else:
            raise RuntimeError('No frequency axis defined.')

    # Setup input axis
    if input is None:
        if axes_from is not None and 'input' in axes_from.index_map:
            input = axes_from.index_map['input']
        else:
            raise RuntimeError('No input axis defined.')

    # Setup time axis
    if time is None:
        if axes_from is not None and 'time' in axes_from.index_map:
            time = axes_from.index_map['time']
        else:
            raise RuntimeError('No time axis defined.')

    # Create CorrData object and setup axies
    from ch_util import andata

    # Initialise distributed container
    data = andata.CorrData.__new__(andata.CorrData)
    memh5.BasicCont.__init__(data, distributed=True, comm=comm)

    # Copy over attributes
    if attrs_from is not None:
        memh5.copyattrs(attrs_from.attrs, data.attrs)

    # Create index map
    data.create_index_map('freq', freq)
    data.create_index_map('input', input)
    data.create_index_map('time', time)

    # Construct and create product map
    if axes_from is not None and 'prod' in axes_from.index_map:
        prodmap = axes_from.index_map['prod']
    else:
        nfeed = len(input)
        prodmap = np.array([[fi, fj] for fi in range(nfeed) for fj in range(fi, nfeed)])
    data.create_index_map('prod', prodmap)

    # Create empty datasets, and add axis attributes to them
    dset = data.create_dataset('vis', shape=(data.nfreq, data.nprod, data.ntime), dtype=np.complex64,
                               distributed=distributed, distributed_axis=distributed_axis)
    dset.attrs['axis'] = np.array(['freq', 'prod', 'time'])
    dset[:] = 0.0

    dset = data.create_flag('vis_weight', shape=(data.nfreq, data.nprod, data.ntime), dtype=np.float32,
                               distributed=distributed, distributed_axis=distributed_axis)
    dset.attrs['axis'] = np.array(['freq', 'prod', 'time'])
    dset[:] = 0.0

    dset = data.create_dataset('gain', shape=(data.nfreq, data.ninput, data.ntime), dtype=np.complex64,
                               distributed=distributed, distributed_axis=distributed_axis)
    dset.attrs['axis'] = np.array(['freq', 'input', 'time'])
    dset[:] = 0.0

    return data
