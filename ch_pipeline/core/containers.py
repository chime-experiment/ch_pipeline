"""
==============================================================
Parallel data containers (:mod:`~ch_pipeline.core.containers`)
==============================================================

.. currentmodule:: ch_pipeline.core.containers

Containers for holding various types of CHIME specific analysis data in a
distributed fashion. See `draco.core.containers` for more general data
containers.

Containers
==========

.. autosummary::
    :toctree: generated/

    CorrInputMask
    CorrInputTest
    CorrInputMonitor
    SiderealDayFlag
    PointSourceTransit
    SunTransit
    RingMap
    Photometry
"""

import numpy as np

from caput import memh5

from draco.core.containers import ContainerBase


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
