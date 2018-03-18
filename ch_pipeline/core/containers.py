"""
==============================================================
Parallel data containers (:mod:`~ch_pipeline.core.containers`)
==============================================================

.. currentmodule:: ch_pipeline.core.containers

Containers for holding various types of CHIME specific analysis data in a
distributed fashion. The module `draco.core.containers` contains general data
containers which are imported into this module.

Containers
==========

.. autosummary::
    :toctree: generated/

    RFIMask
    GainJumps
    CorrInputMask
    CorrInputTest
    CorrInputMonitor
    SiderealDayFlag
    PointSourceTransit
    SunTransit
    Crosstalk
    CrosstalkGain
    RingMap
    Fringestop
    Photometry
    MapNoise

Tasks
=====

.. autosummary::
    :toctree: generated/

    MonkeyPatchContainers
"""

import numpy as np

from caput import memh5, pipeline

from draco.core.containers import *

class RFIMask(ContainerBase):
    """Container for holding mask that indicates RFI events.
    """

    _axes = ('freq', 'input', 'time')

    _dataset_spec = {
        'mask': {
            'axes': ['freq', 'input', 'time'],
            'dtype': np.bool,
            'initialise': True,
            'distributed': True,
            'distributed_axis': 'freq'
        },
        'ndev': {
            'axes': ['freq', 'input', 'time'],
            'dtype': np.float32,
            'initialise': False,
            'distributed': True,
            'distributed_axis': 'freq'
        }
    }

    @property
    def mask(self):
        return self.datasets['mask']

    @property
    def ndev(self):
        return self.datasets['ndev']


class GainJumps(ContainerBase):
    """Container for holding gain jump candidates.
    """

    _axes = ('candidate', 'window')

    _dataset_spec = {
        'freq': {
            'axes': ['candidate'],
            'dtype': np.float32,
            'initialise': True,
            'distributed': False,
        },
        'input': {
            'axes': ['candidate'],
            'dtype': np.int,
            'initialise': True,
            'distributed': False,
        },
        'time': {
            'axes': ['candidate'],
            'dtype': np.float64,
            'initialise': True,
            'distributed': False,
        },
        'flag': {
            'axes': ['candidate', 'window'],
            'dtype': np.bool,
            'initialise': True,
            'distributed': False,
        },
        'atime': {
            'axes': ['candidate', 'window'],
            'dtype': np.float64,
            'initialise': True,
            'distributed': False,
        },
        'auto': {
            'axes': ['candidate', 'window'],
            'dtype': np.float32,
            'initialise': True,
            'distributed': False,
        }
    }

    @property
    def freq(self):
        return self.datasets['freq']

    @property
    def input(self):
        return self.datasets['input']

    @property
    def time(self):
        return self.datasets['time']

    @property
    def flag(self):
        return self.datasets['flag']

    @property
    def atime(self):
        return self.datasets['atime']

    @property
    def auto(self):
        return self.datasets['auto']


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
    def mask(self):
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
    def mask(self):
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
    def mask(self):
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


class PointSourceTransit(StaticGainData):
    """Parallel container for holding the results of a fit to a point source transit.
    """

    _axes = ('freq', 'input', 'ra', 'pol_x', 'pol_y', 'param')

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
            'axes': ['freq', 'input', 'param', 'param'],
            'dtype': np.float64,
            'initialise': True,
            'distributed': True,
            'distributed_axis': 'freq'
        }
    }

    def __init__(self, *args, **kwargs):

        kwargs['param'] = np.array(['peak_amplitude', 'centroid', 'fwhm',
                                    'phase_intercept', 'phase_slope',
                                    'phase_quad', 'phase_cube',
                                    'phase_quart', 'phase_quint'])

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


class SunTransit(ContainerBase):
    """Parallel container for holding the results of a fit to a solar transit.
    """

    _axes = ('freq', 'input', 'time', 'pol', 'eigen', 'good_input1', 'good_input2',
             'udegree', 'vdegree', 'coord', 'param')

    _dataset_spec = {
        'coord': {
            'axes': ['time', 'coord'],
            'dtype': np.float64,
            'initialise': True,
            'distributed': False,
        },
        'evalue1': {
            'axes': ['freq', 'good_input1', 'time'],
            'dtype': np.float64,
            'initialise': True,
            'distributed': True,
            'distributed_axis': 'freq'
        },
        'evalue2': {
            'axes': ['freq', 'good_input2', 'time'],
            'dtype': np.float64,
            'initialise': False,
            'distributed': True,
            'distributed_axis': 'freq'
        },
        'response': {
            'axes': ['freq', 'input', 'time', 'eigen'],
            'dtype': np.complex128,
            'initialise': True,
            'distributed': True,
            'distributed_axis': 'freq'
        },
        'response_error': {
            'axes': ['freq', 'input', 'time', 'eigen'],
            'dtype': np.float64,
            'initialise': True,
            'distributed': True,
            'distributed_axis': 'freq'
        },
        'coeff': {
            'axes': ['freq', 'pol', 'time', 'udegree', 'vdegree'],
            'dtype': np.complex128,
            'initialise': False,
            'distributed': True,
            'distributed_axis': 'freq'
        },
        'is_sun': {
            'axes': ['freq', 'pol', 'time'],
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

        kwargs['param'] = np.array(['peak_amplitude', 'centroid', 'fwhm',
                                    'phase_intercept', 'phase_slope',
                                    'phase_quad', 'phase_cube',
                                    'phase_quart', 'phase_quint'])
        kwargs['coord'] = np.array(['ha', 'dec', 'alt', 'az'])

        super(SunTransit, self).__init__(*args, **kwargs)

    @property
    def coord(self):
        return self.datasets['coord']

    @property
    def evalue1(self):
        return self.datasets['evalue1']

    @property
    def evalue2(self):
        return self.datasets['evalue2']

    @property
    def evalue_x(self):
        return self.datasets['evalue1']

    @property
    def evalue_y(self):
        return self.datasets['evalue2']

    @property
    def response(self):
        return self.datasets['response']

    @property
    def response_error(self):
        return self.datasets['response_error']

    @property
    def coeff(self):
        return self.datasets['coeff']

    @property
    def is_sun(self):
        return self.datasets['is_sun']

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


class Crosstalk(ContainerBase):
    """Parallel container for holding crosstalk model fit.
    """

    _axes = ('prod', 'input', 'path', 'fdegree', 'tdegree', 'freq', 'ra')

    _dataset_spec = {
        'coeff': {
            'axes': ['prod', 'path', 'fdegree', 'tdegree'],
            'dtype': np.complex128,
            'initialise': True,
            'distributed': True,
            'distributed_axis': 'prod'
        },
        'flag': {
            'axes': ['prod', 'path', 'fdegree', 'tdegree'],
            'dtype': np.bool,
            'initialise': True,
            'distributed': True,
            'distributed_axis': 'prod'
        },
        'delay': {
            'axes': ['prod', 'path'],
            'dtype': np.float64,
            'initialise': True,
            'distributed': True,
            'distributed_axis': 'prod'
        },
        'weight': {
            'axes': ['prod', 'freq', 'ra'],
            'dtype': np.bool,
            'initialise': True,
            'distributed': True,
            'distributed_axis': 'prod'
        }
    }

    def __init__(self, *args, **kwargs):

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

        super(Crosstalk, self).__init__(*args, **kwargs)

    @property
    def coeff(self):
        return self.datasets['coeff']

    @property
    def flag(self):
        return self.datasets['flag']

    @property
    def delay(self):
        return self.datasets['delay']

    @property
    def weight(self):
        return self.datasets['weight']

    @property
    def prod(self):
        return self.index_map['prod']


class CrosstalkGain(ContainerBase):
    """Parallel container for holding gain data
    derived from cross talk.
    """

    _axes = ('freq', 'input', 'ra')#, 'pol_x', 'pol_y')

    _dataset_spec = {
        'receiver_temp': {
            'axes': ['freq', 'input', 'ra'],
            'dtype': np.float64,
            'initialise': True,
            'distributed': True,
            'distributed_axis': 'freq'
        },
        'gain': {
            'axes': ['freq', 'input', 'ra'],
            'dtype': np.complex128,
            'initialise': False,
            'distributed': True,
            'distributed_axis': 'freq'
        },
        'chisq_before': {
            'axes': ['freq', 'ra'],
            'dtype': np.float64,
            'initialise': True,
            'distributed': True,
            'distributed_axis': 'freq'
            },
        'chisq_after': {
            'axes': ['freq', 'ra'],
            'dtype': np.float64,
            'initialise': True,
            'distributed': True,
            'distributed_axis': 'freq'
            }
    }

    @property
    def receiver_temp(self):
        return self.datasets['receiver_temp']

    @property
    def gain(self):
        return self.datasets['gain']

    @property
    def chisq_before(self):
        return self.datasets['chisq_before']

    @property
    def chisq_after(self):
        return self.datasets['chisq_after']

    @property
    def freq(self):
        return self.index_map['freq']

    @property
    def input(self):
        return self.index_map['input']

    @property
    def ra(self):
        return self.index_map['ra']


class RingMap(ContainerBase):
    """Container for holding multifrequency ring maps.

    The maps are packed in format `[freq, pol, ra, EW beam, el]` where
    the polarisations are XX, YY, XY, YX.
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


class Fringestop(ContainerBase):
    """Parallel container for holding photometry extracted from a map.
    """

    _axes = ('freq', 'pol', 'param', 'source', 'index')

    _dataset_spec = {
        'ra': {
            'axes': ['index', 'source'],
            'dtype': np.float64,
            'initialise': True,
            'distributed': False
        },
        'vis': {
            'axes': ['freq', 'pol', 'index', 'source'],
            'dtype': np.complex128,
            'initialise': True,
            'distributed': True,
            'distributed_axis': 'freq'
        },
        'weight': {
            'axes': ['freq', 'pol', 'index', 'source'],
            'dtype': np.float64,
            'initialise': True,
            'distributed': True,
            'distributed_axis': 'freq'
        },
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
        }
    }
    @property
    def ra(self):
        return self.datasets['ra']

    @property
    def vis(self):
        return self.datasets['vis']

    @property
    def weight(self):
        return self.datasets['weight']

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
    def param(self):
        return self.index_map['param']

    @property
    def source(self):
        return self.index_map['source']


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


class MapNoise(ContainerBase):
    """Parallel container for holding statistics extracted from a map.
    """

    _axes = ('freq', 'pol', 'region')

    _dataset_spec = {
        'mu': {
            'axes': ['freq', 'pol', 'region'],
            'dtype': np.float64,
            'initialise': True,
            'distributed': True,
            'distributed_axis': 'freq'
        },
        'sigma': {
            'axes': ['freq', 'pol', 'region'],
            'dtype': np.float64,
            'initialise': True,
            'distributed': True,
            'distributed_axis': 'freq'
        },
        'chisq_per_dof': {
            'axes': ['freq', 'pol', 'region'],
            'dtype': np.float64,
            'initialise': True,
            'distributed': True,
            'distributed_axis': 'freq'
        },
        'pte': {
            'axes': ['freq', 'pol', 'region'],
            'dtype': np.float64,
            'initialise': True,
            'distributed': True,
            'distributed_axis': 'freq'
        },
        'shapiro': {
            'axes': ['freq', 'pol', 'region'],
            'dtype': np.float64,
            'initialise': True,
            'distributed': True,
            'distributed_axis': 'freq'
        },
        'anderson': {
            'axes': ['freq', 'pol', 'region'],
            'dtype': np.float64,
            'initialise': True,
            'distributed': True,
            'distributed_axis': 'freq'
        }
    }


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

    # Determine datatype for weights
    if (axes_from is not None) and hasattr(axes_from, 'flags') and ('vis_weight' in axes_from.flags):
        weight_dtype = axes_from.flags['vis_weight'].dtype
    else:
        weight_dtype = np.float32

    # Create empty datasets, and add axis attributes to them
    dset = data.create_dataset('vis', shape=(data.nfreq, data.nprod, data.ntime), dtype=np.complex64,
                               distributed=distributed, distributed_axis=distributed_axis)
    dset.attrs['axis'] = np.array(['freq', 'prod', 'time'])
    dset[:] = 0.0

    dset = data.create_flag('vis_weight', shape=(data.nfreq, data.nprod, data.ntime), dtype=weight_dtype,
                            distributed=distributed, distributed_axis=distributed_axis)
    dset.attrs['axis'] = np.array(['freq', 'prod', 'time'])
    dset[:] = 0.0

    dset = data.create_dataset('gain', shape=(data.nfreq, data.ninput, data.ntime), dtype=np.complex64,
                               distributed=distributed, distributed_axis=distributed_axis)
    dset.attrs['axis'] = np.array(['freq', 'input', 'time'])
    dset[:] = 0.0

    return data


class MonkeyPatchContainers(pipeline.TaskBase):
    """Patch draco to use CHIME timestream containers.

    This task does nothing but perform a monkey patch on `draco.core.containers`
    """

    def __init__(self):

        import ch_pipeline.core.containers as ccontainers
        import draco.core.containers as dcontainers

        # Replace the routine for making an empty timestream. This needs to be replaced
        # in both draco and ch_pipeline because of the ways the imports work
        dcontainers.empty_timestream = ccontainers.make_empty_corrdata
        ccontainers.empty_timestream = ccontainers.make_empty_corrdata

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

            if isinstance(obj, andata.CorrData):
                return dcontainers.empty_timestream(axes_from=obj,
                                                    attrs_from=obj, **kwargs)
            else:
                return _make_empty_like(obj, **kwargs)

        # Replace the empty_like routine
        dcontainers.empty_like = empty_like_patch
