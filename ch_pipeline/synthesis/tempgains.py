# === Start Python 2/3 compatibility
from __future__ import (absolute_import, division,
                        print_function, unicode_literals)
from future.builtins import *  # noqa  pylint: disable=W0401, W0614
from future.builtins.disabled import *  # noqa  pylint: disable=W0401, W0614
# === End Python 2/3 compatibility

import numpy as np
import h5py
from scipy import interpolate
from caput import config, mpiarray, mpiutil, pipeline

from ch_util import data_index, ephemeris

from draco.core import containers, task, io
from draco.synthesis import gain

SOLAR_SEC_PER_DAY = 86400.
SOURCES = {'CygA' : ephemeris.CygA, 'CasA' : ephemeris.CasA, 'TauA' : ephemeris.TauA}


class SiderealTempGains(gain.BaseGains):
    """TThis class comprises a new function called _generate_temperrors which
    is the base class to all temperature dependent gain simulations.

    Attributes
    ----------
    ndays : float
        Number of days that we integrate.
    approx_integration_time : float
        The integration time in solar seconds you are aiming for. Based on this
        information the task will calculate the number of samples per file
        within a sidereal day.
    temperror_type : string, optional
        Temperature error simulation type. Choices are: `random` or `delayed.`
    corr_length_temp : float
        Correlation length for temperature fluctuations in seconds,
        assuming you want to generate gaussian random temp fluctuations.
    sigma_temp : float
        Amplitude of gaussian temperature fluctations. Default: 1K. 
    cadence : int
        Cadence of regridded temperature data when 'delay' temperature
        error type is chosen.
    """
    ndays = config.Property(proptype=float, default=733)
    approx_integration_time = config.Property(proptype=float, default=60.)

    corr_length_temp = config.Property(default=1800.0, proptype=float)
    sigma_temp = config.Property(default=1., proptype=float)

    temperror_type = config.enum(['delay', 'random'], default='random')
    max_temp_delay = config.Property(default=900, proptype=int)
    cal_src = config.Property(default='CygA', proptype=str)

    _prev_caltemp = None

    cadence = 1.
    filt_length = 3600

    def setup(self, bt, sstream):
        """Set up the sidereal gain errors task

        Parameters
        ----------
        bt : BeamTransfer
        sstream : SiderealStream
        """
        self.observer = io.get_telescope(bt)

        sid_sec_per_day = SOLAR_SEC_PER_DAY *  ephemeris.SIDEREAL_S
        self.samples_per_file = int(sid_sec_per_day / self.approx_integration_time)
        self.integration_time = sid_sec_per_day / self.samples_per_file

        # Initialize the current lsd time
        # self._current_lsd = None
        self.counter = 0.
        self.sstream = sstream

        if self.temperror_type == 'delay':
            # Get number of feeds
            ninput = self.observer.nfeed
            self._initialize_tempdelays(ninput)

    def process(self, wstream):
        """Generate a gain timestream for the inputs and times in `data`.

        Parameters
        ----------
        wstream : :class:`ch_util.andata.HKData`
            Generate complex gains for this weather stream.

        Returns
        -------
        gain : :class:`containers.SiderealGainData`
        """

        current_lsd = wstream.attrs['lsd']
        self.log.info("Processing LSD: {}".format(current_lsd))

        # Check if we have reached the end of the requested time
        if self.counter >= self.ndays:
            return None

        # Convert the current lsd day to unix time
        unix_start = self.observer.lsd_to_unix(current_lsd)
        unix_end = self.observer.lsd_to_unix(current_lsd + 1)

        # Distribute the sidereal data and create a time array
        data = self.sstream
        # wdata = self.wstream
        data.redistribute('freq')
        self.freq = data.index_map['freq']['centre'][:]
        ra = np.linspace(0.0, 360.0, self.samples_per_file, endpoint=False)
        time = np.linspace(unix_start, unix_end, self.samples_per_file, endpoint=False)

        # Make a sidereal gain data container
        gain_data = containers.SiderealGainData(axes_from=data, ra=ra)
        gain_data.redistribute('input')

        self.ninput_local = gain_data.gain.local_shape[1]
        self.ninput_global = gain_data.gain.global_shape[1]

        self._gain_hook(time, wstream)

        gain_amp = 1.0
        gain_phase = 0.0

        if self.amp:
            gain_amp = self._generate_amp(time, wstream)

        if self.phase:
            gain_phase = self._generate_phase(time, wstream)

        # Combine into an overall gain fluctuation
        gain_comb = gain_amp * np.exp(1.0J * gain_phase)

        # Copy the gain entries into the output container
        gain = mpiarray.MPIArray.wrap(gain_comb, axis=1)
        gain_data.gain[:] = gain
        gain_data.attrs['lsd'] = current_lsd
        gain_data.attrs['tag'] = 'lsd_%i' % current_lsd
        gain_data.attrs['int_time'] = self.integration_time

        self.counter += 1

        # Keep a reference to time around for the next round
        self._prev_time = time

        return gain_data

    def _gain_hook(self, time, wstream):
        """This hook function is intended to be able to generate stuff for
        both amplitude and phase. Can be implemented in child classes but must not.

        Parameters:
        -----------
        time : np.ndarray
            Generate whatever you need for both amplitude and phase for time
            period given.
        """
        pass

    def _generate_temperrors(self, time, wstream):
        """Generate pertubations in temperature.

        Parameters
        ----------
        time : np.ndarray
            Generate temperature fluctuations for this time period.
        wstream : class
            HKData object.

        Returns
        -------
        temp_error : np.ndarray (ninput, ntime)
            Temperature fluctuations for number of inputs for requested time.
        temp : np.ndarray (ntime)
            outside temperature interpolated to requested time
        """
        if self.temperror_type == 'delay':
            wtime = wstream.time
            outtemp = wstream.datasets['outTemp'][:]

            # Create start and end times for delayed temperatures
            start_time = time[0] - self.max_temp_delay
            end_time = time[-1] + self.max_temp_delay
            t_len = time[-1] - time[0]

            # Create interpolation function and put on 1s grid
            f_temp = interpolate.interp1d(wtime, outtemp)
            regrid_time = np.arange(start_time, end_time, self.cadence)
            regrid_temp = f_temp(regrid_time)

            linp, sinp, einp = mpiutil.split_local(self.ninput_global)
            n_index = int(np.ceil(t_len / self.cadence)) + 1
            slice_list = [slice(i, i + n_index) for i in self._ts[sinp:einp]]

            # Create an array to hold the delayed temperatures
            delayed_temp = np.zeros((self.ninput_local, len(time)), dtype=float)

            # Slice the regridded time
            slice_time = regrid_time[self.max_temp_delay:self.max_temp_delay + n_index]
            # Now slice the regrided temperature for each input
            # and finally interpolate onto file time axis.
            for i in range(self.ninput_local):
                slice_temp = regrid_temp[slice_list[i]]
                f_temp_regrid = interpolate.interp1d(slice_time, slice_temp)
                delayed_temp[i] = f_temp_regrid(time)

            air_temp = f_temp(time)
            errors = delayed_temp - air_temp

            temp_error = self._apply_cal_to_temperrors(errors, time)

        # Otherwise just simulate gaussian random temperature errors
        else:
            # Correlation function for temperature fluctuations
            cf_temp = self._corr_func(self.corr_length_temp, self.sigma_temp)
            # Create temperature fluctuations for each thermal component
            n_realisations = self.ninput_local
            temp_error = gain.generate_fluctuations(time, cf_temp, n_realisations, self._prev_time, self._prev_temp)

            self._prev_temp = temp_error
            self._prev_time = time

        return temp_error

    def _apply_cal_to_temperrors(self, temp_error, time):
        """Apply a filter at calibration source transit to temperature errors
        """
        trans_time = ephemeris.transit_times(SOURCES[self.cal_src], time[0], time[-1])[0]
        trans_idx = np.argmin(abs(time - trans_time))

        filt = np.ones(temp_error.shape[-1])
        n = int(self.filt_length / self.approx_integration_time)
        exp_func = _exponential(n)
        filt[trans_idx:trans_idx + n] = exp_func

        temp_error = temp_error * filt[np.newaxis]

        return temp_error

    def _temp_drift(self, time, wstream):
        """Temperature drifts away from calibration temperature
         Parameters
        ----------
        time : np.ndarray
            Generate temperature fluctuations for this time period.
        wstream : class
            HKData object."""
        wtime = wstream.time
        outtemp = wstream.datasets['outTemp'][:]

        # Create an interpolation function for temperatures
        f_temp = interpolate.interp1d(wtime, outtemp)
        temp = f_temp(time)

        # Assume that we do sidereal calibration once per day
        trans_time = ephemeris.transit_times(SOURCES[self.cal_src], time[0], time[-1])[0]
        trans_idx = np.argmin(abs(time - trans_time))
        caltemp = temp[trans_idx]

        if self._prev_caltemp is None:
            delta_temp = temp - caltemp

        else:
            caltemp_arr = np.ones(temp.shape, dtype=float)
            caltemp_arr[:trans_idx] = self._prev_caltemp
            caltemp_arr[trans_idx:] = caltemp
            delta_temp = temp - caltemp_arr
            self.log.info("caltemp %f", caltemp)
            self.log.info("prev caltemp %f", self._prev_caltemp)

        self._prev_caltemp = caltemp

        return delta_temp

    def _initialize_tempdelays(self, ninput):
        """Initialise temperature delays for input channels if
        temperature error 'delay' is chosen"""

        if mpiutil.rank == 0:
            # Choose random start times for positively and negatively
            # delayed temperature curves
            ts = np.random.randint(0, 2 * self.max_temp_delay, size=ninput)

        else:
            ts = None

        # Broadcast slices to all ranks
        self._ts = mpiutil.world.bcast(ts, root=0)


class CoaxErrors(SiderealTempGains):
    """Task for simulating gain errors in coaxial cables.
    
    Attributes
    ----------
    phase_error_param : bool
        Wheter to include errors in the thermal model best 
        fit parameter or not. Default: True.
    phase_error_temp : bool
        Wheter to include errors in the thermal model temperatures 
        or not. Default: True.
    """
    phase_error_param = config.Property(default=True, proptype=bool)
    phase_error_temp = config.Property(default=True, proptype=bool)

    # For when we want to implement ampltiude errors... just place holders right now
    amp_error_param = config.Property(default=False, proptype=bool)
    amp_error_temp = config.Property(default=False, proptype=bool)

    delay_suscept_mean = config.Property(default=4.5*10**(-12), proptype=float)
    delay_suscept_std = config.Property(default=10**(-13), proptype=float)

    amp = False

    def setup(self, manager, sstream):
        self.log.info("In coax setup")
        super(CoaxErrors, self).setup(manager, sstream)
        tel = io.get_telescope(manager)
        ninput = tel.nfeed
        # Initialize parameters.
        self._initialize_parameters(ninput)

    def _generate_phase(self, time, wstream):

        param_error = 0.0
        temp_error = 0.0

        linp, sinp, einp = mpiutil.split_local(self.ninput_global)
        freq = self.freq

        if self.phase_error_param:   
            # Delay error is susceptibility error times temperature drift since
            # last sidereal calibration
            self.log.info("param errors")
            param_error = (self._suscept_errors[sinp:einp, np.newaxis] * 
                           self.delta_temp[np.newaxis, :])

        if self.phase_error_temp:
            # temp_pert = self._generate_temperrors(time)
            temp_error = (self._susceptibilities[sinp:einp, np.newaxis] *
                          self.temp_pert)
            
        delay_error = param_error + temp_error
        
        # Calculate phase error
        gain_phase = (2.0 * np.pi * freq[:, np.newaxis, np.newaxis] * 1e6 *
                       delay_error[np.newaxis, :])

        return gain_phase

    def _gain_hook(self, time, wstream):
        """Generate temperature errors and/or drift temperature for this day"""

        # Generate temperature pertubations if we want to simulated temp pertubations.
        if self.phase_error_temp or self.amp_error_param:
            self.temp_pert = self._generate_temperrors(time, wstream)

        # Calculate temperature drift away from calibration temperature.
        if self.phase_error_param or self.amp_error_param:
            self.delta_temp = self._temp_drift(time, wstream)

    def _initialize_parameters(self, ninput):
        """Initialise parameters for coaxial cable model and pertubations"""

        if mpiutil.rank == 0:
            self.log.info("Generating random delay susceptibilities and errors")
            # Generate random delay susceptibility errors
            suscept_errors = _draw_random(ninput, self.delay_suscept_std)
            # Add it to the mean to get a susceptiblity for each cable
            # susceptibilities = self.delay_suscept_mean + suscept_errors
            susceptibilities = np.load('/project/rpp-krs/cahofer/ch_pipeline/ch_pipeline/synthesis/suscept.npy')

        else:
            suscept_errors = None
            susceptibilities = None

        # Broadcast input description to all ranks
        self._suscept_errors = mpiutil.world.bcast(suscept_errors, root=0)
        self._susceptibilities = mpiutil.world.bcast(susceptibilities, root=0)


class AmplifierErrors(CoaxErrors):
    """Task for simulating gain errors (phase and amplitude) 
    in amplifiers
    
    Attributes
    ----------
    lna_fname : string
        The path to the best fit parameters of amplifiers.
    std_lin_param : float
        Standard deviation in the linear parameter (from data, measuring 
        100 amplifiers - see doclib xxx)
    std_quad_param : float
        Standard deviation in the quadratic parameter (from data, measuring 
        100 amplifiers - see doclib xxx)
    std_phase : float
        Standard deviation in the phase (from data, measuring 
        100 amplifiers - see doclib xxx)
    """

    lna_fname = config.Property(
        proptype=str,
        default='/project/rpp-krs/cahofer/ch_pipeline/venv/src/ch-util/ch_util/thermal_prms/lna_thermal.hdf5')

    std_lin_param = config.Property(default=4*10**(-4), proptype=float)
    std_quad_param = config.Property(default=4.4*10**(-6), proptype=float)
    std_phase = config.Property(default=3*10**(-3), proptype=float)

    def setup(self, manager, sstream):
        """Set up the gain errors task

        Parameters
        ----------
        bt : ProductManager or BeamTransfer
            BeamTransfer manager.
        sstream : SiderealStream
        """
        # This should call setup from TempGains NOT CoaxErrors check this.
        super(CoaxErrors, super(AmplifierErrors, self)).setup(manager, sstream)

        fname = self.lna_fname
        tel = io.get_telescope(manager)
        ninput = tel.nfeed
        freq = tel.frequencies
        self._initialize_parameters(fname, ninput, freq)

    def _generate_amp(self, time):
        # Create parameter arrays for each input as a function of frequency
        lin_params = self._lin_param[:, np.newaxis] + self._lin_param_pert[np.newaxis]
        quad_params = self._quad_param[:, np.newaxis] + self._quad_param_pert[np.newaxis]

        linp, sinp, einp = mpiutil.split_local(self.ninput_global)

        amp_fit_params_errors = 0.0
        amp_temp_errors = 0.0

        if self.amp_error_param:
            # Calculate the error in amplitude due to fit parameter scatter
            amp_fit_params_errors = (self._lin_param_pert[np.newaxis, sinp:einp, np.newaxis]
                                     * self.delta_temp[np.newaxis, np.newaxis, :]
                                     + self._quad_param_pert[np.newaxis, sinp:einp, np.newaxis]
                                     * self.delta_temp[np.newaxis, np.newaxis, :]**2)

        if self.amp_error_temp:
            # Calculate errors in ampltidue due to tempearture errors
            amp_temp_errors = (lin_params[:, sinp:einp, np.newaxis]
                               + 2 * quad_params[:, sinp:einp, np.newaxis]
                               * self.delta_temp[np.newaxis, np.newaxis, :]) * self.temp_pert[np.newaxis]

        gain_amp = 10.**(0.05 * (amp_fit_params_errors + amp_temp_errors))

        return gain_amp

    def _generate_phase(self, time):

        nfreq = len(self.freq)
        ninput = self.ninput_local
        ntime = len(time)

        linp, sinp, einp = mpiutil.split_local(self.ninput_global)

        phase_device_scatter_error = 0.0
        phase_temp_error = 0.0
        
        if self.phase_error_param:
            # Phase pertubations is RMS spread across amplifiers at maximum slope
            phase_device_scatter_error = (self._phase_pert[np.newaxis, sinp:einp, np.newaxis]
                                          * self.delta_temp[np.newaxis, np.newaxis, :])

        if self.phase_error_temp:
            # Interpolate the phases from file as a function of temperature and frequency
            # Create empty array to populate phase errors due to temperature errors
            phase_temp_error = np.zeros((nfreq, ninput, ntime), dtype=float)
            for f in range(nfreq):
                # Create an interpolation function at every frequency
                f_phase = interpolate.interp1d(self._temp_measured, self._phase_measured[f, :])
                for i in range(ninput):
                    # Difference phase model at perturbed temperature from
                    # model at non-pertubed nominal temperature
                    phase_temp_error[f, i, :] = f_phase(tempi[i, :]) - f_phase(temp)

        gain_phase = phase_device_scatter_error + phase_temp_error

        return gain_phase

    def _initialize_parameters(self, fname, ninput, tstream_freq):
        """Initialise parameters for amplifier model and pertubations"""

        if mpiutil.rank == 0:
            try:
                print("Attempting to read lna file from disk...")
                with h5py.File(fname, 'r') as f:
                    lin_param = f['amp']['prm_lin']  # Linear amplitude parameter
                    quad_param = f['amp']['prm_quad']  # Quadratic amplitude parameter
                    freq_measured = f['phase'].attrs['freq']  # Measured frequencies
                    temp_measured = f['phase'].attrs['temp']  # Measured temperatures
                    phase_measured = f['phase'][...]

            except IOError:
                raise IOError("Could not load lna file from disk [path: %s]."
                              % fname)

            # Interpolate parameters to the frequencies measured
            f_lin_param = interpolate.interp1d(freq_measured, lin_param)
            f_quad_param = interpolate.interp1d(freq_measured, quad_param)
            lin_param = f_lin_param(tstream_freq)
            quad_param = f_quad_param(tstream_freq)

            # Assume for now that scatter in parameters is frequency independent
            print("Generating random fluctuations in linear and quadratic parameters")
            lin_param_pert = _draw_random(ninput, self.std_lin_param)
            quad_param_pert = _draw_random(ninput, self.std_quad_param)

            # Generate pertubations in phase
            phase_pert = _draw_random(ninput, self.std_phase)

        else:
            lin_param = None
            quad_param = None
            freq_measured = None
            temp_measured = None
            lin_param_pert = None
            quad_param_pert = None
            phase_measured = None
            phase_pert = None

        # Broadcast input description to all ranks
        self._lin_param = mpiutil.world.bcast(lin_param, root=0)
        self._quad_param = mpiutil.world.bcast(quad_param, root=0)
        self._freq_measured = mpiutil.world.bcast(freq_measured, root=0)
        self._temp_measured = mpiutil.world.bcast(temp_measured, root=0)
        self._lin_param_pert = mpiutil.world.bcast(lin_param_pert, root=0)
        self._quad_param_pert = mpiutil.world.bcast(quad_param_pert, root=0)
        self._phase_measured = mpiutil.world.bcast(phase_measured, root=0)
        self._phase_pert = mpiutil.world.bcast(phase_pert, root=0)


def _exponential(n):
    """This is an exponential function of size n, such that the first y-value
    is 0 and the last is 1."""
    y = np.exp(np.linspace(0, 1, n) **2) - 1
    y_norm = y / y[-1]

    return y_norm

def _draw_random(n, sigma):
    """Generate Gaussian numbers.

    Parameters
    ----------
    n : integer:
        Number of realisations ot generate.
    sigma : float
        Standard deviation of distribution.

    Returns
    -------
    y : np.nndarray[n]
        Realisations of gaussian field.
    """
    r = np.random.standard_normal(n)
    return r * sigma
