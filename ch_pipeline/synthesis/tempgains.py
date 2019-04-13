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


class TempGains(gain.SiderealGains):
    """This class comprises a new function called _generate_temp which
    is the base class to all temperature dependent gain simulations.

    Attributes
    ----------
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
    corr_length_temp = config.Property(default=1800.0, proptype=float)
    sigma_temp = config.Property(default=1., proptype=float)

    temperror_type = config.enum(['delay', 'random'], default='random')
    # Should be multiples of cadence to make life easier
    max_temp_delay = config.Property(default=900, proptype=int)
    # 1 second cadence, make we are doing ramdom integers
    cadence = 1.
    # Buffer time, ask for extra temperature data for interpolation
    buffer_time = 3600

    _prev_temp = None

    def setup(self, bt, sstream):
        super(TempGains, self).setup(bt, sstream)
        if self.temperror_type == 'delay':
            self.tel = io.get_telescope(bt)
            # Get number of feeds
            ninput = self.tel.nfeed
            self._initialize_tempdelays(ninput)

    def _generate_temperrors(self, time):
        """Generate pertubations in temperature.

        Parameters
        ----------
        time : np.ndarray
            Generate temperature fluctuations for this time period.

        Returns
        -------
        temp_error : np.ndarray (ninput, ntime)
            Temperature fluctuations for number of inputs for requested time.
        temp : np.ndarray (ntime)
            outside temperature interpolated to requested time
        """
        if self.temperror_type == 'delay':
            # Request temperature data according to maximum time delay
            # Here: 15min
            start_time = time[0] - self.max_temp_delay
            end_time = time[-1] + self.max_temp_delay
            # time length file = nsamp * cadence
            t_len = time[-1] - time[0]
            # Query temperatures including some buffer time
            query_start = start_time - self.buffer_time
            query_end = end_time + self.buffer_time
            out_time, out_temp = query_temperatures(query_start, query_end)
            # If out_temp is None, there is no data for this day.
            if out_temp is None:
                print("Using data from previous day")
                # Just use the one from the day before
                return self._prev_temp
            
            # If number of samples for requested time is not expected there is data missing
            samp_per_temp_query = int((query_end - query_start) / 300)
            if abs(samp_per_temp_query - out_time.shape[0]) > 2:
                print("This day probably has missing data - using previous temperature errors")
                return self._prev_temp

            # Create interpolation function with 5min samples
            f_temp = interpolate.interp1d(out_time, out_temp)
            # Put on a different cadence grid - needs to be 1s since we
            # are doing a randint on where to slice the data 
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
            temp_error = delayed_temp - air_temp

            self._prev_temp = temp_error

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

class CoaxErrors(TempGains):
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
    delay_suscept_std = config.Property(default=10**(-12), proptype=float)

    cal_src = config.Property(default='CygA', proptype=str)

    amp = False

    sources = {'CygA' : ephemeris.CygA, 'CasA' : ephemeris.CasA, 'TauA' : ephemeris.TauA}

    _prev_caltemp = None

    def setup(self, manager, sstream):
        super(CoaxErrors, self).setup(manager, sstream)
        # Get telescope object
        tel = io.get_telescope(manager)
        # Get number of feeds
        ninput = tel.nfeed
        # Initialize susceptibilies and errors
        self._initialize_parameters(ninput)

    def _generate_phase(self, time, freq):

        param_error = 0.0
        temp_error = 0.0

        linp, sinp, einp = mpiutil.split_local(self.ninput_global)

        if self.phase_error_param:   
            # Delay error is susceptibility error times temperature drift since
            # last sidereal calibration
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

    def _gain_hook(self, time):
        """Generate temperature errors and query temperatures for this day"""
        # Generate temperature pertubations if we want to simulated temp pertubations.
        if self.phase_error_temp or self.amp_error_param:
            self.temp_pert = self._generate_temperrors(time)

        # This is not really smart because I might need both temp pertubations AND temp fluctuations
        # away from cal temp
        if self.phase_error_param or self.amp_error_param:
            # Generate temperature fluctuations away from calibration temperature if needed.
            # Get the interpolated weather temp
            out_time, out_temp = query_temperatures(time[0] - self.buffer_time, 
                                                    time[-1] + self.buffer_time)
            # Create an interpolation function for temperatures
            f_temp = interpolate.interp1d(out_time, out_temp)
            # Interpolate to time in data
            temp = f_temp(time)

            # Assume that we do sidereal calibration once per day
            # Need to get transit from previous day as well
            trans_time = ephemeris.transit_times(self.sources[self.cal_src], time[0], time[-1])[0]
            trans_idx = np.argmin(abs(time - trans_time))
            caltemp = temp[trans_idx]

            if self._prev_caltemp is None:
                self.delta_temp = temp - caltemp

            else: 
                caltemp_arr = np.ones(temp.shape, dtype=float)
                caltemp_arr[:trans_idx] = self._prev_caltemp
                caltemp_arr[trans_idx:] = caltemp
                self.delta_temp = temp - caltemp_arr

            self._prev_caltemp = caltemp

    def _initialize_parameters(self, ninput):
        """Initialise parameters for coaxial cable model and pertubations"""

        if mpiutil.rank == 0:
            print("Generating random delay susceptibilities and errors")
            # Generate random delay susceptibility errors
            suscept_errors = _draw_random(ninput, self.delay_suscept_std)
            # Add it to the mean to get a susceptiblity for each cable
            susceptibilities = self.delay_suscept_mean + suscept_errors

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
    amp_error_type : enum type
         Choose between susceptibility errors ('params') and temperature
         errors ('temps') or 'both'.
    phase_error_type : enum type
         Choose between susceptibility errors ('params') and temperature
         errors ('temps') or 'both'.    
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

    amp_error_param = config.Property(default=True, proptype=bool)
    amp_error_temp = config.Property(default=True, proptype=bool)

    phase_error_param = config.Property(default=True, proptype=bool)
    phase_error_temp = config.Property(default=True, proptype=bool)

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
        super().setup(manager, sstream)
        fname = self.lna_fname
        tel = io.get_telescope(manager)
        ninput = tel.nfeed
        freq = tel.frequencies
        self._initialize_parameters(fname, ninput, freq)

    def _generate_amp(self, time, freq):
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

    def _generate_phase(self, time, freq):
        nfreq = len(freq)
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

def query_temperatures(start_time, end_time, node_spoof=None):
    """Function to query temperature data,
    could be weather data or lna temperature data

    Parameters
    ----------
    start_time, end_time: float or datetime
        Start and end times of the temperature data. Needs to be either a
        `float` (UNIX time) or a `datetime object in UTC.
    node_spoof : dict, optional
        Optional node spoof argument (default = scinet_online)

    Returns
    -------
    outside_temp: np.array[ntime]
    """

    if node_spoof is None:
        _DEFAULT_NODE_SPOOF = {'cedar_online': '/project/rpp-krs/chime/chime_online/'}
    else:
        _DEFAULT_NODE_SPOOF = node_spoof
    # Query database on rank=0 only
    if mpiutil.rank == 0:
        # Create a finder object limited to the relevant time
        # from datetime import datetime
        f = data_index.Finder(node_spoof=_DEFAULT_NODE_SPOOF)
        f.accept_all_global_flags()
        f.set_time_range(start_time, end_time)
        f.only_weather()

        # Pull out the results and extract files
        f.print_results_summary()
        results = f.get_results()
        # Check if there is weather data for the time requested
        if not results:
            print("No weather data found for this day")
            time = None
            temp_data = None
        else:
            print("Loading weather data")
            weather_data = results[0].as_loaded_data()
            time = weather_data.index_map['time'][:]
            temp_data = weather_data['outTemp']

    else:
        time = None
        temp_data = None

    time = mpiutil.world.bcast(time, root=0)
    temp_data = mpiutil.world.bcast(temp_data, root=0)

    return time, temp_data
