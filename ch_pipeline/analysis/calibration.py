"""
================================================================
Tasks for Calibration (:mod:`~ch_pipeline.analysis.calibration`)
================================================================

.. currentmodule:: ch_pipeline.analysis.calibration

Tasks for calibrating the data.

Tasks
=====

.. autosummary::
    :toctree: generated/

    NoiseSourceCalibration
    NoiseSourceFold
    GatedNoiseCalibration
    SiderealCalibration
"""

import numpy as np
from scipy import interpolate

from caput import config, pipeline
from caput import mpiarray, mpiutil

from ch_util import tools
from ch_util import ephemeris
from ch_util import ni_utils
from ch_util import cal_utils
from ch_util import fluxcat

from draco.core import task
from draco.util import _fast_tools

from ..core import containers


def _extract_diagonal(utmat, axis=1):
    """Extract the diagonal elements of an upper triangular array.

    Parameters
    ----------
    utmat : np.ndarray[..., nprod, ...]
        Upper triangular array.
    axis : int, optional
        Axis of array that is upper triangular.

    Returns
    -------
    diag : np.ndarray[..., ninput, ...]
        Diagonal of the array.
    """

    # Estimate nside from the array shape
    nside = int((2 * utmat.shape[axis])**0.5)

    # Check that this nside is correct
    if utmat.shape[axis] != (nside * (nside + 1) / 2):
        msg = ('Array length (%i) of axis %i does not correspond upper triangle\
                of square matrix' % (utmat.shape[axis], axis))
        raise RuntimeError(msg)

    # Find indices of the diagonal
    diag_ind = [tools.cmap(ii, ii, nside) for ii in range(nside)]

    # Construct slice objects representing the axes before and after the product axis
    slice0 = (np.s_[:],) * axis
    slice1 = (np.s_[:],) * (len(utmat.shape) - axis - 1)

    # Extract wanted elements with a giant slice
    sl = slice0 + (diag_ind,) + slice1
    diag_array = utmat[sl]

    return diag_array


def solve_gain(data, feeds=None, norm=None):
    """
    Steps through each time/freq pixel, generates a Hermitian matrix and
    calculates gains from its largest eigenvector.

    Parameters
    ----------
    data : np.ndarray[nfreq, nprod, ntime]
        Visibility array to be decomposed
    feeds : list
        Which feeds to include. If :obj:`None` include all feeds.
    norm : np.ndarray[nfreq, nfeed, ntime], optional
        Array to use for weighting.

    Returns
    -------
    evalue : np.ndarray[nfreq, nfeed, ntime]
        Eigenvalues obtained from eigenvalue decomposition
        of the visibility matrix.
    gain : np.ndarray[nfreq, nfeed, ntime]
        Gain solution for each feed, time, and frequency
    gain_error : np.ndarray[nfreq, nfeed, ntime]
        Error on the gain solution for each feed, time, and frequency
    """

    # Turn into numpy array to avoid any unfortunate indexing issues
    data = data[:].view(np.ndarray)

    # Calcuate the number of feeds in the data matrix
    tfeed = int((2 * data.shape[1])**0.5)

    # If not set, create the list of included feeds (i.e. all feeds)
    feeds = np.array(feeds) if feeds is not None else np.arange(tfeed)
    nfeed = len(feeds)

    # Create empty arrays to store the outputs
    gain = np.zeros((data.shape[0], nfeed, data.shape[-1]), np.complex64)
    gain_error = np.zeros((data.shape[0], nfeed, data.shape[-1]), np.float32)
    evalue = np.zeros((data.shape[0], nfeed, data.shape[-1]), np.float32)

    # Set up normalisation matrix
    if norm is None:
        norm = (_extract_diagonal(data, axis=1).real)**0.5
        norm = tools.invert_no_zero(norm)
        norm = norm[:, feeds]

    elif norm.shape != gain.shape:
        ValueError("Input normalization matrix has shape %s, should have shape %s." %
                   (norm.shape, gain.shape))

    # Pre-generate the array of inverted norms
    inv_norm = tools.invert_no_zero(norm)

    # Initialise a temporary array for unpacked products
    cd = np.zeros((nfeed, nfeed), dtype=data.dtype)

    # Iterate over frequency/time and solve gains
    for fi in range(data.shape[0]):
        for ti in range(data.shape[-1]):

            # Skip if all zeros
            if not np.any(data[fi, :, ti]):
                continue

            # Unpack visibility and normalisation array into square matrix
            _fast_tools._unpack_product_array_fast(data[fi, :, ti].copy(), cd, feeds, tfeed)

            # Apply weighting
            w = norm[fi, :, ti]
            cd *= np.outer(w, w.conj())

            # Skip if any non-finite values
            if not np.isfinite(cd).all():
                continue

            # Solve for eigenvectors and eigenvalues
            evals, evecs = tools.eigh_no_diagonal(cd, niter=5)

            # Construct gain solutions
            if evals[-1] > 0:
                sign0 = (1.0 - 2.0 * (evecs[0, -1].real < 0.0))
                gain[fi, :, ti] = sign0 * inv_norm[fi, :, ti] * evecs[:, -1] * evals[-1]**0.5

                gain_error[fi, :, ti] = (inv_norm[fi, :, ti] *
                                         1.4826 * np.median(np.abs(evals[:-1] - np.median(evals[:-1]))) /
                                         evals[-1]**0.5)

                evalue[fi, :, ti] = evals

            # Solve for eigenvectors
            # evals, evecs = tools.eigh_no_diagonal(cd, niter=5, eigvals=(nfeed - 2, nfeed - 1))

            # Construct dynamic range and gain, but only if the two highest
            # eigenvalues are positive. If not, we just let the gain and dynamic
            # range stay as zero.
            # if evals[-1] > 0 and evals[-2] > 0:
            #     dr[fi, ti] = evals[-1] / evals[-2]
            #     gain[fi, :, ti] = inv_norm[fi, :, ti] * evecs[:, -1] * evals[-1]**0.5

    return evalue, gain, gain_error


def interp_gains(trans_times, gain_mat, times, axis=-1):
    """ Linearly interpolates gain solutions in sidereal day.

    Parameter
    ---------
    trans_times : array_like
        Unix time of object transit
    gain_mat : array_like
        Array of gains shaped (freq, ncorr, ndays)
    times : array_like
        Timestamps onto which gain solution is interpolated
    axis : int
        Axis along which to interpolate.

    Returns
    -------
    Array of interpolated gains
    """
    f = interpolate.interp1d(trans_times, gain_mat, kind='linear', axis=axis, bounds_error=False)

    gains = f(times)
    gains[..., times < trans_times[0]] = gain_mat[..., 0, np.newaxis]
    gains[..., times > trans_times[-1]] = gain_mat[..., -1, np.newaxis]

    return gains


def _cdiff(ts, dt):
    # Subtract the average of two nearby points from every point in the timestream
    if dt is None:
        return ts

    return ts - 0.5 * (np.roll(ts, dt, axis=-1) + np.roll(ts, -dt, axis=-1))


def _adiff(ts, dt):
    # Subtract the average of the first dt points and last dt points from every point in the timestream
    if dt is None:
        return ts

    return ts - 0.5 * (np.mean(ts[..., :dt], axis=-1) +
                       np.mean(ts[..., -dt:], axis=-1))[..., np.newaxis]


class NoiseSourceFold(task.SingleTask):
    """Fold the noise source for synced data.

    Attributes
    ----------
    period : int, optional
        Period of the noise source in integration samples.
    phase : list, optional
        Phase of noise source on sample.
    """

    period = config.Property(proptype=int, default=None)
    phase = config.Property(proptype=list, default=[])
    only_off = config.Property(proptype=bool, default=False)

    def process(self, ts):
        """Fold on the noise source and generate a gated dataset.

        Parameters
        ----------
        ts : andata.CorrData object
            Timestream to fold on.

        Returns
        -------
        folded_ts : andata.CorrData
            Timestream with a gated_vis0 dataset containing the noise
            source data.
        """

        if (self.period is None) or (not self.phase):
            ni_params = None
        else:
            ni_params = {'ni_period': self.period, 'ni_on_bins': self.phase}

        folded_ts = ni_utils.process_synced_data(ts, ni_params=ni_params,
                                                 only_off=self.only_off)

        return folded_ts


class NoiseInjectionCalibration(pipeline.TaskBase):
    """Calibration using Noise Injection

    Attributes
    ----------
    nchannels : int, optional
        Number of channels (default 16).
    ch_ref : int in the range 0 <= ch_ref <= Nchannels-1, optional
        Reference channel (default 0).
    fbin_ref : int, optional
        Reference frequency bin
    decimate_only : bool, optional
        If set (not default), then we do not apply the gain solution
        and return a decimated but uncalibrated timestream.

    .. deprecated:: pass1G
        This calibration technique only works on old data from before Pass 1G.
        For more recent data, look at :class:`GatedNoiseCalibration`.
    """

    nchannels = config.Property(proptype=int, default=16)
    ch_ref = config.Property(proptype=int, default=None)
    fbin_ref = config.Property(proptype=int, default=None)

    decimate_only = config.Property(proptype=bool, default=False)

    def setup(self, inputmap):
        """Use the input map to set up the calibrator.

        Parameters
        ----------
        inputmap : list of :class:`tools.CorrInputs`
            Describing the inputs to the correlator.
        """
        self.ch_ref = tools.get_noise_channel(inputmap)
        if mpiutil.rank0:
            print "Using input=%i as noise channel" % self.ch_ref

    def next(self, ts):
        """Find gains from noise injection data and apply them to visibilities.

        Parameters
        ----------
        ts : containers.TimeStream
            Parallel timestream class containing noise injection data.

        Returns
        -------
        cts : containers.CalibratedTimeStream
            Timestream with calibrated (decimated) visibilities, gains and
            respective timestamps.
        """
        # This method should derive the gains from the data as it comes in,
        # and apply the corrections to rigidise the data
        #
        # The data will come be received as a containers.TimeStream type. In
        # some ways this looks a little like AnData, but it works in parallel

        # Ensure that we are distributed over frequency

        ts.redistribute('freq')

        # Create noise injection data object from input timestream
        nidata = ni_utils.ni_data(ts, self.nchannels, self.ch_ref, self.fbin_ref)

        # Decimated visibilities without calibration
        vis_uncal = nidata.vis_off_dec

        # Timestamp corresponding to decimated visibilities
        timestamp = nidata.timestamp_dec

        # Find gains
        nidata.get_ni_gains()
        gain = nidata.ni_gains

        # Correct decimated visibilities
        if self.decimate_only:
            vis = vis_uncal.copy()
        else:  # Apply the gain solution
            gain_inv = tools.invert_no_zero(gain)
            vis = tools.apply_gain(vis_uncal, gain_inv)

        # Calculate dynamic range
        ev = ni_utils.sort_evalues_mag(nidata.ni_evals)  # Sort evalues
        dr = abs(ev[:, -1, :] / ev[:, -2, :])
        dr = dr[:, np.newaxis, :]

        # Turn vis, gains and dr into MPIArray
        vis = mpiarray.MPIArray.wrap(vis, axis=0, comm=ts.comm)
        gain = mpiarray.MPIArray.wrap(gain, axis=0, comm=ts.comm)
        dr = mpiarray.MPIArray.wrap(dr, axis=0, comm=ts.comm)

        # Create NoiseInjTimeStream
        cts = containers.TimeStream(timestamp, ts.freq, vis.global_shape[1],
                                    comm=vis.comm, copy_attrs=ts, gain=True)

        cts.vis[:] = vis
        cts.gain[:] = gain
        cts.gain_dr[:] = dr
        cts.common['input'] = ts.input

        cts.redistribute(0)

        return cts


class GatedNoiseCalibration(task.SingleTask):
    """Calibration using Noise Injection

    Attributes
    ----------
    norm : ['gated', 'off', 'identity']
        Specify what to use to normalise the matrix.
    """

    norm = config.Property(proptype=str, default='off')

    def process(self, ts, inputmap):
        """Find gains from noise injection data and apply them to visibilities.

        Parameters
        ----------
        ts : andata.CorrData
            Parallel timestream class containing noise injection data.
        inputmap : list of CorrInputs
            List describing the inputs to the correlator.

        Returns
        -------
        ts : andata.CorrData
            Timestream with calibrated (decimated) visibilities, gains and
            respective timestamps.
        """

        # Ensure that we are distributed over frequency
        ts.redistribute('freq')

        # Figure out which input channel is the noise source (used as gain reference)
        noise_channel = tools.get_noise_channel(inputmap)

        # Get the norm matrix
        if self.norm == 'gated':
            norm_array = _extract_diagonal(ts.datasets['gated_vis0'][:])**0.5
            norm_array = tools.invert_no_zero(norm_array)
        elif self.norm == 'off':
            norm_array = _extract_diagonal(ts.vis[:])**0.5
            norm_array = tools.invert_no_zero(norm_array)

            # Extract the points with zero weight (these will get zero norm)
            w = (_extract_diagonal(ts.weight[:]) > 0)
            w[:, noise_channel] = True  # Make sure we keep the noise channel though!

            norm_array *= w

        elif self.norm == 'none':
            norm_array = np.ones([ts.vis[:].shape[0], ts.ninput, ts.ntime], dtype=np.uint8)
        else:
            raise RuntimeError('Value of norm not recognised.')

        # Take a view now to avoid some MPI issues
        gate_view = ts.datasets['gated_vis0'][:].view(np.ndarray)
        norm_view = norm_array[:].view(np.ndarray)

        # Find gains with the eigenvalue method
        evalue, gain = solve_gain(gate_view, norm=norm_view)[0:2]
        dr = evalue[:, -1, :] * tools.invert_no_zero(evalue[:, -2, :])

        # Normalise by the noise source channel
        gain *= tools.invert_no_zero(gain[:, np.newaxis, noise_channel, :])
        gain = np.nan_to_num(gain)

        # Create container from gains
        gain_data = containers.GainData(axes_from=ts)
        gain_data.add_dataset('weight')

        # Copy data into container
        gain_data.gain[:] = gain
        gain_data.weight[:] = dr

        return gain_data


def contiguous_flag(flag, centre=None):

    nelem = flag.shape[-1]
    shp = flag.shape[:-1]

    if centre is None:
        centre = nelem / 2

    for index in np.ndindex(*shp):

        for ii in range(centre, nelem, 1):
            if not flag[index][ii]:
                flag[index][ii:] = False
                continue

        for ii in range(centre, -1, -1):
            if not flag[index][ii]:
                flag[index][:ii] = False
                continue

    return flag


class SiderealCalibration(task.SingleTask):
    """Use point source as a calibrator for a sidereal stack.

    Attributes
    ----------
    source : str
        Name of the point source to use as calibrator.
        Default CygA.
    model_fit : bool
        Fit a model to the point source transit.
        Default False.
    use_peak : bool
        Relevant if model_fit is True.  If set to True,
        estimate the gain as the response at the
        actual peak location. If set to False, estimate
        the gain as the response at the expected peak location.
        Default False.
    threshold : float
        Relevant if model_fit is True.  The model is only fit to
        time samples with dynamic range greater than threshold.
        Default is 3.

    """

    source = config.Property(proptype=str, default='CygA')
    model_fit = config.Property(proptype=bool, default=False)
    use_peak = config.Property(proptype=bool, default=False)
    threshold = config.Property(proptype=float, default=3.0)

    def process(self, sstream, inputmap, inputmask):
        """Determine calibration from a sidereal stream.

        Parameters
        ----------
        sstream : containers.SiderealStream
            Rigidized sidereal timestream to calibrate.
        inputmap : list of :class:`CorrInput`s
            A list of describing the inputs as they are in the file.
        inputmask : containers.CorrInputMask
            Mask indicating which correlator inputs to use in the
            eigenvalue decomposition.

        Returns
        -------
        gains : containers.PointSourceTransit or containers.StaticGainData
            Response of each feed to the point source and best-fit model
            (model_fit is True), or gains at the expected peak location
            (model_fit is False).
        """

        # Ensure that we are distributed over frequency
        sstream.redistribute('freq')

        # Find the local frequencies
        nfreq = sstream.vis.local_shape[0]
        sfreq = sstream.vis.local_offset[0]
        efreq = sfreq + nfreq

        # Get the local frequency axis
        freq = sstream.freq['centre'][sfreq:efreq]

        # Fetch source
        source = ephemeris.source_dictionary[self.source]

        # Estimate the RA at which the transiting source peaks
        peak_ra = ephemeris.peak_RA(source, deg=True)

        # Find closest array index
        idx = np.argmin(np.abs(sstream.ra - peak_ra))

        # Fetch the transit into this visibility array
        # Cut out a snippet of the timestream
        slice_width_deg = 3.0 * cal_utils.guess_fwhm(400.0, pol='X', dec=source._dec, sigma=True)
        slice_width = int(slice_width_deg / np.median(np.abs(np.diff(sstream.ra))))
        slice_centre = slice_width
        st, et = idx - slice_width, idx + slice_width + 1

        vis_slice = sstream.vis[..., st:et].copy()
        ra_slice = sstream.ra[st:et]

        nra = vis_slice.shape[-1]

        # Determine good inputs
        nfeed = len(inputmap)
        good_input = np.arange(nfeed, dtype=np.int)[inputmask.datasets['input_mask'][:]]

        # Use input map to figure out which are the X and Y feeds
        xfeeds = np.array([idx for idx, inp in enumerate(inputmap) if (idx in good_input) and tools.is_chime_x(inp)])
        yfeeds = np.array([idx for idx, inp in enumerate(inputmap) if (idx in good_input) and tools.is_chime_y(inp)])

        if mpiutil.rank0:
            print("Performing sidereal calibration with %d/%d good feeds (%d xpol, %d ypol)." %
                  (len(good_input), nfeed, len(xfeeds), len(yfeeds)))

        # Extract the diagonal (to be used for weighting)
        # prior to differencing on-source and off-source
        norm = np.sqrt(_extract_diagonal(vis_slice, axis=1).real)
        norm = tools.invert_no_zero(norm)

        # Subtract the average visibility at the start and end of the slice (off source)
        diff = int(slice_width / 3)
        vis_slice = _adiff(vis_slice, diff)

        # Fringestop the data
        vis_slice = tools.fringestop_pathfinder(vis_slice, ra_slice, freq, inputmap, source)

        # Create arrays to hold point source response
        resp = np.zeros([nfreq, nfeed, nra], np.complex128)
        resp_err = np.zeros([nfreq, nfeed, nra], np.float64)

        # Solve for the point source response of each set of polarisations
        evalue_x, resp[:, xfeeds, :], resp_err[:, xfeeds, :] = solve_gain(vis_slice, feeds=xfeeds, norm=norm[:, xfeeds])
        evalue_y, resp[:, yfeeds, :], resp_err[:, yfeeds, :] = solve_gain(vis_slice, feeds=yfeeds, norm=norm[:, yfeeds])

        # Extract flux density of the source
        rt_flux_density = np.sqrt(fluxcat.FluxCatalog[self.source].predict_flux(freq))

        # Divide by the flux density of the point source
        # to convert the response and response_error into
        # units of 'sqrt(correlator units / Jy)'
        resp /= rt_flux_density[:, np.newaxis, np.newaxis]
        resp_err /= rt_flux_density[:, np.newaxis, np.newaxis]

        # Define units
        unit_in = sstream.vis.attrs.get('units', 'rt-correlator-units')
        unit_out = 'rt-Jy'

        # Construct the final gain array from the point source response at transit
        gain = resp[:, :, slice_centre]

        # Construct the dynamic range estimate as the ratio of the first to second
        # largest eigenvalue at the time of transit
        dr_x = evalue_x[:, -1, :] * tools.invert_no_zero(evalue_x[:, -2, :])
        dr_y = evalue_y[:, -1, :] * tools.invert_no_zero(evalue_y[:, -2, :])

        # If requested, fit a model to the point source transit
        if self.model_fit:

            # Only fit ra values above the specified dynamic range threshold
            # that are contiguous about the expected peak position.
            fit_flag = np.zeros([nfreq, nfeed, nra], dtype=np.bool)
            fit_flag[:, xfeeds, :] = contiguous_flag(dr_x > self.threshold, centre=slice_centre)[:, np.newaxis, :]
            fit_flag[:, yfeeds, :] = contiguous_flag(dr_y > self.threshold, centre=slice_centre)[:, np.newaxis, :]

            # Fit model for the complex response of each feed to the point source
            param, param_cov = cal_utils.fit_point_source_transit(ra_slice, resp, resp_err, flag=fit_flag)

            # Overwrite the initial gain estimates for frequencies/feeds
            # where the model fit was successful
            if self.use_peak:
                gain = np.where(np.isnan(param[:, :, 0]), gain,
                                param[:, :, 0] * np.exp(1.0j * np.deg2rad(param[:, :, -2])))
            else:
                for index in np.ndindex(nfreq, nfeed):
                    if np.all(np.isfinite(param[index])):
                        gain[index] = cal_utils.model_point_source_transit(peak_ra, *param[index])

            # Create container to hold results of fit
            gain_data = containers.PointSourceTransit(ra=ra_slice, pol_x=xfeeds, pol_y=yfeeds,
                                                      axes_from=sstream)

            gain_data.evalue_x[:] = evalue_x
            gain_data.evalue_y[:] = evalue_y
            gain_data.response[:] = resp
            gain_data.response_error[:] = resp_err
            gain_data.flag[:] = fit_flag
            gain_data.parameter[:] = param
            gain_data.parameter_cov[:] = param_cov

            # Update units
            gain_data.response.attrs['units'] = unit_in + ' / ' + unit_out
            gain_data.response_error.attrs['units'] = unit_in + ' / ' + unit_out

        else:

            # Create container to hold gains
            gain_data = containers.StaticGainData(axes_from=sstream)

        # Combine dynamic range estimates for both polarizations
        dr = np.minimum(dr_x[:, slice_centre], dr_y[:, slice_centre])

        # Copy to container all quantities that are common to both
        # StaticGainData and PointSourceTransit containers
        gain_data.add_dataset('weight')

        gain_data.gain[:] = gain
        gain_data.weight[:] = dr

        # Update units and unit conversion
        gain_data.gain.attrs['units'] = unit_in + ' / ' + unit_out
        gain_data.gain.attrs['converts_units_to'] = 'Jy'

        # Add attribute with the name of the point source
        # that was used for calibration
        gain_data.attrs['source'] = self.source

        # Return gain data
        return gain_data


class ThermalCalibration(task.SingleTask):
    """Use weather temperature information to correct calibration
       in between point source calibrations.

    Attributes
    ----------
    caltime_path : 


    """

    caltime_path = config.Property(proptype=str)

    def setup(self):
        """
        """
        import h5py
        from datetime import datetime
        # Load calibration times data
        self.caltime_file = h5py.File(self.caltime_path, 'r')

        # Load weather temperatures
        self.log.info("Loading weather temperatures")
        start_time = np.amin(self.caltime_file['tref']) - 10.*3600.
        start_time = ephemeris.unix_to_datetime(start_time)
        end_time = datetime.now()
        self._load_weather(start_time, end_time)

    def process(self, data):
        """Determine calibration from a sidereal stream.

        Parameters
        ----------

        Returns
        -------
        """
        # Frequencies and RA/time
        freq = data.index_map['freq']['centre'][:]
        timestamp = self._ra2unix(data.attrs['lsd'], data.index_map['ra'][:])

        # Find refference times for each timestamp.
        # This is the time of the transit from which the gains
        # applied to the data were derived.
        self.log.info("Getting refference times")
        reftime = self._get_reftime(timestamp, self.caltime_file)

        # Compute gain corrections
        self.log.info("Computing gains corrections")
        g = self._reftime2gain(reftime, timestamp, freq)

        # Ensure data is distributed in something else than RA/time or freq.
        # That is: 'el' for maps, 'stack' or 'prod' for sstreams.
        self.log.info("Redistributing data")
        data.redistribute(['el', 'stack', 'prod'])

        # Apply gain correction
        # For now, just assume the following shapes:
        # For maps: ['freq', 'pol', 'ra', 'beam', 'el']
        # For sstreams: ['freq', 'stack', 'ra']
#        freqslice = np.s_[:10]
        self.log.info("Applying gain correction")
        if ('vis' in data.keys()):
            data['vis'][:] *= g[:, np.newaxis, :]
#            data['vis'][freqslice] *= g[freqslice, np.newaxis, :]
        else: # ('map' in data.keys())
            data['map'][:] *= g[:, np.newaxis, :, np.newaxis, np.newaxis]
#            data['map'][freqslice] *= g[freqslice, np.newaxis, :, np.newaxis, np.newaxis]

        return data


    def _ra2unix(self, csd, ra):
        """ csd must be integer """
        ra0tm = ephemeris.csd_to_unix(csd)
        return ra * (ephemeris.SIDEREAL_S*3600.*24.) / 360. + ra0tm

    def _reftime2gain(self, reftime, timestamp, frequency):
        """
        """
        ntimes = len(timestamp)
        nfreq = len(frequency)
        if not isinstance(frequency, (list, np.ndarray)):
            frequency = np.array([frequency])
        # Ones. Don't modify data where there are no gains
        g = np.ones((nfreq, ntimes), dtype=np.float)
        finitemask = np.isfinite(reftime)
        reftemp = self._interpolate_temperature(
                self.wtime, self.wtemp, reftime[finitemask])
        temp = self._interpolate_temperature(
                self.wtime, self.wtemp, timestamp[finitemask])
        g[:, finitemask] = self.gaincorr(
                reftemp[np.newaxis, :], temp[np.newaxis, :],
                frequency[:, np.newaxis], squared=True)
    
        return g

    def _interpolate_temperature(self, temptime, tempdata, times):
        # Interpolate temperatures
        x = times
        xp = temptime
        fp = tempdata
    
        return np.interp(x, xp, fp)

    # TODO: Should move this to ch_util!
    def gaincorr(self, T0, T, freq, squared=False):#, pol):
        """
        Parameters
        ----------
        freq : float or array of foats
            Frequencies in MHz
        squared : bool
            If true, return 1 + 2*corr. To avoid squaring
            the gain corrections when applying to visibility
            data or maps.

        Returns
        -------
        g : float or array of floats
            Gain amplitude corrections. Multiply by data
            to correct it.
        """
    #     if pol==0:
    #         m_params = [-4.15588627e-09, 8.27318534e-06, -2.02181757e-03]
    #     else:
    #         m_params = [-4.40948632e-09, 8.51834265e-06, -1.99043022e-03]
        m_params = [-4.28268629e-09, 8.39576400e-06, -2.00612389e-03]
        m = np.polyval(m_params, freq)
        
        if squared:
            return 1. + 2.* m * (T - T0)
        else:
            return 1. + m * (T - T0)

    def _get_reftime(self, tms, calfl):
        """
        """
        # Len of tms, indices in calfl.
        last_start_index = np.searchsorted(calfl['tstart'][:], tms) - 1
        # Len of tms, indices in calfl.
        last_end_index = np.searchsorted(calfl['tend'][:], tms) - 1
        # TODO: add test for indices < 0 here.
    
        last_start_time = calfl['tstart'][:][last_start_index]
        
        reftime = np.full(len(tms), np.nan, dtype=np.float)
        is_restart = calfl['is_restart'][:]
        tref = calfl['tref'][:]
        
        # Acquisition restart. We load an old gain.
        acqrestart = (is_restart[last_start_index] == 1)
        reftime[acqrestart] = tref[last_start_index][acqrestart]
        
        # FPGA restart. Data not calibrated.
        fpgarestart = (is_restart[last_start_index] == 2)
        # I think the file has a tref for those.
        # TODO: Should I use them?
        # reftime[fpgarestart] = tref[last_start_index][fpgarestart]
        
        # Gain transition. Need to interpolate gains.
        gaintrans = (last_start_index == (last_end_index+1))
        # Previous update was a restart.
        prev_isrestart = is_restart[last_start_index - 1].astype(bool)
        # Previous update was a gain update.
        prev_isgain = np.invert(prev_isrestart)
        # This update is in gain transition and previous update was a restart.
        # Just use new gain, no interpolation.
        prev_isrestart = prev_isrestart & gaintrans
        reftime[prev_isrestart] = tref[last_start_index][prev_isrestart]
        # This update is in gain transition and previous update was a gain update.
        # Need to interpolate gains.
        # TODO: To correct interpolated gains I need to know what 
        # the applied gains were! For now, just correct for the new gain.
        prev_isgain = np.invert(prev_isrestart) & gaintrans
        reftime[prev_isgain] = tref[last_start_index][prev_isgain]
    
        # Calibrated range. Gain transition has finished.
        calrange = (last_start_index == last_end_index)
        reftime[calrange] = tref[last_start_index][calrange]
    
        return reftime

    def _load_weather(self, start_time, end_time):
        """
        """
        from ch_util import data_index
        ntime = None

        # Can only query the database from one rank.
        if mpiutil.rank == 0:
            f = data_index.Finder(node_spoof={"cedar_archive": '/project/rpp-krs/chime/chime_archive'})
            f.only_weather()
            f.set_time_range(start_time, end_time)
            f.accept_all_global_flags()
            results_list = f.get_results()
            if len(results_list) != 1:
                msg = 'Cannot deal with multiple weather acquisitions'
                raise RuntimeError(msg)
            result = results_list[0]
            wdata = result.as_loaded_data()

            self.wtime, self.wtemp = wdata.time[:], wdata['outTemp'][:]
            ntime = len(self.wtime)

        # Broadcast the times and temperatures to all ranks.
        ntime = mpiutil.world.bcast(ntime, root=0)
        if mpiutil.rank != 0:
            self.wtime = np.empty(ntime, dtype=np.float64)
            self.wtemp = np.empty(ntime, dtype=np.float64)

        # For some reason I need to cast as float here.
        # Bcast chokes when I use np.float64...
        self.wtime = self.wtime.astype(float)
        self.wtemp = self.wtemp.astype(float)
        mpiutil.world.Bcast(self.wtime, root=0)
        mpiutil.world.Bcast(self.wtemp, root=0)

