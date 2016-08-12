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
    ApplyGain
"""
import numpy as np
from scipy import interpolate
from scipy.optimize import curve_fit

from caput import config, pipeline
from caput import mpiarray, mpiutil

from ch_util import tools
from ch_util import ephemeris
from ch_util import ni_utils
from ch_util import data_quality
from ch_util import cal_utils

from ..core import containers, task
from ..util import _fast_tools


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
                sign0 = (1.0 - 2.0*(evecs[0, -1].real < 0.0))
                gain[fi, :, ti] = sign0 * inv_norm[fi, :, ti] * evecs[:, -1] * evals[-1]**0.5
                                                
                gain_error[fi, :, ti] = (inv_norm[fi, :, ti] * 
                                         1.4826 * np.median(np.abs(evals[:-1] - np.median(evals[:-1]))) /
                                         evals[-1]**0.5)

                evalue[fi, :, ti] = evals

            # Solve for eigenvectors
            #evals, evecs = tools.eigh_no_diagonal(cd, niter=5, eigvals=(nfeed - 2, nfeed - 1))

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

    return ts - 0.5*(np.mean(ts[..., :dt], axis=-1) + 
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
        dr = evalue[:, -1, :]*tools.invert_no_zero(evalue[:, -2, :])

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


class SiderealCalibration(task.SingleTask):
    """Use point source as a calibrator for a sidereal stack.

    Attributes
    ----------
    source : str, default CygA
        Name of the point source to use as calibrator.
    model_fit: bool, default False
        Fit a model to the point source transit.
    threshold: float, default 5
        Relevant if model_fit is True.  The model is only fit to
        time samples with dynamic range greater than threshold.
    """

    source = config.Property(proptype=str, default='CygA')
    model_fit = config.Property(proptype=bool, default=False)
    threshold = config.Property(proptype=float, default=5.0)

    _source_dict = {'CygA': ephemeris.CygA,
                    'CasA': ephemeris.CasA,
                    'TauA': ephemeris.TauA}

    def process(self, sstream, inputmap):
        """Determine calibration from a timestream.

        Parameters
        ----------
        sstream : containers.SiderealStream
            Rigidized sidereal timestream to calibrate.

        Returns
        -------
        gains : np.ndarray
            Array of gains.
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
        source = self._source_dict[self.source]

        _PF_ROT = np.radians(1.986)     # Rotation angle of pathfinder
        _PF_LAT = np.radians(49.4991)   # Latitude of pathfinder

        # Estimate the RA at which the transiting source peaks
        peak_ra = source._ra + np.tan(_PF_ROT) * (source._dec - _PF_LAT) / np.cos(_PF_LAT)

        # Find closest array index
        idx = np.abs(sstream.ra - np.degrees(peak_ra)).argmin()
        
        # Fetch the transit into this visibility array
        # Cut out a snippet of the timestream
        slice_width_deg = 6.5 * np.cos(source._dec) / np.cos(_PF_LAT)
        slice_width = int(slice_width_deg / np.median(np.abs(np.diff(sstream.ra))))
        slice_centre = slice_width
        st, et = idx - slice_width, idx + slice_width + 1

        vis_slice = sstream.vis[..., st:et].copy()
        ra_slice = sstream.ra[st:et]
        
        nra = vis_slice.shape[-1]
        
        # Determine good inputs, as indicated by nonzero weight in auto correlation
        nfeed = len(inputmap)        
        good_input = np.arange(nfeed, dtype=np.int)[sstream.input_flag[:]]
                
        # Use input map to figure out which are the X and Y feeds
        xfeeds = np.array([idx for idx, inp in enumerate(inputmap) if tools.is_chime_x(inp) and (idx in good_input)])
        yfeeds = np.array([idx for idx, inp in enumerate(inputmap) if tools.is_chime_y(inp) and (idx in good_input)])
        
        if mpiutil.rank0:
            print("Performing sidereal calibration with %d/%d good feeds (%d xpol, %d ypol)." % 
                  (len(good_input), nfeed, len(xfeeds), len(yfeeds)))
        
        # Extract the diagonal (to be used for weighting) 
        # prior to differencing on-source and off-source
        norm = (_extract_diagonal(vis_slice, axis=1).real)**0.5
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
        ttrans = ephemeris.CSD_ZERO + (sstream.attrs['csd'] + ra_slice[slice_centre]/360.0)*(24.0 * 3600.0 * ephemeris.SIDEREAL_S)
        rt_flux_density = np.sqrt(cal_utils.get_source_flux_density(freq, self.source, time=ttrans))
        
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
        dr_x = evalue_x[:, -1, :]*tools.invert_no_zero(evalue_x[:, -2, :])
        dr_y = evalue_y[:, -1, :]*tools.invert_no_zero(evalue_y[:, -2, :])
        
        # If requested, fit a model to the point source transit
        if self.model_fit:
            
            # Only fit ra values above the specified dynamic range threshold            
            fit_flag = np.zeros([nfreq, nfeed, nra], dtype=np.bool)
            fit_flag[:, xfeeds, :] = (dr_x > self.threshold)[:, np.newaxis, :]
            fit_flag[:, yfeeds, :] = (dr_y > self.threshold)[:, np.newaxis, :]

            # Fit model for the complex response of each feed to the point source
            param, param_cov = cal_utils.fit_point_source_transit(ra_slice, resp, resp_err, flag=fit_flag)
            
            # Overwrite the initial gain estimates for frequencies/feeds
            # where the model fit was successful
            gain = np.where(np.isnan(param[:,:,0]), gain, 
                            param[:,:,0]*np.exp(1.0j*param[:,:,-2]*np.pi/180.0))

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
        dr = np.minimum(dr_x[:,slice_centre], dr_y[:,slice_centre])
            
        # Copy to container all quantities that are common to both
        # StaticGainData and PointSourceTransit containers
        gain_data.add_dataset('weight')

        gain_data.gain[:] = gain
        gain_data.weight[:] = dr
        
        # Update units and unit conversion
        gain_data.gain.attrs['units'] = unit_in + ' / ' + unit_out
        gain_data.gain.attrs['converts_units_to'] = 'Jy'
        
        # Return gain data
        return gain_data


class ApplyGain(task.SingleTask):
    """Apply a set of gains to a timestream or sidereal stack.

    Attributes
    ----------
    inverse : bool, optional
        Apply the gains directly, or their inverse.
    smoothing_length : float, optional
        Smooth the gain timestream across the given number of seconds.
    """

    inverse = config.Property(proptype=bool, default=True)
    update_weight = config.Property(proptype=bool, default=False)
    smoothing_length = config.Property(proptype=float, default=None)

    def process(self, tstream, gain):

        tstream.redistribute('freq')
        gain.redistribute('freq')

        if isinstance(gain, (containers.StaticGainData, containers.PointSourceTransit)):

            # Extract gain array and add in a time axis
            gain_arr = gain.gain[:][..., np.newaxis]

            # Get the weight array if it's there
            weight_arr = gain.weight[:][..., np.newaxis] if gain.weight is not None else None

        elif isinstance(gain, containers.GainData):

            # Extract gain array
            gain_arr = gain.gain[:]

            # Regularise any crazy entries
            gain_arr = np.nan_to_num(gain_arr)

            # Get the weight array if it's there
            weight_arr = gain.weight[:] if gain.weight is not None else None

            # Check that we are defined at the same time samples
            if (gain.time != tstream.time).any():
                raise RuntimeError('Gain data and timestream defined at different time samples.')

            # Smooth the gain data if required
            if self.smoothing_length is not None:
                import scipy.signal as ss

                # Turn smoothing length into a number of samples
                tdiff = gain.time[1] - gain.time[0]
                samp = int(np.ceil(self.smoothing_length / tdiff))

                # Ensure smoothing length is odd
                l = 2 * (samp / 2) + 1

                # Turn into 2D array (required by smoothing routines)
                gain_r = gain_arr.reshape(-1, gain_arr.shape[-1])

                # Smooth amplitude and phase separately
                smooth_amp = ss.medfilt2d(np.abs(gain_r), kernel_size=[1, l])
                smooth_phase = ss.medfilt2d(np.angle(gain_r), kernel_size=[1, l])

                # Recombine and reshape back to original shape
                gain_arr = smooth_amp * np.exp(1.0J * smooth_phase)
                gain_arr = gain_arr.reshape(gain.gain[:].shape)

                # Smooth weight array if it exists
                if weight_arr is not None:
                    weight_arr = ss.medfilt2d(weight_arr, kernel_size=[1, l])

        else:
            raise RuntimeError('Format of `gain` argument is unknown.')

        # Regularise any crazy entries
        gain_arr = np.nan_to_num(gain_arr)
        
        # If requested, scale the weights
        if self.update_weight and (tstream.weight.dtype == np.float):
            tools.apply_gain(tstream.weight[:], np.abs(gain_arr)**2, out=tstream.weight[:])

        # Invert the gains if needed
        if self.inverse:
            gain_arr = tools.invert_no_zero(gain_arr)

        # Apply gains to visibility matrix
        tools.apply_gain(tstream.vis[:], gain_arr, out=tstream.vis[:])
        
        # Update units
        convert_units_to = gain.gain.attrs.get('convert_units_to')
        if convert_units_to:
            tstream.vis.attrs['units'] = convert_units_to

        # Modify the weight array according to the gain weights
        if weight_arr is not None:

            # Convert dynamic range to a binary weight and apply to data
            gain_weight = (weight_arr[:] > 2.0).astype(np.float64)
            tstream.weight[:] *= gain_weight[:, np.newaxis, :]

        return tstream
