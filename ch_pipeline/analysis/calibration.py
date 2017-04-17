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
from ch_util import fluxcat

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

def std_from_mad(a):
    '''
    Calculate standard deviation from Median absolute deviation of time (last) 
    axis for array a. a is assumed to be REAL

    Parameters
    ----------
    a : np.ndarray[freq, prod/input, time]
        3d real array.

    Returns
    -------
    a_std : np.ndarray[freq, prod/input]
        Standard deviation along time (last) axis of a
    '''

    return np.median( abs(a - np.median(a, axis=-1)[..., np.newaxis]), axis=-1 )*1.4826

def get_med_std(vis_array, frames_per_test=None):
    '''Get median and standard deviation from MAD along time (last in general) axis.
    vis_array has the nominal shape [freq, prod/input, time]. 
    Finds the median and std for chunks of frames_per_test frames, and then takes 
    median. frames_per_test must be even
    
    Parameters
    ----------
    vis_array : np.ndarray[freq, prod/input, time]
        3d array.
    frames_per_test: uint8
        Number of samples per test. Must be even

    Returns
    -------
    vis_array_med : np.ndarray[freq, prod/input]
        Median along time (last) axis of a
    vis_array_std : Standard deviation (from MAD) along time (last) axis of a
    '''
    
    Nframes = vis_array.shape[-1]
    if frames_per_test is None:
        frames_per_test = Nframes
        
    Ntests = Nframes/frames_per_test #Number of tests per freq
        
    med = []
    std = []
    for i in range(Ntests):
        v = vis_array[..., i*frames_per_test:(i+1)*frames_per_test]
        v_diff = (v[..., ::2]-v[..., 1::2])/np.sqrt(2) # remove long time scale fluctuations
        std.append(std_from_mad(v_diff)[..., np.newaxis])
        med.append(np.median(v, axis=-1)[..., np.newaxis])
        del v, v_diff
                    
    return np.median(np.concatenate(med, axis=-1), axis=-1), np.median(np.concatenate(std, axis=-1), axis=-1)

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
                evalue[fi, :, ti] = evals

            # Solve for eigenvectors
            #evals, evecs = tools.eigh_no_diagonal(cd, niter=5, eigvals=(nfeed - 2, nfeed - 1))

            # Construct dynamic range and gain, but only if the two highest
            # eigenvalues are positive. If not, we just let the gain and dynamic
            # range stay as zero.
            # if evals[-1] > 0 and evals[-2] > 0:
            #     dr[fi, ti] = evals[-1] / evals[-2]
            #     gain[fi, :, ti] = inv_norm[fi, :, ti] * evecs[:, -1] * evals[-1]**0.5

    return evalue, gain


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
    norm : ['gated', 'off', 'on', 'none']
        Specify what to use to normalise the matrix.

    from_NoiseSource : bool
        If True, the gains are calculated from the crosscorrelations
        with the NoiseSource reference channel only (no eigenvalue decomposition)

    dist_channel : int
        Index (0-255) of the correlator input that tracks the fluctuations
        in the noise injection (ni) distribution system. When available,
        this channel is used to remove the ni distribution gain fluctuations 
        (only magnitude since phase is common to all feeds) from the calculated gains.
        If the noise source power is sigma^2 (as measured by the 
        noise source input) and d is the (voltage) gain of the ni distribution then it 
        is assumed that the gain of the NoiseSource reference channel is sigma and that 
        if the dist_channel input channel is d^2.sigma (d^2 because there is a loopback cable). 
        This parameter is ignored if the NoiseSource reference channel is not available 
        in the data. If dist_channel = None and the NoiseSource reference channel is available
        then dist_channel is set to the NoiseSource id to remove the fluctuations
        of the noise source.
    """

    norm = config.Property(proptype=str, default='on')
    from_NoiseSource = config.Property(proptype=bool, default=False)
    dist_channel = config.Property(proptype=int, default=None)

    def process(self, ts, inputmap, inputmask):
        """Find complex gains from noise injection data.

        Parameters
        ----------
        ts : andata.CorrData
            Parallel timestream class containing noise injection data
            (vis dataset has noise OFF data and gated_vis1 has ON-OFF data).
        inputmap : list of :class:`CorrInput`s
            A list of describing the inputs as they are in the file.
        inputmask : containers.CorrInputMask
            Mask indicating which correlator inputs to use in the
            eigenvalue decomposition.

        Returns
        -------
        gains : containers.GainData 
            Complex gains from noise injection
        """

        # Ensure that we are distributed over frequency
        ts.redistribute('freq')

        # Figure out which input channel is the noise source (used as gain reference)
        noise_channel = tools.get_noise_channel(inputmap)

        # Take a view now to avoid some MPI issues
        vis_gate = ts.datasets['gated_vis1'][:].view(np.ndarray)
        vis_on = (ts.datasets['gated_vis1'][:] + ts.vis[:]).view(np.ndarray)
        nfreq = vis_on.shape[0]
        ntime = vis_on.shape[-1]

        # Determine good inputs
        nfeed = len(inputmap)
        mask = inputmask.datasets['input_mask'][:]
        # Make sure we keep the noise channel (and distribution channel, if exists)
        if noise_channel is not None:
            mask[noise_channel] = True
            if self.dist_channel is None:
                self.dist_channel = noise_channel
            else:
                mask[self.dist_channel] = True

        good_input = np.arange(nfeed, dtype=np.int)[mask]
        N_good_input = len(good_input)
        bad_input = np.delete(np.arange(nfeed, dtype=np.int), good_input)

        # Illumination = 1-Vii_off/Vii_on
        illumination = _extract_diagonal(vis_gate.real)*tools.invert_no_zero(_extract_diagonal(vis_on.real))
        illumination[:, bad_input, :] = 0
        illumination = np.where(illumination<0., 0., illumination) # Negative illumination not allowed

        # Create arrays to hold noise source response
        gain = np.zeros([nfreq, nfeed, ntime], np.complex128)
        gain_error = np.zeros([nfreq, nfeed, ntime], np.float64)

        if self.from_NoiseSource: # Get gains from correlations with NoiseSource only (on gate)
            if noise_channel is not None:
                if mpiutil.rank0:
                    print "Calculating noise inj gains from crosscorrelations with NoiseSource channel."
                # Find corr. indices corresponding to correlations with ref channel
                idx = [tools.cmap(i, noise_channel, nfeed) for i in good_input]
                
                # Correlations with ref channel
                gain[:, good_input, :] = vis_on[:, idx, :]
                
                # Divide by sqrt(abs(Vrr)) to have gains in same format as with evalue decomp (g_i*\sigma)
                gain *= np.sqrt(tools.invert_no_zero(abs(gain[:, np.newaxis, noise_channel, :])))
                
                # Since the gains are obtained from the upper triangle of the
                # correlation matrix, need to conjugate all gains for which
                # idx > noise_channel
                gain[:, noise_channel:, :] = gain[:, noise_channel:, :].conj()
                
                # Refer all phases to the phase of the first good channel
                gain *= np.exp(-1j*np.angle(gain[:, np.newaxis, good_input[0], :]))
                
                # Remove noise source fluctuations (and noise injection distribution system 
                # fluctuations if available)
                gain_dist = np.sqrt(abs(gain[:, np.newaxis, noise_channel, :]*
                                        gain[:, np.newaxis, self.dist_channel, :]))
                gain *= tools.invert_no_zero(gain_dist)
                gain = np.nan_to_num(gain)

                # There are no evalues so create a zero array (hack)
                evalue = np.zeros([nfreq, N_good_input, ntime], np.float64)
            else:
                raise RuntimeError('NoiseSource cannot be None if from_NoiseSource is enabled.')
        else: # Get gains using all data from eigenvalue decomposition
            if mpiutil.rank0:
                    print "Calculating noise inj gains from full correlation matrix."
            # Get the norm matrix
            if self.norm == 'gated':
                norm_array = _extract_diagonal(vis_gate.real)**0.5
                norm_array = tools.invert_no_zero(norm_array)
                norm_array[:, bad_input, :] = 0
            elif self.norm == 'off':
                norm_array = _extract_diagonal(ts.vis[:].real)**0.5
                norm_array = tools.invert_no_zero(norm_array)
                norm_array[:, bad_input, :] = 0
            elif self.norm == 'on':
                norm_array = _extract_diagonal(vis_on.real)**0.5
                norm_array = tools.invert_no_zero(norm_array)
                norm_array[:, bad_input, :] = 0
            elif self.norm == 'none':
                norm_array = np.ones([ts.vis[:].shape[0], ts.ninput, ts.ntime], dtype=np.uint8)
            else:
                raise RuntimeError('Value of norm not recognised.')

            # Find gains with the eigenvalue method
            if mpiutil.rank0: print 'Calculating gains'
            evalue, gain[:, good_input, :] = solve_gain(vis_gate, feeds=good_input, norm=norm_array[:, good_input])
            
            #Calculate dynamic range and gain errors
            sigma_ev = np.std(evalue[:, :-1], axis=1)/np.sqrt(N_good_input-1)
            r_ev = evalue[:, -1]*tools.invert_no_zero(sigma_ev) #dynamic range
            gain_error[:, good_input, :] = np.sqrt((sigma_ev*tools.invert_no_zero(r_ev))[:, np.newaxis, :] * 
                                  tools.invert_no_zero(illumination[:, good_input, :]))

            # Remove noise injection distribution system fluctuations and/or
            # noise source fluctuations if available
            if noise_channel is not None:
                gain_dist = np.sqrt(abs(gain[:, np.newaxis, noise_channel, :]*
                                        gain[:, np.newaxis, self.dist_channel, :]))
                gain *= tools.invert_no_zero(gain_dist)
                gain = np.nan_to_num(gain)

        # Create container to hold gain results
        gain_data = containers.GainData(good_input=good_input, axes_from=ts)

        gain_data.gain[:] = gain
        gain_data.evalue[:] = evalue
        gain_data.gain_error_med[:] = np.median(gain_error, axis=-1)

        illumination_med = np.median(illumination, axis=-1)
        gain_data.illumination_med[:] = illumination_med

        # For now there is one weight per channel per file
        # The weight is just the inverse of the relative variance
        # (sigma_{g_i}/g_i)^2
        Nsamples = ts.attrs['gpu.gpu_intergration_period'][0]
        # alpha is the extra factor you get in the variance of gains when using full matrix
        alpha = np.sum(illumination_med, axis=1)[:, np.newaxis] if self.from_NoiseSource else 1.
        gain_data.add_dataset('weight')
        gain_data.weight[:] = Nsamples*illumination_med*alpha

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
        Default is 5.

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
        slice_width_deg = 3.0*cal_utils.guess_fwhm(400.0, pol='X', dec=source._dec, sigma=True)
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
        vis_slice_diff = _adiff(vis_slice, diff)

        # Illumination = 1-Vii_off/Vii_on
        illumination = _extract_diagonal(vis_slice_diff.real)*tools.invert_no_zero(_extract_diagonal(vis_slice.real))

        # Fringestop the data
        vis_slice_diff = tools.fringestop_pathfinder(vis_slice_diff, ra_slice, freq, inputmap, source)

        # Create arrays to hold point source response
        resp = np.zeros([nfreq, nfeed, nra], np.complex128)
        resp_err = np.zeros([nfreq, nfeed, nra], np.float64)

        # Solve for the point source response of each set of polarisations
        evalue_x, resp[:, xfeeds, :] = solve_gain(vis_slice_diff, feeds=xfeeds, norm=norm[:, xfeeds])
        evalue_y, resp[:, yfeeds, :] = solve_gain(vis_slice_diff, feeds=yfeeds, norm=norm[:, yfeeds])

        #Calculate dynamic range and gain errors
        sigma_ev_x = np.median(abs(evalue_x[:, :-1]-np.median(evalue_x[:, :-1], axis=1)[:, np.newaxis, :]),
                             axis=1)*1.4826/np.sqrt(len(xfeeds)-1)
        r_ev_x = evalue_x[:, -1]*tools.invert_no_zero(sigma_ev_x) #dynamic range
        resp_err[:, xfeeds, :] = np.sqrt((sigma_ev_x*tools.invert_no_zero(r_ev_x))[:, np.newaxis, :] * 
                              tools.invert_no_zero(illumination[:, xfeeds, :]))

        sigma_ev_y = np.median(abs(evalue_y[:, :-1]-np.median(evalue_y[:, :-1], axis=1)[:, np.newaxis, :]),
                             axis=1)*1.4826/np.sqrt(len(yfeeds)-1)
        r_ev_y = evalue_y[:, -1]*tools.invert_no_zero(sigma_ev_y) #dynamic range
        resp_err[:, yfeeds, :] = np.sqrt((sigma_ev_y*tools.invert_no_zero(r_ev_y))[:, np.newaxis, :] * 
                              tools.invert_no_zero(illumination[:, yfeeds, :]))

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
        dr_x = evalue_x[:, -1, :]*tools.invert_no_zero(evalue_x[:, -2, :])
        dr_y = evalue_y[:, -1, :]*tools.invert_no_zero(evalue_y[:, -2, :])

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
                gain = np.where(np.isnan(param[:,:,0]), gain,
                                param[:,:,0]*np.exp(1.0j*np.deg2rad(param[:,:,-2])))
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


class ApplyGain(task.SingleTask):
    """Apply a set of gains to a timestream or sidereal stack.

    Attributes
    ----------
    inverse : bool, optional
        Apply the gains directly, or their inverse.
    smoothing_length : float, optional
        Smooth the gain timestream across the given number of seconds.
    update_weight: bool, optional
        Update timestream weights. For sidereal calibration, the weights
        are rescaled by applying the inverse of the gains applied to 
        visibilities. For noise injection calibration, the weights
        are either rescaled by applying the inverse of the gains applied to 
        visibilities or the weights are updated with the inverse variance
        resulting from propagating the gain errors.
    scale_weight_only: bool, optional
        If True, the data weights are rescaled by applying the inverse of 
        the gains applied to visibilities. If False, the weights are updated 
        with the inverse variance resulting from propagating the gain errors
        (assumes the current weight of the tstream is just the inverse variance 
        from the rad eqn).
    normalize_gains: bool, optional
        If True, the noise injection gains are normalized the median of the
        gain magnitude along the time axis
    """

    inverse = config.Property(proptype=bool, default=True)
    update_weight = config.Property(proptype=bool, default=False)
    scale_weight_only = config.Property(proptype=bool, default=False)
    normalize_gains = config.Property(proptype=float, default=False)
    smoothing_length = config.Property(proptype=float, default=None)

    def process(self, tstream, gain):

        tstream.redistribute('freq')
        gain.redistribute('freq')

        if isinstance(gain, (containers.StaticGainData, containers.PointSourceTransit)):

            # Extract gain array and add in a time axis
            gain_arr = gain.gain[:][..., np.newaxis]

            # Get the weight array if it's there
            weight_arr = gain.weight[:][..., np.newaxis] if gain.weight is not None else None

            # Regularise any crazy entries
            gain_arr = np.nan_to_num(gain_arr)

            # If requested, invert the gains
            inverse_gain_arr = tools.invert_no_zero(gain_arr)

            if self.inverse:
                gweight = gain_arr
                gvis = inverse_gain_arr
                if mpiutil.rank0:
                    print "Applying inverse gain."
            else:
                gweight = inverse_gain_arr
                gvis = gain_arr
                if mpiutil.rank0:
                    print "Applying gain."

            # Apply gains to the weights
            if self.update_weight:
                tools.apply_gain(tstream.weight[:], np.abs(gweight)**2, out=tstream.weight[:])
                if mpiutil.rank0:
                    print "Applying gain to weight."

            # Apply gains to visibility matrix
            tools.apply_gain(tstream.vis[:], gvis, out=tstream.vis[:])

            # Update units
            convert_units_to = gain.gain.attrs.get('convert_units_to')
            if convert_units_to:
                tstream.vis.attrs['units'] = convert_units_to

            # Modify the weight array according to the gain weights
            if weight_arr is not None:
                # Convert dynamic range to a binary weight and apply to data
                gain_weight = (weight_arr[:] > 2.0).astype(tstream.weight.dtype)
                tstream.weight[:] *= gain_weight[:, np.newaxis, :]

        elif isinstance(gain, containers.GainData): #Noise injection gains

            # Extract gain array
            gain_arr = gain.gain[:].view(np.ndarray) # If not viewed as ndarray get error when smoothing
            if self.normalize_gains:
                g_mag_med, g_mag_std = get_med_std(np.abs(gain_arr), 100)
                gain_arr *= tools.invert_no_zero(g_mag_med[..., np.newaxis]) 

            # Get the weight array if it's there
            weight_arr = gain.weight[:].view(np.ndarray) if gain.weight is not None else None

            # Check that we are defined at the same time samples
            if (gain.index_map['time']['ctime'] != tstream.index_map['time']['ctime']).any():
                raise RuntimeError('Gain data and timestream defined at different time samples.')

            # Smooth the gain data if required
            if self.smoothing_length is not None:
                import scipy.signal as ss

                if mpiutil.rank0:
                    print "Smoothing noise injection gains."

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
                    weight_arr_shape = weight_arr.shape
                    if weight_arr_shape[-1] == gain_arr.shape[-1]: # If not assume gain.weight is 2D so don't smooth
                        weight_arr = weight_arr.reshape(-1, weight_arr_shape[-1])
                        weight_arr = ss.medfilt2d(np.abs(weight_arr), kernel_size=[1, l])
                        weight_arr = weight_arr.reshape(weight_arr_shape)

            # Regularise any crazy entries
            gain_arr = np.nan_to_num(gain_arr)

            if mpiutil.rank0:
                print "Applying inverse gain to visibilities."

            # Apply gains to visibility matrix
            vis_raw = tstream.vis[:].copy()
            tools.apply_gain(tstream.vis[:], tools.invert_no_zero(gain_arr), out=tstream.vis[:])

            # Modify the weight array according to the gain weights
            if self.update_weight and (weight_arr is not None):
                if self.scale_weight_only:
                    # Do nt propagate error from applying noise inj gains.
                    # Only rescale inverse rad eqn variance by applied gains
                    # i.e assume perfect knowledge of gains
                    if mpiutil.rank0:
                        print "Applying gain to weights (rescale only)."

                    tools.apply_gain(tstream.weight[:], np.abs(gain_arr)**2, out=tstream.weight[:])
                else:
                    # Propagate error on visibilities from applying noise inj gains.
                    # Assumes the current weight of the tstream is just the inverse variance from the 
                    # rad eqn.

                    if mpiutil.rank0:
                        print "Applying gain to weights (propagate gain errors)."
                    # Variance contribution due to raw visibility
                    var_radeqn = tools.invert_no_zero(tstream.weight[:]*np.abs(vis_raw)**2)
                    # Variance contribution due to inverse gain
                    nfreq = gain_arr.shape[0]
                    nchannels = gain_arr.shape[1]
                    var_gi = np.concatenate([tools.invert_no_zero(weight_arr[:, i:i+1]) * 
                                             np.ones((nfreq, nchannels-i)) 
                                             for i in range(nchannels)], axis=1)
                    var_gj = np.concatenate([tools.invert_no_zero(weight_arr[:, i:])
                                             for i in range(nchannels)], axis=1)

                    tstream.weight[:] = tools.invert_no_zero(np.abs(tstream.vis[:])**2 * 
                                        (var_radeqn + var_gi[..., np.newaxis] + var_gj[..., np.newaxis]))
            else: 
                if mpiutil.rank0:
                    print "Data weights not updated."

        else:
            raise RuntimeError('Format of `gain` argument is unknown.')

        return tstream