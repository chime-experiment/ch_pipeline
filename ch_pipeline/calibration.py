"""
=======================================================
Tasks for Calibration (:mod:`~ch_pipeline.calibration`)
=======================================================

.. currentmodule:: ch_pipeline.calibration

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

from caput import config, pipeline
from caput import mpiarray, mpiutil

from ch_util import tools
from ch_util import ephemeris
from ch_util import ni_utils

from . import containers, task
from . import _fast_tools


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
    dr : np.ndarray[nfreq, ntime]
        Dynamic range of solution.
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
    dr = np.zeros((data.shape[0], data.shape[-1]), np.float64)

    # Set up normalisation matrix
    if norm is None:
        norm = _extract_diagonal(data, axis=1)**0.5

    # Extract only the required feeds for the normalisation
    norm = norm[:, feeds]

    # Pre-generate the array of inverted norms
    inv_norm = tools.invert_no_zero(norm)

    # Initialise a temporary array for unpacked products
    cd = np.zeros((nfeed, nfeed), dtype=data.dtype)

    # Iterate over frequency/time and solve gains
    for fi in range(data.shape[0]):
        for ti in range(data.shape[-1]):

            # Unpack visibility and normalisation array into square matrix
            _fast_tools._unpack_product_array_fast(data[fi, :, ti].copy(), cd, feeds, tfeed)

            if not np.isfinite(cd).all():
                continue

            # Apply weighting
            w = norm[fi, :, ti]
            cd *= np.outer(w, w.conj())

            # Solve for eigenvectors
            evals, evecs = tools.eigh_no_diagonal(cd, niter=5, eigvals=(nfeed - 2, nfeed - 1))

            # Construct dynamic range and gain, but only if the two highest
            # eigenvalues are positive. If not, we just let the gain and dynamic
            # range stay as zero.
            if evals[-1] > 0 and evals[-2] > 0:
                dr[fi, ti] = evals[-1] / evals[-2]
                gain[fi, :, ti] = inv_norm[fi, :, ti] * evecs[:, -1] * evals[-1]**0.5

    return dr, gain


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

    return ts - 0.5*(np.roll(ts, dt, axis=-1) + np.roll(ts, -dt, axis=-1))


class NoiseSourceFold(task.SingleTask):
    """Fold the noise source for synced data.

    Attributes
    ----------
    period : int, optional
        Period of the noise source in integration samples.
    phase : int, optional
        Phase of noise source on sample.
    """

    period = config.Property(proptype=int, default=None)
    phase = config.Property(proptype=int, default=None)

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
        folded_ts = ni_utils.process_synced_data(ts, period=self.period, phase=self.phase)

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
        dr = abs(ev[:, -1, :]/ev[:, -2, :])
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
        dr, gain = solve_gain(gate_view, norm=norm_view)

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
    """Use CasA as a point source calibrator for a sidereal stack.

    Attributes
    ----------
    source : str
        Point source to use as calibrator. Only CasA is supported at this time.
    """

    source = config.Property(proptype=str, default='CasA')

    _source_dict = {'CasA': ephemeris.CasA,
                    'CygA': ephemeris.CygA}

    def process(self, sstream, inputmap):
        """Apply calibration to a timestream.

        Parameters
        ----------
        sstream : containers.SiderealStream
            Rigidized sidereal timestream to calibrate.

        Returns
        -------
        sstream : containers.SiderealStream
            Calibrated sidereal timestream.
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

        # Use input map to figure out which are the X and Y feeds
        xfeeds = [idx for idx, inp in enumerate(inputmap) if tools.is_chime_x(inp)]
        yfeeds = [idx for idx, inp in enumerate(inputmap) if tools.is_chime_y(inp)]

        nfeed = len(inputmap)

        # Fetch source
        source = self._source_dict[self.source]

        _PF_ROT = np.radians(1.986)  # Rotation angle of pathfinder
        _PF_LAT = np.radians(49.0)   # Latitude of pathfinder

        # Estimate the RA at which the transiting source peaks
        peak_ra = source._ra + np.tan(_PF_ROT) * (source._dec - _PF_LAT) / np.cos(_PF_LAT)

        # Find closest array index
        idx = np.abs(sstream.ra - np.degrees(peak_ra)).argmin()
        # Fetch the transit into this visibility array

        # Cut out a snippet of the timestream
        slice_width = 40
        slice_centre = slice_width
        st, et = idx - slice_width, idx + slice_width

        vis_slice = sstream.vis[..., st:et].copy()
        ra_slice = sstream.ra[st:et]

        # Fringestop the data
        vis_slice = tools.fringestop_pathfinder(vis_slice, ra_slice, freq, inputmap, source)

        # Figure out how many samples is ~ 2 degrees, then subtract nearby values
        diff = int(2.0 / np.median(np.abs(np.diff(sstream.ra))))
        vis_slice = _cdiff(vis_slice, diff)

        # Solve for the gains of each set of polarisations
        dr_x, gain_x = solve_gain(vis_slice, feeds=xfeeds)
        dr_y, gain_y = solve_gain(vis_slice, feeds=yfeeds)

        # Construct the final gain arrays
        gain = np.ones([nfreq, nfeed], np.complex128)
        gain[:, xfeeds] = gain_x[:, :, slice_centre]  # slice_width should be the central value i.e. transit
        gain[:, yfeeds] = gain_y[:, :, slice_centre]

        # Combine both dynamic range estimates
        dr = np.minimum(dr_x[:, slice_centre], dr_y[:, slice_centre])

        # Create container from gains
        gain_data = containers.StaticGainData(axes_from=sstream)
        gain_data.add_dataset('weight')

        # Copy data into container
        gain_data.gain[:] = gain
        gain_data.weight[:] = dr

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
    smoothing_length = config.Property(proptype=float, default=None)

    def process(self, tstream, gain):

        tstream.redistribute('freq')
        gain.redistribute('freq')

        if isinstance(gain, containers.StaticGainData):

            # Extract gain array and add in a time axis
            gain_arr = gain.gain[:, :, np.newaxis]

            # Get the weight array if it's there
            weight_arr = gain.weight[:, np.newaxis] if gain.weight is not None else None

        elif isinstance(gain, containers.GainData):

            # Extract gain array
            gain_arr = gain.gain[:]

            # Regularise any crazy entries
            gain_arr = np.nan_to_num(gain_arr)

            # Get the weight array if it's there
            weight_arr = gain.weight[:] if gain.weight is not None else None

            # Check that we are defined at the same time samples
            if (gain.index_map['time'][:] != tstream.index_map['time'][:]).all():
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

        # Invert the gains if needed
        if self.inverse:
            gain_arr = tools.invert_no_zero(gain_arr)

        # Apply gains to visibility matrix
        tools.apply_gain(tstream.vis[:], gain_arr, out=tstream.vis[:])

        # Modify the weight array according to the gain weights
        if weight_arr is not None:

            # Convert dynamic range to a binary weight and apply to data
            gain_weight = (weight_arr[:] > 2.0).astype(np.float64)
            tstream.weight[:] *= gain_weight[:, np.newaxis, :]

        return tstream
