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

    PointSourceCalibration
    NoiseSourceCalibration
"""
import numpy as np
from scipy import interpolate

from caput import pipeline
from caput import config
from caput import mpidataset, mpiutil

from ch_util import andata
from ch_util import tools
from ch_util import ephemeris
from ch_util import ni_utils

import containers

from . import dataspec


def gen_corr_matrix(data, nfeed, feed_loc=None):
    """Generates Hermitian correlation matrix from unique correlations.

    Should probably be in ch_util.tools, or at least not in here.

    Parameters
    ----------
    data : array_like
        Visibility array to be decomposed.
    nfeed : int
        Number of feeds.
    feed_loc : list of ints
        Which feeds to include. If :obj:`None` include all feeds.

    Returns
    -------
    corr_mat : np.ndarray[len(feed_loc), len(feed_loc)]
        Hermitian correlation matrix.
    """

    if feed_loc is None:
        feed_loc = range(nfeed)

    corr_mat = np.zeros((len(feed_loc), len(feed_loc)), np.complex128)

    for ii in range(len(feed_loc)):
        for jj in range(ii, len(feed_loc)):
            corr_mat[ii, jj] = data[tools.cmap(feed_loc[ii], feed_loc[jj], nfeed)]
            corr_mat[jj, ii] = np.conj(data[tools.cmap(feed_loc[ii], feed_loc[jj], nfeed)])

    return corr_mat


def solve_gain(data, nfeed, feed_loc=None):
    """
    Steps through each time/freq pixel, generates a Hermitian matrix and
    calculates gains from its largest eigenvector.

    Parameters
    ----------
    data : array_like
        Visibility array to be decomposed
    nfeed : number
        Number of feeds in configuration
    feed_loc : list
        Which feeds to include. If :obj:`None` include all feeds.

    Returns
    -------
    gain_arr : np.ndarray
        Gain solution for each feed, time, and frequency
    """
    if not feed_loc:
        feed_loc = range(nfeed)

    gain_arr = np.zeros([data.shape[0], len(feed_loc), data.shape[-1]], np.complex128)

    for nu in range(data.shape[0]):
        for tt in range(data.shape[-1]):

            corr_arr = gen_corr_matrix(data[nu, :, tt], nfeed, feed_loc=feed_loc)
            corr_arr[np.diag_indices(len(corr_arr))] = 0.0

            evl, evec = np.linalg.eigh(corr_arr)
            gain_arr[nu, :, tt] = evl[-1]**0.5 * evec[:, -1]

    return gain_arr


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


def list_transits(dspec, obj=ephemeris.CasA, tdel=600):
    """Get a list of files that contain point source transits

    Parameter
    ---------
    dspec : dictionary
        Dataset specification.
    obj : ephem.Body, optional
        Body for which to find transits.
    tdel : float, optional
        Total amount of time to include surrounding the transit in sec.

    Returns
    -------
    interval_list : :class:`DataIntervalList`
        Search results.
    """
    fi = dataspec.finder_from_spec(dspec)
    fi.include_transits(obj, time_delta=tdel)

    return fi.get_results()


# A few internal routines for figuring things out from the input map.
def _is_chime_x(inp):
    return isinstance(inp, tools.CHIMEAntenna) and inp.pol == 'E'


def _is_chime_y(inp):
    return isinstance(inp, tools.CHIMEAntenna) and inp.pol == 'S'


def _get_noise_channel(inputs):
    noise_sources = [ ix for ix, inp in enumerate(inputs) if isinstance(inp, tools.NoiseSource) ]

    return noise_sources[0]


def _apply_gain(vis, gain, axis=1, out=None):
    """Apply per input gains to a set of visibilities packed in upper
    triangular format.

    This allows us to apply the gains while minimising the intermediate products created.

    Parameters
    ----------
    vis : np.ndarray[..., nprod, ...]
        Array of visibility products.
    gain : np.ndarray[..., ninput, ...]
        Array of gains. One gain per input.
    axis : integer, optional
        The axis along which the inputs (or visibilities) are
        contained. Currently only supports axis=1.
    out : np.ndarray
        Array to place output in. If :obj:`None` create a new
        array. This routine can safely use `out = vis`.

    Returns
    -------
    out : np.ndarray
        Visibility array with gains applied. Same shape as :obj:`vis`.
    """

    from ch_util import tools

    ninput = gain.shape[axis]

    if vis.shape[axis] != (ninput * (ninput + 1) / 2):
        raise Exception("Number of inputs does not match the number of products.")

    if out is None:
        out = np.empty_like(vis)
    elif out.shape != vis.shape:
        raise Exception("Output array is wrong shape.")

    # Iterate over input pairs and set gains
    for ii in range(ninput):

        for ij in range(ii, ninput):

            # Calculate the product index
            ik = tools.cmap(ii, ij, ninput)

            # Fetch the gains
            gi = gain[:, ii]
            gj = gain[:, ij].conj()

            # Apply the gains and save into the output array.
            out[:, ik] = vis[:, ik] * gi * gj

    return out


class PointSourceCalibration(pipeline.TaskBase):
    """Use CasA transits as point source calibrators.

    Attributes
    ----------
    source : str
        Point source to use as calibrator. Only CasA is supported at this time.
    """

    source = config.Property(proptype=str, default='CasA')

    _source_dict = { 'CasA': ephemeris.CasA }

    def setup(self, dspec, inputmap):
        """Use a dataspec to derive the calibration solutions.

        Parameters
        ----------
        dspec : dictionary
            Dataspec as a dictionary.
        inputmap : list of :class:`tools.CorrInputs`
            Describing the inputs to the correlator.
        """

        # Use input map to figure out which are the X and Y feeds
        xfeeds = [idx for idx, inp in enumerate(inputmap) if _is_chime_x(inp)]
        yfeeds = [idx for idx, inp in enumerate(inputmap) if _is_chime_y(inp)]

        self.nfeed = len(inputmap)
        self.gain_mat = []
        self.trans_times = []

        if mpiutil.rank0:

            # Fetch source and transit
            source = self._source_dict[self.source]
            transit_list = list_transits(dspec, obj=source)

            # Number of transits
            ndays = len(transit_list)

            # Loop to find gain solutions for each transit
            for k, files in enumerate(transit_list):

                if len(files[0]) > 3:
                    print "Skipping as too many files."
                    continue  # Skip large acqquisitions. Usually high-cadence observations

                flist = files[0]

                print "Reading in:", flist

                # Read in the data
                reader = andata.Reader(flist)
                reader.freq_sel = range(1024)
                data = reader.read()

                times = data.timestamp
                self.nfreq = data.nfreq
                self.ntime = data.ntime

                # Find the exact transit time for this transit
                trans_time = ephemeris.transit_times(source, times[0])
                self.trans_times.append(trans_time)

                # Select only data within a minute of the transit
                times_ind = np.where((times > trans_time - 60.) & (times < trans_time + 60.))
                vis = data.vis[..., times_ind[0]]
                del data

                print "Solving gains."
                # Solve for the gains of each set of polarisations
                gain_arr_x = solve_gain(vis, self.nfeed, feed_loc=xfeeds)
                gain_arr_y = solve_gain(vis, self.nfeed, feed_loc=yfeeds)

                print "Finished eigendecomposition"
                print ""

                # Construct the final gain arrays
                gain = np.ones([self.nfreq, self.nfeed], np.complex128)
                gain[:, xfeeds] = np.median(gain_arr_x, axis=-1)  # Take time avg of gains solution
                gain[:, yfeeds] = np.median(gain_arr_y, axis=-1)

                print "Computing gain matrix for transit %d of %d" % (k+1, ndays)
                self.gain_mat.append(gain[:, :, np.newaxis])

            self.gain_mat = np.concatenate(self.gain_mat, axis=-1)
            self.trans_times = np.concatenate(self.trans_times)
            print "Broadcasting solutions to all ranks."

        self.gain_mat = mpiutil.world.bcast(self.gain_mat, root=0)
        self.trans_times = mpiutil.world.bcast(self.trans_times, root=0)

    def next(self, ts):
        """Apply calibration to a timestream.

        Parameters
        ----------
        ts : containers.TimeStream
            Parallel timestream class.

        Returns
        -------
        calibrated_ts : containers.TimeStream
            Calibrated timestream.
        gains : np.ndarray
            Array of gains.
        """

        # Ensure that we are distributed over frequency
        ts.redistribute(0)

        # Find the local frequencies
        freq_low = ts.vis.local_offset[0]
        freq_up = freq_low + ts.vis.local_shape[0]

        # Find times that are within the calibrated region
        times = ts.timestamp
        #ind_cal = np.where((times > self.trans_times[0]) & (times < self.trans_times[-1]))[0]

        # Construct the gain matrix at all times (using liner interpolation)
        gain = interp_gains(self.trans_times, self.gain_mat[freq_low:freq_up], times)

        # Create TimeStream
        cts = ts.copy(deep=True)
        cts.add_gain()

        # Apply gains to visibility matrix and copy into cts
        _apply_gain(ts.vis, 1.0 / gain, out=cts.vis)

        # Save gains into cts instance
        cts.gain[:] = mpidataset.MPIArray.wrap(gain, axis=0, comm=cts.comm)

        # Ensure distributed over frequency axis
        cts.redistribute(0)

        return cts


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
    """

    nchannels = config.Property(proptype=int, default=16)
    ch_ref = config.Property(proptype=int, default=None)
    fbin_ref = config.Property(proptype=int, default=None)

    def setup(self, inputmap):
        """Use the input map to set up the calibrator.

        Parameters
        ----------
        inputmap : list of :class:`tools.CorrInputs`
            Describing the inputs to the correlator.
        """
        self.ch_ref = _get_noise_channel(inputmap)
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

        ts.redistribute(0)

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
        vis = _apply_gain(vis_uncal, 1.0 / gain)

        # Calculate dynamic range
        ev = ni_utils.sort_evalues_mag(nidata.ni_evals)  # Sort evalues
        dr = abs(ev[:, -1, :]/ev[:, -2, :])
        dr = dr[:, np.newaxis, :]

        # Turn vis, gains and dr into MPIArray
        vis = mpidataset.MPIArray.wrap(vis, axis=0, comm=ts.comm)
        gain = mpidataset.MPIArray.wrap(gain, axis=0, comm=ts.comm)
        dr = mpidataset.MPIArray.wrap(dr, axis=0, comm=ts.comm)

        # Create NoiseInjTimeStream
        cts = containers.TimeStream(timestamp, vis.global_shape[0], vis.global_shape[1],
                                    comm=vis.comm, copy_attrs=ts, gain=True)

        cts.vis[:] = vis
        cts.gain[:] = gain
        cts.gain_dr[:] = dr

        cts.redistribute(0)

        return cts
