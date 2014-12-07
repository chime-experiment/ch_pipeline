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
from caput import mpidataset

from ch_util import andata
from ch_util import tools
from ch_util import ephemeris
from ch_util import ni_utils

import containers

from . import dataspec


def gen_corr_matrix(data, nfeed, feed_loc=False):
    """Generates Hermitian (nfeed, nfeed) correlation matrix from unique
    correlations (should probably be in ch_util.tools, or at least not in
    here)

    Parameters
    ----------
    data : array_like
        Visibility array to be decomposed.
    nfeed : int
        Number of feeds.

    Returns
    -------
    corr_mat : array_like
        Hermitian correlation matrix.
    """

    if not feed_loc:
        feed_loc = range(nfeed)

    corr_mat = np.zeros((len(feed_loc), len(feed_loc)), np.complex128)

    for ii in range(len(feed_loc)):
        for jj in range(ii, len(feed_loc)):
            corr_mat[ii, jj] = data[tools.cmap(feed_loc[ii], feed_loc[jj], nfeed)]
            corr_mat[jj, ii] = np.conj(data[tools.cmap(feed_loc[ii], feed_loc[jj], nfeed)])

    return corr_mat


def solve_gain(data, nfeed, feed_loc=False):
    """
    Steps through each time/freq pixel, generates a Hermitian (nfeed,nfeed)
    matrix and calculates gains from its largest eigenvector.

    Parameters
    ----------
    data : array_like
        Visibility array to be decomposed
    nfeed : number
        Number of feeds in configuration
    feed_loc : list
        Which feeds to include

    Returns
    -------
    gain_arr : array_like
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

    Returns
    -------
    Array of interpolated gains
    """
    f = interpolate.interp1d(trans_times, gain_mat, kind='linear', axis=axis)

    return f(times)


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


class PointSourceCalibration(pipeline.TaskBase):
    """Use CasA transits as point source calibrators.

    Attributes
    ----------
    source : str
        Point source to use as calibrator. Only CasA is supported at this time.
    """

    source = config.Property(proptype=str, default='CasA')

    def setup(self, dspec):
        """Use a dataspec to derive the calibration solutions.

        Parameters
        ----------
        dspec : dictionary
            Dataspec as a dictionary.
        """

        transit_list = list_transits(dspec, obj=ephemeris.CasA)

        ndays = len(transit_list)

        # Assumed to be layout 42
        xfeeds = [0, 1, 2, 3, 12, 13, 14, 15]
        yfeeds = [4, 5, 6, 7, 8, 9, 10, 11]

        self.nfeed = 16
        self.gain_mat = []
        self.trans_times = []

        k = 0
        for files in transit_list:

            if len(files[0]) > 3:
                continue  # Skip large acqquisitions. Usually high-cadence observations

            flist = files[0]

            print "Reading in:", flist

            reader = andata.Reader(flist)
            reader.freq_sel = range(1024)
            data = reader.read()

            times = data.timestamp
            self.nfreq = data.nfreq
            self.ntime = data.ntime
            self.trans_times.append(ephemeris.transit_times(
                                    ephemeris.CasA, times[0]))

            times_ind = np.where((times > self.trans_times[k] - 60.) & (times < self.trans_times[k] + 60.))
            vis = data.vis[..., times_ind[0]]

            del data

            gain_arr_x = solve_gain(vis, self.nfeed, feed_loc=xfeeds)
            gain_arr_y = solve_gain(vis, self.nfeed, feed_loc=yfeeds)

            print "Finished eigendecomposition"
            print ""

            # Use ones since we'll be dividing in PointSourceCalibration.next
            gains = np.ones([self.nfreq, self.nfeed], np.complex128)

            gains[:, xfeeds] = np.median(gain_arr_x, axis=-1)  # Take time avg of gains solution
            gains[:, yfeeds] = np.median(gain_arr_y, axis=-1)

            print "Computing gain matrix for sidereal day %d of %d" % (k+1, ndays)
            self.gain_mat.append((gains[:, :, np.newaxis] * np.conj(gains[:, np.newaxis]))[..., np.newaxis])
            k += 1

        self.gain_mat = np.concatenate(self.gain_mat, axis=-1)
        self.gain_mat = self.gain_mat[:, np.triu_indices(self.nfeed)[0], np.triu_indices(self.nfeed)[1]]
        self.trans_times = np.concatenate(self.trans_times)

    def next_old(self, data):

        times = data.timestamp
        trans_cent = ephemeris.transit_times(ephemeris.CasA, times[0])
        print trans_cent

        #ct = np.where(self.trans_times == trans_cent)[0]

        trans_times = self.trans_times

        ind_cal = np.where((times > trans_times[0]) & (times < trans_times[-1]))[0]
        gain_mat_full = interp_gains(trans_times, self.gain_mat, times[ind_cal])

        calibrated_data = data.vis[..., ind_cal] / gain_mat_full

        return calibrated_data, gain_mat_full

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

        times = ts.timestamp

        trans_cent = ephemeris.transit_times(ephemeris.CasA, times[0])

        ct = np.where(self.trans_times == trans_cent)[0]
        trans_times = self.trans_times[ct-1:ct+1]

        ind_cal = np.where((times > trans_times[0]) & (times < trans_times[-1]))[0]

        gain_mat_full = interp_gains(trans_times, self.gain_mat[freq_low:freq_up, :2], times[ind_cal])

        calibrated_data = ts.vis[..., ind_cal] / gain_mat_full

        return calibrated_data, gain_mat_full


class NoiseInjCalibration(pipeline.TaskBase):
    """Calibration using Noise Injection

    Attributes
    ----------
    Nchannels : int, optional
        Number of channels (default 16).    
    ch_ref : int in the range 0 <= ch_ref <= Nchannels-1, optional
        Reference channel (default 0).
    fbin_ref : int, optional
        Reference frequency bin
    """
    #def setup(self):
        # Initialise any required products in here. This function is only
        # called as the task is being setup, and before any actual data has
        # been sent through

    Nchannels = config.Property(proptype=int, default=16)
    ch_ref = config.Property(proptype=int, default=None)
    fbin_ref = config.Property(proptype=int, default=None)
    #normalize_vis = config.Property(proptype=bool, default=False)
    #masked_channels = config.Property(proptype=list, default=None)
        
    def next(self, ts):
        """Find gains from noise injection data and apply them to visibilities.
        
        Parameters
        ----------
        ts : containers.TimeStream
            Parallel timestream class containing noise injection data.

        Returns
        -------
        nits : containers.NoiseInjTimeStream
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
        nidata = ni_utils.ni_data(ts, self.Nchannels, self.ch_ref, self.fbin_ref)
        
        # Decimated visibilities without calibration
        vis_uncal = nidata.vis_off_dec 
        
        # Timestamp corresponding to decimated visibilities
        timestamp = nidata.timestamp_dec
        
        # Find gains
        nidata.get_ni_gains() 
        g = nidata.ni_gains
        gains = ni_utils.gains2utvec_tf(g) # Convert to gain array
        
        # Correct decimated visibilities
        vis = vis_uncal/gains
        
        # Calculate dynamic range
        ev = ni_utils.sort_evalues_mag(nidata.ni_evals) # Sort evalues
        dr = abs(ev[:, -1, :]/ev[:, -2, :])
        dr = dr[:, np.newaxis, :]
           
        # Turn vis, gains and dr into MPIArray
        vis = mpidataset.MPIArray.wrap(vis, axis=0, comm=ts.comm)  
        gains = mpidataset.MPIArray.wrap(gains, axis=0, comm=ts.comm) 
        dr = mpidataset.MPIArray.wrap(dr, axis=0, comm=ts.comm)   
        
        # Create NoiseInjTimeStream
        nits = containers.NoiseInjTimeStream.from_base_timestream_attrs(vis, gains, dr, timestamp, ts)  
        nits.redistribute(0)  

        return nits