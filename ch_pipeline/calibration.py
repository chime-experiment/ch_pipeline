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


def solve_gain(data, feeds=None):
    """
    Steps through each time/freq pixel, generates a Hermitian matrix and
    calculates gains from its largest eigenvector.

    Parameters
    ----------
    data : np.ndarray[nfreq, nprod, ntime]
        Visibility array to be decomposed
    feed_loc : list
        Which feeds to include. If :obj:`None` include all feeds.

    Returns
    -------
    dr : np.ndarray[nfreq, ntime]
        Dynamic range of solution.
    gain : np.ndarray[nfreq, nfeed, ntime]
        Gain solution for each feed, time, and frequency
    """

    # Expand the products to correlation matrices
    corr_data = tools.unpack_product_array(data, axis=1, feeds=feeds)

    nfeed = corr_data.shape[1]

    gain = np.zeros((data.shape[0], nfeed, data.shape[-1]), np.complex128)
    dr = np.zeros((data.shape[0], data.shape[-1]), np.float64)

    for fi in range(data.shape[0]):
        for ti in range(data.shape[-1]):

            # Normalise and solve for eigenvectors
            xc, ach = tools.normalise_correlations(corr_data[fi, :, :, ti])
            evals, evecs = tools.eigh_no_diagonal(xc, niter=5)

            # Construct dynamic range and gain
            dr[fi, ti] = evals[-1] / np.abs(evals[:-1]).max()
            gain[fi, :, ti] = ach * evecs[:, -1] * evals[-1]**0.5

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


def _cdiff(ts, dt):
    # Subtract the average of two nearby points from every point in the timestream
    if dt is None:
        return ts

    return ts - 0.5*(np.roll(ts, dt, axis=-1) + np.roll(ts, -dt, axis=-1))


class PointSourceCalibration(pipeline.TaskBase):
    """Use CasA transits as point source calibrators.

    Attributes
    ----------
    source : str
        Point source to use as calibrator. Only CasA is supported at this time.
    """

    source = config.Property(proptype=str, default='CasA')

    _source_dict = {'CasA': ephemeris.CasA}

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
        xfeeds = [idx for idx, inp in enumerate(inputmap) if tools.is_chime_x(inp)]
        yfeeds = [idx for idx, inp in enumerate(inputmap) if tools.is_chime_y(inp)]

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
                gain_arr_x = solve_gain(vis, feeds=xfeeds)
                gain_arr_y = solve_gain(vis, feeds=yfeeds)

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
        # ind_cal = np.where((times > self.trans_times[0]) & (times < self.trans_times[-1]))[0]

        # Construct the gain matrix at all times (using liner interpolation)
        gain = interp_gains(self.trans_times, self.gain_mat[freq_low:freq_up], times)

        # Create TimeStream
        cts = ts.copy(deep=True)
        cts.add_gain()

        # Apply gains to visibility matrix and copy into cts
        tools.apply_gain(ts.vis, 1.0 / gain, out=cts.vis)

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
        vis = tools.apply_gain(vis_uncal, 1.0 / gain)

        # Calculate dynamic range
        ev = ni_utils.sort_evalues_mag(nidata.ni_evals)  # Sort evalues
        dr = abs(ev[:, -1, :]/ev[:, -2, :])
        dr = dr[:, np.newaxis, :]

        # Turn vis, gains and dr into MPIArray
        vis = mpidataset.MPIArray.wrap(vis, axis=0, comm=ts.comm)
        gain = mpidataset.MPIArray.wrap(gain, axis=0, comm=ts.comm)
        dr = mpidataset.MPIArray.wrap(dr, axis=0, comm=ts.comm)

        # Create NoiseInjTimeStream
        cts = containers.TimeStream(timestamp, ts.freq, vis.global_shape[1],
                                    comm=vis.comm, copy_attrs=ts, gain=True)

        cts.vis[:] = vis
        cts.gain[:] = gain
        cts.gain_dr[:] = dr

        cts.redistribute(0)

        return cts


class StackCalibration(pipeline.TaskBase):
    """Use CasA as a point source calibrator for a sidereal stack.

    Attributes
    ----------
    source : str
        Point source to use as calibrator. Only CasA is supported at this time.
    """

    source = config.Property(proptype=str, default='CasA')

    _source_dict = {'CasA': ephemeris.CasA}

    def setup(self, inputmap):

        self.inputmap = inputmap

    def next(self, sstream):
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
        sstream.redistribute(0)

        # Find the local frequencies
        nfreq = sstream.vis.local_shape[0]
        sfreq = sstream.vis.local_offset[0]
        efreq = sfreq + nfreq

        # Get the local frequency axis
        freq = sstream.freq['centre'][sfreq:efreq]

        # Use input map to figure out which are the X and Y feeds
        xfeeds = [idx for idx, inp in enumerate(self.inputmap) if tools.is_chime_x(inp)]
        yfeeds = [idx for idx, inp in enumerate(self.inputmap) if tools.is_chime_y(inp)]

        nfeed = len(self.inputmap)

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
        st, et = idx - slice_width, idx + slice_width
        vis_slice = sstream.vis[..., st:et].copy()
        ra_slice = sstream.ra[st:et]

        # Fringestop the data
        vis_slice = tools.fringestop_pathfinder(vis_slice, ra_slice, freq, self.inputmap, source)

        # Figure out how many samples is ~ 2 degrees, then subtract nearby values
        diff = int(2.0 / np.median(np.abs(np.diff(sstream.ra))))
        vis_slice = _cdiff(vis_slice, diff)

        # Solve for the gains of each set of polarisations
        dr_x, gain_x = solve_gain(vis_slice, feeds=xfeeds)
        dr_y, gain_y = solve_gain(vis_slice, feeds=yfeeds)

        # Construct the final gain arrays
        gain = np.ones([nfreq, nfeed], np.complex128)
        gain[:, xfeeds] = gain_x[:, :, slice_width]
        gain[:, yfeeds] = gain_y[:, :, slice_width]

        # Create TimeStream
        cstream = sstream.copy(deep=True)

        # Apply gains to visibility matrix and copy into cts
        tools.apply_gain(cstream.vis, 1.0 / gain[:, :, np.newaxis], out=cstream.vis)

        print np.degrees(np.angle(gain))

        # Save gains into cts instance
        cstream._distributed['gain'] = mpidataset.MPIArray.wrap(gain, axis=0, comm=cstream.comm)

        # Ensure distributed over frequency axis
        cstream.redistribute(0)

        return cstream
