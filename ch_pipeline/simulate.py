"""
====================================================
Timestream Simulation (:mod:`~ch_pipeline.simulate`)
====================================================

.. currentmodule:: ch_pipeline.simulate

Tasks for simulating timestream data.

Tasks
=====

.. autosummary::
    :toctree: generated/

    LoadBeamTransfer
    SimulateSidereal
    DayMask
    Unroll
"""

import numpy as np

from cora.util import hputil
from caput import mpiutil, pipeline, config, memh5, mpiarray

from ch_util import tools, ephemeris

from . import containers, task


class SimulateSidereal(task.SingleTask):
    """Create a simulated timestream.

    Attributes
    ----------
    maps : list
        List of map filenames. The sum of these form the simulated sky.
    ndays : float, optional
        Number of days of observation. Setting `ndays = None` (default) uses
        the default stored in the telescope object; `ndays = 0`, assumes the
        observation time is infinite so that the noise is zero. This allows a
        fractional number to account for higher noise.
    seed : integer, optional
        Set the random seed used for the noise simulations. Default (None) is
        to choose a random seed.
    """
    maps = config.Property(proptype=list)
    ndays = config.Property(proptype=float, default=0.0)
    seed = config.Property(proptype=int, default=None)

    done = False

    def setup(self, telescope, beamtransfer):
        """Setup the simulation.

        Parameters
        ----------
        tel : TransitTelescope
            Telescope object.
        bt : BeamTransfer
            Beam Transfer maanger.
        """
        self.beamtransfer = beamtransfer
        self.telescope = telescope

    def process(self):
        """Simulate a SiderealStream

        Returns
        -------
        ss : SiderealStream
            Stacked sidereal day.
        feeds : list of CorrInput
            Description of the feeds simulated.
        """

        if self.done:
            raise pipeline.PipelineStopIteration

        # Read in telescope system
        bt = self.beamtransfer
        tel = self.telescope

        lmax = tel.lmax
        mmax = tel.mmax
        nfreq = tel.nfreq
        npol = tel.num_pol_sky

        lfreq, sfreq, efreq = mpiutil.split_local(nfreq)
        freqmap = None

        lm, sm, em = mpiutil.split_local(mmax + 1)

        # Set the minimum resolution required for the sky.
        ntime = 2*mmax+1

        # If we want to add maps use the m-mode formalism to project a skymap
        # into visibility space


        # Allocate array to store the local frequencies
        row_map = None

        # Read in and sum up the local frequencies of the supplied maps.
        for mapfile in self.maps:

            mc = containers.Map.from_file(mapfile, distributed=True)
            mc.redistribute('freq')
            freqmap = mc.index_map['freq'][:]

            if (tel.frequencies != freqmap['centre']).all():
                raise RuntimeError('Frequencies in map file (%s) do not match those in Beam Transfers.' % mapfile)

            if row_map is None:
                row_map = mc.map[:]
            else:
                row_map += mc.map[:]

        # Calculate the alm's for the local sections
        row_alm = hputil.sphtrans_sky(row_map, lmax=lmax).reshape((lfreq, npol * (lmax+1), lmax+1))

        # Trim off excess m's and wrap into MPIArray
        row_alm = row_alm[..., :(mmax+1)]
        row_alm = mpiarray.MPIArray.wrap(row_alm, axis=0)

        # Perform the transposition to distribute different m's across processes. Neat
        # tip, putting a shorter value for the number of columns, trims the array at
        # the same time
        col_alm = row_alm.redistribute(axis=2)

        # Transpose and reshape to shift m index first.
        col_alm = col_alm.transpose((2, 0, 1)).reshape((None, nfreq, npol, lmax+1))

        # Create storage for visibility data
        vis_data = mpiarray.MPIArray((mmax+1, nfreq, bt.ntel), axis=0, dtype=np.complex128)
        vis_data[:] = 0.0

        # Iterate over m's local to this process and generate the corresponding
        # visibilities
        for mp, mi in vis_data.enumerate(axis=0):
            vis_data[mp] = bt.project_vector_sky_to_telescope(mi, col_alm[mp].view(np.ndarray))

        # Rearrange axes such that frequency is last (as we want to divide
        # frequencies across processors)
        row_vis = vis_data.transpose((0, 2, 1))

        # Parallel transpose to get all m's back onto the same processor
        col_vis_tmp = row_vis.redistribute(axis=2)
        col_vis_tmp = col_vis_tmp.reshape((mmax + 1, 2, tel.npairs, None))

        # Transpose the local section to make the m's the last axis and unwrap the
        # positive and negative m at the same time.
        col_vis = mpiarray.MPIArray((tel.npairs, nfreq, ntime), axis=1, dtype=np.complex128)
        col_vis[:] = 0.0
        col_vis[..., 0] = col_vis_tmp[0, 0]
        for mi in range(1, mmax+1):
            col_vis[...,  mi] = col_vis_tmp[mi, 0]
            col_vis[..., -mi] = col_vis_tmp[mi, 1].conj()  # Conjugate only (not (-1)**m - see paper)

        del col_vis_tmp

        # If we're simulating noise, create a realisation and add it to col_vis
        # if self.ndays > 0:
        #
        #     # Fetch the noise powerspectrum
        #     noise_ps = tel.noisepower(np.arange(tel.npairs)[:, np.newaxis],
        #                               np.arange(sfreq, efreq)[np.newaxis, :],
        #                               ndays=self.ndays).reshape(tel.npairs, lfreq)[:, :, np.newaxis]
        #
        #     # Seed random number generator to give consistent noise
        #     if self.seed is not None:
        #         # Must include rank such that we don't have massive power deficit from correlated noise
        #         np.random.seed(self.seed + mpiutil.rank)
        #
        #     # Create and weight complex noise coefficients
        #     noise_vis = (np.array([1.0, 1.0J]) * np.random.standard_normal(col_vis.shape + (2,))).sum(axis=-1)
        #     noise_vis *= (noise_ps / 2.0)**0.5
        #
        #     # Reset RNG
        #     if self.seed is not None:
        #         np.random.seed()
        #
        #     # Add into main noise sims
        #     col_vis += noise_vis
        #
        #     del noise_vis

        # Fourier transform m-modes back to get final timestream.
        vis_stream = np.fft.ifft(col_vis, axis=-1) * ntime
        vis_stream = vis_stream.reshape((tel.npairs, lfreq, ntime))
        vis_stream = vis_stream.transpose((1, 0, 2)).copy()

        # Try and fetch out the feed index and info from the telescope object.
        try:
            feed_index = tel.feed_index
        except AttributeError:
            feed_index = tel.nfeed

        # Construct container and set visibility data
        sstream = containers.SiderealStream(freq=freqmap, ra=ntime, input=feed_index,
                                            prod=tel.uniquepairs, distributed=True, comm=mc.comm)
        sstream.vis[:] = mpiarray.MPIArray.wrap(vis_stream, axis=0)
        sstream.weight[:] = 1.0

        self.done = True

        return sstream


def _list_of_timeranges(dlist):

    if not isinstance(list, dlist):
        pass


class MakeFullProducts(task.SingleTask):
    """Un-wrap collated products to full triangle.
    """

    def setup(self, telescope):
        """Get a reference to the telescope class.

        Parameters
        ----------
        tel : :class:`drift.core.TransitTelescope`
            Telescope object.
        """
        self.telescope = telescope

    def process(self, sstream):
        """Transform a sidereal stream to having a full product matrix.

        Parameters
        ----------
        sstream : :class:`containers.SiderealStream`
            Sidereal stream to unwrap.

        Returns
        -------
        new_sstream : :class:`containers.SiderealStream`
            Unwrapped sidereal stream.
        """

        sstream.redistribute('freq')

        ninput = len(sstream.input)

        prod = np.array([ (fi, fj) for fi in range(ninput) for fj in range(fi, ninput)])

        new_stream = containers.SiderealStream(prod=prod, axes_from=sstream)
        new_stream.redistribute('freq')
        new_stream.vis[:] = 0.0
        new_stream.weight[:] = 0.0

        # Iterate over all feed pairs and work out which is the correct index in the sidereal stack.
        for pi, (fi, fj) in enumerate(prod):

            unique_ind = self.telescope.feedmap[fi, fj]
            conj = self.telescope.feedconj[fi, fj]

            # unique_ind is less than zero it has masked out
            if unique_ind < 0:
                continue

            prod_stream = sstream.vis[:, unique_ind]
            new_stream.vis[:, pi] = prod_stream.conj() if conj else prod_stream

            new_stream.weight[:, pi] = 1.0

        return new_stream


class MakeCorrDataFiles(task.SingleTask):
    """Generate a series of time streams files from a sidereal stream.

    Parameters
    ----------
    start_time, end_time : float or datetime
        Start and end times of the timestream to simulate. Needs to be either a
        `float` (UNIX time) or a `datetime` objects in UTC.
    integration_time : float, optional
        Integration time in seconds. Takes precedence over `integration_frame_exp`.
    integration_frame_exp: int, optional
        Specify the integration time in frames. The integration time is
        `2**integration_frame_exp * 2.56 us`.
    samples_per_file : int, optional
        Number of samples per file.
    """

    start_time = config.Property(proptype=ephemeris.ensure_unix)
    end_time = config.Property(proptype=ephemeris.ensure_unix)

    integration_time = config.Property(proptype=float, default=None)
    integration_frame_exp = config.Property(proptype=int, default=23)

    samples_per_file = config.Property(proptype=int, default=1024)

    _cur_time = 0.0  # Hold the current file start time

    def setup(self, sstream):
        """Get the sidereal stream to turn into files.

        Parameters
        ----------
        sstream : SiderealStream
        """
        self.sstream = sstream

        # Initialise the current start time
        self._cur_time = self.start_time

    def process(self):
        """Create a timestream file.

        Returns
        -------
        tstream : :class:`andata.CorrData`
            Time stream object.
        """

        from . import regrid

        # First check to see if we have reached the end of the requested time,
        # and if so stop the iteration.
        if self._cur_time > self.end_time:
            raise pipeline.PipelineStopIteration

        # Calculate the integration time
        if self.integration_time is not None:
            int_time = self.integration_time
        else:
            int_time = 2.56e-6 * 2**self.integration_frame_exp

        # Calculate number of samples in file and timestamps
        nsamp = min(int(np.ceil((self.end_time - self._cur_time) / int_time)), self.samples_per_file)
        timestamps = self._cur_time + (np.arange(nsamp) + 1) * int_time  # +1 as timestamps are at the end of each sample

        # Construct the time axis index map
        if self.integration_time is not None:
            time = timestamps
        else:
            _time_dtype = [('fpga_count', np.uint64), ('ctime', np.float64)]
            time = np.zeros(nsamp, _time_dtype)
            time['ctime'] = timestamps
            time['fpga_count'] = (timestamps - self.start_time) / int_time * 2**self.integration_frame_exp

        # Make the timestream container
        tstream = containers.make_empty_corrdata(axes_from=self.sstream, time=time)

        # Make the interpolation array
        ra = ephemeris.transit_RA(tstream.time)
        lza = regrid.lanczos_forward_matrix(self.sstream.ra, ra, periodic=True)
        lza = lza.T.astype(np.complex64)

        # Apply the interpolation matrix to construct the new timestream, place
        # the output directly into the container
        np.dot(self.sstream.vis[:], lza, out=tstream.vis[:])

        # Set the weights array to the maximum value for CHIME
        tstream.weight[:] = 255.0

        # Increment the current start time for the next iteration
        self._cur_time += nsamp * int_time

        # Output the timestream
        return tstream


class ReceiverTemperature(task.SingleTask):
    """Add a basic receiver temperature term into the data.

    This class adds in an uncorrelated, frequency and time independent receiver
    noise temperature to the data. As it is uncorrelated this will only affect
    the auto-correlations. Note this only adds in the offset to the visibility,
    to add the corresponding random fluctuations to subsequently use the
    :class:`SampleNoise` task.

    Attributes
    ----------
    recv_temp : float
        The receiver temperature in Kelvin.
    """
    recv_temp = config.Property(proptype=float, default=0.0)

    def process(self, data):

        # Iterate over the products to find the auto-correlations and add the noise into them
        for pi, prod in enumerate(data.index_map['prod']):

            # Great an auto!
            if prod[0] == prod[1]:
                data.vis[:, pi] += self.recv_temp

        return data


class SampleNoise(task.SingleTask):
    """Add properly distributed noise to a visibility dataset.

    This task draws properly (complex Wishart) distributed samples from an input
    visibility dataset which is assumed to represent the expectation.

    See http://link.springer.com/article/10.1007%2Fs10440-010-9599-x for a
    discussion of the Bartlett decomposition for complex Wishart distributed
    quantities.

    Attributes
    ----------
    sample_frac : float
        Multiplies the number of samples in each measurement. For instance this
        could be a duty cycle if the correlator was not keeping up, or could be
        larger than one if multiple measurements were combined.
    """

    sample_frac = config.Property(proptype=float, default=1.0)

    def process(self, data_exp):
        """Generate a noisy dataset.

        Parameters
        ----------
        data_exp : :class:`containers.SiderealStream` or :class:`andata.CorrData`
            The expected (i.e. noiseless) visibility dataset. Must be the full
            triangle. Make sure you have added an instrumental noise bias if you
            want instrumental noise.

        Returns
        -------
        data_samp : same as :param:`data_exp`
            The sampled (i.e. noisy) visibility dataset.
        """

        from . import _fast_tools

        data_exp.redistribute('freq')

        nfeed = len(data_exp.index_map['input'])

        # Get a reference to the base MPIArray. Attempting to do this in the
        # loop fails if not all ranks enter the loop (as there is an implied MPI
        # Barrier)
        vis_data = data_exp.vis[:]

        # Iterate over frequencies
        for lfi, fi in vis_data.enumerate(0):

            # Get the time and frequency intervals
            dt = data_exp.time[1] - data_exp.time[0]
            df = data_exp.index_map['freq']['width'][fi] * 1e6

            # Calculate the number of samples
            nsamp = int(self.sample_frac * dt * df)

            # Iterate over time
            for lti, ti in vis_data.enumerate(2):

                # Unpack visibilites into full matrix
                vis_utv = vis_data[lfi, :, lti].view(np.ndarray).copy()
                vis_mat = np.zeros((nfeed, nfeed), dtype=vis_utv.dtype)
                _fast_tools._unpack_product_array_fast(vis_utv, vis_mat, np.arange(nfeed), nfeed)

                vis_samp = draw_complex_wishart(vis_mat, nsamp) / nsamp

                vis_data[lfi, :, lti] = vis_samp[np.triu_indices(nfeed)]

        return data_exp


def standard_complex_wishart(m, n):
    """Draw a standard Wishart matrix.

    Parameters
    ----------
    m : integer
        Number of variables (i.e. size of matrix).
    n : integer
        Number of measurements the covariance matrix is estimated from.

    Returns
    -------
    B : np.ndarray[m, m]
    """

    from scipy.stats import gamma

    # Fill in normal variables in the lower triangle
    T = np.zeros((m, m), dtype=np.complex128)
    T[np.tril_indices(m, k=-1)] = (np.random.standard_normal(m * (m-1) / 2) +
                                   1.0J * np.random.standard_normal(m * (m-1) / 2)) / 2**0.5

    # Gamma variables on the diagonal
    for i in range(m):
        T[i, i] = gamma.rvs(n - i)**0.5

    # Return the square to get the Wishart matrix
    return np.dot(T, T.T.conj())


def draw_complex_wishart(C, n):
    """Draw a complex Wishart matrix.

    Parameters
    ----------
    C_exp : np.ndarray[:, :]
        Expected covaraince matrix.

    n : integer
        Number of measurements the covariance matrix is estimated from.

    Returns
    -------
    C_samp : np.ndarray
        Sample covariance matrix.
    """

    import scipy.linalg as la

    # Find Cholesky of C
    L = la.cholesky(C, lower=True)

    # Generate a standard Wishart
    A = standard_complex_wishart(C.shape[0], n)

    # Transform to get the Wishart variable
    return np.dot(L, np.dot(A, L.T.conj()))


class RandomGains(task.SingleTask):
    """Generate a random gaussian realisation of gain timestreams.

    Attributes
    ----------
    corr_length_amp, corr_length_phase : float
        Correlation length for amplitude and phase fluctuations in seconds.
    sigma_amp, sigma_phase : float
        Size of fluctuations for amplitude (fractional), and phase (radians).
    """

    corr_length_amp = config.Property(default=3600.0, proptype=float)
    corr_length_phase = config.Property(default=3600.0, proptype=float)

    sigma_amp = config.Property(default=0.02, proptype=float)
    sigma_phase = config.Property(default=0.1, proptype=float)

    _prev_gain = None

    def process(self, data):
        """Generate a gain timestream for the inputs and times in `data`.

        Parameters
        ----------
        data : :class:`andata.CorrData`
            Generate a timestream for this dataset.

        Returns
        -------
        gain : :class:`containers.GainData`
        """
        import scipy.linalg as la

        data.redistribute('freq')

        time = data.time

        gain_data = containers.GainData(time=time, axes_from=data)
        gain_data.redistribute('freq')

        ntime = len(time)
        nfreq = data.vis.local_shape[0]
        ninput = len(data.index_map['input'])
        nsamp = nfreq * ninput

        def corr_func(zeta, amp):

            def _cf(x):
                dij = x[:, np.newaxis] - x[np.newaxis, :]
                return amp * np.exp(-0.5 * (dij / zeta)**2)

            return _cf

        # Generate the correlation functions
        cf_amp = corr_func(self.corr_length_amp, self.sigma_amp)
        cf_phase = corr_func(self.corr_length_phase, self.sigma_phase)

        if self._prev_gain is None:

            # Generate amplitude and phase fluctuations
            gain_amp = 1.0 + gaussian_realisation(time, cf_amp, nsamp)
            gain_phase = gaussian_realisation(time, cf_phase, nsamp)

        else:

            # Get the previous set of ampliude and phase fluctuations. Note we
            # need to remove one from the amplitude, and unwrap the phase to
            # make it smooth
            prev_time = self._prev_gain.index_map['time'][:]
            prev_amp = (np.abs(self._prev_gain.gain[:].view(np.ndarray)) - 1.0).reshape(nsamp, len(prev_time))
            prev_phase = np.unwrap(np.angle(self._prev_gain.gain[:].view(np.ndarray))).reshape(nsamp, len(prev_time))

            # Generate amplitude and phase fluctuations consistent with the existing data
            gain_amp = 1.0 + constrained_gaussian_realisation(time, cf_amp, nsamp,
                                                              prev_time, prev_amp)
            gain_phase = constrained_gaussian_realisation(time, cf_phase, nsamp,
                                                          prev_time, prev_phase)

        # Combine into an overall gain fluctuation
        gain_comb = gain_amp * np.exp(1.0J * gain_phase)

        # Copy the gain entries into the output container
        gain_comb = mpiarray.MPIArray.wrap(gain_comb.reshape(nfreq, ninput, ntime), axis=0)
        gain_data.gain[:] = gain_comb

        # Keep a reference to these gains around for the next round
        self._prev_gain = gain_data

        return gain_data


def gaussian_realisation(x, corrfunc, n, rcond=1e-12):
    """Generate a Gaussian random field.

    Parameters
    ----------
    x : np.ndarray[npoints] or np.ndarray[npoints, ndim]
        Co-ordinates of points to generate.
    corrfunc : function(x) -> covariance matrix
        Function that take (vectorized) co-ordinates and returns their
        covariance functions.
    n : integer
        Number of realisations to generate.
    rcond : float, optional
        Ignore eigenmodes smaller than `rcond` times the largest eigenvalue.

    Returns
    -------
    y : np.ndarray[n, npoints]
        Realisations of the gaussian field.
    """
    return _realisation(corrfunc(x), n, rcond)


def _realisation(C, n, rcond):
    """Create a realisation of the given covariance matrix. Regularise by
    throwing away small eigenvalues.
    """

    import scipy.linalg as la

    # Find the eigendecomposition, truncate small modes, and use this to
    # construct a matrix projecting from the non-singular space
    evals, evecs = la.eigh(C)
    num = np.sum(evals > rcond * evals[-1])
    R = evecs[:, -num:] * evals[np.newaxis, -num:]**0.5

    # Generate independent gaussian variables
    w = np.random.standard_normal((n, num))

    # Apply projection to get random field
    return np.dot(w, R.T)


def constrained_gaussian_realisation(x, corrfunc, n, x2, y2, rcond=1e-12):
    """Generate a constrained Gaussian random field.

    Given a correlation function generate a Gaussian random field that is
    consistent with an existing set of values :param:`y2` located at
    co-ordinates :param:`x2`.

    Parameters
    ----------
    x : np.ndarray[npoints] or np.ndarray[npoints, ndim]
        Co-ordinates of points to generate.
    corrfunc : function(x) -> covariance matrix
        Function that take (vectorized) co-ordinates and returns their
        covariance functions.
    n : integer
        Number of realisations to generate.
    x2 : np.ndarray[npoints] or np.ndarray[npoints, ndim]
        Co-ordinates of existing points.
    y2 : np.ndarray[npoints] or np.ndarray[n, npoints]
        Existing values of the random field.
    rcond : float, optional
        Ignore eigenmodes smaller than `rcond` times the largest eigenvalue.

    Returns
    -------
    y : np.ndarray[n, npoints]
        Realisations of the gaussian field.
    """
    import scipy.linalg as la

    if (y2.ndim >= 2) and (n != y2.shape[0]):
        raise ValueError('Array y2 of existing data has the wrong shape.')

    # Calculate the covariance matrix for the full dataset
    xc = np.concatenate([x, x2])
    M = corrfunc(xc)

    # Select out the different blocks
    l = len(x)
    A = M[:l, :l]
    B = M[:l, l:]
    C = M[l:, l:]

    # This method tends to be unstable when there are singular modes in the
    # covariance matrix (i.e. modes with zero variance). We can remove these by
    # projecting onto the non-singular modes.

    # Find the eigendecomposition and construct projection matrices onto the
    # non-singular space
    evals_A, evecs_A = la.eigh(A)
    evals_C, evecs_C = la.eigh(C)

    num_A = np.sum(evals_A > rcond * evals_A.max())
    num_C = np.sum(evals_C > rcond * evals_C.max())

    R_A = evecs_A[:, -num_A:]
    R_C = evecs_C[:, -num_C:]

    # Construct the covariance blocks in the reduced basis
    A_r = np.diag(evals_A[-num_A:])
    B_r = np.dot(R_A.T, np.dot(B, R_C))
    Ci_r = np.diag(1.0 / evals_C[-num_C:])

    # Project the existing data into the new basis
    y2_r = np.dot(y2, R_C)

    # Calculate the mean of the new variables
    z_r = np.dot(y2_r, np.dot(Ci_r, B_r.T))

    # Generate fluctuations for the new variables (in the reduced basis)
    Ap_r = A_r - np.dot(B_r, np.dot(Ci_r, B_r.T))
    y_r = _realisation(Ap_r, n, rcond)

    # Project into the original basis for A
    y = np.dot(z_r + y_r, R_A.T)

    return y
