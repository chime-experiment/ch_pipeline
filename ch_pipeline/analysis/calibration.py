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
    EigenCalibration
    TransitFit
    GainFromTransitFit
    FlagAmplitude
    InterpolateGainOverFrequency
    SiderealCalibration
"""

import numpy as np
from scipy import interpolate
from scipy.constants import c as speed_of_light

from caput import config, pipeline
from caput import mpiarray, mpiutil

from ch_util import tools
from ch_util import ephemeris
from ch_util import ni_utils
from ch_util import cal_utils
from ch_util import fluxcat
from ch_util import rfi

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
    """Calculate gain from largest eigenvector.

    Step through each time/freq pixel, generate a Hermitian matrix,
    perform eigendecomposition, iteratively replacing the diagonal
    elements with a low-rank approximation, and calculate complex gains
    from the largest eigenvector.

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
                                         1.4826 * np.median(np.abs(evals[:-1] -
                                                                   np.median(evals[:-1]))) /
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
    """Linearly interpolates gain solutions in sidereal day.

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
    """Subtract the average of two nearby points from every point in the timestream."""
    if dt is None:
        return ts

    return ts - 0.5 * (np.roll(ts, dt, axis=-1) + np.roll(ts, -dt, axis=-1))


def _adiff(ts, dt):
    """Subtract the average of the first dt points and last dt points from every point."""
    if dt is None:
        return ts

    return ts - 0.5 * (np.mean(ts[..., :dt], axis=-1) +
                       np.mean(ts[..., -dt:], axis=-1))[..., np.newaxis]


def _contiguous_flag(flag, centre=None):
    """Flag everything outside the contiguous unflagged region around centre."""
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
    """Calibration using noise injection.

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
    """Calibration using noise injection.

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


class EigenCalibration(task.SingleTask):
    """Deteremine response of each feed to a point source.

    Extract the feed response from the real-time eigendecomposition
    of the N2 visibility matrix.  Flag frequencies that have low dynamic
    range, orthogonalize the polarizations, fringestop, and reference
    the phases appropriately.

    Attributes
    ----------
    source : str
        Name of the source (same format as `ephemeris.source_dictionary`).
    eigen_ref : int
        Index of the feed that is current phase reference of the eigenvectors.
    phase_ref : list
        Two element list that indicates the chan_id of the feeds to use
        as phase reference for the [Y, X] polarisation.
    med_phase_ref : bool
        Overides `phase_ref`, instead referencing the phase with respect
        to the median value over feeds of a given polarisation.
    window : float
        Fraction of the maximum hour angle considered on source.
    dyn_rng_threshold : float
        Ratio of the second largest eigenvalue on source to the largest eigenvalue
        off source below which frequencies and times will be considered contaminated
        and discarded from further analysis.
    """

    source = config.Property(proptype=str, default='CYG_A')
    eigen_ref = config.Property(proptype=int, default=0)
    phase_ref = config.Property(proptype=list, default=[1152, 1408])
    med_phase_ref = config.Property(proptype=bool, default=False)
    window = config.Property(proptype=float, default=0.75)

    def process(self, data, inputmap):
        """Determine feed response from eigendecomposition.

        Parameters
        ----------
        data : andata.CorrData
            CorrData object that contains the chimecal acquisition datasets,
            specifically vis, weight, erms, evec, and eval.
        inputmap : list of CorrInput's
            List describing the inputs as ordered in data.

        Returns
        -------
        response : containers.SiderealStream
            Response of each feed to the point source.
        """
        from mpi4py import MPI

        # Ensure that we are distributed over frequency
        data.redistribute('freq')

        # Determine local dimensions
        nfreq, neigen, ninput, ntime = data.evec.local_shape

        # Find the local frequencies
        sfreq = data.vis.local_offset[0]
        efreq = sfreq + nfreq

        freq = data.freq[sfreq:efreq]

        # Compute flux of source
        source_obj = fluxcat.FluxCatalog[self.source]
        inv_rt_flux_density = tools.invert_no_zero(np.sqrt(source_obj.predict_flux(freq)))

        # Determine source coordinates
        ttrans = ephemeris.transit_times(source_obj.skyfield, data.time[0])[0]
        csd = int(np.floor(ephemeris.unix_to_csd(ttrans)))

        src_ra, src_dec = ephemeris.object_coords(source_obj.skyfield, date=ttrans, deg=True)

        ra = ephemeris.lsa(data.time)

        ha = ra - src_ra
        ha = ((ha + 180.0) % 360.0) - 180.0
        ha = np.radians(ha)

        window = self.window * np.max(np.abs(ha))
        off_source = np.abs(ha) > window

        itrans = np.argmin(np.abs(ha))

        src_dec = np.radians(src_dec)
        lat = np.radians(ephemeris.CHIMELATITUDE)

        # Dereference datasets
        evec = data.datasets['evec'][:]
        evalue = data.datasets['eval'][:]
        erms = data.datasets['erms'][:]
        vis = data.datasets['vis'][:]
        weight = data.flags['weight'][:]

        # Find inputs that were not included in the eigenvalue decomposition
        eps = 10.0 * np.finfo(evec.dtype).eps
        input_flag = np.all(np.abs(evec[:, 0]) < eps,  axis=(0, 2))
        input_flag = np.logical_not(mpiutil.allreduce(input_flag, op=MPI.LAND, comm=data.comm))

        self.log.info("%d inputs missing from eigenvalue decomposition." %
                      np.sum(~input_flags))

        # Check that we have data for the phase reference
        for ref in self.phase_ref:
            if not input_flag[ref]:
                ValueError("Requested phase reference (%d) "
                           "was not included in decomposition." % ref)

        # Determine x and y pol index
        xfeeds = np.array([idf for idf, inp in enumerate(inputmap) if input_flag[idf] and
                           tools.is_array_x(inp)])
        yfeeds = np.array([idf for idf, inp in enumerate(inputmap) if input_flag[idf] and
                           tools.is_array_y(inp)])

        nfeed = xfeeds.size + yfeeds.size

        pol = [yfeeds,  xfeeds]
        polstr = ['Y', 'X']
        npol = len(pol)

        phase_ref_by_pol = [pol[pp].tolist().index(self.phase_ref[pp]) for pp in range(npol)]

        # Compute distances
        dist = tools.get_feed_positions(inputmap)
        for pp, feeds in enumerate(pol):
            dist[feeds, :] -= dist[self.phase_ref[pp], np.newaxis, :]

        # Determine the number of eigenvalues to include in the orthogonalization
        neigen = min(max(npol, self.neigen), neigen)

        # Calculate dynamic range
        eval0_off_source = np.median(evalue[:, 0, off_source], axis=-1)

        dyn = evalue[:, 1, :] * tools.invert_no_zero(eval0_off_source[:, np.newaxis])

        # Determine frequencies to mask
        not_rfi = ~rfi.frequency_mask(data.freq)
        not_rfi = not_rfi[:, np.newaxis]

        dyn_flg = dyn > self.dyn_rng_threshold

        flag = dyn_flag & not_rfi

        # Calculate base error
        base_err = erms[:, np.newaxis, :]

        # Check for sign flips
        ref_resp = evec[:, 0:neigen, self.eigen_ref, :]

        sign0 = 1.0 - 2.0 * (ref_resp.real < 0.0)

        # Check that we have the correct reference feed
        if np.any(np.abs(ref_resp.imag) > eps):
            ValueError("Reference feed %d is incorrect." % self.eigen_ref)

        # Create output container
        response = containers.SiderealStream(ra=ra, attrs_from=data, axes_from=data,
                                             distributed=data.distributed, comm=data.comm)

        response.attrs['source_name'] = source
        response.attrs['transit_time'] = ttrans
        response.attrs['lsd'] = csd

        reponse.input_flags[:] = input_flag[:, np.newaxis]

        out_vis = response.vis[:]
        out_weight = response.weight[:]

        # Loop over polarizations
        for pp, feeds in enumerate(pol):

            # Create the polarization masking vector
            P = np.zeros((1, ninput, 1), dtype=np.float64)
            P[:, feeds, :] = 1.0

            # Loop over frequencies
            for ff in range(nfreq):

                flg = flag[ff, :]
                ww = weight[ff, feeds, :]

                # Normalize by eigenvalue and correct for pi phase flips in process.
                resp = (sign0[ff, :, np.newaxis, :] * evec[ff, 0:neigen, :, :] *
                        np.sqrt(evalue[ff, 0:neigen, np.newaxis, :]))

                # Rotate to single-pol response
                # Move time to first axis for the matrix multiplication
                invL = tools.invert_no_zero(np.rollaxis(evalue[ff, 0:neigen, np.newaxis, :],
                                                        -1, 0))
                UT = np.rollaxis(resp, -1, 0)
                U = np.swapaxes(UT, -1, -2)

                mu, vp = np.linalg.eigh(np.matmul(UT.conj(), P * U))

                rsign0 = (1.0 - 2.0 * (vp[:, 0, np.newaxis, :].real < 0.0))

                resp = mu[:, np.newaxis, :] * np.matmul(U, rsign0 * vp * invL)

                # Extract feeds of this pol
                # Transpose so that time is back to last axis
                resp = resp[:, feeds, -1].T

                # Compute error on response
                dataflg = (flg[np.newaxis, :] & (np.abs(resp) > 0.0) &
                           (ww > 0.0) & np.isfinite(ww)).astype(np.float32)

                resp_err = (dataflg * base_err[ff, :, :] * np.sqrt(vis[ff, feeds, :].real) *
                            tools.invert_no_zero(np.sqrt(mu[np.newaxis, :, -1])))

                # Reference to specific input
                resp *= np.exp(-1.0J * np.angle(resp[phase_ref_by_pol[pp], np.newaxis, :]))

                # Fringestop
                lmbda = speed_of_light * 1e-6 / freq[ff]

                resp *= tools.fringestop_phase(ha[np.newaxis, :], lat, src_dec,
                                               dist[feeds, 0, np.newaxis] / lmbda,
                                               dist[feeds, 1, np.newaxis] / lmbda)

                # Normalize by source flux
                resp *= inv_rt_flux_density[ff]
                resp_err *= inv_rt_flux_density[ff]

                # If requested, reference phase to the median value
                if self.med_phase_ref:
                    phi0 = np.angle(resp[:, itrans, np.newaxis])
                    resp *= np.exp(-1.0J * phi0)
                    resp *= np.exp(-1.0J * np.median(np.angle(resp), axis=0, keepdims=True))
                    resp *= np.exp(1.0J * phi0)

                out_vis[ff, feeds, :] = resp
                out_weight[ff, feeds, :] = tools.invert_no_zero(resp_err**2)

        return response


class TransitFit(task.SingleTask):
    """Fit model to the transit of a point source.

    Multiple model choices are available.  Default is a nonlinear fit
    of a gaussian in amplitude and a polynomial in phase to the complex data.
    Setting the config property `poly = True` will result instead in an
    iterative weighted least squares fit of a polynomial to log amplitude and phase.
    Type of polynomial can be chosen through `poly_type` config property.

    Attributes
    ----------
    nsigma : float
        Number of standard deviations away from transit to fit.
    poly : bool
        Peform linear fit of a polynomial to log amplitude and phase
        instead of a nonlinear fit to complex valued data.
    poly_type : str
        Type of polynomial.  Either 'standard', 'hermite', or 'chebychev'.
        Relevant if `poly = True`.
    poly_deg_amp : int
        Degree of the polynomial to fit to amplitude.
        Relevant if `poly = True`.
    poly_deg_phi : int
        Degree of the polynomial to fit to phase.
        Relevant if `poly = True`.
    niter : int
        Number of times to update the errors using model amplitude.
        Relevant if `poly = True`.
    nsigma_move : int
        Number of standard deviations away from peak to fit.
        The peak location is updated with each iteration.
        Must be less than `nsigma`.  Relevant if `poly = True`.
    """

    # Parameters relevant for nonlinear gaussian fit or linear polynomial fit
    absolute_sigma = config.Property(proptype=bool, default=False)
    alpha = config.Property(proptype=float, default=0.32)
    nsigma = config.Property(proptype=(lambda x: x if x is None else float(x)), default=0.60)

    # Parameters for polynomial fit
    poly = config.Property(proptype=bool, default=False)
    poly_type = config.Property(proptype=str, default='standard')
    poly_deg_amp = config.Property(proptype=int, default=5)
    poly_deg_phi = config.Property(proptype=int, default=5)
    niter = config.Property(proptype=int, default=5)
    nsigma_move = config.Property(proptype=(lambda x: x if x is None else float(x)), default=0.30)

    def process(self, response, inputmap):
        """Fit model to the point source response for each feed and frequency.

        Parameters
        ----------
        response : containers.SiderealStream
            SiderealStream covering the source transit.  Must contain
            `source_name` and `transit_time` attributes.
        inputmap : list of CorrInput's
            List describing the inputs as ordered in response.

        Returns
        -------
        fit : containers.TransitFitParams
            Parameters of the model fit and their covariance.
        """
        # Ensure that we are distributed over frequency
        response.redistribute('freq')

        # Determine local dimensions
        nfreq, ninput, nra = response.vis.local_shape

        # Find the local frequencies
        sfreq = response.vis.local_offset[0]
        efreq = sfreq + nfreq

        freq = data.freq[sfreq:efreq]

        # Calculate the hour angle using the source and transit time saved to attributes
        source_obj = ephemeris.source_dictionary[response.attrs['source_name']]
        ttrans = response.attrs['transit_time']

        src_ra, src_dec = ephemeris.object_coords(source_obj, date=ttrans, deg=True)

        ha = response.ra[:] - src_ra
        ha = ((ha + 180.0) % 360.0) - 180.0

        # Determine the fit window
        input_flag = np.any(response.input_flags[:], axis=-1)

        xfeeds = np.array([idf for idf, inp in enumerate(inputmap) if input_flag[idf] and
                           tools.is_array_x(inp)])
        yfeeds = np.array([idf for idf, inp in enumerate(inputmap) if input_flag[idf] and
                           tools.is_array_y(inp)])

        pol = {'X': xfeeds, 'Y': yfeeds}

        sigma = np.zeros((nfreq, ninput), dtype=np.float32)
        for pstr, feed in pol.items():
            sigma[:, feed] = cal_utils.guess_fwhm(freq, pol=pstr, dec=src_dec,
                                                  sigma=True, voltage=True)[:, np.newaxis]

        # Dereference datasets
        vis = response.vis[:]
        weight = response.weight[:]
        err = np.sqrt(tools.invert_no_zero(weight))

        # Flag data with zero weight or data that is outside the fit window
        # set by nsigma config parameter
        flag = (weight > 0.0)
        if self.nsigma is not None:
            flag &= ha[np.newaxis, np.newaxis, :] < (self.nsigma * sigma[:, :, np.newaxis])

        # Call the fitting routine
        if poly:
            moving_window = self.nsigma_move and self.nsigma_move * sigma

            param, param_cov, fit_info, chisq, ndof = cal_utils.fit_poly_point_source_transit(
                                                            ha, vis, err, flag=flag,
                                                            poly_type=self.poly_type,
                                                            poly_deg_amp=self.poly_deg_amp,
                                                            poly_deg_phi=self.poly_deg_phi,
                                                            niter=self.niter,
                                                            window=moving_window,
                                                            absolute_sigma=self.absolute_sigma,
                                                            alpha=self.alpha)
        else:
            param, param_cov, fit_info, chisq, ndof = cal_utils.fit_gauss_point_source_transit(
                                                            ha, vis, err, flag=flag,
                                                            fwhm=2.35482 * sigma,
                                                            verbose=False,
                                                            absolute_sigma=self.absolute_sigma,
                                                            alpha=self.alpha)

        # Create an output container
        param_axis = fit_info.pop('parameter_names')
        component_axis = fit_info.pop('component')

        fit = containers.TransitFitParams(param=param_axis, component=component_axis,
                                          axes_from=response, attrs_from=response,
                                          distributed=response.distributed,
                                          comm=response.comm)

        # Transfer fit information to container attributes
        for key, val in fit_info.items():
            fit.attrs[key] = val

        # Save datasets
        fit.parameter[:] = param
        fit.parameter_cov[:] = param_cov
        fit.chisq[:] = chisq
        fit.ndof[:] = ndof

        return fit


class GainFromTransitFit(task.SingleTask):
    """Determine gain by evaluating the best-fit model for the point source transit.

    Attributes
    ----------
    evaluate : str
        Evaluate the model at this location, either 'transit' or 'peak'.
    chisq_per_dof_threshold : float
        Set gain to zero if the chisq per degree of freedom of the fit is less
        than this threshold.
    """

    evaluate = config.Property(proptype=str, default='transit')
    chisq_per_dof_threshold = config.Property(proptype=float, default=20.0)

    def process(self, fit):
        """Determine gain from best-fit model.

        Parameters
        ----------
        fit : containers.TransitFitParams
            Parameters of the model fit and their covariance.
            Must also contain 'model_function', 'model_error_function',
            'model_peak_function', and 'model_kwargs' attributes that
            can be used to evaluate the model.

        Returns
        -------
        gain : containers.StaticGainData
            Gain and uncertainty on the gain.
        """
        from pydoc import locate

        # Distribute over frequency
        fit.redistribute('freq')

        nfreq, ninput, _ = fit.parameters.local_shape

        # Import the function for evaluating the model and keyword arguments
        model_function = locate(fit.attrs['model_function'])
        model_error_function = locate(fit.attrs['model_error_function'])
        model_peak_function = locate(fit.attrs['model_peak_function'])

        model_kwargs = fit.attrs.get('model_kwargs', {})

        # Create output container
        out = containers.StaticGainData(axes_from=fit, attrs_from=fit,
                                        distributed=fit.distributed,
                                        comm=fit.comm)

        # Initialize gains and weights to zeros
        out.gain[:] = np.zeros(out.gain.local_shape, dtype=out.gain.dtype)
        out.weight[:] = np.zeros(out.weight.local_shape, dtype=out.weight.dtype)

        # Determine hour angle of evaluation
        ha = 0.0 if self.evaluate == 'transit' else None

        # Dereference datasets
        param = fit.parameter[:]
        param_cov = fit.parameter_cov[:]
        chisq_per_dof = fit.chisq[:] * tools.invert_no_zero(fit.ndof[:].astype(np.float32))

        gain = out.gain[:]
        weight = out.weight[:]

        # Loop over local frequencies
        for ff in range(nfreq):

            # Loop over inputs
            for ii in range(ninput):

                # Make sure all parameters are finite
                if np.any(~np.isfinite(param[ff, ii, :])):
                    continue

                # Make sure chisq per degree of freedom is below threshold
                if np.any(chisq_per_dof[ff, ii, :] > self.chisq_per_dof_threshold):
                    continue

                if self.evaluate == 'peak':
                    ha = model_peak_function(param[ff, ii, :], **model_kwargs)

                if ha is None:
                    continue

                g = model_function(ha, param[ff, ii], **model_kwargs)
                gerr = model_error_function(ha, param[ff, ii], param_cov[ff, ii], **model_kwargs)

                # Use convention that you multiply by gain to calibrate
                gain[ff, ii] = tools.invert_no_zero(g)
                weight[ff, ii] = tools.invert_no_zero(err**2) * np.abs(g)**4

        return out


class FlagAmplitude(task.SingleTask):
    """Flag feeds and frequencies with outlier gain amplitude.

    Attributes
    ----------
    min_amp_scale_factor : float
        Flag feeds and frequencies where the amplitude of the gain
        is less than `min_amp_scale_factor` times the median amplitude
        over all feeds and frequencies.
    max_amp_scale_factor : float
        Flag feeds and frequencies where the amplitude of the gain
        is greater than `max_amp_scale_factor` times the median amplitude
        over all feeds and frequencies.
    nsigma_outlier : float
        Flag a feed at a particular frequency if the gain amplitude
        is greater than `nsigma_outlier` from the median value over
        all feeds of the same polarisation at that frequency.
    nsigma_med_outlier : float
        Flag a frequency if the median gain amplitude over all feeds of a
        given polarisation is `nsigma_med_outlier` away from the local median.
    window_med_outlier : int
        Number of frequency bins to use to determine the local median for
        the test outlined in the description of `nsigma_med_outlier`.
    threshold_good_freq: float
        If a frequency has less than this fraction of good inputs, then
        it is considered bad and the data for all inputs is flagged.
    threshold_good_input : float
        If an input has less than this fraction of good frequencies, then
        it is considered bad and the data for all frequencies is flagged.
        Note that the fraction is relative to the number of frequencies
        that pass the test described in `threshold_good_freq`.
    """

    min_amp_scale_factor = config.Property(proptype=float, default=0.05)
    max_amp_scale_factor = config.Property(proptype=float, default=20.0)
    nsigma_outlier = config.Property(proptype=float, default=10.0)
    nsigma_med_outlier = config.Property(proptype=float, default=10.0)
    window_med_outlier = config.Property(proptype=int, default=24)
    threshold_good_freq = config.Property(proptype=float, default=0.70)
    threshold_good_input = config.Property(proptype=float, default=0.80)

    def process(self, gain, inputmap):
        """Set weight to zero for feeds and frequencies with outlier gain amplitude.

        Parameters
        ----------
        gain : containers.StaticGain
            Gain derived from point source transit.
        inputmap : list of CorrInput's
            List describing the inputs as ordered in gain.

        Returns
        -------
        gain : containers.StaticGain
            The input gain container with modified weights.
        """
        # Distribute over frequency
        gain.redistribute('freq')

        nfreq, ninput = gain.gain.local_shape

        sfreq = gain.gain.local_offset[0]
        efreq = sfreq + nfreq

        # Dereference datasets
        flag = gain.weight[:] > 0.0
        amp = np.abs(gain.gain[:])

        # Determine x and y pol index
        xfeeds = np.array([idf for idf, inp in enumerate(inputmap) if tools.is_array_x(inp)])
        yfeeds = np.array([idf for idf, inp in enumerate(inputmap) if tools.is_array_y(inp)])
        pol = [yfeeds, xfeeds]
        polstr = ['Y', 'X']

        # Hard cutoffs on the amplitude
        med_amp = np.median(amp[flag])
        min_amp = med_amp * self.min_amp_scale_factor
        max_amp = med_amp * self.max_amp_scale_factor

        flag &= ((amp >= min_amp) & (amp <= max_amp))

        # Flag outliers in amplitude for each frequency
        for pp, feeds in enumerate(pol):

            med_amp_by_pol = np.zeros(nfreq, dtype=np.float32)
            sig_amp_by_pol = np.zeros(nfreq, dtype=np.float32)

            for ff in range(nfreq):

                this_flag = flag[ff, feeds]

                if np.any(this_flag):

                    med, slow, shigh = cal_utils.estimate_directional_scale(
                                            amp[ff, feeds[this_flag]])
                    lower = med - self.nsigma_outlier * slow
                    upper = med + self.nsigma_outlier * shigh

                    flag[ff, feeds] &= ((amp[ff, feeds] >= lower) & (amp[ff, feeds] <= upper))

                    med_amp_by_pol[ff] = med
                    sig_amp_by_pol[ff] = (0.5 * (shigh - slow) /
                                          np.sqrt(np.sum(this_flag, dtype=np.float32)))

            # Flag frequencies that are outliers with respect to local median
            if self.nsigma_med_outlier:

                # Collect med_amp_by_pol for all frequencies on rank 0
                if gain.comm.rank == 0:
                    full_med_amp_by_pol = np.zeros(gain.freq.size, dtype=np.float32)
                else:
                    full_med_amp_by_pol = None

                mpiutil.gather_local(full_med_amp_by_pol, med_amp_by_pol, sfreq,
                                     root=0, comm=gain.comm)

                # Flag outlier frequencies on rank 0
                not_outlier = None
                if gain.comm.rank == 0:

                    med_flag = full_med_amp_by_pol > 0.0

                    not_outlier = cal_utils.flag_outliers(full_med_amp_by_pol, med_flag,
                                                          window=self.window_med_outlier,
                                                          nsigma=self.nsigma_med_outlier)

                # Broadcast outlier frequencies to other ranks
                gain.comm.bcast(not_outlier, root=0)
                gain.comm.Barrier()

                flag[:, feeds] &= not_outlier[sfreq:efreq, np.newaxis]

                self.log.info("Pol %s:  %d frequencies are outliers." %
                              (polstr[pp], np.sum(~not_outlier & med_flag, dtype=np.int)))

        # Determine bad frequencies
        flag_freq = ((np.sum(flag, axis=1, dtype=np.float32) / float(ninput)) >
                     self.threshold_good_freq)

        good_freq = list(sfreq + np.flatnonzero(flag_freq))
        good_freq = mpiutil.allreduce(good_freq, op=MPI.SUM, comm=gain.comm)

        flag &= flag_freq[:, np.newaxis]

        # Determine bad inputs
        flag = mpiarray.MPIArray.wrap(flag, axis=0, comm=gain.comm)
        flag.redistribute(1)

        fraction_good = (np.sum(flag[good_freq, :], axis=0, dtype=np.float32) *
                         tools.invert_no_zero(float(good_freq.size)))
        flag_input = fraction_good > self.threshold_good_input

        flag[:] &= flag_input[np.newaxis, :]

        # Redistribute flags back over frequencies and update container
        flag.redistribute(0)

        gain.weight[:] *= flag.astype(gain.weight.dtype)

        return gain


class InterpolateGainOverFrequency(task.SingleTask):
    """Replace gain at flagged frequencies with interpolated values.

    Uses a gaussian process regression to perform the interpolation
    with a Matern function describing the covariance between frequencies.

    Attributes
    ----------
    interp_scale : float
        Correlation length of the gain with frequency in MHz.
    """

    interp_scale = config.Property(proptype=float, default=30.0)

    def process(self, gain):
        """Interpolate the gain over the frequency axis.

        Parameters
        ----------
        gain : containers.StaticGainData
            Complex gains at single time.

        Returns
        -------
        gain : containers.StaticGainData
            Complex gains with flagged frequencies (`weight = 0.0`)
            replaced with interpolated values and `weight` dataset
            updated to reflect the uncertainty on the interpolation.
        """
        # Redistribute over input
        gain.redistribute('input')

        # Determine flagged frequencies
        flag = gain.weight[:] > 0.0

        # Interpolate the gain at non-flagged frequencies to the flagged frequencies
        interp_gain, interp_weight = cal_utils.interpolate_gain(gain.freq[:], gain.gain[:],
                                                                gain.weight[:], flag=flag,
                                                                length_scale=self.interp_scale)

        # Replace the gain and weight datasets with the interpolated arrays
        # Note that the gain and weight for non-flagged frequencies have not changed
        gain.gain[:] = interp_gain
        gain.weight[:] = interp_weight

        gain.redistribute('freq')

        return gain


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
        xfeeds = np.array([idx for idx, inp in enumerate(inputmap) if (idx in good_input) and
                           tools.is_chime_x(inp)])
        yfeeds = np.array([idx for idx, inp in enumerate(inputmap) if (idx in good_input) and
                           tools.is_chime_y(inp)])

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
        evalue_x, resp[:, xfeeds, :], resp_err[:, xfeeds, :] = solve_gain(vis_slice, feeds=xfeeds,
                                                                          norm=norm[:, xfeeds])
        evalue_y, resp[:, yfeeds, :], resp_err[:, yfeeds, :] = solve_gain(vis_slice, feeds=yfeeds,
                                                                          norm=norm[:, yfeeds])

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
            fit_flag[:, xfeeds, :] = _contiguous_flag(dr_x > self.threshold,
                                                     centre=slice_centre)[:, np.newaxis, :]
            fit_flag[:, yfeeds, :] = _contiguous_flag(dr_y > self.threshold,
                                                     centre=slice_centre)[:, np.newaxis, :]

            # Fit model for the complex response of each feed to the point source
            param, param_cov = cal_utils.fit_point_source_transit(ra_slice, resp, resp_err,
                                                                  flag=fit_flag)

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
