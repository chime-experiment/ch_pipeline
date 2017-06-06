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
    LoadCalibration
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
from ..core.io import _list_or_glob


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
                                         np.median(np.abs(evals[:-1] - np.median(evals[:-1]))) /
                                         (nfeed * evals[-1])**0.5)

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

    def setup(self):

        self.gain = None
        self.weight = None

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
        st, et = idx - slice_width, idx + slice_width + 1

        islice = np.arange(st, et, dtype=np.int)
        vis_slice = sstream.vis[:].view(np.ndarray).take(islice, mode='wrap', axis=-1)
        ra_slice = sstream.ra[:].take(islice, mode='wrap')

        nra = vis_slice.shape[-1]

        slice_centre = np.argmin(np.abs(ra_slice - peak_ra))

        # Determine good inputs
        nfeed = len(inputmap)
        good_input = np.flatnonzero(inputmask.datasets['input_mask'][:])

        # Use input map to figure out which are the X and Y feeds
        xfeeds = np.array([idf for idf, inp in enumerate(inputmap) if (idf in good_input) and tools.is_chime_x(inp)])
        yfeeds = np.array([idf for idf, inp in enumerate(inputmap) if (idf in good_input) and tools.is_chime_y(inp)])

        if mpiutil.rank0:
            print("Performing sidereal calibration with %d/%d good feeds (%d xpol, %d ypol)." %
                  (len(good_input), nfeed, len(xfeeds), len(yfeeds)))

        # If this is the first call to process, then create
        # internal variable to hold most recent gains.
        if self.gain is None:
            self.gain = np.zeros([nfreq, nfeed], np.complex128)
            self.weight = np.zeros(nfreq, np.float64)

        # Only perform calibration for frequencies where the
        # weights at source transit are not set to zero.
        nexp_input = int(0.8 * (xfeeds.size + yfeeds.size))
        nexp_prod = nexp_input * (nexp_input + 1) / 2

        good_freq_local = np.flatnonzero(np.sum(sstream.weight[:, :, idx].view(np.ndarray) > 0,
                                                axis=-1) > nexp_prod)

        if mpiutil.rank0:
            print("Performing sidereal calibration for %d/%d frequencies." % (good_freq_local.size, nfreq))

        # Create container to hold results
        if self.model_fit:
            gain_data = containers.PointSourceTransit(ra=ra_slice, pol_x=xfeeds, pol_y=yfeeds,
                                                      axes_from=sstream, attrs_from=sstream)
        else:
            gain_data = containers.StaticGainData(axes_from=sstream, attrs_from=sstream)

        gain_data.redistribute('freq')

        # Define units
        unit_in = sstream.vis.attrs.get('units', 'rt-correlator-units')
        unit_out = 'rt-Jy'

        # Solve for gains for available frequencies
        nfreq = good_freq_local.size
        if nfreq > 0:

            good_freq = np.arange(sfreq, efreq, dtype=np.int)[good_freq_local]

            vis_slice = vis_slice[good_freq_local]
            freq = freq[good_freq_local]

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

            # Construct the final gain array from the point source response at transit
            self.gain[good_freq_local] = resp[:, :, slice_centre]

            # Construct the dynamic range estimate as the ratio of the first to second
            # largest eigenvalue at the time of transit
            dr_x = evalue_x[:, -1, :] * tools.invert_no_zero(evalue_x[:, -2, :])
            dr_y = evalue_y[:, -1, :] * tools.invert_no_zero(evalue_y[:, -2, :])

            # Combine dynamic range estimates for both polarizations
            self.weight[good_freq_local] = np.minimum(dr_x[:, slice_centre], dr_y[:, slice_centre])

            # If requested, fit a model to the point source transit
            if self.model_fit:

                # Obtain initial estimate of beam FWHM
                initial_fwhm = np.full([nfreq, nfeed], 2.0, dtype=np.float64)
                initial_fwhm[:, xfeeds] = cal_utils.guess_fwhm(freq, pol='X', dec=source._dec, sigma=False)[:, np.newaxis]
                initial_fwhm[:, yfeeds] = cal_utils.guess_fwhm(freq, pol='Y', dec=source._dec, sigma=False)[:, np.newaxis]

                # Only fit ra values above the specified dynamic range threshold
                # that are contiguous about the expected peak position.
                fit_flag = np.zeros([nfreq, nfeed, nra], dtype=np.bool)
                fit_flag[:, xfeeds, :] = contiguous_flag(dr_x > self.threshold, centre=slice_centre)[:, np.newaxis, :]
                fit_flag[:, yfeeds, :] = contiguous_flag(dr_y > self.threshold, centre=slice_centre)[:, np.newaxis, :]

                # Fit model for the complex response of each feed to the point source
                param, param_cov = cal_utils.fit_point_source_transit(ra_slice, resp, resp_err,
                                                                      flag=fit_flag, fwhm=initial_fwhm)

                # Overwrite the initial gain estimates for frequencies/feeds
                # where the model fit was successful
                if self.use_peak:
                    for index in np.ndindex(nfreq, nfeed):
                        if np.all(np.isfinite(param[index])):
                            out_index = (good_freq_local[index[0]], index[1])
                            self.gain[out_index] = param[index][0] * np.exp(1.0j * np.deg2rad(param[index][3]))

                else:
                    for index in np.ndindex(nfreq, nfeed):
                        if np.all(np.isfinite(param[index])):
                            out_index = (good_freq_local[index[0]], index[1])
                            self.gain[out_index] = cal_utils.model_point_source_transit(peak_ra, *param[index])

                # Save results to container
                gain_data.evalue_x[good_freq_local, :, :] = evalue_x
                gain_data.evalue_y[good_freq_local, :, :] = evalue_y
                gain_data.response[good_freq_local, :, :] = resp
                gain_data.response_error[good_freq_local, :, :] = resp_err
                gain_data.flag[good_freq_local, :, :] = fit_flag
                gain_data.parameter[good_freq_local, :, :] = param
                gain_data.parameter_cov[good_freq_local, :, :, :] = param_cov

                # Update units
                gain_data.response.attrs['units'] = unit_in + ' / ' + unit_out
                gain_data.response_error.attrs['units'] = unit_in + ' / ' + unit_out

        # Copy to container all quantities that are common to both
        # StaticGainData and PointSourceTransit containers
        gain_data.add_dataset('weight')

        gain_data.gain[:] = self.gain
        gain_data.weight[:] = self.weight

        # Update units and unit conversion
        gain_data.gain.attrs['units'] = unit_in + ' / ' + unit_out
        gain_data.gain.attrs['converts_units_to'] = 'Jy'

        # Add attribute with the name of the point source
        # that was used for calibration
        gain_data.attrs['source'] = self.source

        # Return gain data
        return gain_data


class LoadCalibration(task.SingleTask):
    """Load sidereal calibrations previously saved to disk
    and extract the appropriate gain for an input stream.

    Attributes
    ----------
    files : list or glob
        Files containing the sidereal calibration.
    interpolate : bool
        If True, interpolate the gains found in files.  If False,
        use the gain found in the file closest in time.
    """

    files = config.Property(proptype=_list_or_glob)
    interpolate = config.Property(proptype=bool, default=False)

    def setup(self, observer=None):
        """Load the sidereal calibration found in files.

        Parameters
        ----------
        observer : caput.time.Observer, optional
            Details of the observer, if not set default to CHIME.
        """

        # Set observer attribute
        self.observer = ephemeris.chime_observer() if observer is None else observer

        # Set gain attribute to None, will be initialized after loading first file.
        self.gain = None

        # Loop over files
        nfiles = len(self.files)
        for ff, filename in enumerate(self.files):

            # Load the static gains from the file.
            cont = containers.GainData.from_file(filename, distributed=True)
            cont.redistribute('freq')

            # If the gain variable does not exist yet, then create with proper size.
            if self.gain is None:

                nfreq = cont.gain.local_shape[0]
                nfeed = cont.gain.local_shape[1]

                self.gain = np.zeros((nfreq, nfeed, nfiles), dtype=cont.gain.dtype)
                self.time = np.zeros(nfiles, dtype=np.float64)

            # Save the static gains to the attribute.
            self.gain[:, :, ff] = cont.gain[:].view(np.ndarray)

            # Extract the time of transit
            lsd = cont.attrs['lsd'] if 'lsd' in cont.attrs else cont.attrs['csd']

            source = ephemeris.source_dictionary[cont.attrs['source']]
            peak_ra = ephemeris.peak_RA(source, deg=True)

            self.time[ff] = self.observer.lsd_to_unix(lsd + peak_ra / 360.0)


    def process(self, sstream):
        """ Return appropriate gain for input stream.

        Parameters
        ----------
        sstream : containers.SiderealStream or andata.CorrData

        Returns
        -------
        outcont : containers.GainData or containers.StaticGainData
            If interpolate attribute is set to True, then a GainData
            container is returned.  If set to False, then a
            StaticGainData container is returned.
        """

        # Extract the time of the stream.
        if hasattr(sstream, 'time'):
            time = sstream.time[:]
        else:
            ra = sstream.index_map['ra'][:]
            lsd = sstream.attrs['lsd'] if 'lsd' in sstream.attrs else sstream.attrs['csd']
            time = self.observer.lsd_to_unix(lsd + ra / 360.0)

        # Determine appropriate gain to apply based on time of stream.
        if self.interpolate:

            # Interpolate the gains from sidereal calibration.
            gain_data = containers.GainData(time=time, axes_from=sstream, attrs_from=sstream)
            gain_data.redistribute('freq')

            gain_data.gain[:] = interp_gains(self.time, self.gain, time, axis=-1)

        else:

            # Use the nearest sidereal calibration.
            gain_data = containers.StaticGainData(axes_from=sstream, attrs_from=sstream)
            gain_data.redistribute('freq')

            index_time = np.argmin(np.abs(np.median(time) - self.time))

            if mpiutil.rank0:
                print "Using calibration from lsd %d" % int(self.observer.unix_to_lsd(self.time[index_time]))

            gain_data.gain[:] = self.gain[:, :, index_time]

        # Return gains
        return gain_data

