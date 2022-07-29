"""Tasks for calibrating the data."""

import json

import numpy as np
from scipy import interpolate
from scipy.constants import c as speed_of_light

from caput import config, pipeline, memh5
from caput import mpiarray, mpiutil

from ch_util import tools
from ch_util import ephemeris
from ch_util import ni_utils
from ch_util import cal_utils
from ch_util import fluxcat
from ch_util import finder
from ch_util import rfi

from draco.core import task, containers
from draco.util import _fast_tools

from ..core import containers as ccontainers
from ..core.dataquery import _DEFAULT_NODE_SPOOF


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
    nside = int((2 * utmat.shape[axis]) ** 0.5)

    # Check that this nside is correct
    if utmat.shape[axis] != (nside * (nside + 1) // 2):
        msg = (
            "Array length (%i) of axis %i does not correspond upper triangle\
                of square matrix"
            % (utmat.shape[axis], axis)
        )
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
    tfeed = int((2 * data.shape[1]) ** 0.5)

    # If not set, create the list of included feeds (i.e. all feeds)
    feeds = np.array(feeds) if feeds is not None else np.arange(tfeed)
    nfeed = len(feeds)

    # Create empty arrays to store the outputs
    gain = np.zeros((data.shape[0], nfeed, data.shape[-1]), np.complex64)
    gain_error = np.zeros((data.shape[0], nfeed, data.shape[-1]), np.float32)
    evalue = np.zeros((data.shape[0], nfeed, data.shape[-1]), np.float32)

    # Set up normalisation matrix
    if norm is None:
        norm = (_extract_diagonal(data, axis=1).real) ** 0.5
        norm = tools.invert_no_zero(norm)
        norm = norm[:, feeds]

    elif norm.shape != gain.shape:
        ValueError(
            "Input normalization matrix has shape %s, should have shape %s."
            % (norm.shape, gain.shape)
        )

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
            _fast_tools._unpack_product_array_fast(
                data[fi, :, ti].copy(), cd, feeds, tfeed
            )

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
                sign0 = 1.0 - 2.0 * (evecs[0, -1].real < 0.0)
                gain[fi, :, ti] = (
                    sign0 * inv_norm[fi, :, ti] * evecs[:, -1] * evals[-1] ** 0.5
                )

                gain_error[fi, :, ti] = (
                    inv_norm[fi, :, ti]
                    * 1.4826
                    * np.median(np.abs(evals[:-1] - np.median(evals[:-1])))
                    / evals[-1] ** 0.5
                )

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

    Parameters
    ----------
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
    f = interpolate.interp1d(
        trans_times, gain_mat, kind="linear", axis=axis, bounds_error=False
    )

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

    return (
        ts
        - 0.5
        * (np.mean(ts[..., :dt], axis=-1) + np.mean(ts[..., -dt:], axis=-1))[
            ..., np.newaxis
        ]
    )


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
            ni_params = {"ni_period": self.period, "ni_on_bins": self.phase}

        folded_ts = ni_utils.process_synced_data(
            ts, ni_params=ni_params, only_off=self.only_off
        )

        return folded_ts


class NoiseInjectionCalibration(task.MPILoggedTask):
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
        self.log.debug("Using input=%i as noise channel", self.ch_ref)

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

        ts.redistribute("freq")

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
        cts = containers.TimeStream(
            timestamp,
            ts.freq,
            vis.global_shape[1],
            comm=vis.comm,
            copy_attrs=ts,
            gain=True,
        )

        cts.vis[:] = vis
        cts.gain[:] = gain
        cts.gain_dr[:] = dr
        cts.common["input"] = ts.input

        cts.redistribute(0)

        return cts


class GatedNoiseCalibration(task.SingleTask):
    """Calibration using noise injection.

    Attributes
    ----------
    norm : ['gated', 'off', 'identity']
        Specify what to use to normalise the matrix.
    """

    norm = config.Property(proptype=str, default="off")

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
        ts.redistribute("freq")

        # Figure out which input channel is the noise source (used as gain reference)
        noise_channel = tools.get_noise_channel(inputmap)

        # Get the norm matrix
        if self.norm == "gated":
            norm_array = _extract_diagonal(ts.datasets["gated_vis0"][:]) ** 0.5
            norm_array = tools.invert_no_zero(norm_array)
        elif self.norm == "off":
            norm_array = _extract_diagonal(ts.vis[:]) ** 0.5
            norm_array = tools.invert_no_zero(norm_array)

            # Extract the points with zero weight (these will get zero norm)
            w = _extract_diagonal(ts.weight[:]) > 0
            w[:, noise_channel] = True  # Make sure we keep the noise channel though!

            norm_array *= w

        elif self.norm == "none":
            norm_array = np.ones(
                [ts.vis[:].shape[0], ts.ninput, ts.ntime], dtype=np.uint8
            )
        else:
            raise RuntimeError("Value of norm not recognised.")

        # Take a view now to avoid some MPI issues
        gate_view = ts.datasets["gated_vis0"][:].view(np.ndarray)
        norm_view = norm_array[:].view(np.ndarray)

        # Find gains with the eigenvalue method
        evalue, gain = solve_gain(gate_view, norm=norm_view)[0:2]
        dr = evalue[:, -1, :] * tools.invert_no_zero(evalue[:, -2, :])

        # Normalise by the noise source channel
        gain *= tools.invert_no_zero(gain[:, np.newaxis, noise_channel, :])
        gain = np.nan_to_num(gain)

        # Create container from gains
        gain_data = containers.GainData(axes_from=ts)
        gain_data.add_dataset("weight")

        # Copy data into container
        gain_data.gain[:] = gain
        gain_data.weight[:] = dr

        return gain_data


class DetermineSourceTransit(task.SingleTask):
    """Determine the sources that are transiting within time range covered by container.

    Attributes
    ----------
    source_list : list of str
        List of source names to consider.  If not specified, all sources
        contained in `ch_util.ephemeris.source_dictionary` will be considered.
    freq : float
        Frequency in MHz.  Sort the sources by the flux at this frequency.
    require_transit: bool
        If this is True and a source transit is not found in the container,
        then the task will return None.
    """

    source_list = config.Property(proptype=list, default=[])
    freq = config.Property(proptype=float, default=600.0)
    require_transit = config.Property(proptype=bool, default=True)

    def setup(self):
        """Set list of sources, sorted by flux in descending order."""
        self.source_list = reversed(
            sorted(
                self.source_list or ephemeris.source_dictionary.keys(),
                key=lambda src: fluxcat.FluxCatalog[src].predict_flux(self.freq),
            )
        )

    def process(self, sstream):
        """Add attributes to container describing source transit contained within.

        Parameters
        ----------
        sstream : containers.SiderealStream, containers.TimeStream, or equivalent
            Container covering the source transit.

        Returns
        -------
        sstream : containers.SiderealStream, containers.TimeStream, or equivalent
            Container covering the source transit, now with `source_name` and
            `transit_time` attributes.
        """
        # Determine the time covered by input container
        if "time" in sstream.index_map:
            timestamp = sstream.time
        else:
            lsd = sstream.attrs.get("lsd", sstream.attrs.get("csd"))
            timestamp = ephemeris.csd_to_unix(lsd + sstream.ra / 360.0)

        # Loop over sources and check if there is a transit within time range
        # covered by container.  If so, then add attributes describing that source
        # and break from the loop.
        contains_transit = False
        for src in self.source_list:
            transit_time = ephemeris.transit_times(
                ephemeris.source_dictionary[src], timestamp[0], timestamp[-1]
            )
            if transit_time.size > 0:
                self.log.info(
                    "Data stream contains %s transit on LSD %d."
                    % (src, ephemeris.csd(transit_time[0]))
                )
                sstream.attrs["source_name"] = src
                sstream.attrs["transit_time"] = transit_time[0]
                contains_transit = True
                break

        if contains_transit or not self.require_transit:
            return sstream
        else:
            return None


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
    neigen : int
        Number of eigenvalues to include in the orthogonalization.
    max_hour_angle : float
        The maximum hour angle in degrees to consider in the analysis.
        Hour angles between [window * max_hour_angle, max_hour_angle] will
        be used for the determination of the off source eigenvalue.
    window : float
        Fraction of the maximum hour angle considered still on source.
    dyn_rng_threshold : float
        Ratio of the second largest eigenvalue on source to the largest eigenvalue
        off source below which frequencies and times will be considered contaminated
        and discarded from further analysis.
    telescope_rotation : float
        Rotation of the telescope from true north in degrees.  A positive rotation is
        anti-clockwise when looking down at the telescope from the sky.
    """

    source = config.Property(default=None)
    eigen_ref = config.Property(proptype=int, default=0)
    phase_ref = config.Property(proptype=list, default=[1152, 1408])
    med_phase_ref = config.Property(proptype=bool, default=False)
    neigen = config.Property(proptype=int, default=2)
    max_hour_angle = config.Property(proptype=float, default=10.0)
    window = config.Property(proptype=float, default=0.75)
    dyn_rng_threshold = config.Property(proptype=float, default=3.0)
    telescope_rotation = config.Property(proptype=float, default=tools._CHIME_ROT)

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
        data.redistribute("freq")

        # Determine local dimensions
        nfreq, neigen, ninput, ntime = data.datasets["evec"].local_shape

        # Find the local frequencies
        sfreq = data.vis.local_offset[0]
        efreq = sfreq + nfreq

        freq = data.freq[sfreq:efreq]

        # Determine source name.  If not provided as config property, then check data attributes.
        source_name = self.source or data.attrs.get("source_name", None)
        if source_name is None:
            raise ValueError(
                "The source name must be specified as a configuration property "
                "or added to input container attributes by an earlier task."
            )

        # Compute flux of source
        source_obj = fluxcat.FluxCatalog[source_name]
        inv_rt_flux_density = tools.invert_no_zero(
            np.sqrt(source_obj.predict_flux(freq))
        )

        # Determine source coordinates
        ttrans = ephemeris.transit_times(source_obj.skyfield, data.time[0])[0]
        csd = int(np.floor(ephemeris.unix_to_csd(ttrans)))

        src_ra, src_dec = ephemeris.object_coords(
            source_obj.skyfield, date=ttrans, deg=True
        )

        ra = ephemeris.lsa(data.time)

        ha = ra - src_ra
        ha = ((ha + 180.0) % 360.0) - 180.0
        ha = np.radians(ha)

        max_ha_off_source = np.minimum(
            np.max(np.abs(ha)), np.radians(self.max_hour_angle)
        )
        min_ha_off_source = self.window * max_ha_off_source
        off_source = (np.abs(ha) >= min_ha_off_source) & (
            np.abs(ha) <= max_ha_off_source
        )

        itrans = np.argmin(np.abs(ha))

        src_dec = np.radians(src_dec)
        lat = np.radians(ephemeris.CHIMELATITUDE)

        # Dereference datasets
        evec = data.datasets["evec"][:].view(np.ndarray)
        evalue = data.datasets["eval"][:].view(np.ndarray)
        erms = data.datasets["erms"][:].view(np.ndarray)
        vis = data.datasets["vis"][:].view(np.ndarray)
        weight = data.flags["vis_weight"][:].view(np.ndarray)

        # Check for negative autocorrelations (bug observed in older data)
        negative_auto = vis.real < 0.0
        if np.any(negative_auto):
            vis[negative_auto] = 0.0 + 0.0j
            weight[negative_auto] = 0.0

        # Find inputs that were not included in the eigenvalue decomposition
        eps = 10.0 * np.finfo(evec.dtype).eps
        evec_all_zero = np.all(np.abs(evec[:, 0]) < eps, axis=(0, 2))

        input_flags = np.zeros(ninput, dtype=np.bool)
        for ii in range(ninput):
            input_flags[ii] = np.logical_not(
                mpiutil.allreduce(evec_all_zero[ii], op=MPI.LAND, comm=data.comm)
            )

        self.log.info(
            "%d inputs missing from eigenvalue decomposition." % np.sum(~input_flags)
        )

        # Check that we have data for the phase reference
        for ref in self.phase_ref:
            if not input_flags[ref]:
                ValueError(
                    "Requested phase reference (%d) "
                    "was not included in decomposition." % ref
                )

        # Update input_flags to include feeds not present in database
        for idf, inp in enumerate(inputmap):
            if not tools.is_chime(inp):
                input_flags[idf] = False

        # Determine x and y pol index
        xfeeds = np.array(
            [
                idf
                for idf, inp in enumerate(inputmap)
                if input_flags[idf] and tools.is_array_x(inp)
            ]
        )
        yfeeds = np.array(
            [
                idf
                for idf, inp in enumerate(inputmap)
                if input_flags[idf] and tools.is_array_y(inp)
            ]
        )

        nfeed = xfeeds.size + yfeeds.size

        pol = [yfeeds, xfeeds]
        polstr = ["Y", "X"]
        npol = len(pol)

        phase_ref_by_pol = [
            pol[pp].tolist().index(self.phase_ref[pp]) for pp in range(npol)
        ]

        # Create new product map for the output container that has `input_b` set to
        # the phase reference feed.  Necessary to apply the timing correction later.
        prod = np.copy(data.prod)
        for pp, feeds in enumerate(pol):
            prod["input_b"][feeds] = self.phase_ref[pp]

        # Compute distances
        tools.change_chime_location(rotation=self.telescope_rotation)
        dist = tools.get_feed_positions(inputmap)
        for pp, feeds in enumerate(pol):
            dist[feeds, :] -= dist[self.phase_ref[pp], np.newaxis, :]

        # Check for feeds that do not have a valid distance (feedpos are set to nan)
        no_distance = np.flatnonzero(np.any(np.isnan(dist), axis=1))
        if (no_distance.size > 0) and np.any(input_flags[no_distance]):
            raise RuntimeError(
                "Do not have positions for feeds: %s"
                % str(no_distance[input_flags[no_distance]])
            )

        # Determine the number of eigenvalues to include in the orthogonalization
        neigen = min(max(npol, self.neigen), neigen)

        # Calculate dynamic range
        eval0_off_source = np.median(evalue[:, 0, off_source], axis=-1)

        dyn = evalue[:, 1, :] * tools.invert_no_zero(eval0_off_source[:, np.newaxis])

        # Determine frequencies and times to mask
        not_rfi = ~rfi.frequency_mask(freq)
        not_rfi = not_rfi[:, np.newaxis]

        self.log.info(
            "Using a dynamic range threshold of %0.2f." % self.dyn_rng_threshold
        )
        dyn_flag = dyn > self.dyn_rng_threshold

        converged = erms > 0.0

        flag = converged & dyn_flag & not_rfi

        # Calculate base error
        base_err = erms[:, np.newaxis, :]

        # Check for sign flips
        ref_resp = evec[:, 0:neigen, self.eigen_ref, :]

        sign0 = 1.0 - 2.0 * (ref_resp.real < 0.0)

        # Check that we have the correct reference feed
        if np.any(np.abs(ref_resp.imag) > eps):
            ValueError("Reference feed %d is incorrect." % self.eigen_ref)

        # Create output container
        response = containers.SiderealStream(
            ra=ra,
            prod=prod,
            stack=None,
            attrs_from=data,
            axes_from=data,
            distributed=data.distributed,
            comm=data.comm,
        )

        response.input_flags[:] = input_flags[:, np.newaxis]

        # Create attributes identifying the transit
        response.attrs["source_name"] = source_name
        response.attrs["transit_time"] = ttrans
        response.attrs["lsd"] = csd
        response.attrs["tag"] = "%s_lsd_%d" % (source_name.lower(), csd)

        # Add an attribute that indicates if the transit occured during the daytime
        is_daytime = 0
        solar_rise = ephemeris.solar_rising(ttrans - 86400.0)
        for sr in solar_rise:
            ss = ephemeris.solar_setting(sr)[0]
            if (ttrans >= sr) and (ttrans <= ss):
                is_daytime = 1
                break
        response.attrs["daytime_transit"] = is_daytime

        # Dereference the output datasets
        out_vis = response.vis[:]
        out_weight = response.weight[:]

        # Loop over polarizations
        for pp, feeds in enumerate(pol):

            # Create the polarization masking vector
            P = np.zeros((1, ninput, 1), dtype=np.float64)
            P[:, feeds, :] = 1.0

            # Loop over frequencies
            for ff in range(nfreq):

                ww = weight[ff, feeds, :]

                # Normalize by eigenvalue and correct for pi phase flips in process.
                resp = (
                    sign0[ff, :, np.newaxis, :]
                    * evec[ff, 0:neigen, :, :]
                    * np.sqrt(evalue[ff, 0:neigen, np.newaxis, :])
                )

                # Rotate to single-pol response
                # Move time to first axis for the matrix multiplication
                invL = tools.invert_no_zero(
                    np.rollaxis(evalue[ff, 0:neigen, np.newaxis, :], -1, 0)
                )
                UT = np.rollaxis(resp, -1, 0)
                U = np.swapaxes(UT, -1, -2)

                mu, vp = np.linalg.eigh(np.matmul(UT.conj(), P * U))

                rsign0 = 1.0 - 2.0 * (vp[:, 0, np.newaxis, :].real < 0.0)

                resp = mu[:, np.newaxis, :] * np.matmul(U, rsign0 * vp * invL)

                # Extract feeds of this pol
                # Transpose so that time is back to last axis
                resp = resp[:, feeds, -1].T

                # Compute error on response
                dataflg = (
                    flag[ff, np.newaxis, :]
                    & (np.abs(resp) > 0.0)
                    & (ww > 0.0)
                    & np.isfinite(ww)
                ).astype(np.float32)

                resp_err = (
                    dataflg
                    * base_err[ff, :, :]
                    * np.sqrt(vis[ff, feeds, :].real)
                    * tools.invert_no_zero(np.sqrt(mu[np.newaxis, :, -1]))
                )

                # Reference to specific input
                resp *= np.exp(
                    -1.0j * np.angle(resp[phase_ref_by_pol[pp], np.newaxis, :])
                )

                # Fringestop
                lmbda = speed_of_light * 1e-6 / freq[ff]

                resp *= tools.fringestop_phase(
                    ha[np.newaxis, :],
                    lat,
                    src_dec,
                    dist[feeds, 0, np.newaxis] / lmbda,
                    dist[feeds, 1, np.newaxis] / lmbda,
                )

                # Normalize by source flux
                resp *= inv_rt_flux_density[ff]
                resp_err *= inv_rt_flux_density[ff]

                # If requested, reference phase to the median value
                if self.med_phase_ref:
                    phi0 = np.angle(resp[:, itrans, np.newaxis])
                    resp *= np.exp(-1.0j * phi0)
                    resp *= np.exp(
                        -1.0j * np.median(np.angle(resp), axis=0, keepdims=True)
                    )
                    resp *= np.exp(1.0j * phi0)

                out_vis[ff, feeds, :] = resp
                out_weight[ff, feeds, :] = tools.invert_no_zero(resp_err**2)

        return response


class TransitFit(task.SingleTask):
    """Fit model to the transit of a point source.

    Multiple model choices are available and can be specified through the `model`
    config property.  Default is `gauss_amp_poly_phase`, a nonlinear fit
    of a gaussian in amplitude and a polynomial in phase to the complex data.
    There is also `poly_log_amp_poly_phase`, an iterative weighted least squares
    fit of a polynomial to log amplitude and phase.  The type of polynomial can be
    chosen through the `poly_type`, `poly_deg_amp`, and `poly_deg_phi` config properties.

    Attributes
    ----------
    model : str
        Name of the model to fit.  One of 'gauss_amp_poly_phase' or
        'poly_log_amp_poly_phase'.
    nsigma : float
        Number of standard deviations away from transit to fit.
    absolute_sigma : bool
        Set to True if the errors provided are absolute.  Set to False if
        the errors provided are relative, in which case the parameter covariance
        will be scaled by the chi-squared per degree-of-freedom.
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
    moving_window : int
        Number of standard deviations away from peak to fit.
        The peak location is updated with each iteration.
        Must be less than `nsigma`.  Relevant if `poly = True`.
    """

    model = config.enum(
        ["gauss_amp_poly_phase", "poly_log_amp_poly_phase"],
        default="gauss_amp_poly_phase",
    )
    nsigma = config.Property(
        proptype=(lambda x: x if x is None else float(x)), default=0.60
    )
    absolute_sigma = config.Property(proptype=bool, default=False)
    poly_type = config.Property(proptype=str, default="standard")
    poly_deg_amp = config.Property(proptype=int, default=5)
    poly_deg_phi = config.Property(proptype=int, default=5)
    niter = config.Property(proptype=int, default=5)
    moving_window = config.Property(
        proptype=(lambda x: x if x is None else float(x)), default=0.30
    )

    def setup(self):
        """Define model to fit to transit."""
        self.fit_kwargs = {"absolute_sigma": self.absolute_sigma}

        if self.model == "gauss_amp_poly_phase":
            self.ModelClass = cal_utils.FitGaussAmpPolyPhase
            self.model_kwargs = {
                "poly_type": self.poly_type,
                "poly_deg_phi": self.poly_deg_phi,
            }

        elif self.model == "poly_log_amp_poly_phase":
            self.ModelClass = cal_utils.FitPolyLogAmpPolyPhase
            self.model_kwargs = {
                "poly_type": self.poly_type,
                "poly_deg_amp": self.poly_deg_amp,
                "poly_deg_phi": self.poly_deg_phi,
            }
            self.fit_kwargs.update(
                {"niter": self.niter, "moving_window": self.moving_window}
            )

        else:
            raise ValueError(
                "Do not recognize model %s.  Options are %s and %s."
                % (self.model, "gauss_amp_poly_phase", "poly_log_amp_poly_phase")
            )

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
        fit : ccontainers.TransitFitParams
            Parameters of the model fit and their covariance.
        """
        # Ensure that we are distributed over frequency
        response.redistribute("freq")

        # Determine local dimensions
        nfreq, ninput, nra = response.vis.local_shape

        # Find the local frequencies
        sfreq = response.vis.local_offset[0]
        efreq = sfreq + nfreq

        freq = response.freq[sfreq:efreq]

        # Calculate the hour angle using the source and transit time saved to attributes
        source_obj = ephemeris.source_dictionary[response.attrs["source_name"]]
        ttrans = response.attrs["transit_time"]

        src_ra, src_dec = ephemeris.object_coords(source_obj, date=ttrans, deg=True)

        ha = response.ra[:] - src_ra
        ha = ((ha + 180.0) % 360.0) - 180.0

        # Determine the fit window
        input_flags = np.any(response.input_flags[:], axis=-1)

        xfeeds = np.array(
            [
                idf
                for idf, inp in enumerate(inputmap)
                if input_flags[idf] and tools.is_array_x(inp)
            ]
        )
        yfeeds = np.array(
            [
                idf
                for idf, inp in enumerate(inputmap)
                if input_flags[idf] and tools.is_array_y(inp)
            ]
        )

        pol = {"X": xfeeds, "Y": yfeeds}

        sigma = np.zeros((nfreq, ninput), dtype=np.float32)
        for pstr, feed in pol.items():
            sigma[:, feed] = cal_utils.guess_fwhm(
                freq, pol=pstr, dec=np.radians(src_dec), sigma=True, voltage=True
            )[:, np.newaxis]

        # Dereference datasets
        vis = response.vis[:].view(np.ndarray)
        weight = response.weight[:].view(np.ndarray)
        err = np.sqrt(tools.invert_no_zero(weight))

        # Flag data that is outside the fit window set by nsigma config parameter
        if self.nsigma is not None:
            err *= (
                np.abs(ha[np.newaxis, np.newaxis, :])
                <= (self.nsigma * sigma[:, :, np.newaxis])
            ).astype(err.dtype)

        # Instantiate the model fitter
        model = self.ModelClass(**self.model_kwargs)

        # Fit the model
        model.fit(ha, vis, err, width=sigma, **self.fit_kwargs)

        # Create an output container
        fit = ccontainers.TransitFitParams(
            param=model.parameter_names,
            component=model.component,
            axes_from=response,
            attrs_from=response,
            distributed=response.distributed,
            comm=response.comm,
        )

        fit.add_dataset("chisq")
        fit.add_dataset("ndof")

        # Transfer fit information to container attributes
        fit.attrs["model_kwargs"] = json.dumps(model.model_kwargs)
        fit.attrs["model_class"] = ".".join(
            [getattr(self.ModelClass, key) for key in ["__module__", "__name__"]]
        )

        # Save datasets
        fit.parameter[:] = model.param[:]
        fit.parameter_cov[:] = model.param_cov[:]
        fit.chisq[:] = model.chisq[:]
        fit.ndof[:] = model.ndof[:]

        return fit


class GainFromTransitFit(task.SingleTask):
    """Determine gain by evaluating the best-fit model for the point source transit.

    Attributes
    ----------
    evaluate : str
        Evaluate the model at this location, either 'transit' or 'peak'.
    chisq_per_dof_threshold : float
        Set gain and weight to zero if the chisq per degree of freedom
        of the fit is less than this threshold.
    alpha : float
        Use confidence level 1 - alpha for the uncertainty on the gain.
    """

    evaluate = config.enum(["transit", "peak"], default="transit")
    chisq_per_dof_threshold = config.Property(proptype=float, default=20.0)
    alpha = config.Property(proptype=float, default=0.32)

    def process(self, fit):
        """Determine gain from best-fit model.

        Parameters
        ----------
        fit : ccontainers.TransitFitParams
            Parameters of the model fit and their covariance.
            Must also contain 'model_class' and 'model_kwargs'
            attributes that can be used to evaluate the model.

        Returns
        -------
        gain : containers.StaticGainData
            Gain and uncertainty on the gain.
        """
        from pydoc import locate

        # Distribute over frequency
        fit.redistribute("freq")

        nfreq, ninput, _ = fit.parameter.local_shape

        # Import the function for evaluating the model and keyword arguments
        ModelClass = locate(fit.attrs["model_class"])
        model_kwargs = json.loads(fit.attrs["model_kwargs"])

        # Create output container
        out = containers.StaticGainData(
            axes_from=fit, attrs_from=fit, distributed=fit.distributed, comm=fit.comm
        )
        out.add_dataset("weight")

        # Dereference datasets
        param = fit.parameter[:].view(np.ndarray)
        param_cov = fit.parameter_cov[:].view(np.ndarray)
        chisq = fit.chisq[:].view(np.ndarray)
        ndof = fit.ndof[:].view(np.ndarray)

        chisq_per_dof = chisq * tools.invert_no_zero(ndof.astype(np.float32))

        gain = out.gain[:]
        weight = out.weight[:]

        # Instantiate the model object
        model = ModelClass(
            param=param, param_cov=param_cov, chisq=chisq, ndof=ndof, **model_kwargs
        )

        # Suppress numpy floating errors
        with np.errstate(all="ignore"):

            # Determine hour angle of evaluation
            if self.evaluate == "peak":
                ha = model.peak()
                elementwise = True
            else:
                ha = 0.0
                elementwise = False

            # Predict model and uncertainty at desired hour angle
            g = model.predict(ha, elementwise=elementwise)

            gerr = model.uncertainty(ha, alpha=self.alpha, elementwise=elementwise)

            # Use convention that you multiply by gain to calibrate
            gain[:] = tools.invert_no_zero(g)
            weight[:] = tools.invert_no_zero(np.abs(gerr) ** 2) * np.abs(g) ** 4

            # Can occassionally get Infs when evaluating fits to anomalous data.
            # Replace with zeros. Also zero data where the chi-squared per
            # degree of freedom is greater than threshold.
            not_valid = ~(
                np.isfinite(gain)
                & np.isfinite(weight)
                & np.all(chisq_per_dof <= self.chisq_per_dof_threshold, axis=-1)
            )

            if np.any(not_valid):
                gain[not_valid] = 0.0 + 0.0j
                weight[not_valid] = 0.0

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
    valid_gains_frac_good_freq : float
        If the fraction of frequencies that remain after flagging is less than
        this value, then the task will return None and the processing of the
        sidereal day will not proceed further.
    """

    min_amp_scale_factor = config.Property(proptype=float, default=0.05)
    max_amp_scale_factor = config.Property(proptype=float, default=20.0)
    nsigma_outlier = config.Property(proptype=float, default=10.0)
    nsigma_med_outlier = config.Property(proptype=float, default=10.0)
    window_med_outlier = config.Property(proptype=int, default=24)
    threshold_good_freq = config.Property(proptype=float, default=0.70)
    threshold_good_input = config.Property(proptype=float, default=0.80)
    valid_gains_frac_good_freq = config.Property(proptype=float, default=0.0)

    def process(self, gain, inputmap):
        """Set weight to zero for feeds and frequencies with outlier gain amplitude.

        Parameters
        ----------
        gain : containers.StaticGainData
            Gain derived from point source transit.
        inputmap : list of CorrInput's
            List describing the inputs as ordered in gain.

        Returns
        -------
        gain : containers.StaticGainData
            The input gain container with modified weights.
        """
        from mpi4py import MPI

        # Distribute over frequency
        gain.redistribute("freq")

        nfreq, ninput = gain.gain.local_shape

        sfreq = gain.gain.local_offset[0]
        efreq = sfreq + nfreq

        # Dereference datasets
        flag = gain.weight[:].view(np.ndarray) > 0.0
        amp = np.abs(gain.gain[:].view(np.ndarray))

        # Determine x and y pol index
        xfeeds = np.array(
            [idf for idf, inp in enumerate(inputmap) if tools.is_array_x(inp)]
        )
        yfeeds = np.array(
            [idf for idf, inp in enumerate(inputmap) if tools.is_array_y(inp)]
        )
        pol = [yfeeds, xfeeds]
        polstr = ["Y", "X"]

        # Hard cutoffs on the amplitude
        med_amp = np.median(amp[flag])
        min_amp = med_amp * self.min_amp_scale_factor
        max_amp = med_amp * self.max_amp_scale_factor

        flag &= (amp >= min_amp) & (amp <= max_amp)

        # Flag outliers in amplitude for each frequency
        for pp, feeds in enumerate(pol):

            med_amp_by_pol = np.zeros(nfreq, dtype=np.float32)
            sig_amp_by_pol = np.zeros(nfreq, dtype=np.float32)

            for ff in range(nfreq):

                this_flag = flag[ff, feeds]

                if np.any(this_flag):

                    med, slow, shigh = cal_utils.estimate_directional_scale(
                        amp[ff, feeds[this_flag]]
                    )
                    lower = med - self.nsigma_outlier * slow
                    upper = med + self.nsigma_outlier * shigh

                    flag[ff, feeds] &= (amp[ff, feeds] >= lower) & (
                        amp[ff, feeds] <= upper
                    )

                    med_amp_by_pol[ff] = med
                    sig_amp_by_pol[ff] = (
                        0.5
                        * (shigh - slow)
                        / np.sqrt(np.sum(this_flag, dtype=np.float32))
                    )

            # Flag frequencies that are outliers with respect to local median
            if self.nsigma_med_outlier:

                # Collect med_amp_by_pol for all frequencies on rank 0
                if gain.comm.rank == 0:
                    full_med_amp_by_pol = np.zeros(gain.freq.size, dtype=np.float32)
                else:
                    full_med_amp_by_pol = None

                mpiutil.gather_local(
                    full_med_amp_by_pol,
                    med_amp_by_pol,
                    (sfreq,),
                    root=0,
                    comm=gain.comm,
                )

                # Flag outlier frequencies on rank 0
                not_outlier = None
                if gain.comm.rank == 0:

                    med_flag = full_med_amp_by_pol > 0.0

                    not_outlier = cal_utils.flag_outliers(
                        full_med_amp_by_pol,
                        med_flag,
                        window=self.window_med_outlier,
                        nsigma=self.nsigma_med_outlier,
                    )

                    self.log.info(
                        "Pol %s:  %d frequencies are outliers."
                        % (polstr[pp], np.sum(~not_outlier & med_flag, dtype=np.int))
                    )

                # Broadcast outlier frequencies to other ranks
                not_outlier = gain.comm.bcast(not_outlier, root=0)
                gain.comm.Barrier()

                flag[:, feeds] &= not_outlier[sfreq:efreq, np.newaxis]

        # Determine bad frequencies
        flag_freq = (
            np.sum(flag, axis=1, dtype=np.float32) / float(ninput)
        ) > self.threshold_good_freq

        good_freq = list(sfreq + np.flatnonzero(flag_freq))
        good_freq = np.array(mpiutil.allreduce(good_freq, op=MPI.SUM, comm=gain.comm))

        flag &= flag_freq[:, np.newaxis]

        self.log.info("%d good frequencies after flagging amplitude." % good_freq.size)

        # If fraction of good frequencies is less than threshold, stop and return None
        frac_good_freq = good_freq.size / float(gain.freq.size)
        if frac_good_freq < self.valid_gains_frac_good_freq:
            self.log.info(
                "Only %0.1f%% of frequencies remain after flagging amplitude.  Will "
                "not process this sidereal day further." % (100.0 * frac_good_freq,)
            )
            return None

        # Determine bad inputs
        flag = mpiarray.MPIArray.wrap(flag, axis=0, comm=gain.comm)
        flag = flag.redistribute(1)

        fraction_good = np.sum(
            flag[good_freq, :], axis=0, dtype=np.float32
        ) * tools.invert_no_zero(float(good_freq.size))
        flag_input = fraction_good > self.threshold_good_input

        good_input = list(flag.local_offset[1] + np.flatnonzero(flag_input))
        good_input = np.array(mpiutil.allreduce(good_input, op=MPI.SUM, comm=gain.comm))

        flag[:] &= flag_input[np.newaxis, :]

        self.log.info("%d good inputs after flagging amplitude." % good_input.size)

        # Redistribute flags back over frequencies and update container
        flag = flag.redistribute(0)

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
        gain.redistribute("input")

        # Deference datasets
        g = gain.gain[:].view(np.ndarray)
        w = gain.weight[:].view(np.ndarray)

        # Determine flagged frequencies
        flag = w > 0.0

        # Interpolate the gain at non-flagged frequencies to the flagged frequencies
        ginterp, winterp = cal_utils.interpolate_gain_quiet(
            gain.freq[:], g, w, flag=flag, length_scale=self.interp_scale
        )

        # Replace the gain and weight datasets with the interpolated arrays
        # Note that the gain and weight for non-flagged frequencies have not changed
        gain.gain[:] = ginterp
        gain.weight[:] = winterp

        gain.redistribute("freq")

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

    source = config.Property(proptype=str, default="CygA")
    model_fit = config.Property(proptype=bool, default=False)
    use_peak = config.Property(proptype=bool, default=False)
    threshold = config.Property(proptype=float, default=3.0)

    def process(self, sstream, inputmap, inputmask):
        """Determine calibration from a sidereal stream.

        Parameters
        ----------
        sstream : containers.SiderealStream
            Rigidized sidereal timestream to calibrate.
        inputmap : list of :class:`CorrInput`
            List describing the inputs as they are in the file.
        inputmask : ccontainers.CorrInputMask
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
        sstream.redistribute("freq")

        # Find the local frequencies
        nfreq = sstream.vis.local_shape[0]
        sfreq = sstream.vis.local_offset[0]
        efreq = sfreq + nfreq

        # Get the local frequency axis
        freq = sstream.freq["centre"][sfreq:efreq]

        # Fetch source
        source = ephemeris.source_dictionary[self.source]

        # Estimate the RA at which the transiting source peaks
        peak_ra = ephemeris.peak_RA(source, deg=True)

        # Find closest array index
        idx = np.argmin(np.abs(sstream.ra - peak_ra))

        # Fetch the transit into this visibility array
        # Cut out a snippet of the timestream
        slice_width_deg = 3.0 * cal_utils.guess_fwhm(
            400.0, pol="X", dec=source._dec, sigma=True
        )
        slice_width = int(slice_width_deg / np.median(np.abs(np.diff(sstream.ra))))
        slice_centre = slice_width
        st, et = idx - slice_width, idx + slice_width + 1

        vis_slice = sstream.vis[..., st:et].copy()
        ra_slice = sstream.ra[st:et]

        nra = vis_slice.shape[-1]

        # Determine good inputs
        nfeed = len(inputmap)
        good_input = np.arange(nfeed, dtype=np.int)[inputmask.datasets["input_mask"][:]]

        # Use input map to figure out which are the X and Y feeds
        xfeeds = np.array(
            [
                idx
                for idx, inp in enumerate(inputmap)
                if (idx in good_input) and tools.is_chime_x(inp)
            ]
        )
        yfeeds = np.array(
            [
                idx
                for idx, inp in enumerate(inputmap)
                if (idx in good_input) and tools.is_chime_y(inp)
            ]
        )

        self.log.info(
            "Performing sidereal calibration with %d/%d good feeds (%d xpol, %d ypol).",
            len(good_input),
            nfeed,
            len(xfeeds),
            len(yfeeds),
        )

        # Extract the diagonal (to be used for weighting)
        # prior to differencing on-source and off-source
        norm = np.sqrt(_extract_diagonal(vis_slice, axis=1).real)
        norm = tools.invert_no_zero(norm)

        # Subtract the average visibility at the start and end of the slice (off source)
        diff = int(slice_width / 3)
        vis_slice = _adiff(vis_slice, diff)

        # Fringestop the data
        vis_slice = tools.fringestop_pathfinder(
            vis_slice, ra_slice, freq, inputmap, source
        )

        # Create arrays to hold point source response
        resp = np.zeros([nfreq, nfeed, nra], np.complex128)
        resp_err = np.zeros([nfreq, nfeed, nra], np.float64)

        # Solve for the point source response of each set of polarisations
        evalue_x, resp[:, xfeeds, :], resp_err[:, xfeeds, :] = solve_gain(
            vis_slice, feeds=xfeeds, norm=norm[:, xfeeds]
        )
        evalue_y, resp[:, yfeeds, :], resp_err[:, yfeeds, :] = solve_gain(
            vis_slice, feeds=yfeeds, norm=norm[:, yfeeds]
        )

        # Extract flux density of the source
        rt_flux_density = np.sqrt(fluxcat.FluxCatalog[self.source].predict_flux(freq))

        # Divide by the flux density of the point source
        # to convert the response and response_error into
        # units of 'sqrt(correlator units / Jy)'
        resp /= rt_flux_density[:, np.newaxis, np.newaxis]
        resp_err /= rt_flux_density[:, np.newaxis, np.newaxis]

        # Define units
        unit_in = sstream.vis.attrs.get("units", "rt-correlator-units")
        unit_out = "rt-Jy"

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
            fit_flag[:, xfeeds, :] = _contiguous_flag(
                dr_x > self.threshold, centre=slice_centre
            )[:, np.newaxis, :]
            fit_flag[:, yfeeds, :] = _contiguous_flag(
                dr_y > self.threshold, centre=slice_centre
            )[:, np.newaxis, :]

            # Fit model for the complex response of each feed to the point source
            param, param_cov = cal_utils.fit_point_source_transit(
                ra_slice, resp, resp_err, flag=fit_flag
            )

            # Overwrite the initial gain estimates for frequencies/feeds
            # where the model fit was successful
            if self.use_peak:
                gain = np.where(
                    np.isnan(param[:, :, 0]),
                    gain,
                    param[:, :, 0] * np.exp(1.0j * np.deg2rad(param[:, :, -2])),
                )
            else:
                for index in np.ndindex(nfreq, nfeed):
                    if np.all(np.isfinite(param[index])):
                        gain[index] = cal_utils.model_point_source_transit(
                            peak_ra, *param[index]
                        )

            # Create container to hold results of fit
            gain_data = ccontainers.PointSourceTransit(
                ra=ra_slice, pol_x=xfeeds, pol_y=yfeeds, axes_from=sstream
            )

            gain_data.evalue_x[:] = evalue_x
            gain_data.evalue_y[:] = evalue_y
            gain_data.response[:] = resp
            gain_data.response_error[:] = resp_err
            gain_data.flag[:] = fit_flag
            gain_data.parameter[:] = param
            gain_data.parameter_cov[:] = param_cov

            # Update units
            gain_data.response.attrs["units"] = unit_in + " / " + unit_out
            gain_data.response_error.attrs["units"] = unit_in + " / " + unit_out

        else:

            # Create container to hold gains
            gain_data = containers.StaticGainData(axes_from=sstream)

        # Combine dynamic range estimates for both polarizations
        dr = np.minimum(dr_x[:, slice_centre], dr_y[:, slice_centre])

        # Copy to container all quantities that are common to both
        # StaticGainData and PointSourceTransit containers
        gain_data.add_dataset("weight")

        gain_data.gain[:] = gain
        gain_data.weight[:] = dr

        # Update units and unit conversion
        gain_data.gain.attrs["units"] = unit_in + " / " + unit_out
        gain_data.gain.attrs["converts_units_to"] = "Jy"

        # Add attribute with the name of the point source
        # that was used for calibration
        gain_data.attrs["source"] = self.source

        # Return gain data
        return gain_data


def find_contiguous_time_ranges(timestamp, dt=3600.0):
    """Find contiguous ranges within an array of unix timestamps.

    Used by ThermalCalibration to determine the ranges of time
    to load temperature data.

    Parameters
    ----------
    timestamp: np.ndarray[ntime,]
        Unix timestamps.
    dt: float
        Maximum time difference in seconds.
        If consecutive timestamps are separated
        by more than 2 * dt, then they will be
        placed into separate time ranges. Note that
        each time range will be expanded by dt
        on either end.

    Returns
    -------
    time_ranges: [(start_time, stop_time), ...]
        List of 2 element tuples, which each tuple
        containing the start and stop time covering
        a contiguous range of timestamps.
    """

    timestamp = np.sort(timestamp)

    start = [timestamp[0] - dt]
    stop = []

    for tt in range(timestamp.size - 1):

        if (timestamp[tt + 1] - timestamp[tt]) > (2 * dt):

            stop.append(timestamp[tt] + dt)
            start.append(timestamp[tt + 1] - dt)

    stop.append(timestamp[-1] + dt)

    return list(zip(start, stop))


class ThermalCalibration(task.SingleTask):
    """Use weather temperature to correct calibration in between point source transits.

    Attributes
    ----------
    caltime_path : string
        Full path to file describing the calibration times.
    node_spoof : dictionary
        (default: {'cedar_online': '/project/rpp-krs/chime/chime_online/'} )
        host and directory in which to find data.
    """

    caltime_path = config.Property(proptype=str)
    node_spoof = config.Property(proptype=dict, default=_DEFAULT_NODE_SPOOF)

    def setup(self):
        """Load calibration times."""
        self.caltime_file = memh5.MemGroup.from_hdf5(self.caltime_path)

    def process(self, data):
        """Determine thermal calibration for a sidereal stream or time stream.

        Parameters
        ----------
        data : `containers.SiderealStream` or `containers.TimeStream`
            Data to generate calibration for.

        Returns
        -------
        gain : Either `containers.SiderealGainData` or `containers.GainData`
            The type depends on the type of `data`.

        """
        # Frequencies and RA/time
        freq = data.freq[:]
        if "ra" in data.index_map.keys():
            timestamp = self._ra2unix(data.attrs["lsd"], data.ra[:])
            # Create container
            gain = containers.CommonModeSiderealGainData(
                axes_from=data, distributed=True, comm=data.comm
            )
        else:
            timestamp = data.time[:]
            gain = containers.CommonModeGainData(
                time=timestamp, axes_from=data, distributed=True, comm=data.comm
            )
        # Redistribute
        gain.redistribute("freq")
        lo = gain.gain.local_offset[0]
        ls = gain.gain.local_shape[0]

        # Find refference times for each timestamp.
        # This is the time of the transit from which the gains
        # applied to the data were derived.
        self.log.info("Getting refference times")
        reftime_result = self._get_reftime(timestamp, self.caltime_file)

        # Compute gain corrections
        self.log.info("Computing gains corrections")
        g = self._reftime2gain(reftime_result, timestamp, freq[lo : lo + ls])

        # Copy data into container
        gain.gain[:] = g[:]
        # gain.weight[:] = dr

        return gain

    def _ra2unix(self, csd, ra):
        """csd must be integer"""
        return ephemeris.csd_to_unix(csd + ra / 360.0)

    def _reftime2gain(self, reftime_result, timestamp, frequency):
        """
        Parameters
        ----------
        timestamp : array of foats
            Unix time of data points to be calibrated.
        reftime : array of floats
            Unix time of same length as `timestamp'. Reference times of transit of the
            source used to calibrate the data at each time in `times'.
        frequency : array of floats
            Frequencies to obtain the gain corrections for, in MHz.

        Returns
        -------
        g : 2D array of floats of shape (nfreq, ntimes)
            Per-input gain amplitude corrections. Multiply by data
            to correct it.
        """
        ntimes = len(timestamp)
        nfreq = len(frequency)

        reftime = reftime_result["reftime"]
        reftime_prev = reftime_result["reftime_prev"]
        interp_start = reftime_result["interp_start"]
        interp_stop = reftime_result["interp_stop"]

        # Ones. Don't modify data where there are no gains
        g = np.ones((nfreq, ntimes), dtype=np.float)

        # Simple gains. No interpolation.
        direct_gains = np.isfinite(reftime) & (~np.isfinite(reftime_prev))
        # Gains that need interpolation
        to_interpolate = np.isfinite(reftime_prev)

        # Load weather data for this time range
        #######################################################
        trng = find_contiguous_time_ranges(
            np.concatenate((timestamp, reftime, reftime_prev[to_interpolate]))
        )
        wtime, wtemp = self._load_weather(trng)

        # Gain corrections for direct gains (no interpolation).
        #######################################################
        # Reference temperatures
        reftemp = self._interpolate_temperature(wtime, wtemp, reftime[direct_gains])
        # Current temperatures
        temp = self._interpolate_temperature(wtime, wtemp, timestamp[direct_gains])
        # Gain corrections
        g[:, direct_gains] = cal_utils.thermal_amplitude(
            temp[np.newaxis, :] - reftemp[np.newaxis, :], frequency[:, np.newaxis]
        )

        # Gain corrections for interpolated gains.
        ##########################################
        # Reference temperatures
        reftemp = self._interpolate_temperature(wtime, wtemp, reftime[to_interpolate])
        # Reference temperatures of previous update
        reftemp_prev = self._interpolate_temperature(
            wtime, wtemp, reftime_prev[to_interpolate]
        )
        # Current temperatures
        temp = self._interpolate_temperature(wtime, wtemp, timestamp[to_interpolate])
        # Current gain corrections
        current_gain = cal_utils.thermal_amplitude(
            temp[np.newaxis, :] - reftemp[np.newaxis, :], frequency[:, np.newaxis]
        )
        # Previous gain corrections
        previous_gain = cal_utils.thermal_amplitude(
            temp[np.newaxis, :] - reftemp_prev[np.newaxis, :], frequency[:, np.newaxis]
        )
        # Compute interpolation coefficient. Use a Hanning (cos^2) function.
        # The same that is used for gain interpolation in the real-time pipeline.
        transition_period = interp_stop[to_interpolate] - interp_start[to_interpolate]
        time_into_transition = timestamp[to_interpolate] - interp_start[to_interpolate]
        interpolation_factor = (
            np.cos(time_into_transition / transition_period * np.pi / 2) ** 2
        )
        g[:, to_interpolate] = previous_gain * interpolation_factor + current_gain * (
            1 - interpolation_factor
        )

        return g

    def _interpolate_temperature(self, temperature_time, temperature_data, times):
        # Interpolate temperatures
        return np.interp(times, temperature_time, temperature_data)

    def _get_reftime(self, times, cal_file):
        """
        Parameters
        ----------
        times : array of foats
            Unix time of data points to be calibrated
        cal_file : memh5.MemGroup object
            File which containes the reference times
            for calibration source transits.

        Returns
        -------
        reftime : array of floats
            Unix time of same length as `times'. Reference times of transit of the
            source used to calibrate the data at each time in `times'. Returns `NaN'
            for times without a reference.
        """
        # Data from calibration file.
        is_restart = cal_file["is_restart"][:]
        tref = cal_file["tref"][:]
        tstart = cal_file["tstart"][:]
        tend = cal_file["tend"][:]
        # Length of calibration file and of data points
        n_cal_file = len(tstart)
        ntimes = len(times)

        # Len of times, indices in cal_file.
        last_start_index = np.searchsorted(tstart, times, side="right") - 1
        # Len of times, indices in cal_file.
        last_end_index = np.searchsorted(tend, times, side="right") - 1
        # Check for times before first update or after last update.
        too_early = last_start_index < 0
        n_too_early = np.sum(too_early)
        if n_too_early > 0:
            msg = (
                "{0} out of {1} time entries have no reference update."
                + "Cannot correct gains for those entries."
            )
            self.log.warning(msg.format(n_too_early, ntimes))
        # Fot times after the last update, I cannot be sure the calibration is valid
        # (could be that the cal file is incomplete. To be conservative, raise warning.)
        too_late = (last_start_index >= (n_cal_file - 1)) & (
            last_end_index >= (n_cal_file - 1)
        )
        n_too_late = np.sum(too_late)
        if n_too_late > 0:
            msg = (
                "{0} out of {1} time entries are beyond calibration file time values."
                + "Cannot correct gains for those entries."
            )
            self.log.warning(msg.format(n_too_late, ntimes))

        # Array to contain reference times for each entry.
        # NaN for entries with no reference time.
        reftime = np.full(ntimes, np.nan, dtype=np.float)
        # Array to hold reftimes of previous updates
        # (for entries that need interpolation).
        reftime_prev = np.full(ntimes, np.nan, dtype=np.float)
        # Arrays to hold start and stop times of gain transition
        # (for entries that need interpolation).
        interp_start = np.full(ntimes, np.nan, dtype=np.float)
        interp_stop = np.full(ntimes, np.nan, dtype=np.float)

        # Acquisition restart. We load an old gain.
        acqrestart = is_restart[last_start_index] == 1
        reftime[acqrestart] = tref[last_start_index][acqrestart]

        # FPGA restart. Data not calibrated.
        # There shouldn't be any time points here. Raise a warning if there are.
        fpga_restart = is_restart[last_start_index] == 2
        n_fpga_restart = np.sum(fpga_restart)
        if n_fpga_restart > 0:
            msg = (
                "{0} out of {1} time entries are after an FPGA restart but before the "
                + "next kotekan restart. Cannot correct gains for those entries."
            )
            self.log.warning(msg.format(n_fpga_restart, ntimes))

        # This is a gain update
        gainupdate = is_restart[last_start_index] == 0

        # This is the simplest case. Last update was a gain update and
        # it is finished. No need to interpolate.
        calrange = (last_start_index == last_end_index) & gainupdate
        reftime[calrange] = tref[last_start_index][calrange]

        # The next cases might need interpolation. Last update was a gain
        # update and it is *NOT* finished. Update is in transition.
        gaintrans = last_start_index == (last_end_index + 1)

        # This update is in gain transition and previous update was an
        # FPGA restart. Just use new gain, no interpolation.
        prev_is_fpga = is_restart[last_start_index - 1] == 2
        prev_is_fpga = prev_is_fpga & gaintrans & gainupdate
        reftime[prev_is_fpga] = tref[last_start_index][prev_is_fpga]

        # The next two cases need interpolation of gain corrections.
        # It's not possible to correct interpolated gains because the
        # products have been stacked. Just interpolate the gain
        # corrections to avoide a sharp transition.

        # This update is in gain transition and previous update was a
        # Kotekan restart. Need to interpolate gain corrections.
        prev_is_kotekan = is_restart[last_start_index - 1] == 1
        to_interpolate = prev_is_kotekan & gaintrans & gainupdate

        # This update is in gain transition and previous update was a
        # gain update. Need to interpolate.
        prev_is_gain = is_restart[last_start_index - 1] == 0
        to_interpolate = to_interpolate | (prev_is_gain & gaintrans & gainupdate)

        # Reference time of this update
        reftime[to_interpolate] = tref[last_start_index][to_interpolate]
        # Reference time of previous update
        reftime_prev[to_interpolate] = tref[last_start_index - 1][to_interpolate]
        # Start and stop times of gain transition.
        interp_start[to_interpolate] = tstart[last_start_index][to_interpolate]
        interp_stop[to_interpolate] = tend[last_start_index][to_interpolate]

        # For times too early or too late, don't correct gain.
        # This might mean we don't correct gains right after the last update
        # that could in principle be corrected. But there is no way to know
        # If the calibration file is up-to-date and the last update applies
        # to all entries that come after it.
        reftime[too_early | too_late] = np.nan

        # Test for un-identified NaNs
        known_bad_times = (too_early) | (too_late) | (fpga_restart)
        n_bad_times = np.sum(~np.isfinite(reftime[~known_bad_times]))
        if n_bad_times > 0:
            msg = (
                "{0} out of {1} time entries don't have a reference calibration time "
                + "without an identifiable cause. Cannot correct gains for those entries."
            )
            self.log.warning(msg.format(n_bad_times, ntimes))

        # Bundle result in dictionary
        result = {
            "reftime": reftime,
            "reftime_prev": reftime_prev,
            "interp_start": interp_start,
            "interp_stop": interp_stop,
        }

        return result

    def _load_weather(self, time_ranges):
        """Load the chime_weather acquisitions covering the input time ranges."""
        ntime = None

        # Can only query the database from one rank.
        if self.comm.rank == 0:

            f = finder.Finder(node_spoof=self.node_spoof)
            f.only_chime_weather()  # Excludes MingunWeather
            for start_time, end_time in time_ranges:
                f.include_time_interval(start_time, end_time)
            f.accept_all_global_flags()

            times, temperatures = [], []
            results_list = f.get_results()
            for result in results_list:
                wdata = result.as_loaded_data()
                times.append(wdata.time[:])
                temperatures.append(wdata.temperature[:])

            wtime = np.concatenate(times)
            wtemp = np.concatenate(temperatures)

            ntime = len(wtime)

        # Broadcast the times and temperatures to all ranks.
        ntime = self.comm.bcast(ntime, root=0)
        if self.comm.rank != 0:
            wtime = np.empty(ntime, dtype=np.float64)
            wtemp = np.empty(ntime, dtype=np.float64)

        self.comm.Bcast(wtime, root=0)
        self.comm.Bcast(wtemp, root=0)

        # Ensure times are increasing. Needed for np.interp().
        sort_index = np.argsort(wtime)
        wtime = wtime[sort_index]
        wtemp = wtemp[sort_index]

        return wtime, wtemp


class CalibrationCorrection(task.SingleTask):
    """Base class for applying multiplicative corrections based on a DataFlag.

    This task is not functional.  It simply defines `setup` and `process`
    methods that are common to several subclasses.  All subclasses must
    define the `_get_correction` and `_correction_is_nonzero` methods.

    Parameters
    ----------
    rotation : float
        Current best estimate of telescope rotation.
    name_of_flag : str
        The name of the DataFlag.
    """

    rotation = config.Property(proptype=float, default=tools._CHIME_ROT)
    name_of_flag = config.Property(proptype=str, default="")

    def setup(self):
        """Query the database for all DataFlags with name equal to the `name_of_flag` property."""
        flags = []

        # Query flag database if on 0th node
        if self.comm.rank == 0:
            finder.connect_database()
            flag_types = finder.DataFlagType.select()
            for ft in flag_types:
                if ft.name == self.name_of_flag:
                    ftemp = list(
                        finder.DataFlag.select().where(finder.DataFlag.type == ft)
                    )
                    # Only keep flags that will produce nonzero corrections, as defined by
                    # the _correction_is_nonzero method
                    flags += [
                        flg
                        for flg in ftemp
                        if self._correction_is_nonzero(**flg.metadata)
                    ]

        # Share flags with other nodes
        flags = self.comm.bcast(flags, root=0)

        # Save flags to class attribute
        self.log.info("Found %d %s flags in total." % (len(flags), self.name_of_flag))
        self.flags = flags

    def process(self, sstream, inputmap):
        """Apply a multiplicative correction to visiblities during range of time covered by flags.

        Parameters
        ----------
        sstream : andata.CorrData, containers.SiderealStream, or equivalent
            Apply a correction to the `vis` dataset in this container.
        inputmap : list of :class:`CorrInput`
            List describing the inputs as they are in the file, output from
            `tools.get_correlator_inputs()`

        Returns
        ----------
        sstream_out : same as sstream
            The input container with the correction applied.
        """
        # Determine if there are flags pertinent to this range of time
        if "ra" in sstream.index_map:
            ra = sstream.ra
            csd = (
                sstream.attrs["lsd"] if "lsd" in sstream.attrs else sstream.attrs["csd"]
            )
            if hasattr(csd, "__iter__"):
                csd = sorted(csd)[len(csd) // 2]
            timestamp = ephemeris.csd_to_unix(csd + ra / 360.0)
        else:
            timestamp = sstream.time

        covered = False
        for flag in self.flags:
            if np.any((timestamp >= flag.start_time) & (timestamp <= flag.finish_time)):
                covered = True
                break

        # If the flags do not cover this range of time, then do nothing
        # and return the input container
        if not covered:
            return sstream

        # We are covered by the flags, so set up for correction
        sstream.redistribute("freq")

        # Determine local dimensions
        nfreq, nstack, ntime = sstream.vis.local_shape

        # Find the local frequencies
        sfreq = sstream.vis.local_offset[0]
        efreq = sfreq + nfreq

        freq = sstream.freq[sfreq:efreq]

        # Extract representative products for the stacked visibilities
        stack_new, stack_flag = tools.redefine_stack_index_map(
            inputmap, sstream.prod, sstream.stack, sstream.reverse_map["stack"]
        )
        do_not_apply = np.flatnonzero(~stack_flag)
        prod = sstream.prod[stack_new["prod"]].copy()

        # Swap the product pair order for conjugated stack indices
        cj = np.flatnonzero(stack_new["conjugate"].astype(np.bool))
        if cj.size > 0:
            prod["input_a"][cj], prod["input_b"][cj] = (
                prod["input_b"][cj],
                prod["input_a"][cj],
            )

        # Dereference dataset
        ssv = sstream.vis[:]

        # Loop over flags again
        for flag in self.flags:

            in_range = (timestamp >= flag.start_time) & (timestamp <= flag.finish_time)
            if np.any(in_range):

                msg = (
                    "%d (of %d) samples require phase correction according to "
                    "%s DataFlag covering %s to %s."
                    % (
                        np.sum(in_range),
                        in_range.size,
                        self.name_of_flag,
                        ephemeris.unix_to_datetime(flag.start_time).strftime(
                            "%Y%m%dT%H%M%SZ"
                        ),
                        ephemeris.unix_to_datetime(flag.finish_time).strftime(
                            "%Y%m%dT%H%M%SZ"
                        ),
                    )
                )

                self.log.info(msg)

                correction = self._get_correction(
                    freq, prod, timestamp[in_range], inputmap, **flag.metadata
                )

                if do_not_apply.size > 0:
                    self.log.warning(
                        "Do not have valid baseline distance for stack indices: %s"
                        % str(do_not_apply)
                    )
                    correction[:, do_not_apply, :] = 1.0 + 0.0j

                ssv.local_array[:, :, in_range] *= correction

        # Return input container with phase correction applied
        return sstream

    def _correction_is_nonzero(self, **kwargs):
        return True

    def _get_correction(self, freq, prod, timestamp, inputmap, **kwargs):
        pass


class CorrectTimeOffset(CalibrationCorrection):
    """Correct stacked visibilities for a different time standard used during calibration.

    Parameters
    ----------
    name_of_flag : str
        The name of the DataFlag that contains the time offset.
    """

    name_of_flag = config.Property(proptype=str, default="calibration_time_offset")

    def _correction_is_nonzero(self, **kwargs):
        return kwargs["time_offset"] != 0.0

    def _get_correction(self, freq, prod, timestamp, inputmap, **kwargs):

        time_offset = kwargs["time_offset"]
        calibrator = kwargs["calibrator"]
        self.log.info(
            "Applying a phase correction for a %0.2f second "
            "time offset on the calibrator %s." % (time_offset, calibrator)
        )

        body = ephemeris.source_dictionary[calibrator]

        lat = np.radians(ephemeris.CHIMELATITUDE)

        # Compute feed positions with rotation
        tools.change_chime_location(rotation=self.rotation)
        uv = _calculate_uv(freq, prod, inputmap)

        # Return back to default rotation
        tools.change_chime_location(default=True)

        # Determine location of calibrator
        ttrans = ephemeris.transit_times(body, timestamp[0] - 24.0 * 3600.0)[0]

        ra, dec = ephemeris.object_coords(body, date=ttrans, deg=False)

        ha = np.radians(ephemeris.lsa(ttrans + time_offset)) - ra

        # Calculate and return the phase correction, which is old offset minus new time offset
        # since we previously divided the chimestack data by the response to the calibrator.
        correction = tools.fringestop_phase(ha, lat, dec, *uv) * tools.invert_no_zero(
            tools.fringestop_phase(0.0, lat, dec, *uv)
        )

        return correction[:, :, np.newaxis]


class CorrectTelescopeRotation(CalibrationCorrection):
    """Correct stacked visibilities for a different telescope rotation used during calibration.

    Parameters
    ----------
    name_of_flag : str
        The name of the DataFlag that contains the telescope rotation
        used was during calibration.
    """

    name_of_flag = config.Property(
        proptype=str, default="calibration_telescope_rotation"
    )

    def _correction_is_nonzero(self, **kwargs):
        return kwargs["rotation"] != self.rotation

    def _get_correction(self, freq, prod, timestamp, inputmap, **kwargs):

        rotation = kwargs["rotation"]
        calibrator = kwargs["calibrator"]

        self.log.info(
            "Applying a phase correction to convert from a telescope rotation "
            "of %0.3f deg to %0.3f deg for the calibrator %s."
            % (rotation, self.rotation, calibrator)
        )

        body = ephemeris.source_dictionary[calibrator]

        lat = np.radians(ephemeris.CHIMELATITUDE)

        # Compute feed positions with old rotation
        tools.change_chime_location(rotation=rotation)
        old_uv = _calculate_uv(freq, prod, inputmap)

        # Compute feed positions with current rotation
        tools.change_chime_location(rotation=self.rotation)
        current_uv = _calculate_uv(freq, prod, inputmap)

        # Return back to default rotation
        tools.change_chime_location(default=True)

        # Determine location of calibrator
        ttrans = ephemeris.transit_times(body, timestamp[0] - 24.0 * 3600.0)[0]

        ra, dec = ephemeris.object_coords(body, date=ttrans, deg=False)

        # Calculate and return the phase correction, which is old positions minus new positions
        # since we previously divided the chimestack data by the response to the calibrator.
        correction = tools.fringestop_phase(
            0.0, lat, dec, *old_uv
        ) * tools.invert_no_zero(tools.fringestop_phase(0.0, lat, dec, *current_uv))

        return correction[:, :, np.newaxis]


def _calculate_uv(freq, prod, inputmap):
    """Generate baseline distances in wavelengths from the frequency, products, and inputmap."""
    feedpos = tools.get_feed_positions(inputmap).T
    dist = feedpos[:, prod["input_a"]] - feedpos[:, prod["input_b"]]

    lmbda = speed_of_light * 1e-6 / freq
    uv = dist[:, np.newaxis, :] / lmbda[np.newaxis, :, np.newaxis]

    return uv
