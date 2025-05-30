"""Tasks for calibrating the data."""

import json

import caput.time as ctime
import numpy as np
import scipy.signal
from caput import config, memh5, mpiarray, mpiutil, weighted_median
from ch_ephem import coord, sources
from ch_ephem.observers import chime
from ch_util import andata, cal_utils, ephemeris, finder, fluxcat, ni_utils, rfi, tools
from draco.core import containers, task
from draco.util import _fast_tools
from mpi4py import MPI
from scipy import interpolate
from scipy.constants import c as speed_of_light

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
        msg = f"Array length ({utmat.shape[axis]:d}) of axis {axis:d} does not correspond upper triangle\
                of square matrix"
        raise RuntimeError(msg)

    # Find indices of the diagonal
    diag_ind = [tools.cmap(ii, ii, nside) for ii in range(nside)]

    # Construct slice objects representing the axes before and after the product axis
    slice0 = (np.s_[:],) * axis
    slice1 = (np.s_[:],) * (len(utmat.shape) - axis - 1)

    # Extract wanted elements with a giant slice
    sl = (*slice0, diag_ind, *slice1)

    return utmat[sl]


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
    data = data[:].local_array

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
            f"Input normalization matrix has shape {norm.shape}, "
            f"should have shape {gain.shape}."
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

        return ni_utils.process_synced_data(
            ts, ni_params=ni_params, only_off=self.only_off
        )


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
            norm_array = (
                _extract_diagonal(ts.datasets["gated_vis0"][:].local_array) ** 0.5
            )
            norm_array = tools.invert_no_zero(norm_array)
        elif self.norm == "off":
            norm_array = _extract_diagonal(ts.vis[:].local_array) ** 0.5
            norm_array = tools.invert_no_zero(norm_array)

            # Extract the points with zero weight (these will get zero norm)
            w = _extract_diagonal(ts.weight[:].local_array) > 0
            w[:, noise_channel] = True  # Make sure we keep the noise channel though!

            norm_array *= w

        elif self.norm == "none":
            norm_array = np.ones(
                (ts.vis[:].local_shape[0], ts.ninput, ts.ntime), dtype=np.uint8
            )
        else:
            raise RuntimeError("Value of norm not recognised.")

        # Take a view now to avoid some MPI issues
        gate_view = ts.datasets["gated_vis0"][:].local_array

        # Find gains with the eigenvalue method
        evalue, gain = solve_gain(gate_view, norm=norm_array)[0:2]
        dr = evalue[:, -1, :] * tools.invert_no_zero(evalue[:, -2, :])

        # Normalise by the noise source channel
        gain *= tools.invert_no_zero(gain[:, np.newaxis, noise_channel, :])
        gain = np.nan_to_num(gain)

        # Create container from gains
        gain_data = containers.GainData(axes_from=ts)
        gain_data.add_dataset("weight")

        # Copy data into container
        gain_data.gain[:] = gain
        gain_data.weight[:] = dr[:, np.newaxis, :]

        return gain_data


class DetermineSourceTransit(task.SingleTask):
    """Determine the sources that are transiting within time range covered by container.

    Attributes
    ----------
    source_list : list of str
        List of source names to consider.  If not specified, all sources
        contained in `ch_ephem.sources.source_dictionary` will be considered.
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
        self.source_list = sorted(
            self.source_list or ephemeris.source_dictionary.keys(),
            key=lambda src: fluxcat.FluxCatalog[src].predict_flux(self.freq),
            reverse=True,
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
            timestamp = chime.lsd_to_unix(lsd + sstream.ra / 360.0)

        # Loop over sources and check if there is a transit within time range
        # covered by container.  If so, then add attributes describing that source
        # and break from the loop.
        contains_transit = False
        for src in self.source_list:
            transit_time = chime.transit_times(
                sources.source_dictionary[src], timestamp[0], timestamp[-1]
            )
            if transit_time.size > 0:
                self.log.info(
                    f"Data stream contains {src} transit on LSD {chime.unix_to_lsd(transit_time[0]):d}."
                )
                sstream.attrs["source_name"] = src
                sstream.attrs["transit_time"] = transit_time[0]
                contains_transit = True
                break

        if contains_transit or not self.require_transit:
            return sstream

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
        Name of the source (same format as `sources.source_dictionary`).
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
    telescope_rotation = config.Property(proptype=float, default=chime.rotation)

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
        # Ensure that we are distributed over frequency
        data.redistribute("freq")

        # Determine local dimensions
        nfreq, neigen, ninput, ntime = data.datasets["evec"].local_shape

        # Find the local frequencies
        freq = data.freq[data.vis[:].local_bounds]

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
        ttrans = chime.transit_times(source_obj.skyfield, data.time[0])[0]
        csd = int(np.floor(chime.unix_to_lsd(ttrans)))

        src_ra, src_dec = chime.object_coords(
            source_obj.skyfield, date=ttrans, deg=True
        )

        ra = chime.unix_to_lsa(data.time)

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
        lat = np.radians(chime.latitude)

        # Dereference datasets
        evec = data.datasets["evec"][:].local_array
        evalue = data.datasets["eval"][:].local_array
        erms = data.datasets["erms"][:].local_array
        vis = data.datasets["vis"][:].local_array
        weight = data.flags["vis_weight"][:].local_array

        # Check for negative autocorrelations (bug observed in older data)
        negative_auto = vis.real < 0.0
        if np.any(negative_auto):
            vis[negative_auto] = 0.0 + 0.0j
            weight[negative_auto] = 0.0

        # Find inputs that were not included in the eigenvalue decomposition
        eps = 10.0 * np.finfo(evec.dtype).eps
        evec_all_zero = np.all(np.abs(evec[:, 0]) < eps, axis=(0, 2))

        input_flags = np.zeros(ninput, dtype=bool)
        for ii in range(ninput):
            input_flags[ii] = np.logical_not(
                mpiutil.allreduce(evec_all_zero[ii], op=MPI.LAND, comm=data.comm)
            )

        self.log.info(
            f"{np.sum(~input_flags):d} inputs missing from eigenvalue decomposition."
        )

        # Check that we have data for the phase reference
        for ref in self.phase_ref:
            if not input_flags[ref]:
                ValueError(
                    f"Requested phase reference ({ref:d}) "
                    "was not included in decomposition."
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

        pol = [yfeeds, xfeeds]
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
                f"Do not have positions for feeds: {no_distance[input_flags[no_distance]]!s}"
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
            f"Using a dynamic range threshold of {self.dyn_rng_threshold:0.2f}."
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
            ValueError(f"Reference feed {self.eigen_ref:d} is incorrect.")

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
        response.attrs["tag"] = f"{source_name.lower()}_lsd_{csd:d}"

        # Add an attribute that indicates if the transit occured during the daytime
        is_daytime = 0
        solar_rise = chime.solar_rising(ttrans - 86400.0)
        for sr in solar_rise:
            ss = chime.solar_setting(sr)[0]
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
                f"Do not recognize model {self.model}.  Options are "
                "`gauss_amp_poly_phase` and `poly_log_amp_poly_phase`."
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
        freq = response.freq[response.vis[:].local_bounds]

        # Calculate the hour angle using the source and transit time saved to attributes
        source_obj = sources.source_dictionary[response.attrs["source_name"]]
        ttrans = response.attrs["transit_time"]

        src_ra, src_dec = chime.object_coords(source_obj, date=ttrans, deg=True)

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
        vis = response.vis[:].local_array
        weight = response.weight[:].local_array
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
        param = fit.parameter[:].local_array
        param_cov = fit.parameter_cov[:].local_array
        chisq = fit.chisq[:].local_array
        ndof = fit.ndof[:].local_array

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
        # Distribute over frequency
        gain.redistribute("freq")

        nfreq, ninput = gain.gain.local_shape

        sfreq = gain.gain.local_offset[0]
        efreq = sfreq + nfreq

        # Dereference datasets
        flag = gain.weight[:].local_array > 0.0
        amp = np.abs(gain.gain[:].local_array)

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

                    noutlier = np.sum(~not_outlier & med_flag)
                    self.log.info(
                        f"Pol {polstr[pp]}: {noutlier:d} frequencies are outliers."
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

        self.log.info(f"{good_freq.size:d} good frequencies after flagging amplitude.")

        # If fraction of good frequencies is less than threshold, stop and return None
        frac_good_freq = good_freq.size / float(gain.freq.size)
        if frac_good_freq < self.valid_gains_frac_good_freq:
            self.log.info(
                f"Only {100.0 * frac_good_freq:0.1f}% of frequencies remain after flagging amplitude.  Will "
                "not process this sidereal day further."
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

        self.log.info(f"{good_input.size:d} good inputs after flagging amplitude.")

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
    in_place: bool
        Save the interpolated gains to the input container.
    """

    interp_scale = config.Property(proptype=float, default=30.0)
    in_place = config.Property(proptype=bool, default=False)

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
        g = gain.gain[:].local_array
        w = gain.weight[:].local_array

        # Determine flagged frequencies
        flag = w > 0.0

        # Interpolate the gain at non-flagged frequencies to the flagged frequencies
        ginterp, winterp = cal_utils.interpolate_gain_quiet(
            gain.freq[:], g, w, flag=flag, length_scale=self.interp_scale
        )

        if self.in_place:
            out = gain
        else:
            out = containers.StaticGainData(
                axes_from=gain,
                attrs_from=gain,
                distributed=gain.distributed,
                comm=gain.comm,
            )
            out.add_dataset("weight")
            out.redistribute("input")
            gain.redistribute("freq")

        # Replace the gain and weight datasets with the interpolated arrays
        # Note that the gain and weight for non-flagged frequencies have not changed
        out.gain[:] = ginterp
        out.weight[:] = winterp

        out.redistribute("freq")

        return out


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

        # Get the local frequency axis
        nfreq = sstream.vis.local_shape[0]
        freq = sstream.freq["centre"][sstream.vis[:].local_bounds]

        # Fetch source
        source = sources.source_dictionary[self.source]

        # Estimate the RA at which the transiting source peaks
        peak_ra = coord.peak_ra(source, deg=True)

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
        good_input = np.arange(nfeed, dtype=np.int64)[
            inputmask.datasets["input_mask"][:]
        ]

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
            fit_flag = np.zeros([nfreq, nfeed, nra], dtype=bool)
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

    caltime_file = None

    def process(self, data):
        """Determine thermal calibration for a sidereal stream or time stream.

        Parameters
        ----------
        data : `containers.SiderealStream` or `containers.TimeStream`
            Data to generate calibration for.

        Returns
        -------
        gain : Either `containers.SiderealGainData` or `containers.GainData`
            The type depends on the type of `data`. Returns `None` if a thermal
            correction could not be determined.

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
                axes_from=data, distributed=True, comm=data.comm
            )
        # Redistribute
        gain.redistribute("freq")
        lo = gain.gain.local_offset[0]
        ls = gain.gain.local_shape[0]

        # Find reference times for each timestamp.
        # This is the time of the transit from which the gains
        # applied to the data were derived.
        self.log.info("Getting reference times")

        reftime_result = None

        # Try and lookup the cal times. Only do this on rank = 0, and then broadcast the
        # results
        try:
            # First attempt to group all dataset_ids for all frequencies on
            # rank=0 so it can do the full lookup
            dataset_ids = data.dataset_id[:].gather(rank=0)

            if self.comm.rank == 0:
                # Try to use the dataset ID scheme in here, if the time range passed won't
                # work an exception will be raised...
                reftime_result = cal_utils.get_reference_times_dataset_id(
                    timestamp, dataset_ids, logger=self.log
                )

        # ... catch it and then try to load a calibration time file
        except (ValueError, KeyError):
            if self.comm.rank == 0:
                self.log.debug(
                    "Could not get cal times via dataset IDs, trying caltime file."
                )

                if self.caltime_file is None:
                    self._load_cal_file()

                if timestamp[0] > self._file_start and timestamp[-1] < self._file_end:
                    reftime_result = cal_utils.get_reference_times_file(
                        timestamp, self.caltime_file, logger=self.log
                    )
                else:
                    self.log.error("Cal time file does not cover the period requested.")

        reftime_result = self.comm.bcast(reftime_result, root=0)

        if reftime_result is None:
            self.log.error(
                "Could not find cal time for incoming data. Check the logs for rank=0 "
                "to see why."
            )
            return None

        # Compute gain corrections
        self.log.info("Computing gains corrections")
        g = self._reftime2gain(reftime_result, timestamp, freq[lo : lo + ls])

        # Copy data into container
        gain.gain[:] = g[:]

        return gain

    def _load_cal_file(self):
        """Load the cal time file."""
        self.caltime_file = memh5.MemGroup.from_hdf5(self.caltime_path)
        self._file_start = self.caltime_file["tstart"][0]
        self._file_end = self.caltime_file["tend"][-1]

    def _ra2unix(self, csd, ra):
        """Csd must be integer."""
        return chime.lsd_to_unix(csd + ra / 360.0)

    def _reftime2gain(self, reftime_result, timestamp, frequency):
        """Get gain corrections based on source transit times.

        Parameters
        ----------
        reftime_result : array of floats
            Unix time of same length as `timestamp'. Reference times of transit of the
            source used to calibrate the data at each time in `times'.
        timestamp : array of foats
            Unix time of data points to be calibrated.
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
        g = np.ones((nfreq, ntimes), dtype=np.float64)

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

        # Exclude NaNs and Infs in weather data from the interpolation
        weather_sel = np.isfinite(wtemp)
        wtime = wtime[weather_sel]
        wtemp = wtemp[weather_sel]

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


class ApplyDigitalGain(task.SingleTask):
    """Multiply calibration gains by the digital gains.

    This yields the complex number that was applied
    to the voltage data by the real-time pipeline.

    Attributes
    ----------
    invert: bool
        Multiply calibration gains by the *inverse* of the
        digital gains.

    normalize: bool
        Normalize the digital gains by the median value over
        input and frequency, so that the overall magnitude of
        the calibration gains does not change.
    """

    invert = config.Property(proptype=bool, default=False)
    normalize = config.Property(proptype=bool, default=False)

    def setup(self, files):
        """Load digital gain files that cover full span of time to be processed.

        Parameters
        ----------
        files: list of str
            List of paths to files containing the digital gains.
        """
        # Load all of the digital gains into a single container
        digi_gain_data = andata.DigitalGainData.from_acq_h5(files)

        # Save as class attribute
        self.digi_gain_data = digi_gain_data

        # Extract the gain
        dg = digi_gain_data.gain[:]

        # If requested, normalize the digital gains by the
        # median over frequency and input.
        if self.normalize:
            self.log.info("Normalizing the digital gains.")
            med_dg = np.median(np.abs(dg), axis=(1, 2), keepdims=True)
            dg = dg * tools.invert_no_zero(med_dg)

        # If requested, invert the digital gains.
        if self.invert:
            self.log.info("Inverting the digital gains.")
            dg = tools.invert_no_zero(dg)

        # Save digital gains as class attribute.
        self.dg = dg

    def process(self, gain):
        """Lookup and apply the relevant digital gain update.

        Parameters
        ----------
        gain: StaticGainData
            The calibration gains at a particular time.

        Returns
        -------
        gain: StaticGainData
            The input container with the gain and weight dataset
            scaled by the appropriate digital gains.
        """
        gain.redistribute("freq")

        # Find the local frequencies
        fsel = gain.gain[:].local_bounds

        # Look up the most recent digital gain update using
        # the timestamp in the input container
        tindex = self.digi_gain_data.search_update_time(gain.attrs["time"])[0]

        # Apply in place
        dg = self.dg[tindex][fsel]

        gain.gain[:].local_array[:] *= dg
        gain.weight[:].local_array[:] *= tools.invert_no_zero(np.abs(dg) ** 2)

        # Save digital gain update_id as attribute
        gain.attrs["digitalgain_update_id"] = self.digi_gain_data.update_id[
            tindex
        ].decode()

        # Return the scaled gains
        return gain


class InvertGain(task.SingleTask):
    """Invert gains."""

    def process(self, gain):
        """Invert gains.

        Parameters
        ----------
        gain: StaticGainData or GainData
            gain data container

        Returns
        -------
        gain: StaticGainData or GainData
            The input container with the gain dataset
            inverted and the uncertainty contained
            in the weight dataset propagated appropriately.
        """
        g = gain.gain[:]
        w = gain.weight[:] * np.abs(g) ** 4

        gain.gain[:] = tools.invert_no_zero(g)
        gain.weight[:] = w

        return gain


class BaseCommonMode(task.SingleTask):
    """Base class for calculating the common-mode gain."""

    use_cylinder = config.Property(proptype=bool, default=True)

    def setup(self, pm):
        """Use telescope instance to identify groups of similar feeds.

        Parameters
        ----------
        pm : ProductManager
            Object describing the telescope.
        """
        self.input_map = pm.telescope.feeds

        self._set_groups(self.input_map)

        self.dataset_map = {
            containers.GainData: "gain",
            containers.StaticGainData: "gain",
            containers.TrackBeam: "beam",
        }

    def _set_groups(self, inputmap):
        """Group inputs based on their cylinder and polarisation.

        Override to define a different grouping.

        Parameters
        ----------
        inputmap: list of CorrInput
            map of inputs

        Attributes
        ----------
        groups: np.ndarray[ngroup,]
            Names of the groups that will be averaged over.
        gindex: dict
            Dictionary of the format {group_id: group_indices}.
            Each entry in gindex contains the indices of all inputs
            of a particular polarisation and (optionally) on a
            particular cylinder.
        glookup: dict
            Dictionary of the format {input_index: group_index}.
        """
        index = np.flatnonzero([tools.is_chime(inp) for inp in inputmap])

        fmt = "{inp.pol}"
        if self.use_cylinder:
            fmt += "-{inp.cyl}"

        idd = np.array([fmt.format(inp=inputmap[ii]) for ii in index])

        self.groups = np.unique(idd)
        self.gindex = {ug: index[idd == ug] for ug in self.groups}
        self.glookup = {
            ii: gg for gg, ug in enumerate(self.groups) for ii in self.gindex[ug]
        }


class ComputeCommonMode(BaseCommonMode):
    """Compute the common-mode gain amplitude.

    Attributes
    ----------
    use_amplitude: bool
        Take the absolute value before calculating the
        average/percentile over inputs.
    use_percentile: bool
        If False, then calculate the average over inputs.
        If True, then calculate a percentile over inputs.
    percentile: float
        Percentile over inputs to calculate.
        Only used if use_percentile is True.
    """

    use_amplitude = config.Property(proptype=bool, default=False)
    use_percentile = config.Property(proptype=bool, default=True)
    percentile = config.Property(proptype=float, default=50.0)

    def process(self, data):
        """Calculate the common-mode gain.

        Parameters
        ----------
        data: StaticGainData or GainData
            gain data container

        Returns
        -------
        group: StaticGainData or GainData
            The common-mode gain *amplitude* for the different
            groups of inputs.
        """
        # Determine what dataset we are dealing with based on the
        # input container type and find the input axis of that dataset.
        dset = self.dataset_map[data.__class__]
        inp_axis = list(data[dset].attrs["axis"]).index("input")

        data.redistribute("freq")

        # Dereference datasets.
        vis = data[dset][:].local_array
        if self.use_amplitude:
            # If requested use only the amplitude.
            self.log.info("Taking the amplitude of the gain.")
            vis = np.abs(vis)

        iscomplex = np.any(np.iscomplex(vis))

        weight = data.weight[:].local_array
        flag = weight > 0.0

        # Create output container
        self.log.info(f"There are {len(self.groups):0.0f} groups in total.")

        group = data.__class__(
            attrs_from=data,
            axes_from=data,
            input=self.groups,
            distributed=data.distributed,
            comm=data.comm,
        )

        group.add_dataset("weight")
        group.redistribute("freq")

        grp_vis = group[dset][:].local_array
        grp_weight = group.weight[:].local_array

        # Calculate the mean (or percentile) for each group of inputs
        for gg, glbl in enumerate(self.groups):
            gind = self.gindex[glbl]

            gsi = tuple([slice(None)] * inp_axis + [gind])
            gso = tuple([slice(None)] * inp_axis + [gg])

            norm = tools.invert_no_zero(np.sum(flag[gsi], axis=inp_axis))

            if self.use_percentile:
                temp_re = np.nanpercentile(
                    np.where(flag[gsi], vis[gsi].real, np.nan),
                    self.percentile,
                    axis=inp_axis,
                )

                if iscomplex:
                    temp_im = np.nanpercentile(
                        np.where(flag[gsi], vis[gsi].imag, np.nan),
                        self.percentile,
                        axis=inp_axis,
                    )

                    temp = temp_re + 1.0j * temp_im

                else:
                    temp = temp_re

                grp_vis[gso] = np.where(np.isfinite(temp), temp, 0.0)

                grp_weight[gso] = (norm > 0).astype(np.float32)

            else:
                grp_vis[gso] = np.sum(flag[gsi] * vis[gsi], axis=inp_axis) * norm

                grp_weight[gso] = tools.invert_no_zero(
                    np.sum(flag[gsi] * tools.invert_no_zero(weight[gsi]), axis=inp_axis)
                    * norm**2
                )

        return group


class ExpandCommonMode(BaseCommonMode):
    """Expand the common mode so that it can be applied to the original input axis."""

    def process(self, cmn):
        """Expand the common mode gain amplitude.

        Parameters
        ----------
        cmn: StaticGainData or GainData
            The common-mode gain *amplitude* for the different
            groups of inputs.

        Returns
        -------
        out: StaticGainData or GainData
            The common-mode gain amplitude replicated for all inputs
            in a group.
        """
        dset = self.dataset_map[cmn.__class__]
        inp_axis = list(cmn[dset].attrs["axis"]).index("input")

        cmn.redistribute("freq")

        cvis = cmn[dset][:].local_array
        cweight = cmn.weight[:].local_array

        # Create output container
        inputs = np.array(
            [(inp.id, inp.input_sn) for inp in self.input_map],
            dtype=[("chan_id", "u2"), ("correlator_input", "U32")],
        )
        ninput = inputs.size

        out = cmn.__class__(
            attrs_from=cmn,
            axes_from=cmn,
            input=inputs,
            distributed=cmn.distributed,
            comm=cmn.comm,
        )

        if "weight" not in out:
            out.add_dataset("weight")

        out.redistribute("freq")

        # Dereference datasets
        ovis = out[dset][:].local_array
        oweight = out.weight[:].local_array

        # Loop over local inputs
        for ii in range(ninput):
            gso = tuple([slice(None)] * inp_axis + [ii])

            try:
                gg = self.glookup[ii]

            except KeyError:
                ovis[gso] = 1.0
                oweight[gso] = 0.0

            else:
                gsi = tuple([slice(None)] * inp_axis + [gg])

                ovis[gso] = cvis[gsi]
                oweight[gso] = cweight[gsi]

        # Return
        return out


class IdentifyNarrowbandFeatures(task.SingleTask):
    """Identify and flag narrowband features in gains.

    Attributes
    ----------
    tau_cut: float
        Cutoff of the high-pass filter in microseconds.
    epsilon: float
        Stop-band rejection of the filter.
    window: int
        Width of the window, in number of frequency channnels,
        used to estimate the noise by calculating a local
        median absolute deviation.
    threshold: float
        Number of median absolute deviations beyond which
        a frequency channel is considered an outlier.
    nperiter: int
        Maximum number of frequency channels to flag
        on any iteration.
    niter: int
        Maximum number of iterations.
    """

    tau_cut = config.Property(proptype=float, default=0.6)
    epsilon = config.Property(proptype=float, default=1e-10)
    window = config.Property(proptype=int, default=151)
    threshold = config.Property(proptype=float, default=6.0)
    nperiter = config.Property(proptype=int, default=1)
    niter = config.Property(proptype=int, default=40)

    def process(self, data):
        """Identify and flag narrowband features in the gain.

        Parameters
        ----------
        data: StaticGainData
            Gain applied to the voltage data.

        Returns
        -------
        out: StaticGainData
            Copy of the input container with the weight
            dataset set to zero if a narrowband feature
            has been identified for that frequency and input.
        """
        # Create output container.  Problems with copy when distributed over inputs.
        data.redistribute("freq")
        out = data.copy()
        out.redistribute("input")
        data.redistribute("input")

        # Dereference datasets and calculate amplitude
        freq = data.freq
        amp = np.abs(data.gain[:].local_array)
        weight = data.weight[:].local_array

        oweight = out.weight[:].local_array

        nfreq, ninput = amp.shape

        # Flag RFI
        rfi_flag = ~rfi.frequency_mask(freq)
        flag = (weight > 0.0) & rfi_flag[:, np.newaxis]

        # Loop over local inputs
        for ii in range(ninput):
            self.log.debug(f"Processing input {ii} of {ninput}.")

            if not np.any(flag[:, ii]):
                oweight[:, ii] = 0.0
                continue

            # Generate the mask
            try:
                amp_hpf, flag_hpf, rsigma_hpf = rfi.iterative_hpf_masking(
                    freq,
                    amp[:, ii],
                    flag=flag[:, ii],
                    tau_cut=self.tau_cut,
                    epsilon=self.epsilon,
                    window=self.window,
                    threshold=self.threshold,
                    nperiter=self.nperiter,
                    niter=self.niter,
                )
            except np.linalg.LinAlgError as exc:
                self.log.warning(
                    f"Failed to create delay filter for input {ii} (of {ninput}): {exc}"
                )
                oweight[:, ii] = 0.0
            else:
                # Update the weight dataset to flag the narrowband features
                oweight[:, ii] *= flag_hpf.astype(np.float32)

        # Return the original gain container with modified weights
        return out


class ReconstructGainError(task.SingleTask):
    """Estimate the fractional error in the calibration gain.

    The "true" gain is estimated by low-pass filtering the
    applied gain along the frequency axis.  The ratio of the
    applied gain to the true gain is then output.

    The low-pass filtering is sensitive to narrowband features in
    the gain due to RFI.  An iterative algorithm is used to
    identify these narrowband features by low-pass filtering,
    taking the ratio, averaging over all baselines, masking the
    `nperiter` largest absolute deviations relative to the
    median absolute deviation, and then repeating the procedure.
    The frequencies identified as containing narrowband features
    should be masked in the visibility dataset.

    This procedure requires access to all frequencies and inputs.
    For this reason, it is recommended to provide the gain over
    many nights so that the procedure can be distributed over time.

    Attributes
    ----------
    full_output : bool
        If False, then only output the fractional error in the
        calibration gains.  If True, then the low-pass filtered gains,
        frequency-time mask, and an archive of the baseline-averaged
        fractional error at each iteration will be output as well.
    simple_lpf : bool
        If True, then use a simple FIR low-pass filter.  If False,
        then use the DAYENUREST technique to construct the filter.
    numtaps : float
        Number of taps in the FIR low-pass filter if simple_lpf is True.
        Note this is specified in MHz with the actual number of taps
        obtained by dividing this number by the frequency channel width.
    remove_hpf : bool
        Remove a high-pass filtered version of the gains using DAYENU
        prior to fitting DPSS modes to obtain the low-pass filtered version.
        Can suffer from numerical issues when inverting the signal covariance.
        Only relevant if simple_lpf is False.
    rcond : float
        The condition number used to calculate the pseudo-inverse of the masked
        frequency-frequency covariance matrix when using DAYENU.  Only relevant
        if remove_hpf is True and simple_lpf is False.
    niter : int
        Maximum number of iterations used to mask narrowband features in the gains.
    window: int
        Width of the window, in number of frequency channnels, used to estimate
        the noise by calculating a local median absolute deviation. If not provided,
        then will use the entire band.
    nsigma_outlier : float
        Number of median absolute deviations beyond which a frequency channel
        is considered an outlier.  The algorithm terminates when there are
        no longer any frequency channels that exceed this threshold
        or the maximum number of iterations is reached.
    nperiter: int
        Maximum number of frequency channels to flag on any iteration. Note that
        the low-pass filtering will leak power from outliers to neighboring
        frequencies, so it is recommended to keep this number small in order to
        avoid accidentally masking good frequencies contaminated by a neighbor.
    mask_rfi_bands : bool
        Ignore the persistent RFI bands when calculating the gain error.
        These bands are specified in ch_util.rfi.frequency_mask.
    tau_centre : float or np.ndarray[nstopband,]
        The centre of the pass-band regions in micro-seconds.
    tau_width : float or np.ndarray[nstopband,]
        The half width of the pass-band regions in micro-seconds.
    epsilon: float
        Stop-band rejection of the filter.
    threshold: float
        Filter is constructed from eigenmodes of the signal covariance whose
        eigenvalue is larger than this factor times the maximum eigenvalue.
    """

    full_output = config.Property(proptype=bool, default=True)

    simple_lpf = config.Property(proptype=bool, default=False)
    numtaps = config.Property(proptype=float, default=20.0)

    remove_hpf = config.Property(proptype=bool, default=False)
    rcond = config.Property(proptype=float, default=1.0e-15)

    niter = config.Property(proptype=int, default=40)
    window = config.Property(proptype=int)
    nsigma_outlier = config.Property(proptype=float, default=4.0)
    nperiter = config.Property(proptype=int, default=1)
    mask_rfi_bands = config.Property(proptype=bool, default=True)

    tau_centre = config.Property(proptype=np.atleast_1d, default=0.0)
    tau_width = config.Property(proptype=np.atleast_1d, default=0.2)
    epsilon = config.Property(proptype=np.atleast_1d, default=1.0e-12)
    threshold = config.Property(proptype=float, default=1.0e-12)

    def setup(self):
        """Determine the frequency axis.

        This is necessary because the frequency axis in the gains is saved
        as float32 instead of float64, which causes issues when constructing
        the delay filter.
        """
        self.freq = np.linspace(800.0, 400.0, 1024, endpoint=False, dtype=float)
        self.dfreq = np.median(np.abs(np.diff(self.freq)))

    def process(self, gain):
        """Identify features in the gains at high delay.

        Parameters
        ----------
        gain: GainData
            Original gain data.

        Returns
        -------
        out_err: GainData
            Ratio of the original gain and a low-pass filtered version of the gain.
        out_lpf: GainData
            Low-pass filtered version of the gain.  Only output if full_output is True.
        out_mask: RFIMask
            Frequencies and times that were identified as outliers by the iterative
            high-pass-filter masking algorithm.  Only output if full_output is True.
        archive: GainData
            The gain error averaged over all baselines for each iteration of the
            high-pass-filter masking algorithm.  The input axis of this container is
            used to accomodate iteration number.  Only output if full_output is True.
        """
        # Redistribute over time
        gain.redistribute("time")

        # Dereference datasets
        g = gain.gain[:].local_array
        w = gain.weight[:].local_array

        nfreq, ninput, ntime = g.shape

        freq = self._get_freq(gain.freq)

        unix_times = gain.time[gain.gain[:].local_bounds]

        # Create a mask that identifies flagged data
        flag = w > 0.0

        # Create output gain errors
        out_err = containers.GainData(
            freq=freq,
            axes_from=gain,
            attrs_from=gain,
            distributed=gain.distributed,
            comm=gain.comm,
        )

        out_err.add_dataset("weight")
        out_err.add_dataset("update_id")
        out_err.redistribute("time")

        out_err.update_id[:] = gain.update_id[:]

        gout_err = out_err.gain[:].local_array
        wout_err = out_err.weight[:].local_array

        gout_err[:] = 0.0
        wout_err[:] = 0.0

        # The following containers are only output if requested
        if self.full_output:

            # Create output LPF gain
            out_lpf = containers.GainData(
                freq=freq,
                axes_from=gain,
                attrs_from=gain,
                distributed=gain.distributed,
                comm=gain.comm,
            )

            out_lpf.add_dataset("weight")
            out_lpf.add_dataset("update_id")
            out_lpf.redistribute("time")

            out_lpf.update_id[:] = gain.update_id[:]

            gout_lpf = out_lpf.gain[:].local_array
            wout_lpf = out_lpf.weight[:].local_array

            gout_lpf[:] = 0.0
            wout_lpf[:] = 0.0

            # Create archive of iterations
            archive = containers.GainData(
                freq=freq,
                input=np.arange(self.niter - 1, dtype=int),
                axes_from=gain,
                attrs_from=gain,
                distributed=gain.distributed,
                comm=gain.comm,
            )

            archive.add_dataset("weight")
            archive.redistribute("time")

            garch = archive.gain[:].local_array
            warch = archive.weight[:].local_array

            garch[:] = 0.0
            warch[:] = 0.0

            # Create output RFI mask
            out_mask = containers.RFIMask(freq=freq, axes_from=gain, attrs_from=gain)
            out_mask.mask[:] = False

            mask = np.zeros((nfreq, ntime), dtype=bool)

        # Determine eigen-modes of the signal covariance
        freq = out_err.freq

        cov = self._get_cov(freq)
        evalue, evec = np.linalg.eig(cov)

        isort = np.argsort(evalue)[::-1]
        evalue = evalue[isort] / evalue[isort[0]]
        evec = evec[:, isort]

        imax = np.min(np.flatnonzero(np.abs(evalue) < self.threshold))

        self.log.info(f"Fitting {imax} DPSS modes (of {evalue.size}).")

        A = evec[:, 0:imax]
        AT = A.T.conj()

        # Loop over times
        for tt, timestamp in enumerate(unix_times):

            # Make sure the flags can be factorized into an input flag and frequency flag
            fmeas = flag[..., tt]

            input_flag = np.any(fmeas, axis=0)
            good_input = np.flatnonzero(input_flag)

            if not np.all(fmeas[:, good_input] == fmeas[:, good_input[0], np.newaxis]):
                raise RuntimeError(
                    "Must have the same frequency mask for all good inputs."
                )

            # Extact gain for good inputs and global frequency mask
            gmeas = g[..., tt][:, good_input]
            wmeas = w[..., tt][:, good_input]
            vmeas = tools.invert_no_zero(wmeas)

            if self.mask_rfi_bands:
                freq_flag0 = fmeas[:, good_input[0]] & ~rfi.frequency_mask(
                    freq, timestamp=timestamp
                )
            else:
                freq_flag0 = fmeas[:, good_input[0]]

            freq_flag = freq_flag0.copy()

            rcond = self.rcond
            for ii in range(self.niter):

                self.log.info(f"Iteration {ii} of {self.niter}.")

                # Low-pass filter the gains
                if self.simple_lpf and (ii == (self.niter - 1)):

                    # If the simple LPF was requested, then we first need to interpolate
                    # over the masked frequencies.  This is slow, so we only do this for
                    # the last iteration.  Otherwise we use the faster DAYENUREST method
                    # for identify outlier frequencies.
                    ginterp = self._interpolate(freq, gmeas, wmeas, freq_flag)
                    inv_ginterp = tools.invert_no_zero(
                        self._apply_simple_lpf(freq, ginterp)
                    )

                else:
                    # If requested, first subtract off a HPF version of the gains obtained
                    # by applying DAYENU.  If the signal covariance matrix inversion fails,
                    # then we increase the condition number (rcond) and try again.
                    if self.remove_hpf:

                        try:
                            H = self._get_hpf(cov, freq_flag, rcond=rcond)

                        except np.linalg.LinAlgError as exc:
                            self.log.error(
                                "Failed to converge while processing "
                                f"iteration {ii} (rcond is {rcond:0.1e}):  "
                                f"{exc}"
                            )
                            rcond = rcond * 10.0
                            continue
                        else:
                            rcond = self.rcond

                        ghpf = np.matmul(H, gmeas)
                        glpf = freq_flag[:, np.newaxis] * (gmeas - ghpf)

                    else:
                        glpf = freq_flag[:, np.newaxis] * gmeas

                    # Obtain low-pass-filtered gains by fitting the DPSS modes,
                    # ignoring masked frequencies.
                    E = np.matmul(AT, freq_flag[:, np.newaxis] * A)

                    ginterp = np.matmul(A, np.linalg.solve(E, np.matmul(AT, glpf)))
                    inv_ginterp = tools.invert_no_zero(ginterp)

                # Calculate the ratio of the gains to the low-pass-filtered gains.
                ratio = gmeas * inv_ginterp

                if ii < (self.niter - 1):

                    # Collapse over feeds
                    avg_ratio = self._average_over_feeds(ratio)

                    # Flag
                    dratio = np.where(freq_flag, avg_ratio - 1.0, 0.0)

                    masked_freq = self._identify_outliers(dratio, freq_flag)

                    if masked_freq is None:
                        perc_masked = 100.0 * np.mean(~freq_flag & freq_flag0)
                        self.log.info(
                            f"Iteration {ii}, finished.  "
                            f"Masked {perc_masked:0.3f} percent of frequencies in total."
                        )
                        break

                    # Update the frequency mask
                    freq_flag &= ~masked_freq

                    # Print total number off frequencies masked thus far
                    perc_masked = 100.0 * np.mean(~freq_flag & freq_flag0)
                    self.log.info(
                        f"Iteration {ii}, finished.  "
                        f"Masked {perc_masked:0.3f} percent of frequencies in total."
                    )

                    if self.full_output:
                        garch[:, ii, tt] = avg_ratio
                        warch[:, ii, tt] = freq_flag

            # Save final result
            var = vmeas * np.abs(inv_ginterp) ** 2

            gout_err[:, :, tt][:, good_input] = ratio
            wout_err[:, :, tt][:, good_input] = np.where(
                freq_flag[:, np.newaxis], tools.invert_no_zero(var), 0.0
            )

            if self.full_output:
                gout_lpf[:, :, tt][:, good_input] = ginterp
                wout_lpf[:, :, tt][:, good_input] = vmeas

                mask[:, tt] = ~freq_flag & freq_flag0

        # Prepare output depending on config parameters
        out_err.redistribute("freq")

        if self.full_output:
            out_mask.mask[:] = mpiarray.MPIArray.wrap(
                mask, axis=1, comm=gain.comm
            ).allgather()

            out_lpf.redistribute("freq")
            archive.redistribute("freq")

            out = (out_err, out_lpf, out_mask, archive)
        else:
            out = out_err

        return out

    def _average_over_feeds(self, ratio):
        """Average the product of the fractional gain error over all pairs of feeds.

        Parameters
        ----------
        ratio : np.ndarray[nfreq, ninput]
            Gain divided by a low-pass-filtered version of the gain.

        Returns
        -------
        vis : np.ndarray[nfreq,]
            Effective error in the beamformed visibilities due to gain errors.
        """
        ratio_conj = ratio.conj()

        shp = ratio.shape[:-1]
        nfeed = ratio.shape[-1]
        nprod = (nfeed * (nfeed + 1)) // 2

        vis = np.zeros(shp, dtype=ratio.dtype)
        for ii in range(nfeed):
            vis += np.sum(ratio[..., ii, np.newaxis] * ratio_conj[..., ii:], axis=-1)

        return vis / nprod

    def _identify_outliers(self, dy, flag):
        """Identify frequencies where fractional gain error exceeds some threshold.

        Parameters
        ----------
        dy : np.ndarray[nfreq,]
            Fractional errors in the gain, averaged over all pairs of feeds.
        flag : np.ndarray[nfreq,]
            Boolean flag where True indicates good frequency channels and False
            indicates previously masked frequency channels.

        Returns
        -------
        new_mask : np.ndarray[nfreq,]
            Boolean mask where True indicates the frequency channel is an outlier.
        """
        # Calculate the local median absolute deviation
        ady = np.ascontiguousarray(np.abs(dy), dtype=np.float64)
        w = np.ascontiguousarray(flag, dtype=np.float64)

        if self.window is not None:
            sigma = 1.48625 * weighted_median.moving_weighted_median(
                ady, w, self.window, method="split"
            )
        else:
            sigma = 1.48625 * weighted_median.weighted_median(ady, w, method="split")

        # Calculate the signal to noise
        s2n = ady * tools.invert_no_zero(sigma)

        # Identify frequency channels that are above the signal to noise threshold
        above_threshold = np.flatnonzero(s2n > self.nsigma_outlier)

        if above_threshold.size == 0:
            return None

        # Find the largest nperiter frequency channels that are above the threshold
        ibad = above_threshold[np.argsort(-ady[above_threshold])[0 : self.nperiter]]

        # Flag those frequency channels
        new_mask = np.zeros_like(flag)
        new_mask[ibad] = True

        return new_mask

    def _get_freq(self, freq):
        """Find the appropriate float64 representation of the frequencies.

        Parameters
        ----------
        freq : np.ndarray[nfreq,]
            Frequencies in MHz.

        Returns
        -------
        faxis : np.ndarray[nfreq,] with dtype [("centre", <f8), ("width", <f8)]
            Float64 representation of the frequencies to be used
            in filter construction.
        """
        imatch = np.array([np.argmin(np.abs(nu - self.freq)) for nu in freq])

        freq_match = self.freq[imatch]
        if np.any(np.abs(freq - freq_match) > (0.1 * self.dfreq)):
            raise RuntimeError("Frequency axis unexpected.")

        faxis = np.zeros(freq_match.size, dtype=[("centre", float), ("width", float)])
        faxis["centre"] = freq_match
        faxis["width"] = self.dfreq

        return faxis

    def _get_cov(self, freq):
        """Construct a model for the signal covariance.

        Assumes the signal is the sum of one or more
        top hats in delay space located at tau_centre
        with half-width tau_width.

        Parameters
        ----------
        freq : np.ndarray[nfreq,]
            Frequency in MHz.

        Returns
        -------
        cov : np.ndarray[nfreq, nfreq]
            Model for the signal covariance.
        """
        args = (self.tau_centre, self.tau_width, self.epsilon)

        nfreq = freq.size
        dfreq = freq[:, np.newaxis] - freq[np.newaxis, :]

        cov = np.zeros((nfreq, nfreq), dtype=complex)

        for tt, (tc, tw, eps) in enumerate(zip(*args)):

            self.log.info(
                f"Filter component {tt}: "
                f"tau_c = {tc:0.2f} usec, "
                f"tau_w = {tw:0.2f} usec, "
                f"eps = {eps:0.1e}"
            )

            cov += np.exp(-2.0j * np.pi * tc * dfreq) * np.sinc(2.0 * tw * dfreq) / eps

        return cov

    def _get_hpf(self, cov, flag, rcond=None):
        """Construct a high pass filter from foreground covariance and frequency mask.

        Parameters
        ----------
        cov : np.ndarray[nfreq, nfreq]
            Model for the signal covariance.
        flag : np.ndarray[nfreq,]
            Boolean flag where True indicates a good frequency and
            False indicates a bad frequency.
        rcond : float, optional
            Cutoff for small singular values passed to `np.linalg.pinv`.
            Default is None.

        Returns
        -------
        pinv : np.ndarray[nfreq, nfreq]
            Pseudo-inverse of the foreground covariance times
            outer product of the mask with itself.
        """
        nfreq = flag.size

        uflag = flag[:, np.newaxis] & flag[np.newaxis, :]
        ucov = uflag * (np.eye(nfreq, dtype=cov.dtype) + cov)

        return np.linalg.pinv(ucov, hermitian=True, rcond=rcond) * uflag

    def _interpolate(self, freq, gain, weight, flag):
        """Use Gaussian Process Regression to interpolate gains to missing frequencies.

        Parameters
        ----------
        freq : np.ndarray[nfreq,]
            Frequency in MHz.
        gain : np.ndarray[nfreq, ninput]
            Complex gain as a function of frequency for each input.
        weight : np.ndarray[nfreq, ninput]
            Uncertainty on the gain expressed as an inverse variance.
        flag : np.ndarray[nfreq,]
            Boolean flag where True indicates a good frequency and
            False indicates a bad frequency.

        Returns
        -------
        ginterp : np.ndarray[nfreq, ninput]
            Gain at all frequencies.  Previously flagged frequencies
            have been interpolated from neighboring channels.
        """
        if np.all(flag):
            return gain

        flag = np.ones(weight.shape, dtype=bool) & flag[:, np.newaxis]

        ginterp, _ = cal_utils.interpolate_gain_quiet(freq, gain, weight, flag=flag)

        return ginterp

    def _apply_simple_lpf(self, freq, gain):
        """Apply a simple FIR low-pass filter to the gains as a function of frequency.

        Parameters
        ----------
        freq : np.ndarray[nfreq,]
            Frequency in MHz.
        gain : np.ndarray[nfreq, ninput]
            Complex gain as a function of frequency for each input.

        Returns
        -------
        gfilt : np.ndarray[nfreq, ninput]
            Gains low-pass filtered along the frequency axis.
        """
        cutoff = self.tau_width[0]

        dfreq = np.median(np.abs(np.diff(freq)))
        fs = 1.0 / dfreq

        numtaps = np.round(self.numtaps / dfreq)
        numtaps = int(numtaps + int(not (numtaps % 2)))

        coeff = scipy.signal.firwin(numtaps, cutoff, window=("dpss", 5), fs=fs)

        return scipy.signal.filtfilt(coeff, [1.0], gain.astype(np.complex128), axis=0)


class CorrectGainError(task.SingleTask):
    """Correct stacked visibilities for errors in the gains applied in real-time.

    This correction is imperfect because the redundant baselines are not actually
    redundant.

    Attributes
    ----------
    ignore_input_flags : bool
        When calculating the correction, do not exclude
        feeds that were identified as bad and excluded
        from the stacked visibilities (faster).
    """

    ignore_input_flags = config.Property(proptype=bool, default=False)

    def setup(self, gains):
        """Prepare the gain errors as a function of time.

        Parameters
        ----------
        gains: containers.GainData
            Narrowband gain errors generated by the
            ReconstructGainError task.
        """
        gains.redistribute("freq")

        if "time" in gains.index_map:

            self.timestamp = gains.time

            self.frac_error = gains.gain[:].local_array
            self.flag = gains.weight[:].local_array > 0.0

        elif "time" in gains.attrs:

            self.timestamp = np.atleast_1d(gains.attrs["time"])

            self.frac_error = gains.gain[:].local_array[:, :, np.newaxis]
            self.flag = gains.weight[:].local_array[:, :, np.newaxis] > 0.0

        else:
            raise RuntimeError("gain must have a time axis or attribute.")

        self.gains = gains

    def process(self, data):
        """Look up gain errors, stack over baselines, and apply to visibilities.

        Parameters
        ----------
        data: TimeStream or SiderealStream
            Visibilities.

        Returns
        -------
        data: TimeStream or SiderealStream
            Visibilities with the gain errors removed.
        """
        # Make sure the frequencies are the same
        if not np.array_equal(self.gains.freq, data.freq):
            raise ValueError("Frequencies do not match for gain error and timestream.")

        data.redistribute("freq")

        stacked_vis = data.vis[:].local_array
        stacked_weight = data.weight[:].local_array

        stack = data.reverse_map["stack"]["stack"]
        conj = data.reverse_map["stack"]["conjugate"]
        nstack = stacked_vis.shape[1]

        # Determine the time axis.  This will be an (nlsd, ntime) array.
        if "ra" in data.index_map:
            ra = data.ra
            lsd = np.atleast_1d(
                data.attrs["lsd"] if "lsd" in data.attrs else data.attrs["csd"]
            )

            if lsd.size > 1:
                raise RuntimeError(
                    "Currently only able to handle single sidereal days."
                )

            timestamp = ephemeris.csd_to_unix(lsd[0] + ra / 360.0)

        else:
            # The input container has a time axis.
            timestamp = data.time

        # Determine dimensions
        ntime = timestamp.size

        nfreq, ninput, nupdate = self.frac_error.shape

        # We may want to replace the code below with something
        # based on the dataset id.
        tindex = np.digitize(timestamp, self.timestamp) - 1

        self.log.info(f"Unique time indices are: {np.unique(tindex)}")

        before = tindex < 0
        if np.any(before):
            nbefore = np.sum(before)
            tbefore = np.max(self.timestamp.min() - timestamp[before]) / 3600.0
            self.log.warning(
                f"{nbefore:0.0f} requested timestamps are before the earliest "
                f"gain update time by as much as {tbefore:0.1f} hours."
            )

        after = tindex == (nupdate - 1)
        if np.any(after):
            nafter = np.sum(after)
            tafter = np.max(timestamp[after] - self.timestamp.max()) / 3600.0
            self.log.warning(
                f"{nafter:0.0f} requested timestamps are after the latest"
                f"gain update time by as much as {tafter:0.1f} hours."
            )

        # Determine if we have valid input flags
        try:
            input_flag = data.input_flags[:].astype(bool)
        except AttributeError:
            input_flag = None
        else:
            if not np.any(input_flag):
                input_flag = None

        # Different calculation whether or not we have input flags
        index = np.zeros((ntime, 2), dtype=int)
        index[:, 0] = tindex

        if self.ignore_input_flags or input_flag is None:
            self.log.info(f"Ignoring input flags ({self.ignore_input_flags})")
            uniq_input_flag = np.ones((ninput, 1), dtype=bool)

        else:
            uniq_input_flag, windex = np.unique(input_flag, return_inverse=True, axis=1)
            nuw = uniq_input_flag.shape[1]

            self.log.info(f"Found {nuw} unique sets of input flags.")

            index[:, 1] = windex

        uindex, oindex = np.unique(index, return_inverse=True, axis=0)
        nuniq = uindex.shape[0]

        # Loop over the unique combinations of gains and input flags
        for uu in range(nuniq):

            self.log.info(
                f"Processing unique gains/flags {uu} of {nuniq}. "
                f"Time index is {uindex[uu, 0]}.  Flag index is {uindex[uu, 1]}."
            )

            oind = np.flatnonzero(oindex == uu)

            flag = (
                uniq_input_flag[np.newaxis, :, uindex[uu, 1]]
                & self.flag[:, :, uindex[uu, 0]]
            )

            ratio = flag * self.frac_error[:, :, uindex[uu, 0]]

            # Calculate outer product
            flag = tools.fast_pack_product_array(
                flag[:, :, np.newaxis] & flag[:, np.newaxis, :]
            )

            vis = tools.fast_pack_product_array(
                ratio[:, :, np.newaxis] * ratio[:, np.newaxis, :].conj()
            )
            vis = np.where(conj, vis.conj(), vis)

            for ff in range(nfreq):

                corr = np.bincount(
                    stack, weights=vis[ff].real, minlength=nstack + 1
                ) + 1.0j * np.bincount(
                    stack, weights=vis[ff].imag, minlength=nstack + 1
                )

                count = np.bincount(
                    stack, weights=flag[ff].astype(float), minlength=nstack + 1
                )

                corr *= tools.invert_no_zero(count)

                stacked_vis[ff][:, oind] *= tools.invert_no_zero(
                    corr[:nstack, np.newaxis]
                )
                stacked_weight[ff][:, oind] *= np.abs(corr[:nstack, np.newaxis]) ** 2

        return data


class CollapseGainError(task.SingleTask):
    """Average gain errors over all pairs of feeds."""

    def process(self, gain):
        """Calculate the baseline-averaged gain error.

        Approximates the effect of the gain errors on
        the frequency dependence of a ringmap.

        Parameters
        ----------
        gain : StaticGainData
            Ratio of the applied gain to the "true" gain.
            The "true" gain is usually estimated by
            low-pass filtering along the frequency axis.

        Returns
        -------
        out : StaticGainData
            Baseline-averaged gain error placed into a
            StaticGainData container with a size 1
            input axis.
        """
        # Redistribute over freq
        gain.redistribute("freq")

        # Dereference datasets
        g = gain.gain[:].local_array
        w = gain.weight[:].local_array
        v = tools.invert_no_zero(w)

        nfreq, ninput = g.shape

        # Identify flagged data
        gflag = w > 0.0

        # Calculate the outer product of the gains
        flag = tools.fast_pack_product_array(
            gflag[:, :, np.newaxis] & gflag[:, np.newaxis, :]
        )
        vis = tools.fast_pack_product_array(
            g[:, :, np.newaxis] * g[:, np.newaxis, :].conj()
        )
        var = tools.fast_pack_product_array(v[:, :, np.newaxis] + v[:, np.newaxis, :])

        count = np.sum(flag, axis=-1)
        avg_vis = np.sum(flag * vis, axis=-1) * tools.invert_no_zero(count)
        avg_var = np.sum(flag * var, axis=-1) * tools.invert_no_zero(count**2)

        # Save to output container
        out = containers.StaticGainData(
            input=np.array(["baseline-averaged"]),
            axes_from=gain,
            attrs_from=gain,
            distributed=gain.distributed,
            comm=gain.comm,
        )

        out.add_dataset("weight")
        out.redistribute("freq")

        out.gain[:].local_array[:] = avg_vis[:, np.newaxis]
        out.weight[:].local_array[:] = tools.invert_no_zero(avg_var)[:, np.newaxis]

        return out


class EstimateNarrowbandGainError(task.SingleTask):
    """Estimate error in gains due to narrowband features.

    Attributes
    ----------
    ignore_rfi: bool
        Ignore the persistent RFI bands when calculating
        the gain error.  These bands are specified in
        ch_util.rfi.frequency_mask.
    """

    ignore_rfi = config.Property(proptype=bool, default=True)

    def process(self, gain, gain_mask, gain_smooth):
        """Construct the correction for narrowband gain errors.

        Parameters
        ----------
        gain: StaticGainData
            Original (unmasked) gains.
        gain_mask: StaticGainData
            Gains after masking.
        gain_smooth: StaticGainData
            Gains after masking and interpolating
            over the narrowband features.

        Returns
        -------
        out: StaticGainData
            Here the gain dataset is the ratio of the original gain
            and a version of the gain where all narrowband feature have
            been flagged and smoothly interpolated over.  The weight
            dataset is 1.0 if a narrowband feature was identified at
            that (freq, input, time) and 0.0 otherwise.
        """
        # Redistribute over input
        gain.redistribute("input")
        gain_mask.redistribute("input")
        gain_smooth.redistribute("input")

        # Dereference datasets
        gm = gain_mask.gain[:].local_array
        gs = gain_smooth.gain[:].local_array

        wi = gain.weight[:].local_array
        wm = gain_mask.weight[:].local_array

        # Create a mask that identifies newly flagged data
        mask = (wi > 0.0) & (wm == 0.0)

        if self.ignore_rfi:
            rfi_mask = rfi.frequency_mask(gain.freq)

            mask &= ~rfi_mask[:, np.newaxis]

        # Calculate the ratio of the gain and smooth version of the gain
        out = containers.StaticGainData(
            axes_from=gain,
            attrs_from=gain,
            distributed=gain.distributed,
            comm=gain.comm,
        )

        out.add_dataset("weight")
        out.redistribute("input")

        out.gain[:] = np.where(mask, gm * tools.invert_no_zero(gs), 1.0)
        out.weight[:] = mask.astype(np.float32)

        # Set the weight to zero for bad inputs
        out.weight[:].local_array[:] *= np.any(wi > 0.0, axis=0, keepdims=True).astype(
            np.float32
        )

        # Return the ratio of gains
        return out


class ConcatenateGains(task.SingleTask):
    """Repackage a list of StaticGainData/GainData into a single GainData container."""

    def process(self, gains):
        """Concatenate gain updates.

        Parameters
        ----------
        gains: list of GainData or StaticGainData
            List of gain updates.

        Returns
        -------
        out: GainData
            The list of gain updates sorted by time and
            placed in a single container with a time axis.
        """
        # Sort by time
        timestamp, index = [], []
        for ii, gain in enumerate(gains):
            t = list(gain.time) if "time" in gain.index_map else [gain.attrs["time"]]
            timestamp += t
            index += [(ii, jj) for jj in range(len(t))]
            gain.redistribute("freq")

        isort = np.argsort(timestamp)
        timestamp = np.array(timestamp)[isort]
        index = [index[iso] for iso in isort]

        g0 = gains[index[0][0]]
        out = containers.GainData(axes_from=g0, time=timestamp, comm=g0.comm)
        out.add_dataset("weight")

        if "update_id" in g0.datasets:
            out.add_dataset("update_id")

        out.redistribute("freq")

        for tt, (gg, ii) in enumerate(index):

            gain = gains[gg]

            if "time" in gain.index_map:
                out.gain[:, :, tt] = gain.gain[:, :, ii]
                out.weight[:, :, tt] = gain.weight[:, :, ii]
                if "update_id" in out.datasets:
                    out.update_id[tt] = gain.update_id[ii]

            else:
                out.gain[:, :, tt] = gain.gain[:]
                out.weight[:, :, tt] = gain.weight[:]
                if "update_id" in out.datasets:
                    out.update_id[tt] = gain.attrs["update_id"]

        return out


class FlagNarrowbandGainError(task.SingleTask):
    """Mask frequencies and times where narrowband gain errors were identified.

    Attributes
    ----------
    transition: float
        Duration in seconds over which we transitioned between gains in the
        real-time pipeline.
    threshold: float
        Mask any frequency and time where the fractional gain error is larger
        than this threshold.
    ignore_input_flags: bool
        Ignore the input flags when calculating the average gain error.
    """

    transition = config.Property(proptype=float, default=600.0)
    threshold = config.Property(proptype=float, default=1.0e-3)
    ignore_input_flags = config.Property(proptype=bool, default=False)

    def setup(self, gains):
        """Prepare the gain errors as a function of time.

        Parameters
        ----------
        gains: containers.GainData
            Narrowband gain errors generated by the
            EstimateNarrowbandGainError task.
        """
        gains.redistribute("freq")
        self.gains = gains

        # Identify the bad inputs when the gains were generated
        weight_local = np.any(gains.weight[:].local_array > 0.0, axis=0, keepdims=True)
        self.weight = np.zeros_like(weight_local)
        self.comm.Allreduce(weight_local, self.weight, op=MPI.LOR)

        # Compute the fractional error in the gain
        self.frac_error = gains.gain[:].local_array - 1.0

    def process(self, data):
        """Look up appropriate gain errors, construct flag, and apply to weights.

        Parameters
        ----------
        data: TimeStream or SiderealStream
            Data to flag.

        Returns
        -------
        out: RFIMask or SiderealRFIMask
            Mask that removes frequencies and times.
        """
        # Make sure the frequencies are the same
        if not np.array_equal(self.gains.freq, data.freq):
            raise ValueError("Frequencies do not match for gain error and timestream.")

        data.redistribute("freq")

        # Determine the time axis.  This will be an (nlsd, ntime) array.
        if "ra" in data.index_map:
            ra = data.ra
            lsd = np.atleast_1d(
                data.attrs["lsd"] if "lsd" in data.attrs else data.attrs["csd"]
            )

            timestamp = chime.lsd_to_unix(
                lsd[:, np.newaxis] + ra[np.newaxis, :] / 360.0
            )

            out = containers.SiderealRFIMask(axes_from=data, attrs_from=data)

        else:
            # The input container has a time axis.  Use a singleton lsd axis.
            timestamp = data.time[np.newaxis, :]

            out = containers.RFIMask(axes_from=data, attrs_from=data)

        # Determine dimensions
        nfreq, ninput, nupdate = self.frac_error.shape
        nlsd, ntime = timestamp.shape

        ftimestamp = timestamp.flatten()
        nftime = ftimestamp.size

        # We may want to replace the code below with something
        # based on the dataset id.  However this time-based look up
        # should work okay if we allow some reasonable transition window
        # during which we take the maximum possible error.
        index = np.zeros((2, nftime), dtype=int)

        index[0] = np.digitize(ftimestamp - self.transition, self.gains.time) - 1
        index[1] = np.digitize(ftimestamp, self.gains.time) - 1

        before = index[0] < 0
        if np.any(before):
            nbefore = np.sum(before)
            tbefore = np.max(self.gains.time.min() - ftimestamp[before]) / 3600.0
            raise RuntimeError(
                f"{nbefore:0.0f} requested timestamps are before the earliest "
                f"gain update time by as much as {tbefore:0.1f} hours."
            )

        after = index[1] == (nftime - 1)
        if np.any(after):
            nafter = np.sum(after)
            tafter = np.max(ftimestamp[after] - self.gains.time.max()) / 3600.0
            self.log.warning(
                f"{nafter:0.0f} requested timestamps are after the latest"
                f"gain update time by as much as {tafter:0.1f} hours."
            )

        # Determine if we have valid input flags
        try:
            w = data.input_flags[:].local_array
        except AttributeError:
            no_input_flags = True
        else:
            if np.any(w):
                no_input_flags = False
                w = w.astype(np.float32)
            else:
                # The sidereal stacks currently set all input flags to zero.
                # We want to ignore the input flags in this case.
                no_input_flags = True

        # Different calculation whether or not we have input flags
        if self.ignore_input_flags or no_input_flags:
            # We do not have an input flag.  Perform a straight average over all good inputs.
            err = np.sum(self.weight * self.frac_error, axis=1) * tools.invert_no_zero(
                np.sum(self.weight, axis=1)
            )

            # Use the gain update with the maximum error for times
            # where there may have been a transition between two gain updates
            err_avgi = np.where(
                np.abs(err[:, index[0]]) > np.abs(err[:, index[1]]),
                err[:, index[0]],
                err[:, index[1]],
            )

        else:
            w = w * tools.invert_no_zero(np.sum(w, axis=1, keepdims=True))
            w = w[np.newaxis, :, :]

            # Caculate the average error for the inputs that were
            # flagged as good at each time.
            err = np.zeros((2, nfreq, nftime), dtype=self.frac_error.dtype)

            # Loop over the two possible gain updates
            for ii, ind in enumerate(index):
                for tt, uu in enumerate(ind):
                    # The modulus with the number of times is for the
                    # sidereal stacks where we have gain errors on
                    # many days, but a single set of input flags.
                    wit = w[:, :, tt % ntime] * self.weight[:, :, uu]

                    err[ii, :, tt] = np.sum(
                        wit * self.frac_error[:, :, uu], axis=1
                    ) * tools.invert_no_zero(np.sum(wit, axis=1))

            # Again, during transitions we take the update with the maximum error.
            err_avgi = np.where(np.abs(err[0]) > np.abs(err[1]), err[0], err[1])

        # Average over sidereal days
        err_avgi_avgd = np.mean(err_avgi.reshape(nfreq, nlsd, ntime), axis=1)

        # Mask any frequency and time where the magnitude
        # of the input-averaged fractional gain error is greater
        # than the specified threshold
        mask = np.abs(err_avgi_avgd) > self.threshold

        # The RFIMask containers are not distributed.  Perform an allgather to
        # acquire the mask for all frequencies.
        out.mask[:] = mpiarray.MPIArray.wrap(mask, 0, comm=data.comm).allgather()

        # Print the total fraction of data flagged
        frac_masked = np.sum(out.mask[:]) / float(out.mask[:].size)
        self.log.info(
            f"Flagging {100 * frac_masked:0.2f}% "
            "of data due to narrow band gain errors."
        )

        return out


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

    rotation = config.Property(proptype=float, default=chime.rotation)
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
        self.log.info(f"Found {len(flags):d} {self.name_of_flag} flags in total.")
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
        -------
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
            timestamp = chime.lsd_to_unix(csd + ra / 360.0)
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
        freq = sstream.freq[sstream.vis[:].local_bounds]

        # Extract representative products for the stacked visibilities
        stack_new, stack_flag = tools.redefine_stack_index_map(
            inputmap, sstream.prod, sstream.stack, sstream.reverse_map["stack"]
        )
        do_not_apply = np.flatnonzero(~stack_flag)
        prod = sstream.prod[stack_new["prod"]].copy()

        # Swap the product pair order for conjugated stack indices
        cj = np.flatnonzero(stack_new["conjugate"].astype(bool))
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
                datestr_start = ctime.unix_to_datetime(flag.start_time).strftime(
                    "%Y%m%dT%H%M%SZ"
                )
                datestr_end = ctime.unix_to_datetime(flag.finish_time).strftime(
                    "%Y%m%dT%H%M%SZ"
                )
                msg = (
                    f"{np.sum(in_range):d} (of {in_range.size:d}) samples require "
                    f"phase correction according to {self.name_of_flag} DataFlag "
                    f"covering {datestr_start} to {datestr_end}."
                )

                self.log.info(msg)

                correction = self._get_correction(
                    freq, prod, timestamp[in_range], inputmap, **flag.metadata
                )

                if do_not_apply.size > 0:
                    self.log.warning(
                        f"Do not have valid baseline distance for stack indices: {do_not_apply!s}"
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
            f"Applying a phase correction for a {time_offset:0.2f} second "
            f"time offset on the calibrator {calibrator}."
        )

        body = sources.source_dictionary[calibrator]

        lat = np.radians(chime.latitude)

        # Compute feed positions with rotation
        tools.change_chime_location(rotation=self.rotation)
        uv = _calculate_uv(freq, prod, inputmap)

        # Return back to default rotation
        tools.change_chime_location(default=True)

        # Determine location of calibrator
        ttrans = chime.transit_times(body, timestamp[0] - 24.0 * 3600.0)[0]

        ra, dec = chime.object_coords(body, date=ttrans, deg=False)

        ha = np.radians(chime.unix_to_lsa(ttrans + time_offset)) - ra

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
            f"of {rotation:.3f} deg to {self.rotation:.3f} deg for the calibrator {calibrator}."
        )

        body = sources.source_dictionary[calibrator]

        lat = np.radians(chime.latitude)

        # Compute feed positions with old rotation
        tools.change_chime_location(rotation=rotation)
        old_uv = _calculate_uv(freq, prod, inputmap)

        # Compute feed positions with current rotation
        tools.change_chime_location(rotation=self.rotation)
        current_uv = _calculate_uv(freq, prod, inputmap)

        # Return back to default rotation
        tools.change_chime_location(default=True)

        # Determine location of calibrator
        ttrans = chime.transit_times(body, timestamp[0] - 24.0 * 3600.0)[0]

        ra, dec = chime.object_coords(body, date=ttrans, deg=False)

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

    return dist[:, np.newaxis, :] / lmbda[np.newaxis, :, np.newaxis]
