"""Tasks for analysis of the radio sun.

Includes grouping individual files into a solar day;
solar calibration; solar beamforming; and solar excision.
"""

import datetime

import caput.astro.time as ctime
import numpy as np
import pytz
import scipy.constants
import skyfield
from caput import config
from caput.astro.skyfield import skyfield_wrapper
from caput.containers import tod
from caput.pipeline import tasklib
from ch_ephem import coord, sources
from ch_ephem.observers import chime
from ch_util import cal_utils, tools
from draco.util import interferometry

from ..core import containers


def unix_to_localtime(unix_time):
    """Converts unix time to a :class:`datetime.datetime` object.

    Parameters
    ----------
    unix_time : float
        Unix/POSIX time.

    Returns
    -------
    dt : :class:`datetime.datetime`

    """
    utc_time = pytz.utc.localize(datetime.datetime.utcfromtimestamp(unix_time))

    return utc_time.astimezone(pytz.timezone("Canada/Pacific"))


def sun_coord(unix_time, deg=True):
    """Calculate the coordinates of the sun at a given time.

    Parameters
    ----------
    unix_time : np.ndarray
        1D array of size `ntime` containing unix timestamps.

    deg : bool
        Return all coordinates in degrees,
        otherwise return all coordinates in radians.

    Returns
    -------
    coord : np.ndarray
        2D array of type `float` with shape `(ntime, 4)` that contains
        the hour angle, declination, altitude, and azimuth of the
        sun at each time.

    """
    date = ctime.ensure_unix(np.atleast_1d(unix_time))
    skyfield_time = ctime.unix_to_skyfield_time(date)
    ntime = date.size

    coord = np.zeros((ntime, 4), dtype=np.float64)

    planets = skyfield_wrapper.ephemeris
    # planets = skyfield.api.load('de421.bsp')
    sun = planets["sun"]

    observer = chime.skyfield_obs()

    apparent = observer.at(skyfield_time).observe(sun).apparent()

    radec = apparent.cirs_radec(epoch=skyfield_time)
    coord[:, 0] = radec[0].radians
    coord[:, 1] = radec[1].radians

    altaz = apparent.altaz()
    coord[:, 2] = altaz[0].radians
    coord[:, 3] = altaz[1].radians

    # Convert to hour angle
    # defined as local stellar angle minus source right ascension
    coord[:, 0] = _correct_phase_wrap(np.radians(chime.unix_to_lsa(date)) - coord[:, 0])

    if deg:
        coord = np.degrees(coord)

    return coord


class SolarGrouper(tasklib.base.ContainerTask):
    """Group individual timestreams together into whole solar days.

    Attributes
    ----------
    min_span : float
        The minimum solar day length (in hours) to process.
        Default is 2.
    """

    min_span = config.Property(proptype=float, default=2.0)

    def __init__(self):
        super().__init__()
        self._timestream_list = []
        self._current_day = None

    def process(self, tstream):
        """Load in each solar day.

        Parameters
        ----------
        tstream : andata.CorrData
            Timestream to group together.

        Returns
        -------
        ts : andata.CorrData or None
            Returns the timestream of each solar day when we have received
            the last file, otherwise returns :obj:`None`.
        """
        # Get the start and end day of the file as an int with format YYYYMMDD
        day_start = int(unix_to_localtime(tstream.time[0]).strftime("%Y%m%d"))
        day_end = int(unix_to_localtime(tstream.time[-1]).strftime("%Y%m%d"))

        # If current_day is None then this is the first time we've run
        if self._current_day is None:
            self._current_day = day_start

        # If this file started during the current day add it onto the list
        if self._current_day == day_start:
            self._timestream_list.append(tstream)

        self.log.info("Adding file into group for date: %i", day_start)

        # If this file ends during a later day then we need to process the
        # current list and restart the system
        if self._current_day < day_end:
            self.log.info("Concatenating files for date: %i", day_start)

            # Combine timestreams into a single container for the whole day this
            # could get returned as None if there wasn't enough data
            tstream_all = self._process_current_day()

            # Reset list and current day for the new file
            self._timestream_list = [tstream]
            self._current_day = day_end

            return tstream_all

        return None

    def process_finish(self):
        """Return the final day.

        Returns
        -------
        ts : andata.CorrData or None
            Returns the timestream of the final day if it's long
            enough, otherwise returns :obj:`None`.
        """
        # If we are here there is no more data coming, we just need to process any remaining data
        return self._process_current_day()

    def _process_current_day(self):
        # Combine the current set of files into a timestream

        day = str(self._current_day)

        # Calculate the length of data in the current day
        start = datetime.datetime.utcfromtimestamp(self._timestream_list[0].time[0])
        end = datetime.datetime.utcfromtimestamp(self._timestream_list[-1].time[-1])
        tdelta = end - start
        day_length = tdelta.days * 24.0 + tdelta.seconds / 3600.0

        # If the amount of data for this day is too small, then just skip
        if day_length < self.min_span:
            return None

        self.log.info("Constructing %s [%i files]", day, len(self._timestream_list))

        # Construct the combined timestream
        ts = tod.concatenate(self._timestream_list)

        # Add attributes for the date and a tag for labelling saved files
        ts.attrs["tag"] = day
        ts.attrs["date"] = self._current_day

        return ts


class SolarCalibrationN2(tasklib.base.ContainerTask):
    """Use Sun to measure antenna beam pattern.

    Must be run prior to averaging redundant baselines.

    Attributes
    ----------
    dualpol: bool, default True
        Model all polarization products together.
        Otherwise model XX and YY independently.
    extended:  bool, default True
        Model the extended nature of the sun.
    hermweight:  bool, default True
        Normalize the hermite polynomials used to model the extended
        nature of the sun.  Only relevant if extended is True.
    neigen: int, default 2
        Number of eigenvalues to use to model response to sun.
    ymin: float, default 1.2
        Do not include baselines with N-S separation
        less than ymin in the fit.  Only relevant if
        extended is True.
    max_iter: int, default 4
        Maximum number of iterations.
    """

    dualpol = config.Property(proptype=bool, default=True)
    extended = config.Property(proptype=bool, default=True)
    hermweight = config.Property(proptype=bool, default=True)
    neigen = config.Property(proptype=int, default=2)
    ymin = config.Property(proptype=float, default=1.2)
    max_iter = config.Property(proptype=int, default=4)

    def process(self, sstream, inputmap, inputmask):
        """Determine solar response from input timestream.

        Parameters
        ----------
        sstream : andata.CorrData or containers.SiderealStream
            Timestream collected during the day.  Must contain
            the upper triangle of the N^2 visibility matrix.
        inputmap : list of :class:`CorrInput`
            A list describing the inputs as they are in the file.
        inputmask : containers.CorrInputMask
            Mask indicating which correlator inputs to use in the
            eigenvalue decomposition.

        Returns
        -------
        suntrans : containers.SunTransit
            Response to the sun.
        """
        from itertools import groupby
        from operator import itemgetter

        from .calibration import _extract_diagonal, solve_gain

        # Hardcoded parameters related to size of sun
        poly_deg = np.array([3, 3])
        scale = np.array([np.radians(0.5), np.radians(0.5)])

        poly_ncoeff = poly_deg + 1

        # Ensure that we are distributed over frequency
        sstream.redistribute("freq")

        # Find the local frequencies
        nfreq = sstream.vis.local_shape[0]
        sfreq = sstream.vis.local_offset[0]
        efreq = sfreq + nfreq

        # Get the local frequency axis
        freq = sstream.freq[sfreq:efreq]
        wv = scipy.constants.c / (freq * 1e6)

        # Get times (ra in degrees)
        if hasattr(sstream, "time"):
            time = sstream.time
            ra = chime.unix_to_lsa(time)
        else:
            ra = sstream.ra
            csd = (
                sstream.attrs["lsd"] if "lsd" in sstream.attrs else sstream.attrs["csd"]
            )
            time = chime.lsd_to_unix(csd + ra / 360.0)

        # Only examine data between sunrise and sunset
        time_flag = np.zeros(len(time), dtype=bool)
        rise = chime.solar_rising(time[0] - 24.0 * 3600.0, t1=time[-1])
        for rr in rise:
            ss = chime.solar_setting(rr)[0]
            time_flag |= (time >= rr) & (time <= ss)

        if not np.any(time_flag):
            self.log.debug(
                "No daytime data between %s and %s.",
                ctime.unix_to_datetime(time[0]).strftime("%b %d %H:%M"),
                ctime.unix_to_datetime(time[-1]).strftime("%b %d %H:%M"),
            )
            return None

        # Convert boolean flag to slices
        time_index = np.flatnonzero(time_flag)

        time_slice = []
        ntime = 0
        for key, group in groupby(
            enumerate(time_index), lambda index_item: index_item[0] - index_item[1]
        ):
            group = list(map(itemgetter(1), group))
            ngroup = len(group)

            time_slice.append(
                (range(group[0], group[-1] + 1), range(ntime, ntime + ngroup))
            )
            ntime += ngroup

        time = np.concatenate([time[slc[0]] for slc in time_slice])
        ra = np.concatenate([ra[slc[0]] for slc in time_slice])

        # Get ra, dec, alt of sun (in radians)
        sun_pos = sun_coord(time, deg=False)

        # Determine good inputs
        nfeed = len(inputmap)

        good_input = np.flatnonzero(inputmask.datasets["input_mask"][:])

        prodmap = sstream.index_map["prod"][:]

        # Use input map to figure out which are the X and Y feeds
        xfeeds = np.array(
            [
                idx
                for idx, inp in enumerate(inputmap)
                if tools.is_chime_x(inp) and (idx in good_input)
            ]
        )
        yfeeds = np.array(
            [
                idx
                for idx, inp in enumerate(inputmap)
                if tools.is_chime_y(inp) and (idx in good_input)
            ]
        )

        self.log.info(
            f"Performing sun calibration with {len(good_input):d}/{nfeed:d} "
            f"good feeds ({len(xfeeds):d} xpol, {len(yfeeds):d} ypol)."
        )

        # Construct baseline vector for each visibility
        feed_pos = tools.get_feed_positions(inputmap)
        vis_pos = np.array([feed_pos[ii] - feed_pos[ij] for ii, ij in prodmap])
        vis_pos = np.where(np.isnan(vis_pos), np.zeros_like(vis_pos), vis_pos)

        # Deal with different options for fitting dual polarisation data
        if self.dualpol:
            feeds = np.sort(np.concatenate((xfeeds, yfeeds)))

            prods = np.array(
                [
                    idx
                    for idx, (fi, fj) in enumerate(prodmap)
                    if ((fi in feeds) and (fj in feeds))
                ]
            )

            polmap = [(feeds, prods)]

            # Create container to hold results of fit
            suntrans = containers.SunTransit(
                time=time,
                eigen=self.neigen,
                pol=np.array(["DUAL"]),
                good_input1=feeds,
                good_input2=feeds,
                udegree=poly_ncoeff[0],
                vdegree=poly_ncoeff[1],
                axes_from=sstream,
                attrs_from=sstream,
            )

        else:
            xprods = np.array(
                [
                    idx
                    for idx, (fi, fj) in enumerate(prodmap)
                    if ((fi in xfeeds) and (fj in xfeeds))
                ]
            )
            yprods = np.array(
                [
                    idx
                    for idx, (fi, fj) in enumerate(prodmap)
                    if ((fi in yfeeds) and (fj in yfeeds))
                ]
            )

            polmap = [(xfeeds, xprods), (yfeeds, yprods)]

            # Create container to hold results of fit
            suntrans = containers.SunTransit(
                time=time,
                eigen=self.neigen,
                pol=np.array(["XX", "YY"]),
                good_input1=xfeeds,
                good_input2=yfeeds,
                udegree=poly_ncoeff[0],
                vdegree=poly_ncoeff[1],
                axes_from=sstream,
                attrs_from=sstream,
            )

            suntrans.add_dataset("evalue2")

        # Initialize datasets
        if self.extended:
            suntrans.add_dataset("coeff")

        suntrans.redistribute("freq")

        for key in suntrans.datasets.keys():
            suntrans.datasets[key][:] = 0.0

        # Set coordinates (in radians)
        suntrans.coord[:] = sun_pos

        # Create slice for expanding dimensions for solve_gain
        edim = (None, slice(None), None)

        # Loop over polarizations
        for ipol, (ifeed, iprod) in enumerate(polmap):
            iauto = np.array(
                [idx for idx, (fi, fj) in enumerate(prodmap[iprod]) if (fi == fj)]
            )
            iadj = np.flatnonzero(
                (np.abs(vis_pos[iprod, 1]) <= self.ymin)
                & (np.abs(vis_pos[iprod, 0]) < 10.0)
            )
            intercyl = np.flatnonzero(np.abs(vis_pos[iprod, 0]) > 10.0)

            polid = np.array(
                [
                    2 * int(fi in yfeeds) + int(fj in yfeeds)
                    for (fi, fj) in prodmap[iprod]
                ]
            )
            uniq_polid = np.unique(polid)

            icross = np.ones(iprod.size, dtype=bool)
            # icross[iadj] = False
            icross[iauto] = False
            icross = np.flatnonzero(icross)

            # Loop over frequency
            for ff_local, ff_global in enumerate(range(sfreq, efreq)):
                # Create baseline vectors
                u = vis_pos[iprod, 0] / wv[ff_local]
                v = vis_pos[iprod, 1] / wv[ff_local]

                # Create hermite polynomials to model extended emission
                if self.extended:
                    H = np.polynomial.hermite.hermvander2d(
                        u * scale[0], v * scale[1], poly_deg
                    ).astype(np.complex64)
                    if self.hermweight:
                        H *= np.polynomial.hermite.hermweight(u * scale[0])[
                            :, np.newaxis
                        ]
                        H *= np.polynomial.hermite.hermweight(v * scale[1])[
                            :, np.newaxis
                        ]

                # Define windows around bright point source transits
                source_window = []
                for ss, src in sources.source_dictionary.iteritems():
                    if isinstance(src, skyfield.starlib.Star):
                        peak_ra = coord.peak_ra(src, deg=True)
                        window_ra = 3.0 * cal_utils.guess_fwhm(
                            freq[ff_local],
                            pol=["X", "Y"][ipol],
                            dec=src._dec,
                            sigma=True,
                        )
                        source_window.append((src, peak_ra, window_ra))

                # Loop over time slices
                for slc_in, slc_out in time_slice:
                    # Loop over times within a slice
                    for tt_in, tt_out in zip(slc_in, slc_out):
                        # Extract visibility and weight at this frequency and time
                        vis = (
                            sstream.vis[ff_global, iprod, tt_in].view(np.ndarray).copy()
                        )
                        weight = (
                            sstream.weight[ff_global, iprod, tt_in]
                            .view(np.ndarray)
                            .copy()
                        )

                        # Set weight for autocorrelation and adjacent feeds equal to zero
                        weight[iauto] = 0.0
                        weight[iadj] = 0.0

                        # Extract the diagonal (to be used for weighting)
                        norm = (_extract_diagonal(vis, axis=0).real) ** 0.5
                        norm = tools.invert_no_zero(norm)

                        # Project out bright point sources so that they do not confuse the sun calibration
                        for src, center, span in source_window:
                            sha = _correct_phase_wrap(ra[tt_out] - center, deg=True)

                            if np.abs(sha) < span:
                                src_phase = interferometry.fringestop_phase(
                                    np.radians(sha),
                                    np.radians(chime.latitude),
                                    src._dec,
                                    u,
                                    v,
                                )

                                for upid in uniq_polid:
                                    pp = np.flatnonzero(polid == upid)

                                    asrc = np.sum(
                                        weight[pp] * vis[pp] * src_phase[pp]
                                    ) * tools.invert_no_zero(np.sum(weight[pp]))

                                    vis[pp] -= asrc * src_phase[pp].conj()

                        # Fringestop
                        vis *= interferometry.fringestop_phase(
                            sun_pos[tt_out, 0],
                            np.radians(chime.latitude),
                            sun_pos[tt_out, 1],
                            u,
                            v,
                        )

                        # Solve for the solar response
                        ev, resp, err_resp = (
                            np.squeeze(var)
                            for var in solve_gain(
                                vis[edim], norm=norm[edim], neigen=self.neigen
                            )
                        )

                        if len(resp.shape) == 1:
                            resp = resp[:, np.newaxis]
                            err_resp = err_resp[:, np.newaxis]

                        G = _upper_triangle_gain_vector(resp)

                        # Analysis of extended source
                        if self.extended:
                            A = G[:, np.newaxis] * H

                            iters = 0
                            while iters < self.max_iter:
                                vism = vis.copy()

                                # Calculate covariance of model coefficients
                                C = np.dot(A.T.conj(), weight[:, np.newaxis] * A)

                                # Solve for model coefficients
                                coeff = np.linalg.lstsq(
                                    C, np.dot(A.T.conj(), weight * vis)
                                )[0]

                                # Compute model for extended source structure
                                model = np.dot(H, coeff)

                                # Correct for the extended source structure
                                vism[icross] *= tools.invert_no_zero(model[icross])

                                # Re-solve for the solar response
                                ev, resp, err_resp = (
                                    np.squeeze(var)
                                    for var in solve_gain(
                                        vism[edim], norm=norm[edim], neigen=self.neigen
                                    )
                                )

                                if len(resp.shape) == 1:
                                    resp = resp[:, np.newaxis]
                                    err_resp = err_resp[:, np.newaxis]

                                G = _upper_triangle_gain_vector(resp)

                                A = G[:, np.newaxis] * H

                                # Increase iteration counter
                                iters += 1

                            # Save model coefficients to output container
                            suntrans.coeff[ff_global, ipol, tt_out, :, :] = (
                                coeff.reshape(*poly_ncoeff)
                            )

                            # Calculate residual
                            residual = vis - np.dot(A, coeff)

                        else:
                            # Not modeling extended source, just calculate residual
                            residual = vis - G

                        # Check reduction in intercylinder power to determine if we should perform subtraction
                        chisq_before = np.sum(
                            weight[intercyl] * np.abs(vis[intercyl]) ** 2
                        )
                        chisq_after = np.sum(
                            weight[intercyl] * np.abs(residual[intercyl]) ** 2
                        )

                        suntrans.is_sun[ff_global, ipol, tt_out] = (
                            chisq_before * tools.invert_no_zero(chisq_after) - 1.0
                        )

                        # Save results to container
                        suntrans.response[ff_global, ifeed, tt_out, :] = resp
                        suntrans.response_error[ff_global, ifeed, tt_out, :] = err_resp

                        if ipol == 0:
                            suntrans.evalue1[ff_global, :, tt_out] = ev
                        else:
                            suntrans.evalue2[ff_global, :, tt_out] = ev

        # Update attributes
        units = "sqrt(" + sstream.vis.attrs.get("units", "correlator-units") + ")"
        suntrans.response.attrs["units"] = units
        suntrans.response_error.attrs["units"] = units

        suntrans.attrs["source"] = "Sun"

        suntrans.attrs["uscale"] = scale[0]
        suntrans.attrs["vscale"] = scale[1]
        suntrans.attrs["hermweight"] = self.hermweight
        suntrans.attrs["ymin"] = self.ymin

        # Return sun transit
        return suntrans


class SolarCleanN2(tasklib.base.ContainerTask):
    """Clean sun from daytime data.

    Subtracts a model for the sun determined by the
    SolarCalibrationN2 task from the N^2 visibilities.

    Attributes
    ----------
    threshold : float, default 2.5
        Do not subtract sun if the is_sun metric defined in
        SolarCalibration module is less than threshold.
    """

    threshold = config.Property(proptype=float, default=2.5)

    def process(self, sstream, suntrans, inputmap):
        """Clean the sun.

        Parameters
        ----------
        sstream : andata.CorrData or containers.SiderealStream
            Timestream collected during the day.
        suntrans : containers.SolarTransit
            Response to the sun.
        inputmap : list of :class:`CorrInput`
            A list describing the inputs as they are in the file.

        Returns
        -------
        mstream : containers.SiderealStream
            Sidereal stream with sun removed
        """
        # Redistribute over frequency
        sstream.redistribute("freq")
        suntrans.redistribute("freq")

        # Find the local frequencies
        nfreq = sstream.vis.local_shape[0]
        sfreq = sstream.vis.local_offset[0]
        efreq = sfreq + nfreq

        freq = sstream.freq[sfreq:efreq]
        wv = scipy.constants.c / (freq * 1e6)

        # Determine time mapping
        if hasattr(sstream, "time"):
            stime = sstream.time
        else:
            ra = sstream.ra
            csd = (
                sstream.attrs["lsd"] if "lsd" in sstream.attrs else sstream.attrs["csd"]
            )
            stime = chime.lsd_to_unix(csd + ra / 360.0)

        # Extract gain array
        gtime = suntrans.time[:]
        gain = suntrans.response[:].view(np.ndarray)

        # Match ra
        match = np.array([np.argmin(np.abs(gt - stime)) for gt in gtime])
        ntime = gtime.size

        # Determine product map
        prodmap = sstream.index_map["prod"][:]
        nprod = prodmap.size
        ninput = gain.shape[1]

        if nprod != (ninput * (ninput + 1) // 2):
            raise Exception("Number of inputs does not match the number of products.")

        feed_list = [(inputmap[ii], inputmap[jj]) for ii, jj in prodmap]

        # Determine polarisation for each visibility
        pol_ind = np.full(len(feed_list), -1, dtype=np.int64)
        for bb, (fi, fj) in enumerate(feed_list):
            if tools.is_chime(fi) and tools.is_chime(fj):
                pol_ind[bb] = 2 * tools.is_chime_y(fi) + tools.is_chime_y(fj)

        npol = suntrans.index_map["pol"].size
        if npol == 1:
            pol_ind = (pol_ind < 0).astype(np.int64)
            pol_sub = np.array([0])
        elif npol == 2:
            pol_sub = np.array([0, 3])
        else:
            ValueError(f"npol = {npol:d}, must be either 1 or 2.")

        # Construct baseline vector for each visibility
        feed_pos = tools.get_feed_positions(inputmap)
        vis_pos = np.array([feed_pos[ii] - feed_pos[jj] for ii, jj in prodmap])
        vis_pos = np.where(np.isnan(vis_pos), np.zeros_like(vis_pos), vis_pos)

        # Extract coordinates
        ha = suntrans.coord[:, 0]
        dec = suntrans.coord[:, 1]

        # Set up for removal
        scale = np.array([suntrans.attrs["uscale"], suntrans.attrs["vscale"]])
        hermweight = suntrans.attrs["hermweight"]
        extended = "coeff" in suntrans.datasets

        # Loop over frequencies
        for ff_local, ff_global in enumerate(range(sfreq, efreq)):
            # Loop over polarisations
            for pp, psub in enumerate(pol_sub):
                # Apply threshold to the is_sun metric defined in SolarCalibration
                # to determine what time samples to subtract.
                subtract_sun = (
                    suntrans.is_sun[ff_global, pp, :] > self.threshold
                ).astype(np.float64)

                # Determine baselines for this polarisation
                this_pol = np.flatnonzero(pol_ind == psub)

                # Loop over baselines
                for bb in this_pol:
                    ii, jj = prodmap[bb]

                    # Do not subtract from autocorrelations
                    if ii != jj:
                        # Create baseline vectors
                        u = vis_pos[bb, 0] / wv[ff_local]
                        v = vis_pos[bb, 1] / wv[ff_local]

                        # Determine phase of sun
                        sunphase = interferometry.fringestop_phase(
                            ha, np.radians(chime.latitude), dec, u, v
                        ).conj()

                        # Determine model for sun's extended emission
                        if extended:
                            coeff = np.rollaxis(
                                suntrans.coeff[ff_global, pp, :, :, :], 0, 3
                            )
                            model = np.polynomial.hermite.hermval2d(
                                u * scale[0], v * scale[1], coeff
                            )
                            if hermweight:
                                model *= np.polynomial.hermite.hermweight(
                                    u * scale[0]
                                ) * np.polynomial.hermite.hermweight(v * scale[1])
                        else:
                            model = np.ones(ntime, dtype=np.float64)

                        # Fetch the gains
                        gi = gain[ff_local, ii, :, :]
                        gj = gain[ff_local, jj, :, :].conj()

                        gout = np.sum(gi * gj, axis=-1)

                        # Outer product of the gains times the sun phase
                        # Subtract the outer product of the gains times the sun phase
                        sstream.vis[ff_global, bb, match] -= (
                            subtract_sun * model * gout * sunphase
                        )

        # Return the clean sidereal stream
        return sstream


class SolarBeamform(tasklib.base.ContainerTask):
    """Estimate the average primary beam by beamforming on the Sun.

    Formerly called SunCalibration.

    Attributes
    ----------
    ymax: float, default 10.0
        Do not include baselines with N-S separation
        greater than ymax to avoid resolving out the Sun.
        Default is 10.0 (meters)
    exclude_intercyl : bool, default True
        Exclude intercylinder baselines to avoid resolving
        out the Sun. Default is True
    sep_cyl : bool, default False
        Do not average over cylinder pairs when beamforming.
        If False, will yield a single measurement of the sun.
        If True will yield a separate measurement of the sun
        for each cylinder pair.  Default is False.
    """

    ymax = config.Property(proptype=float, default=10.0)
    exclude_intercyl = config.Property(proptype=bool, default=True)
    sep_cyl = config.Property(proptype=bool, default=False)

    def process(self, sstream, inputmap):
        """Beamform visibilities to the location of the Sun.

        Parameters
        ----------
        sstream: andata.CorrData, containers.TimeStream, containers.SiderealStream
            Timestream collected during the day.
        inputmap : list of :class:`CorrInput`
            A list describing the inputs as they are in the file.

        Returns
        -------
        sunstream : containers.FormedBeamTime
            Formed beam at the location of the sun.
        """
        # Determine the time axis
        if hasattr(sstream, "time"):
            time = sstream.time
        else:
            csd = (
                sstream.attrs["lsd"] if "lsd" in sstream.attrs else sstream.attrs["csd"]
            )
            time = chime.lsd_to_unix(csd + sstream.ra / 360.0)

        lat = np.radians(chime.latitude)

        # Get position of sun at every time sample (in radians)
        sun_pos = sun_coord(time, deg=False)

        ha = sun_pos[:, 0]
        dec = sun_pos[:, 1]
        el = sun_pos[:, 2]

        # Only process times when sun is above the horizon
        valid_time = np.flatnonzero(el > 0.0)

        if valid_time.size == 0:
            return None

        # Redistribute over frequency
        sstream.redistribute("freq")

        freq = sstream.freq[sstream.vis[:].local_bounds]
        wv = scipy.constants.c / (freq * 1e6)

        # Get polarisations of feeds
        pol = tools.get_feed_polarisations(inputmap)

        # Get positions of feeds
        pos = tools.get_feed_positions(inputmap)

        # Get cylinder each feed is on
        cyl = np.array(
            [chr(63 + inp.cyl) if tools.is_chime(inp) else "X" for inp in inputmap]
        )

        # Make sure that none of our typical products reference non-array feeds
        stack_new, stack_flag = tools.redefine_stack_index_map(
            inputmap, sstream.prod, sstream.stack, sstream.reverse_map["stack"]
        )

        valid_stack = np.flatnonzero(stack_flag)
        ninvalid = stack_new.size - valid_stack.size

        if ninvalid > 0:
            stack_new = stack_new[valid_stack]
            self.log.info(
                "Could not find appropriate reference inputs for "
                f"{ninvalid:0.0f} stacked products.  Ignoring these "
                "products in solar beamform."
            )

        # Extract the typical products for each stack.
        # Make sure to swap inputs if the data was conjugated.
        prodstack = _swap_inputs(
            sstream.prod[stack_new["prod"]], stack_new["conjugate"]
        )

        # Figure out what data we will need to conjugate in order to convert YX to XY
        conj_pol = pol[prodstack["input_a"]] > pol[prodstack["input_b"]]

        prodstack = _swap_inputs(prodstack, conj_pol)

        nconj = np.sum(conj_pol)
        if nconj > 0:
            self.log.debug(f"Conjugating {nconj} products (of {conj_pol.size}).")

        # Calculate baseline distance, polarisation pair, and cylinder pair
        index_a = prodstack["input_a"]
        index_b = prodstack["input_b"]

        bdist = pos[index_a] - pos[index_b]
        bpol = np.core.defchararray.add(pol[index_a], pol[index_b])
        bcyl = np.core.defchararray.add(cyl[index_a], cyl[index_b])

        # Exclude autocorrelations
        flag = index_a != index_b

        # Exclude intercylinder baselines if requested
        if self.exclude_intercyl:
            flag &= cyl[index_a] == cyl[index_b]

        # Exclude long north-south baselines if requested
        if self.ymax is not None:
            flag &= np.abs(bdist[:, 1]) <= self.ymax

        # Get the indices into the stack axis that will be processed
        to_process = np.flatnonzero(flag)

        bdist = bdist[to_process]
        conj_pol = conj_pol[to_process]
        ikeep = valid_stack[to_process]

        # Group the polarisation pairs
        upol, pol_map = np.unique(bpol[to_process], return_inverse=True)
        npol = upol.size

        # If requested group the cylinder pairs
        if self.sep_cyl:
            ucyl, cyl_map = np.unique(bcyl[to_process], return_inverse=True)
            index = cyl_map * npol + pol_map
        else:
            ucyl = np.array(["all"])
            index = pol_map

        ncyl = ucyl.size
        object_id = np.empty(ncyl, dtype=[("source", "<U16"), ("cylinder", "<U3")])
        object_id["source"] = "sun"
        object_id["cylinder"] = ucyl

        # Create output container
        sunstream = containers.FormedBeamTime(
            time=time[valid_time],
            object_id=object_id,
            pol=upol,
            axes_from=sstream,
            attrs_from=sstream,
            distributed=sstream.distributed,
            comm=sstream.comm,
        )

        sunstream.redistribute("freq")
        sunstream.beam[:] = 0.0
        sunstream.weight[:] = 0.0

        # Dereference datasets
        vis_local = sstream.vis[:].local_array
        weight_local = sstream.weight[:].local_array

        vis_out = sunstream.beam[:].local_array
        weight_out = sunstream.weight[:].local_array

        nfreq = vis_local.shape[0]

        # Iterate over frequencies
        for fi in range(nfreq):
            # Get the baselines in wavelengths
            u = bdist[:, 0] / wv[fi]
            v = bdist[:, 1] / wv[fi]

            # Iterate over times
            for tt, ti in enumerate(valid_time):
                # Initialize the visiblities matrix
                vis = vis_local[fi, ikeep, ti]
                weight = weight_local[fi, ikeep, ti]

                vis = np.where(conj_pol, vis.conj(), vis)

                # Calculate the phase that the sun would have using the fringestop routine
                sun_vis = interferometry.fringestop_phase(ha[ti], lat, dec[ti], u, v)

                # Fringestop to the sun
                vs = weight * vis * sun_vis

                # Accumulate the fringestopped visibilities based on what group their
                # baseline belongs to (as specified by index)
                vds = np.bincount(
                    index, weights=vs.real, minlength=ncyl * npol
                ) + 1.0j * np.bincount(index, weights=vs.imag, minlength=ncyl * npol)

                sds = np.bincount(index, weights=weight, minlength=ncyl * npol)

                isds = tools.invert_no_zero(sds)

                vis_out[:, :, fi, tt] = (vds * isds).reshape(ncyl, npol)
                weight_out[:, :, fi, tt] = sds.reshape(ncyl, npol)

        # Return the beamformed data
        return sunstream


class SolarClean(tasklib.base.ContainerTask):
    """Clean the sun from data by projecting out signal from its location.

    Formerly called SunClean.
    """

    def process(self, sstream, inputmap):
        """Clean the sun.

        Parameters
        ----------
        sstream: andata.CorrData, containers.TimeStream, or containers.SiderealStream
            Timestream collected during the day.
        inputmap : list of :class:`CorrInput`
            A list describing the inputs as they are in the file.

        Returns
        -------
        sscut : andata.CorrData, containers.TimeStream, or containers.SiderealStream
            Sidereal stack with sun projected out.
        """
        # Get the unix time
        if hasattr(sstream, "time"):
            times = sstream.time
        else:
            csd = (
                sstream.attrs["lsd"] if "lsd" in sstream.attrs else sstream.attrs["csd"]
            )
            times = chime.lsd_to_unix(csd + sstream.ra / 360.0)

        lat = np.radians(chime.latitude)

        # Get position of sun at every time sample (in radians)
        sun_pos = sun_coord(times, deg=False)

        ha = sun_pos[:, 0]
        dec = sun_pos[:, 1]
        el = sun_pos[:, 2]

        # Only process times when sun is above the horizon
        valid_time = np.flatnonzero(el > 0.0)

        if valid_time.size == 0:
            return sstream

        # Redistribute over frequency
        sstream.redistribute("freq")

        freq = sstream.freq[sstream.vis[:].local_bounds]
        wv = scipy.constants.c / (freq * 1e6)

        # Get polarisations of feeds
        pol = tools.get_feed_polarisations(inputmap)

        # Get positions of feeds
        pos = tools.get_feed_positions(inputmap)

        # Make sure that none of our typical products reference non-array feeds
        stack_new, stack_flag = tools.redefine_stack_index_map(
            inputmap, sstream.prod, sstream.stack, sstream.reverse_map["stack"]
        )

        valid_stack = np.flatnonzero(stack_flag)

        ninvalid = stack_new.size - valid_stack.size
        if ninvalid > 0:
            stack_new = stack_new[valid_stack]
            self.log.info(
                "Could not find appropriate reference inputs for "
                f"{ninvalid:0.0f} stacked products.  Ignoring these "
                "products in solar beamform."
            )

        # Extract the typical products for each stack.
        # Make sure to swap inputs if the data was conjugated.
        prodstack = _swap_inputs(
            sstream.prod[stack_new["prod"]], stack_new["conjugate"]
        )

        # Figure out what data we will need to conjugate in order to convert YX to XY
        conj_pol = pol[prodstack["input_a"]] > pol[prodstack["input_b"]]

        prodstack = _swap_inputs(prodstack, conj_pol)

        nconj = np.sum(conj_pol)
        if nconj > 0:
            self.log.debug(f"Conjugating {nconj} products (of {conj_pol.size}).")

        # Calculate baseline distance and polarisation pair
        index_a = prodstack["input_a"]
        index_b = prodstack["input_b"]

        bdist = pos[index_a] - pos[index_b]
        bpol = np.core.defchararray.add(pol[index_a], pol[index_b])

        # Group the polarisation pairs
        upol, pol_map = np.unique(bpol, return_inverse=True)
        npol = upol.size

        # Exclude autocorrelations
        flag = index_a != index_b

        # Initialise new container
        sscut = sstream.copy()
        sscut.redistribute("freq")

        # Dereference datasets
        vis_local = sstream.vis[:].local_array
        weight_local = sstream.weight[:].local_array

        vis_out = sscut.vis[:].local_array

        nfreq = vis_local.shape[0]

        # Iterate over frequencies and polarisations to null out the sun
        for fi in range(nfreq):
            # Get the baselines in wavelengths
            u = bdist[:, 0] / wv[fi]
            v = bdist[:, 1] / wv[fi]

            # Loop over time to reduce memory usage
            for ti in valid_time:
                # Extract the valid visibilities and weights for this freq and time.
                # Multiply weights by flag so autocorrelations are not used in fit.
                vis = vis_local[fi, valid_stack, ti]
                weight = flag * weight_local[fi, valid_stack, ti]

                # We will transform YX to XY and solve for single solar amplitude.
                vis = np.where(conj_pol, vis.conj(), vis)

                # Calculate the phase that the sun would have using the fringestop routine
                sun_phase = interferometry.fringestop_phase(ha[ti], lat, dec[ti], u, v)

                # Fringestop to the sun
                vs = weight * vis * sun_phase

                # Loop over polarisation pairs
                for pp in range(npol):
                    ipol = np.flatnonzero(pol_map == pp)

                    # Calculate weighted average of the fringestopped visibilities
                    vds = np.sum(vs[ipol])
                    sds = np.sum(weight[ipol])

                    amp = vds * tools.invert_no_zero(sds)

                    # Construct model for sun
                    model = amp * sun_phase[ipol].conj()

                    # Subtract model, conjugating if necessary
                    vis_out[fi, valid_stack[ipol], ti] -= np.where(
                        conj_pol[ipol], model.conj(), model
                    )

        # Return the clean sidereal stream
        return sscut


def _correct_phase_wrap(phi, deg=False):
    if deg:
        return ((phi + 180.0) % 360.0) - 180.0

    return ((phi + np.pi) % (2.0 * np.pi)) - np.pi


def _upper_triangle_gain_vector(gain):
    nfeed = gain.shape[0]
    nprod = nfeed * (nfeed + 1) / 2

    G = np.zeros(nprod, dtype=gain.dtype)

    count = 0
    for fi in range(nfeed):
        for fj in range(fi, nfeed):
            G[count] = np.sum(gain[fi, :] * gain[fj, :].conj())
            count += 1

    return G


def _swap_inputs(prod, conj):
    tmp = prod.copy()
    tmp["input_a"] = np.where(conj, prod["input_b"], prod["input_a"])
    tmp["input_b"] = np.where(conj, prod["input_a"], prod["input_b"])
    return tmp


class SolarCleanVisGrid(tasklib.base.ContainerTask):
    """Clean the Sun from gridded visibilities with a fit smooth in frequency and time.

    For each baseline (east-west and north-south separation) and polarisation of a
    :class:`draco.core.containers.VisGridStream`, the samples taken while the Sun is
    up are fringestopped to the Sun and fitted with

        V(nu, t) exp(-2 pi i b . s_sun(t) / lambda) = sum_kl c_kl S_k(nu) B_l(t),

    and the fit, times the Sun's fringe, is subtracted.  ``S_k`` are discrete prolate
    spheroidal sequences on the frequency channels with delays up to ``delay_max``
    (the eigenvectors of ``sinc(2 delay_max (nu_i - nu_j))`` with eigenvalues above
    ``eig_threshold`` times the largest), and ``B_l`` are cubic B-splines in time
    with knots every ``knot_spacing`` minutes, separately for each period with the
    Sun above ``alt_min``.

    The fit absorbs the Sun's response on each baseline whatever its origin: the
    primary beam of the feeds of that baseline (which differs between baselines by
    ~10% near transit and more in the sidelobes), the resolved disk and its
    projection, and slow changes in the solar flux.  It is a filter in delay and
    fringe rate centred on those of the Sun, which follows the Sun across the sky.
    A transient within the primary beam is within ``delay_max`` of the Sun's delay
    on all but the longest baselines, so only its shortness protects it: a pulse of
    duration T loses roughly T / ``knot_spacing`` of its amplitude while the Sun is
    up (e.g. 17% for 30 s with 3 minute knots).  Finer knots follow the Sun more
    closely and cost more.

    Each sample touches four splines, so the normal equations are block-banded in
    time.  They are built sample by sample (the frequency Gram matrix of each
    sample, weighted by the visibility weights) and solved by a banded Cholesky
    decomposition, with ``ridge`` times the mean of the diagonal added to it.

    Attributes
    ----------
    delay_max : float
        Largest delay of the fit in ns.  Default is 120.
    knot_spacing : float
        Spacing of the spline knots in minutes.  Default is 3.
    alt_min : float
        Fit and subtract while the Sun is above this altitude, in degrees.  Default
        is 0.
    freq_chunk : float
        If set, fit separate pieces of the band of this width (MHz), which is
        cheaper for wide bands.  Default is None (the whole band).
    eig_threshold : float
        Keep the frequency basis functions with eigenvalues above this fraction of
        the largest.  Default is 1e-10 (1e-6 leaves ~1e-4 of a bright Sun).
    ridge : float
        Regularisation, as a fraction of the mean of the diagonal of the normal
        equations.  Default is 1e-8.
    in_place : bool
        Subtract from the input instead of a copy.  Default is False.
    """

    delay_max = config.Property(proptype=float, default=120.0)
    knot_spacing = config.Property(proptype=float, default=3.0)
    alt_min = config.Property(proptype=float, default=0.0)
    freq_chunk = config.Property(proptype=float, default=None)
    eig_threshold = config.Property(proptype=float, default=1e-10)
    ridge = config.Property(proptype=float, default=1e-8)
    in_place = config.Property(proptype=bool, default=False)

    def process(self, gstream):
        """Clean the Sun.

        Parameters
        ----------
        gstream : draco.core.containers.VisGridStream
            Gridded visibilities, with an ``lsd`` (or ``csd``) attribute.

        Returns
        -------
        out : draco.core.containers.VisGridStream
            The visibilities with the Sun subtracted while it is up.
        """
        out = gstream if self.in_place else gstream.copy()
        out.redistribute("ns")

        lsd = out.attrs.get("lsd", out.attrs.get("csd"))
        if lsd is None:
            raise ValueError("The stream has no lsd attribute.")
        lsd = float(np.atleast_1d(lsd)[0])
        ra = out.ra[:]
        times = chime.lsd_to_unix(lsd + ra / 360.0)

        # The Sun in telescope coordinates
        sun = sun_coord(times, deg=False)
        ha, dec, alt = sun[:, 0], sun[:, 1], sun[:, 2]
        lat = np.radians(chime.latitude)
        rot = np.radians(getattr(chime, "rotation", 0.0))
        xs = -np.cos(dec) * np.sin(ha)
        ys = np.cos(lat) * np.sin(dec) - np.sin(lat) * np.cos(dec) * np.cos(ha)
        ew_sun = np.cos(rot) * xs + np.sin(rot) * ys
        ns_sun = -np.sin(rot) * xs + np.cos(rot) * ys

        segments = _true_runs(alt > np.radians(self.alt_min))
        if not segments:
            self.log.info("The Sun is not up.")
            return out

        freq = out.freq[:]
        chunks = self._freq_chunks(freq)
        sbases = [
            _dpss_basis(freq[c], self.delay_max * 1e-3, self.eig_threshold)
            for c in chunks
        ]
        self.log.info(
            f"Fitting {len(segments)} periods with the Sun up, "
            f"{sum(s.stop - s.start for s in segments)} samples, with "
            f"{sum(b.shape[1] for b in sbases)} frequency functions in "
            f"{len(chunks)} pieces of the band."
        )

        tbases = [
            _spline_basis(times[seg], self.knot_spacing * 60.0) for seg in segments
        ]

        ew = out.index_map["ew"][:]
        ns_local = out.vis[:].local_bounds
        ns = out.index_map["ns"][:][ns_local]
        vis = out.vis[:].local_array  # pol, freq, ew, ns, ra
        weight = out.weight[:].local_array

        inv_lambda = freq * 1e6 / scipy.constants.c
        npol, _, new, nns, _ = vis.shape
        removed = np.zeros(2)
        for seg, (ib, bv, nbasis) in zip(segments, tbases):
            for ie in range(new):
                for jn in range(nns):
                    # Fringe of the Sun [freq, time]
                    delay = ew[ie] * ew_sun[seg] + ns[jn] * ns_sun[seg]
                    fringe = np.exp(2.0j * np.pi * inv_lambda[:, np.newaxis] * delay)
                    for pp in range(npol):
                        w = weight[pp, :, ie, jn, seg]
                        if not np.any(w > 0):
                            continue
                        d = vis[pp, :, ie, jn, seg] * fringe.conj()
                        model = np.zeros(d.shape, dtype=np.complex128)
                        for chunk, sb in zip(chunks, sbases):
                            model[chunk] = _fit_smooth(
                                d[chunk], w[chunk], sb, ib, bv, nbasis, self.ridge
                            )
                        removed += [
                            np.sum(w * np.abs(model) ** 2),
                            np.sum(w * np.abs(d) ** 2),
                        ]
                        vis[pp, :, ie, jn, seg] -= (model * fringe).astype(vis.dtype)

        total = np.zeros(2)
        out.comm.Allreduce(removed, total)
        self.log.info(
            f"The Sun model holds {total[0] / max(total[1], 1e-30):.3f} of the weighted "
            "power while the Sun is up."
        )
        out.attrs["solar_clean"] = (
            f"delay_max {self.delay_max} ns, knot_spacing {self.knot_spacing} min, "
            f"alt_min {self.alt_min} deg"
        )
        return out

    def _freq_chunks(self, freq):
        """Index arrays of the pieces of the band that are fitted separately."""
        if not self.freq_chunk:
            return [np.arange(freq.size)]
        edges = np.arange(freq.min(), freq.max() + self.freq_chunk, self.freq_chunk)
        piece = np.clip(np.digitize(freq, edges) - 1, 0, edges.size - 1)
        return [np.flatnonzero(piece == p) for p in np.unique(piece)]


def _true_runs(mask):
    """Slices of the runs of True values of a 1D boolean array."""
    padded = np.concatenate([[False], mask, [False]])
    change = np.flatnonzero(np.diff(padded.astype(int)))
    return [slice(a, b) for a, b in zip(change[::2], change[1::2])]


def _dpss_basis(freq, delay_max, threshold):
    """Discrete prolate spheroidal basis on frequencies (MHz) for delays up to delay_max (us).

    Returns
    -------
    basis : np.ndarray[nfreq, nfunc]
        Orthonormal columns, ordered by decreasing concentration.
    """
    dnu = freq[:, np.newaxis] - freq[np.newaxis, :]
    kernel = np.sinc(2.0 * delay_max * dnu)
    lam, vec = np.linalg.eigh(kernel)
    keep = lam > threshold * lam.max()
    return vec[:, keep][:, ::-1]


def _spline_basis(t, spacing):
    """Cubic B-splines in time with knots every ``spacing`` seconds.

    Returns
    -------
    index : np.ndarray[ntime, 4]
        Index of the four splines that are non-zero at each sample.
    value : np.ndarray[ntime, 4]
        Their values.
    nbasis : int
        Number of splines.
    """
    from scipy.interpolate import BSpline

    span = max(t[-1] - t[0], 1e-6)
    nint = max(int(np.ceil(span / spacing)), 1)
    inner = np.linspace(t[0], t[-1], nint + 1)
    knots = np.concatenate([[inner[0]] * 3, inner, [inner[-1]] * 3])
    nbasis = knots.size - 4
    dm = BSpline.design_matrix(np.clip(t, inner[0], inner[-1]), knots, 3).tocsr()
    index = np.zeros((t.size, 4), dtype=np.int64)
    value = np.zeros((t.size, 4))
    first = np.clip(np.searchsorted(inner, t, side="right") - 1, 0, nint - 1)
    for ii in range(4):
        index[:, ii] = first + ii
        value[:, ii] = np.asarray(dm[np.arange(t.size), first + ii]).ravel()
    return index, value, nbasis


def _fit_smooth(d, w, sbasis, tindex, tvalue, nbasis, ridge):
    """Weighted least-squares fit of ``d[freq, time]`` by ``S(freq) x B(time)``.

    The normal equations are block-banded in time and solved by a banded Cholesky
    decomposition.  Returns the model on the same grid.
    """
    from scipy.linalg import solveh_banded

    nk = sbasis.shape[1]
    nt = d.shape[1]
    w = np.where(np.isfinite(d), w, 0.0)
    d = np.where(w > 0, d, 0.0)

    # Gram matrix and projection of each sample on the frequency basis
    gram = np.einsum("fk,fl,ft->tkl", sbasis, sbasis, w, optimize=True)  # t, k, k
    proj = np.einsum("fk,ft->tk", sbasis, w * d, optimize=True)  # t, k

    # Upper blocks (l, l + o) for o = 0..3 (the matrix is symmetric), and the
    # right-hand side
    blocks = np.zeros((nbasis, 4, nk, nk))
    rhs = np.zeros((nbasis, nk), dtype=np.complex128)
    for ii in range(4):
        np.add.at(rhs, tindex[:, ii], tvalue[:, ii, np.newaxis] * proj)
        for jj in range(ii, 4):
            np.add.at(
                blocks,
                (tindex[:, ii], jj - ii),
                (tvalue[:, ii] * tvalue[:, jj])[:, np.newaxis, np.newaxis] * gram,
            )
    # Upper banded storage: ab[u + i - j, j] = A[i, j] for i <= j
    npar = nbasis * nk
    u = 4 * nk - 1
    ab = np.zeros((u + 1, npar))
    for o in range(4):
        for a in range(nk):
            for b in range(nk):
                i = np.arange(nbasis - o) * nk + a
                j = (np.arange(nbasis - o) + o) * nk + b
                ok = i <= j
                ab[u + i[ok] - j[ok], j[ok]] = blocks[: nbasis - o, o, a, b][ok]
    diag = ab[u]
    ab[u] = diag + ridge * max(np.mean(diag), 1e-30)

    rhs = rhs.reshape(npar)
    sol = solveh_banded(ab, np.stack([rhs.real, rhs.imag], axis=-1), lower=False)
    coeff = (sol[:, 0] + 1j * sol[:, 1]).reshape(nbasis, nk)

    # Model: S @ coeff[l]^T, combined over the four splines of each sample
    h = sbasis @ coeff.T  # freq, nbasis
    model = np.zeros((sbasis.shape[0], nt), dtype=np.complex128)
    for ii in range(4):
        model += h[:, tindex[:, ii]] * tvalue[:, ii]
    return model
