"""
Tasks for analysis of the radio sun

Tasks for analysis of the radio sun.  Includes grouping individual files
into a solar day; solar calibration; and sun excision from sidereal stream.

Tasks
=====

.. autosummary::
    :toctree: generated/

    SolarGrouper
    SolarCalibration
    SolarClean
    SolarCleanProject
    SolarBeamform

Usage
=====


"""

from datetime import datetime
import numpy as np

from caput import config
from caput import mpiutil
from caput import time as ctime
from ch_util import andata, ephemeris, tools, cal_utils
from draco.core import task

from ..core import containers


def unix_to_localtime(unix_time):
    """Converts unix time to a :class:`datetime.datetime` object.

    Parameters
    ----------
    unix_time : float
        Unix/POSIX time.

    Returns
    --------
    dt : :class:`datetime.datetime`

    """
    import pytz

    utc_time = pytz.utc.localize(datetime.utcfromtimestamp(unix_time))

    return utc_time.astimezone(pytz.timezone("Canada/Pacific"))


def _correct_phase_wrap(phi):
    return ((phi + np.pi) % (2.0 * np.pi)) - np.pi


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

    date = ephemeris.ensure_unix(np.atleast_1d(unix_time))
    skyfield_time = ephemeris.unix_to_skyfield_time(date)
    ntime = date.size

    coord = np.zeros((ntime, 4), dtype=np.float32)

    planets = ephemeris.skyfield_wrapper.ephemeris
    # planets = skyfield.api.load('de421.bsp')
    sun = planets["sun"]

    observer = ephemeris._get_chime().skyfield_obs()

    apparent = observer.at(skyfield_time).observe(sun).apparent()
    radec = apparent.radec(epoch=skyfield_time)

    coord[:, 0] = radec[0].radians
    coord[:, 1] = radec[1].radians

    altaz = apparent.altaz()
    coord[:, 2] = altaz[0].radians
    coord[:, 3] = altaz[1].radians

    # Correct RA from equinox to CIRS coords using
    # the equation of the origins
    era = np.radians(ctime.unix_to_era(date))
    gast = 2 * np.pi * skyfield_time.gast / 24.0
    coord[:, 0] = coord[:, 0] + (era - gast)

    # Convert to hour angle
    # defined as local stellar angle minus source right ascension
    coord[:, 0] = _correct_phase_wrap(np.radians(ephemeris.lsa(date)) - coord[:, 0])

    if deg:
        coord = np.degrees(coord)

    return coord


def upper_triangle_gain_vector(gain):

    nfeed = gain.shape[0]
    nprod = nfeed * (nfeed + 1) / 2

    G = np.zeros(nprod, dtype=gain.dtype)

    count = 0
    for fi in range(nfeed):
        for fj in range(fi, nfeed):
            G[count] = np.sum(gain[fi, :] * gain[fj, :].conj())
            count += 1

    return G


class SolarGrouper(task.SingleTask):
    """Group individual timestreams together into whole solar days.

    Attributes
    ----------
    min_span : float
        The minimum solar day length (in hours) to process.
        Default is 2.
    """

    min_span = config.Property(proptype=float, default=2.0)

    def __init__(self):
        super(SolarGrouper, self).__init__()
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

        self.log.debug("Adding file into group for date: %i", day_start)

        # If this file ends during a later day then we need to process the
        # current list and restart the system
        if self._current_day < day_end:

            self.log.debug("Concatenating files for date: %i", day_start)

            # Combine timestreams into a single container for the whole day this
            # could get returned as None if there wasn't enough data
            tstream_all = self._process_current_day()

            # Reset list and current day for the new file
            self._timestream_list = [tstream]
            self._current_day = day_end

            return tstream_all
        else:
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
        tstream_all = self._process_current_day()

        return tstream_all

    def _process_current_day(self):

        # Combine the current set of files into a timestream

        day = str(self._current_day)

        # Calculate the length of data in the current day
        start = datetime.utcfromtimestamp(self._timestream_list[0].time[0])
        end = datetime.utcfromtimestamp(self._timestream_list[-1].time[-1])
        tdelta = end - start
        day_length = tdelta.days * 24.0 + tdelta.seconds / 3600.0

        # If the amount of data for this day is too small, then just skip
        if day_length < self.min_span:
            return None

        self.log.debug("Constructing %s [%i files]", day, len(self._timestream_list))

        # Construct the combined timestream
        ts = andata.concatenate(self._timestream_list)

        # Add attributes for the date and a tag for labelling saved files
        ts.attrs["tag"] = day
        ts.attrs["date"] = self._current_day

        return ts


class SolarCalibration(task.SingleTask):
    """Use Sun to measure antenna beam pattern.

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
    model_fit: bool, default False
        Fit a model to the primary beam.
    nsig: float, default 2.0
        Relevant if model_fit is True.  The model is only fit to
        time samples within +/- nsig sigma from the expected
        peak location.
    """

    dualpol = config.Property(proptype=bool, default=True)
    extended = config.Property(proptype=bool, default=True)
    hermweight = config.Property(proptype=bool, default=True)
    neigen = config.Property(proptype=int, default=2)
    ymin = config.Property(proptype=float, default=1.2)
    max_iter = config.Property(proptype=int, default=4)

    model_fit = config.Property(proptype=bool, default=False)
    nsig = config.Property(proptype=float, default=2.0)

    def process(self, sstream, inputmap, inputmask):
        """Determine solar response from input timestream.

        Parameters
        ----------
        sstream : andata.CorrData or containers.SiderealStream
            Timestream collected during the day.
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

        from operator import itemgetter
        from itertools import groupby
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
        freq = sstream.index_map["freq"]["centre"][sfreq:efreq]
        wv = 3e2 / freq

        # Get times (ra in degrees)
        if hasattr(sstream, "time"):
            time = sstream.time
            ra = ephemeris.lsa(time)
        else:
            ra = sstream.index_map["ra"][:]
            csd = (
                sstream.attrs["lsd"] if "lsd" in sstream.attrs else sstream.attrs["csd"]
            )
            csd = csd + ra / 360.0
            time = ephemeris.csd_to_unix(csd)

        # Only examine data between sunrise and sunset
        time_flag = np.zeros(len(time), dtype=np.bool)
        rise = ephemeris.solar_rising(time[0] - 24.0 * 3600.0, end_time=time[-1])
        for rr in rise:
            ss = ephemeris.solar_setting(rr)[0]
            time_flag |= (time >= rr) & (time <= ss)

        if not np.any(time_flag):
            self.log.debug(
                "No daytime data between %s and %s.",
                ephemeris.unix_to_datetime(time[0]).strftime("%b %d %H:%M"),
                ephemeris.unix_to_datetime(time[-1]).strftime("%b %d %H:%M"),
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

        if mpiutil.rank0:
            print(
                "Performing sun calibration with %d/%d good feeds (%d xpol, %d ypol)."
                % (len(good_input), nfeed, len(xfeeds), len(yfeeds))
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

            p_nfeed, p_nprod = ifeed.size, iprod.size
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

            icross = np.ones(iprod.size, dtype=np.bool)
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
                for ss, src in ephemeris.source_dictionary.iteritems():
                    if isinstance(src, skyfield.starlib.Star):
                        peak_ra = ephemeris.peak_RA(src, deg=True)
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

                            sha = wrap_phase(ra[tt_out] - center, deg=True)

                            if np.abs(sha) < span:

                                src_phase = tools.fringestop_phase(
                                    np.radians(sha),
                                    np.radians(ephemeris.CHIMELATITUDE),
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
                        vis *= tools.fringestop_phase(
                            sun_pos[tt_out, 0],
                            np.radians(ephemeris.CHIMELATITUDE),
                            sun_pos[tt_out, 1],
                            u,
                            v,
                        )

                        # Solve for the solar response
                        ev, resp, err_resp = [
                            np.squeeze(var)
                            for var in solve_gain(
                                vis[edim], norm=norm[edim], neigen=self.neigen
                            )
                        ]

                        if len(resp.shape) == 1:
                            resp = resp[:, np.newaxis]
                            err_resp = err_resp[:, np.newaxis]

                        G = upper_triangle_gain_vector(resp)

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
                                ev, resp, err_resp = [
                                    np.squeeze(var)
                                    for var in solve_gain(
                                        vism[edim], norm=norm[edim], neigen=self.neigen
                                    )
                                ]

                                if len(resp.shape) == 1:
                                    resp = resp[:, np.newaxis]
                                    err_resp = err_resp[:, np.newaxis]

                                G = upper_triangle_gain_vector(resp)

                                A = G[:, np.newaxis] * H

                                # Increase iteration counter
                                iters += 1

                            # Save model coefficients to output container
                            suntrans.coeff[
                                ff_global, ipol, tt_out, :, :
                            ] = coeff.reshape(*poly_ncoeff)

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

        # If requested, fit a model to the primary beam of the sun transit
        if self.model_fit:

            # Estimate peak RA
            i_transit = np.argmin(np.abs(sun_pos[:, 0]))

            body = ephemeris.skyfield_wrapper.ephemeris["sun"]
            obs = ephemeris._get_chime()
            obs.date = ephemeris.unix_to_ephem_time(time[i_transit])
            body.compute(obs)
            peak_ra = ephemeris.peak_RA(body, date=time[i_transit], deg=True)
            dra = ra - peak_ra
            dra = np.abs(wrap_phase(dra, deg=True))

            # Estimate FWHM
            sig_x = cal_utils.guess_fwhm(freq, pol="X", dec=body.dec, sigma=True)[
                :, np.newaxis, np.newaxis
            ]
            sig_y = cal_utils.guess_fwhm(freq, pol="Y", dec=body.dec, sigma=True)[
                :, np.newaxis, np.newaxis
            ]

            # Only fit ra values above the specified dynamic range threshold
            fit_flag = np.zeros((nfreq, nfeed, ntime), dtype=np.bool)
            fit_flag[:, xfeeds, :] = dra < (self.nsig * sig_x)
            fit_flag[:, yfeeds, :] = dra < (self.nsig * sig_y)

            # Fit model for the complex response of each feed to the point source
            param, param_cov = cal_utils.fit_point_source_transit(
                ra,
                suntrans.response[..., 0].view(np.ndarray),
                suntrans.response_error[..., 0].view(np.ndarray),
                flag=fit_flag,
            )

            # Save to container
            suntrans.add_dataset("flag")
            suntrans.flag[:] = fit_flag

            suntrans.add_dataset("parameter")
            suntrans.parameter[:] = param

            suntrans.add_dataset("parameter_cov")
            suntrans.parameter_cov[:] = param_cov

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


class SolarClean(task.SingleTask):
    """Clean sun from daytime data by subtracting a model for the
       sun visibility determined by the SolarCalibration module.

    Attributes
    ----------
    threshold : float, default 2.5
        Do not subtract sun if the is_sun metric defined in
        SolarCalibration module is less than threshold.
    savesun : bool, default False
        Save solar model to be subtracted
    output_dir : str, default None
        Directory path to save output file
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

        freq = sstream.freq["centre"][sfreq:efreq]
        wv = 3e2 / freq

        # Determine time mapping
        if hasattr(sstream, "time"):
            stime = sstream.time[:]
        else:
            ra = sstream.index_map["ra"][:]
            csd = (
                sstream.attrs["lsd"] if "lsd" in sstream.attrs else sstream.attrs["csd"]
            )
            stime = ephemeris.csd_to_unix(csd + ra / 360.0)

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
        pol_ind = np.full(len(feed_list), -1, dtype=np.int)
        for bb, (fi, fj) in enumerate(feed_list):
            if tools.is_chime(fi) and tools.is_chime(fj):
                pol_ind[bb] = 2 * tools.is_chime_y(fi) + tools.is_chime_y(fj)

        npol = suntrans.index_map["pol"].size
        if npol == 1:
            pol_ind = (pol_ind < 0).astype(np.int)
            pol_sub = np.array([0])
        elif npol == 2:
            pol_sub = np.array([0, 3])
        else:
            ValueError("npol = %d, must be either 1 or 2." % npol)

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
                        sunphase = tools.fringestop_phase(
                            ha, np.radians(ephemeris.CHIMELATITUDE), dec, u, v
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


class SolarCleanProject(task.SingleTask):
    """Clean the sun from data by projecting out signal from its location.
    Formerly called SunClean."""

    def process(self, sstream, inputmap):
        """Clean the sun.

        Parameters
        ----------
        sstream: andata.CorrData or containers.SiderealStream
            Timestream collected during the day.
        inputmap : list of :class:`CorrInput`s
            A list describing the inputs as they are in the file.

        Returns
        -------
        mstream : containers.SiderealStream
            Sidereal stack with sun projected out.
        """

        sstream.redistribute("freq")

        # Get array of CSDs for each sample (ra in degrees)
        if hasattr(sstream, "time"):
            times = sstream.time
            ra = ephemeris.lsa(time)
        else:
            ra = sstream.index_map["ra"][:]
            csd = (
                sstream.attrs["lsd"] if "lsd" in sstream.attrs else sstream.attrs["csd"]
            )
            csd = csd + ra / 360.0
            times = ephemeris.csd_to_unix(csd)

        nprod = len(sstream.index_map["prod"])

        # Get position of sun at every time sample (in radians)
        sun_pos = sun_coord(times, deg=False)

        # Get hour angle and dec of sun, in radians
        ha = sun_pos[:, 0]
        dec = sun_pos[:, 1]
        el = sun_pos[:, 2]

        # Construct baseline vector for each visibility
        feed_pos = tools.get_feed_positions(inputmap)
        vis_pos = np.array(
            [feed_pos[ii] - feed_pos[ij] for ii, ij in sstream.index_map["prod"][:]]
        )

        feed_list = [
            (inputmap[fi], inputmap[fj]) for fi, fj in sstream.index_map["prod"][:]
        ]

        # Determine polarisation for each visibility
        pol_ind = np.full(len(feed_list), -1, dtype=np.int)
        for ii, (fi, fj) in enumerate(feed_list):
            if tools.is_chime(fi) and tools.is_chime(fj):
                pol_ind[ii] = 2 * tools.is_chime_y(fi) + tools.is_chime_y(fj)

        # Change vis_pos for non-CHIME feeds from NaN to 0.0
        vis_pos[(pol_ind == -1), :] = 0.0

        # Initialise new container
        sscut = sstream.__class__(axes_from=sstream, attrs_from=sstream)
        sscut.redistribute("freq")

        wv = 3e2 / sstream.index_map["freq"]["centre"]

        # Iterate over frequencies and polarisations to null out the sun
        for lfi, fi in sstream.vis[:].enumerate(0):

            # Get the baselines in wavelengths
            u = vis_pos[:, 0] / wv[fi]
            v = vis_pos[:, 1] / wv[fi]

            # Loop over ra to reduce memory usage
            for ri in range(len(ra)):

                # Copy over the visiblities and weights
                vis = sstream.vis[fi, :, ri]
                weight = sstream.weight[fi, :, ri]
                sscut.vis[fi, :, ri] = vis
                sscut.weight[fi, :, ri] = weight

                # Check if sun has set
                if el[ri] > 0.0:

                    # Calculate the phase that the sun would have using the fringestop routine
                    sun_vis = tools.fringestop_phase(
                        ha[ri], np.radians(ephemeris.CHIMELATITUDE), dec[ri], u, v
                    )

                    # Mask out the auto-correlations
                    sun_vis *= np.logical_or(u != 0.0, v != 0.0)

                    # Iterate over polarisations to do projection independently for each.
                    # This is needed because of the different beams for each pol.
                    for pol in range(4):

                        # Mask out other polarisations in the visibility vector
                        sun_vis_pol = sun_vis * (pol_ind == pol)

                        # Calculate various projections
                        vds = (vis * sun_vis_pol * weight).sum(axis=0)
                        sds = weight.sum(axis=0)
                        isds = tools.invert_no_zero(sds)

                        # Subtract sun contribution from visibilities and place in new array
                        sscut.vis[fi, :, ri] -= sun_vis_pol.conj() * vds * isds

        # Return the clean sidereal stream
        return sscut


class SolarBeamform(task.SingleTask):
    """Beamform to the location of the Sun.
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
    single_cyl : bool, default False
        Include data from just a single cylinder. Default
        is False
    cyl_id :  int, default 0
        Cylinder number (0-3) for single cylinder
        measurements. Only relevant if exclude_intercyl
        and single_cyl are both True. Default is 0.
    """

    ymax = config.Property(proptype=float, default=10.0)
    exclude_intercyl = config.Property(proptype=bool, default=True)
    single_cyl = config.Property(proptype=bool, default=False)
    cyl_id = config.Property(proptype=int, default=0)

    def process(self, sstream, inputmap):
        """Beamform visibilities to the location of the Sun

        Parameters
        ----------
        sstream: andata.CorrData or containers.SiderealStream
            Timestream collected during the day.
        inputmap : list of :class:`CorrInput`s
            A list describing the inputs as they are in the file.

        Returns
        -------
        sunstream : containers.SiderealStream
            Sun's contribution to sidereal stack
        """

        sstream.redistribute("freq")

        # Get array of CSDs for each sample (ra in degrees)
        if hasattr(sstream, "time"):
            time = sstream.time
            ra = ephemeris.lsa(time)
        else:
            ra = sstream.index_map["ra"][:]
            csd = (
                sstream.attrs["lsd"] if "lsd" in sstream.attrs else sstream.attrs["csd"]
            )
            csd = csd + ra / 360.0
            time = ephemeris.csd_to_unix(csd)

        nprod = len(sstream.index_map["prod"][sstream.index_map["stack"]["prod"]])

        # Get position of sun at every time sample (in radians)
        sun_pos = sun_coord(time, deg=False)

        # Get hour angle and dec of sun, in radians
        ha = sun_pos[:, 0]
        dec = sun_pos[:, 1]
        el = sun_pos[:, 2]

        # Construct baseline vector for each visibility
        feed_pos = tools.get_feed_positions(inputmap)
        vis_pos = np.array(
            [
                feed_pos[fi] - feed_pos[fj]
                for fi, fj in sstream.index_map["prod"][
                    sstream.index_map["stack"]["prod"]
                ][:]
            ]
        )

        feed_list = [
            (inputmap[fi], inputmap[fj])
            for fi, fj in sstream.index_map["prod"][sstream.index_map["stack"]["prod"]][
                :
            ]
        ]
        
        # Determine polarisation for each visibility
        pol_ind = np.full(nprod, -1, dtype=np.int)
        cyl_i = np.full(nprod, -1, dtype=np.int)
        cyl_j = np.full(nprod, -1, dtype=np.int)

        for ii, (fi, fj) in enumerate(feed_list):

            if tools.is_chime(fi) and tools.is_chime(fj):

                pol_ind[ii] = 2 * tools.is_array_y(fi) + tools.is_array_y(fj)

                if fi.reflector == "cylinder_A":
                    cyl_i[ii] = 0
                elif fi.reflector == "cylinder_B":
                    cyl_i[ii] = 1
                elif fi.reflector == "cylinder_C":
                    cyl_i[ii] = 2
                elif fi.reflector == "cylinder_D":
                    cyl_i[ii] = 3

                if fj.reflector == "cylinder_A":
                    cyl_j[ii] = 0
                elif fj.reflector == "cylinder_B":
                    cyl_j[ii] = 1
                elif fj.reflector == "cylinder_C":
                    cyl_j[ii] = 2
                elif fj.reflector == "cylinder_D":
                    cyl_j[ii] = 3

        # Change vis_pos for non-CHIME feeds from NaN to 0.0
        vis_pos[(pol_ind == -1), :] = 0.0

        newprod = [[0, 0], [0, 1], [1, 0], [1, 1]]

        newprod = np.array(newprod, dtype=sstream.index_map["prod"].dtype)
        newstack = np.zeros(len(newprod), dtype=[("prod", "<u4"), ("conjugate", "u1")])
        newstack["prod"][:] = np.arange(len(newprod))
        newstack["conjugate"] = 0

        if isinstance(sstream, containers.SiderealStream):
            OutputContainer = containers.SiderealStream
        else:
            OutputContainer = containers.TimeStream

        sunstream = OutputContainer(
            prod=newprod, stack=newstack, axes_from=sstream, attrs_from=sstream
        )

        sunstream.redistribute("freq")
        sunstream.vis[:] = 0.0
        sunstream.weight[:] = 0.0

        wv = 3e2 / sstream.index_map["freq"]["centre"]

        # Iterate over frequencies and polarisations to null out the sun
        for lfi, fi in sstream.vis[:].enumerate(0):

            # Get the baselines in wavelengths
            u = vis_pos[:, 0] / wv[fi]
            v = vis_pos[:, 1] / wv[fi]

            # Loop over ra to reduce memory usage
            for ri in range(len(ra)):

                # Initialize the visiblities matrix
                vis = sstream.vis[fi, :, ri]
                weight = sstream.weight[fi, :, ri]

                # Check if sun has set
                if el[ri] > 0.0:

                    # Calculate the phase that the sun would have using the fringestop routine
                    sun_vis = tools.fringestop_phase(
                        ha[ri], np.radians(ephemeris.CHIMELATITUDE), dec[ri], u, v
                    )

                    # Mask out the auto-correlations
                    sun_vis *= np.logical_or(u != 0.0, v != 0.0)

                    # Mask out long NS baselines
                    sun_vis *= np.abs(vis_pos[:, 1]) <= self.ymax

                    if self.exclude_intercyl:
                        # Mask out inter-cylinder visibilities
                        sun_vis *= cyl_i == cyl_j

                        if self.single_cyl:
                            sun_vis *= cyl_i == self.cyl_id

                    # Iterate over polarizations
                    for pi in range(4):

                        # Mask out other polarisations in the visibility vector
                        sun_vis_pol = sun_vis * (pol_ind == pi)

                        # Beamform to Sun
                        vds = (vis * sun_vis_pol * weight).sum(axis=0)
                        sds = (sun_vis_pol.conj() * sun_vis_pol * weight).sum(axis=0)
                        isds = tools.invert_no_zero(sds)

                        sunstream.vis[fi, pi, ri] = vds * isds
                        sunstream.weight[fi, pi, ri] = sds

        # Return the clean sidereal stream
        return sunstream
