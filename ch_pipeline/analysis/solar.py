"""
Tasks for analysis of the radio sun

Tasks for analysis of the radio sun.  Includes grouping individual files
into a solar day; solar calibration; and sun excision from sidereal stream.
"""

from datetime import datetime
import numpy as np

from caput import config
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


def ra_dec_of(body, time):
    """Calculate the coordinates of a celestial
    body at a particular time.

    Parameters
    ----------
    body : ephem.Body
        PyEphem celestial body.

    time : float
        Unix/POSIX time.

    Returns
    --------
    coord : (ra, dec, alt, az)
        Coordinates of the sources

    """
    obs = ephemeris._get_chime()
    obs.date = ephemeris.unix_to_ephem_time(time)

    body.compute(obs)

    return body.ra, body.dec, body.alt, body.az


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
        ts.attrs["date"] = day

        return ts


class SolarCalibration(task.SingleTask):
    """Use Sun to measure antenna beam pattern.

    Attributes
    ----------
    fringestop:  bool, default False
        Fringestop prior to solving for the sun response.
    model_fit: bool, default False
        Fit a model to the primary beam.
    nsig: float, default 2.0
        Relevant if model_fit is True.  The model is only fit to
        time samples within +/- nsig sigma from the expected
        peak location.
    """

    fringestop = config.Property(proptype=bool, default=False)
    model_fit = config.Property(proptype=bool, default=False)
    nsig = config.Property(proptype=float, default=2.0)

    def process(self, sstream, inputmap, inputmask):
        """Determine calibration from a timestream.

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

        # Ensure that we are distributed over frequency
        sstream.redistribute("freq")

        # Find the local frequencies
        nfreq = sstream.vis.local_shape[0]
        sfreq = sstream.vis.local_offset[0]
        efreq = sfreq + nfreq

        # Get the local frequency axis
        freq = sstream.freq["centre"][sfreq:efreq]
        wv = 3e2 / freq

        # Get times
        if hasattr(sstream, "time"):
            time = sstream.time
            ra = ephemeris.transit_RA(time)
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
        time_index = np.where(time_flag)[0]

        time_slice = []
        ntime = 0
        for key, group in groupby(
            enumerate(time_index), lambda index_item: index_item[0] - index_item[1]
        ):
            group = list(map(itemgetter(1), group))
            ngroup = len(group)
            time_slice.append(
                (slice(group[0], group[-1] + 1), slice(ntime, ntime + ngroup))
            )
            ntime += ngroup

        time = np.concatenate([time[slc[0]] for slc in time_slice])
        ra = np.concatenate([ra[slc[0]] for slc in time_slice])

        # Get ra, dec, alt of sun
        sun_pos = np.array(
            [ra_dec_of(ephemeris.skyfield_wrapper.ephemeris["sun"], t) for t in time]
        )

        # Convert from ra to hour angle
        sun_pos[:, 0] = np.radians(ra) - sun_pos[:, 0]

        # Determine good inputs
        nfeed = len(inputmap)
        good_input = np.arange(nfeed, dtype=np.int)[inputmask.datasets["input_mask"][:]]

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

        self.log.debug(
            "Performing sun calibration with %d/%d good feeds (%d xpol, %d ypol).",
            len(good_input),
            nfeed,
            len(xfeeds),
            len(yfeeds),
        )

        # Construct baseline vector for each visibility
        feed_pos = tools.get_feed_positions(inputmap)
        vis_pos = np.array(
            [feed_pos[ii] - feed_pos[ij] for ii, ij in sstream.index_map["prod"][:]]
        )
        vis_pos = np.where(np.isnan(vis_pos), np.zeros_like(vis_pos), vis_pos)

        u = (vis_pos[np.newaxis, :, 0] / wv[:, np.newaxis])[:, :, np.newaxis]
        v = (vis_pos[np.newaxis, :, 1] / wv[:, np.newaxis])[:, :, np.newaxis]

        # Create container to hold results of fit
        suntrans = containers.SunTransit(
            time=time, pol_x=xfeeds, pol_y=yfeeds, axes_from=sstream
        )
        for key in suntrans.datasets.keys():
            suntrans.datasets[key][:] = 0.0

        # Set coordinates
        suntrans.coord[:] = sun_pos

        # Loop over time slices
        for slc_in, slc_out in time_slice:

            # Extract visibility slice
            vis_slice = sstream.vis[..., slc_in].copy()

            ha = (sun_pos[slc_out, 0])[np.newaxis, np.newaxis, :]
            dec = (sun_pos[slc_out, 1])[np.newaxis, np.newaxis, :]

            # Extract the diagonal (to be used for weighting)
            norm = (_extract_diagonal(vis_slice, axis=1).real) ** 0.5
            norm = tools.invert_no_zero(norm)

            # Fringestop
            if self.fringestop:
                vis_slice *= tools.fringestop_phase(
                    ha, np.radians(ephemeris.CHIMELATITUDE), dec, u, v
                )

            # Solve for the point source response of each set of polarisations
            ev_x, resp_x, err_resp_x = solve_gain(
                vis_slice, feeds=xfeeds, norm=norm[:, xfeeds]
            )
            ev_y, resp_y, err_resp_y = solve_gain(
                vis_slice, feeds=yfeeds, norm=norm[:, yfeeds]
            )

            # Save to container
            suntrans.evalue_x[..., slc_out] = ev_x
            suntrans.evalue_y[..., slc_out] = ev_y

            suntrans.response[:, xfeeds, slc_out] = resp_x
            suntrans.response[:, yfeeds, slc_out] = resp_y

            suntrans.response_error[:, xfeeds, slc_out] = err_resp_x
            suntrans.response_error[:, yfeeds, slc_out] = err_resp_y

        # If requested, fit a model to the primary beam of the sun transit
        if self.model_fit:

            # Estimate peak RA
            i_transit = np.argmin(np.abs(sun_pos[:, 0]))

            body = ephemeris.skyfield_wrapper.ephemeris["sun"]
            obs = ephemeris._get_chime()
            obs.date = ephemeris.unix_to_ephem_time(time[i_transit])
            body.compute(obs)

            peak_ra = ephemeris.peak_RA(body)
            dra = ra - peak_ra
            dra = np.abs(dra - (dra > np.pi) * 2.0 * np.pi)[np.newaxis, np.newaxis, :]

            # Estimate FWHM
            sig_x = cal_utils.guess_fwhm(freq, pol="X", dec=body.dec, sigma=True)[
                :, np.newaxis, np.newaxis
            ]
            sig_y = cal_utils.guess_fwhm(freq, pol="Y", dec=body.dec, sigma=True)[
                :, np.newaxis, np.newaxis
            ]

            # Only fit ra values above the specified dynamic range threshold
            fit_flag = np.zeros([nfreq, nfeed, ntime], dtype=np.bool)
            fit_flag[:, xfeeds, :] = dra < (self.nsig * sig_x)
            fit_flag[:, yfeeds, :] = dra < (self.nsig * sig_y)

            # Fit model for the complex response of each feed to the point source
            param, param_cov = cal_utils.fit_point_source_transit(
                ra, suntrans.response[:], suntrans.response_error[:], flag=fit_flag
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

        # Return sun transit
        return suntrans


class SolarClean(task.SingleTask):
    """Clean the sun from daytime data by removing the outer product of the
    eigenvector corresponding to the largest eigenvalue.
    """

    def process(self, sstream, suntrans, inputmap):
        """Clean the sun.

        Parameters
        ----------
        sstream : containers.SiderealStream
            Sidereal stream.
        suntrans : containers.SolarTransit
            Response to the sun.
        inputmap : list of :class:`CorrInput`
            A list describing the inputs as they are in the file.

        Returns
        -------
        mstream : containers.SiderealStream
            Sidereal stream with sun removed
        """

        sstream.redistribute("freq")
        suntrans.redistribute("freq")

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

        ninput = gain.shape[1]

        # Determine product map
        prod_map = sstream.index_map["prod"][:]
        nprod = prod_map.size

        if nprod != (ninput * (ninput + 1) // 2):
            raise Exception("Number of inputs does not match the number of products.")

        feed_list = [(inputmap[ii], inputmap[jj]) for ii, jj in prod_map]

        # Determine polarisation for each visibility
        same_pol = np.zeros(nprod, dtype=np.bool)
        for pp, (ii, jj) in enumerate(feed_list):
            if tools.is_chime(ii) and tools.is_chime(jj):
                same_pol[pp] = tools.is_chime_y(ii) == tools.is_chime_y(jj)

        # Match ra
        match = np.array([np.argmin(np.abs(gt - stime)) for gt in gtime])

        # Loop over frequencies and products
        for lfi, fi in sstream.vis[:].enumerate(0):

            for pp in range(nprod):

                if same_pol[pp]:

                    ii, jj = prod_map[pp]

                    # Fetch the gains
                    gi = gain[lfi, ii, :]
                    gj = gain[lfi, jj, :].conj()

                    # Subtract the gains
                    sstream.vis[fi, pp, match] -= gi * gj

        # Return the clean sidereal stream
        return sstream


class SunClean(task.SingleTask):
    """Clean the sun from data by projecting out signal from its location."""

    def process(self, sstream, inputmap):
        """Clean the sun.

        Parameters
        ----------
        sstream : containers.SiderealStream
            Sidereal stream.

        Returns
        -------
        mstream : containers.SiderealStream
            Sidereal stack with sun projected out.
        """

        sstream.redistribute("freq")

        # Get array of CSDs for each sample
        ra = sstream.index_map["ra"][:]
        csd = sstream.attrs["lsd"] if "lsd" in sstream.attrs else sstream.attrs["csd"]
        csd = csd + ra / 360.0

        # Get position of sun at every time sample
        times = ephemeris.csd_to_unix(csd)
        sun_pos = np.array(
            [ra_dec_of(ephemeris.skyfield_wrapper.ephemeris["sun"], t) for t in times]
        )

        # Get hour angle and dec of sun, in radians
        ha = 2 * np.pi * (ra / 360.0) - sun_pos[:, 0]
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

                    # Calculate the visibility vector for the sun
                    sun_vis = sun_vis.conj()

                    # Mask out the auto-correlations
                    sun_vis *= np.logical_or(u != 0.0, v != 0.0)

                    # Iterate over polarisations to do projection independently for each.
                    # This is needed because of the different beams for each pol.
                    for pol in range(4):

                        # Mask out other polarisations in the visibility vector
                        sun_vis_pol = sun_vis * (pol_ind == pol)

                        # Calculate various projections
                        vds = (vis * sun_vis_pol.conj() * weight).sum(axis=0)
                        sds = (sun_vis_pol * sun_vis_pol.conj() * weight).sum(axis=0)
                        isds = tools.invert_no_zero(sds)

                        # Subtract sun contribution from visibilities and place in new array
                        sscut.vis[fi, :, ri] -= sun_vis_pol * vds * isds

        # Return the clean sidereal stream
        return sscut
