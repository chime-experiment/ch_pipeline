"""Tasks for analysis of the radio Jupiter

Includes grouping individual files;
"""

import datetime
import pytz

import numpy as np
import scipy.constants

from caput import config, mpiutil, tod
from draco.core import task
from ch_util import ephemeris, tools, cal_utils

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

    utc_time = pytz.utc.localize(datetime.datetime.utcfromtimestamp(unix_time))

    return utc_time.astimezone(pytz.timezone("Canada/Pacific"))


def jupiter_coord(unix_time, deg=True):
    """Calculate the coordinates of the jupiter at a given time.

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
        jupiter at each time.

    """

    date = ephemeris.ensure_unix(np.atleast_1d(unix_time))
    skyfield_time = ephemeris.unix_to_skyfield_time(date)
    ntime = date.size

    coord = np.zeros((ntime, 4), dtype=np.float64)

    planets = ephemeris.skyfield_wrapper.ephemeris
    # planets = skyfield.api.load('de421.bsp')
    jup = planets["Jupiter Barycenter"]

    observer = ephemeris.chime.skyfield_obs()

    apparent = observer.at(skyfield_time).observe(jup).apparent()

    radec = apparent.cirs_radec(epoch=skyfield_time)
    coord[:, 0] = radec[0].radians
    coord[:, 1] = radec[1].radians

    altaz = apparent.altaz()
    coord[:, 2] = altaz[0].radians
    coord[:, 3] = altaz[1].radians

    # Convert to hour angle
    # defined as local stellar angle minus source right ascension
    coord[:, 0] = _correct_phase_wrap(np.radians(ephemeris.lsa(date)) - coord[:, 0])

    if deg:
        coord = np.degrees(coord)

    return coord


class JupiterGrouper(task.SingleTask):
    """Group individual timestreams together into whole solar days.

    Attributes
    ----------
    min_span : float
        The minimum solar day length (in hours) to process.
        Default is 2.
    """

    min_span = config.Property(proptype=float, default=2.0)

    def __init__(self):
        super(JupiterGrouper, self).__init__()
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







class JupiterBeamform(task.SingleTask):
    """Estimate the average primary beam by beamforming on the Jupiter.

    Formerly called JupiterCalibration.

    Attributes
    ----------
    ymax: float, default 10.0
        Do not include baselines with N-S separation
        greater than ymax to avoid resolving out the Jupiter.
        Default is 10.0 (meters)
    exclude_intercyl : bool, default True
        Exclude intercylinder baselines to avoid resolving
        out the Jupiter. Default is True
    sep_cyl : bool, default False
        Do not average over cylinder pairs when beamforming.
        If False, will yield a single measurement of the Jupiter.
        If True will yield a separate measurement of the Jupiter
        for each cylinder pair.  Default is False.
    """

    #ymax = config.Property(proptype=float, default=10.0)
    exclude_intracyl = config.Property(proptype=bool, default=True)
    sep_cyl = config.Property(proptype=bool, default=False)

    def process(self, sstream, inputmap):
        """Beamform visibilities to the location of the Jupiter.

        Parameters
        ----------
        sstream: andata.CorrData, containers.TimeStream, containers.SiderealStream
            Timestream collected during the day.
        inputmap : list of :class:`CorrInput`
            A list describing the inputs as they are in the file.

        Returns
        -------
        jupitertream : containers.FormedBeamTime
            Formed beam at the location of the jupiter.
        """

        # Determine the time axis
        if hasattr(sstream, "time"):
            time = sstream.time
        else:
            csd = (
                sstream.attrs["lsd"] if "lsd" in sstream.attrs else sstream.attrs["csd"]
            )
            time = ephemeris.csd_to_unix(csd + sstream.ra / 360.0)

        lat = np.radians(ephemeris.CHIMELATITUDE)

        # Get position of jupiter at every time sample (in radians)
        jup_pos = jupiter_coord(time, deg=False)

        ha = jup_pos[:, 0]
        dec = jup_pos[:, 1]
        el = jup_pos[:, 2]

        # Only process times when jupiter is above the horizon
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
                "products in jupiter beamform."
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
        if self.exclude_intracyl:
            flag &= cyl[index_a] != cyl[index_b]

        # Exclude long north-south baselines if requested
        #if self.ymax is not None:
            #flag &= np.abs(bdist[:, 1]) <= self.ymax

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
        object_id["source"] = "jupiter"
        object_id["cylinder"] = ucyl

        # Create output container
        jupiterstream = containers.FormedBeamTime(
            time=time[valid_time],
            object_id=object_id,
            pol=upol,
            axes_from=sstream,
            attrs_from=sstream,
            distributed=sstream.distributed,
            comm=sstream.comm,
        )

        jupiterstream.redistribute("freq")
        jupiterstream.beam[:] = 0.0
        jupiterstream.weight[:] = 0.0

        # Dereference datasets
        vis_local = sstream.vis[:].local_array
        weight_local = sstream.weight[:].local_array

        vis_out = jupiterstream.beam[:].local_array
        weight_out = jupiterstream.weight[:].local_array

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

                # Calculate the phase that the jupiter would have using the fringestop routine
                jup_vis = tools.fringestop_phase(ha[ti], lat, dec[ti], u, v)

                # Fringestop to the jupiter
                vs = weight * vis * jup_vis

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
        return jupiterstream



