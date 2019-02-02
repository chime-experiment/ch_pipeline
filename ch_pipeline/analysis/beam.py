""" Docstring goes here.
"""

import numpy as np

from caput import config, tod, mpiarray
from caput.pipeline import PipelineConfigError

from draco.core import task
from draco.analysis.transform import Regridder
from draco.core.containers import SiderealStream

from ch_util import ephemeris as ephem


class TransitGrouper(task.SingleTask):
    """ Group transits from a sequence of TimeStream objects.

        Attributes
        ----------
        ha_span: float
            Span in degrees surrounding transit.
        source: str
            Name of the transiting source. (Must match what is used in `ch_util.ephemeris`.)
    """

    ha_span = config.Property(proptype=float, default=180.)
    source = config.Property(proptype=str)

    def setup(self, observer=None):
        """Set the local observers position if not using CHIME.

        Parameters
        ----------
        observer : caput.time.Observer, optional
            Details of the observer, if not set default to CHIME.
        """
        self.observer = ephem.chime_observer() if observer is None else observer
        self.sky_obs = wrap_observer(self.observer)
        try:
            self.src = ephem.source_dictionary[self.source]
        except KeyError:
            msg = ("Could not find source {} in catalogue. "
                   "Must use same spelling as in `ch_util.ephemeris`.".format(self.source))
            self.log.error(msg)
            raise PipelineConfigError(msg)
        self.cur_transit = None
        self.tstreams = []
        self.last_time = 0

    def process(self, tstream):
        """ Take in a timestream and accumulate, group into whole transits.

        Parameters
        ----------
        tstream : TimeStream

        Returns
        -------
        ts : TimeStream
            Timestream with containing transits of specified length.
        """

        # shortcut
        obs = self.sky_obs

        # Update observer time
        self.sky_obs.date = tstream.time[0]

        # placeholder for finalized transit when it is ready
        final_ts = None

        # check if we jumped to another acquisition
        if (tstream.time[0] - self.last_time) > 5*(tstream.time[1] - tstream.time[0]):
            if self.cur_transit is None:
                # will be true when we start a new transit
                pass
            else:
                # start on a new transit and return current one
                self.cur_transit = None
                final_ts = self._finalize_transit()

        # if this is the start of a new grouping, setup transit bounds
        if self.cur_transit is None:
            self.cur_transit = obs.next_transit(self.src)
            # subtract half a day from start time to ensure we don't get following day
            self.start_t = obs.lsa_to_unix(obs.unix_to_lsa(self.cur_transit) - self.ha_span / 2,
                                           tstream.time[0] - 43200)
            self.end_t = obs.lsa_to_unix(obs.unix_to_lsa(self.cur_transit) + self.ha_span / 2,
                                         tstream.time[0])
            self.last_time = tstream.time[-1]

        # check if we've accumulated enough past the transit
        if tstream.time[-1] > self.end_t:
            self.cur_transit = None
            self.tstreams.append(tstream)
            final_ts = self._finalize_transit()
        else:
            self.tstreams.append(tstream)

        return final_ts

    def process_finish(self):
        """ Return the current transit before finishing.

            Returns
            -------
            ts: TimeStream
                Last (possibly incomplete) transit.
        """
        return self._finalize_transit()

    def _finalize_transit(self):

        # Find where transit starts and ends
        if len(self.tstreams) == 0:
            self.log.info("Did not find any transits.")
            return None
        all_t = np.concatenate([ts.time for ts in self.tstreams])
        start_ind = np.argmin(np.abs(all_t - self.start_t))
        stop_ind = np.argmin(np.abs(all_t - self.end_t))

        # Concatenate timestreams
        ts = tod.concatenate(self.tstreams, start={'time': start_ind},
                             stop={'time': stop_ind})
        _, dec = self.sky_obs.radec(self.src)
        ts.attrs['dec'] = dec._degrees
        ts.attrs['source_name'] = self.source

        self.tstreams = []

        return ts


class TransitRegridder(Regridder):
    """ Interpolate TimeStream transits onto a regular grid in hour angle.

        Attributes
        ----------
        samples : int
            Number of samples to interpolate onto.
        ha_span: float
            Span in degrees surrounding transit.
        lanczos_width : int
            Width of the Lanczos interpolation kernel.
        snr_cov: float
            Ratio of signal covariance to noise covariance (used for Wiener filter).
        source: str
            Name of the transiting source. (Must match what is used in `ch_util.ephemeris`.)
    """

    samples = config.Property(proptype=int, default=1024)
    lanczos_width = config.Property(proptype=int, default=5)
    snr_cov = config.Property(proptype=float, default=1e-8)
    ha_span = config.Property(proptype=float, default=180.)
    source = config.Property(proptype=str)

    def setup(self, observer=None):
        """Set the local observers position if not using CHIME.

        Parameters
        ----------
        observer : caput.time.Observer, optional
            Details of the observer, if not set default to CHIME.
        """
        self.observer = ephem.chime_observer() if observer is None else observer
        self.sky_obs = wrap_observer(self.observer)

        # Setup bounds for interpolation grid
        self.start = - self.ha_span / 2
        self.end = self.ha_span / 2

        try:
            self.src = ephem.source_dictionary[self.source]
        except KeyError:
            msg = ("Could not find source {} in catalogue. "
                   "Must use same spelling as in `ch_util.ephemeris`.".format(self.source))
            self.log.error(msg)
            raise PipelineConfigError(msg)

    def process(self, data):
        """Regrid visibility data onto a regular grid in hour angle.

        Parameters
        ----------
        data : TimeStream
            Time-ordered data.

        Returns
        -------
        new_data : SiderealStream
            The regridded data centered on the source RA.
        """

        # Redistribute if needed
        data.redistribute('freq')

        # View of data
        weight = data.weight[:].view(np.ndarray)
        vis_data = data.vis[:].view(np.ndarray)

        # Update observer time
        self.sky_obs.date = data.time[0]

        ra, _ = self.sky_obs.radec(self.src)
        ra = ra._degrees

        # Get input time grid
        lha = self.sky_obs.unix_to_lsa(data.time) - ra

        # perform regridding
        new_grid, new_vis, ni = self._regrid(vis_data, weight, lha)

        # Wrap to produce MPIArray
        new_vis = mpiarray.MPIArray.wrap(new_vis, axis=data.vis.distributed_axis)
        ni = mpiarray.MPIArray.wrap(ni, axis=data.vis.distributed_axis)

        # Create new container for output
        new_data = SiderealStream(axes_from=data, attrs_from=data,
                                  ra=new_grid + ra)
        new_data.redistribute('freq')
        new_data.vis[:] = new_vis
        new_data.weight[:] = ni

        return new_data


class MakeHolographyBeam(task.SingleTask):
    """ Repackage a holography transit into a beam container.
    """

    def setup(self, inputmap):
        self.inputmap = inputmap

    def process(self, data):
        pass
        # Sort products by polarization
        # Make a new feed index map
        # Rotate polarization into correct basis?
        # Return a TrackBeam


def wrap_observer(obs):
    return ephem.SkyfieldObserverWrapper(
            lon=obs.longitude,
            lat=obs.latitude,
            alt=obs.altitude,
            lsd_start=obs.lsd_start_day
    )
