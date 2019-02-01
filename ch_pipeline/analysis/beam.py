""" Docstring goes here.
"""

import numpy as np

from caput import config, tod
from caput.pipeline import PipelineConfigError

from draco.core import task
from draco.analysis.transform import Regridder

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
        try:
            self.src = ephem.source_dictionary[self.source]
        except KeyError as e:
            msg = ("Could not find source {} in catalogue. "
                   "Must use same spelling as in `ch_util.ephemeris`.")
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
        obs = self.observer

        # placeholder for finalized transit when it is ready
        final_ts = None

        # check if we jumped to another acquisition
        if (tstream.times[0] - self.last_time) > 5*(tstream.times[1] - tstream.times[0]):
            if self.cur_transit is None:
                # will be true when we start a new transit
                pass
            else:
                # start on a new transit and return current one
                self.cur_transit = None
                final_ts = self._finalize_transit()

        # if this is the start of a new grouping, setup transit bounds
        if self.cur_transit is None:
            self.cur_transit = obs.next_transit(tstream.time[0])
            self.start_t = obs.lsa_to_unix(obs.unix_to_lsa(self.cur_transit) - self.ha_span / 2)
            self.end_t = obs.lsa_to_unix(obs.unix_to_lsa(self.cur_transit) + self.ha_span / 2)
            self.last_time = tstream.times[-1]

        # check if we've accumulated enough past the transit
        if tstream.times[-1] > self.end_t:
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
        all_t = np.concatenate([ ts.time for ts in self.tstreams])
        start_ind = np.argmin(np.abs(all_t - self.start_t))
        stop_ind = np.argmin(np.abs(all_t - self.end_t))

        # Concatenate timestreams
        ts = tod.concatenate(self.tstreams, start=start_ind, stop=stop_ind)
        _, dec = self.observer.radec(self.src)
        ts.attrs['dec'] = dec.degrees

        self.tstreams = []

        return ts
