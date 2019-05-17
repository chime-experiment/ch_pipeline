"""Module for thermal model stuff"""


import numpy as np

from caput import config, mpiutil, mpiarray, tod

from ch_util import ephemeris
# from ..core import task, containers, io
from .sidereal import SiderealGrouper

class WeatherGrouper(SiderealGrouper):
    """Group Weather files together, slice the files at the correct timestamps
    including the specified padding."""

    def _process_current_lsd(self):
        # Check if we have weather data for this day.
        if len(self._timestream_list) == 0:
            self.log.info("No weather data for this sidereal day")
            return None

        # Check if there is data missing
        # Calculate the length of data in this current LSD
        start = self._timestream_list[0].time[0]
        end = self._timestream_list[-1].time[-1]
        sid_seconds = 86400. / ephemeris.SIDEREAL_S

        if (end - start) < (sid_seconds + 2 * self.padding):
            self.log.info("Not enough weather data - skipping this day")
            return None

        lsd = self._current_lsd

        # Convert the current lsd day to unix time and pad it.
        unix_start = self.observer.lsd_to_unix(lsd)
        unix_end = self.observer.lsd_to_unix(lsd + 1)
        self.pad_start = unix_start - self.padding
        self.pad_end = unix_end + self.padding

        times = np.concatenate([ts.time for ts in self._timestream_list])
        start_ind = int(np.argmin(np.abs(times - self.pad_start)))
        stop_ind = int(np.argmin(np.abs(times - self.pad_end)))

        self.log.info("Constructing LSD:%i [%i files]",
                      lsd, len(self._timestream_list))
        # Concatenate timestreams
        ts = tod.concatenate(self._timestream_list, start=start_ind, stop=stop_ind)

        # Make sure that our timestamps of the concatenated files don't fall
        # out of the requested lsd time span
        if (ts.time[0] > unix_start) or (ts.time[-1] < unix_end):
            return None

        ts.attrs['tag'] = ('lsd_%i' % lsd)
        ts.attrs['lsd'] = lsd

        return ts
