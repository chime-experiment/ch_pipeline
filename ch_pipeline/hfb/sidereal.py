"""Regrid the data to sidereal time."""
import numpy as np

from caput import mpiarray
from ch_util import ephemeris
from draco.analysis.sidereal import SiderealRegridderLinear

from .containers import HFBData, HFBRingMap


# TODO: this is really just a copy-paste of much of the underlying .process(...), with
# some refactoring of the base task this task could probably be eliminated
class HFBSiderealRegridder(SiderealRegridderLinear):
    """Regrid HFB data."""

    def setup(self, observer=None):
        """Setup the SiderealRegridder task.

        Parameters
        ----------
        observer : caput.time.Observer, optional
            Details of the observer, if not set default to CHIME.
        """

        # Set up the default Observer
        self.observer = ephemeris.chime if observer is None else observer

    def process(self, data: HFBData) -> HFBRingMap:
        """Regrid HFB timestream data onto the sidereal day.

        Parameters
        ----------
        data
            The time ordered HFB data. This should span the whole sidereal day.

        Returns
        -------
        sdata
            The sidereal gridded data.
        """
        self.log.info(f"Regridding HFB data on {data.attrs['lsd']}.")
        data.redistribute("freq")

        # Convert data timestamps into LSDs
        timestamp_lsd = self.observer.unix_to_lsd(data.time)

        # Fetch which LSD this is to set bounds
        self.start = data.attrs["lsd"]
        self.end = self.start + 1

        # Get view of data
        weight = data.weight[:].local_array
        hfb_data = data.hfb[:].local_array

        # Massage down to a 3D array by combining the subfreq and beam axes, this is to
        # fit the expectations of the base class
        lfreq, *sb, ntime = hfb_data.shape
        hfb_data = hfb_data.reshape(lfreq, -1, ntime)
        weight = weight.reshape(lfreq, -1, ntime)

        # perform regridding
        # TODO: regrid each ew beam separately so we can fix up the offsets
        _, sts, ni = self._regrid(hfb_data, weight, timestamp_lsd)

        # Get back to the 4D shape we need in here
        sts = sts.reshape(lfreq, *sb, -1)
        ni = ni.reshape(lfreq, *sb, -1)

        # Calculate the EW-NS grid covering the beams in the data container (*_beams),
        # and what indices (*_map) the beams will take in the output
        beams = data.index_map["beam"]
        ew_beams, ew_map = np.unique(beams // 256, return_inverse=True)
        ns_beams, ns_map = np.unique(beams % 256, return_inverse=True)

        # Wrap to produce MPIArray
        sts = mpiarray.MPIArray.wrap(sts, axis=0)
        ni = mpiarray.MPIArray.wrap(ni, axis=0)

        # TODO: look up the x and y coordinates of the beams and provide a proper el
        # axis
        sdata = HFBRingMap(
            axes_from=data, attrs_from=data, ra=self.samples, beam=ew_beams, el=ns_beams
        )
        sdata.redistribute("freq")
        sdata.attrs["lsd"] = self.start
        sdata.attrs["tag"] = "lsd_%i" % self.start

        sh = sdata.hfb[:]
        sw = sdata.weight[:]
        for ii in range(len(beams)):
            ewi, nsi = ew_map[ii], ns_map[ii]

            sh[:, :, ewi, nsi] = sts[:, :, ii]
            sw[:, :, ewi, nsi] = ni[:, :, ii]

        return sdata