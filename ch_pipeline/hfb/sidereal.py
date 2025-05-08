"""Regrid the data to sidereal time."""

import numpy as np
from beam_model.formed import FFTFormedActualBeamModel
from caput import mpiarray
from draco.analysis.sidereal import SiderealRegridderLinear

from .containers import HFBData, HFBRingMap


class HFBSiderealRegridder(SiderealRegridderLinear):
    """Regrid HFB data."""

    def setup(self, observer=None):
        """Setup the SiderealRegridder task.

        Parameters
        ----------
        observer : caput.astro.time.Observer, optional
            Details of the observer, if not set default to CHIME.
        """
        # Set up the default Observer
        if observer is None:
            from ch_ephem.observers import chime as observer

        self.observer = observer

        # Load beam model to look up reference zenith angles and hour angles of EW beams
        self.beam_mdl = FFTFormedActualBeamModel()

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

        # Get lengths of dimensions (local for frequency)
        lfreq, nsubfreq, nbeam, ntime = hfb_data.shape
        nra = self.samples

        # Massage down to a 3D array by combining the subfreq and beam axes;
        # this is to fit the expectations of the base class
        lfreq, *_, ntime = hfb_data.shape
        hfb_data = hfb_data.reshape(lfreq, -1, ntime)
        weight = weight.reshape(lfreq, -1, ntime)

        # Perform regridding
        _, sts, ni = self._regrid(hfb_data, weight, timestamp_lsd)

        # Get back to the 4D shape we need in here
        sts = sts.reshape(lfreq, nsubfreq, nbeam, nra)
        ni = ni.reshape(lfreq, nsubfreq, nbeam, nra)

        # Wrap to produce MPIArray
        sts = mpiarray.MPIArray.wrap(sts, axis=0)
        ni = mpiarray.MPIArray.wrap(ni, axis=0)

        # Calculate the EW-NS grid covering the beams in the data container (*_beams),
        # and what indices (*_map) the beams will take in the output
        ew_beams, ew_map = np.unique(data.beam // 256, return_inverse=True)
        ns_beams, ns_map = np.unique(data.beam % 256, return_inverse=True)

        # Look up reference zenith angles from beam model and convert to el = sin(za)
        za_deg = self.beam_mdl.reference_angles[ns_beams]
        el = np.sin(za_deg / 180.0 * np.pi)

        # Create container to hold regridded data
        sdata = HFBRingMap(
            axes_from=data,
            attrs_from=data,
            beam_ew=ew_beams,
            beam_ns=ns_beams,
            el=el,
            ra=self.samples,
        )
        sdata.redistribute("freq")
        sdata.attrs["lsd"] = self.start
        sdata.attrs["tag"] = f"lsd_{self.start:d}"

        # Put regridded data into output container, one beam at a time
        sh = sdata.hfb[:]
        sw = sdata.weight[:]
        for ii in range(nbeam):
            ewi, nsi = ew_map[ii], ns_map[ii]

            sh[:, :, ewi, nsi] = sts[:, :, ii]
            sw[:, :, ewi, nsi] = ni[:, :, ii]

        return sdata
