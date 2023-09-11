"""Regrid the data to sidereal time."""
import numpy as np

from caput import mpiarray
from ch_util import ephemeris
from draco.analysis.sidereal import SiderealRegridderLinear

from beam_model.formed import FFTFormedActualBeamModel

from .containers import HFBData, HFBRingMap


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

        # Get hour angles of EW beams (in deg) from beam model
        beam_mdl = FFTFormedActualBeamModel()
        self.ew_beam_offset_deg = beam_mdl.config["ew_spacing"]

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

        # Calculate the EW-NS grid covering the beams in the data container (*_beams),
        # and what indices (*_map) the beams will take in the output
        beams = data.index_map["beam"]
        ew_beams, ew_map = np.unique(beams // 256, return_inverse=True)
        ns_beams, ns_map = np.unique(beams % 256, return_inverse=True)

        # Look up reference zenith angles from beam model and convert to el = sin(za)
        za_deg = self.beam_mdl.reference_angles[ns_beams]
        el = np.sin(za_deg / 180.0 * np.pi)

        sdata = HFBRingMap(
            axes_from=data, attrs_from=data, ra=nra, beam=ew_beams, el=el
        )
        sdata.redistribute("freq")
        sdata.attrs["lsd"] = self.start
        sdata.attrs["tag"] = "lsd_%i" % self.start
        sh = sdata.hfb[:]
        sw = sdata.weight[:]

        for iewb, ewb in enumerate(ew_beams):
            # Select data and weights of each EW beam
            ewb_mask = ew_map == iewb
            hfb_data_ewb = hfb_data[:, :, ewb_mask, :]
            weight_ewb = weight[:, :, ewb_mask, :]

            # Massage down to a 3D array by combining the subfreq and beam axes,
            # this is to fit the expectations of the base class
            hfb_data_ewb = hfb_data_ewb.reshape(lfreq, -1, ntime)
            weight_ewb = weight_ewb.reshape(lfreq, -1, ntime)

            # Implement time offset for each EW beam
            timestamp_lsd_ewb = timestamp_lsd + self.ew_beam_offset_deg[ewb] / 360.0

            # Perform regridding
            _, sts_ewb, ni_ewb = self._regrid(
                hfb_data_ewb, weight_ewb, timestamp_lsd_ewb
            )

            # Get back to the 4D shape we need in here
            sts_ewb = sts_ewb.reshape(lfreq, nsubfreq, -1, nra)
            ni_ewb = ni_ewb.reshape(lfreq, nsubfreq, -1, nra)

            # Wrap to produce MPIArray
            sts_ewb = mpiarray.MPIArray.wrap(sts_ewb, axis=0)
            ni_ewb = mpiarray.MPIArray.wrap(ni_ewb, axis=0)

            # Insert regridded data and weights at correct EW and NS beam indices
            insb = ns_map[ewb_mask]
            sh[:, :, iewb, insb, :] = sts_ewb
            sw[:, :, iewb, insb, :] = ni_ewb

        return sdata
