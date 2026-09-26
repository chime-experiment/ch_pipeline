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
        hfb_data = hfb_data.reshape(lfreq, nsubfreq * nbeam, ntime)
        weight = weight.reshape(lfreq, nsubfreq * nbeam, ntime)

        # Take everything we need from the input before the regrid, so the
        # timestream can be released as soon as _regrid is done.
        ew_beams, ew_map = np.unique(data.beam // 256, return_inverse=True)
        ns_beams, ns_map = np.unique(data.beam % 256, return_inverse=True)

        za_deg = self.beam_mdl.reference_angles[ns_beams]
        el = np.sin(za_deg / 180.0 * np.pi)

        freq = data.index_map["freq"]
        subfreq = data.index_map["subfreq"]
        attrs = dict(data.attrs)

        # Perform regridding
        _, sts, ni = self._regrid(hfb_data, weight, timestamp_lsd)

        # Release the input timestream before allocating the output.
        del hfb_data, weight, data

        # Get back to the 4D shape we need in here
        sts = sts.reshape(lfreq, nsubfreq, nbeam, nra)
        ni = ni.reshape(lfreq, nsubfreq, nbeam, nra)

        # Wrap to produce MPIArray
        sts = mpiarray.MPIArray.wrap(sts, axis=0)
        ni = mpiarray.MPIArray.wrap(ni, axis=0)

        # Create container to hold regridded data
        sdata = HFBRingMap(
            freq=freq,
            subfreq=subfreq,
            beam_ew=ew_beams,
            beam_ns=ns_beams,
            el=el,
            ra=self.samples,
        )
        sdata.attrs.update(attrs)

        sdata.redistribute("freq")
        sdata.attrs["lsd"] = self.start
        sdata.attrs["tag"] = f"lsd_{int(self.start)}"

        # Put regridded data into output container, one beam at a time
        sh = sdata.hfb[:]
        sw = sdata.weight[:]
        for ii in range(nbeam):
            ewi, nsi = ew_map[ii], ns_map[ii]

            sh[:, :, ewi, nsi] = sts[:, :, ii]
            sw[:, :, ewi, nsi] = ni[:, :, ii]

        return sdata
