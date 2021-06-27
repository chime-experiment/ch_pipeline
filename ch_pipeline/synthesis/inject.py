"""
==============================================================
Catalog injection tasks (:mod:`~ch_pipeline.synthesis.inject`)
==============================================================

.. currentmodule:: ch_pipeline.synthesis.inject

Tasks for injecting source catalogs into visibility data.

Tasks
=====

.. autosummary::
    :toctree: generated/

    FluxCatalog
    FluxCatalogWithBeam
    FluxCatalogWithBeamExternal
    SpectroscopicCatalog
    SpectroscopicCatalogWithBeam
    SpectroscopicCatalogWithBeamExternal

Task Base Classes
-----------------
.. autosummary::
    :toctree: generated/

    BaseInject

"""
import os
import itertools

import numpy as np
import scipy.stats
import scipy.interpolate

from cora.util import units
from caput import config, interferometry
from draco.util import tools
from draco.core import task, io
from draco.analysis.beamform import icrs_to_cirs
from ch_util import fluxcat

from ..core import containers

# Constants
NU21 = units.nu21
C = units.c

SIGMA = {"XX": 14.87857614, "YY": 9.95746878, "XY": 12.17180995, "YX": 12.17180995}


class BaseInject(task.SingleTask):
    """Base class for injecting point source catalogs into visibility data.

    Attributes
    ----------
    polarization : str
        Determines the polarizations that will be output:
            - 'single' : 'XX' only (default)
            - 'copol' : 'XX' and 'YY' only
            - 'full' : 'XX', 'XY', 'YX' and 'YY' in this order
        The stack or prod axis of the output container will be the same as
        that of the input container, however nothing will be injected into
        products whose polarisation is not in whichever list was chosen above.
    inject : bool
        If True, then add the signal from the source catalog to the visibilities
        in the input container.  If False, then create a new container where the
        visibilities consist of the signal from the source catalog alone.  Default
        is False.
    nsigma : float
        Track sources for nsigma times the standard deviation of the theoretical
        beam response on each side of transit.  Default is 0.0, which results in
        point sources.
    """

    polarization = config.enum(["full", "copol", "single"], default="single")
    inject = config.Property(proptype=bool, default=False)
    nsigma = config.Property(proptype=float, default=0.0)

    def setup(self, manager):
        """Save the telescope that will be used by the process method.

        Parameters
        ----------
        manager : ProductManager, BeamTransfer, or TransitTelescope
            Contains a TransitTelescope object describing the telescope.
        """
        # Get the TransitTelescope object
        self.telescope = io.get_telescope(manager)
        self.latitude = np.deg2rad(self.telescope.latitude)

        # Polarizations.
        if self.polarization == "full":
            self.process_pol = ["XX", "XY", "YX", "YY"]
        elif self.polarization == "copol":
            self.process_pol = ["XX", "YY"]
        else:
            self.process_pol = ["XX"]

        self.output_attrs = {}

    def process(self, data):
        """Add sources from a catalog to the visibilities.

        Parameters
        ----------
        data : SiderealStream or TimeStream
            Data to inject the signal into.

        Returns
        -------
        out : SiderealStream or TimeStream
            If the inject config parameter is True, this will be the input container.
            If the inject config parameter is False, this will be a new container
            that has the same type as the input container.
        """
        # Determine the epoch
        if "ra" in data.index_map:
            ra = np.deg2rad(data.index_map["ra"])

            # Determine the epoch in order to convert to CIRS coordinates
            if "lsd" not in data.attrs:
                raise ValueError(
                    "SiderealStream must have an LSD attribute to calculate the epoch."
                )

            lsd = np.mean(data.attrs["lsd"])
            self.epoch = self.telescope.lsd_to_unix(lsd)

        else:
            # Convert data timestamps into LSAs (degrees)
            ra = np.deg2rad(self.telescope.unix_to_lsa(data.time))
            self.epoch = data.time.mean()

        nra = ra.size

        # Redistribute over frequency axis
        data.redistribute("freq")

        # Create the output container
        if self.inject:
            self.log.info("Injecing signal into existing container.")
            out = data
            if "tag" in out.attrs:
                out.attrs["tag"] = "_".join(
                    [out.attrs["tag"], "inject", self.catalog_tag]
                )
            else:
                out.attrs["tag"] = self.catalog_tag

        else:
            self.log.info("Creating new container to inject signal.")
            out = containers.empty_like(data)
            out.redistribute("freq")

            out.vis[:] = 0.0
            out.weight[:] = data.weight[:].copy()

            out.attrs["tag"] = self.catalog_tag

        # Save attributes describing the source catalog
        for key, val in self.output_attrs.items():
            out.attrs[key] = val

        # Extract the frequencies
        nfreq = out.vis.local_shape[0]
        ind_freq_start = out.vis.local_offset[0]
        ind_freq_end = ind_freq_start + nfreq

        freq = out.index_map["freq"][ind_freq_start:ind_freq_end]
        self.freq_local = freq

        min_freq, max_freq = np.percentile(freq["centre"], [0, 100])

        # Process the catalog
        self._process_catalog()

        # Initialize the beam
        self._initialize_beam_with_data()

        # Determine baseline distance and polariation
        dist, pol = _get_baseline_info(out, self.telescope)

        pol_uniq = [bp for bp in np.unique(pol) if bp in self.process_pol]
        pol_index = {pu: np.flatnonzero(pol == pu) for pu in pol_uniq}

        ibaseline = np.sort(np.concatenate([pol_index[pu] for pu in pol_uniq]))
        nbaseline = ibaseline.size

        dist = dist[:, ibaseline]
        dist = dist[:, np.newaxis, :, np.newaxis]

        pol = pol[ibaseline]
        pol_index = {pu: np.flatnonzero(pol == pu) for pu in pol_uniq}

        # Calculate the baseline distance in wavelengths
        lmbda = C * 1e-6 / freq["centre"][np.newaxis, :, np.newaxis, np.newaxis]
        uv = dist / lmbda

        # Print some info
        self.log.info(
            "Evaluating %d sources in frequency range [%0.2f MHz, %0.2f MHz]."
            % (self.nsource, min_freq, max_freq)
        )

        self.log.info(
            "Processing %s baselines."
            % str({key: len(val) for key, val in pol_index.items()})
        )

        # Create array to hold sky model
        vis = np.zeros((nfreq, nbaseline, nra), dtype=np.complex128)

        # Loop over sources
        for ss, (sra, sdec) in enumerate(zip(self.sra, self.sdec)):

            # Print message indicating how far along we are
            if ss % 1000 == 0:
                self.log.info("Adding source %d of %d." % (ss, self.nsource))

            # Determine the frequencies that have to be processed
            fslc = self._get_freq_slice(ss)

            # Get the amplitude
            amp = self._ampfunc(ss, freq[fslc])[:, np.newaxis, np.newaxis]

            # Determine the hour angle
            ha = _correct_phase_wrap(ra - sra)

            # Split the ra axis into contiguous groups
            groups = self._get_ha_slice(ha, min_freq, sdec)

            for slc in groups:

                sha = ha[slc]

                phase = (
                    amp
                    * interferometry.fringestop_phase(
                        sha[np.newaxis, np.newaxis, :],
                        self.latitude,
                        sdec,
                        *uv[:, fslc],
                    ).conj()
                )

                # Modulate by the beam
                if self.apply_beam:

                    for pstr, ipol in pol_index.items():
                        phase[:, ipol, :] *= self._beamfunc(
                            freq["centre"][fslc], pstr, sdec, sha
                        )[:, np.newaxis, :]

                # Add to visibility
                vis[fslc, :, slc] += phase

        # Add the sky visibilities to the output container
        out.vis[:][:, ibaseline, :] += vis

        # Return container
        return out

    def _ampfunc(self, source_index, freq):
        """Return the source amplitude.

        Parameters
        ----------
        source_index : int
            Index into the catalog of the source under consideration.
        freq : np.ndarray[nfreq,] of dtype [("centre"), ("width")]
            Frequency bins in MHz.

        Returns
        -------
        amp : np.ndarray[nfreq,]
            Amplitude of the source at the requested frequencies.
        """

        return np.ones(freq.size, dtype=np.float32)

    @property
    def apply_beam(self):
        """Modulate the source amplitude by a primary beam model."""
        return False

    def _initialize_beam_with_data(self):
        """Initialize the beam.

        This method is called during process and can be overridden to perform
        any beam initialization that requires the data to be parsed first.
        """
        pass

    def _get_ha_slice(self, ha, freq, dec):
        """Find the RAs that should be processed for each source.

        Parameters
        ----------
        ha : np.ndarray
            The hour angle of the source in radians.
        freq : float
            The frequency in MHz.
        dec : float
            The declination of the source in radians.

        Returns
        -------
        group : list of slices
            List of slices into the RA/time axis.
        """

        if self.nsigma > 0.0:
            # Split the ra axis into contiguous groups
            sigma = SIGMA["XX"] / freq / np.cos(dec)
            nearby = np.flatnonzero(np.abs(ha) < (self.nsigma * sigma))
            groups = _find_contiguous_slices(nearby)

        else:
            # Find the nearest ra bin
            sind = np.argmin(np.abs(ha))
            groups = [slice(sind, sind + 1)]

        return groups

    def _get_freq_slice(self, source_index):
        """Find the frequencies that should be processed for each source.

        Paramters
        ---------
        source_index : int
            Index into the catalog of the source under consideration.

        Returns
        -------
        slc : slice
            Slice into the frequency axis.
        """

        return slice(None)


class FluxCatalog(BaseInject):
    """Base class for injecting flux catalogs into visibility data.

    Attributes
    ----------
    min_dec : float
        Only include sources with a declination in degrees
        greater than this value.
    max_dec : float
        Only include sources with a declination in degrees
        less than this value.
    min_flux : float
        Only include sources with a flux at NOMINAL_FREQ in mJy
        greater than this value.
    filename : str
        Name of the file containing the flux catalog.  Must be a .json
        file that can be loaded with ch_util.fluxcat.FluxCatalog.load.
    """

    min_dec = config.Property(proptype=float, default=0.0)
    max_dec = config.Property(proptype=float, default=70.0)
    min_flux = config.Property(proptype=float, default=0.0)
    filename = config.Property(proptype=str, default="")

    NOMINAL_FREQ = 600.0

    def setup(self, manager):
        """Load the requested flux catalog.

        Parameters
        ----------
        manager : ProductManager, BeamTransfer, or TransitTelescope
            Contains a TransitTelescope object describing the telescope.
        """

        super().setup(manager)

        # Delete all previously loaded catalogs
        collections = fluxcat.FluxCatalog.loaded_collections()
        for collection in collections:
            fluxcat.FluxCatalog.delete_loaded_collection(collection[0])

        # Load the flux catalog
        fluxcat.FluxCatalog.load(self.filename, overwrite=2)

        self.catalog_tag = os.path.splitext(os.path.basename(self.filename))[0]

        # Save attributes describing the source selection to the output container
        self.output_attrs["catalog"] = self.catalog_tag
        self.output_attrs["min_dec"] = self.min_dec
        self.output_attrs["max_dec"] = self.max_dec
        self.output_attrs["min_flux"] = self.min_flux

    def _ampfunc(self, source_index, freq):
        """Return the flux of the source.

        Uses FluxCatalog to evaluate a model that has been fit to
        measurements of the source flux made by other telescopes.

        Parameters
        ----------
        source_index : int
            Index into the catalog of the source under consideration.
        freq : np.ndarray[nfreq,] of dtype [("centre"), ("width")]
            Frequency bins in MHz.

        Returns
        -------
        amp : np.ndarray[nfreq,]
            Flux of the source in Jy at the requested frequencies.
        """

        return fluxcat.FluxCatalog[self.sources[source_index]].predict_flux(
            freq["centre"]
        )

    def _process_catalog(self):
        """Process the catalog.

        This method filters the catalog based on the provided config parameters.
        It also converts from ICRS to CIRS coordinates.  For this reason, the
        epoch attribute must be defined before calling this method.
        """
        sources = fluxcat.FluxCatalog.keys()

        sra = np.array([fluxcat.FluxCatalog[src].ra for src in sources])
        sdec = np.array([fluxcat.FluxCatalog[src].dec for src in sources])

        nominal_flux = np.array(
            [
                fluxcat.FluxCatalog[src].predict_flux(self.NOMINAL_FREQ)
                for src in sources
            ]
        )
        min_flux = 1e-3 * self.min_flux

        ikeep = np.flatnonzero(
            (sdec >= self.min_dec) & (sdec <= self.max_dec) & (nominal_flux >= min_flux)
        )

        self.log.info(
            "There are %d sources between declination [%d, %d] with S(600) > %0.3f."
            % (ikeep.size, self.min_dec, self.max_dec, min_flux)
        )

        self.sources = [sources[ik] for ik in ikeep]
        self.nsource = len(self.sources)

        sra, sdec = icrs_to_cirs(sra[ikeep], sdec[ikeep], self.epoch)

        self.sra = np.deg2rad(sra)
        self.sdec = np.deg2rad(sdec)


class SpectroscopicCatalog(BaseInject):
    """Base class for injecting spectroscopic catalogs into visibility data.

    Model for the signal is

        V(freq, z) = amplitude * exp(-|freq - freq_21cm(z)| / scale).

    This function is integrated over each frequency bin within +/- nscale * scale.

    Attributes
    ----------
    amplitude : float
        Amplitude of the signal in Jy.
    scale : float
        Scale parameter for the signal in MHz.  If set to zero,
        then the signal is assumed to be a delta function in frequency.
    nscale : float
        Inject the model for the signal into all frequency bins
        within +/- nscale * scale.
    perturb_z : float
        Perturb the redshifts by a random error drawn from a Gaussian
        with standard deviation equal to the "redshift/z_error" dataset,
        or if that dataset is entirely zero, from a tracer-specific
        distribution inferred from the catalog name.
    filename : str
        Name of the hdf5 file containing the spectroscopic catalog.
    """

    amplitude = config.Property(proptype=float, default=1.0)
    scale = config.Property(proptype=float, default=0.0)
    nscale = config.Property(proptype=float, default=5.0)
    perturb_z = config.Property(proptype=bool, default=False)
    filename = config.Property(proptype=str, default="")

    def setup(self, manager):
        """Save the spectroscopic that will be used by process.

        Parameters
        ----------
        manager : ProductManager, BeamTransfer, or TransitTelescope
            Contains a TransitTelescope object describing the telescope.
        source_cat : SpectroscopicCatalog
            Catalog of sources with redshift information.
        """

        super().setup(manager)

        # Save the catalog as attribute
        self.catalog = containers.SpectroscopicCatalog.from_file(self.filename)
        self.catalog_tag = self.catalog.attrs.get("tag", None)

        # Save attributes describing the signal model to the output container
        self.output_attrs["catalog"] = self.catalog_tag
        self.output_attrs["amplitude"] = self.amplitude
        self.output_attrs["scale"] = self.scale
        self.output_attrs["nscale"] = self.nscale
        self.output_attrs["perturb_z"] = self.perturb_z

        # Print some info
        self.log.info(
            "21cm amplitude set to %0.1e Jy.  Scale set to %0.2f MHz."
            % (self.amplitude, self.scale)
        )

    def _ampfunc(self, source_index, freq):
        """Return the source amplitude.

        Parameters
        ----------
        source_index : int
            Index into the catalog of the source under consideration.
        freq : np.ndarray[nfreq,] of dtype [("centre"), ("width")]
            Frequency bins in MHz.

        Returns
        -------
        amp : np.ndarray[nfreq,]
            Amplitude of the source at the requested frequencies.
        """

        sfreq = self.sfreq[source_index]

        fedge_lower = freq["centre"] - 0.5 * freq["width"]
        fedge_upper = freq["centre"] + 0.5 * freq["width"]

        # If scale is 0, then model the correlation function
        # as a delta function in frequency.
        if self.scale == 0.0:
            return self.amplitude * (
                (sfreq > fedge_lower) & (sfreq <= fedge_upper)
            ).astype(np.float32)

        # Model the correlation function as an exponential.
        # Integrate the exponential over the bin width.
        dlower = fedge_lower - sfreq
        dupper = fedge_upper - sfreq

        adlower = np.abs(dlower / self.scale)
        adupper = np.abs(dupper / self.scale)

        amp = (self.amplitude / self.scale) * np.where(
            (dlower < 0.0) & (dupper > 0.0),
            2.0 - np.exp(-adlower) - np.exp(-adupper),
            np.abs(np.exp(-adlower) - np.exp(-adupper)),
        )

        return amp

    def _process_catalog(self):
        """Process the spectroscopic catalog.

        This method converts the catalog from ICRS to CIRS coordinates.
        For this reason, the epoch attribute must be defined. This method
        also perturbs redshifts (if requested) and determines what frequency
        bins should be processed for each source.
        """

        catalog = self.catalog

        # Convert redshift to radio frequency of the redshifted 21cm line
        if "redshift" not in catalog:
            raise ValueError("Input is missing a required redshift table.")

        z = catalog["redshift"]["z"][:]

        if self.perturb_z:
            zerr = catalog["redshift"]["z_error"][:]
            if np.any(zerr):
                self.log.info("Perturbing redshift by error in catalog.")
                z = z + np.random.normal(scale=zerr)

            else:

                if "tracer" in catalog.index_map["object_id"].dtype.fields:
                    tracer = catalog.index_map["object_id"]["tracer"][:]

                    utracers, uindex = np.unique(tracer, return_inverse=True)
                    for uu, utracer in enumerate(utracers):
                        this_tracer = np.flatnonzero(uu == uindex)
                        self.log.info(
                            f"Perturbing {this_tracer.size} source redshifts by "
                            f"{utracer} error distribution."
                        )
                        z[this_tracer] = perturb_redshift(
                            z[this_tracer], tracer=utracer
                        )

                else:

                    tracer = catalog.attrs.get("tracer", None)
                    if tracer is None:
                        for tr in ["ELG", "LRG", "QSO"]:
                            if tr in catalog.attrs["tag"]:
                                tracer = tr
                                break

                    if tracer is None:
                        raise RuntimeError(
                            "To perturb redshifts, must provide "
                            "the redshift error dataset OR the "
                            "name of the tracer in the catalog "
                            "'object_id', 'tracer' attribute, "
                            "or 'tag' attribute."
                        )

                    self.log.info(
                        f"Perturbing redshift by {tracer} error distribution."
                    )
                    z = perturb_redshift(z, tracer=tracer)

        sfreq = NU21 / (z + 1.0)  # MHz

        # Sort by frequency
        isort = np.argsort(sfreq)
        sfreq = sfreq[isort]

        # Filter for sources that overlap with the range of frequencies
        # processed by this node
        fedge_lb = self.freq_local["centre"] - 0.5 * self.freq_local["width"]
        fedge_ub = self.freq_local["centre"] + 0.5 * self.freq_local["width"]

        min_freq = np.min(fedge_lb)
        max_freq = np.max(fedge_ub)

        win = self.nsigma * self.scale
        sf_lb = sfreq - win
        sf_ub = sfreq + win

        keep = np.flatnonzero((sf_ub >= min_freq) & (sf_lb <= max_freq))

        sf_lb = sf_lb[keep]
        sf_ub = sf_ub[keep]

        self.sfreq = sfreq[keep]
        isort = isort[keep]

        # Identify the frequency bins that have non-zero signal for each source
        bin_sort = np.argsort(fedge_lb)
        aa = bin_sort[
            np.maximum(0, np.searchsorted(fedge_lb[bin_sort], sf_lb, side="left") - 1)
        ]
        bb = bin_sort[
            np.maximum(0, np.searchsorted(fedge_lb[bin_sort], sf_ub, side="left") - 1)
        ]

        # The next two lines are necessary because the frequency axis may be reversed
        self.lower_freq_bin = np.minimum(aa, bb)
        self.upper_freq_bin = np.maximum(aa, bb)

        # Convert positions from ICRS to CIRS coordinates.
        # Also converts to radians.
        if "position" not in catalog:
            raise ValueError("Input is missing a position table.")

        sra, sdec = icrs_to_cirs(
            catalog["position"]["ra"][isort],
            catalog["position"]["dec"][isort],
            self.epoch,
        )

        self.sra = np.deg2rad(sra)
        self.sdec = np.deg2rad(sdec)

        self.nsource = len(self.sra)

    def _get_freq_slice(self, source_index):
        """Look up the frequencies to be processed for each source.

        This is pre-determined by the _process_catalog method.

        Paramters
        ---------
        source_index : int
            Index into the catalog of the source under consideration.

        Returns
        -------
        slc : slice
            Slice into the frequency axis.
        """

        return slice(
            self.lower_freq_bin[source_index], self.upper_freq_bin[source_index] + 1
        )


class Beam(BaseInject):
    """Base class for modulating source amplitudes by a primary beam model.

    The beam model is obtained from the beam method of the telescope provided
    during setup.
    """

    def setup(self, manager):
        """Perform additional setup required for telescope based beam model.

        Parameters
        ----------
        manager : ProductManager, BeamTransfer, or TransitTelescope
            Contains a TransitTelescope object describing the telescope.
        """

        super().setup(manager)
        self.map_pol_feed = {
            pstr: list(self.telescope.polarisation).index(pstr) for pstr in ["X", "Y"]
        }

    def _beamfunc(self, freq, pol, dec, ha):
        """Calculate the primary beam at the location of a source as it transits.

        Parameters
        ----------
        freq : np.ndarray[nfreq,]
            The frequency in MHz.
        pol : str
            String specifying the polarisation,
            either 'XX', 'XY', 'YX', or 'YY'.
        dec : float
            The declination of the source in radians.
        ha : np.ndarray[nha,]
            The hour angle of the source in radians.

        Returns
        -------
        primary_beam : np.ndarray[nfreq, nha]
            The primary beam as a function of frequency and hour angle
            at the sources declination for the requested polarisation.
        """

        index_freq = np.array(
            [np.argmin(np.abs(nu - self.telescope.frequencies)) for nu in freq]
        )

        p_stokesI = np.array([[1.0, 0.0], [0.0, 1.0]])

        angpos = np.array([(0.5 * np.pi - dec) * np.ones_like(ha), ha]).T

        primary_beam = np.zeros((freq.size, ha.size), dtype=np.float64)

        for ff, freq in enumerate(index_freq):

            bii = self.telescope.beam(self.map_pol_feed[pol[0]], freq, angpos)

            if pol[0] != pol[1]:
                bjj = self.telescope.beam(self.map_pol_feed[pol[1]], freq, angpos)
            else:
                bjj = bii

            primary_beam[ff] = np.sum(bii * np.dot(bjj.conjugate(), p_stokesI), axis=1)

        return primary_beam

    @property
    def apply_beam(self):
        """Modulate the source amplitude by a primary beam model."""
        return True


class BeamExternal(BaseInject):
    """Base class that provides methods for evaluating an external beam model."""

    def setup(self, manager, beam):
        """Perform the generic setup and initialize the beam.

        Parameters
        ----------
        manager : either ProductManager, BeamTransfer, or TransitTelescope
            Contains a TransitTelescope object describing the telescope.

        beam : GridBeam`
            Model for the primary beam.
        """

        super().setup(manager)
        self._initialize_beam(beam)

    @property
    def apply_beam(self):
        """Modulate the source amplitude by a primary beam model."""
        return True

    def _initialize_beam(self, beam):
        """Initialize based on the beam container type.

        Parameters
        ----------
        beam : containers.GridBeam
            Container holding the model for the primary beam.
            Currently only accepts GridBeam type containers.
        """

        if isinstance(beam, containers.GridBeam):
            self._initialize_grid_beam(beam)
            self._beamfunc = self._grid_beam

        else:
            raise ValueError(f"Do not recognize beam container: {beam.__class__}")

    def _initialize_beam_with_data(self):
        """Check that the beam and visibility data have the same local frequencies."""

        freq = self.freq_local["centre"]

        if (freq.size != self._beam_freq.size) or np.any(freq != self._beam_freq):
            raise RuntimeError("Beam and visibility frequency axes do not match.")

    def _initialize_grid_beam(self, gbeam):
        """Create an interpolator for a GridBeam.

        Parameters
        ----------
        gbeam : containers.GridBeam
            Model for the primary beam on a celestial grid where
            (theta, phi) = (declination, hour angle) in degrees.  The beam
            must be in power units and must have a length 1 input axis that
            contains the "baseline averaged" beam, which will be applied to
            all baselines of a given polarisation.
        """

        # Make sure the beam is in celestial coordinates
        if gbeam.coords != "celestial":
            raise RuntimeError(
                "GridBeam must be converted to celestial coordinates for beamforming."
            )

        # Make sure there is a single beam to use for all inputs
        if gbeam.input.size > 1:
            raise NotImplementedError(
                "Do not support input-dependent beams at the moment."
            )

        # Distribute over frequencies, extract local frequencies
        gbeam.redistribute("freq")

        lo = gbeam.beam.local_offset[0]
        nfreq = gbeam.beam.local_shape[0]
        self._beam_freq = gbeam.freq[lo : lo + nfreq]

        # Find the relevant indices into the polarisation axis
        ipol = np.array([list(gbeam.pol).index(pstr) for pstr in self.process_pol])
        npol = ipol.size
        self._beam_pol = [gbeam.pol[ip] for ip in ipol]

        # Extract beam
        flag = gbeam.weight[:, :, 0][:, ipol] > 0.0
        beam = np.where(flag, gbeam.beam[:, :, 0][:, ipol].real, 0.0)

        # Convert the declination and hour angle axis to radians, make sure they are sorted
        ha = (gbeam.phi + 180.0) % 360.0 - 180.0
        isort = np.argsort(ha)
        ha = np.radians(ha[isort])

        dec = np.radians(gbeam.theta)

        # Create an interpolator for each frequency and polarisation
        self._beam = [
            [
                scipy.interpolate.RectBivariateSpline(dec, ha, beam[ff, pp][:, isort])
                for pp in range(npol)
            ]
            for ff in range(nfreq)
        ]

        self._beam_flag = [
            [
                scipy.interpolate.RectBivariateSpline(
                    dec, ha, flag[ff, pp][:, isort].astype(np.float32)
                )
                for pp in range(npol)
            ]
            for ff in range(nfreq)
        ]

        self.log.info("Grid beam initialized.")

    def _grid_beam(self, freq, pol, dec, ha):
        """Interpolate a GridBeam to the requested declination and hour angles.

        Parameters
        ----------
        freq : np.ndarray[nfreq,]
            The frequency in MHz.
        pol : str
            String specifying the polarisation,
            either 'XX', 'XY', 'YX', or 'YY'.
        dec : float
            The declination of the source in radians.
        ha : np.ndarray[nha,]
            The hour angle of the source in radians.

        Returns
        -------
        primay_beam : np.ndarray[nfreq, nha]
            The primary beam as a function of frequency and hour angle
            at the sources declination for the requested polarisation.
        """

        index_freq = np.array([np.argmin(np.abs(nu - self._beam_freq)) for nu in freq])

        pp = self._beam_pol.index(pol)

        primay_beam = np.array([self._beam[ff][pp](dec, ha)[0] for ff in index_freq])

        flag = np.array(
            [
                np.abs(self._beam_flag[ff][pp](dec, ha)[0] - 1.0) < 0.01
                for ff in index_freq
            ]
        )

        return np.where(flag, primay_beam, 0.0)


class FluxCatalogWithBeam(Beam, FluxCatalog):
    """Inject a flux catalog using an analytical beam model."""


class FluxCatalogWithBeamExternal(BeamExternal, FluxCatalog):
    """Inject a flux catalog using an external beam model."""


class SpectroscopicCatalogWithBeam(Beam, SpectroscopicCatalog):
    """Inject a spectroscopic catalog using an analytical beam model."""


class SpectroscopicCatalogWithBeamExternal(BeamExternal, SpectroscopicCatalog):
    """Inject a spectroscopic catalog using an external beam model."""


def qso_velocity_error(nsample):

    QSO_SIG1 = 150.0
    QSO_SIG2 = 1000.0
    QSO_F = 4.478

    dv1 = np.random.normal(scale=QSO_SIG1, size=nsample)
    dv2 = np.random.normal(scale=QSO_SIG2, size=nsample)

    u = np.random.uniform(size=nsample)
    flag = u >= (1.0 / (1.0 + QSO_F))

    dv = np.where(flag, dv1, dv2)

    return dv


def lrg_velocity_error(nsample):

    LRG_SIG = 91.8

    dv = np.random.normal(scale=LRG_SIG, size=nsample)

    return dv


def elg_velocity_error(nsample):

    ELG_SIG = 11.877
    ELG_LAMBDA = -0.4028

    dv = scipy.stats.tukeylambda.rvs(ELG_LAMBDA, scale=ELG_SIG, size=nsample)

    return dv


velocity_error_function_lookup = {
    "QSO": qso_velocity_error,
    "ELG": elg_velocity_error,
    "LRG": lrg_velocity_error,
}


def perturb_redshift(z, tracer="QSO"):

    err_func = velocity_error_function_lookup[tracer]

    dv = err_func(z.size)

    dz = (1.0 + z) * dv / (C * 1e-3)

    return z + dz


def _get_baseline_info(data, telescope):
    # Determine stack axis
    stack_new, stack_flag = tools.redefine_stack_index_map(
        telescope, data.input, data.prod, data.stack, data.reverse_map["stack"]
    )

    # Extract the local products
    nstack = data.vis.local_shape[1]
    start = data.vis.local_offset[1]
    end = start + nstack

    stack_new = stack_new[start:end]

    # Get representative products
    ps = data.prod[stack_new["prod"]]
    conj = stack_new["conjugate"]

    prodstack = ps.copy()
    prodstack["input_a"] = np.where(conj, ps["input_b"], ps["input_a"])
    prodstack["input_b"] = np.where(conj, ps["input_a"], ps["input_b"])

    # Figure out mapping between inputs in data file and inputs in telescope
    tel_index = tools.find_inputs(
        telescope.input_index, data.input, require_match=False
    )

    # Use the mapping to extract polarisation and position of each input
    input_pos = np.zeros((2, data.input.size), dtype=np.float32)
    for ii, ti in enumerate(tel_index):
        if ti is not None:
            input_pos[:, ii] = telescope.feedpositions[ti, :]

    input_pol = np.array(
        [telescope.polarisation[ti] if ti is not None else "N" for ti in tel_index]
    )

    # Construct polarisation and distance for each baseline
    aa, bb = prodstack["input_a"], prodstack["input_b"]

    baseline_dist = input_pos[:, aa] - input_pos[:, bb]
    baseline_pol = np.core.defchararray.add(input_pol[aa], input_pol[bb])

    return baseline_dist, baseline_pol


def _correct_phase_wrap(phi):
    return ((phi + np.pi) % (2.0 * np.pi)) - np.pi


def _find_contiguous_slices(index):
    slices = []
    for w, z in itertools.groupby(index, lambda x, y=itertools.count(): next(y) - x):
        grouped = list(z)
        slices.append(slice(grouped[0], grouped[-1] + 1))
    return slices
