"""Tasks for injecting source catalogs into visibility data."""

import json
from pydoc import locate

import numpy as np
import scipy.interpolate
from caput import config
from caput.astro import constants
from caput.astro import skyfield as skf
from caput.astro import time as ctime
from caput.astro.coordinates import spherical
from caput.pipeline import exceptions, tasklib
from caput.util import mpitools
from ch_util import fluxcat
from draco.core import containers, io
from draco.util import interferometry, tools
from mpi4py import MPI
from scipy.special import erf

# Constants
NU21 = constants.nu21
C = constants.c

SIGMA = {"XX": 14.87857614, "YY": 9.95746878, "XY": 12.17180995, "YX": 12.17180995}


class BaseInject(tasklib.base.ContainerTask):
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
    copy_weight = config.Property(proptype=bool, default=True)
    anti_podal = config.Property(proptype=bool, default=False)
    full_track = config.Property(proptype=bool, default=False)
    nsigma = config.Property(proptype=float, default=0.0)

    def setup(self, manager, catalog):
        """Save the telescope that will be used by the process method.

        Parameters
        ----------
        manager : ProductManager, BeamTransfer, or TransitTelescope
            Contains a TransitTelescope object describing the telescope.
        catalog : SourceCatalog or FluxModelCatalog
            Sources to inject.
        """
        # Get the TransitTelescope object
        self.telescope = io.get_telescope(manager)
        self.latitude = np.deg2rad(self.telescope.latitude)

        self.transits = (
            [0.0, np.pi] if self.anti_podal and not self.full_track else [0.0]
        )
        self.log.info(f"Injecting at the following hour angles:  {self.transits}")

        # Polarizations.
        if self.polarization == "full":
            self.process_pol = ["XX", "XY", "YX", "YY"]
        elif self.polarization == "copol":
            self.process_pol = ["XX", "YY"]
        else:
            self.process_pol = ["XX"]

        # Save the catalog as attribute
        self.catalog = catalog
        self.catalog_tag = self.catalog.attrs.get("tag", None)

        # Save output attributes
        self.output_attrs = {}
        self.output_attrs["catalog"] = self.catalog_tag

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
                self.epoch = None
                self.log.warning(
                    "LSD attribute not available.  Using ICRS coordinates."
                )
            else:
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
            out = data.__class__(
                axes_from=data,
                attrs_from=data,
                distributed=data.distributed,
                comm=data.comm,
            )
            out.redistribute("freq")

            out.vis[:] = 0.0
            if self.copy_weight:
                out.weight[:] = data.weight[:].copy()
            else:
                out.weight[:] = 1.0

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
            f"Evaluating {self.nsource} sources in frequency range "
            f"[{min_freq:0.2f} MHz, {max_freq:0.2f} MHz]."
        )

        baseline_count = {key: len(val) for key, val in pol_index.items()}
        self.log.info(f"Processing {baseline_count} baselines.")

        # Create array to hold sky model
        vis = np.zeros((nfreq, nbaseline, nra), dtype=np.complex128)

        # Loop over sources
        for ss, (sra, sdec) in enumerate(zip(self.sra, self.sdec)):

            # Print message indicating how far along we are
            if ss % 100 == 0:
                self.log.info(f"Adding source {ss} of {self.nsource}.")

            # Determine the frequencies that have to be processed
            fslc = self._get_freq_slice(ss)

            # Get the amplitude
            amp = self._ampfunc(ss, freq[fslc])[:, np.newaxis, np.newaxis]

            # Determine the hour angle
            ha = tools.correct_phase_wrap(ra - sra)

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
        out.vis[:].local_array[:, ibaseline, :] += vis

        # Return container
        return out

    def _process_catalog(self):
        """Process the catalog.

        This method filters the catalog based on the provided config parameters.
        It also converts from ICRS to CIRS coordinates.  For this reason, the
        epoch attribute must be defined before calling this method.
        """
        self.sources = self.catalog.index_map["object_id"][:]
        self.nsource = len(self.sources)

        sra = self.catalog["position"]["ra"][:]
        sdec = self.catalog["position"]["dec"][:]

        coordinate_system = self.catalog["position"].attrs.get("coords", "icrs")
        if self.epoch is None:
            self.log.info(
                f"Epoch not available.  Using {coordinate_system} coordinates."
            )

        elif coordinate_system == "icrs":
            date_string = ctime.unix_to_datetime(self.epoch).strftime("%Y%m%dT%H%M%SZ")
            self.log.info(f"Converting from ICRS to CIRS coordinates at {date_string}.")
            sra, sdec = _icrs_to_cirs(sra, sdec, self.epoch, self.telescope)

        elif coordinate_system == "cirs":
            self.log.info("Using the CIRS coordinates provided with the catalog.")

        else:
            raise RuntimeError(
                f"Do not recognize coordinate system {coordinate_system}."
                "Options are ICRS and CIRS."
            )

        self.sra = np.deg2rad(sra)
        self.sdec = np.deg2rad(sdec)

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
        sin_alt = np.sin(dec) * np.sin(self.latitude) + np.cos(dec) * np.cos(
            self.latitude
        ) * np.cos(ha)
        alt = np.arcsin(np.minimum(np.maximum(sin_alt, -1.0), 1.0))
        above_horizon = alt > 0.0

        if self.full_track:
            groups = tools.find_contiguous_slices(np.flatnonzero(above_horizon))

        else:
            # The RAs near the podal and anti-podal transit.  Take their union, since
            # near the pole the two windows overlap and the RAs in both must be
            # processed once.
            nearby = np.zeros(ha.size, dtype=bool)

            for hac in self.transits:

                dha = tools.correct_phase_wrap(ha - hac)

                if self.nsigma > 0.0:
                    sigma = SIGMA["XX"] / freq / np.cos(dec)
                    nearby |= np.abs(dha) < (self.nsigma * sigma)

                else:
                    # Find the nearest ra bin
                    nearby[np.argmin(np.abs(dha))] = True

            # Split the ra axis into contiguous groups
            groups = tools.find_contiguous_slices(
                np.flatnonzero(above_horizon & nearby)
            )

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


class ConvertICRStoCIRS(tasklib.base.ContainerTask):
    """Convert the ICRS positions of a catalogue to CIRS at the epoch of the data."""

    find_nearest = config.Property(proptype=bool, default=True)
    apparent = config.Property(proptype=bool, default=True)

    def setup(self, manager):
        """Store the telescope object.

        Parameters
        ----------
        manager : either `ProductManager`, `BeamTransfer` or `TransitTelescope`
            Contains a TransitTelescope object describing the telescope.
        catalog : SourceCatalog or FluxModelCatalog
            Sources to inject.
        """
        self.telescope = io.get_telescope(manager)

    def process(self, catalog, data):
        """Convert ICRS coordinates to the CIRS coordinates on the epoch in the data.

        Parameters
        ----------
        catalog : SourceCatalog, SpectroscopicCatalog
            Catalog of source positions in ICRS coordinates.

        data : StackMask, SiderealStream, TimeStream
            Data whose

        Returns
        -------
        catalog_cirs : SourceCatalog, SpectroscopicCatalog, SourceCatalogVersusRA
            Catalog of source positions in CIRS coordinates.
        """
        # Extract the source coordinates
        input_coords = catalog["position"].attrs.get("coords", "icrs")
        if input_coords != "icrs":
            raise RuntimeError("Source positions must be in ICRS coordinates.")

        ra_icrs = catalog["position"]["ra"][:]
        dec_icrs = catalog["position"]["dec"][:]

        # Each rank will process a subset of the sources
        nsource = ra_icrs.size
        nsource_per_rank, ssrc, esrc = mpitools.split_all(nsource, comm=self.comm)
        sslc = slice(ssrc[self.comm.rank], esrc[self.comm.rank])
        nlocal = nsource_per_rank[self.comm.rank]

        positions = skf.skyfield_star_from_ra_dec(ra_icrs[sslc], dec_icrs[sslc])

        # Determine the local sidereal days that contributed to data
        if False:  # isinstance(data, containers.StackMask):

            lsds = data.lsd[:]
            ra = data.ra[:]
            weight = data.flag[:].astype(float)
            self.log.info(
                f"Calculating positions using epoch and flags from {data.lsd.size} LSDs."
            )

        elif "ra" in data.index_map:

            lsds = np.atleast_1d(
                data.attrs["lsd"] if "lsd" in data.attrs else data.attrs["csd"]
            )
            ra = data.ra
            weight = np.ones((lsds.size, ra.size), dtype=float)
            self.log.info(f"Calculating positions using epoch from {lsds.size} LSDs.")

        elif "time" in data.index_map:

            lsds = np.atleast_1d(np.fix(np.mean(self.telescope.unix_to_lsd(data.time))))
            ra = self.telescope.lsa(data.time)
            self.log.info(
                f"Calculating positions using epoch corresponding to LSD {lsds[0]}."
            )

        else:
            raise RuntimeError("Unable to extract LSD from input container.")

        nlsd = lsds.size

        # Calculate the source coordinates on each LSD
        ra_cirs = np.zeros((nlocal, nlsd), dtype=ra_icrs.dtype)
        dec_cirs = np.zeros((nlocal, nlsd), dtype=dec_icrs.dtype)

        for ii, lsd in enumerate(lsds):

            epoch = skf.unix_to_skyfield_time(self.telescope.lsd_to_unix(lsd))

            obs = self.telescope.skyfield_obs().at(epoch).observe(positions)
            if self.apparent:
                obs = obs.apparent()

            sra, sdec, _ = obs.cirs_radec(epoch)

            ra_cirs[:, ii] = sra._degrees
            dec_cirs[:, ii] = sdec._degrees

        # If data contains the average visibilities over multiple LSD, then take the
        # corresponding average of the source positions.  Note that if significant
        # flagging was applied to the data prior to averaging, then this should be
        # described in a StackMask that is supplied to this task as data instead of
        # the actual sidereal stack.
        if nlsd > 1:

            weight = weight * tools.invert_no_zero(
                np.sum(weight, axis=0, keepdims=True)
            )

            span = np.max(ra_cirs, axis=-1) - np.min(ra_cirs, axis=-1)
            wraps = np.flatnonzero(span > 180.0)
            if wraps.size > 0:
                ra_cirs[wraps] = tools.correct_phase_wrap(ra_cirs[wraps], deg=True)

            ra_cirs = np.matmul(ra_cirs, weight) % 360.0
            dec_cirs = np.matmul(dec_cirs, weight)

            # If requested transform the source positions from an (nsource, nra)
            # array to an (nsource,) by finding the nearest RA to the sources RA.
            if self.find_nearest:

                imin = np.argmin(
                    np.abs(
                        tools.correct_phase_wrap(ra_cirs - ra[np.newaxis, :], deg=True)
                    ),
                    axis=1,
                )
                nearest = (np.arange(imin.size), imin)

                ra_cirs = ra_cirs[nearest]
                dec_cirs = dec_cirs[nearest]

                out = catalog.copy()

                all_ra_cirs = np.zeros(nsource, dtype=ra_cirs.dtype)
                all_dec_cirs = np.zeros(nsource, dtype=dec_cirs.dtype)

                self.comm.Allgatherv(
                    ra_cirs,
                    [all_ra_cirs, tuple(nsource_per_rank), tuple(ssrc), MPI.DOUBLE],
                )
                self.comm.Allgatherv(
                    dec_cirs,
                    [all_dec_cirs, tuple(nsource_per_rank), tuple(ssrc), MPI.DOUBLE],
                )

                out["position"]["ra"][:] = all_ra_cirs
                out["position"]["dec"][:] = all_dec_cirs

            else:

                out = containers.SourceCatalogVersusRA(
                    ra=ra, axes_from=catalog, attrs_from=catalog, distributed=True
                )
                out.redistribute("object_id")

                out["position"][:].local_array["ra"][:] = ra_cirs
                out["position"][:].local_array["dec"][:] = dec_cirs

        else:
            # The data corresponds to a single day.  Use the source positions
            # calculated on this day.
            out = catalog.copy()

            all_ra_cirs = np.zeros(nsource, dtype=ra_cirs.dtype)
            all_dec_cirs = np.zeros(nsource, dtype=dec_cirs.dtype)

            self.comm.Allgatherv(
                ra_cirs[:, 0],
                [all_ra_cirs, tuple(nsource_per_rank), tuple(ssrc), MPI.DOUBLE],
            )
            self.comm.Allgatherv(
                dec_cirs[:, 0],
                [all_dec_cirs, tuple(nsource_per_rank), tuple(ssrc), MPI.DOUBLE],
            )

            out["position"]["ra"][:] = all_ra_cirs
            out["position"]["dec"][:] = all_dec_cirs

        # Note that we have changed coordinate system by updating the coords attribute
        out["position"].attrs["coords"] = "cirs"

        return out


class MakeTemplateStream(tasklib.base.ContainerTask):
    """An empty sidereal stream on a fixed grid, to simulate a sky model that serves many days.

    The output has the frequencies, inputs, products and stack of a reference day,
    zero visibilities on a regular grid of ``nra`` samples of the sidereal day
    (``ra`` from 0 to 360 degrees), and weights that do not depend on time: for
    each frequency and stacked baseline, the median of the reference day's non-zero
    weights, or zero if fewer than ``min_fraction`` of its samples have weight (the
    channels and baselines masked for most of the day).  So nothing of the day's
    time axis or time-dependent flags survives.  The ``lsd`` attribute sets the
    epoch at which the source positions are converted to CIRS
    (:class:`ConvertICRStoCIRS`, :class:`BaseInject`).

    Attributes
    ----------
    nra : int
        Number of samples of the sidereal day.  Default is 8668 (~10 s).
    lsd : float
        Epoch of the simulation as an LSD.  Default is the reference day's.
    min_fraction : float
        Smallest fraction of the reference day's samples with weight for a
        frequency and baseline to keep it.  Default is 0.5.
    """

    nra = config.Property(proptype=int, default=8668)
    lsd = config.Property(proptype=float, default=None)
    min_fraction = config.Property(proptype=float, default=0.5)

    def process(self, data):
        """Make the stream.

        Parameters
        ----------
        data : TimeStream or SiderealStream
            The reference day (masked as for the search).

        Returns
        -------
        out : SiderealStream
        """
        data.redistribute("freq")
        out = containers.SiderealStream(
            ra=self.nra,
            axes_from=data,
            attrs_from=data,
            distributed=True,
            comm=data.comm,
        )
        out.redistribute("freq")
        out.vis[:] = 0.0

        win = data.weight[:].local_array  # freq, stack, time
        wout = out.weight[:].local_array
        nkeep = 0
        for ff in range(win.shape[0]):
            w = np.asarray(win[ff], dtype=np.float64)
            pos = w > 0.0
            frac = pos.mean(axis=-1)
            med = np.zeros(w.shape[0])
            ok = frac >= self.min_fraction
            if ok.any():
                med[ok] = [np.median(w[ii][pos[ii]]) for ii in np.flatnonzero(ok)]
            wout[ff] = med[:, np.newaxis]
            nkeep += int(ok.sum())
        nkeep = self.comm.allreduce(nkeep)
        ntot = self.comm.allreduce(int(np.prod(win.shape[:2])))
        self.log.info(
            f"{nkeep} of {ntot} frequencies and baselines have weight on at least "
            f"{self.min_fraction:.0%} of the reference day."
        )

        if self.lsd is not None:
            out.attrs["lsd"] = self.lsd
        elif "lsd" not in out.attrs:
            raise ValueError("The reference day has no lsd attribute; set lsd.")
        out.attrs["reference_lsd"] = data.attrs.get("lsd", np.nan)
        out.attrs["tag"] = f"template_lsd_{float(np.mean(out.attrs['lsd'])):g}"
        return out


class CreateFluxCatalogs(tasklib.base.ContainerTask):
    """Package sources in a FluxCatalog collection into a container.

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
    files : str
        Name of the files containing the flux catalogs.  Must be .json
        files that can be loaded with ch_util.fluxcat.FluxCatalog.load.
    """

    min_dec = config.Property(proptype=float, default=-90.0)
    max_dec = config.Property(proptype=float, default=90.0)
    min_flux = config.Property(proptype=float, default=None)
    max_flux = config.Property(proptype=float, default=None)
    exclude = config.Property(proptype=list)
    files = config.Property(proptype=tasklib.io.list_or_glob)

    NOMINAL_FREQ = 600.0

    def process(self):
        """Load the next flux catalog and package into SourceCatalog container."""
        # Exit this task if we have created all of the requested catalogs
        if len(self.files) == 0:
            raise exceptions.PipelineStopIteration

        filename = self.files.pop(0)

        # Delete all previously loaded catalogs
        collections = fluxcat.FluxCatalog.loaded_collections()
        for collection in collections:
            fluxcat.FluxCatalog.delete_loaded_collection(collection[0])

        # Load the flux catalog
        fluxcat.FluxCatalog.load(filename, overwrite=2)

        catalog_tag = fluxcat.FluxCatalog.loaded_collections()[0][0]

        # Extract source names and positions
        sources = np.array(fluxcat.FluxCatalog.keys())

        sra = np.array([fluxcat.FluxCatalog[src].ra for src in sources])
        sdec = np.array([fluxcat.FluxCatalog[src].dec for src in sources])

        if self.exclude is not None:
            not_excluded = np.array([src not in self.exclude for src in sources])
        else:
            not_excluded = np.ones(sources.size, dtype=bool)

        # Filter based on flux and declination
        nominal_flux = np.array(
            [
                fluxcat.FluxCatalog[src].predict_flux(self.NOMINAL_FREQ)
                for src in sources
            ]
        )
        min_flux = self.min_flux if self.min_flux is not None else 0.0
        max_flux = self.max_flux if self.max_flux is not None else np.inf

        ikeep = np.flatnonzero(
            not_excluded
            & (sdec >= self.min_dec)
            & (sdec <= self.max_dec)
            & (nominal_flux >= min_flux)
            & (nominal_flux <= max_flux)
        )

        self.log.info(
            f"There are {ikeep.size} sources between declination "
            f"[{self.min_dec:0.2f}, {self.max_dec:0.2f}] with "
            f"{min_flux:0.3f} < S(600) < {max_flux:0.3f}."
        )

        sources = sources[ikeep]

        # Create output container
        catalog = containers.SourceCatalog(object_id=sources)

        catalog["position"]["ra"][:] = sra[ikeep]
        catalog["position"]["dec"][:] = sdec[ikeep]
        catalog["position"].attrs["coords"] = "icrs"

        # Save attributes describing the source selection to the output container
        catalog.attrs["tag"] = catalog_tag
        catalog.attrs["catalog"] = catalog_tag
        catalog.attrs["filename"] = filename

        catalog.attrs["min_dec"] = self.min_dec
        catalog.attrs["max_dec"] = self.max_dec
        catalog.attrs["min_flux"] = min_flux
        catalog.attrs["max_flux"] = max_flux

        return catalog


class FluxCatalog(BaseInject):
    """Base class for injecting flux catalogs into visibility data.

    The flux of each source is the model of the fluxcat collection named by the
    catalog (its ``filename`` attribute), or, if the catalog has a ``flux_model``
    table (:class:`ch_pipeline.core.containers.FluxModelCatalog`), the curved
    power law given there.
    """

    def setup(self, manager, catalog):
        """Load the requested flux catalog.

        Parameters
        ----------
        manager : ProductManager, BeamTransfer, or TransitTelescope
            Contains a TransitTelescope object describing the telescope.
        catalog : SourceCatalog or FluxModelCatalog
            Catalog of sources.
        """
        super().setup(manager, catalog)

        # The spectra given with the catalog
        self._flux_model = None
        if "flux_model" in self.catalog:
            fm = self.catalog["flux_model"][:]
            self._flux_model = (
                np.asarray(fm["freq_pivot"], dtype=np.float64),
                np.stack([fm[f"p{k}"] for k in range(6)], axis=1).astype(np.float64),
            )
            self.log.info(f"Using the flux models of {fm.size} sources in the catalog.")
            return

        # Make sure that the FluxCatalog contains the sources in the input catalog
        skip_load = False
        for collection in fluxcat.FluxCatalog.loaded_collections():
            if collection[0] == self.catalog_tag:
                self.log.info(
                    "FluxCatalog already contains the "
                    f"{self.catalog_tag} collection.  Skipping load."
                )
                skip_load = True
            else:
                fluxcat.FluxCatalog.delete_loaded_collection(collection[0])

        if not skip_load:
            filename = self.catalog.attrs["filename"]
            self.log.info(f"Loading {filename} into FluxCatalog.")
            fluxcat.FluxCatalog.load(filename, overwrite=2)

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
        if self._flux_model is not None:
            pivot, param = self._flux_model
            x = np.log(freq["centre"] / pivot[source_index])
            p = param[source_index]
            return p[0] * np.exp(sum(p[k] * x**k for k in range(1, p.size)))

        return fluxcat.FluxCatalog[self.sources[source_index]].predict_flux(
            freq["centre"]
        )


class SpectroscopicCatalog(BaseInject):
    """Inject a spectroscopic catalog into visibility data.

    This task supports two signal models.

    source_model = "fixed"
        Use one global amplitude and scale for every source.  This preserves
        the legacy behaviour used for stacking on spectroscopic catalogs.

    source_model = "catalog"
        Use source-dependent amplitudes and widths from a SpectralLineCatalog:

            catalog["line"]["amplitude"]
            catalog["line"]["width"]

        The amplitude is signed in Jy.  Negative values are absorption.
        The width is an observed-frame scale in MHz.  A width of zero means
        unresolved: inject the amplitude into the single coarse frequency
        channel containing the redshifted line.

    Model for finite-width sources
    ------------------------------
    The finite-width model is the existing exponential profile,

        V(freq, z) = amplitude * exp(-|freq - freq_21cm(z)| / width),

    integrated over each frequency bin within +/- nscale * width.

    Attributes
    ----------
    source_model : {"fixed", "catalog"}
        Whether to use global fixed signal parameters or source-dependent
        signal parameters from the input catalog.

    amplitude : float
        Fixed amplitude of the signal in Jy.  Used when source_model="fixed".

    scale : float
        Fixed scale parameter for the signal in MHz.  Used when
        source_model="fixed".  If zero, the signal is assumed to be unresolved
        in frequency.

    nscale : float
        Inject the finite-width model into all frequency bins within
        +/- nscale * width.
    """

    source_model = config.enum(["fixed", "catalog"], default="fixed")
    spectral_model = config.enum(
        ["channel", "exponential", "gaussian"], default="exponential"
    )

    amplitude = config.Property(proptype=float, default=1.0)
    scale = config.Property(proptype=float, default=0.0)
    nscale = config.Property(proptype=float, default=5.0)

    def setup(self, manager, catalog):
        """Save the spectroscopic catalog that will be used by process.

        Parameters
        ----------
        manager : ProductManager, BeamTransfer, or TransitTelescope
            Contains a TransitTelescope object describing the telescope.
        catalog : SpectroscopicCatalog
            Catalog of sources with position and redshift information.  If
            source_model="catalog", this must also contain a "line" table with
            "amplitude" and "width" columns.
        """
        super().setup(manager, catalog)

        self.output_attrs["source_model"] = self.source_model
        self.output_attrs["spectral_model"] = self.spectral_model
        self.output_attrs["amplitude"] = self.amplitude
        self.output_attrs["scale"] = self.scale
        self.output_attrs["nscale"] = self.nscale

        if self.source_model == "fixed":
            self.log.info(
                f"Using fixed spectroscopic signal model: "
                f"amplitude={self.amplitude:0.3e} Jy, "
                f"scale={self.scale:0.3f} MHz."
            )

        elif self.source_model == "catalog":
            self.log.info(
                "Using source-dependent spectroscopic signal model from "
                "catalog['line']['amplitude'] and catalog['line']['width']."
            )

        else:
            raise ValueError(f"Do not recognize source_model={self.source_model}.")

    def _ampfunc(self, source_index, freq):
        """Return the channel-averaged source spectral profile.

        Parameters
        ----------
        source_index : int
            Index into the processed source catalog.
        freq : np.ndarray[nfreq,] of dtype [("centre"), ("width")]
            Frequency bins in MHz.

        Returns
        -------
        profile : np.ndarray[nfreq,]
            Channel-averaged flux density in Jy.

        Notes
        -----
        For spectral_model="channel", amplitude is assigned directly to the
        channel containing the line centre and width is ignored.

        For spectral_model="exponential", width is the observed-frame exponential
        e-folding scale in MHz.

        For spectral_model="gaussian", width is the observed-frame Gaussian sigma
        in MHz.
        """
        sfreq = self.sfreq[source_index]
        samp = self.samp[source_index]
        swidth = self.swidth[source_index]

        fedge_lower = freq["centre"] - 0.5 * freq["width"]
        fedge_upper = freq["centre"] + 0.5 * freq["width"]
        channel_width = fedge_upper - fedge_lower

        # Unresolved/channel model.  This matches the coarse-channel
        # absorption observable: inject the amplitude into the channel
        # containing the line frequency.
        if swidth == 0.0:
            in_channel = (sfreq > fedge_lower) & (sfreq <= fedge_upper)

            return samp * in_channel

        # Existing exponential model, integrated over each frequency bin.
        dlower = fedge_lower - sfreq
        dupper = fedge_upper - sfreq

        if self.spectral_model == "exponential":

            adlower = np.abs(dlower / swidth)
            adupper = np.abs(dupper / swidth)

            integral = swidth * np.where(
                (dlower < 0.0) & (dupper > 0.0),
                2.0 - np.exp(-adlower) - np.exp(-adupper),
                np.abs(np.exp(-adlower) - np.exp(-adupper)),
            )

        else:  # self.spectral_model == "gaussian"

            arg_lower = dlower / (np.sqrt(2.0) * swidth)
            arg_upper = dupper / (np.sqrt(2.0) * swidth)

            integral = np.sqrt(0.5 * np.pi) * swidth * (erf(arg_upper) - erf(arg_lower))

        return samp * integral / channel_width

    def _process_catalog(self):
        """Process the spectroscopic catalog.

        This method converts the catalog from ICRS to CIRS coordinates.
        It also determines the source frequency, amplitude, width, and local
        frequency bins to process.
        """
        catalog = self.catalog

        # Convert redshift to observed 21 cm frequency.
        if "redshift" not in catalog:
            raise ValueError("Input is missing a required redshift table.")

        z = catalog["redshift"]["z"][:]
        sfreq = NU21 / (z + 1.0)  # MHz
        nsource_total = sfreq.size

        # Get amplitudes and widths.
        if self.source_model == "fixed":
            samp = np.full(nsource_total, self.amplitude, dtype=np.float64)
            swidth = np.full(nsource_total, self.scale, dtype=np.float64)

        elif self.source_model == "catalog":
            if "line" not in catalog:
                raise ValueError(
                    "source_model='catalog' requires a SpectralLineCatalog "
                    "with a 'line' table."
                )

            line = catalog["line"]
            samp = np.asarray(line["amplitude"][:], dtype=np.float64)
            swidth = np.asarray(line["width"][:], dtype=np.float64)

            if samp.size != nsource_total:
                raise ValueError(
                    f"catalog['line']['amplitude'] has length {samp.size}, "
                    f"expected {nsource_total}."
                )

            if swidth.size != nsource_total:
                raise ValueError(
                    f"catalog['line']['width'] has length {swidth.size}, "
                    f"expected {nsource_total}."
                )

        # If the channel spectral model was selected, then reset the width to zero
        if self.spectral_model == "channel":
            swidth = np.zeros_like(swidth)

        elif np.any(swidth < 0.0):
            raise ValueError("Line widths must be non-negative.")

        # Sort by frequency.
        isort = np.argsort(sfreq)

        sfreq = sfreq[isort]
        samp = samp[isort]
        swidth = swidth[isort]

        # Filter for sources that overlap with the range of frequencies
        # processed by this node.
        fedge_lb = self.freq_local["centre"] - 0.5 * self.freq_local["width"]
        fedge_ub = self.freq_local["centre"] + 0.5 * self.freq_local["width"]

        min_freq = np.min(fedge_lb)
        max_freq = np.max(fedge_ub)

        # This is source-dependent.  For unresolved sources, swidth = 0,
        # so sf_lb == sf_ub == sfreq.
        win = self.nscale * swidth

        sf_lb = sfreq - win
        sf_ub = sfreq + win

        keep = np.flatnonzero((sf_ub >= min_freq) & (sf_lb <= max_freq))

        sf_lb = sf_lb[keep]
        sf_ub = sf_ub[keep]

        self.sfreq = sfreq[keep]
        self.samp = samp[keep]
        self.swidth = swidth[keep]
        isort = isort[keep]

        # Identify the frequency bins that have non-zero signal for each source.
        bin_sort = np.argsort(fedge_lb)

        aa = bin_sort[
            np.maximum(
                0,
                np.searchsorted(fedge_lb[bin_sort], sf_lb, side="left") - 1,
            )
        ]
        bb = bin_sort[
            np.maximum(
                0,
                np.searchsorted(fedge_lb[bin_sort], sf_ub, side="left") - 1,
            )
        ]

        # These lines are necessary because the frequency axis may be reversed.
        self.lower_freq_bin = np.minimum(aa, bb)
        self.upper_freq_bin = np.maximum(aa, bb)

        # Convert positions from ICRS to CIRS coordinates.
        if "position" not in catalog:
            raise ValueError("Input is missing a required position table.")

        sra = catalog["position"]["ra"][isort]
        sdec = catalog["position"]["dec"][isort]

        coordinate_system = catalog["position"].attrs.get("coords", "icrs")

        if self.epoch is None:
            self.log.info(
                f"Epoch not available.  Using {coordinate_system} coordinates."
            )

        elif coordinate_system == "icrs":
            date_string = ctime.unix_to_datetime(self.epoch).strftime("%Y%m%dT%H%M%SZ")
            self.log.info(f"Converting from ICRS to CIRS coordinates at {date_string}.")
            sra, sdec = _icrs_to_cirs(sra, sdec, self.epoch, self.telescope)

        elif coordinate_system == "cirs":
            self.log.info("Using the CIRS coordinates provided with the catalog.")

        else:
            raise RuntimeError(
                f"Do not recognize coordinate system {coordinate_system}. "
                "Options are ICRS and CIRS."
            )

        self.sra = np.deg2rad(sra)
        self.sdec = np.deg2rad(sdec)

        self.nsource = len(self.sra)

        self.log.info(
            f"Selected {self.nsource} sources overlapping local frequency range."
        )

    def _get_freq_slice(self, source_index):
        """Look up the frequencies to be processed for each source."""
        return slice(
            self.lower_freq_bin[source_index],
            self.upper_freq_bin[source_index] + 1,
        )


class Beam(BaseInject):
    """Base class for modulating source amplitudes by a primary beam model.

    The beam model is obtained from the beam method of the telescope provided
    during setup.
    """

    def setup(self, manager, catalog):
        """Perform additional setup required for telescope based beam model.

        Parameters
        ----------
        manager : ProductManager, BeamTransfer, or TransitTelescope
            Contains a TransitTelescope object describing the telescope.
        catalog : SourceCatalog or FluxModelCatalog
            Sources to inject.
        """
        super().setup(manager, catalog)
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

        angpos = np.array([(0.5 * np.pi - dec) * np.ones_like(ha), ha]).T

        primary_beam = np.zeros((freq.size, ha.size), dtype=np.float64)

        for ff, freq in enumerate(index_freq):

            bii = self.telescope.beam(self.map_pol_feed[pol[0]], freq, angpos)

            if pol[0] != pol[1]:
                bjj = self.telescope.beam(self.map_pol_feed[pol[1]], freq, angpos)
            else:
                bjj = bii

            primary_beam[ff] = np.sum(bii * bjj.conjugate(), axis=1)

        return primary_beam

    @property
    def apply_beam(self):
        """Modulate the source amplitude by a primary beam model."""
        return True


class BeamExternal(BaseInject):
    """Base class that provides methods for evaluating an external beam model."""

    input_index = config.Property(proptype=int, default=0)

    def setup(self, manager, catalog, beam):
        """Perform the generic setup and initialize the beam.

        Parameters
        ----------
        manager : either ProductManager, BeamTransfer, or TransitTelescope
            Contains a TransitTelescope object describing the telescope.
        catalog : SourceCatalog or FluxModelCatalog
            Sources to inject.
        beam : GridBeam, TrackBeam
            Model for the primary beam.
        """
        super().setup(manager, catalog)
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

        elif isinstance(beam, containers.TrackBeam):
            self._initialize_track_beam(beam)
            self._beamfunc = self._track_beam

        else:
            raise ValueError(f"Do not recognize beam container: {beam.__class__}")

    def _initialize_beam_with_data(self):
        """Check that the beam and visibility data have the same local frequencies."""
        freq = self.freq_local["centre"]

        if not np.array_equal(freq, self._beam_freq):
            raise RuntimeError("Beam and visibility frequency axes do not match.")

    def _initialize_track_beam(self, tbeam):
        """Create a beam model from the coefficients saved in a TrackBeam.

        Parameters
        ----------
        tbeam : containers.TrackBeam
            Coefficients of a linear model for the primary beam,
            which will be evaluated using the beso.beam module.
        """
        # Distribute over frequencies, extract local frequencies
        tbeam.redistribute("freq")

        lo = tbeam.beam.local_offset[0]
        nfreq = tbeam.beam.local_shape[0]
        self._beam_freq = tbeam.freq[lo : lo + nfreq]

        # Find the relevant indices into the polarisation axis
        ipol = np.array([list(tbeam.pol).index(pstr) for pstr in self.process_pol])
        self._beam_pol = [tbeam.pol[ip] for ip in ipol]

        # Extract the model coefficients
        self._beam_coeff = tbeam.beam[:].local_array[:, ipol, self.input_index, :]

        # Create the class used to evaluate the beam model
        BeamModelClass = locate(tbeam.attrs["beam_model_class"])
        beam_model_kwargs = json.loads(tbeam.attrs["beam_model_kwargs"])

        self._beam_model = BeamModelClass(
            self.telescope, comm=None, **beam_model_kwargs
        )

    def _track_beam(self, freq, pol, dec, ha):
        """Evaluate a beam model at the requested declination and hour angles.

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
        pp = self._beam_pol.index(pol)

        nha = ha.size
        rdec = np.full(nha, dec)

        primary_beam = np.zeros((freq.size, nha), dtype=float)

        for ii, nu in enumerate(freq):

            ff = np.argmin(np.abs(self._beam_freq - nu))
            assert np.abs(nu - self._beam_freq[ff]) < 0.1

            self._beam_model.prepare(freq=nu, pol=pol)

            primary_beam[ii] = self._beam_model.evaluate_beam(
                self._beam_coeff[ff, pp, :],
                grid=False,
                ha=ha,
                dec=rdec,
            ).real

        return primary_beam

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
        if gbeam.coords not in ["celestial", "telescope"]:
            raise RuntimeError(
                "GridBeam must be converted to celestial or telescope coordinates for simulation."
            )

        self.coords = gbeam.coords

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
        flag = gbeam.weight[:].local_array[:, :, 0][:, ipol] > 0.0
        beam = np.where(flag, gbeam.beam[:].local_array[:, :, 0][:, ipol].real, 0.0)

        # Convert the declination and hour angle axis to radians, make sure they are sorted
        if self.coords == "celestial":

            phi = (gbeam.phi + 180.0) % 360.0 - 180.0
            isort = np.argsort(phi)
            phi = np.radians(phi[isort])

            theta = np.radians(gbeam.theta)

        else:

            phi = gbeam.phi
            isort = np.argsort(phi)
            phi = phi[isort]

            theta = gbeam.theta

        # Create an interpolator for each frequency and polarisation
        self._beam = [
            [
                scipy.interpolate.RectBivariateSpline(
                    theta, phi, beam[ff, pp][:, isort]
                )
                for pp in range(npol)
            ]
            for ff in range(nfreq)
        ]

        self._beam_flag = [
            [
                scipy.interpolate.RectBivariateSpline(
                    theta, phi, flag[ff, pp][:, isort].astype(np.float32)
                )
                for pp in range(npol)
            ]
            for ff in range(nfreq)
        ]

        self.log.info(f"Grid beam initialized ({self.coords} coordinates).")

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
        if self.coords == "celestial":

            podal = np.abs(ha) <= (0.5 * np.pi)

            phi = np.where(podal, ha, ((ha + 2.0 * np.pi) % (2.0 * np.pi)) - np.pi)
            theta = np.where(podal, dec, np.pi - dec)

        else:
            phi, theta, _ = spherical.sph_to_ground(ha, self.latitude, dec)

        index_freq = np.array([np.argmin(np.abs(nu - self._beam_freq)) for nu in freq])

        pp = self._beam_pol.index(pol)

        primay_beam = np.array(
            [self._beam[ff][pp](theta, phi, grid=False) for ff in index_freq]
        )

        flag = np.array(
            [
                np.abs(self._beam_flag[ff][pp](theta, phi, grid=False) - 1.0) < 0.01
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


def _icrs_to_cirs(ra, dec, epoch, telescope):
    """Convert ICRS coordinates to apparent site-observed CIRS coordinates."""
    source = skf.skyfield_star_from_ra_dec(ra, dec)

    return telescope.object_coords(
        source,
        date=epoch,
        deg=True,
    )


def _get_baseline_info(data, telescope):
    # Determine stack axis
    stack_new, _stack_flag = tools.redefine_stack_index_map(
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
