"""Tasks for selecting subsets of HFB data.

Tasks
=====
- :py:class:`SelectBeam`
- :py:class:`SelectBeamsAroundSources`
- :py:class:`SelectFreqAroundSources`
- :py:class:`SelectTransit`
- :py:class:`ExtractAbsorberCutouts`
"""

import beam_model.formed as fm
import numpy as np
from beam_model.composite import FutureMostAccurateCompositeBeamModel
from caput import config
from caput.containers import copy_datasets_filter, empty_like
from caput.memdata import MemDiskGroup
from caput.pipeline import tasklib
from ch_ephem.coord import bmxy_to_hadec
from ch_util.hfbcat import HFBCatalog
from draco.core.io import get_telescope
from mpi4py import MPI

from . import containers
from .io import BeamSelectionMixin
from .util import ra_window_pixels


class SelectBeam(BeamSelectionMixin, tasklib.base.ContainerTask):
    """Select a subset of EW and/or NS beams from a container.

    The selection is made by passing `beam_ew_include` and/or `beam_ns_index` or
    `beam_ns_range` attributes (see documentation in :class:`BeamSelectionsMixin`).
    """

    def setup(self):
        """Resolve the beam selection."""
        # Resolve the selections provided through the `beam_ew_include`,
        # `beam_ns_index`, and `beam_ns_range` attributes (via
        # `ch_pipeline.hfb.io.BeamSelectionsMixin`)
        self.beam_sel = self.resolve_beam_sel()

    def process(self, stream):
        """Select a subset of beams.

        Parameters
        ----------
        stream : containers.HFBBeamContainer
            An HFB container with a beam axis.

        Returns
        -------
        newstream : containers.HFBBeamContainer
            New container with a selection of beams.
        """
        # Create new container with subset of beams
        newstream = empty_like(stream, beam=self.beam_sel)

        # Make sure all datasets are initialised
        for name in stream.datasets.keys():
            if name not in newstream.datasets:
                newstream.add_dataset(name)

        # Find indices in current beam axis of selected subset of beams
        selindex = np.flatnonzero(np.isin(stream.beam, self.beam_sel)).tolist()

        # Copy over datasets
        copy_datasets_filter(stream, newstream, "beam", selindex)

        return newstream


class SelectBeamsAroundSources(tasklib.base.ContainerTask):
    """Select the beams around the sources recorded in the attributes.

    The source declination and frequency are read from the container
    attributes ('source_dec', 'source_freq'), which are set by
    LoadFilesFromCatalog.

    Attributes
    ----------
    n_beams_ns : int
        Number of NS beams to include on either side of the beam closest
        to each source (i.e. 2 * n_beams_ns + 1 NS beams per source).
        Default is 2.
    beam_ew_include : list
        List of East-West beam indices to include. Default is [0, 1, 2].
    """

    n_beams_ns = config.Property(proptype=int, default=2)
    beam_ew_include = config.Property(proptype=list, default=[0, 1, 2])

    def setup(self, manager, catalog):
        """Load the beam model used to locate the sources' NS beams.

        Parameters
        ----------
        manager : caput.astro.observer.Observer
            An Observer object holding the geographic location of the telescope.
        catalog : containers.HFBAbsorberCatalog
            Catalog of sources whose beams are to be selected.
        """
        self.beam_mdl = fm.FFTFormedActualBeamModel()
        self.observer = get_telescope(manager)
        self.latitude = self.observer.latitude

        # Number of NS beams from the beam model
        self.nbeam_ns = len(self.beam_mdl.reference_angles)

        if not isinstance(catalog, containers.HFBAbsorberCatalog):
            raise TypeError(f"Expected an HFBAbsorberCatalog, got {type(catalog)}.")
        catalog.validate()

        self.src_dec = np.asarray(catalog["position"]["dec"][:])
        self.src_freq = np.asarray(catalog["absorber"]["freq"][:])

    def process(self, stream):
        """Select the beams around the catalog's sources.

        Parameters
        ----------
        stream : containers.HFBData
            Container with a beam axis and 'source_dec' / 'source_freq'
            attributes from 'LoadFilesFromCatalog'

        Returns
        -------
        newstream : containers.HFBData
            New container holding only the selected beams.
        """
        # Union of the NS-beam neighbourhoods of all absorbers in the catalog.
        ns_beams: set = set()
        for dec, freq in zip(self.src_dec, self.src_freq, strict=True):
            angles = self.beam_mdl.get_beam_positions(
                np.arange(0, self.nbeam_ns), [freq]
            )
            bmy = angles[:, 0, 1]
            _, dg = bmxy_to_hadec(np.zeros_like(bmy), bmy)
            ns0 = int(np.argmin(np.abs(dg - dec)))
            ns_beams.update(
                range(
                    max(0, ns0 - self.n_beams_ns),
                    min(self.nbeam_ns, ns0 + self.n_beams_ns + 1),
                )
            )

        ns_sel = np.array(sorted(ns_beams))
        ew_sel = np.array(self.beam_ew_include)

        # Full beam indices, restricted to beams present in the stream
        beam_sel = (ew_sel[:, np.newaxis] * self.nbeam_ns + ns_sel).flatten()
        beam_sel = np.sort(beam_sel[np.isin(beam_sel, stream.beam)])

        self.log.info(
            f"Selecting {beam_sel.size} beams for {self.src_dec.size} absorbers "
            f"({ns_sel.size} NS beams, EW {ew_sel.tolist()})."
        )

        # Create new container with the subset of beams
        newstream = empty_like(stream, beam=beam_sel)

        # Make sure all datasets are initialised
        for dname in stream.datasets.keys():
            if dname not in newstream.datasets:
                newstream.add_dataset(dname)

        # Find indices in current beam axis of selected subset of beams
        selindex = np.flatnonzero(np.isin(stream.beam, beam_sel)).tolist()

        # Copy over datasets
        copy_datasets_filter(stream, newstream, "beam", selindex)

        return newstream


class SelectFreqAroundSources(tasklib.base.ContainerTask):
    """Keep only the coarse channels around each source in a catalog.

    The catalog-driven counterpart of 'draco.core.task.Downselect' for the
    frequency axis: 'Downselect' takes fixed indices from the config, while
    the channels wanted here depend on the sources.

    Attributes
    ----------
    n_coarse : int
        Coarse channels either side of the one nearest each source's
        frequency (i.e. 2 * n_coarse + 1 per source). Default is 6.
    """

    n_coarse = config.Property(proptype=int, default=6)

    def setup(self, catalog):
        """Take the source frequencies from the catalog.

        Parameters
        ----------
        catalog : containers.HFBAbsorberCatalog
            Catalog of sources whose channels are to be kept.
        """
        if not isinstance(catalog, containers.HFBAbsorberCatalog):
            raise TypeError(f"Expected an HFBAbsorberCatalog, got {type(catalog)}.")
        catalog.validate()

        self.src_freq = np.asarray(catalog["absorber"]["freq"][:])

    def process(self, stream):
        """Cut the freq axis down to the union of the sources' windows.

        Parameters
        ----------
        stream : containers.HFBData
            Container with a freq axis.

        Returns
        -------
        newstream : containers.HFBData
            New container holding only the selected channels.
        """
        cfreq = stream.index_map["freq"]["centre"]
        nfreq = len(cfreq)

        sel = set()
        for f0 in self.src_freq:
            i0 = int(np.argmin(np.abs(cfreq - f0)))
            sel.update(
                range(max(0, i0 - self.n_coarse), min(nfreq, i0 + self.n_coarse + 1))
            )
        fsel = sorted(sel)

        self.log.info(
            f"Keeping {len(fsel)}/{nfreq} coarse channels for "
            f"{self.src_freq.size} sources."
        )

        # freq is the distributed axis, so move the distribution elsewhere
        # before cutting it.
        stream.redistribute("time")

        newstream = stream.__class__(
            axes_from=stream,
            attrs_from=stream,
            skip_datasets=True,
            freq=stream.index_map["freq"][fsel],
        )
        copy_datasets_filter(stream, newstream, selection={"freq": fsel})

        newstream.redistribute("freq")

        return newstream


class SelectTransit(tasklib.base.ContainerTask):
    """Find transit data through FRB beams.

    Task for selecting on- and off-source data. The task can do both, but only
    one at a time (which one is set by the `selection` attribute). The selection
    of data is determined by the following criteria (a single one or both can be
    used, as specified by the `criteria` attribute):

    1) The offset in RA of the time samples with respect to the nominal centre of
       the synthetic beam being considered. This results in a window of data that
       is includes (in the case of on-source selection) or excluded (in the case
       of off-source selection). The size of the window is measured in units of
       the half-width of the synthetic beams in the EW direction, and set by the
       `on_source_include_window` and `off_source_exclude_window` attributes.
    2) The sensitivity of the synthetic beam toward the source. This must be greater
       than some threshold (in the case of on-source selection) or lower than some
       ceiling (in the case of off-source selection). The threshold/ceiling is set
       by the `on_source_sens_threshold` and `off_source_sens_ceiling` attributes,
       as a fraction of the maximum sensitivity in the beam's main lobe. (The
       sensitivity of the last row of EW beams, EW index 3, is higher in the side
       lobes than the main lobe. If we were to use the maximum sensitivity of the
       beam per se to calculate the threshold needed for data selection, selected
       data won't be in the main lobe.)

    Attributes
    ----------
    selection : str
        What part of the data to select. Options are: 'on-source', 'off-source'.
        Default is 'on-source'
    criteria : list of strings
        Criteria used to decide which time samples are selected. Possible entries:
        'offset' (i.e., the RA offset of the source to the centre of the beam),
        'sensitivity' (i.e., the synthetic beam sensitivity).
        Default is ['offset', 'sensitivity']
    on_source_include_window : float
        Half-width of the window to include when selecting on-source data in units
        of the EW half-width of the synthetic beams (beam and frequency dependent).
        Default is 1.0
    on_source_sens_threshold : float
        Fraction of maximum beam sensitivity above which samples are considered
        on-source.
        Default is 0.6
    off_source_exclude_window : float
        Half-width of the window to exclude when selecting off-source data in units
        of the EW half-width of the synthetic beams (beam and frequency dependent).
        A value of 10.0 excludes the first main sidelobe of the synthetic beams.
        A value of 18.0 excludes a further minor sidelobe, at the cost of requiring
        larger chunks of data around transit, especially at lower frequecies.
        Default is 18.0
    off_source_sens_ceiling : float
        Fraction of maximum beam sensitivity below which samples are considered
        off-source.
        Default is 0.005
    source_name : str
        Name of source, which should be in `ch_util.hfbcat.HFBCatalog`. If this is
        not provided, the task will look for it in the input container attributes.
    source_ra : float
        Right ascension of the source in degrees, in case this is not available in
        `ch_util.hfbcat.HFBCatalog` (or in case the value given there needs to be
        overridden manually).
    source_dec : float
        Declination of the source in degrees, in case this is not available in
        `ch_util.hfbcat.HFBCatalog` (or in case the value given there needs to be
        overridden manually).
    time_offset : float
        Time (is seconds) to add to the data times before comparison with the
        beam model. This is a temporary hack, to be removed when the data times
        are fully understood.
        Default is 0.0
    """

    selection = config.enum(["on-source", "off-source"], default="on-source")
    criteria = config.list_type(type_=str, default=["offset"])
    on_source_include_window = config.Property(proptype=float, default=1.0)
    on_source_sens_threshold = config.Property(proptype=float, default=0.6)
    off_source_exclude_window = config.Property(proptype=float, default=18.0)
    off_source_sens_ceiling = config.Property(proptype=float, default=0.005)
    source_name = config.Property(proptype=str, default=None)
    source_ra = config.Property(proptype=float, default=None)
    source_dec = config.Property(proptype=float, default=None)
    time_offset = config.Property(proptype=float, default=0.0)

    def setup(self):
        """Check criteria attribute."""
        if not self.criteria:
            raise ValueError("No selection criteria set.")

        available_criteria = ["offset", "sensitivity"]
        for criterion in self.criteria:
            if criterion not in available_criteria:
                raise ValueError(
                    f"Unknown selection criterion passed ({criterion})."
                    f"Available criteria are: {available_criteria}"
                )

    def process(self, stream):
        """Find transit data for the given beams.

        Parameters
        ----------
        stream : containers.HFBData
            Container with HFB data and weights.

        Returns
        -------
        out : containers.HFBData
            Container with same HFB data, but weights adjusted to make selection.
        """
        # On the first pass, obtain the source's coordinates from the HFB catalog,
        # unless they are provided manually via the task's source_ra / source_dec
        # attributes. To do this, get the source name from the container attributes,
        # unless manually overridden via the task's source_name attribute.
        if not self.source_ra or not self.source_dec:
            if not self.source_name:
                self.source_name = stream.attrs["source_name"]
            hfb_cat = HFBCatalog[self.source_name]
            if not self.source_ra:
                self.source_ra = hfb_cat.ra
            if not self.source_dec:
                self.source_dec = hfb_cat.dec

        # Extract axes lengths
        nfreq, nsubfreq, nbeam, ntime = stream.hfb.shape

        # Extract physical frequencies by combining CHIME frequency channels and
        # HFB subfrequencies into one vector
        freq = stream._data["index_map"]["freq"]["centre"][:]
        subfreq = stream._data["index_map"]["subfreq"]
        phys_freq = (freq.reshape(nfreq, 1) + subfreq).flatten()

        # Extract beam indices, then change format for sensitivity calculations.
        # The beam_model package uses a format (referred to here as `beam_number`)
        # where the EW beam index is indicated by thousands, and the NS beam index
        # by the remaining digits (a number ranging form 0 and 255). For example,
        # the indices [12, 268, 524, 780] will be changed to [12, 1012, 2012, 3012]
        beam_index = stream._data["index_map"]["beam"][:]
        dtype = beam_index.dtype
        nbeam_ns = 256
        beam_number = (
            beam_index % nbeam_ns + 1000 * np.floor(beam_index / nbeam_ns)
        ).astype(dtype)

        # Extract time array. HACK: Possibly add offset to time.
        time = stream.time[:] + self.time_offset

        # Load beam model used for sensitivity calculations, for looking up beam
        # positions and beam widths, and for finding source positions in beam-model
        # xy-coordinates. Set `interpolate_bad_freq=True` to avoid issues at
        # frequencies where the data-driven primary-beam model lacks data.
        beam_mdl = FutureMostAccurateCompositeBeamModel(interpolate_bad_freq=True)

        # Obtain the track of the source in beam-model xy-coordinates from its
        # equatorial position and time. The array has shape (ntime, 2).
        posxy_source = np.zeros((ntime, 2))
        for itime, time_sample in enumerate(time):
            posxy_source[itime, :] = beam_mdl.get_position_from_equatorial(
                self.source_ra, self.source_dec, time_sample
            )

        # Calculate the centre positions of the synthetic beams and their widths
        # (FWHM in the EW and NS directions) in beam-model xy-coordinates for all
        # combinations of beam and frequency. The shape of both resulting arrays
        # is (nbeam, nfreq, 2), with the final dimension giving x and y.
        posxy_beams = beam_mdl.get_beam_positions(beam_number, phys_freq)
        beam_width = beam_mdl.get_beam_widths(beam_number, phys_freq)

        # Compute the offset between the source position and the centres of the
        # synthetic beams in terms of beam model x coordinate. Make sure this
        # array gets axes (time, beam, phys_freq).
        x_offset = np.abs(
            posxy_source[:, 0][:, np.newaxis, np.newaxis] - posxy_beams[:, :, 0]
        )

        # Create a boolean array that selects the main lobe of the synthetic beam.
        # It is defined as time samples that are within `on-source-include-window`
        # synthetic beam EW half-widths of the centre of the beam. The array has axes
        # (time, beam, phys_freq).
        main_lobe = x_offset < self.on_source_include_window * beam_width[:, :, 0] / 2

        # Compute beam sensitivities only if necessary
        if "sensitivity" in self.criteria:
            # Calculate the (composite) beam sesitivities for all combinations of
            # source position (i.e., time), beam, and frequency, resulting in an
            # array of size (ntime, nbeam, nfreq * nsubfreq).
            sens = beam_mdl.get_sensitivity(beam_number, posxy_source, phys_freq)

            # Find the maximum sensitivity within the main lobe of the beam.
            # The resulting array has axes (beam, phys_freq)
            sens_only_main_lobe = sens * main_lobe.astype(float)
            max_sens = sens_only_main_lobe.max(axis=0)

        # Create a boolean mask that selects certain samples of the data:
        # True for selected samples, False for non-selected samples.
        # Initialize the array as all True, then set parts of it to False below
        # based on choosen criteria
        mask = np.full((ntime, nbeam, nfreq * nsubfreq), True)

        if self.selection == "on-source":
            # Make mask to select on-source data

            if "offset" in self.criteria:
                # Ensure that on-source data corresponds to time samples where the
                # source is inside the main lobe of the synthetic beam.
                mask &= main_lobe

            if "sensitivity" in self.criteria:
                # Ensure that on-source data corresponds to time samples where the
                # beam sensitivity is above the choosen threshold (i.e., at least
                # `on_source_sens_threshold` times the maximum sensitivity found
                # in the main lobe of the beam being considered).
                sens_threshold = self.on_source_sens_threshold * max_sens
                mask &= sens > sens_threshold[np.newaxis, :, :]

        elif self.selection == "off-source":
            # Make mask to select off-source data

            if "offset" in self.criteria:
                # Ensure that off-source data corresponds to time samples where
                # the source is far enough away from the synthetic beam (at least
                # `off_source_exclude_window` synthetic beam half-widths away from
                # the beam centre).
                x_offset_lim = self.off_source_exclude_window * beam_width[:, :, 0] / 2
                mask &= x_offset > x_offset_lim

            if "sensitivity" in self.criteria:
                # Ensure that off-source data corresponds to time samples where
                # the beam sensitivity is below the choosen threshold (at most
                # `off_source_sens_ceiling` times the maximum sensitivity found
                # in the main lobe of the beam being considered).
                sens_threshold = self.off_source_sens_ceiling * max_sens
                mask &= sens < sens_threshold[np.newaxis, :, :]

        # The mask created above has axes (time, beam, phys_freq). Convert this
        # back to the axes used in the hfb containers (freq, subfreq, beam, time):
        # First swap axes to get (phys_freq, beam, time). Then reshape the array
        # to convert phys_freq to freq and subfreq.
        mask = np.swapaxes(mask, 0, 2)
        mask = mask.reshape(nfreq, nsubfreq, nbeam, ntime)

        self.log.info(
            f"Average number of {self.selection} time samples selected: "
            f"{mask.sum(axis=3).mean():.1f}"
        )

        # Create container to hold output
        out = containers.HFBData(copy_from=stream)

        # Set weights of non-selected samples to zero
        weight = stream.weight[:] * mask.astype(float)
        out.weight[:] = weight

        # Return output container
        return out


class ExtractAbsorberCutouts(tasklib.base.ContainerTask):
    """Extract each absorber's cutout from a day's ringmap, MPI-combined.

    The RA window is 'dra_deg' physical degrees (converted to a per-absorber
    pixel count via 'ra_window_pixels'), wrapping correctly around
    RA = 0/360. The el window is a fixed number of pixels, clipped at the edges
    of the el axis. The freq window is 'n_coarse' whole coarse channels,
    clipped at the edges of the loaded freq axis.

    Attributes
    ----------
    dra_deg : float
        RA half-width in degrees at the best-resolution declination.
        Default is 2.0.
    n_coarse : int
        Coarse channels either side of the one nearest the absorber's
        frequency (i.e. 2 * n_coarse + 1 channels, each expanded to
        high-res sub-frequencies). Default is 6.
    n_el : int
        El pixels (beams) to include on either side of the pixel nearest
        the absorber's el (i.e. 2 * n_el + 1 pixels total, clipped at the
        axis edges). Default is 2.
    """

    dra_deg = config.Property(proptype=float, default=2.0)
    n_coarse = config.Property(proptype=int, default=6)
    n_el = config.Property(proptype=int, default=2)

    def setup(self, manager, catalog):
        """Set the local observer's position and load the beam model.

        Parameters
        ----------
        manager : caput.astro.observer.Observer
            An Observer object holding the geographic location of the telescope.
        catalog : containers.HFBAbsorberCatalog
            Catalog of sources to cut out.
        """
        self.observer = get_telescope(manager)
        self.latitude = self.observer.latitude
        self._beam_mdl = fm.FFTFormedActualBeamModel()

        if not isinstance(catalog, containers.HFBAbsorberCatalog):
            raise TypeError(f"Expected an HFBAbsorberCatalog, got {type(catalog)}.")
        catalog.validate()

        self._names = [str(n) for n in catalog.index_map["object_id"]]
        self._src_ra = np.asarray(catalog["position"]["ra"][:]).astype(float)
        self._src_dec = np.asarray(catalog["position"]["dec"][:]).astype(float)
        self._src_freq = np.asarray(catalog["absorber"]["freq"][:]).astype(float)

    def process(self, stream):
        """Extract and MPI-combine one cutout per absorber in the catalog.

        Parameters
        ----------
        stream : containers.HFBHighResRingMap
            High frequency resolution ringmap container.

        Returns
        -------
        cont : caput.memdata.MemDiskGroup
            One group per absorber (cont[name]), each holding
            beam_ew / el / ra / freq / hfb / weight datasets and a
            'centre_freq' attribute, plus cont.attrs["csd"].
        """
        # Ensure only the freq axis is distributed. After this,
        # beam_ew, el and ra are complete on every rank.
        stream.redistribute("freq")
        rank0 = self.comm.rank == 0

        # Source list from the catalog taken in setup.
        names = self._names
        src_ra = self._src_ra
        src_dec = self._src_dec
        src_freq = self._src_freq

        lsd = stream.attrs.get("lsd", stream.attrs.get("csd", None))
        if lsd is None:
            if rank0:
                self.log.warning("No lsd/csd in stream attrs; nothing extracted.")
            return None
        lsd = np.atleast_1d(lsd)
        if lsd.size != 1:
            raise ValueError(f"Stream holds several days ({lsd}); expected one.")
        csd = int(lsd[0])

        # Axes of the ringmap as plain numpy arrays.
        freq = np.asarray(stream.index_map["freq"])
        ra = np.asarray(stream.index_map["ra"])
        el = np.asarray(stream.index_map["el"])
        beam_ew = np.asarray(stream.index_map["beam_ew"])
        nra = ra.size
        nel = el.size
        ra_step = float(np.median(np.diff(ra)))

        # Number of HFB sub-frequencies per coarse channel.
        if "nsubfreq" not in stream.attrs:
            raise KeyError(
                "Stream has no 'nsubfreq' attribute, so the coarse-channel "
                "structure of the freq axis cannot be recovered. It is set "
                "by MakeHighFreqResRingMap."
            )
        nsubfreq = int(stream.attrs["nsubfreq"])

        # The freq axis is made of whole coarse channels of nsubfreq
        # sub-frequencies each. Get each channel's centre frequency and the
        # channel width, both needed to build the frequency window.
        if freq.size % nsubfreq != 0:
            raise ValueError(
                f"Length of freq axis ({freq.size}) is not a multiple of "
                f"nsubfreq ({nsubfreq}); cannot identify coarse channels."
            )
        block_freq = freq.reshape(-1, nsubfreq)
        block_centre = block_freq.mean(axis=1)
        nblock = block_centre.size

        n_el_win = 2 * self.n_el + 1
        if n_el_win > nel:
            raise RuntimeError(
                f"el window ({n_el_win}) is wider than the el axis ({nel})."
            )

        # Find axis order of the hfb dataset
        axis_names = list(stream.hfb.attrs["axis"])
        faxis = axis_names.index("freq")
        ewaxis = axis_names.index("beam_ew")

        hfb_local = stream.hfb[:].local_array
        weight_local = stream.weight[:].local_array
        freq_offset = stream.hfb[:].local_offset[faxis]
        nfreq_local = hfb_local.shape[faxis]
        nbeam_ew = hfb_local.shape[ewaxis]

        cont = MemDiskGroup()
        cont.attrs["csd"] = csd
        cont.attrs["tag"] = f"lsd_{csd}"
        nextracted = 0

        for iobj, name in enumerate(names):
            # RA window: +/- physical degrees -> per-absorber pixel count,
            # wrapping at 0/360
            n_ra = ra_window_pixels(
                src_dec[iobj],
                src_freq[iobj],
                self.dra_deg,
                ra_step,
                self._beam_mdl,
                self.latitude,
            )
            if 2 * n_ra + 1 > nra:
                raise RuntimeError(
                    f"Source {name}: RA window ({2 * n_ra + 1}) is wider than "
                    f"the RA axis ({nra}). Reduce dra_deg."
                )
            dra = np.abs((ra - src_ra[iobj] + 180.0) % 360.0 - 180.0)
            ira0 = int(np.argmin(dra))
            rsel = (ira0 + np.arange(-n_ra, n_ra + 1)) % nra

            # Freq window: +/- n_coarse whole coarse channels, clipped
            # at the true edges of the loaded axis.
            ib0 = int(np.argmin(np.abs(block_centre - src_freq[iobj])))
            bsel = np.arange(
                max(0, ib0 - self.n_coarse), min(nblock, ib0 + self.n_coarse + 1)
            )
            fsel = (bsel[:, np.newaxis] * nsubfreq + np.arange(nsubfreq)).ravel()

            # El window: +/- n_el pixels, clipped at edges
            bmy_all = self._beam_mdl.get_beam_positions(
                np.arange(len(self._beam_mdl.reference_angles)), [src_freq[iobj]]
            )[:, 0, 1]
            _, dg_all = bmxy_to_hadec(np.zeros_like(bmy_all), bmy_all)
            ns0 = int(np.argmin(np.abs(dg_all - src_dec[iobj])))

            beam_ns = np.asarray(stream.index_map["beam_ns"])
            ie0 = int(np.argmin(np.abs(beam_ns - ns0)))
            esel = np.arange(max(0, ie0 - self.n_el), min(nel, ie0 + self.n_el + 1))

            # Extract this rank's freq chunk
            shape = (nbeam_ew, esel.size, rsel.size, fsel.size)
            data_cut = np.zeros(shape, dtype=hfb_local.dtype)
            weight_cut = np.zeros(shape, dtype=weight_local.dtype)

            # Convert global freq indices to this rank's local indices
            fsel_local = fsel - freq_offset
            valid = (fsel_local >= 0) & (fsel_local < nfreq_local)
            fsel_local = fsel_local[valid]
            fsel_out = np.flatnonzero(valid)

            if fsel_local.size > 0:
                ix = np.ix_(np.arange(nbeam_ew), esel, rsel, fsel_local)
                data_cut[..., fsel_out] = hfb_local[ix]
                weight_cut[..., fsel_out] = weight_local[ix]

            # Combine freq chunks from all ranks
            # (This works for the current cutout sizes, but every rank holds a
            # full copy, so it may not scale. If that becomes a problem,
            # consider a Gatherv to rank 0 instead.)
            self.comm.Allreduce(MPI.IN_PLACE, data_cut, op=MPI.SUM)
            self.comm.Allreduce(MPI.IN_PLACE, weight_cut, op=MPI.SUM)

            # Fill the output container
            cutout_grp = cont.create_group(name)
            cutout_grp.attrs["centre_freq"] = float(src_freq[iobj])
            cutout_grp.create_dataset("beam_ew", data=beam_ew)
            cutout_grp.create_dataset("el", data=el[esel])
            cutout_grp.create_dataset("ra", data=ra[rsel])
            cutout_grp.create_dataset("freq", data=freq[fsel])
            cutout_grp.create_dataset("hfb", data=data_cut)
            cutout_grp.create_dataset("weight", data=weight_cut)
            nextracted += 1

        if rank0:
            self.log.info(f"Extracted {nextracted}/{len(names)} cutouts.")

        return cont
