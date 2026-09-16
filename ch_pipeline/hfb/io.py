"""HFB tasks for reading and writing files."""

import gc
import json
import logging
import os
from pathlib import Path

import caput.astro.time as ctime
import h5py
import numpy as np
from beam_model.formed import FFTFormedActualBeamModel, FFTFormedBeamModel
from caput import config
from caput.containers.tod import concatenate as _concatenate_time
from caput.pipeline import exceptions
from caput.pipeline.tasklib import base, io
from caput.util import mpitools
from ch_ephem.coord import bmxy_to_hadec
from ch_ephem.observers import chime
from ch_util.hfbcat import HFBCatalog
from draco.core.io import get_telescope

from .containers import HFBAbsorberCatalog, HFBData, HFBHighResRingMapStack, HFBReader


class BeamSelectionMixin:
    """Mixin for parsing beam selections, typically from a yaml config.

    Attributes
    ----------
    beam_ew_include : list
        List of East-West beam indices (i.e., in the range 0-3) to include.
        By default all four EW beams are included.
    beam_ns_index : list
        Selection of North-South beam indices (i.e., in the range 0-255) to
        include, given as an explicit list of indices.
    beam_ns_range : list
        Selection of North-South beam indices (i.e., in the range 0-255) to
        include, given as a slice with `[start, stop]` or `[start, stop, step]`
        as the value.

    Notes
    -----
    These attributes will result in an error if they are used together with the
    attribute `freq_phys_list` in :class:`ch_pipeline.hfb.io.BaseLoadFiles`.
    It seems that only one axis can be indexed using "fancy indexing" (i.e.,
    passing an array of indices to access multiple array elements at once).

    Here's an example in the YAML format that the pipeline uses:

    .. code-block:: yaml

        beam_ew_include: [0, 1, 2]      # Excludes EW beam 3
        beam_ns_index: [105, 118, 127]  # A sparse selection (CygA, absorber, Zenith)
        beam_ns_range: [100, 130]       # Will override the selection above
    """

    beam_ew_include = config.Property(proptype=list, default=None)
    beam_ns_index = config.Property(proptype=list, default=None)
    beam_ns_range = config.Property(proptype=list, default=None)

    def resolve_beam_sel(self):
        """Resolve the beam selection.

        Returns
        -------
        beam_sel : np.ndarray
            Array of beam indices to select.
        """
        # Grid of all beam indices, with shape (4, 256) (i.e., EW x NS)
        beam_index_grid = np.arange(1024).reshape(4, 256)

        # Resolve selection of EW beams, creating a column vector to allow
        # correct broadcasting in case `beam_ns_index` is used for NS beams
        if self.beam_ew_include:
            beam_ew_sel = np.array(self.beam_ew_include)[:, np.newaxis]
        else:
            beam_ew_sel = slice(None)

        # Resolve selection of NS beams, with `beam_ns_range` taking
        # precedence over `beam_ns_index`
        if self.beam_ns_range:
            beam_ns_sel = slice(*self.beam_ns_range)
        elif self.beam_ns_index:
            beam_ns_sel = self.beam_ns_index
        else:
            beam_ns_sel = slice(None)

        # Select beam indices from grid
        return beam_index_grid[beam_ew_sel, beam_ns_sel].flatten()


class BaseLoadFiles(BeamSelectionMixin, io.BaseLoadFiles):
    """Base class for loading CHIME HFB data from files on disk into containers.

    Attributes
    ----------
    source_name : str
        Name of source, which should be in `ch_util.hfbcat.HFBCatalog`.
    source_dec : float
        Declination of source in degrees.
    freq_phys_range : list
        Start and stop of physical frequencies (in MHz) to read. The mean is
        used as reference frequency in evaluating beam positions (for selecting
        the beams closest to a transiting source).
    freq_phys_list : list
        List of physical frequencies (in MHz) to read. The first frequency
        in this list is also used in evaluating beam positions (for selecting
        the beams closest to a transiting source). Does not work in combination
        with `beam_ns_range` or `beam_ew_include`.
    freq_phys_delta : float
        Half-width of frequency chuck (in MHz) that is selected around the
        frequency listed in the HFB target list in case a source is provided
        via `source_name`.
        Default is 1.

    Selections
    ----------
    Selections in frequency and beams can be done in multiple ways:
    1. By passing a `source_name` attribute, in which case the HFB target list
       is consulted for the declination of the source (for the beam selection)
       and the frequency of its absorption feature (for the frequency selection,
       together with `freq_phys_delta`).
    2. By passing a `source_dec` attribute (for the beam selection) and/or
       a `freq_phys_range` or `freq_phys_list` attribute (for the frequency
       selection). If both `freq_phys_range` and `freq_phys_list` are given the
       former will take precedence, but you should clearly avoid doing this.
       The `source_dec`, `freq_phys_range`, and `freq_phys_list` attributes
       cancel the look-up of declination and frequency from the HFB target list
       triggered by the `source_name` attribute.
    3. By passing `beam_ew_include` and/or `beam_ns_index` or `beam_ns_range`
       attributes (see documentation in :class:`BeamSelectionsMixin`).
    4. By manually passing indices in the `selections` attribute
       (see documentation in :class:`caput.task.io.SelectionsMixin`).
    Method 1 takes precedence over method 2. If no relevant attributes are
    passed, all frequencies/beams are read.
    """

    source_name = config.Property(proptype=str, default=None)
    source_dec = config.Property(proptype=float, default=None)
    freq_phys_range = config.Property(proptype=list, default=[])
    freq_phys_list = config.Property(proptype=list, default=[])
    freq_phys_delta = config.Property(proptype=float, default=1.0)

    def setup(self, observer=None):
        """Set up observer, and frequency and beam selection.

        Parameters
        ----------
        observer : caput.time.Observer, optional
            Details of the observer, if not set default to CHIME.
        """
        # Set up the default Observer
        self.observer = chime if observer is None else observer

        # Resolve any selections provided through the `selections` attribute
        # (via `caput.task.io.SelectionsMixin`)
        super().setup()

        # Look up source in catalog
        if self.source_name:
            hfb_cat = HFBCatalog[self.source_name]

            # Load source declination, unless manually overridden
            if not self.source_dec:
                self.source_dec = hfb_cat.dec

            # Load frequency(ies) of absorption features, unless a range or list
            # of frequecies to load is provided via the task's attributes
            if (
                hasattr(hfb_cat, "freq_abs")
                and not self.freq_phys_range
                and not self.freq_phys_list
            ):
                nfreq_abs = len(hfb_cat.freq_abs)
                if nfreq_abs == 1:
                    self.freq_phys_range = [
                        hfb_cat.freq_abs[0] - self.freq_phys_delta,
                        hfb_cat.freq_abs[0] + self.freq_phys_delta,
                    ]
                else:
                    raise NotImplementedError(
                        f"Source {hfb_cat.name} has {nfreq_abs} absorption features"
                        "listed in the catalog. Please manually select frequencies"
                        "to load, e.g., using `freq_phys_range` or `freq_phys_list`."
                    )

        # Set up frequency selection.
        cfreq = np.linspace(
            self.observer.freq_start,
            self.observer.freq_end,
            self.observer.num_freq,
            endpoint=False,
        )
        if self.freq_phys_range:
            freq_phys_start = np.max(self.freq_phys_range)
            freq_phys_stop = np.min(self.freq_phys_range)
            freq_index_start = np.argmin(np.abs(cfreq - freq_phys_start))
            freq_index_stop = np.argmin(np.abs(cfreq - freq_phys_stop))
            self.freq_sel = slice(freq_index_start, freq_index_stop)
        elif self.freq_phys_list:
            self.freq_sel = sorted(
                {np.argmin(np.abs(cfreq - freq)) for freq in self.freq_phys_list}
            )
        elif "freq_sel" in self._sel:
            self.freq_sel = self._sel["freq_sel"]
        else:
            self.freq_sel = slice(None)

        # Set up beam selection
        if self.source_dec:
            # NS beam selection from the source's declination, with optional
            # EW beam selection via the `beam_ew_include` attribute
            beam_index_ns = self._find_beam()
            self.beam_sel = slice(beam_index_ns, 1024, 256)
            if self.beam_ew_include:
                self.beam_sel = list(
                    np.arange(1024)[self.beam_sel][self.beam_ew_include]
                )
        elif self.beam_ew_include or self.beam_ns_index or self.beam_ns_range:
            # Beam selection via the `BeamSelectionMixin`
            self.beam_sel = self.resolve_beam_sel()
        elif "beam_sel" in self._sel:
            # Manual beam index selection via the `selections` attribute
            self.beam_sel = self._sel["beam_sel"]
        else:
            self.beam_sel = slice(None)

    def _find_beam(self):
        """Find NS beam number of beam closest to source at transit.

        Returns
        -------
        beam_index_ns : int
            North-south index of beam closest to source at transit.
        """
        # Find source's telescope-y coordinate
        src_y = self.source_dec - chime.latitude

        # Choose beam model
        mdl = FFTFormedActualBeamModel()

        # Grid of beam numbers with EW beam number 1
        beams_ind = np.arange(1000, 1256)

        # Decide frequency (in MHz) at which to evaluate beam positions
        if self.freq_phys_range:
            freq = np.mean(self.freq_phys_range)
        elif self.freq_phys_list:
            freq = self.freq_phys_list[0]
        else:
            freq = 600.0

        # Find beam positions
        beams_xy = mdl.get_beam_positions(beams_ind, freq).squeeze()

        # Find NS beam number of beam closest to calibration source
        return np.abs(beams_xy[:, 1] - src_y).argmin()

    def _load_filelist(self, files, time_range=(None, None)):
        """Load a list of files into the HFBData container.

        Parameters
        ----------
        files : list
            List of filenames to load into container.
        time_range: tuple
            Unix timestamps bracketing the part of the data to be loaded.
        """
        for filename in files:
            if not os.path.exists(filename):
                raise RuntimeError(f"File does not exist: {filename}")

        self.log.info(f"Loading files {files}")
        self.log.debug(f"Reading with time range: {time_range}")
        self.log.debug(f"Reading with freq selections: {self.freq_sel}")
        self.log.debug(f"Reading with beam selections: {self.beam_sel}")

        if len(files) > 1 or time_range != (None, None):
            # Set up the reader
            rd = HFBReader(files)

            # Select time range
            rd.select_time_range(time_range[0], time_range[1])

            # Select frequency range
            rd.freq_sel = self.freq_sel

            # Select beams
            rd.beam_sel = self.beam_sel

            # Read files
            cont = rd.read()
        else:
            kwargs = {}
            if self.freq_sel is not None:
                kwargs["freq_sel"] = self.freq_sel
            if self.beam_sel is not None:
                kwargs["beam_sel"] = self.beam_sel

            cont = HFBData.from_file(files[0], distributed=self.distributed, **kwargs)

        if self.redistribute is not None:
            cont.redistribute(self.redistribute)

        return cont


class LoadFilesFromParams(BaseLoadFiles):
    """Load CHIME HFB data from files given in the task's parameters.

    Attributes
    ----------
    filegroups : list or dict
        A dictionary specifying a file group, or a list of them. In addition to
        the standard components of file groups ('tag' and 'files'; see documentation
        in :class:`caput.task.io`), the file groups can also have a 'time_range',
        given as a list of two unix timestamps. Example YAML content:

    .. code-block:: yaml

        filegroups:
          - tag: '20230108'
            files: ['/mnt/gong/archive/20221221T181623Z_chime_hfb/hfb_01504956_0000.h5',
                    '/mnt/gong/archive/20221221T181623Z_chime_hfb/hfb_01510110_0000.h5']
            time_range: [1673156146.031947, 1673157946.031947]
          - tag: '20230109'
            files: ['/mnt/gong/archive/20221221T181623Z_chime_hfb/hfb_01592573_0000.h5']
            time_range: [1673242310.130873, 1673244110.130873]
    """

    filegroups = config.Property(proptype=io.list_of_filegroups)

    _fgroup_ptr = 0

    def process(self):
        """Load in each filegroup (e.g., a sidereal day).

        Returns
        -------
        ts : HFBData
            The timestream of each filegroup.
        """
        if len(self.filegroups) == self._fgroup_ptr:
            raise exceptions.PipelineStopIteration

        # Collect garbage to remove any prior data objects
        gc.collect()

        # Fetch and remove the first item in the list
        filegroup = self.filegroups[self._fgroup_ptr]
        self._fgroup_ptr += 1

        if "time_range" in filegroup:
            time_range = filegroup["time_range"]
        else:
            time_range = (None, None)

        # Read filegroup
        self.log.info(
            f"Reading filegroup {self._fgroup_ptr} of {len(self.filegroups)}."
        )
        ts = self._load_filelist(filegroup["files"], time_range)

        # Find the time to use to compute the container's LSD
        if time_range and time_range != (None, None):
            # Use middle of time_range, which normally corresponds to the transit time
            container_time = np.mean(time_range)
        else:
            # Use the start time of the container
            container_time = ts.time[0]

        # Compute LSD and add to container attributes.
        lsd = int(self.observer.unix_to_lsd(container_time))
        ts.attrs["lsd"] = lsd

        # Add calendar date in YYYYMMDD format to attributes
        calendar_date = ctime.unix_to_datetime(container_time).strftime("%Y%m%d")
        ts.attrs["calendar_date"] = calendar_date

        # Create tag from LSD, unless manually overridden
        if "tag" in filegroup:
            ts.attrs["tag"] = filegroup["tag"]
        else:
            ts.attrs["tag"] = f"lsd_{lsd:d}"

        # Add list of files (full paths) to container attributes
        ts.attrs["files"] = filegroup["files"]

        # Add source name to container attributes, to allow catalog use
        if self.source_name:
            ts.attrs["source_name"] = self.source_name

        # Return timestream
        return ts


class LoadFiles(LoadFilesFromParams):
    """Load CHIME HFB data from file lists passed into the setup routine.

    Attributes
    ----------
    single_group : bool
        If this task receives a single list of files should they be considered
        as forming a single group (True), or is each file its own group (False,
        default).
    """

    single_group = config.Property(proptype=bool, default=False)

    filelists = None

    def setup(self, filelists):
        """Parse the file lists and set up frequency and beam selection.

        Parameters
        ----------
        filelists : list
            A specification of the set of files to load and how they should be
            grouped. Entries in the list must be a homogeneous set of:

            - Lists of filenames. Each of these forms a filegroup where the
              member files will be loaded into and returned in a single
              container.
            - 2-tuples. The first entry is a list of filenames that forms
              the filegroup, the second entry gives the time range of data in
              the group to read (given as float UTC Unix seconds).
            - String filenames/Path objects. Depending on the `single_group`
              config option this will either interpret the parent list as a
              single filegroup incorporating all entries, or as each entry
              forming its own filegroup.
        """
        if not isinstance(filelists, list):
            raise RuntimeError("Argument must be a list.")

        # If we just get a single list of files then convert into a single
        # group is specified. Otherwise each file will be its own group.
        if filelists and not isinstance(filelists[0], list) and self.single_group:
            filelists = [filelists]

        # Convert list of filelists to list of filegroups
        self.filegroups = []
        for flist in filelists:
            # Handle lists including time ranges
            if isinstance(flist, tuple):
                fgroup = {"files": flist[0], "time_range": flist[1]}
            elif isinstance(flist, list):
                fgroup = {"files": flist, "time_range": (None, None)}
            elif isinstance(flist, str | Path):
                fgroup = {"files": [flist], "time_range": (None, None)}
            else:
                raise ValueError(
                    f"Did not expect to get an object of type {type(flist)}"
                )

            # Avoid adding filegroups with empty filelists (the output of
            # QueryDatabase with return_intervals can include days with no files)
            if fgroup["files"]:
                self.filegroups.append(fgroup)

        # Call the baseclass setup to resolve any selections
        super().setup()


class MakeAbsorberCatalog(base.ContainerTask):
    """Build an HFBAbsorberCatalog from JSON target lists, resolving calibrators.

    Each JSON entry holds 'ra' (degrees), 'dec' (degrees), 'freq' (MHz), and
    optionally 'cal_src' (a calibration source name). Entries with the same
    key across files count as one object. (The first occurrence wins,
    adopting a later entry's 'cal_src' if it has none itself.)

    For each unique (cal_src, coarse channel) pair referenced by an
    absorber, get position from 'combinedps_file', frequency borrowed from that
    absorber (so the same window is extracted downstream), and name
    "<ra><+/-dec>_<cal_src>", e.g. "141+79_3C_220.3".

    Attributes
    ----------
    json_files : list of str
        Paths of the JSON target list files.
    combinedps_file : str, optional
        Point-source table for calibrator positions. Required if any
        entry has a 'cal_src'.
    """

    json_files = config.Property(proptype=list)
    combinedps_file = config.Property(proptype=str, default=None)

    _done = False

    def setup(self, manager):
        """Set up the observer.

        Parameters
        ----------
        manager :
            An Observer object holding the geographic location of the teles>
        """
        self.observer = get_telescope(manager)

    def _load_combinedps(self):
        """Load {name: (ra, dec)} from the combined point-source table."""
        with open(self.combinedps_file) as f:
            header = f.readline().split()
            rows = [line.split() for line in f if line.strip()]

        ira = header.index("RA")
        idec = header.index("DEC")
        i3c = header.index("3CNAME")

        pos = {}
        for row in rows:
            pos.setdefault(row[i3c], (float(row[ira]), float(row[idec])))
        return pos

    def process(self):
        """Read the JSON files, resolve calibrators, return the catalog.

        Returns
        -------
        cat : HFBAbsorberCatalog
        """
        if self._done:
            raise exceptions.PipelineStopIteration
        self._done = True

        # Check we were given at least one file.
        files = [str(f) for f in (self.json_files or [])]
        if not files:
            raise config.CaputConfigError(
                "At least one JSON file must be given via 'json_files'."
            )

        # CHIME coarse channel centres
        cfreq = np.linspace(
            self.observer.freq_start,
            self.observer.freq_end,
            self.observer.num_freq,
            endpoint=False,
        )

        # Read every file in order. Duplicate keys (across files) count as one object
        # The first occurrence wins, but a later occurrence's cal_src is adopted if
        # the first one didn't have one.
        entries = {}  # name -> entry dict, in first-seen order
        nmerged = 0

        for fname in files:
            with open(fname) as f:
                data = json.load(f)

            for name, entry in data.items():
                name = str(name)
                try:
                    ra = float(entry["ra"])
                    dec = float(entry["dec"])
                    freq = float(entry["freq"])
                except (KeyError, TypeError, ValueError) as e:
                    raise RuntimeError(
                        f"Entry {name} in {fname} is missing or has an "
                        f"invalid required key: {e}."
                    ) from e

                cal_src = entry.get("cal_src", None)
                cal_src = str(cal_src) if cal_src else ""

                if name in entries:
                    nmerged += 1
                    kept = entries[name]
                    self.log.info(
                        f"Entry {name} ({fname}) already read from "
                        f"{kept['source_file']}; treating as the same object."
                    )
                    if cal_src and not kept["cal_src"]:
                        kept["cal_src"] = cal_src
                    continue

                channel = int(np.argmin(np.abs(cfreq - freq)))
                entries[name] = {
                    "name": name,
                    "ra": ra,
                    "dec": dec,
                    "freq": freq,
                    "cal_src": cal_src,
                    "channel": channel,
                }

        entry_list = list(entries.values())
        seen = {e["name"] for e in entry_list}

        # Build one calibrator entry per unique (cal_src, coarse channel) pair,
        # at the calibrator's own position but the absorber's frequency, so the same window
        # is extracted downstream.
        cal_entries = []
        if any(e["cal_src"] for e in entry_list):
            if not self.combinedps_file:
                raise config.CaputConfigError(
                    "Entries have 'cal_src' but no 'combinedps_file' was "
                    "given to look up calibrator positions."
                )
            positions = self._load_combinedps()

            cal_seen = set()
            for e in entry_list:
                cs = e["cal_src"]
                if not cs or (cs, e["channel"]) in cal_seen:
                    continue
                cal_seen.add((cs, e["channel"]))

                if cs not in positions:
                    raise RuntimeError(
                        f"Calibration source {cs} (for {e['name']}) not "
                        f"found in {self.combinedps_file}."
                    )
                cra, cdec = positions[cs]

                # Name from the calibrator's own position: e.g. "041+79_3C_220.3".
                base_cname = f"{round(cra):03d}{round(cdec):+d}_{cs}"
                cname, n = base_cname, 1
                while cname in seen:
                    n += 1
                    cname = f"{base_cname}_{n}"
                seen.add(cname)

                cal_entries.append(
                    {"name": cname, "ra": cra, "dec": cdec, "freq": e["freq"]}
                )

        all_entries = entry_list + cal_entries

        # Build the catalog container. validate() raises on any out-of-range ra/dec/freq
        # before it's used downstream.
        names = [e["name"] for e in all_entries]
        ras = [e["ra"] for e in all_entries]
        decs = [e["dec"] for e in all_entries]
        freqs = [e["freq"] for e in all_entries]

        cat = HFBAbsorberCatalog(object_id=np.array(names, dtype="U64"))
        cat["position"]["ra"][:] = np.array(ras)
        cat["position"]["dec"][:] = np.array(decs)
        cat["absorber"]["freq"][:] = np.array(freqs)

        cat.validate()

        self.log.info(
            f"Read {len(files)} files and merged {nmerged} as duplicates; "
            f"{len(entry_list)} unique absorbers and {len(cal_entries)} calibrator entries, "
            f"totaling {len(names)}."
        )

        return cat


class LoadFilesForAbsorbers(BaseLoadFiles):
    """Load HFB data for the absorbers in a catalog.

    Works like LoadFiles but computes the frequency selection as the union
    of the absorbers' frequency windows, loading only those channels, then
    handing back one time-chunk per file group.

    The absorber parameters are stored as arrays in each returned
    container's attributes ('source_names', 'source_ra', 'source_dec',
    'source_freq') and are propagated through the pipeline via
    'attrs_from'.

    Attributes
    ----------
    n_coarse : int
        Coarse channels either side of the one nearest each absorber's
        frequency (i.e. 2 * n_coarse + 1 channels), clipped at the true
        telescope band edges. Default is 6.
    """

    n_coarse = config.Property(proptype=int, default=6)

    _fgroup_ptr = 0

    def setup(self, manager, filelists, catalog):
        """Parse the file groups, take the catalog and set up the observer.

        Parameters
        ----------
        manager :
            An Observer object holding the geographic location of the telescope.
        filelists : list
            A specification of the set of files for the day.
        catalog : HFBAbsorberCatalog
            The catalog of absorbers to load.
        """
        self.observer = get_telescope(manager)

        if not isinstance(filelists, list):
            raise RuntimeError("Argument must be a list.")

        self.filegroups = []
        for flist in filelists:
            if isinstance(flist, tuple):
                fgroup = {"files": flist[0], "time_range": flist[1]}
            elif isinstance(flist, list):
                fgroup = {"files": flist, "time_range": (None, None)}
            elif isinstance(flist, str | Path):
                fgroup = {"files": [flist], "time_range": (None, None)}
            else:
                raise ValueError(
                    f"Did not expect to get an object of type {type(flist)}"
                )
            # Avoid adding filegroups with empty filelists (the output of
            # QueryDatabase with return_intervals can include days with no files)
            if fgroup["files"]:
                self.filegroups.append(fgroup)

        self.log.info(f"Will iterate over {len(self.filegroups)} file groups.")

        # Take the absorber parameters from the catalog container
        if not isinstance(catalog, HFBAbsorberCatalog):
            raise TypeError(f"Expected an HFBAbsorberCatalog, got {type(catalog)}.")
        catalog.validate()

        names = np.array([str(n) for n in catalog.index_map["object_id"]])
        freqs = np.asarray(catalog["absorber"]["freq"][:])
        freq_ax = np.linspace(
            self.observer.freq_start,
            self.observer.freq_end,
            self.observer.num_freq,
            endpoint=False,
        )
        nfreq = len(freq_ax)

        # Per-absorber coarse-channel windows
        self._slices = []
        for f0 in freqs:
            i0 = int(np.argmin(np.abs(freq_ax - f0)))
            self._slices.append(
                slice(max(0, i0 - self.n_coarse), min(nfreq, i0 + self.n_coarse + 1))
            )

        # Union of all absorber channel windows, as a sorted list of indices.
        self.freq_sel = sorted(
            {ch for sl in self._slices for ch in range(sl.start, sl.stop)}
        )

        self.log.info(
            f"Catalog has {len(names)} absorbers, {len(self.freq_sel)} total "
            f"channels to distribute across {mpitools.size} MPI ranks."
        )

        self._source_names = names
        self._source_ra = np.asarray(catalog["position"]["ra"][:])
        self._source_dec = np.asarray(catalog["position"]["dec"][:])
        self._source_freq = freqs

    def process(self):
        """Load the next file group for all absorbers.

        Returns
        -------
        tstream : HFBData
            One time-chunk with all absorbers' frequency channels
            distributed across MPI ranks.
        """
        gc.collect()

        if self._fgroup_ptr >= len(self.filegroups):
            raise exceptions.PipelineStopIteration

        filegroup = self.filegroups[self._fgroup_ptr]
        self._fgroup_ptr += 1

        self.log.info(f"Reading file group {self._fgroup_ptr}/{len(self.filegroups)}.")

        time_range = filegroup.get("time_range", (None, None))

        # Load each file into an HFBData container, applying the union
        # frequency selection, then concatenate along the time axis
        fgroup_containers = []
        for fname in filegroup["files"]:
            cont = HFBData.from_file(
                fname,
                distributed=self.distributed,
                freq_sel=self.freq_sel,
            )
            fgroup_containers.append(cont)

        tstream = (
            _concatenate_time(fgroup_containers)
            if len(fgroup_containers) > 1
            else fgroup_containers[0]
        )

        # Redistribute so channels are evenly spread across MPI ranks
        tstream.redistribute("freq")

        # Find the time to use to compute the container's LSD
        if time_range and time_range != (None, None):
            container_time = float(np.mean(time_range))
        else:
            container_time = 0.5 * (float(tstream.time[0]) + float(tstream.time[-1]))

        lsd = int(self.observer.unix_to_lsd(container_time))
        tstream.attrs["lsd"] = lsd
        tstream.attrs["tag"] = f"lsd_{lsd}"

        tstream.attrs["source_names"] = self._source_names
        tstream.attrs["source_ra"] = self._source_ra
        tstream.attrs["source_dec"] = self._source_dec
        tstream.attrs["source_freq"] = self._source_freq

        return tstream


class UpdateAbsorberStacks(base.ContainerTask):
    """Merge each absorber's cutout from ExtractAbsorberCutouts into its stack file.

    The cutout is written into this day's csd slot of <stack_dir>/<name>.h5, which
    must already exist with matching axes: 'beam_ew', 'ra' and 'freq' must equal
    the stack file's axes exactly; 'el' only has to have the same length.

    This task takes the container ExtractAbsorberCutouts produced ('csd' in its attrs,
    one named group per absorber).

    Attributes
    ----------
    stack_dir : str
        Directory holding the per-absorber stack files.
    """

    stack_dir = config.Property(proptype=str)

    def process(self, cont):
        """Merge this day's cutouts into their stack files.

        Parameters
        ----------
        cont : caput.memdata.MemDiskGroup
            The output of ExtractAbsorberCutouts with 'cont.attrs["csd"]'
        """
        if cont is None:
            self.log.warning("Nothing to merge (no cutouts this day).")
            return

        csd = int(cont.attrs["csd"])
        names = list(cont.keys())

        error = None
        nsaved = 0

        if mpitools.rank0:
            for name in names:
                grp = cont[name]
                axes = {
                    "beam_ew": grp["beam_ew"][:],
                    "el": grp["el"][:],
                    "ra": grp["ra"][:],
                    "freq": grp["freq"][:],
                }
                data = {"hfb": grp["hfb"][:], "weight": grp["weight"][:]}
                try:
                    self._merge_into_stack(name, csd, axes, data)
                    nsaved += 1
                except Exception as e:
                    self.log.exception(f"Absorber {name}: stack update failed: {e}")
                    error = f"{name}: {type(e).__name__}: {e}"
                    break

        # Stop the job cleanly if it failed.
        error = self.comm.bcast(error, root=0)
        if error is not None:
            raise RuntimeError(f"UpdateAbsorberStacks failed on rank 0: {error}")

        self.log.info(f"Merged {nsaved}/{len(names)} cutouts into their stacks.")

    def _merge_into_stack(self, name, csd, axes, data):
        """Write one absorber's cutout into its stack file. Rank 0 only.

        Parameters
        ----------
        name : str
            Absorber name; the stack file is <stack_dir>/<name>.h5.
        csd : int
            Day this cutout is from.
        axes : dict
            {'beam_ew', 'el', 'ra', 'freq'} -> axis values of the cutout.
            beam_ew, ra and freq must equal the stack file's axes while el only has to
            have the same length.
        data : dict
            Dataset name -> array, e.g. {'hfb': ..., 'weight': ...}.
        """
        path = os.path.join(self.stack_dir, f"{name}.h5")

        if not os.path.exists(path):
            raise RuntimeError(
                f"{name}: stack file {path} does not exist. Stack files need to be "
                "created with ch_pipeline.hfb.io.create_absorber_stacks()"
                "before running this task"
            )

        # Open on disk in append mode
        stack = HFBHighResRingMapStack.from_file(
            path, ondisk=True, mode="a", distributed=False
        )

        try:
            for axis, values in axes.items():
                existing = np.asarray(stack.index_map[axis])
                values = np.asarray(values)
                if existing.shape != values.shape:
                    raise RuntimeError(
                        f"{name}: cutout '{axis}' axis has shape {values.shape}, "
                        f"but stack file {path} has {existing.shape}."
                    )
                if axis != "el" and not np.allclose(
                    existing, values, rtol=1e-5, atol=1e-8
                ):
                    raise RuntimeError(
                        f"{name}: cutout '{axis}' axis values differ from stack "
                        f"file {path} (same shape, different values)."
                    )

            csd_index = np.asarray(stack.index_map["csd"])
            matches = np.flatnonzero(csd_index == csd)
            if matches.size == 0:
                raise RuntimeError(
                    f"{name}: csd={csd} is outside the range of stack file "
                    f"{path} ({csd_index.min()}-{csd_index.max()}). Recreate "
                    "the stacks with a wider csd_range."
                )
            ic = int(matches[0])

            for dset, arr in data.items():
                stack[dset][ic] = arr
        finally:
            stack.close()


def create_absorber_stacks(
    json_files,
    stack_dir,
    csd_range,
    combinedps_file=None,
    samples=4280,
    dra_deg=2.0,
    n_coarse=6,
    n_el=2,
    beam_ew=(0, 1, 2),
    freq_start=800.0,
    freq_end=400.0,
    num_freq=1024,
    nsubfreq=128,
    overwrite=False,
    log=None,
):
    """Create empty per-source stack files for UpdateAbsorberStacks to fill.

    Run once, by hand, before running the search pipeline. Writes one
    zero-filled <stack_dir>/<name>.h5 per source, with a 'csd' axis spanning
    'csd_range' and the other axes sized to match what ExtractAbsorberCutouts
    produces for that source.

    Parameters
    ----------
    json_files : list of str
        JSON target lists. Each entry needs 'ra' (deg), 'dec' (deg) and
        'freq' (MHz), and may have a 'cal_src'. Duplicate keys across files
        are the same object.
    stack_dir : str or Path
        Directory to write into. Created if missing.
    csd_range : list
        [start, stop) CSDs each file spans.
    combinedps_file : str, optional
        Point-source table, for looking up 'cal_src' positions.
    samples : int, optional
        RA samples in the sidereal ringmap. Default is 4280.
    dra_deg : float, optional
        RA half-width in degrees. Default is 2.0.
    n_coarse : int, optional
        Coarse channels either side of the source's own. Default is 6.
    n_el : int, optional
        NS beams either side of the one nearest the source. Default is 2.
    beam_ew : array_like, optional
        EW beam indices. Default is (0, 1, 2).
    freq_start : float, optional
        Frequency (MHz) of the top of the band. Default is 800.0.
    freq_end : float, optional
        Frequency (MHz) of the bottom of the band. Default is 400.0.
    num_freq : int, optional
        Number of coarse channels across the band. Default is 1024.
    nsubfreq : int, optional
        HFB sub-frequency bins per coarse channel. Default is 128.
    overwrite : bool, optional
        Replace existing files. Default is False, which skips them.
    log : logging.Logger, optional
        Logger for progress messages. Defaults to this module's logger.

    Returns
    -------
    created : list of str
        Paths written.
    """
    from .analysis import ra_window_pixels

    if log is None:
        log = logging.getLogger(__name__)

    if len(csd_range) != 2 or csd_range[1] <= csd_range[0]:
        raise ValueError(f"csd_range must be increasing, got {csd_range}.")
    csd_array = np.arange(int(csd_range[0]), int(csd_range[1]))

    stack_dir = Path(stack_dir)
    stack_dir.mkdir(parents=True, exist_ok=True)

    cfreq = np.linspace(freq_start, freq_end, num_freq, endpoint=False)
    freq_width = (freq_start - freq_end) / num_freq

    # Read the sources: (name, ra, dec, freq)
    if not json_files:
        raise ValueError("At least one JSON file must be given.")

    entries = {}
    for fname in json_files:
        with open(fname) as f:
            data = json.load(f)

        for name, entry in data.items():
            name = str(name)
            if name in entries:
                continue
            try:
                entries[name] = (
                    float(entry["ra"]),
                    float(entry["dec"]),
                    float(entry["freq"]),
                    str(entry.get("cal_src") or ""),
                )
            except (KeyError, TypeError, ValueError) as e:
                raise RuntimeError(
                    f"Entry {name} in {fname} is missing or has an invalid "
                    f"required key: {e}."
                ) from e

    sources = [(n, ra, dec, freq) for n, (ra, dec, freq, _) in entries.items()]
    seen = {n for n, _, _, _ in sources}

    # One entry per unique (cal_src, coarse channel), at the calibrator's own
    # position but the absorber's frequency, so the same window is extracted.
    if any(cs for _, _, _, cs in entries.values()):
        if not combinedps_file:
            raise ValueError(
                "Entries have 'cal_src' but no 'combinedps_file' was given."
            )

        # Calibrator positions, keyed by 3C name.
        with open(combinedps_file) as f:
            header = f.readline().split()
            rows = [line.split() for line in f if line.strip()]

        ira = header.index("RA")
        idec = header.index("DEC")
        i3c = header.index("3CNAME")

        positions = {}
        for row in rows:
            positions.setdefault(row[i3c], (float(row[ira]), float(row[idec])))

        cal_seen = set()
        for name, (_, _, freq, cs) in entries.items():
            if not cs:
                continue
            channel = int(np.argmin(np.abs(cfreq - freq)))
            if (cs, channel) in cal_seen:
                continue
            cal_seen.add((cs, channel))

            if cs not in positions:
                raise RuntimeError(
                    f"Calibration source {cs} (for {name}) not found in "
                    f"{combinedps_file}."
                )
            cra, cdec = positions[cs]

            cname = f"{round(cra):03d}{round(cdec):+d}_{cs}"
            n = 1
            while cname in seen:
                n += 1
                cname = f"{round(cra):03d}{round(cdec):+d}_{cs}_{n}"
            seen.add(cname)

            sources.append((cname, cra, cdec, freq))

    # Axes shared by every source
    beam_mdl = FFTFormedBeamModel()
    latitude = chime.latitude
    beam_ew = np.asarray(beam_ew)

    nbeam_ns = len(beam_mdl.reference_angles)
    ns_grid = np.arange(nbeam_ns)

    ra_full = np.arange(samples) * (360.0 / samples)
    ra_step = 360.0 / samples

    subfreq_offset = np.linspace(
        freq_width / 2, -freq_width / 2, nsubfreq, endpoint=False
    )

    # Create one file per source
    created = []

    for name, src_ra, src_dec, src_freq in sources:
        path = stack_dir / f"{name}.h5"
        if path.exists() and not overwrite:
            log.info(f"{name}: {path} exists, skipping.")
            continue

        # Freq: nearest coarse channel +/- n_coarse, clipped at the band
        # edges, each expanded to nsubfreq sub-frequencies.
        ic0 = int(np.argmin(np.abs(cfreq - src_freq)))
        channels = np.arange(
            max(0, ic0 - n_coarse), min(cfreq.size, ic0 + n_coarse + 1)
        )
        freq_win = (cfreq[channels][:, np.newaxis] + subfreq_offset).flatten()

        # RA: nearest pixel +/- n_ra, wrapping through 0/360.
        n_ra = ra_window_pixels(src_dec, src_freq, dra_deg, ra_step, beam_mdl, latitude)
        dra = np.abs((ra_full - src_ra + 180.0) % 360.0 - 180.0)
        ira0 = int(np.argmin(dra))
        rsel = (ira0 + np.arange(-n_ra, n_ra + 1)) % ra_full.size

        # El: the NS beams nearest this source, +/- n_el, clipped at the ends
        # of the NS beam range.
        bmy_all = beam_mdl.get_beam_positions(ns_grid, [src_freq])[:, 0, 1]
        ns0 = int(np.argmin(np.abs(bmy_all - (src_dec - latitude))))
        beam_ns_win = np.arange(max(0, ns0 - n_el), min(nbeam_ns, ns0 + n_el + 1))

        bmy_win = bmy_all[beam_ns_win]
        _, dec_grid = bmxy_to_hadec(np.zeros_like(bmy_win), bmy_win)
        el_win = np.sin(np.radians(dec_grid - latitude))

        # Temp file then rename, so an interrupted run leaves nothing behind.
        tmp_path = f"{path}.tmp.{os.getpid()}"

        try:
            shape = (
                csd_array.size,
                beam_ew.size,
                el_win.size,
                rsel.size,
                freq_win.size,
            )

            with h5py.File(tmp_path, "w") as fh:
                imap = fh.create_group("index_map")
                imap.create_dataset("csd", data=csd_array)
                imap.create_dataset("beam_ew", data=beam_ew)
                imap.create_dataset("beam_ns", data=beam_ns_win)
                imap.create_dataset("el", data=el_win)
                imap.create_dataset("ra", data=ra_full[rsel])
                imap.create_dataset("freq", data=freq_win)

                for dname, dspec in HFBHighResRingMapStack._dataset_spec.items():
                    chunks = dspec.get("chunks")
                    if chunks is not None:
                        # Clamp: HDF5 rejects a chunk larger than the dataset.
                        chunks = tuple(
                            min(c, s) for c, s in zip(chunks, shape, strict=True)
                        )

                    dset = fh.create_dataset(
                        dname,
                        shape=shape,
                        dtype=dspec["dtype"],
                        chunks=chunks,
                        compression=dspec.get("compression"),
                        compression_opts=dspec.get("compression_opts"),
                    )
                    dset.attrs["axis"] = np.array(dspec["axes"], dtype="S")

                fh.attrs["object_id"] = name

            os.replace(tmp_path, path)
        finally:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)

        log.info(
            f"{name}: created {path} (csd={csd_array.size}, "
            f"el={el_win.size}, ra={rsel.size}, freq={freq_win.size})."
        )
        created.append(str(path))

    log.info(f"Created {len(created)}/{len(sources)} stack files.")

    return created
