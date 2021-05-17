"""
Dataset Specification

Lookup information from the database about the data, particularly which files
are contained in a specific dataset (defined by a `run` global flag) and what
each correlator input is connected to.

Dataspec Format
===============

.. deprecated:: pass1
    Use `run` global flag instead.

A dataspec is just a dictionary with three required keys.

:name:
    A short name given to the dataset.
:instrument:
    The name of the instrument that took the data.
:timerange:
    A specification of the time range, or ranges that make up the dataset.

The time range is specified either as a dictionary with `start` and `end`
keys, containing datetime objects (in UTC). Or it can be a list of such
ditionaries, to specify multiple time ranges to include. This can be contained
in a dataspec YAML file, and loaded using :class:`LoadDataspec`. Example:

.. code-block:: yaml

    datasets:
        -   name:       A
            instrument: blanchard
            timerange:
                -   start:  2014-07-26 03:00:00
                    end:    2014-07-28 01:00:00

                -   start:  2014-07-28 11:00:00
                    end:    2014-07-31 00:00:00
"""

import os

from caput import mpiutil, config, pipeline
from draco.core import task
from chimedb import data_index as di
from ch_util import tools, ephemeris, finder, layout


_DEFAULT_NODE_SPOOF = {"cedar_online": "/project/rpp-krs/chime/chime_online/"}


def _force_list(val):
    """Ensure configuration property is a list."""
    if val is None:
        return []
    elif hasattr(val, "__iter__"):
        return val
    else:
        return [val]


class QueryDatabase(task.MPILoggedTask):
    """Find files from specified database queries.

    This routine will query the database as specified in the runtime
    configuration file.

    Attributes
    ----------
    node_spoof : dictionary
        (default: {'cedar_archive': '/project/rpp-krs/chime/chime_archive/'} )
        host and directory in which to find data.
    start_time, end_time : string (default: None)
        start and end times to restrict the database search to
        can be in any format ensure_unix will support, including eg
        20190116T150323 and 2019-1-16 08:03:23 -7
    start_csd, end_csd : float
        Start and end CSDs. Only used if `start_time` is not set.
    instrument : string (default: 'chimestack')
        data set to use
    source_26m : string (default: None)
        holography source to include. If None, do not include holography data.
    exclude_daytime : bool (default: False)
        exclude daytime data
    exclude_sun : bool (default: False)
        exclude data around Sun
    exclude_sun_time_delta : float (default: None)
        time_delta parameter passed to exclude_sun()
    exclude_sun_time_delta_rise_set : float (default: None)
        time_delta_rise_set parameter passed to exclude_sun()
    exclude_transits : list of string or float (default: [])
        if set, call exclude_transits(). Pass list of sources or RA to exclude.
    exclude_transits_time_delta : list of float (default : [])
        time in seconds to exclude around each source transit given in `exclude_transits`.
        if single value is passed, then that value will be applied to all source transits.
    include_transits : list of string or float (default : [])
        if set, call include_transits(). Pass list of sources or RA to include.
    include_transits_time_delta : list of float (default : [])
        time in seconds to include around each source transit given in `include_transits`.
        if single value is passed, then that value will be applied to all source transits.
    start_RA, end_RA : float (default: None)
        starting and ending RA to include. Both values must be included or
        no effect
    run_name : string (default: None)
        run name to include. If used, all other parameters will be ignored.
    accept_all_global_flags : bool (default: False)
        Accept all global flags. Due to a bug as of 2019-1-16, this may need to
        be set to True
    exclude_data_flag_types: list of string
        Reject time intervals that overlap with DataFlags of these types.
    return_intervals : bool (default: False)
        Return the full interval from the Finder. Otherwise only a list of file names.
    """

    return_intervals = config.Property(proptype=bool, default=False)

    node_spoof = config.Property(proptype=dict, default=_DEFAULT_NODE_SPOOF)

    instrument = config.Property(proptype=str, default="chimestack")

    source_26m = config.Property(proptype=str, default=None)

    start_time = config.Property(default=None)
    end_time = config.Property(default=None)

    start_csd = config.Property(proptype=float, default=None)
    end_csd = config.Property(proptype=float, default=None)

    exclude_daytime = config.Property(proptype=bool, default=False)

    exclude_sun = config.Property(proptype=bool, default=False)
    exclude_sun_time_delta = config.Property(proptype=float, default=None)
    exclude_sun_time_delta_rise_set = config.Property(proptype=float, default=None)

    exclude_transits = config.Property(proptype=_force_list, default=[])
    exclude_transits_time_delta = config.Property(proptype=_force_list, default=[])

    include_transits = config.Property(proptype=_force_list, default=[])
    include_transits_time_delta = config.Property(proptype=_force_list, default=[])

    start_RA = config.Property(proptype=float, default=None)
    end_RA = config.Property(proptype=float, default=None)

    run_name = config.Property(proptype=str, default=None)

    accept_all_global_flags = config.Property(proptype=bool, default=False)

    exclude_data_flag_types = config.Property(proptype=list, default=[])

    def setup(self):
        """Query the database and fetch the files

        Returns
        -------
        files : list
            List of files to load
        """
        files = None

        # Query the database on rank=0 only, and broadcast to everywhere else
        if mpiutil.rank0:

            if self.run_name:
                return self.QueryRun()

            layout.connect_database()

            f = finder.Finder(node_spoof=self.node_spoof)

            # should be redundant if an instrument has been specified
            f.only_corr()

            if self.accept_all_global_flags:
                f.accept_all_global_flags()

            # Use start and end times if set, or try and use the start and end CSDs
            if self.start_time:
                st, et = self.start_time, self.end_time
            elif self.start_csd:
                st = ephemeris.csd_to_unix(self.start_csd)
                et = (
                    ephemeris.csd_to_unix(self.end_csd)
                    if self.end_csd is not None
                    else None
                )

            # Note: include_time_interval includes the specified time interval
            # Using this instead of set_time_range, which only narrows the interval
            # f.include_time_interval(self.start_time, self.end_time)
            f.set_time_range(st, et)

            if self.start_RA and self.end_RA:
                f.include_RA_interval(self.start_RA, self.end_RA)
            elif self.start_RA or self.start_RA:
                self.log.warning(
                    "One but not both of start_RA and end_RA " "are set. Ignoring both."
                )

            f.filter_acqs(di.ArchiveInst.name == self.instrument)

            if self.exclude_daytime:
                f.exclude_daytime()

            if self.exclude_sun:
                f.exclude_sun(
                    time_delta=self.exclude_sun_time_delta,
                    time_delta_rise_set=self.exclude_sun_time_delta_rise_set,
                )

            if self.include_transits:
                time_delta = self.include_transits_time_delta
                ntime_delta = len(time_delta)
                if 1 < ntime_delta < len(self.include_transits):
                    raise ValueError(
                        "Must specify `time_delta` for each source in "
                        "`include_transits` or provide single value for all sources."
                    )
                for ss, src in enumerate(self.include_transits):
                    tdelta = time_delta[ss % ntime_delta] if ntime_delta > 0 else None
                    bdy = (
                        ephemeris.source_dictionary[src]
                        if isinstance(src, str)
                        else src
                    )
                    f.include_transits(bdy, time_delta=tdelta)

            if self.exclude_transits:
                time_delta = self.exclude_transits_time_delta
                ntime_delta = len(time_delta)
                if 1 < ntime_delta < len(self.exclude_transits):
                    raise ValueError(
                        "Must specify `time_delta` for each source in "
                        "`exclude_transits` or provide single value for all sources."
                    )
                for ss, src in enumerate(self.exclude_transits):
                    tdelta = time_delta[ss % ntime_delta] if ntime_delta > 0 else None
                    bdy = (
                        ephemeris.source_dictionary[src]
                        if isinstance(src, str)
                        else src
                    )
                    f.exclude_transits(bdy, time_delta=tdelta)

            if self.source_26m:
                f.include_26m_obs(self.source_26m)

            if len(self.exclude_data_flag_types) > 0:
                f.exclude_data_flag_type(self.exclude_data_flag_types)

            results = f.get_results()
            if not self.return_intervals:
                files = [fname for result in results for fname in result[0]]
                files.sort()
            else:
                files = results
                files.sort(key=lambda x: x[1][0])

        files = mpiutil.world.bcast(files, root=0)

        # Make sure all nodes have container before return
        mpiutil.world.Barrier()

        return files


class QueryRun(task.MPILoggedTask):
    """Find the files belonging to a specific `run`.

    This routine will query the database for the global flag corresponding to
    the given run, and will use the start and end times (as well as the
    instrument) to return a list of the contained files.

    Attributes
    ----------
    run_name : str
        Name of the `run` defined in the database.
    node_spoof : str, optional
        Node spoof argument. See documentation of :class:`ch_util.finder.Finder`.
    """

    run_name = config.Property(proptype=str)
    node_spoof = config.Property(proptype=dict, default=_DEFAULT_NODE_SPOOF)

    def setup(self):
        """Fetch the files in the specified run.

        Returns
        -------
        files : list
            List of files to load
        """
        files = None

        # Query the database on rank=0 only, and broadcast to everywhere else
        if mpiutil.rank0:

            layout.connect_database()

            cat_run = (
                layout.global_flag_category.select()
                .where(layout.global_flag_category.name == "run")
                .get()
            )

            # Find run in database
            run_query = layout.global_flag.select().where(
                layout.global_flag.category == cat_run,
                layout.global_flag.name == self.run_name,
            )

            # Make sure we only have flags with active events
            run_query = (
                run_query.join(layout.graph_obj)
                .join(layout.event)
                .where(layout.event.active)
            )

            if run_query.count() == 0:
                raise RuntimeError("Run %s not found in database" % self.run_name)
            elif run_query.count() > 1:
                raise RuntimeError(
                    "Multiple global flags found in database for run %s" % self.run_name
                )

            run = run_query.get()

            # Fetch run start and end time
            run_event = run.event().get()
            start, end = run_event.start.time, run_event.end.time

            # Fetch the instrument
            if run.inst is None:
                raise RuntimeError("Instrument is not specified in database.")
            inst_obj = run.inst

            # Create a finder object limited to the relevant time
            fi = finder.Finder(node_spoof=self.node_spoof)
            fi.only_corr()

            # Set the time range that encapsulates all the intervals
            fi.set_time_range(start, end)

            # Add in all the time ranges
            # for ti in timerange:
            #     fi.include_time_interval(ti['start'], ti['end'])

            # Only include the required instrument
            fi.filter_acqs(di.ArchiveAcq.inst == inst_obj)

            # Pull out the results and extract all the files
            results = fi.get_results()
            files = [fname for result in results for fname in result[0]]
            files.sort()

        files = mpiutil.world.bcast(files, root=0)

        # Make sure all nodes have container before return
        mpiutil.world.Barrier()

        return files


class QueryDataspecFile(task.MPILoggedTask):
    """Find the available files given a dataspec from a file.

    .. deprecated:: pass1
        Use the `run` global flag in the database,
        combined with :class:`LoadRun` instead.

    Attributes
    ----------
    dataset_file : str
        YAML file containing dataset specification. If not specified, use the
        one contained within the ch_pipeline repository.
    dataset_name : str
        Name of dataset to use.
    archive_root : str
        Root of archive to add to file paths.
    """

    dataset_file = config.Property(proptype=str, default="")
    dataset_name = config.Property(proptype=str, default="")
    node_spoof = config.Property(proptype=dict, default=_DEFAULT_NODE_SPOOF)

    def setup(self):
        """Fetch the files in the given dataspec.

        Returns
        -------
        files : list
            List of files to load
        """

        import yaml

        # Set to default datasets file
        if self.dataset_file == "":
            self.dataset_file = os.path.dirname(__file__) + "/data/datasets.yaml"

        # Check existense and read yaml datasets file
        if not os.path.exists(self.dataset_file):
            raise Exception("Dataset file not found.")

        with open(self.dataset_file, "r") as f:
            dconf = yaml.safe_load(f)

        if "datasets" not in dconf:
            raise Exception("No datasets.")

        dsets = dconf["datasets"]

        # Find the correct dataset
        dspec = None
        for ds in dsets:
            if ds["name"] == self.dataset_name:
                dspec = ds
                break

        # Raise exception if it's not found
        if dspec is None:
            raise Exception(
                "Dataset %s not found in %s." % (self.dataset_name, self.dataset_file)
            )

        if ("instrument" not in dspec) or ("timerange" not in dspec):
            raise Exception("Invalid dataset.")

        # Add archive root if exists
        if self.node_spoof is not None:
            dspec["node_spoof"] = self.node_spoof

        files = files_from_spec(dspec, node_spoof=self.node_spoof)

        return files


class QueryDataspec(task.MPILoggedTask):
    """Find the available files given a dataspec in the config file.

    Attributes
    ----------
    instrument : str
        Name of the instrument.
    timerange : list
        List of time ranges as documented above.
    node_spoof : dict, optional
        Optional node spoof argument.
    """

    instrument = config.Property(proptype=str)
    timerange = config.Property(proptype=list)
    node_spoof = config.Property(proptype=dict, default=_DEFAULT_NODE_SPOOF)

    def setup(self):
        """Fetch the files in the given dataspec.

        Returns
        -------
        files : list
            List of files to load
        """

        dspec = {"instrument": self.instrument, "timerange": self.timerange}

        # Add archive root if exists
        if self.node_spoof is not None:
            dspec["node_spoof"] = self.node_spoof

        files = files_from_spec(dspec, node_spoof=self.node_spoof)

        return files


class QueryAcquisitions(task.MPILoggedTask):
    """Iterate over acquisitions.

    This routine will query the database as specified in the runtime
    configuration file.  It will iterate over the returned acquisitions
    in chronological order, outputing a list of the corresponding files.

    Attributes
    ----------
    node_spoof : dict
        Host and directory in which to find data.
    start_time, end_time : str
        Find all acquisitions between this start and end time.
    instrument : str
        Find all acquisitions from this instrument.
    accept_all_global_flags : bool
        Accept all global flags.
    min_num_files : int
        Do not process acquisitions that contain less than this number of files.
    max_num_files : int
        Maximum number of files to return at once.  If an acquisition
        contains more than this number of files, then it will be split
        up into multiple blocks (pipeline iterations) of size roughly
        equal to max_num_files.
    """

    node_spoof = config.Property(proptype=dict, default=_DEFAULT_NODE_SPOOF)
    instrument = config.Property(proptype=str, default="chimestack")
    start_time = config.Property(default=None)
    end_time = config.Property(default=None)
    accept_all_global_flags = config.Property(proptype=bool, default=False)
    min_num_files = config.Property(proptype=int, default=None)
    max_num_files = config.Property(proptype=int, default=None)

    def setup(self):
        """Query the database, fetch the files, and save to attribute."""

        # Function to break a list of files into groups of roughly the same size
        def _choose_group_size(n, m, accept):
            if (n % m) < accept:
                return m
            l, u = m - 1, m + 1
            while ((n % l) > accept) and ((n % u) > accept):
                l, u = l - 1, u + 1
            if (n % l) < (n % u):
                return l
            else:
                return u

        # Query the database on rank=0 only, and broadcast to everywhere else
        files = None
        if self.comm.rank == 0:

            layout.connect_database()

            fi = finder.Finder(node_spoof=self.node_spoof)
            fi.only_corr()
            if self.accept_all_global_flags:
                fi.accept_all_global_flags()
            fi.set_time_range(self.start_time, self.end_time)
            fi.filter_acqs(di.ArchiveInst.name == self.instrument)

            files = []
            for aa, _ in enumerate(fi.acqs):

                acq_results = fi.get_results_acq(aa)

                filelist = [ff for acqr in acq_results for ff in acqr[0]]
                nfiles = len(filelist)

                if (self.min_num_files is not None) and (nfiles < self.min_num_files):
                    continue

                if (self.max_num_files is None) or (nfiles <= self.max_num_files):
                    files.append(filelist)

                else:
                    group_size = _choose_group_size(
                        nfiles,
                        self.max_num_files,
                        max(1, int(0.10 * self.max_num_files)),
                    )

                    ngroup, offset = nfiles // group_size, (nfiles % group_size) // 2
                    bnd = [offset + gg * group_size for gg in range(ngroup + 1)]
                    bnd[0], bnd[-1] = 0, nfiles

                    files += [
                        filelist[bnd[ii] : bnd[ii + 1]] for ii in range(len(bnd) - 1)
                    ]

        # Broadcast the files to the other nodes
        files = self.comm.bcast(files, root=0)
        self.comm.Barrier()

        self.files = files

    def next(self):
        """Return the files from the next acquisition.

        Returns
        -------
        files : list
            List of files to load
        """
        if len(self.files) == 0:
            raise pipeline.PipelineStopIteration

        files = self.files.pop(0)

        return files


class QueryInputs(task.MPILoggedTask):
    """From a dataspec describing the data create a list of objects describing
    the inputs in the files.

    Attributes
    ----------
    cache : bool
        Only query for the inputs for the first container received. For all
        subsequent files just return the initial set of inputs. This can help
        minimise the number of potentially fragile database operations.
    """

    cache = config.Property(proptype=bool, default=False)

    _cached_inputs = None

    def next(self, ts):
        """Generate an input description from the timestream passed in.

        Parameters
        ----------
        ts : andata.CorrData
            Timestream container.

        Returns
        -------
        inputs : list of :class:`CorrInput`
            A list of describing the inputs as they are in the file.
        """

        # Fetch from the cache if we can
        if self.cache and self._cached_inputs:
            self.log.debug("Using cached inputs.")
            return self._cached_inputs

        inputs = None

        if mpiutil.rank0:

            # Get the datetime of the middle of the file
            time = ephemeris.unix_to_datetime(0.5 * (ts.time[0] + ts.time[-1]))
            inputs = tools.get_correlator_inputs(time)

            inputs = tools.reorder_correlator_inputs(ts.index_map["input"], inputs)

        # Broadcast input description to all ranks
        inputs = mpiutil.world.bcast(inputs, root=0)

        # Save into the cache for the next iteration
        if self.cache:
            self._cached_inputs = inputs

        # Make sure all nodes have container before return
        mpiutil.world.Barrier()

        return inputs


def finder_from_spec(spec, node_spoof=None):
    """Get a `Finder` object from the dataspec.

    Parameters
    ----------
    dspec : dict
        Dataspec dictionary.

    Returns
    -------
    fi : ch_util.finder.Finder
    """

    instrument = spec["instrument"]
    timerange = spec["timerange"]

    fi = None
    if mpiutil.rank0:

        # Get instrument
        inst_obj = (
            di.ArchiveInst.select().where(di.ArchiveInst.name == instrument).get()
        )

        # Ensure timerange is a list
        if not isinstance(timerange, list):
            timerange = [timerange]

        # Find the earliest and latest times
        earliest = min([tr["start"] for tr in timerange])
        latest = max([tr["end"] for tr in timerange])

        # Set the archive_root
        if node_spoof is None and "node_spoof" in spec:
            node_spoof = spec["node_spoof"]

        # Create a finder object limited to the relevant time
        fi = finder.Finder(node_spoof=node_spoof)

        # Set the time range that encapsulates all the intervals
        fi.set_time_range(earliest, latest)

        # Add in all the time ranges
        for ti in timerange:
            fi.include_time_interval(ti["start"], ti["end"])

        # Only include the required instrument
        fi.filter_acqs(di.ArchiveAcq.inst == inst_obj)

    return fi


def files_from_spec(spec, node_spoof=None):
    """Get the names of files in a dataset.

    Parameters
    ----------
    dspec : dict
        Dataspec dictionary.

    Returns
    -------
    files : list
    """
    # Get a list of files in a dataset from an instrument name and timerange.

    files = None

    if mpiutil.rank0:

        # Get the finder object
        fi = finder_from_spec(spec, node_spoof)

        # Pull out the results and extract all the files
        results = fi.get_results()
        files = [fname for result in results for fname in result[0]]
        files.sort()

    files = mpiutil.world.bcast(files, root=0)

    return files
