"""HFB Daily Pipeline processing type.

This processing type is used to run the HFB daily pipeline.

Classes
=======
- :py:class:`HFBDailyProcessing`
"""

import logging
from datetime import datetime
from typing import ClassVar

import chimedb.core as db
import chimedb.data_index as di
import peewee as pw
from caput.util.arraytools import unique_ordered
from ch_ephem.observers import chime

from . import base
from .daily import (
    expand_csd_range,
    files_in_timespan,
    get_filenames_used_by_csds,
    get_flagged_csds,
    get_recent_csds,
)

logger = logging.getLogger(__name__)

DEFAULT_SCRIPT = """
# Cluster configuration
cluster:
  name: {jobname}
  account: rpp-chime

  directory: {dir}
  temp_directory: {tempdir}

  time: {time}
  system: slurm
  nodes: {nodes}
  ompnum: {ompnum}
  pernode: {pernode}
  mem: 768000M

  invoke: mpirun
  invoke_args: "--map-by l3cache:PE=$SLURM_CPUS_PER_TASK"

  venv: {venv}
  module_path: {modpath}
  module_list: {modlist}

# Pipeline task configuration
pipeline:

  logging:
    root: DEBUG
    peewee: INFO
    matplotlib: INFO

  save_versions:
    - caput
    - ch_util
    - ch_pipeline
    - chimedb.core
    - chimedb.data_index
    - chimedb.dataflag
    - cora
    - draco
    - drift
    - fluxcat
    - numpy
    - scipy
    - h5py
    - mpi4py

  tasks:

    - type: caput.pipeline.tasklib.base.SetMPILogging
      params:
        level_rank0: DEBUG
        level_all: WARNING

    - type: caput.pipeline.tasklib.debug.CheckMPIEnvironment
      params:
        timeout: 420

    - type: ch_pipeline.core.dataquery.ConnectDatabase
      params:
        timeout: 5
        ntries: 5

    # Load the telescope model that we need for several steps
    - type: draco.core.io.LoadProductManager
      out: manager
      params:
        product_directory: "{product_path}"

    # Query for all the data for the sidereal day we are processing
    - type: ch_pipeline.core.dataquery.QueryDatabase
      out: filelist_hfb
      params:
        start_csd: {csd[0]:.2f}
        end_csd: {csd[1]:.2f}
        node_spoof:
          fir_online: "/project/rpp-chime/chime/chime_online/"
        acqtype: 'hfb'
        instrument: 'chime'

    # Load HFB files
    - type: ch_pipeline.hfb.io.LoadFiles
      requires: filelist_hfb
      out: tstream_hfb
      params:
        distributed: true
        single_group: false

    # Run HFB directional RFI flagging / bitmap creation
    - type: ch_pipeline.hfb.flagging.HFBDirectionalRFIFlagging
      in: tstream_hfb
      out: rfibitmap_hfb
      params:
        sigma: "{sigma_list}"

    # Group into one sidereal-day container
    - type: draco.analysis.sidereal.SiderealGrouper
      requires: manager
      in: rfibitmap_hfb
      out: rfibitmap_hfb_grouped
      params:
	compression:
          subfreq_rfi:
            chunks: [64, 128, 512]
        save: true
        output_name: "rfibitmap_hfb_{{tag}}.h5"
"""


class HFBDailyProcessing(base.ProcessingType):
    """Daily HFB pipeline.

    Defines the configuration and scheduling logic for running the CHIME
    HFB daily pipeline. For each sidereal day (CSD), this pipeline:

    - Queries the database for HFB acquisitions
    - Loads the corresponding HFB files
    - Runs directional RFI flagging to produce an RFI bitmap
    - Groups results into a sidereal container
    - Writes a truncated output product

    This class also manages job submission, prioritization of recent days,
    exclusion of flagged CSDs, and online/offline file staging.
    """

    type_name = "hfb_daily"
    tag_pattern = r"\d+"

    # Parameters of the job processing
    default_params: ClassVar = {
        # Time range(s) to process
        "intervals": [{"start": "CSD2660", "step": 7}],
        # Fixed CSDS to ignore
        "flags": {},
        # Any days flagged by these flags are excluded from
        # the days to run
        "exclude_flags": ["corrupted_file"],
        # Whether to look for offline data and request it be brought online
        "include_offline_files": True,
        # Number of recent days to prioritize in queue
        "num_recent_days_first": 7,
        # The beam transfers to use
        "product_path": "/project/rpp-chime/chime/bt_empty/chime_4cyl_allfreq/",
        # System modules to use/load
        "modpath": "/project/rpp-chime/chime/chime_env/modules/modulefiles",
        "modlist": "chime/python/2025.10",
        # Job params
        "time": 40,  # How long in minutes?
        "nodes": 4,  # Number of nodes to use.
        "ompnum": 8,  # Number of OpenMP threads
        "pernode": 24,  # Jobs per node
    }
    default_script = DEFAULT_SCRIPT
    # Make sure not to remove hfb files corresponding to CSDs below
    # Absorber search for October 2021
    daemon_config: ClassVar = {
        "keep_online": {"start": "CSD2872", "end": "CSD2943"},
    }

    def _create_hook(self):
        """Produce a list of bad days based on the following criteria."""
        intervals = []
        for t in self.default_params["intervals"]:
            intervals.append((t["start"], t.get("end", None), t.get("step", 1)))
        # Save out a list of heavily flagged days
        self.default_params["flags"].update(
            get_flagged_csds(
                [csd for i in intervals for csd in expand_csd_range(*i)],
                self.default_params["exclude_flags"],
            )
        )

    def _load_hook(self):
        # Process the intervals
        self._intervals = []
        for t in self._revparams["intervals"]:
            self._intervals.append((t["start"], t.get("end", None), t.get("step", 1)))

        self._data_coverage = self._revparams.get("data_coverage", 0.0)
        self._num_recent_days = self._revparams.get("num_recent_days_first", 0)
        self._include_offline_files = self._revparams.get(
            "include_offline_files", False
        )
        self._exclude = set()
        for flag in self._revparams.get("exclude_flags", []):
            for csd in self._revparams["flags"].get(flag, []):
                self._exclude.add(int(csd[3:]))

    def _available_tags(self) -> list:
        """Return all the tags that are available to run.

        This includes any that currently exist or are in the job queue.
        """
        csds = self.runnable_tags(require_online=True)

        return [str(int(csd)) for csd in csds]

    @property
    def _config_tags(self) -> list:
        """Return all tags listed in the revision config.

        Order is preserved, except priority CSDS are placed
        at the start of the list.
        """
        csds = unique_ordered(
            csd for i in self._intervals for csd in expand_csd_range(*i)
        )

        if self._num_recent_days > 0:
            priority = get_recent_csds(csds, self._num_recent_days)
            csds = priority + [csd for csd in csds if csd not in priority]

        return csds

    def runnable_tags(self, require_online=False) -> list:
        """Return all runnable tags, whether or not the data is available."""
        csds = [csd for csd in self._config_tags if csd not in self._exclude]

        runnable = available_csds(
            sorted(csds),
            self._data_coverage,
            require_online=require_online,
        )

        return [csd for csd in csds if csd in runnable]

    def _finalise_jobparams(self, tag, jobparams):
        """Modify the job parameters before the final config is generated.

        Does the following:

        - Set bounds for this CSD.
        - Select the correct template file
        """
        csd = float(tag)

        jobparams.update(
            {
                "csd": [csd, csd + 1],
            }
        )

        return jobparams

    def update_files(self, user=None, retrieve=True, clear=True):
        """Update the status of files used by this revision.

        This includes requesting that soon-to-be needed files get brought
        online, and that files that are no longer needed be moved offline.
        """
        nfiles = {"nretrieve": 0, "nclear": 0}

        if not self._include_offline_files:
            # Cannot try to retrieve files if forbidden. Can
            # still clear out files which are no longer needed
            retrieve = False

        rev_stats = self.status(user=user)
        # Get the upcoming jobs
        upcoming = rev_stats["not_yet_submitted"]
        # Get all the runnable tags
        all_tags = [str(tag) for tag in self.runnable_tags(require_online=False)]

        if retrieve:
            # If there are any upcoming jobs which require offline data,
            # request that these be moved online
            exclude_tags = {
                *rev_stats["pending"],
                *rev_stats["running"],
                *rev_stats["successful"],
                *rev_stats["failed"],
            }
            all_tags = [tag for tag in all_tags if tag not in exclude_tags]

            # Search the next 5 tags and request any that we may want to be brought online.
            online_request_tags = sorted(
                [int(tag) for tag in all_tags[:5] if tag not in upcoming]
            )
            if online_request_tags:
                # Submit the request to bring files online
                nfiles["nretrieve"] = request_offline_csds(online_request_tags)

        if clear:
            # Clear out data that we don't need anymore
            remove_request_tags = sorted([int(tag) for tag in rev_stats["successful"]])
            exclude_tags = [
                *rev_stats["running"],
                *rev_stats["pending"],
                *rev_stats["failed"],
                *rev_stats["not_yet_submitted"],
                *expand_csd_range(*self.daemon_config["keep_online"].values()),
            ]
            exclude_tags = sorted([int(tag) for tag in exclude_tags])
            if remove_request_tags:
                # Submit the request to remove these files
                nfiles["nclear"] = remove_online_csds(remove_request_tags, exclude_tags)

        return nfiles

    def _generate_hook(self, user=None, priority_only=False, check_failed=False):
        # Get the list of tags remaining to run, in order
        to_run = self.status(user=user)["not_yet_submitted"]

        if check_failed:
            requeue = {"chimedb_error", "time_limit", "mpi_error"}

            # Place failed jobs at the start of the queue
            to_run = [
                tag
                for key, tags in self.failed().items()
                for tag in tags
                if key in requeue
            ] + to_run

        # Ensure that the current in-progress acquisition does not get queued
        today = chime.get_current_lsd()
        to_run = [csd for csd in to_run if (today - float(csd)) > 1]

        if priority_only:
            to_run = get_recent_csds(to_run, self._num_recent_days)

        return to_run


def available_csds(
    csds: list,
    data_coverage: float = 0.0,
    require_online: bool = False,
) -> set:
    """Return the subset of csds in `csds` for whom all files are online.

    Parameters
    ----------
    csds
        sorted list of csds to check
    data_coverage
      What fraction of the day must exist in order to be considered available.
      Even if all files are online, if the data coverage is less than this
      fraction, the day shouldn't be processed.
    require_online
        If True, a CSD must have all files available online to be considered
        available. Default is True.

    Returns
    -------
    available
        set of all available csds
    """
    # Figure out which files exist online and which ones exist entirely
    # Repeat the process for hfb data and weather data
    hfb_online, hfb_that_exist = db_get_hfb_files_in_range(csds[0], csds[-1])

    def _available(filenames_online, filenames_that_exist, coverage):
        available = set()
        coverage = coverage * 86400

        def _check_coverage(x):
            time_range = 0
            for tr in x:
                time_range += tr[2] - tr[1]

            if time_range > coverage:
                return True

            return False

        for csd in csds:
            start_time = chime.lsd_to_unix(csd)
            end_time = chime.lsd_to_unix(csd + 1)

            exists, index_exists = files_in_timespan(
                start_time, end_time, filenames_that_exist
            )

            available_offline = _check_coverage(exists)

            # The final file in the span may contain more than one sidereal day
            index_exists = max(index_exists - 1, 0)
            filenames_that_exist = filenames_that_exist[index_exists:]

            if not available_offline:
                # We can't get this data no matter what
                continue

            if require_online:
                # online - list of file start and end times that are online
                # between start_time and end_time
                # index_online - the final index in which data was located
                online, index_online = files_in_timespan(
                    start_time, end_time, filenames_online
                )

                if _check_coverage(online) & (len(online) == len(exists)):
                    available.add(csd)

                index_online = max(index_online - 1, 0)
                filenames_online = filenames_online[index_online:]
            else:
                available.add(csd)

        return available

    return _available(hfb_online, hfb_that_exist, data_coverage)


def db_get_hfb_files_in_range(start_csd: int, end_csd: int):
    """Get the name and start and end times of hfbdata files in the CSD range.

    Return both a list of files which were found on the online node and those
    which were found anywhere. If `has_file` is `N`, these lists should be
    the same.

    Return an empty list if files between start_csd and end_csd are only
    partially available online.

    Total file count is verified by checking files that exist everywhere.

    Parameters
    ----------
    start_csd
        Start date in sidereal day format
    end_csd
        End date in sidereal day format

    Returns
    -------
    filenames_online
        chimestack files available in the timespan, if
        all of them are available online
    filenames_that_exist
        all chimestack files available in the timespan
    """
    # Query all the files in this time range
    start_time = chime.lsd_to_unix(start_csd)
    end_time = chime.lsd_to_unix(end_csd + 1)

    db.connect()

    hfb_inst = di.ArchiveInst.get(name="chime")
    hfb_actype = di.AcqType.get(name="hfb")

    # Query for all chimestack files in the time range
    archive_files = (
        di.ArchiveFileCopy.select(
            di.ArchiveFileCopy.file,
            di.HFBFileInfo.start_time,
            di.HFBFileInfo.finish_time,
        )
        .join(di.ArchiveFile)
        .join(di.ArchiveAcq)
        .switch(di.ArchiveFile)
        .join(di.HFBFileInfo)
    ).where(
        di.ArchiveAcq.inst == hfb_inst,
        di.ArchiveAcq.type == hfb_actype,
        di.HFBFileInfo.start_time < end_time,
        di.HFBFileInfo.finish_time >= start_time,
        di.ArchiveFileCopy.has_file == "Y",
    )

    # Figure out which files are online
    online_node = di.StorageNode.get(name="fir_online", active=True)
    files_online = archive_files.where(di.ArchiveFileCopy.node == online_node)

    files_online = sorted(files_online.tuples(), key=lambda x: x[1])
    # files_that_exist might contain the same file multiple files
    # if it exists in multiple locations (nearline, online, gossec, etc)
    # we only want to include it once, so we initially create a set
    files_that_exist = sorted(set(archive_files.tuples()), key=lambda x: x[1])

    return files_online, files_that_exist


def request_offline_csds(csds: list):
    """Given a list of csds, request that all required data be copied online.

    Request that all data required by the list of CSDS get copied to the
    fir_online node.

    Parameters
    ----------
    csds
        list of integer csds to copy
    pad
        fraction of data from adjacent days that should also be copied online
    """

    def _make_copy_request(file, sources, target):
        # Find a source with the file
        for source in sources:
            try:
                di.ArchiveFileCopy.get(file=file, node=source, has_file="Y")
            except pw.DoesNotExist:
                continue

            # There is a copy of the file on this node, try to copy it.
            try:
                # Check if an active request already exists. If so,
                # leave alpenhorn alone to do its thing
                di.ArchiveFileCopyRequest.get(
                    file=file,
                    group_to=target,
                    node_from=source,
                    completed=False,
                    cancelled=False,
                )
                return 0
            except pw.DoesNotExist:
                di.ArchiveFileCopyRequest.insert(
                    file=file_,
                    group_to=target,
                    node_from=source,
                    cancelled=0,
                    completed=0,
                    n_requests=1,
                    timestamp=datetime.now(),
                ).execute()
                return 1

        # No source with the file
        return 0

    # Figure out which chimestack files are needed
    online_files, files = db_get_hfb_files_in_range(csds[0], csds[-1])
    # Only check files that are not online
    files = [f for f in files if f not in online_files]
    request_hfb_files = get_filenames_used_by_csds(csds, files)

    db.connect(read_write=True)

    target_node = di.StorageGroup.get(name="fir_online")
    offline_node = di.StorageNode.get(name="fir_nearline")
    smallfile_node = di.StorageNode.get(name="fir_smallfile")

    nrequests = 0

    # Request chimestack files be brought back online
    for file_ in request_hfb_files:
        nr = _make_copy_request(file_, [offline_node, smallfile_node], target_node)
        nrequests += nr

    return nrequests


def remove_online_csds(csds_remove: list, csds_keep: list):
    """Remove online files which are solely used by specified csds.

    Check the files required by `csds_remove` and check against those used
    by `csds_keep`. Any files which are _only_ used by csds in `csds_remove`
    are removed from the `fir_online` node, provided that a copy exists
    elsewhere.

    Parameters
    ----------
    csds_remove
        list of integer csds to clear data where possible
    csds_keep
        list of integer csds for which all data is still required
    """
    files, _ = db_get_hfb_files_in_range(
        min(csds_remove[0], csds_keep[0]),
        max(csds_remove[-1], csds_keep[-1]),
    )
    # Get all the files we want to keep and remove. Choosing to
    # keep a file supercedes choosing to remove one
    keep_files = get_filenames_used_by_csds(csds_keep, files)
    remove_files = get_filenames_used_by_csds(csds_remove, files)
    remove_files = [file for file in remove_files if file not in keep_files]

    online_node = di.StorageNode.get(name="fir_online")
    # Establish a read-write database connection
    db.connect(read_write=True)
    # Request that these files be removed from the online node
    di.ArchiveFileCopy.update(wants_file="N").where(
        di.ArchiveFileCopy.file << remove_files,
        di.ArchiveFileCopy.node == online_node,
    ).execute()

    return len(remove_files)
