"""Processing type to reprocess daily data products."""

from typing import ClassVar

from caput.util.arraytools import unique_ordered

from . import base, client, daily

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

    # Load the telescope model that we need for several steps
    - type: draco.core.io.LoadProductManager
      out: manager
      params:
        product_directory: "{product_path}"

    # Load the sidereal stream we want to process
    - type: caput.pipeline.tasklib.io.LoadFilesFromParams
      out: sstream
      params:
        files: "{src_type_path}/{csd}/{file_match}*{csd}.*"
"""


class SiderealReprocessing(base.ProcessingType):
    """Baseclass for a reprocessing type.

    Reprocess from a product of a daily pipeline revision.

    A revision of this processing type is directly linked to an existing revision of the Daily
    processing type and pulls available files from there. By default, it is linked to the most
    recent Daily Processing revision, and can be changed in the revconfig file after creation.
    In theory, this will be used to quickly re-generate products from an already-processed
    Daily type, with custom changes made to the config.

    This is non-functional on its own and is designed to be subclassed.
    """

    type_name = "reprocess"
    tag_pattern = r"\d+"

    # Default processing parameters
    default_params: ClassVar = {
        # Which type of file to glob against
        "file_match": "tstream",
        # Just include all CSDS for now - user can specify narrower range
        "intervals": [
            {"start": "CSD1000", "end": "CSD9999"},
        ],
        # The beam transfers to use (need to have the same freq range as above)
        "product_path": "/project/rpp-chime/chime/bt_empty/chime_4cyl_allfreq/",
        # Calibration times for thermal correction
        "caltimes_file": (
            "/project/rpp-chime/chime/chime_processed/gain/calibration_times/"
            "20180902_20201230_calibration_times.h5"
        ),
        # File for the timing correction
        "timing_file": (
            "/project/rpp-chime/chime/chime_processed/timing/rev_00/referenced/"
            "*_chimetiming_delay.h5"
        ),
        # File containing the freq map being used for processing the data
        "freqmap_file": (
            "/project/rpp-chime/chime/chime_processed/freq_map/"
            "20180902_20220927_freq_map.h5"
        ),
        # Catalogs to extract fluxes of
        "catalogs": [
            "/project/rpp-chime/chime/chime_processed/catalogs/ps_cora_10Jy.h5",
            "/project/rpp-chime/chime/chime_processed/catalogs/ps_QSO_05Jy.h5",
            "/project/rpp-chime/chime/chime_processed/catalogs/ps_OVRO.h5",
            "/project/rpp-chime/chime/chime_processed/catalogs/ps_requested.h5",
        ],
        # Annual template files for template subtraction/blending
        "template_file": {
            2018: "/project/rpp-chime/chime/templates/rev_08/sstack_2019.h5",
            2019: "/project/rpp-chime/chime/templates/rev_08/sstack_2019.h5",
            2020: "/project/rpp-chime/chime/templates/rev_08/sstack_2020.h5",
            2021: "/project/rpp-chime/chime/templates/rev_08/sstack_2021.h5",
            2022: "/project/rpp-chime/chime/templates/rev_08/sstack_2022.h5",
            2023: "/project/rpp-chime/chime/templates/rev_08/sstack_2023.h5",
            2024: "/project/rpp-chime/chime/templates/rev_08/sstack_2024.h5",
            2025: "/project/rpp-chime/chime/templates/rev_08/sstack_2025.h5",
        },
        # System modules to use/load
        "modpath": "/project/rpp-chime/chime/chime_env/modules/modulefiles",
        "modlist": "chime/python/2026.05",
        "nfreq_delay": 1025,
        # Job params
        "time": 60,  # How long in minutes?
        "nodes": 2,  # Number of nodes to use.
        "ompnum": 8,  # Number of OpenMP threads
        "pernode": 24,  # Jobs per node
    }

    default_script = DEFAULT_SCRIPT

    def _available_tags(self):
        """Return all the tags that are available to run.

        This includes any that currently exist or are in the job queue.
        """
        # Get the csds which are available in the source daily revision
        csds_available = [int(i) for i in self.source_rev.ls()]
        # Get all unique csds from the config, maintaining order
        csds = unique_ordered(
            csd for i in self._intervals for csd in daily.expand_csd_range(*i)
        )
        # Return any tags which are both available and in the config
        return [f"{csd:.0f}" for csd in csds if csd in csds_available]

    def _finalise_jobparams(self, tag, jobparams):
        """Set the csd to process."""
        jobparams.update({"csd": int(tag)})

        return jobparams

    def _load_hook(self):
        """Process relevant items from the rev config file."""
        # get a reference to the daily processing instance that
        # we are pulling data from
        source_rev = self._revparams["src_rev"]
        self.source_rev = client.PRev().convert(f"daily:{source_rev}", None, None)

        # Process the intervals given in the rev config
        self._intervals = []
        for t in self._revparams["intervals"]:
            self._intervals.append((t["start"], t.get("end", None), t.get("step", 1)))

    def _create_hook(self):
        """Finalize the default configuration."""
        # Include the daily revision that we want to pull data from.
        # By default, this is the most recent revision
        latest_daily_rev = daily.DailyProcessing.latest()
        self.default_params.update(
            {
                "src_rev": latest_daily_rev.revision,
                "src_type_path": str(latest_daily_rev.base_path),
            }
        )
        self._update_default_params_hook()

    def _update_default_params_hook(self):
        """Overwrite to add custom default_params modifications."""
        pass
