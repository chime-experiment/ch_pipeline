"""Holography pipeline processing type.

Classes
=======
- :py:class:`HolographyFringestop`
"""

from typing import ClassVar

from caput import time as ctime
from ch_util import holography as holo
from chimedb.core import connect as connect_db

from . import base

DEFAULT_SCRIPT = """
# Cluster configuration
cluster:
  name: {jobname}

  directory: {dir}
  temp_directory: {tempdir}

  time: {time}
  system: fir
  nodes: {nodes}
  ompnum: {ompnum}
  pernode: {pernode}

  venv: {venv}
  module_path: {modpath}
  module_list: {modlist}

# source, hour angle span will be used by multiple tasks
param_anchors:
  - &source_name {src}
  - &hour_angle {ha_span}
  - &db_source_name {src_db}

# Pipeline task configuration
pipeline:
  tasks:

    - type: draco.core.task.SetMPILogging
      params:
          level_all: INFO
          level_rank0: DEBUG

    - type: ch_pipeline.core.dataquery.QueryDatabase
      out: files_wait
      params:
        node_spoof:
            fir_online: /project/rpp-chime/chime/chime_online
        instrument: chime26m
        source_26m: *db_source_name
        start_time: {start_time}
        end_time: {end_time}
        accept_all_global_flags: True
        exclude_data_flag_types: ["misc"]
        return_intervals: True

    - type: draco.core.io.LoadFilesFromParams
      out: tcorr
      params:
        files: {timing_corr}
        distributed: No

    - type: draco.core.misc.AccumulateList
      in: tcorr
      out: tcorrlist

    - type: draco.core.misc.WaitUntil
      requires: tcorrlist
      in: files_wait
      out: files

    - type: ch_pipeline.analysis.beam.FilterHolographyProcessed
      in: files
      out: files_filt
      params:
        source: *db_source_name
        processed_dir: [{dir}, {tempdir}]

    - type: ch_pipeline.core.io.LoadCorrDataFiles
      requires: files_filt
      out: tstreams

    - type: ch_pipeline.analysis.timing.ApplyTimingCorrection
      requires: tcorrlist
      in: tstreams
      out: tstreams_corr
      params:
        pass_if_missing: True

    - type: ch_pipeline.analysis.beam.TransitGrouper
      in: tstreams_corr
      out: transits
      params:
        source: *source_name
        db_source: *db_source_name
        ha_span: *hour_angle

    - type: ch_pipeline.core.dataquery.QueryInputs
      in: transits
      out: inputs

    - type: ch_pipeline.analysis.fringestop.FringeStop
      in: [transits, inputs]
      out: transits_fs
      params:
        source: *source_name
        wterm: True
        overwrite: True

    - type: ch_pipeline.analysis.decorrelation.CorrectDecorrelation
      in: [transits_fs, inputs]
      out: transits_corr
      params:
        source: *source_name
        wterm: True
        overwrite: True

    - type: ch_pipeline.analysis.beam.TransitRegridder
      in: transits_corr
      out: transit_regrid
      params:
        ha_span: *hour_angle
        samples: {num_samples}
        snr_cov: 1.
        source: *source_name
        lanczos_width: 7
        mask_zero_weight: True

    - type: ch_pipeline.analysis.beam.MakeHolographyBeam
      in: [transit_regrid, inputs]
      params:
        save:  Yes
"""


class HolographyFringestop(base.ProcessingType):
    """Holography beam pipeline processing type."""

    type_name = "holo_fstop"
    # tag by name of source and processing run
    tag_pattern = r"(.+)_run(\d{3})"

    # Parameters of the job processing
    default_params: ClassVar = {
        "src": ["CYG_A", "CAS_A"],
        "src_db": ["CygA", "CasA"],
        "start_time": "20180101T000000",
        "end_time": "20200101T000000",
        "transits_per_run": 10,
        "ha_span": 60.0,
        "num_samples": 720,
        "nodes": 1,
        "pernode": 8,
        "ompnum": 24,
        "time": "0-4:00:00",
        "timing_corr": "/project/rpp-chime/chime/chime_processed/timing/rev_00/not_referenced/*_chimetiming_delay.h5",
        # System modules to use/load
        "modpath": "/project/rpp-chime/chime/chime_env/modules/modulefiles",
        "modlist": "chime/python/2022.06",
    }
    default_script = DEFAULT_SCRIPT

    def _available_tags(self):
        self._tags = {}
        # Divide observations by source and in groups
        start_t = ctime.ensure_unix(self._revparams["start_time"])
        end_t = ctime.ensure_unix(self._revparams["end_time"])
        connect_db()
        for src in self._revparams["src_db"]:
            # query database for observations within time range and sort by time
            db_src = holo.HolographySource.get(holo.HolographySource.name == src)
            db_obs = (
                holo.HolographyObservation.select()
                .where(holo.HolographyObservation.source == db_src)
                .where(
                    (holo.HolographyObservation.start_time > start_t)
                    & (holo.HolographyObservation.finish_time < end_t)
                )
                .where(
                    (holo.HolographyObservation.quality_flag == 0)
                    | (holo.HolographyObservation.quality_flag == None)  # noqa: E711
                )
                .order_by(holo.HolographyObservation.start_time)
            )

            # divide into groups to process together
            n_per = self._revparams["transits_per_run"]
            n_groups = len(db_obs) // n_per + (len(db_obs) % n_per != 0)
            for i in range(n_groups):
                tag = f"{src}_run{i:0>3d}"
                # set up time range for these transits, with 1h padding
                i *= n_per
                j = i + n_per - 1
                if j >= len(db_obs):
                    j = len(db_obs) - 1
                bnds = (db_obs[i].start_time - 3600, db_obs[j].finish_time + 3600)
                self._tags[tag] = {"start": bnds[0], "end": bnds[1], "src_db": src}

        return self._tags.keys()

    def _finalise_jobparams(self, tag, jobparams):
        # adjust start and end time
        jobparams["start_time"] = self._tags[tag]["start"]
        jobparams["end_time"] = self._tags[tag]["end"]

        # set source
        jobparams["src_db"] = self._tags[tag]["src_db"]
        jobparams["src"] = self._revparams["src"][
            self._revparams["src_db"].index(self._tags[tag]["src_db"])
        ]

        return jobparams
