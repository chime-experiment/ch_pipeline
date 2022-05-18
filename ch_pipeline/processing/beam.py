from ch_util import holography as holo
from ch_util import ephemeris as ephem
from chimedb.core import connect as connect_db

from . import base


DEFAULT_SCRIPT = """
# Cluster configuration
cluster:
  name: {jobname}

  directory: {dir}
  temp_directory: {tempdir}

  time: {time}
  system: cedar
  nodes: {nodes}
  ompnum: {ompnum}
  pernode: {pernode}

  venv: {venv}

# source, hour angle span will be used by multiple tasks
param_anchors:
  - &source_name {src}
  - &hour_angle {ha_span}
  - &db_source_name {src_db}

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
    - draco
    - numpy
    - scipy
    - h5py
    - mpi4py

  tasks:

    - type: draco.core.task.SetMPILogging
      params:
          level_all: WARNING
          level_rank0: INFO

    - type: ch_pipeline.core.dataquery.QueryDatabase
      out: files_wait
      params:
        node_spoof:
          cedar_online: /project/rpp-chime/chime/chime_online
        instrument: {inst}
        source_26m: *db_source_name
        start_time: {start_time}
        end_time: {end_time}
        accept_all_global_flags: True
        exclude_data_flag_types: ["misc"]

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

    - type: ch_pipeline.core.io.LoadCorrDataFiles
      requires: files
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
        fail_if_missing: True

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
    """ """

    type_name = "holo_fstop"
    # tag by name of source and processing run
    tag_pattern = r"(.+)_(\d{4})"

    # Parameters of the job processing
    default_params = {
        "sources": {"CygA": "CYG_A", "CasA": "CAS_A"},
        "gated_sources": {"B0329+54": "B0329+54"},
        "start_time": "20180101T000000",
        "ha_span": 60.0,
        "num_samples": 720,
        "nodes": 4,
        "pernode": 4,
        "ompnum": 12,
        "time": "0-0:15:00",
        "timing_corr": "/project/rpp-chime/chime/chime_processed/timing/rev_00/not_referenced/*_chimetiming_delay.h5",
    }
    default_script = DEFAULT_SCRIPT

    def _available_tags(self):

        self._tags = {}

        # Query database for holography observations of given sources
        start_t = ephem.ensure_unix(self._revparams["start_time"])
        connect_db()

        def get_obs(src):
            db_src = holo.HolographySource.get(holo.HolographySource.name == src)
            return (
                holo.HolographyObservation.select()
                .where(
                    (holo.HolographyObservation.source == db_src),
                    (holo.HolographyObservation.start_time > start_t),
                    (holo.HolographyObservation.quality_flag == 0)
                    | (holo.HolographyObservation.quality_flag == None),
                )
                .order_by(holo.HolographyObservation.start_time)
            )

        for src in self._revparams["sources"]:
            db_obs = get_obs(src)
            for obs in db_obs:
                tag = f"{src}_{obs.id:0>4d}"
                self._tags[tag] = {
                    "start": obs.start_time,
                    "end": obs.finish_time,
                    "src_db": src,
                    "gated": False,
                }

        for src in self._revparams["gated_sources"]:
            db_obs = get_obs(src)
            for obs in db_obs:
                tag = f"{src}_gated_{obs.id:0>4d}"
                self._tags[tag] = {
                    "start": obs.start_time,
                    "end": obs.finish_time,
                    "src_db": src,
                    "gated": True,
                }

        return self._tags.keys()

    def _finalise_jobparams(self, tag, jobparams):
        # adjust start and end time
        jobparams["start_time"] = self._tags[tag]["start"]
        jobparams["end_time"] = self._tags[tag]["end"]

        # set source name
        # some sources are named differently in the database and the ephemeris
        # catalog so we need to map from one to the other
        src_db = self._tags[tag]["src_db"]
        jobparams["src_db"] = src_db
        if self._tags[tag]["gated"]:
            # instrument is different for gated observations
            jobparams["src"] = self._revparams["gated_sources"][src_db]
            jobparams["inst"] = "chime26mgated"
        else:
            jobparams["src"] = self._revparams["sources"][src_db]
            jobparams["inst"] = "chime26m"

        return jobparams
