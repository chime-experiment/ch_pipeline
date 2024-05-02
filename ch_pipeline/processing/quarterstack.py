import re

import numpy as np

from caput import time as ctime
from ch_util import ephemeris
from chimedb import core
from chimedb import dataflag as df

from . import base, daily


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
  mem: 192000M

  venv: {venv}

days: &days
{days}

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
    - numpy
    - scipy
    - h5py
    - mpi4py

  tasks:

    - type: draco.core.task.SetMPILogging
      params:
        level_rank0: DEBUG
        level_all: WARNING

    - type: draco.core.io.LoadProductManager
      out: manager
      params:
        product_directory: "{product_path}"

    - type: draco.core.io.LoadBasicCont
      out: sstream
      params:
        files: *days
        selections:
          freq_range: [{freq[0]:d}, {freq[1]:d}]

    - type: ch_pipeline.analysis.flagging.MaskDay
      in: sstream
      out: sstream_mask

    - type: ch_pipeline.analysis.flagging.MaskMoon
      in: sstream_mask
      out: sstream_mask2

    - type: ch_pipeline.analysis.flagging.DataFlagger
      in: sstream_mask2
      out: sstream_mask3
      params:
        flag_type:
          - acjump_sd
          - rain1mm_sd
          - srs/bad_ringmap_broadband
          - bad_calibration_gains
          - bad_calibration_fpga_restart
          - bad_calibration_acquisition_restart
          - snow
          - decorrelated_cylinder

    # Load gain errors as a function of time
    - type: ch_pipeline.core.io.LoadSetupFile
      out: gain_err
      params:
        filename: {gain_err_file}
        distributed: true
        selections:
          freq_range: [{freq[0]:d}, {freq[1]:d}]

    # Apply a mask that removes frequencies and times that suffer from gain errors
    - type: ch_pipeline.analysis.calibration.FlagNarrowbandGainError
      requires: gain_err
      in: sstream_mask3
      out: mask_gain_err
      params:
        transition: 600.0
        threshold: 1.0e-3
        ignore_input_flags: Yes
        save: false

    - type: draco.analysis.flagging.ApplyRFIMask
      in: [sstream_mask3, mask_gain_err]
      out: sstream_mask4

    # Flag out low weight samples to remove transient RFI artifacts at the edges of
    # flagged regions
    - type: draco.analysis.flagging.ThresholdVisWeightBaseline
      requires: manager
      in: sstream_mask4
      out: full_tvwb_mask
      params:
        relative_threshold: 0.5
        ignore_absolute_threshold: -1
        average_type: "mean"
        pols_to_flag: "all"

    # Apply the tvwb mask. This will modify the data inplace.
    - type: draco.analysis.flagging.ApplyBaselineMask
      in: [sstream_mask4, full_tvwb_mask]
      out: sstream_mask5

    - type: draco.analysis.flagging.RFIMask
      in: sstream_mask5
      out: rfi_mask
      params:
          stack_ind: 66

    - type: draco.analysis.flagging.ApplyRFIMask
      in: [sstream_mask5, rfi_mask]
      out: sstream_mask6

    - type: ch_pipeline.analysis.sidereal.SiderealMean
      in: sstream_mask6
      out: med
      params:
        mask_ra: [[{ra_range[0]:.2f}, {ra_range[1]:.2f}]]
        median: true
        missing_threshold: 0.3
        inverse_variance: false

    - type: ch_pipeline.analysis.sidereal.ChangeSiderealMean
      in: [sstream_mask6, med]
      out: sstream_mask7

    - type: draco.analysis.sidereal.SiderealStacker
      in: sstream_mask7
      out: sstack_stack
      params:
        tag: {tag}

    # Precision truncate the sidereal stack data
    - type: draco.core.io.Truncate
      in: sstack_stack
      out: sstack_trunc
      params:
        dataset:
          vis:
            weight_dataset: vis_weight
            variance_increase: 1.0e-4
          vis_weight: 1.0e-6
        compression:
          vis:
            chunks: [16, 512, 512]
          vis_weight:
            chunks: [16, 512, 512]

    # Save the sstack out to a zarr zip file
    - type: draco.core.io.SaveZarrZip
      in: sstack_trunc
      out: zip_handle
      params:
        output_name: "sstack.zarr.zip"

    - type: draco.analysis.ringmapmaker.RingMapMaker
      requires: manager
      in: sstack_trunc
      out: ringmap
      params:
        single_beam: true
        weight: "natural"
        exclude_intracyl: false
        include_auto: false
        npix: 1024

    # Precision truncate the chunked normal ringmap
    - type: draco.core.io.Truncate
      in: ringmap
      out: ringmap_trunc
      params:
        dataset:
          map:
            weight_dataset: weight
            variance_increase: 1.0e-4
          weight: 1.0e-6
        compression:
          map:
            chunks: [1, 1, 16, 512, 512]
          weight:
            chunks: [1, 16, 512, 512]
          dirty_beam:
            chunks: [1, 1, 16, 512, 512]

    # Save the ringmap out to a ZarrZip file
    - type: draco.core.io.SaveZarrZip
      in: ringmap_trunc
      out: zip_handle
      params:
        output_name: "ringmap.zarr.zip"

    # Estimate the delay spectrum
    - type: draco.analysis.delay.DelaySpectrumEstimator
      requires: manager
      in: sstack_stack
      params:
        freq_zero: 800.0
        complex_timedomain: true
        nfreq: {nfreq_delay}
        nsamp: 40
        save: true
        output_name: "dspec.h5"

    # Wait for the Zipping to finish
    - type: draco.core.io.WaitZarrZip
      in: zip_handle
"""


class QuarterStackProcessing(base.ProcessingType):
    """Stacks the daily data within quarters, subdivided into interleaved jack knifes.

    This uses opinions in the dataflag database `chimedb.dataflag` to determine which
    days are good and bad for each revision of the daily processing. It will then
    take each good day (taking from the latest revision if multiple revisions contain
    a good version), and then perform the stacking.

    Implementation
    --------------
    The implementation of this processing type is a little different to others, so
    it's worth discussing in more detail. Notably all of the decisions about what
    data to include are made at the time the revision is *created*. At that point the
    database is queried, good days are found, checked for existence, and they are
    divided up into groups for each of the output stacks. These decisions are saved
    into the `revconfig.yaml` file. When jobs to create the individual items (i.e.
    stacks) are generated, the set of days to stack are simply pulled from this
    revision config.

    This has the important result that after the revision has been created, changes
    to the daily data, e.g. processing new days or changing opinions about them in
    the database, will not effect the stacks to be generated.
    """

    type_name = "quarterstack"
    tag_pattern = r"(?P<year>\d{4})q(?P<quarter>[1-4])p(?P<partition>\d)"

    # Parameters of the job processing
    default_params = {
        # Daily processing revisions to use (later entries in this list take precedence
        # over earlier ones)
        "daily_revisions": ["rev_07"],
        # Usually the opinions are queried for each revision, this dictionary allows
        # that to be overridden. Each `data_rev: opinion_rev` pair means that the
        # opinions used to select days for `data_rev` will instead be taken from
        # `opinion_rev`.
        "opinion_overrides": {
            "rev_03": "rev_02",
        },
        "daily_root": "/project/rpp-chime/chime/chime_processed/",
        # Frequencies to process
        "freq": [0, 1024],
        "nfreq_delay": 1025,
        # The beam transfers to use (need to have the same freq range as above)
        "product_path": "/project/rpp-chime/chime/bt_empty/chime_4cyl_allfreq/",
        "partitions": 2,
        # Don't generate quarter stacks with less days than this
        "min_days": 5,
        # RA range used to estimate the cross talk for each quarter
        "crosstalk_ra": {
            "q1": [165, 180],
            "q2": [240, 255],
            "q3": [315, 330],
            "q4": [45, 60],
        },
        "gain_error_file": {
            2018: (
                "/project/rpp-chime/chime/chime_processed/gain/gain_errors/rev_00/"
                "20180905_20191231_gain_inverted_error_input_flagged.h5"
            ),
            2019: (
                "/project/rpp-chime/chime/chime_processed/gain/gain_errors/rev_00/"
                "20180905_20191231_gain_inverted_error_input_flagged.h5"
            ),
            2020: (
                "/project/rpp-chime/chime/chime_processed/gain/gain_errors/rev_00/"
                "20200101_20201231_gain_inverted_error_input_flagged.h5"
            ),
            # Update these below when they become available
            2021: (
                "/project/rpp-chime/chime/chime_processed/gain/gain_errors/rev_00/"
                "20210101_20211231_gain_inverted_error_input_flagged.h5"
            ),
            2022: (
                "/project/rpp-chime/chime/chime_processed/gain/gain_errors/rev_00/"
                "20200101_20201231_gain_inverted_error_input_flagged.h5"
            ),
        },
        # Job params
        "time": 180,  # How long in minutes?
        "nodes": 16,  # Number of nodes to use.
        "ompnum": 6,  # Number of OpenMP threads
        "pernode": 8,  # Jobs per node
    }

    default_script = DEFAULT_SCRIPT

    def _create_hook(self):
        """Create the revision.

        This tries to determine which days are good and bad, and partitions the
        available good days into the individual stacks.
        """

        days = {}

        core.connect()

        opinion_overrides = self.default_params.get("opinion_overrides", {})

        # Go over each revision and construct the set of LSDs we should stack, and save
        # the path to each.
        # NOTE: later entries in `daily_revisions` will override LSDs found in earlier
        # revisions.
        for rev in self.default_params["daily_revisions"]:
            daily_path = (
                self.root_path
                if self.default_params["daily_root"] is None
                else self.default_params["daily_root"]
            )
            daily_rev = daily.DailyProcessing(rev, root_path=daily_path)

            # Get the revision used to determine the opinions, by default this is the
            # revision, but it can be overriden
            opinion_rev = opinion_overrides.get(rev, rev)

            # Get all the bad days in this revision
            revision = df.DataRevision.get(name=opinion_rev)
            query = (
                df.DataFlagOpinion.select(df.DataFlagOpinion.lsd)
                .distinct()
                .where(
                    df.DataFlagOpinion.revision == revision,
                    df.DataFlagOpinion.decision == "bad",
                )
            )
            bad_days = [x[0] for x in query.tuples()]

            # Get all the good days
            query = (
                df.DataFlagOpinion.select(df.DataFlagOpinion.lsd)
                .distinct()
                .where(
                    df.DataFlagOpinion.revision == revision,
                    df.DataFlagOpinion.decision == "good",
                )
            )
            good_days = [x[0] for x in query.tuples()]

            for d in daily_rev.ls():
                try:
                    lsd = int(d)
                except ValueError as e:
                    raise RuntimeError(
                        f'Could not parse string tag "{d}" into a valid LSD'
                    ) from e

                # Filter out known bad days here
                if (lsd in bad_days) or (lsd not in good_days):
                    continue

                # Insert the day and path into the dict, this will replace the entries
                # from prior revisions
                path = daily_rev.base_path / d
                days[lsd] = path

        lsds = sorted(days)

        # Map each LSD into the quarter it belongs in and find which quarters we have
        # data for
        dates = ctime.unix_to_datetime(ephemeris.csd_to_unix(np.array(lsds)))
        yq = np.array([f"{d.year}q{(d.month - 1) // 3 + 1}" for d in dates])
        quarters = np.unique(yq)

        npart = self.default_params["partitions"]

        lsd_partitions = {}

        # For each quarter divide the LSDs it contains into a number of partitions to
        # give jack knifes
        for quarter in quarters:
            lsds_in_quarter = sorted(np.array(lsds)[yq == quarter])

            # Skip quarters with too few days in them
            if len(lsds_in_quarter) < self.default_params["min_days"] * npart:
                continue

            for i in range(npart):
                lsd_partitions[f"{quarter}p{i}"] = [
                    int(d) for d in lsds_in_quarter[i::npart]
                ]

        # Save the relevant parameters into the revisions configuration
        self.default_params["days"] = {
            int(day): str(path) for day, path in days.items()
        }
        self.default_params["stacks"] = lsd_partitions

    def _available_tags(self):
        """Return all the tags that are available to run.

        This includes any that currently exist or are in the job queue.
        """

        return list(self._revparams["stacks"].keys())

    def _finalise_jobparams(self, tag, jobparams):
        """Modify the job parameters before the final config is generated.

        Unfortunately this needs to by hand generate the list of daily file paths to
        process and insert it as a string into the YAML. It would be nice to find a
        better way to do this.
        """

        days = self._revparams["stacks"][tag]
        paths = self._revparams["days"]

        # TODO: find a better way to do this. Some kind of configuration language
        # (Jsonnet/YTT/...) seems like it would be a better idea here
        day_list_str = "\n" + "\n".join(
            [f"- {paths[day]}/sstream_lsd_{day}.zarr.zip" for day in days]
        )

        year, quarter, _ = self._parse_tag(tag)
        ra_range = self._revparams["crosstalk_ra"][f"q{quarter}"]
        gain_err_file = self._revparams["gain_error_file"][year]

        jobparams.update(
            {
                "days": day_list_str,
                "ra_range": ra_range,
                "gain_err_file": gain_err_file,
            }
        )

        return jobparams

    def _parse_tag(self, tag):
        """Extract the year, quarter and partition from the tag."""

        mo = re.match(self.tag_pattern, tag)

        if not mo:
            raise ValueError(f'Tag "{tag}" is invalid.')

        return tuple(int(mo[k]) for k in ["year", "quarter", "partition"])
