# === Start Python 2/3 compatibility
from __future__ import absolute_import, division, print_function, unicode_literals
from future.builtins import *  # noqa  pylint: disable=W0401, W0614
from future.builtins.disabled import *  # noqa  pylint: disable=W0401, W0614

# === End Python 2/3 compatibility

import datetime

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
  mem: 192000M

  venv: {venv}

# Pipeline task configuration
pipeline:
  tasks:
    - type: ch_pipeline.core.containers.MonkeyPatchContainers

    - type: draco.core.task.SetMPILogging
      params:
        level_rank0: DEBUG
        level_all: WARNING

    - type: ch_pipeline.core.dataquery.QueryDatabase
      out: filelist
      params:
        start_csd: {csd[0]:.2f}
        end_csd: {csd[1]:.2f}

        accept_all_global_flags: true
        node_spoof:
          cedar_archive: "/project/rpp-krs/chime/chime_archive/"
        instrument: chimestack

    - type: draco.core.io.LoadProductManager
      out: manager
      params:
        product_directory: "{product_path}"

    - type: ch_pipeline.core.io.LoadSetupFile
      out: tcorr
      params:
        filename: "{timing_file}"

    - type: ch_pipeline.core.io.LoadCorrDataFiles
      requires: filelist
      out: tstream
      params:
        channel_range: [{freq[0]:d}, {freq[1]:d}]

    - type: ch_pipeline.analysis.timing.ApplyTimingCorrection
      requires: tcorr
      in: tstream
      out: tstream_corr
      params:
        use_input_flags: Yes

    - type: draco.analysis.transform.CollateProducts
      requires: manager
      in: tstream_corr
      out: tstream_col
      params:
        weight: "natural"

    - type: draco.analysis.sidereal.SiderealGrouper
      requires: manager
      in: tstream_col
      out: groupday

    - type: ch_pipeline.analysis.flagging.RFIFilter
      in: groupday
      out: rfimask
      params:
        stack: Yes
        flag1d: No
        rolling: Yes
        apply_static_mask: Yes
        keep_auto: Yes
        keep_ndev: Yes
        freq_width: 10.0
        time_width: 420.0
        threshold_mad: 6.0
        save: Yes
        output_root: "rfi_mask_"
        nan_check: No

    - type: ch_pipeline.analysis.flagging.ApplyCorrInputMask
      in: [groupday, rfimask]
      out: groupday_rfi

    - type: draco.analysis.flagging.SmoothVisWeight
      in: groupday_rfi
      out: groupday_smoothweight

    - type: draco.analysis.sidereal.SiderealRegridder
      requires: manager
      in: groupday_smoothweight
      out: sstream
      params:
        samples: 4096
        weight: natural
        save: true
        output_root: "sstream_col_"

    - type: ch_pipeline.analysis.mapmaker.RingMapMaker
      requires: manager
      in: sstream
      out: ringmap
      params:
        single_beam: Yes
        weight: natural
        exclude_intracyl: No
        include_auto: No
        save: Yes
        output_root: "ringmap_"

    - type: ch_pipeline.analysis.mapmaker.RingMapMaker
      requires: manager
      in: sstream
      out: ringmapint
      params:
        single_beam: Yes
        weight: natural
        exclude_intracyl: Yes
        include_auto: No
        save: Yes
        output_root: "ringmap_intercyl_"

    - type: ch_pipeline.analysis.flagging.DayMask
      in: sstream
      out: sstream_mask

    - type: draco.analysis.flagging.RFIMask
      in: sstream_mask
      out: sstream_rfi2
      params:
        stack_ind: 66

    - type: draco.analysis.delay.DelaySpectrumEstimator
      requires: manager
      in: sstream_rfi2
      params:
        freq_zero: 800.0
        nfreq: 1025
        nsamp: 40
        save: Yes
        output_root: "delayspectrum_"
"""


class DailyProcessing(base.ProcessingType):
    """
    """

    type_name = "daily"
    tag_pattern = r"\d+"

    # Parameters of the job processing
    default_params = {
        # Time range(s) to process
        "intervals": [
            # Intervals as defined by Mateus
            # Pass A (start trimmed for timing sols)
            {"start": "20181221T000000Z", "end": "20181229T000000Z"},
            # Pass B
            {"start": "20190111T000000Z", "end": "20190207T000000Z"},
            # Pass C (end trimmed for timing sols)
            {"start": "20190210T000000Z", "end": "20190304T000000Z"},
        ],
        # Amount of padding each side of sidereal day to load
        "padding": 0.02,
        # Frequencies to process
        "freq": [0, 1024],
        # The beam transfers to use (need to have the same freq range as above)
        "product_path": "/scratch/jrs65/bt_empty/chime_4cyl_allfreq/",
        # File for the timing correction
        "timing_file": (
            "/scratch/ssiegel/timing/v1/referenced/"
            + "20181220T235147Z_to_20190304T135948Z_chimetiming_delay.h5"
        ),
        # Job params
        "time": 180,  # How long in minutes?
        "nodes": 16,  # Number of nodes to use.
        "ompnum": 6,  # Number of OpenMP threads
        "pernode": 8,  # Jobs per node
    }
    default_script = DEFAULT_SCRIPT

    def _load_hook(self):

        # Process the intervals
        self._intervals = []
        for t in self._revparams["intervals"]:
            self._intervals.append((t["start"], t.get("end", None)))
        self._padding = self._revparams["padding"]

    def _available_tags(self):
        """Return all the tags that are available to run.

        This includes any that currently exist or are in the job queue.
        """

        # TODO: should decide availability based on what data is actually
        # available
        # - Need to find all correlator data in the range
        # - Figure out which ones are on cedar
        # - Figure out which sidereal days are covered by this range

        csds = []
        for interval in self._intervals:
            csds += csds_in_range(*interval)

        tags = ["%i" % csd for csd in csds]

        return tags

    def _finalise_jobparams(self, tag, jobparams):
        """Set bounds for this CSD."""

        csd = float(tag)
        jobparams.update({"csd": [csd - self._padding, csd + 1 + self._padding]})

        return jobparams


class TestDailyProcessing(DailyProcessing):
    """A test version of the daily processing.

    Processes only 16 frequencies on a single node.
    """

    type_name = "test_daily"

    # Override params above
    default_params = DailyProcessing.default_params.copy()
    default_params.update(
        {
            "intervals": [{"start": "20181221T000000Z", "end": "20181225T000000Z"}],
            "freq": [192, 208],
            "product_path": "/scratch/jrs65/bt_empty/chime_4cyl_10freq/",
            "time": 60,  # How long in minutes?
            "nodes": 1,  # Number of nodes to use.
            "ompnum": 12,  # Number of OpenMP threads
            "pernode": 4,  # Jobs per node
        }
    )


def csds_in_range(start, end):
    """Get the CSDs within a time range.

    Parameters
    ----------
    start : datetime
        Start of interval.
    end : datetime
        End of interval. If `None` use now.

    Returns
    -------
    csds : list of ints
    """

    import math
    from ch_util import ephemeris

    if end is None:
        end = datetime.datetime.utcnow()

    start_csd = ephemeris.unix_to_csd(ephemeris.ensure_unix(start))
    end_csd = ephemeris.unix_to_csd(ephemeris.ensure_unix(end))

    start_csd = math.floor(start_csd)
    end_csd = math.ceil(end_csd)

    csds = [day for day in range(start_csd, end_csd + 1)]
    return csds
