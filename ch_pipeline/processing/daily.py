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
    - type: ch_pipeline.core.containers.MonkeyPatchContainers

    - type: draco.core.task.SetMPILogging
      params:
        level_rank0: DEBUG
        level_all: WARNING

    # Query for all the data for the sidereal day we are processing
    - type: ch_pipeline.core.dataquery.QueryDatabase
      out: filelist
      params:
        start_csd: {csd[0]:.2f}
        end_csd: {csd[1]:.2f}
        accept_all_global_flags: true
        node_spoof:
          cedar_archive: "/project/rpp-krs/chime/chime_archive/"
        instrument: chimestack

    # Load the telescope model that we need for several steps
    - type: draco.core.io.LoadProductManager
      out: manager
      params:
        product_directory: "{product_path}"

    # Load and accumulate the available timing correction files.
    - type: draco.core.io.LoadFilesFromParams
      out: tcorr
      params:
        files: "{timing_file}"
        distributed: false

    - type: draco.core.misc.AccumulateList
      in: tcorr
      out: tcorrlist

    # We need to block the loading of the files until all the timing correction files are available
    - type: draco.core.misc.WaitUntil
      requires: tcorrlist
      in: filelist
      out: filelist2

    # Load the individual data files
    - type: ch_pipeline.core.io.LoadCorrDataFiles
      requires: filelist2
      out: tstream_orig
      params:
        channel_range: [{freq[0]:d}, {freq[1]:d}]

    # Correct the timing distribution errors
    - type: ch_pipeline.analysis.timing.ApplyTimingCorrection
      requires: tcorrlist
      in: tstream_orig
      out: tstream_corr
      params:
        use_input_flags: true

    # Get the telescope layout that was active when this data file was recorded
    - type: ch_pipeline.core.dataquery.QueryInputs
      in: tstream_corr
      out: inputmap
      params:
        cache: true

    # Correct the miscalibration of the data due to the varying estimates of telescope
    # rotation
    - type: ch_pipeline.analysis.calibration.CorrectTelescopeRotation
      in: [tstream_corr, inputmap]
      out: tstream_rot
      params:
        rotation: -0.071

    # Correct an early bug in the interpretation of the timestamps that effected the
    # calibration
    - type: ch_pipeline.analysis.calibration.CorrectTimeOffset
      in: [tstream_rot, inputmap]
      out: tstream

    # Calculate the system sensitivity for this file
    - type: draco.analysis.sensitivity.ComputeSystemSensitivity
      requires: manager
      in: tstream_corr
      out: sensitivity
      params:
        exclude_intracyl: true

    # Average over redundant baselines across all cylinder pairs
    - type: draco.analysis.transform.CollateProducts
      requires: manager
      in: tstream_corr
      out: tstream_col
      params:
        weight: "natural"

    # Concatenate together all the days timestream information
    - type: draco.analysis.sidereal.SiderealGrouper
      requires: manager
      in: tstream_col
      out: tstream_day

    # Concatenate together all the days sensitivity information and output it
    # for validation
    - type: draco.analysis.sidereal.SiderealGrouper
      requires: manager
      in: sensitivity
      out: sensitivity_day
      params:
        save: true
        output_name: "sensitivity_{{tag}}.h5"

    # Calculate the RFI mask from the autocorrelation data
    - type: ch_pipeline.analysis.flagging.RFIFilter
      in: tstream_day
      out: rfimask
      params:
        stack: true
        flag1d: false
        rolling: true
        apply_static_mask: true
        keep_auto: true
        keep_ndev: true
        freq_width: 10.0
        time_width: 420.0
        threshold_mad: 6.0
        save: true
        output_name: "rfi_mask_{{tag}}.h5"
        nan_check: false

    # Apply the RFI mask. This will modify the data in place.
    - type: ch_pipeline.analysis.flagging.ApplyCorrInputMask
      in: [tstream_day, rfimask]
      out: tstream_day_rfi

    # Calculate the thermal gain correction
    - type: ch_pipeline.analysis.calibration.ThermalCalibration
      in: tstream_day_rfi
      out: thermal_gain
      params:
        caltime_path: "{caltimes_file}"

    # Apply the thermal correction
    - type: draco.core.misc.ApplyGain
      in: [tstream_day_rfi, thermal_gain]
      out: tstream_thermal_corrected

    # Smooth the noise estimates which suffer from sample variance
    - type: draco.analysis.flagging.SmoothVisWeight
      in: tstream_thermal_corrected
      out: tstream_day_smoothweight

    # Regrid the data onto a regular grid in sidereal time
    - type: draco.analysis.sidereal.SiderealRegridderCubic
      requires: manager
      in: tstream_day_smoothweight
      out: sstream
      params:
        samples: 4096
        weight: natural
        save: true
        output_name: "sstream_{{tag}}.h5"

    - type: draco.analysis.flagging.RFIMask
      in: sstream
      out: sstream_mask
      params:
          stack_ind: 66

    # Make a map of the full dataset
    - type: ch_pipeline.analysis.mapmaker.RingMapMaker
      requires: manager
      in: sstream_mask
      out: ringmap
      params:
        single_beam: true
        weight: natural
        exclude_intracyl: false
        include_auto: false
        save: true
        output_name: "ringmap_{{tag}}.h5"

    # Make a map from the inter cylinder baselines. This is less sensitive to
    # cross talk and emphasis point sources
    - type: ch_pipeline.analysis.mapmaker.RingMapMaker
      requires: manager
      in: sstream_mask
      out: ringmapint
      params:
        single_beam: true
        weight: natural
        exclude_intracyl: true
        include_auto: false
        save: true
        output_name: "ringmap_intercyl_{{tag}}.h5"

    # Mask out intercylinder baselines before beam forming to minimise cross
    # talk. This creates a copy of the input that shares the vis dataset (but
    # with a distinct weight dataset) to save memory
    - type: draco.analysis.flagging.MaskBaselines
      requires: manager
      in: sstream_mask
      out: sstream_inter
      params:
        share: vis
        mask_short_ew: 1.0

    # Load the source catalogs to measure fluxes of
    - type: draco.core.io.LoadBasicCont
      out: source_catalog
      params:
        files:
        - "{catalogs[0]}"
        - "{catalogs[1]}"
        - "{catalogs[2]}"

    # Measure the observed fluxes of the point sources in the catalogs
    - type: draco.analysis.beamform.BeamFormCat
      requires: [manager, sstream_inter]
      in: source_catalog
      params:
        timetrack: 60.0
        save: true
        output_name: "sourceflux_{{tag}}.h5"

    # Mask out day time data
    - type: ch_pipeline.analysis.flagging.DayMask
      in: sstream_mask
      out: sstream_mask1

    - type: ch_pipeline.analysis.flagging.MaskMoon
      in: sstream_mask1
      out: sstream_mask2

    # Remove ranges of time known to be bad that may effect the delay power
    # spectrum estimate
    - type: ch_pipeline.analysis.flagging.DataFlagger
      in: sstream_mask2
      out: sstream_mask3
      params:
        flag_type:
          - acjump
          - bad_calibration_acquisition_restart
          - bad_calibration_fpga_restart
          - bad_calibration_gains
          - decorrelated_cylinder
          - globalflag
          - rain1mm

    # Load the stack that we will blend into the daily data
    - type: draco.core.io.LoadBasicCont
      out: sstack
      params:
        files:
            - "{blend_stack_file}"
        selections:
            freq_range: [{freq[0]:d}, {freq[1]:d}]

    - type: draco.analysis.flagging.BlendStack
      requires: sstack
      in: sstream_mask3
      out: sstream_blend1
      params:
        frac: 1e-4

    # Mask the daytime data again. This ensures that we only see the point sources in
    # the delay spectrum we would expect
    - type: ch_pipeline.analysis.flagging.MaskDay
      in: sstream_blend1
      out: sstream_blend2

    # Estimate the delay power spectrum of the data. This is a good diagnostic
    # of instrument performance
    - type: draco.analysis.delay.DelaySpectrumEstimator
      requires: manager
      in: sstream_blend2
      params:
        freq_zero: 800.0
        nfreq: 1025
        nsamp: 40
        save: true
        output_name: "delayspectrum_{{tag}}.h5"
"""


class DailyProcessing(base.ProcessingType):
    """"""

    type_name = "daily"
    tag_pattern = r"\d+"

    # Parameters of the job processing
    default_params = {
        # Time range(s) to process
        "intervals": [
            ## Priority days to reprocess
            # Good ranges from rev_00
            {"start": "CSD1868", "end": "CSD1875"},
            {"start": "CSD1927", "end": "CSD1935"},
            {"start": "CSD1973", "end": "CSD1977"},
            {"start": "CSD2071", "end": "CSD2080"},
            {"start": "CSD2162", "end": "CSD2166"},
            # A good looking interval from late 2019 (determined from run
            # notes, dataflags and data availability)
            {"start": "CSD2143", "end": "CSD2148"},
            ## Runs processed for rev_00
            # October run
            {"start": "20181011T140000Z", "end": "20181019T220000Z"},
            # Winter run periods - as defined by Mateus
            # Pass A (start trimmed for timing solutions)
            {"start": "20181223T000000Z", "end": "20181229T000000Z"},
            # Pass B
            {"start": "20190111T000000Z", "end": "20190207T000000Z"},
            # Pass C (end trimmed for timing sols)
            {"start": "20190210T000000Z", "end": "20190304T000000Z"},
            # April run
            {"start": "20190406T000000Z", "end": "20190418T000000Z"},
            # July run
            {"start": "20190713T000000Z", "end": "20190723T000000Z"},
        ],
        # Amount of padding each side of sidereal day to load
        "padding": 0.02,
        # Frequencies to process
        "freq": [0, 1024],
        # The beam transfers to use (need to have the same freq range as above)
        "product_path": "/project/rpp-krs/chime/bt_empty/chime_4cyl_allfreq/",
        # Calibration times for thermal correction
        "caltimes_file": (
            "/project/rpp-krs/chime/chime_processed/gain/calibration_times/"
            "20180902_20200404_calibration_times.h5"
        ),
        # File for the timing correction
        "timing_file": (
            "/project/rpp-krs/chime/chime_processed/timing/rev_00/referenced/"
            "*_chimetiming_delay.h5"
        ),
        "catalogs": (
            "/project/rpp-krs/chime/chime_processed/catalogs/ps_cora_10Jy.h5",
            "/project/rpp-krs/chime/chime_processed/catalogs/ps_CGRaBS.h5",
            "/project/rpp-krs/chime/chime_processed/catalogs/ps_QSO_05Jy.h5",
        ),
        "blend_stack_file": (
            "/project/rpp-krs/chime/chime_processed/seth_tmp/stacks/rev_00/all/"
            "sidereal_stack.h5"
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

        # For each interval find and add all CSDs that have not already been added
        for interval in self._intervals:
            csd_i = csds_in_range(*interval)
            csd_set = set(csds)
            csds += [csd for csd in csd_i if csd not in csd_set]

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
            "intervals": [{"start": "20181224T000000Z", "end": "20181228T000000Z"}],
            "freq": [400, 416],
            "product_path": "/project/rpp-krs/chime/bt_empty/chime_4cyl_16freq/",
            "time": 60,  # How long in minutes?
            "nodes": 1,  # Number of nodes to use.
            "ompnum": 12,  # Number of OpenMP threads
            "pernode": 4,  # Jobs per node
        }
    )


def csds_in_range(start, end):
    """Get the CSDs within a time range.

    The start and end parameters must either be strings of the form "CSD\d+"
    (i.e. CSD followed by an int), which specifies an exact CSD start, or a
    form that `ephemeris.ensure_unix` understands.

    Parameters
    ----------
    start : str or parseable to datetime
        Start of interval.
    end : str or parseable to datetime
        End of interval. If `None` use now. Note that for CSD intervals the
        end is *inclusive* (unlike a `range`).

    Returns
    -------
    csds : list of ints
    """

    import math
    from ch_util import ephemeris

    if end is None:
        end = datetime.datetime.utcnow()

    if start.startswith("CSD"):
        start_csd = int(start[3:])
    else:
        start_csd = ephemeris.unix_to_csd(ephemeris.ensure_unix(start))
        start_csd = math.floor(start_csd)

    if end.startswith("CSD"):
        end_csd = int(end[3:])
    else:
        end_csd = ephemeris.unix_to_csd(ephemeris.ensure_unix(end))
        end_csd = math.ceil(end_csd)

    csds = [day for day in range(start_csd, end_csd + 1)]
    return csds
