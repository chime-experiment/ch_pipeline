import math

import chimedb.core as db
import chimedb.data_index as di
from ch_util import ephemeris
from caput.tools import unique_ordered

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

    - type: draco.core.task.SetMPILogging
      params:
        level_rank0: DEBUG
        level_all: WARNING

    - type: draco.core.misc.CheckMPIEnvironment
      params:
        timeout: 420

    # Query for all the data for the sidereal day we are processing
    - type: ch_pipeline.core.dataquery.QueryDatabase
      out: filelist
      params:
        start_csd: {csd[0]:.2f}
        end_csd: {csd[1]:.2f}
        accept_all_global_flags: true
        node_spoof:
          cedar_online: "/project/rpp-chime/chime/chime_online/"
        connection_attempts: 5
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
      in: tstream
      out: sensitivity
      params:
        exclude_intracyl: true

    # Mask out any weights that are abnormally large or small, since these are likely
    # numerical issues and represent bad data
    - type: draco.analysis.flagging.SanitizeWeights
      in: tstream
      out: tstream_sanitized
      params:
        max_thresh: 1e30
        min_thresh: 1e-30

    # Identify individual baselines with much lower weights than expected
    - type: draco.analysis.flagging.ThresholdVisWeightBaseline
      requires: manager
      in: tstream_sanitized
      out: full_bad_baseline_mask
      params:
        average_type: "median"
        absolute_threshold: 1e-7
        relative_threshold: 1e-5
        pols_to_flag: "copol"

    # Collapse bad-baseline mask over baseline, so that any time-freq sample
    # with a low weight at any baseline is masked
    - type: draco.analysis.flagging.CollapseBaselineMask
      in: full_bad_baseline_mask
      out: bad_baseline_mask

    # Load the frequency map that was active when this data was collected
    - type: ch_pipeline.core.dataquery.QueryFrequencyMap
      in: tstream_sanitized
      out: freqmap
      params:
        cache: false

    # Identify decorrelated cylinders
    - type: ch_pipeline.analysis.flagging.MaskDecorrelatedCylinder
      in: [tstream_sanitized, inputmap, freqmap]
      out: decorr_cyl_mask
      params:
        threshold: 5.0
        max_frac_freq: 0.1

    # Average over redundant baselines across all cylinder pairs
    - type: draco.analysis.transform.CollateProducts
      requires: manager
      in: tstream_sanitized
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
      params:
        save: true
        output_name: "sensitivity_{{tag}}.h5"

    # Concatenate together all the masks for bad baselines
    - type: draco.analysis.sidereal.SiderealGrouper
      requires: manager
      in: bad_baseline_mask
      out: bad_baseline_mask_day
      params:
        save: true
        output_name: "bad_baseline_mask_{{tag}}.h5"

    # Concatenate together all the decorrelated cylinder masks
    - type: draco.analysis.sidereal.SiderealGrouper
      requires: manager
      in: decorr_cyl_mask
      out: decorr_cyl_mask_day

    # Expand the decorrelated cylinder mask along the time axis
    - type: ch_pipeline.analysis.flagging.ExpandMask
      in: decorr_cyl_mask_day
      out: decorr_cyl_mask_day_exp
      params:
        nexpand: 6
        in_place: true
        save: true
        output_name: "decorrelated_cylinder_mask_expanded_{{tag}}.h5"

    # Apply the mask from the bad baselines. This will modify the data in
    # place.
    - type: draco.analysis.flagging.ApplyTimeFreqMask
      in: [tstream_day, bad_baseline_mask_day]
      out: tstream_bbm

    # Apply the mask from the decorrelated cylinders. This will modify the data
    # in place.
    - type: draco.analysis.flagging.ApplyTimeFreqMask
      in: [tstream_bbm, decorr_cyl_mask_day_exp]
      out: tstream_dcm

    # Calculate the RFI mask from the autocorrelation data
    - type: ch_pipeline.analysis.flagging.RFIFilter
      in: tstream_dcm
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
        use_draco_container: true
        save: true
        output_name: "rfi_mask_{{tag}}.h5"
        nan_check: false

    # Apply the RFI mask. This will modify the data in place.
    - type: draco.analysis.flagging.ApplyTimeFreqMask
      in: [tstream_dcm, rfimask]
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
      params:
        inverse: false

    # Smooth the noise estimates which suffer from sample variance
    - type: draco.analysis.flagging.SmoothVisWeight
      in: tstream_thermal_corrected
      out: tstream_day_smoothweight

    # Regrid the data onto a regular grid in sidereal time
    - type: draco.analysis.sidereal.SiderealRegridder
      requires: manager
      in: tstream_day_smoothweight
      out: sstream
      params:
        samples: 4096

    # Precision truncate the sidereal stream data
    - type: draco.core.io.Truncate
      in: sstream
      out: sstream_trunc
      params:
        dataset:
          vis:
            weight_dataset: vis_weight
            variance_increase: 1.0e-3
          vis_weight: 1.0e-5

    # Save the truncated sidereal stream to a .zarr file and start a background
    # task to zip it
    - type: draco.core.io.SaveZarrZip
      in: sstream_trunc
      out: sstream_trunc_handle
      params:
        compression:
          vis:
            chunks: [32, 512, 512]
          vis_weight:
            chunks: [32, 512, 512]
        save: true
        output_name: "sstream_{{tag}}.zarr.zip"
        remove: true

    # Flag out low weight samples to remove transient RFI artifacts at the edges of
    # flagged regions
    - type: draco.analysis.flagging.ThresholdVisWeightBaseline
      requires: manager
      in: sstream
      out: full_tvwb_mask
      params:
        relative_threshold: 0.5
        ignore_absolute_threshold: -1
        average_type: "mean"
        pols_to_flag: "all"

    # Apply the tvwb mask. This will modify the data inplace.
    - type: draco.analysis.flagging.ApplyBaselineMask
      in: [sstream, full_tvwb_mask]
      out: sstream_tvwb

    # Generate the second RFI mask using targeted knowledge of the instrument
    - type: draco.analysis.flagging.RFIMask
      in: sstream_tvwb
      out: rfimask2
      params:
        stack_ind: 66
        output_name: "rfi_mask2_{{tag}}.h5"
        save: true

    # Apply the RFI mask. This will modify the data in place.
    - type: draco.analysis.flagging.ApplyTimeFreqMask
      in: [sstream_tvwb, rfimask2]
      out: sstream_mask

    # Make a map of the full dataset
    - type: draco.analysis.ringmapmaker.RingMapMaker
      requires: manager
      in: sstream_mask
      out: ringmap
      params:
        single_beam: true
        weight: natural
        exclude_intracyl: false
        include_auto: false

    # Precision truncate and write out the chunked normal ringmap
    - type: draco.core.io.Truncate
      in: ringmap
      out: ringmap_trunc
      params:
        dataset:
          map:
            weight_dataset: weight
            variance_increase: 1.0e-3
          weight: 1.0e-5

    # Save the truncated ringmap to a .zarr file and start a background
    # task to zip it
    - type: draco.core.io.SaveZarrZip
      in: ringmap_trunc
      out: ringmap_trunc_handle
      params:
        compression:
          map:
            chunks: [1, 1, 32, 512, 512]
          weight:
            chunks: [1, 32, 512, 512]
          dirty_beam:
            chunks: [1, 1, 32, 512, 512]
        save: true
        output_name: "ringmap_{{tag}}.zarr.zip"
        remove: true

    # Make a map from the inter cylinder baselines. This is less sensitive to
    # cross talk and emphasis point sources
    - type: draco.analysis.ringmapmaker.RingMapMaker
      requires: manager
      in: sstream_mask
      out: ringmap_int
      params:
        single_beam: true
        weight: natural
        exclude_intracyl: true
        include_auto: false

    # Precision truncate and write out the chunked intercylinder ringmap
    # NOTE: this cannot be combined with the above Truncate task as it would
    # result in both ringmaps existing in memory at the same time.
    - type: draco.core.io.Truncate
      in: ringmap_int
      out: ringmap_int_trunc
      params:
        dataset:
          map:
            weight_dataset: weight
            variance_increase: 1.0e-3
          weight: 1.0e-5

    # Save the truncated intercylinder ringmap to a .zarr file and start a background
    # task to zip it
    - type: draco.core.io.SaveZarrZip
      in: ringmap_int_trunc
      out: ringmap_int_trunc_handle
      params:
        compression:
          map:
            chunks: [1, 1, 32, 512, 512]
          weight:
            chunks: [1, 32, 512, 512]
          dirty_beam:
            chunks: [1, 1, 32, 512, 512]
        save: true
        output_name: "ringmap_intercyl_{{tag}}.zarr.zip"
        remove: true

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

    # Load the source catalogs to measure flux as a function of hour angle
    - type: draco.core.io.LoadBasicCont
      out: source_catalog_nocollapse
      params:
        files:
        - "{catalogs[0]}"

    # Measure the beamformed visibility as a function of hour angle
    - type: draco.analysis.beamform.BeamFormCat
      requires: [manager, sstream_inter]
      in: source_catalog_nocollapse
      out: sourceflux_nocollapse
      params:
        timetrack: 300.0
        collapse_ha: false
        save: true
        output_name: "sourceflux_vs_ha_{{tag}}.h5"

    # Wait until the additional catalog is loaded, otherwise this task will
    # run its setup method and significantly increase memory used
    - type: draco.core.misc.WaitUntil
      requires: sourceflux_nocollapse
      in: sstream_inter
      out: sstream_inter2

    # Load the source catalogs to measure fluxes of
    - type: draco.core.io.LoadBasicCont
      out: source_catalog
      params:
        files:
        - "{catalogs[0]}"
        - "{catalogs[1]}"
        - "{catalogs[2]}"
        - "{catalogs[3]}"

    # Measure the observed fluxes of the point sources in the catalogs
    - type: draco.analysis.beamform.BeamFormCat
      requires: [manager, sstream_inter2]
      in: source_catalog
      params:
        timetrack: 300.0
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

    # Mask the daytime data again. This ensures that we only see the time range in
    # the delay spectrum we would expect
    - type: ch_pipeline.analysis.flagging.MaskDay
      in: sstream_blend1
      out: sstream_blend2

    # Mask out the bright sources so we can see the high delay structure more easily
    - type: ch_pipeline.analysis.flagging.MaskSource
      in: sstream_blend2
      out: sstream_blend3
      params:
        source: ["CAS_A", "CYG_A", "TAU_A", "VIR_A"]

    # Try and derive an optimal time-freq factorizable mask that covers the
    # existing masked entries
    # Also, mask out some additional frequencies:
    # 506.25 - 511.71 MHz: band which isn't that visible in the data but generates
    # a lot of high delay power
    # 519.14 - 525.4 MHz: unmasked band in the stack that seems to generate a lot
    # of power across all delays
    # 601.56 - 608.59 MHz: sporadic band which generates some power
    # 459.38 - 465.23 MHz: another high-power band in the stack
    # 609.77 - 610.55 MHz: narrow band in stack creating artifacts in delay-filtered
    # ringmaps 
    - type: draco.analysis.flagging.MaskFreq
      in: sstream_blend3
      out: factmask
      params:
        factorize: true
        bad_freq_ind: [[738, 753], [703, 720], [490, 509], [857, 873], [485, 488]]

    # Apply the RFI mask. This will modify the data in place.
    - type: draco.analysis.flagging.ApplyTimeFreqMask
      in: [sstream_blend3, factmask]
      out: sstream_blend4

    # Estimate the delay power spectrum of the data. This is a good diagnostic
    # of instrument performance
    - type: draco.analysis.delay.DelaySpectrumEstimator
      requires: manager
      in: sstream_blend4
      params:
        freq_zero: 800.0
        nfreq: {nfreq_delay}
        nsamp: 40
        complex_timedomain: true
        save: true
        output_name: "delayspectrum_{{tag}}.h5"

    # Apply delay filter to stream
    - type: draco.analysis.delay.DelayFilter
      requires: manager
      in: sstream_blend4
      out: sstream_dfilter
      params:
        delay_cut: 0.1
        za_cut: 1.0
        window: true

    # Estimate the delay power spectrum of the data after applying
    # the delay filter
    - type: draco.analysis.delay.DelaySpectrumEstimator
      requires: manager
      in: sstream_dfilter
      params:
        freq_zero: 800.0
        nfreq: {nfreq_delay}
        nsamp: 40
        complex_timedomain: true
        save: true
        output_name: "delayspectrum_hpf_{{tag}}.h5"

    # Make an intercylinder ringmap from the delay-filtered visibilities,
    - type: draco.analysis.ringmapmaker.RingMapMaker
      requires: manager
      in: sstream_dfilter
      out: ringmap_int_hpf
      params:
        single_beam: true
        weight: natural
        exclude_intracyl: true
        include_auto: false

    # Select 10 frequencies from the delay-filtered map that are useful for validation
    - type: draco.analysis.transform.SelectFreq
      in: ringmap_int_hpf
      out: ringmap_int_hpf_sel
      params:
        channel_index: {val_freq}

    # Precision truncate and write out the chunked filtered ringmap.
    # Don't truncate the map itself, to preserve low-amplitude pixels.
    - type: draco.core.io.Truncate
      in: ringmap_int_hpf_sel
      out: ringmap_int_hpf_sel_trunc
      params:
        dataset:
          map: No
          weight: 1.0e-5

    # Save the truncated filtered intercylinder ringmap to a .zarr file and
    # start a background task to zip it
    - type: draco.core.io.SaveZarrZip
      in: ringmap_int_hpf_sel_trunc
      out: ringmap_int_hpf_sel_trunc_handle
      params:
        compression:
          map:
            chunks: [1, 1, 32, 512, 512]
          weight:
            chunks: [1, 32, 512, 512]
          dirty_beam:
            chunks: [1, 1, 32, 512, 512]
        save: true
        output_name: "ringmap_intercyl_hpf_{{tag}}.zarr.zip"
        remove: true

    # Downselect the ringmap to keep only the XX and YY pols
    - type: draco.analysis.transform.Downselect
      in: ringmap_int_hpf
      out: ringmap_int_hpf_xx_yy
      params:
        selections:
          pol_index: [0, 3]

    # Take the variance of the map across elevation. Apply weights
    # as a mask only
    - type: draco.analysis.transform.ReduceVar
      in: ringmap_int_hpf_xx_yy
      params:
        axes:
          - el
        dataset: map
        weighting: weighted
        compression:
          map:
            chunks: [1, 1, 32, 512, 512]
          weight:
            chunks: [1, 32, 512, 512]
        save: true
        output_name: "ringmap_intercyl_el_var_{{tag}}.h5"

    # Take the variance of the map across frequency. Apply weights
    # as a mask only
    - type: draco.analysis.transform.ReduceVar
      in: ringmap_int_hpf_xx_yy
      params:
        axes:
          - freq
        dataset: map
        weighting: weighted
        compression:
          map:
            chunks: [1, 1, 32, 512, 512]
          weight:
            chunks: [1, 32, 512, 512]
        save: true
        output_name: "ringmap_intercyl_freq_var_{{tag}}.h5"

    # Wait for all the zarr zipping tasks to complete
    # Wait for the sstream last since it will likely take the
    # longest to complete
    - type: draco.core.io.WaitZarrZip
      in: ringmap_trunc_handle

    - type: draco.core.io.WaitZarrZip
      in: ringmap_int_trunc_handle

    - type: draco.core.io.WaitZarrZip
      in: ringmap_int_hpf_sel_trunc_handle

    - type: draco.core.io.WaitZarrZip
      in: sstream_trunc_handle
"""


class DailyProcessing(base.ProcessingType):
    """ """

    type_name = "daily"
    tag_pattern = r"\d+"

    # Parameters of the job processing
    default_params = {
        # Time range(s) to process
        "intervals": [
            # Two short two-day intervals either side of the caltime change
            # (1878, 3000), one with the weird 4 baseline issue (1912), one
            # with no actual data (1898), one with a single baseline-freq issue
            # (1960), one to test the thermal correction which otherwise shows
            # a large spread (1965), one with a different set of vertical
            # stripes (1983), and one with a decorrelated cylinder (2325) in
            # order to have a few good test days at the very start
            # NOTE: these intervals are *inclusive*
            {"start": "CSD1878", "end": "CSD1879"},
            {"start": "CSD1898", "end": "CSD1898"},
            {"start": "CSD1912", "end": "CSD1912"},
            {"start": "CSD1960", "end": "CSD1960"},
            {"start": "CSD1965", "end": "CSD1965"},
            {"start": "CSD1983", "end": "CSD1983"},
            {"start": "CSD2325", "end": "CSD2325"},
            {"start": "CSD3000", "end": "CSD3001"},
            # Good short ranges from rev_00, these are spread over the year and
            # quickly cover the full sky
            {"start": "CSD1868", "end": "CSD1875"},
            {"start": "CSD1927", "end": "CSD1935"},
            {"start": "CSD1973", "end": "CSD1977"},
            {"start": "CSD2071", "end": "CSD2080"},
            {"start": "CSD2162", "end": "CSD2166"},
            # Intervals covering the days used in the stacking analysis
            {"start": "CSD1878", "end": "CSD1939"},
            {"start": "CSD1958", "end": "CSD1990"},
            {"start": "CSD2012", "end": "CSD2013"},
            {"start": "CSD2029", "end": "CSD2046"},
            {"start": "CSD2059", "end": "CSD2060"},
            {"start": "CSD2072", "end": "CSD2143"},
            # Intervals for which we process only one day every 7 days, this
            # will slowly build up coverage over the whole timespan, extending
            # as new days become available
            {"start": "CSD1878", "step": 7},
            {"start": "CSD1879", "step": 7},
            {"start": "CSD1880", "step": 7},
            {"start": "CSD1881", "step": 7},
            {"start": "CSD1882", "step": 7},
            {"start": "CSD1883", "step": 7},
            {"start": "CSD1884", "step": 7},
        ],
        # Amount of padding each side of sidereal day to load
        "padding": 0.02,
        # Number of recent days to prioritize in queue
        "num_recent_days_first": 0,
        # Frequencies to process
        "freq": [0, 1024],
        # Frequencies to save for validation ringmaps
        "val_freq": [65, 250, 325, 399, 470, 605, 730, 830, 950, 990],
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
        # Delay spectrum estimation
        "blend_stack_file": (
            "/project/rpp-chime/chime/chime_processed/stacks/rev_03/all/sstack.h5"
        ),
        "nfreq_delay": 1025,
        # Job params
        "time": 100,  # How long in minutes?
        "nodes": 12,  # Number of nodes to use.
        "ompnum": 4,  # Number of OpenMP threads
        "pernode": 12,  # Jobs per node
    }
    default_script = DEFAULT_SCRIPT

    def _load_hook(self):
        # Process the intervals
        self._intervals = []
        for t in self._revparams["intervals"]:
            self._intervals.append((t["start"], t.get("end", None), t.get("step", 1)))

        self._padding = self._revparams["padding"]
        self._num_recent_days = self._revparams.get("num_recent_days_first", 0)

    def _available_files(self, start_csd, end_csd):
        """
        Return chimestack files available in cedar_online between start_csd and
        end_csd, if all of the files for that period are available online.

        Return an empty list if files between start_csd and end_csd are only
        partially available online.

        Total file count is verified by checking files that exist everywhere.

        Parameters
        ----------
        start_csd : int
            Start date in sidereal day format
        end_csd : int
            End date in sidereal day format

        Returns
        -------
        list
            List contains the chimestack files available in the timespan, if
            all of them are available online
        """

        # Connect to databases
        db.connect()

        # Get timestamps in unix format
        # Needed for queries
        start_time = ephemeris.csd_to_unix(start_csd)
        end_time = ephemeris.csd_to_unix(end_csd)

        # We will want to know which files are in chime_online and nearline on cedar
        online_node = di.StorageNode.get(name="cedar_online", active=True)
        chimestack_inst = di.ArchiveInst.get(name="chimestack")

        # TODO: if the time range is so small that it’s completely contained
        # within a single file, nothing will be returned have to special-case
        # it by looking for files which start before the start time and end
        # after the end time).
        archive_files = (
            di.ArchiveFileCopy.select(
                di.CorrFileInfo.start_time,
                di.CorrFileInfo.finish_time,
            )
            .join(di.ArchiveFile)
            .join(di.ArchiveAcq)
            .switch(di.ArchiveFile)
            .join(di.CorrFileInfo)
        )

        # chimestack files available online which include between start and end_time
        files_that_exist = archive_files.where(
            di.ArchiveAcq.inst
            == chimestack_inst,  # specifically looking for chimestack files
            di.CorrFileInfo.start_time
            < end_time,  # which contain data that includes start time and end time
            di.CorrFileInfo.finish_time >= start_time,
            di.ArchiveFileCopy.has_file == "Y",
        )

        files_online = files_that_exist.where(
            di.ArchiveFileCopy.node == online_node,  # that are online
        )

        filenames_online = sorted([t for t in files_online.tuples()])

        # files_that_exist might contain the same file multiple files
        # if it exists in multiple locations (nearline, online, gossec, etc)
        # we only want to include it once
        filenames_that_exist = sorted({t for t in files_that_exist.tuples()})

        return filenames_online, filenames_that_exist

    def _available_tags(self):
        """Return all the tags that are available to run.

        This includes any that currently exist or are in the job queue.
        """
        # Lets us get only unique csds in the order we want them
        # with a single pass.
        csds = unique_ordered(
            # This is a generator
            (csd for i in self._intervals for csd in csds_in_range(*i))
        )
        csds_sorted = sorted(csds)

        # grab the list of files that are online, and that exist anywhere,
        # from the earliest csd to the latest
        filenames_online, filenames_that_exist = self._available_files(
            csds_sorted[0], csds_sorted[-1] + 1
        )
        # only queue jobs for which all data is available online in chime_online
        csds_available = self._csds_available_data(
            csds_sorted, filenames_online, filenames_that_exist
        )

        tags = [f"{csd:.0f}" for csd in csds if csd in csds_available]

        return tags

    def _csds_available_data(self, csds, filenames_online, filenames_that_exist) -> set:
        """
        Return the subset of csds in `csds` for whom all files are online.

        `filenames_online` and `filenames_that_exist` are a list of tuples
        (start_time, finish_time)

        All 3 input lists should be sorted.
        """
        csds_available = set()

        for csd in csds:
            start_time = ephemeris.csd_to_unix(csd)
            end_time = ephemeris.csd_to_unix(csd + 1)

            # online - list of filenames that are online between start_time and end_time
            # index_online, the final index in which data was located
            online, index_online = self._files_in_timespan(
                start_time, end_time, filenames_online
            )
            exists, index_exists = self._files_in_timespan(
                start_time, end_time, filenames_that_exist
            )

            if (len(online) == len(exists)) and (len(online) != 0):
                csds_available.add(csd)

            # The final file in the span may contain more than one sidereal day
            index_online = max(index_online - 1, 0)
            index_exists = max(index_exists - 1, 0)

            filenames_online = filenames_online[index_online:]
            filenames_that_exist = filenames_that_exist[index_exists:]

        return csds_available

    def _files_in_timespan(self, start, end, files):
        """
        Parameters
        ----------
        start : float
            unix timestamp
        end : float
            unix timestamp
        files : list of tuple
            tuple: (start_time, finish_time)

        Returns
        -------
        list of elements of `files`, whose start_time and finish_time
        fall between `start` and `end`

        index of the first file whose `start_time` is after `end`
        """
        available = []
        for i in range(0, len(files)):
            f = files[i]
            if (f[0] < end) and (f[1] >= start):
                available.append(f)
            # files are in chronological order
            # once we hit this conditional, there are no files
            # further in the list, which will fall within
            # our timewindow
            elif f[0] > end:
                return available, i
        return available, len(files) - 1

    def _finalise_jobparams(self, tag, jobparams):
        """Set bounds for this CSD."""

        csd = float(tag)
        jobparams.update({"csd": [csd - self._padding, csd + 1 + self._padding]})

        return jobparams

    def _generate_hook(self, user=None):
        to_run = self.status(user=user)["not_yet_submitted"]

        buffer = 2
        today = math.floor(ephemeris.chime.get_current_lsd()) - buffer

        # Remove any days which are within the buffer window
        to_run = [csd for csd in to_run if today - int(csd) > 0]
        # Get any recent tags to run first
        priority = [csd for csd in to_run if today - int(csd) <= self._num_recent_days]
        to_run = priority + [csd for csd in to_run if csd not in priority]

        return to_run


class TestDailyProcessing(DailyProcessing):
    """A test version of the daily processing.

    Processes only 16 frequencies on a single node.
    """

    type_name = "test_daily"

    # Override params above
    default_params = DailyProcessing.default_params.copy()
    default_params.update(
        {
            "intervals": [
                # 1878 and 1885 have files available online
                {"start": "CSD1878", "end": "CSD1889", "step": 7},
                {"start": "CSD3000", "end": "CSD3014", "step": 7},
                {"start": "20181224T000000Z", "end": "20181228T000000Z"},
            ],
            "freq": [400, 416],
            "val_freq": [2, 3, 7],
            "nfreq_delay": 17,
            "product_path": "/project/rpp-chime/chime/bt_empty/chime_4cyl_16freq/",
            "time": 60,  # How long in minutes?
            "nodes": 1,  # Number of nodes to use.
            "ompnum": 12,  # Number of OpenMP threads
            "pernode": 4,  # Jobs per node
        }
    )


def csds_in_range(start, end, step=1):
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

    if start.startswith("CSD"):
        start_csd = int(start[3:])
    else:
        start_csd = ephemeris.unix_to_csd(ephemeris.ensure_unix(start))
        start_csd = math.floor(start_csd)

    if end is None:
        end_csd = int(ephemeris.chime.get_current_lsd())
    elif end.startswith("CSD"):
        end_csd = int(end[3:])
    else:
        end_csd = ephemeris.unix_to_csd(ephemeris.ensure_unix(end))
        end_csd = math.ceil(end_csd)

    csds = [day for day in range(start_csd, end_csd + 1, step)]
    return csds
