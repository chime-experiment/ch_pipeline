import math
import numpy as np
from datetime import datetime
import peewee as pw

import chimedb.core as db
import chimedb.data_index as di
import chimedb.dataflag as df
from ch_util import ephemeris
from caput.tools import unique_ordered
from ch_pipeline.processing import base

import logging

logger = logging.getLogger(__name__)

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

    - type: ch_pipeline.core.dataquery.ConnectDatabase
      params:
        timeout: 5
        ntries: 5

    # Query for all the data for the sidereal day we are processing
    - type: ch_pipeline.core.dataquery.QueryDatabase
      out: filelist
      params:
        start_csd: {csd[0]:.2f}
        end_csd: {csd[1]:.2f}
        accept_all_global_flags: true
        node_spoof:
          cedar_online: "/project/rpp-chime/chime/chime_online/"
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
      out: sensitivity_day
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

    # Calculate a RFI mask from the sensitivity metric
    - type: ch_pipeline.analysis.flagging.RFISensitivityMask
      in: sensitivity_day
      out: rfimask_sensitivity
      params:
        save: true
        output_name: "rfi_mask_sensitivity_{{tag}}.h5"

    # Calculate a RFI mask from Stokes I visibilities
    - type: ch_pipeline.analysis.flagging.RFIStokesIMask
      requires: manager
      in: tstream_dcm
      out: [rfimask_stokesi, _]
      params:
        save: true
        output_name:
          - "rfi_mask_stokesi_{{tag}}.h5"
          - "lowpass_power_2cyl_{{tag}}.h5"

    # Apply the StokesI RFI mask. This will modify the data in place.
    - type: draco.analysis.flagging.ApplyTimeFreqMask
      in: [tstream_dcm, rfimask_stokesi]
      out: tstream_day_rfi

    # Apply the Sensitivity RFI mask. This will modify the data in place.
    - type: draco.analysis.flagging.ApplyTimeFreqMask
      in: [tstream_day_rfi, rfimask_sensitivity]
      out: tstream_day_rfi2

    # Fully remove any frequencies which are mostly flagged. A threshold
    # of 0.3 (30%) generally only removes ~0.4-0.8% of additional data,
    # but has a noticeably positive effect on high-delay noise
    - type: draco.analysis.flagging.MaskFreq
      in: tstream_day_rfi2
      out: freq_mask
      params:
        freq_frac: 0.3
        save: true
        output_name: "rfi_mask_freq_{{tag}}.h5"

    # Apply the frequency mask
    - type: draco.analysis.flagging.ApplyTimeFreqMask
      in: [tstream_day_rfi2, freq_mask]
      out: tstream_day_freq_cut

    # Calculate the thermal gain correction
    - type: ch_pipeline.analysis.calibration.ThermalCalibration
      in: tstream_day_freq_cut
      out: thermal_gain
      params:
        caltime_path: "{caltimes_file}"

    # Apply the thermal correction
    - type: draco.core.misc.ApplyGain
      in: [tstream_day_freq_cut, thermal_gain]
      out: tstream_thermal_corrected
      params:
        inverse: false

    # Smooth the noise estimates which suffer from sample variance
    - type: draco.analysis.flagging.SmoothVisWeight
      in: tstream_thermal_corrected
      out: tstream_day_smoothweight

    # Apply an aggressive delay filter and
    # check consistency of data with noise at high delay.
    - type: draco.analysis.dayenu.DayenuDelayFilterFixedCutoff
      requires: manager
      in: tstream_day_smoothweight
      out: chisq_day_filtered
      params:
        tauw: 0.400
        single_mask: false
        atten_threshold: 0.0
        reduce_baseline: true
        mask_short: 20.0
        save: true
        output_name: "chisq_{{tag}}.h5"

    # Generate an RFI mask from the chi-squared test statistic.
    - type: ch_pipeline.analysis.flagging.RFIMaskChisqHighDelay
      in: chisq_day_filtered
      out: rfimask_chisq
      params:
        save: true
        output_name: "rfi_mask_chisq_{{tag}}.h5"

    # Apply the RFI mask. This will modify the data in place.
    - type: draco.analysis.flagging.ApplyTimeFreqMask
      in: [tstream_day_smoothweight, rfimask_chisq]
      out: tstream_day_rfi3

    # Regrid the data onto a regular grid in sidereal time
    - type: draco.analysis.sidereal.SiderealRebinner
      requires: manager
      in: tstream_day_rfi3
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

    # Load the stack used to calculate a binning gradient correction.
    # This is the same dataset used in blending, but we don't want to
    # keep it in memory while making ringmaps
    - type: draco.core.io.LoadBasicCont
      out: sstack_grad_fix
      params:
        files:
          - "{blend_stack_file}"
        selections:
          freq_range: [{freq[0]:d}, {freq[1]:d}]

    # Apply a gradient correction to the rebinned sidereal stream
    - type: draco.analysis.sidereal.RebinGradientCorrection
      requires: sstack_grad_fix
      in: sstream
      out: sstream_grad_fix

    # Make a map of the full dataset
    - type: draco.analysis.ringmapmaker.RingMapMaker
      requires: manager
      in: sstream_grad_fix
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
      in: sstream_grad_fix
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
      in: sstream_grad_fix
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

    # Wait until the catalog is loaded, otherwise this task will
    # run its setup method and significantly increase memory used
    - type: draco.core.misc.WaitUntil
      requires: source_catalog_nocollapse
      in: sstream_inter
      out: sstream_inter2

    # Measure the beamformed visibility as a function of hour angle
    - type: draco.analysis.beamform.BeamFormCat
      requires: [manager, sstream_inter2]
      in: source_catalog_nocollapse
      out: sourceflux_nocollapse
      params:
        timetrack: 300.0
        collapse_ha: false
        save: true
        output_name: "sourceflux_vs_ha_{{tag}}.h5"
        limit_outputs: 1

    # Wait until the first beam forming task is done in order to
    # avoid unnecessary memory usage
    - type: draco.core.misc.WaitUntil
      requires: sourceflux_nocollapse
      in: sstream_inter
      out: sstream_inter3

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
      requires: [manager, sstream_inter3]
      in: source_catalog
      params:
        timetrack: 300.0
        save: true
        output_name: "sourceflux_{{tag}}.h5"
        limit_outputs: 4

    # Mask out day time data
    - type: ch_pipeline.analysis.flagging.DayMask
      in: sstream_grad_fix
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

    # Find and flag periods of rainfall over 1mm
    - type: ch_pipeline.analysis.flagging.FlagRainfall
      in: sstream_mask3
      out: sstream_mask4
      params:
        accumulation_time: 30.0
        threshold: 1.0

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
      in: sstream_mask4
      out: sstream_blend1
      params:
        frac: 1e-4
        mask_freq: true

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
    # Also, mask out some additional frequencies that are not masked in the stack:
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
        save: true
        output_name: "rfi_mask_factorized_{{tag}}.h5"

    # Apply the RFI mask. This will modify the data in place.
    - type: draco.analysis.flagging.ApplyTimeFreqMask
      in: [sstream_blend3, factmask]
      out: sstream_blend4

    # Estimate the delay power spectrum of the data using the NRML
    # estimator. This is a good diagnostic of instrument performance
    - type: draco.analysis.delay.DelayPowerSpectrumStokesIEstimator
      requires: manager
      in: sstream_blend4
      params:
        freq_frac: 0.01
        time_frac: 0.01
        remove_mean: true
        freq_zero: 800.0
        nfreq: {nfreq_delay}
        nsamp: 100
        maxpost: true
        maxpost_tol: 1.0e-4
        complex_timedomain: true
        save: true
        output_name: "delayspectrum_{{tag}}.h5"

    # Use the gibbs estimator to produce a non-converged backup. This
    # estimator is slower to converge, but was used in the past, so
    # this will provide a good comparison with older data
    - type: draco.analysis.delay.DelayPowerSpectrumStokesIEstimator
      requires: manager
      in: sstream_blend4
      params:
        freq_frac: 0.01
        time_frac: 0.01
        remove_mean: true
        freq_zero: 800.0
        nfreq: {nfreq_delay}
        nsamp: 40
        maxpost: false
        complex_timedomain: true
        save: true
        output_name: "delayspectrum_gibbs_{{tag}}.h5"

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
    - type: draco.analysis.delay.DelayPowerSpectrumStokesIEstimator
      requires: manager
      in: sstream_dfilter
      params:
        freq_frac: 0.01
        time_frac: 0.01
        remove_mean: true
        freq_zero: 800.0
        nfreq: {nfreq_delay}
        nsamp: 100
        maxpost: true
        maxpost_tol: 1.0e-4
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
        # Fixed CSDS to ignore
        "flags": {},
        # Any days flagged by these flags are excluded from
        # the days to run
        "exclude_flags": [
            "no_timing_correction",
            "timing_correction_edge",
            "no_weather_data",
            "corrupted_file",
        ],
        # Maximum fraction of day flagged to exclude
        "frac_flagged": 0.8,
        # Amount of padding each side of sidereal day to load
        "padding": 0.02,
        # Minimum data coverage in order to process a day
        "required_coverage": 0.3,
        # Weather files are usuall only produced once per day, so
        # full coverage should generally be required
        "weather_coverage": 1.0,
        # Whether to look for offline data and request it be brought online
        "include_offline_files": True,
        # Number of recent days to prioritize in queue
        "num_recent_days_first": 7,
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
        # System modules to use/load
        "modpath": "/project/rpp-chime/chime/chime_env/modules/modulefiles",
        "modlist": "chime/python/2024.04",
        "nfreq_delay": 1025,
        # Job params
        "time": 150,  # How long in minutes?
        "nodes": 12,  # Number of nodes to use.
        "ompnum": 4,  # Number of OpenMP threads
        "pernode": 12,  # Jobs per node
    }
    default_script = DEFAULT_SCRIPT
    # Make sure not to remove chimestack files before this CSD
    daemon_config = {"keep_online": {"start": "CSD1800", "end": "CSD2650"}}

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
                self.default_params["frac_flagged"],
            )
        )

    def _load_hook(self):
        # Process the intervals
        self._intervals = []
        for t in self._revparams["intervals"]:
            self._intervals.append((t["start"], t.get("end", None), t.get("step", 1)))

        self._padding = self._revparams.get("padding", 0)
        self._min_coverage = self._revparams.get("required_coverage", 0.0)
        self._weather_coverage = self._revparams.get("weather_coverage", 1.0)
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
        # Get all desired csds from the config file
        csds = self._all_tags
        # Automatically exclude bad days
        csds = [csd for csd in csds if csd not in self._exclude]
        csds_sorted = sorted(csds)
        # Get a list of CSDS whose files are entirely available online
        csds_to_run = available_csds(
            csds_sorted,
            pad=self._padding,
            required_coverage=self._min_coverage,
            weather_coverage=self._weather_coverage,
        )

        tags = [f"{csd:.0f}" for csd in csds if csd in csds_to_run]

        return tags

    @property
    def _all_tags(self) -> list:
        """Return all tags desired from the config."""
        return unique_ordered(
            (csd for i in self._intervals for csd in expand_csd_range(*i))
        )

    def _finalise_jobparams(self, tag, jobparams):
        """Set bounds for this CSD."""

        csd = float(tag)
        jobparams.update({"csd": [csd - self._padding, csd + 1 + self._padding]})

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
        # Get all the tags requested by the config, including those that
        # are not currently available online
        all_tags = [str(tag) for tag in self._all_tags]

        if retrieve:
            # If there are any upcoming jobs which require offline data,
            # request that this be moved online
            exclude_tags = {
                *rev_stats["pending"],
                *rev_stats["running"],
                *rev_stats["successful"],
                *rev_stats["failed"],
            }
            all_tags = [tag for tag in all_tags if tag not in exclude_tags]
            # Search the next 20 tags and request any that we may want to be brought online.
            online_request_tags = sorted(
                [int(tag) for tag in all_tags[:20] if tag not in upcoming]
            )
            if online_request_tags:
                # Submit the request to bring files online
                nfiles["nretrieve"] = request_offline_csds(
                    online_request_tags, self._padding
                )

        if clear:
            # Clear out data that we don't need anymore
            remove_request_tags = sorted([int(tag) for tag in rev_stats["successful"]])
            exclude_tags = [
                *rev_stats["running"],
                *rev_stats["pending"],
                *rev_stats["failed"],
                *rev_stats["not_yet_submitted"],
            ] + list(expand_csd_range(*self.daemon_config["keep_online"].values()))
            exclude_tags = sorted([int(tag) for tag in exclude_tags])
            if remove_request_tags:
                # Submit the request to remove these files
                nfiles["nclear"] = remove_online_csds(
                    remove_request_tags, exclude_tags, self._padding
                )

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
        today = ephemeris.chime.get_current_lsd()
        to_run = [csd for csd in to_run if (today - float(csd)) > (1 + self._padding)]

        # Prioritize some number of recent days
        today = np.floor(today).astype(int)
        priority = [
            csd for csd in to_run if (today - int(csd)) <= self._num_recent_days
        ]

        if priority_only:
            return priority

        return priority + [csd for csd in to_run if csd not in priority]


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


def expand_csd_range(start, end, step=1):
    """Get the CSDs within a time range.

    The start and end parameters must either be strings of the form "CSD\d+"
    (i.e. CSD followed by an int), which specifies an exact CSD start, or a
    form that `ephemeris.ensure_unix` understands.

    Parameters
    ----------
    start : str or parseable to datetime
        Start of interval. If `None`, use CSD 0
    end : str or parseable to datetime
        End of interval. If `None` use now. Note that for CSD intervals the
        end is *inclusive* (unlike a `range`).

    Returns
    -------
    csds : list of ints
    """

    if start is None:
        start_csd = 0
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


def files_in_timespan(start, end, file_times):
    """Iterate over a list of file times and find those in a range.

    Given a list of file (name start, end) times, iterate over the list and
    find which files fall within start, end

    Parameters
    ----------
    start : float
        unix timestamp
    end : float
        unix timestamp
    file_times : list of tuples
        tuple: (name, start_time, finish_time)

    Returns
    -------
    list of elements of `files`, whose start_time and finish_time
    fall between `start` and `end`

    index of the first file whose `start_time` is after `end`
    """
    available = []
    for i, ts in enumerate(file_times):
        if (ts[1] < end) and (ts[2] >= start):
            available.append(ts)
        # files are in chronological order, so once we hit this, there are no files
        # further in the list which will fall within the time window
        elif ts[1] > end:
            return available, i

    return available, len(file_times) - 1


def available_csds(
    csds: list,
    pad: float = 0.0,
    required_coverage: float = 0.0,
    weather_coverage: float = 1.0,
) -> set:
    """Return the subset of csds in `csds` for whom all files are online.

    Parameters
    ----------
    csds
        sorted list of csds to check
    pad
      fraction of a day to pad on either end. This data should also be available
      in order for a day to be considered available
    required_coverage
      What fraction of the day must exist in order to be considered available.
      This *includes* the padding fraction, so values greater than 1 are
      permitted. For example, if `pad=0.2`, setting `required_coverage=1.4`
      would indicate that all the data must be available.
      Even if all files are online, if the data coverage is less than this
      fraction, the day shouldn't be processed.
    weather_coverage
        What fraction of the day must have weather coverage. Generally,
        weather files are only produced once per day, so this should be
        set to 1.0 unless there are changes to data format.

    Returns
    -------
    available
        set of all available csds
    """

    # Figure out which files exist online and which ones exist entirely
    # Repeat the process for corr data and weather data
    corr_online, corr_that_exist = db_get_corr_files_in_range(
        csds[0] - pad, csds[-1] + pad
    )
    weather_online, weather_that_exist = db_get_weather_files_in_range(
        csds[0] - pad, csds[-1] + pad
    )

    def _available(filenames_online, filenames_that_exist, coverage):
        available = set()
        coverage = coverage * 86400

        for csd in csds:
            start_time = ephemeris.csd_to_unix(csd - pad)
            end_time = ephemeris.csd_to_unix(csd + 1 + pad)

            # online - list of file start and end times that are online
            # between start_time and end_time
            # index_online - the final index in which data was located
            online, index_online = files_in_timespan(
                start_time, end_time, filenames_online
            )
            exists, index_exists = files_in_timespan(
                start_time, end_time, filenames_that_exist
            )

            if (len(online) == len(exists)) and (len(online) != 0):
                # All files for this CSD are online
                # Check that we have the required data coverage
                time_range = 0
                for tr in online:
                    time_range += tr[2] - tr[1]

                if time_range > coverage:
                    available.add(csd)

            # The final file in the span may contain more than one sidereal day
            index_online = max(index_online - 1, 0)
            index_exists = max(index_exists - 1, 0)

            filenames_online = filenames_online[index_online:]
            filenames_that_exist = filenames_that_exist[index_exists:]

        return available

    corr_available = _available(corr_online, corr_that_exist, required_coverage)
    weather_available = _available(weather_online, weather_that_exist, weather_coverage)

    return corr_available & weather_available


def db_get_corr_files_in_range(start_csd: int, end_csd: int):
    """Get the name and start and end times of corrdata files in the CSD range.

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
    start_time = ephemeris.csd_to_unix(start_csd)
    end_time = ephemeris.csd_to_unix(end_csd + 1)

    db.connect()

    chimestack_inst = di.ArchiveInst.get(name="chimestack")

    # Query for all chimestack files in the time range
    archive_files = (
        di.ArchiveFileCopy.select(
            di.ArchiveFileCopy.file,
            di.CorrFileInfo.start_time,
            di.CorrFileInfo.finish_time,
        )
        .join(di.ArchiveFile)
        .join(di.ArchiveAcq)
        .switch(di.ArchiveFile)
        .join(di.CorrFileInfo)
    ).where(
        di.ArchiveAcq.inst == chimestack_inst,
        di.CorrFileInfo.start_time < end_time,
        di.CorrFileInfo.finish_time >= start_time,
        di.ArchiveFileCopy.has_file == "Y",
    )
    # Figure out which files are online
    online_node = di.StorageNode.get(name="cedar_online", active=True)
    files_online = archive_files.where(di.ArchiveFileCopy.node == online_node)

    files_online = sorted([t for t in files_online.tuples()], key=lambda x: x[1])
    # files_that_exist might contain the same file multiple files
    # if it exists in multiple locations (nearline, online, gossec, etc)
    # we only want to include it once, so we initially create a set
    files_that_exist = sorted({t for t in archive_files.tuples()}, key=lambda x: x[1])

    return files_online, files_that_exist


def db_get_weather_files_in_range(start_csd: int, end_csd: int):
    """Get the name and start and end times of weather files in the CSD range.

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
    start_time = ephemeris.csd_to_unix(start_csd)
    end_time = ephemeris.csd_to_unix(end_csd + 1)

    db.connect()

    archive_files = (
        di.ArchiveFileCopy.select(
            di.ArchiveFileCopy.file,
            di.WeatherFileInfo.start_time,
            di.WeatherFileInfo.finish_time,
        )
        .join(di.ArchiveFile)
        .join(di.ArchiveAcq)
        .join(di.ArchiveInst)
        .switch(di.ArchiveFile)
        .join(di.WeatherFileInfo)
    ).where(
        di.ArchiveInst.name == "chime",
        di.WeatherFileInfo.start_time < end_time,
        di.WeatherFileInfo.finish_time >= start_time,
        di.ArchiveFileCopy.has_file == "Y",
    )
    # Figure out which files are online
    online_node = di.StorageNode.get(name="cedar_online", active=True)
    files_online = archive_files.where(di.ArchiveFileCopy.node == online_node)

    files_online = sorted([t for t in files_online.tuples()], key=lambda x: x[1])
    # files_that_exist might contain the same file multiple files
    # if it exists in multiple locations (nearline, online, gossec, etc)
    # we only want to include it once, so we initially create a set
    files_that_exist = sorted({t for t in archive_files.tuples()}, key=lambda x: x[1])

    return files_online, files_that_exist


def get_filenames_used_by_csds(csds: list, files: list, pad: float = 0):
    """For a list of CSDS, return all files from `files` which are used by the CSDS.

    This only returns the file names.

    Parameters
    ----------
    csds
        list of integer CSDS
    files
        list of files in format (name, start_time, end_time). This list
        is searched for files with times overlapping any CSD
    pad
        A fraction of a full CSD to pad when searching for files - i.e., a
        given CSD may want a small amount of data from adajacent days

    Returns
    -------
    files_used
        list of file names that are needed by the CSDS
    """
    files_used = set()
    for csd in csds:
        start_time = ephemeris.csd_to_unix(csd - pad)
        end_time = ephemeris.csd_to_unix(csd + 1 + pad)
        # Get all the files in this timespan which exist in `files`
        f, index = files_in_timespan(start_time, end_time, files)
        files_used.update(f)
        # Update the view of the list to reduce future iterations
        index = max(index - 1, 0)
        files = files[index:]

    return [f[0] for f in files_used]


def request_offline_csds(csds: list, pad: float = 0):
    """Given a list of csds, request that all required data be copied online.

    Request that all data required by the list of CSDS get copied to the
    cedar_online node.

    Parameters
    ----------
    csds
        list of integer csds to copy
    pad
        fraction of data from adjacent days that should also be copied online
    """

    def _make_copy_request(file, source, target):
        try:
            # Check if an activate request already exists. If so,
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

    # Figure out which chimestack files are needed
    online_files, files = db_get_corr_files_in_range(csds[0] - pad, csds[-1] + pad)
    # Only check files that are not online
    files = [f for f in files if f not in online_files]
    request_corr_files = get_filenames_used_by_csds(csds, files, pad)

    # Repeat for weather data. These are usually always online
    online_files, files = db_get_weather_files_in_range(csds[0] - pad, csds[-1] + pad)
    files = [f for f in files if f not in online_files]
    request_weather_files = get_filenames_used_by_csds(csds, files, pad)

    db.connect(read_write=True)

    target_node = di.StorageGroup.get(name="cedar_online")
    offline_node = di.StorageNode.get(name="cedar_nearline")
    smallfile_node = di.StorageNode.get(name="cedar_smallfile")

    nrequests = 0

    # Request chimestack files be brought back online
    for file_ in request_corr_files:
        nr = _make_copy_request(file_, offline_node, target_node)
        nrequests += nr

    # Request weather files be brought back online
    for file_ in request_weather_files:
        nr = _make_copy_request(file_, smallfile_node, target_node)
        nrequests += nr

    return nrequests


def remove_online_csds(csds_remove: list, csds_keep: list, pad: float = 0):
    """Remove online files which are solely used by specified csds.

    Check the files required by `csds_remove` and check against those used
    by `csds_keep`. Any files which are _only_ used by csds in `csds_remove`
    are removed from the `cedar_online` node, provided that a copy exists
    elsewhere.

    Parameters
    ----------
    csds_remove
        list of integer csds to clear data where possible
    csds_keep
        list of integer csds for which all data is still required
    pad
        fraction of data required by adjacent days. This will prevent
        a csd in `csds_remove` from clearing all its data if that fraction
        of data is needed by an adjacent csd in `csds_keep`
    """
    files, _ = db_get_corr_files_in_range(
        min(csds_remove[0], csds_keep[0]),
        max(csds_remove[-1], csds_keep[-1]),
    )
    # Get all the files we want to keep and remove. Choosing to
    # keep a file supercedes choosing to remove one
    keep_files = get_filenames_used_by_csds(csds_keep, files, pad)
    remove_files = get_filenames_used_by_csds(csds_remove, files, pad)
    remove_files = [file for file in remove_files if file not in keep_files]

    online_node = di.StorageNode.get(name="cedar_online")
    # Establish a read-write database connection
    db.connect(read_write=True)
    # Request that these files be removed from the online node
    di.ArchiveFileCopy.update(wants_file="N").where(
        di.ArchiveFileCopy.file << remove_files,
        di.ArchiveFileCopy.node == online_node,
    ).execute()

    return len(remove_files)


def get_flagged_csds(csds: list, flags: list, frac_flagged: float) -> dict:
    """Return a subset of csds which are heavily flagged.

    Return a set of csds from the input list for which more than
    `frac_flagged` of the data is flagged.

    Parameters
    ----------
    csds
        List of integer csds to check

    flags
        List of flag names to consider

    frac_flagged
        Fraction between 0 and 1 of flagged data for which to
        return a csd

    Returns
    -------
    flagged
        Subset of days in `csds` for which more than the threshold of
        data is flagged
    """
    # No point in doing database queries if there's nothing
    # to looks for
    if not csds or not flags:
        return {}

    out_flags = {}
    flagged_days = {}

    # Open a database connection
    db.connect()
    # Get flags from the database
    flag_types = df.DataFlagType.select()
    for ft in flag_types:
        if ft.name in flags:
            out_flags[ft.name] = list(
                df.DataFlag.select().where(df.DataFlag.type == ft)
            )
            flagged_days[ft.name] = []

    for name in flags:
        if name not in out_flags:
            logger.debug(f"Ignoring invalid flag {name}.")

    # Iterate over csds and flags to find which ones need to be excluded
    for csd in csds:
        start = ephemeris.csd_to_unix(csd)
        end = ephemeris.csd_to_unix(csd + 1)

        for flag_name, flag_list in out_flags.items():
            flagged_intervals = []
            # Iterate over the individual flags
            for flag in flag_list:
                # Iterate over flagged intervals for this flag
                if (start < flag.finish_time) & (end >= flag.start_time):
                    # some part of this day is flagged
                    overlap = False
                    for interval in flagged_intervals:
                        # Check if this flag overlaps with an already-flagged region
                        if (interval[0] < flag.finish_time) & (
                            interval[1] >= flag.start_time
                        ):
                            overlap = True
                            interval[0] = min(interval[0], flag.start_time)
                            interval[1] = max(interval[1], flag.finish_time)

                    if not overlap:
                        # This is part of a new interval
                        flagged_intervals.append(
                            [max(flag.start_time, start), min(flag.finish_time, end)]
                        )

            # Figure out what fraction of this day is flagged
            flagged_time = np.sum([t[1] - t[0] for t in flagged_intervals])
            if flagged_time > frac_flagged * (end - start):
                flagged_days[flag_name].append(f"CSD{csd}")

    # Ensure flagged CSDS are unique and sorted
    for key, val in flagged_days.items():
        flagged_days[key] = sorted(set(val))

    return flagged_days
