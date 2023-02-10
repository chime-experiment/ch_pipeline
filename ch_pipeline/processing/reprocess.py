"""Make a delay spectrum from a sidereal stream. By default, this is the same as the standard
daily pipeline configuration, so the config for a revision should be modified as needed.
"""

from caput.tools import unique_ordered

from . import base
from . import daily
from . import client


class SiderealReprocessing(base.ProcessingType):
    """Baseclass for a reprocessing type.

    Reprocess from the SiderealStream product of a daily pipeline revision.

    A revision of this processing type is directly linked to an existing revision of the Daily
    processing type and pulls available files from there. By default, it is linked to the most
    recent Daily Processing revision, and can be changed in the revconfig file after creation.
    In theory, this will be used to quickly re-generate products from an already-processed
    Daily type, with custom changes made to the config.

    This is non-functional on its own and is designed to be subclassed.
    """

    # Use default DailyProcessing params
    default_params = daily.DailyProcessing.default_params.copy()

    # No script by default
    default_script = """ """

    def _available_tags(self):
        """Return all the tags that are available to run.

        This includes any that currently exist or are in the job queue.
        """

        # Get the csds which are available in the source daily revision
        csds_available = [int(i) for i in self.source_rev.ls()]
        # Get all unique csds from the config, maintaining order
        csds = unique_ordered(
            (csd for i in self._intervals for csd in daily.csds_in_range(*i))
        )
        # Return any tags which are both available and in the config
        tags = [f"{csd:.0f}" for csd in csds if csd in csds_available]

        return tags

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


DELAY_SCRIPT = """
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

    # Load the telescope model that we need for several steps
    - type: draco.core.io.LoadProductManager
      out: manager
      params:
        product_directory: "{product_path}"

    # Load the sidereal stream we want to process
    - type: draco.core.io.LoadFilesFromParams
      out: sstream
      params:
        files: "{src_type_path}/{csd}/sstream*{csd}.*"

    # Flag out low weight samples to remove transient RFI artifacts at the edges of
    # flagged regions
    - type: draco.analysis.flagging.ThresholdVisWeightBaselineAlt
      in: sstream
      out: sstream_tvwb
      params:
        relative_threshold: 0.5

    # Generate the second RFI mask using targeted knowledge of the instrument
    - type: draco.analysis.flagging.RFIMask
      in: sstream_tvwb
      out: rfimask2
      params:
        stack_ind: 66
        output_name: "rfi_mask2_{{tag}}.h5"
        save: true

    # Apply the RFI mask. This will modify the data in place.
    - type: draco.analysis.flagging.ApplyRFIMask
      in: [sstream_tvwb, rfimask2]
      out: sstream_mask

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
    # Also, mask out an additional frequency band which isn't that visible in
    # the data but generates a lot of high delay power
    - type: draco.analysis.flagging.MaskFreq
      in: sstream_blend3
      out: factmask
      params:
        factorize: true
        bad_freq_ind: [[738, 753]]

    # Apply the RFI mask. This will modify the data in place.
    - type: draco.analysis.flagging.ApplyRFIMask
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
"""


class DelaySpectrumReprocessing(SiderealReprocessing):
    """Generate a Delay Spectrum from a SiderealStream product of a daily revision

    This runs the basic pipeline config to generate both the Delay Spectrum and
    hpf Delay Spectrum from a given Sidereal Stream product.
    """

    type_name = "delay"
    tag_pattern = r"\d+"

    default_script = DELAY_SCRIPT

    def _update_default_params_hook(self):
        self.default_params.update(
            {
                "time": 60,
                "nodes": 8,
                "ompnum": 4,
                "pernode": 12,
            }
        )
        # Since we've copied the daily processing config, we can remove
        # any irrelevant items from the config
        del self.default_params["padding"]
        del self.default_params["caltimes_file"]
        del self.default_params["timing_file"]
        del self.default_params["freqmap_file"]
        del self.default_params["catalogs"]
