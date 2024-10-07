"""Chime Full Stack processing type.

Combines quarterstacks into a full sidereal day.

Classes
=======
- :py:class:`FullStackProcessing`
"""

from typing import ClassVar

from . import base, client

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

inputs: &inputs
{inputs}

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

    - type: draco.core.io.LoadProductManager
      out: manager
      params:
        product_directory: "{product_path}"

    - type: draco.core.io.LoadBasicCont
      out: sstream
      params:
        files: *inputs
        selections:
          freq_range: [{freq[0]:d}, {freq[1]:d}]

    # Apply a gradient correction fix for rebinned data. If the sstream
    # dataset has already been corrected, or if it uses a different
    # gridding method, this task behaves as a no-op
    - type: draco.analysis.sidereal.RebinGradientCorrection
      requires: sstream
      in: sstream
      out: sstream_grad

    - type: draco.analysis.sidereal.{stacker_instance}
      in: sstream_grad
      out: fstack
      params:
        tag: "fullstack"

    - type: draco.core.io.Truncate
      in: fstack
      out: fstack_trunc
      params:
        dataset:
          vis:
            weight_dataset: vis_weight
            variance_increase: 1.0e-4
          vis_weight: 1.0e-5

    - type: draco.core.io.SaveZarrZip
      in: fstack_trunc
      out: fstack_trunc_handle
      params:
        compression:
          vis:
            chunks: [32, 512, 2048]
          vis_weight:
            chunks: [32, 512, 2048]
        save: true
        output_name: "sstack.zarr.zip"
        remove: true

    # Block the pipeline until the stack is written out
    - type: draco.core.misc.WaitUntil
      requires: fstack_trunc_handle
      in: fstack
      out: fstack2

    - type: draco.analysis.ringmapmaker.RingMapMaker
      requires: manager
      in: fstack2
      out: ringmap
      params:
        single_beam: true
        weight: natural
        weight_ew: natural
        exclude_intracyl: false
        include_auto: false
        npix: 1024

    - type: draco.core.io.Truncate
      in: ringmap
      out: ringmap_trunc
      params:
        dataset:
          map:
            weight_dataset: weight
            variance_increase: 1.0e-4
          vis_weight: 1.0e-5

    - type: draco.core.io.SaveZarrZip
      in: ringmap_trunc
      out: ringmap_trunc_handle
      params:
        compression:
          map:
            chunks: [1, 1, 32, 512, 2048]
          weight:
            chunks: [1, 32, 512, 2048]
          dirty_beam:
            chunks: [1, 1, 32, 512, 2048]
        save: true
        output_name: "ringmap.zarr.zip"
        remove: true

    - type: draco.analysis.ringmapmaker.RingMapMaker
      requires: manager
      in: fstack2
      out: ringmap_int
      params:
        single_beam: true
        weight: natural
        weight_ew: natural
        exclude_intracyl: true
        include_auto: false
        npix: 1024

    - type: draco.core.io.Truncate
      in: ringmap_int
      out: ringmap_int_trunc
      params:
        dataset:
          map:
            weight_dataset: weight
            variance_increase: 1.0e-4
          vis_weight: 1.0e-5

    - type: draco.core.io.SaveZarrZip
      in: ringmap_int_trunc
      out: ringmap_int_trunc_handle
      params:
        compression:
          map:
            chunks: [1, 1, 32, 512, 2048]
          weight:
            chunks: [1, 32, 512, 2048]
          dirty_beam:
            chunks: [1, 1, 32, 512, 2048]
        save: true
        output_name: "ringmap_intercyl.zarr.zip"
        remove: true

    # Mask out the bright sources so we can see the high delay structure more easily
    - type: ch_pipeline.analysis.flagging.MaskSource
      in: fstack2
      out: fstack_flag_src
      params:
        source: ["CAS_A", "CYG_A", "TAU_A", "VIR_A"]

    # Try and derive an optimal time-freq factorizable mask that covers the
    # existing masked entries
    - type: draco.analysis.flagging.MaskFreq
      in: fstack_flag_src
      out: factmask
      params:
        factorize: true
        save: true
        output_name: "factorized_mask.h5"

    # Apply the RFI mask. This will modify the data in place.
    - type: draco.analysis.flagging.ApplyTimeFreqMask
      in: [fstack_flag_src, factmask]
      out: fstack_factmask

    - type: draco.analysis.delay.DelayPowerSpectrumStokesIEstimator
      requires: manager
      in: fstack_factmask
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
        output_name: "delayspectrum.h5"

    - type: draco.analysis.delay.DelayFilter
      requires: manager
      in: fstack_factmask
      out: fstack_lf
      params:
        delay_cut: 0.1
        za_cut: 1.0
        window: true

    - type: draco.analysis.delay.DelaySpectrumEstimator
      requires: manager
      in: fstack_lf
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
        output_name: "delayspectrum_hpf.h5"

    - type: draco.core.io.WaitZarrZip
      in: fstack_trunc_handle

    - type: draco.core.io.WaitZarrZip
      in: ringmap_trunc_handle

    - type: draco.core.io.WaitZarrZip
      in: ringmap_int_trunc_handle
"""


class FullStackProcessing(base.ProcessingType):
    """Stacks days or quarterstacks into a full stack.

    This creates a revision based of off another revision type with existing
    data to stack. This can be either a quarterstack revision or a daily revision,
    the latter of which can include either standard regridded days or semi-regridded
    dirty days. The type of input data affects whic stacker is used.
    """

    type_name = "fullstack"
    tag_pattern = r"stack((_p\d){1}|_all|$)"

    # Parameters of the job processing
    default_params: ClassVar = {
        # Frequencies to proces
        "freq": [0, 1024],
        "nfreq_delay": 1025,
        # The beam transfers to use (need to have the same freq range as above)
        "product_path": "/project/rpp-chime/chime/bt_empty/chime_4cyl_allfreq/",
        # System modules to use/load
        "modpath": "/project/rpp-chime/chime/chime_env/modules/modulefiles",
        "modlist": "chime/python/2024.04",
        # Job params
        "time": 45,  # Time in minutes
        "nodes": 16,  # Number of nodes to use
        "ompnum": 6,  # Number of OpenMP threads
        "pernode": 8,  # Jobs per node
    }

    default_script = DEFAULT_SCRIPT

    def _create_hook(self):
        """Create the revision.

        This looks at the input revision and determines which stacking method
        to use accordingly. Right now, only quarterstacks are supported.
        """
        # Prompt the user to provide a source revision path
        input_rev = input("Enter the input revision <type>:<rev>:")
        # Load the source revision
        input_rev = client.PRev().convert(input_rev, None, None)
        # Set the input type
        input_type = input_rev.type_name

        # Select the stacking class to use based on the input type
        if input_type == "quarterstack":
            self.default_params["stacker_instance"] = "SiderealStackerMatch"
        elif input_type == "daily":
            self.default_params["stacker_instance"] = "SiderealStackerDeconvolve"
        else:
            raise NotImplementedError(
                f"Stacking from type {input_type} is not currently supported."
            )

        # Set the default parameters for this revision
        self.default_params["input_type"] = input_type
        self.default_params["input_rev"] = input_rev.revision

        stacks = {"stack_all": []}
        inputs = {}

        for tag in input_rev.ls():
            # Add all tags to the full stack
            stacks["stack_all"].append(tag)
            # Construct the full file path and add to inputs
            inputs[tag] = str(input_rev.base_path) + "/" + tag

            if input_type == "quarterstack":
                # Figure out which partition these belong to
                _, _, part = input_rev._parse_tag(tag)
                # Add this tag to the partition stack
                stacks.setdefault(f"p{part}", []).append(tag)

        # Store the full file path for each input to this revision
        self.default_params["inputs"] = inputs
        self.default_params["stacks"] = stacks

    def _available_tags(self):
        """Return all the tags that are available to run.

        Returns the full stack and each partition stack.
        """
        return list(self._revparams["stacks"].keys())

    def _finalise_jobparams(self, tag, jobparams):
        """Modify the job parameters before the final config is made."""
        inputs = self._revparams["stacks"][tag]
        paths = self._revparams["inputs"]
        input_type = self._revparams["input_type"]

        if input_type == "quarterstack":
            input_list_str = "\n" + "\n".join(
                [f"- {paths[t]}/sstack.zarr.zip" for t in inputs]
            )
        elif input_type == "daily":
            input_list_str = "\n" + "\n".join(
                [f"- {paths[t]}/sstream_lsd_{t}.zarr.zip" for t in inputs]
            )

        jobparams.update({"inputs": input_list_str})

        return jobparams
