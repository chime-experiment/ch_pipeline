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

    # Apply a gradient correction fix for rebinned data
    - type: draco.analysis.sidereal.RebinGradientFix
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
            variance_increase: 1.0e-3
          vis_weight: 1.0e-5

    - type: draco.core.io.SaveZarrZip
      in: fstack_trunc
      out: fstack_trunc_handle
      params:
        compression:
          vis:
            chunks: [32, 512, 512]
          vis_weight:
            chunks: [32, 512, 512]
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
      params:
        single_beam: true
        weight: natural
        weight_ew: natural
        exclude_intracyl: false
        include_auto: false
        save: true
        output_name: "ringmap.h5"
        npix: 1024

    - type: draco.analysis.ringmapmaker.RingMapMaker
      requires: manager
      in: fstack2
      params:
        single_beam: true
        weight: natural
        weight_ew: natural
        exclude_intracyl: true
        include_auto: false
        save: true
        output_name: "ringmap_intercyl.h5"
        npix: 1024

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
        output_name: "fact_mask.h5"

    # Apply the RFI mask. This will modify the data in place.
    - type: draco.analysis.flagging.ApplyTimeFreqMask
      in: [fstack_flag_src, factmask]
      out: fstack_factmask

    - type: draco.analysis.delay.DelaySpectrumEstimator
      requires: manager
      in: fstack_factmask
      params:
        freq_zero: 800.0
        nfreq: {nfreq_delay}
        nsamp: 200
        complex_timedomain: true
        save: true
        output_name: "dspec.h5"

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
        freq_zero: 800.0
        nfreq: {nfreq_delay}
        nsamp: 200
        complex_timedomain: true
        save: true
        output_name: "dspec_hpf.h5"

    - type: draco.core.io.WaitZarrZip
      in: fstack_trunc_handle
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
    default_params = {
        # Frequencies to proces
        "freq": [0, 1024],
        # The beam transfers to use (need to have the same freq range as above)
        "product_path": "/project/rpp-chime/chime/bt_empty/chime_4cyl_allfreq/",
        "nfreq_delay": 1025,
        # Job params
        "time": 180,  # Time in minutes
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
        # Set the input type. Right now, only `quarterstack` is supported
        input_type = input_rev.type_name

        # Select the stacking class to use based on the input type
        if input_type == "quarterstack":
            self.default_params["stacker_instance"] = "SiderealStackerMatch"
        else:
            self.default_params["stacker_instance"] = None
            raise NotImplementedError(
                "Stacking directly from individual days is not currently supported."
            )

        # Set the default parameters for this revision
        self.default_params["input_type"] = input_type
        self.default_params["input_rev"] = input_rev.revision

        stacks = {"stack_all": []}
        inputs = {}

        for tag in input_rev.ls():
            # Figure out which partition these belong to
            _, _, part = input_rev._parse_tag(tag)
            # Add this tag to the full stack and partition stack
            stacks["stack_all"].append(tag)
            stacks.setdefault(f"p{part}", []).append(tag)
            # Construct the full file path and add to inputs
            inputs[tag] = str(input_rev.base_path) + "/" + tag

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

        input_list_str = "\n" + "\n".join(
            [f"- {paths[t]}/sstack.zarr.zip" for t in inputs]
        )

        jobparams.update({"inputs": input_list_str})

        return jobparams
