"""Chime Full Stack processing type.

Combines quarterstacks into a full sidereal day.
"""

import os
from typing import ClassVar

from . import base, client

DEFAULT_SCRIPT = """
# Cluster configuration
cluster:
  name: {jobname}
  account: rpp-chime

  directory: {dir}
  temp_directory: {tempdir}

  time: {time}
  system: slurm
  nodes: {nodes}
  ompnum: {ompnum}
  pernode: {pernode}
  mem: 768000M

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
    - fluxcat
    - numpy
    - scipy
    - h5py
    - mpi4py

  tasks:

    - type: caput.pipeline.tasklib.base.SetMPILogging
      params:
        level_rank0: DEBUG
        level_all: WARNING

    - type: caput.pipeline.tasklib.debug.CheckMPIEnvironment
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

    - type: caput.pipeline.tasklib.io.LoadFilesFromParams
      out: sstream
      params:
        files: *inputs
        selections:
          freq_range: [{freq[0]:d}, {freq[1]:d}]

    - type: draco.analysis.sidereal.{stacker_instance}
      in: sstream
      out: fstack
      params:
        tag: "fullstack"

    - type: caput.pipeline.tasklib.io.Truncate
      in: fstack
      out: fstack_trunc
      params:
        dataset:
          vis:
            weight_dataset: vis_weight
            variance_increase: 1.0e-5
          vis_weight: 1.0e-6
        save: true
        output_name: "sstack.h5"

    - type: draco.analysis.ringmapmaker.RingMapMaker
      requires: manager
      in: fstack_trunc
      out: ringmap
      params:
        single_beam: true
        weight: natural
        weight_ew: natural
        exclude_intracyl: false
        include_auto: false

    - type: caput.pipeline.tasklib.io.Truncate
      in: ringmap
      out: ringmap_trunc
      params:
        dataset:
          map:
            weight_dataset: weight
            variance_increase: 1.0e-4
          vis_weight: 1.0e-5
        save: true
        output_name: "ringmap.h5"

    - type: draco.analysis.ringmapmaker.RingMapMaker
      requires: manager
      in: fstack
      out: ringmap_int
      params:
        single_beam: true
        weight: natural
        weight_ew: natural
        exclude_intracyl: true
        include_auto: false

    - type: caput.pipeline.tasklib.io.Truncate
      in: ringmap_int
      out: ringmap_int_trunc
      params:
        dataset:
          map:
            weight_dataset: weight
            variance_increase: 1.0e-4
          vis_weight: 1.0e-5
        save: true
        output_name: "ringmap_intercyl.h5"

    # Block
    - type: caput.pipeline.tasklib.debug.WaitUntil
      requires: ringmap_int_trunc
      in: fstack
      out: fstack2

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

    # Get Stokes I visibilities
    - type: draco.analysis.transform.StokesIVis
      requires: manager
      in: fstack_factmask
      out: fstack_stokesI

    # Make a delay power spectrum with noise removed
    - type: draco.analysis.delay.DelayPowerSpectrumNRML
      in: fstack_stokesI
      params:
        dataset: "vis"
        sample_axis: "ra"
        freq_zero: 800.0
        nfreq: {nfreq_delay}
        nsamp: 150
        complex_timedomain: true
        save: true
        output_name: "delayspectrum.h5"

    - type: draco.analysis.delay.DelayFilter
      requires: manager
      in: fstack_stokesI
      out: fstack_lf
      params:
        delay_cut: 0.1
        za_cut: 1.0
        window: true

    # Get Stokes I visibilities
    - type: draco.analysis.transform.StokesIVis
      requires: manager
      in: fstack_lf
      out: fstack_lf_stokesI

    # Make a delay power spectrum with noise removed
    - type: draco.analysis.delay.DelayPowerSpectrumNRML
      in: fstack_lf_stokesI
      params:
        dataset: "vis"
        sample_axis: "ra"
        freq_zero: 800.0
        nfreq: {nfreq_delay}
        nsamp: 150
        complex_timedomain: true
        save: true
        output_name: "delayspectrum_hpf.h5"
"""


class FullStackProcessing(base.ProcessingType):
    """Stacks days or quarterstacks into a full stack.

    This creates a revision based of off another revision type with existing
    data to stack. This can be either a quarterstack revision or a daily revision,
    the latter of which can include either standard regridded days or semi-regridded
    dirty days. The type of input data affects which stacker is used.
    """

    type_name = "fullstack"
    tag_pattern = r"stack((_p\d){1}|_all|$)"

    # Parameters of the job processing
    default_params: ClassVar = {
        # Frequencies to process
        "freq": [0, 1024],
        "nfreq_delay": 1025,
        "split_by_year": True,
        # The beam transfers to use (need to have the same freq range as above)
        "product_path": "/project/rpp-chime/chime/bt_empty/chime_4cyl_allfreq/",
        # System modules to use/load
        "modpath": "/project/rpp-chime/chime/chime_env/modules/modulefiles",
        "modlist": "chime/python/2025.10",
        # Job params
        "time": 30,  # Time in minutes
        "nodes": 3,  # Number of nodes to use
        "ompnum": 8,  # Number of OpenMP threads per MPI process
        "pernode": 24,  # MPI processes per node
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
            raise NotImplementedError(
                "Stacking from a set of days is not currently supported."
            )
            # self.default_params["stacker_instance"] = "SiderealStackerDeconvolve"
        else:
            raise NotImplementedError(
                f"Stacking from type {input_type} is not currently supported."
            )

        # Set the default parameters for this revision
        self.default_params["input_type"] = input_type
        self.default_params["input_rev"] = input_rev.revision

        stacks = {"stack_all": {}}
        inputs = {}

        for tag in input_rev.ls():
            year, _, part = input_rev._parse_tag(tag)
            # Add all tags to the full stack
            stacks["stack_all"].setdefault(year, []).append(tag)
            # Add this tag to the partition stack
            stacks.setdefault(f"p{part}", {}).setdefault(year, []).append(tag)
            # Construct the full file path and add to inputs
            inputs[tag] = str(input_rev.base_path) + "/" + tag

        # Store the full file path for each input to this revision
        self.default_params["inputs"] = inputs
        self.default_params["stacks"] = stacks

    def _load_hook(self):
        # Figure out if we're splitting years into separate stacks
        by_year = self._revparams["split_by_year"]

        stacks = {}

        for stack_key, year_inputs in self._revparams["stacks"].items():
            for year_key, inputs in year_inputs.items():
                if by_year:
                    stacks[f"{year_key}_{stack_key}"] = inputs
                else:
                    # Just combine all years for each partition
                    stacks.setdefault(stack_key, []).extend(inputs)

        self._revparams["stacks"] = stacks

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
            fnames = {"sstack", "quarterstack"}
        elif input_type == "daily":
            fnames = {"sstream_lsd"}

        # Check the file extension and file name format
        for name in fnames:
            testpaths = (name, name + f"_{inputs[0]}")

            for ii, tst in enumerate(testpaths):
                # This is a bit annoying - we just have to be able to
                # deal with different output formats that have existed
                # over the years
                _include_tag = ii == 1

                extension = _check_file_extension(paths[inputs[0]] + "/" + tst)

                if extension is not None:
                    break

            if extension is not None:
                break
        else:
            raise ValueError("Could not resolve any possible source file type.")

        input_paths = [
            (
                f"- {paths[t]}/{tst}_{t}{extension}"
                if _include_tag
                else f"- {paths[t]}/{tst}{extension}"
            )
            for t in inputs
        ]

        input_list_str = "\n" + "\n".join(input_paths)

        jobparams.update({"inputs": input_list_str})

        return jobparams


def _check_file_extension(path: os.PathLike):
    path: str = str(path)

    for ext in {".h5", ".zarr.zip"}:
        if os.path.exists(path + ext):
            return ext

    return None
