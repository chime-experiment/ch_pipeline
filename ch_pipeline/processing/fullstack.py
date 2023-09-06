from . import base, daily, quarterstack

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

    - type: draco.core.misc.CheckMPIEnvironment
      params:
        timeout: 420

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

    - type: draco.analysis.sidereal.{stacker_instance}
      in: sstream
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

    - type: draco.analysis.ringmapmaker.RingMapMaker
      requires: manager
      in: fstack
      params:
        single_beam: true
        weight: "natural"
        weight_ew: "natural"
        exclude_intracyl: false
        include_auto: false
        save: true
        output_name: "ringmap.h5"
        npix: 1024

    - type: draco.analysis.ringmapmaker.RingMapMaker
      requires: manager
      in: fstack
      params:
        single_beam: true
        weight: "natural"
        weight_ew: "natural"
        exclude_intracyl: true
        include_auto: false
        save: true
        output_name: "ringmap_intercyl.h5"
        npix: 1024

    - type: draco.analysis.delay.DelaySpectrumEstimator
      requires: manager
      in: fstack
      params:
        freq_zero: 800.0
        nfreq: {nfreq_delay}
        nsamp: 100
        save: Yes
        complex_timedomain: true
        output_name: "dspec.h5"

    - type: draco.analysis.delay.DelayFilter
      requires: manager
      in: fstack
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
        nsamp: 100
        complex_timedomain: true
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
    tag_pattern = "stack"

    # Parameters of the job processing
    default_params = {
        # Frequencies to proces
        "freq": [0, 1024],
        # The beam transfers to use
        "product_path": "/project/rpp-chime/chime/bt_empty/chime_4cyl_allfreq/",
        "nfreq_delay": 1025,
        # Job params
        "time": 70,
        "nodes": 12,
        "ompnum": 4,
        "pernode": 12,
    }

    default_script = DEFAULT_SCRIPT

    def _create_hook(self):
        """Create the revision.

        This looks at the input revision and determines which stacking method
        to use accordingly.
        """

        # Prompt the user to provide a source revision path
        source_path = input("Enter the input revision path:")
        source_path = source_path.strip("/").split("/")
        input_rev = source_path[-1]
        input_type = source_path[-2]

        base = "/" + "/".join(source_path)
        root = "/" + "/".join(source_path[:-2])

        if input_type == "quarterstack":
            self.default_params["stacker_instance"] = "SiderealStackerMatch"
            input_rev = quarterstack.QuarterStackProcessing(input_rev, root_path=root)
        elif input_type == "daily":
            self.default_params["stacker_instance"] = "SiderealStackerDeconvolve"
            input_rev = daily.DailyProcessing(input_rev, root_path=root)

        self.default_params["input_type"] = input_type
        self.default_params["days"] = {d: base + "/" + d for d in input_rev.ls()}

    def _available_tags(self):
        """Return all the tags that are available to run.

        Right now, this just returns a single tag.
        """
        return ["stack"]

    def _finalise_jobparams(self, tag, jobparams):
        """Modify the job parameters before the final config is made."""

        days = self._revparams["days"]
        input_type = self._revparams["input_type"]

        if input_type == "quarterstack":
            day_list_str = "\n" + "\n".join(
                [f"- {path}/sstack.zarr.zip" for path in days.values()]
            )

        elif input_type == "daily":
            day_list_str = "\n" + "\n".join(
                [f"- {path}/sstream_lsd_{d}.zarr.zip" for d, path in days.items()]
            )

        jobparams.update({"days": day_list_str})

        return jobparams
