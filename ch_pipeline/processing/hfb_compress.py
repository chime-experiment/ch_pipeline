"""CHIME HFB compression processing type.

Classes
=======
- :py:class:`HFBCompressProcessing`
"""

from typing import ClassVar

from . import base

DEFAULT_SCRIPT = """
# Cluster configuration
cluster:
  name: {jobname}

  directory: {dir}
  temp_directory: {tempdir}

  time: {time}
  system: fir
  nodes: {nodes}
  ompnum: {ompnum}
  pernode: {pernode}
  mem: 2900G  # use fat nodes for now

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

    - type: draco.core.misc.DebugInfo

    # Load the telescope model that we need for several steps
    - type: draco.core.io.LoadProductManager
      out: manager
      params:
        product_directory: {product_path}

    - type: ch_pipeline.core.dataquery.ConnectDatabase
      params:
        timeout: 2
        ntries: 3

    - type: ch_pipeline.core.dataquery.QueryDatabase
      out: qfilelist
      params:
        node_spoof:
          fir_online: /project/rpp-chime/chime/chime_online
        acqtype: 'hfb'
        instrument: 'chime'
        start_time: {start_time}
        end_time: {end_time}

    - type: draco.core.io.Print
      out: qfilelist_p
      in: qfilelist

    - type: ch_pipeline.hfb.io.LoadFiles
      requires: qfilelist
      out: tstream_orig
      params:
        distributed: True
        single_group: False

    - type: ch_pipeline.hfb.compress.CompressHFBWeights
      in: tstream_orig
      out: tstream_compressed
      params:
        method: {compress_method}
        save: True
        output_name: "compressed-hfb-example.h5"

    - type:       draco.core.task.Delete
      in:         tstream_orig
      out:        del_tstream_orig
"""

class HFBCompressProcessing(base.ProcessingType):
    """CHIME HFB compression pipeline processing type."""

    type_name = "test_hfb_compress"

    # Parameters of the job processing
    default_params: ClassVar = {
        "nodes": 1,
        "pernode": 1,
        "ompnum": 96,
        "time": "0-3:00:00",
        # The beam transfers to use (need to have the same freq range as above)
        "product_path": "/project/rpp-chime/chime/bt_empty/chime_4cyl_allfreq/",
        # System modules to use/load
        "modpath": "/project/rpp-chime/chime/chime_env/modules/modulefiles",
        "modlist": "chime/python/2025.10",
        "compress_method": "med"
    }
    default_script = DEFAULT_SCRIPT
