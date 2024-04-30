from caput import time as ctime
from ch_util import ephemeris
from chimedb import core

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
  mem: 19200M

  venv: {venv}

# Pipeline task configuration 
pipeline: 

  logging:
    root: DEBUG
  tasks:
    - type: draco.core.task.SetMPILogging 
      params:
        level_all: INFO
        level_rank0: DEBUG

    - type: draco.core.io.LoadProductManager 
      out: manager
      params:
          product_directory: /project/rpp-krs/chime/bt_empty/chime_4cyl_allfreq

    - type: draco.core.io.LoadFilesFromParams
      out: transit
        params:
          files: {filename}

    - type: ch_pipeline.core.dataquery.QueryLoadFromTransit
      in: transit
      out: stackstream
        
    - type: ch_pipeline.analysis.beam.ComputeHolographicSensitivity
      in:
        - transit
        - stackstream
      out: sensitivity
      params:
        gridding_factor: {gridding_factor}
        save: true
      requires: manager
"""

class HolographySensitivity(base.ProcessingType):

    type_name = "holo_sens"
    tag_pattern = ""

    default_params = {
        "files": "/project/rpp-chime/chime/chime_processed/holography/holo_fstop/rev_01/CygA_run000/",
        "gridding_factor": 2,
    }

    default_script = DEFAULT_SCRIPT
