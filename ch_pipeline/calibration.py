"""
=======================================================
Tasks for Calibration (:mod:`~ch_pipeline.calibration`)
=======================================================

.. currentmodule:: ch_pipeline.calibration

Tasks for calibrating the data.

Tasks
=====

.. autosummary::
    :toctree: generated/

    PointSourceCalibration
"""

from caput import pipeline
from caput import config


class PointSourceCalibration(pipeline.TaskBase):

    def setup(self, files):
        # Derive calibration solution from input
        pass

    def next(self, data):

        # Calibrate data as it comes in.

        return calibrated_data
