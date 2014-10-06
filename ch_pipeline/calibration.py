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

    source = config.Property(proptype=str, default='CasA')

    def setup(self, files):
        # Derive calibration solution from input
        pass

    def next(self, data):

        # Calibrate data as it comes in.
        calibrated_data = data

        return calibrated_data


class NoiseSourceCalibration(pipeline.TaskBase):

    def setup(self):
        # Initialise any required products in here. This function is only
        # called as the task is being setup, and before any actual data has
        # been sent through

        pass

    def next(self, data):
        # This method should derive the gains from the data as it comes in,
        # and apply the corrections to rigidise the data
        #
        # The data will come be received as a containers.TimeStream type. In
        # some ways this looks a little like AnData, but it works in parallel

        calibrated_data = data

        return calibrated_data
