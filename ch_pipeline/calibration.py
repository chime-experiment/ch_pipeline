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

import gain_sol_transit as gsol # Temp until the eigensolver is in ch_util. 
# Or should the gainsol code go here?

class PointSourceCalibration(pipeline.TaskBase):

    def setup(self, files):
        """ Derive calibration solution from input
        start with the calibration solution itself
        How much time do I want from each transit? 
        Enough to actually get the transit at all frequencies 
        Do I want to fringestop? No prolly not

        Need to know feed layout. Which are P0 which are P1? 

        This is parallelized over frequency, right?

        Should the linear interpolation go in here?
        """
        ts = containers.TimeStream.from_acq_files(files)
        # Need to select subset of this data
        ts_xpol, ts_ypol = ts, ts # Need to figure out a way to select pols
        
        gain_arr_x = gsol.solve_gain(ts_xpol)
        gain_arr_y = gsol.solve_gain(tx_ypol)


    def next(self, data):

        # Calibrate data as it comes in.

        return calibrated_data
