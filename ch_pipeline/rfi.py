"""
================================================
Tasks for RFI cleaning (:mod:`~ch_pipeline.rfi`)
================================================

.. currentmodule:: ch_pipeline.rfi

Tasks for calculating RFI masks for timestream data.

Tasks
=====

.. autosummary::
    :toctree: generated/

    RFIFilter
"""
from caput import pipeline, config
from caput import mpiutil
from ch_util import rfi

from . import task


class RFIFilter(task.SingleTask):
    """Filter RFI from a Timestream.

    This task works on the parallel
    :class:`~ch_pipeline.containers.TimeStream` objects.

    Attributes
    ----------
    threshold_mad : float
        Threshold above which we mask the data.
    """

    threshold_mad = config.Property(proptype=float, default=5.0)

    def process(self, data):

        if mpiutil.rank0:
            print "RFI filtering %s" % data.attrs['tag']

        data.redistribute('time')

        # Construct RFI mask
        mask = rfi.flag_dataset(data, only_autos=False, threshold=self.threshold_mad)

        data.weight[:] *= (1 - mask)  # Switch from mask to inverse noise weight

        # Redistribute across frequency
        data.redistribute('freq')

        return data
