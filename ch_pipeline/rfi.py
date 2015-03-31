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
from ch_util import rfi_pipeline


class RFIFilter(pipeline.TaskBase):
    """Filter RFI from a Timestream.

    This task works on the parallel
    :class:`~ch_pipeline.containers.TimeStream` objects.

    Attributes
    ----------
    threshold_mad : float
        Threshold above which we mask the data.
    """

    threshold_mad = config.Property(proptype=float, default=5.0)

    def next(self, data):

        if mpiutil.rank0:
            print "RFI filtering %s" % data.attrs['tag']

        data.redistribute(axis=2)

        # Construct RFI mask
        mask = rfi_pipeline.flag_dataset_with_mad(data, only_autos=False,
                                                  threshold=self.threshold_mad)

        # Add weight dataset, and copy mask into it
        data.add_weight()
        data.weight[:] = (1 - mask)  # Switch from mask to inverse noise weight

        # Redistribute across frequency
        data.redistribute(0)

        return data
