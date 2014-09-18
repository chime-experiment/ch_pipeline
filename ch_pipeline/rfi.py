
import numpy as np

from caput import pipeline, config
from caput import mpidataset
from ch_util import rfi_pipeline

import containers


class RFIFilter(pipeline.TaskBase):
    """Filter RFI from a Timestream.

    Attributes
    ----------
    threshold_mad : float
        Threshold above which we mask the data.
    """

    threshold_mad = config.Property(proptype=float, default=5.0)

    def next(self, data):

        data.redistribute(axis=2)

        # Construct RFI mask
        mask = rfi_pipeline.flag_dataset_with_mad(data, only_autos=False,
                                                  threshold=self.threshold_mad)

        # Turn mask into MPIArray
        mask = mpidataset.MPIArray.wrap(mask.view(np.uint8), axis=2)

        # Create MaskedTimeStream instance and redistribute back over frequency
        mts = containers.MaskedTimeStream.from_timestream_and_mask(data, mask)
        mts.redistribute(0)

        return mts
