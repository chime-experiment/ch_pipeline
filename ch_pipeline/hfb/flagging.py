"""HFB Tasks for flagging data
"""
import numpy as np

from caput import config

from draco.core import task

from containers import HFBRFIMask


class HFBRNTMask(task.SingleTask):
    """Identify RFI in HFB data using the radiometer noise test, averaging over beams.

    Attributes
    ----------
    threshold : float
        The desired threshold for the RFI fliter. Data with sensitivity metric
        values above this thershold will be considered RFI.
    keep_sens : bool
        Save the sensitivity metric data that were used to construct
        the mask in the output container.
    """

    threshold = config.Property(proptype=float, default=1.6)
    keep_sens = config.Property(proptype=bool, default=False)

    def process(self, stream):
        """Derive an RFI mask.

        Parameters
        ----------
        stream : containers.HFBData
            Container with HFB data and weights.

        Returns
        -------
        out : containers.HFBRFIMask
            Boolean mask that can be applied to an HFB data container
            with the task `ApplyHFBMask` to mask contaminated
            frequencies, subfrequencies and time samples.
        """

        # Extract data and weight arrays, averaging over beams
        data = np.mean(stream.hfb[:], axis=2)
        weight = np.mean(stream.weight[:], axis=2)

        # Number of samples per data point in the HFB data:
        # N_S = Delta nu * Delta t, where Delta nu is the frequency resolution
        # (390.625 kHz / 128) and Delta t the integration time (10 s)
        # NOTE: Should this be extracted from the container attributes? In that
        # case this information needs to be added to the attributes upstream.
        N_SAMP = 390625.0 / 128.0 * 10.0

        # Ideal radiometer equation
        radiometer = data**2 / N_SAMP

        # Radiometer noise test: the sensitivity metric would be unity for
        # an ideal radiometer, it would be higher for data with RFI
        sensitivity_metric = 2.0 / (radiometer * weight)

        # Boolean mask idicating data that are contaminated by RFI
        mask = sensitivity_metric > self.threshold

        # Create container to hold output
        out = HFBRFIMask(axes_from=stream, attrs_from=stream)

        if self.keep_sens:
            out.add_dataset("sens")
            out.sens[:] = sensitivity_metric

        # Save mask to output container
        out.mask[:] = mask

        # Add beam selection to RFI mask attributes
        out.attrs["beam"] = stream._data["index_map"]["beam"][:]

        # Return output container
        return out


class ApplyHFBMask(task.SingleTask):
    """Apply a mask to an HFB stream.

    Attributes
    ----------
    zero_data : bool, optional
        Zero the data in addition to modifying the weights. Default is True.
    """

    zero_data = config.Property(proptype=bool, default=True)

    def process(self, stream, mask):
        """Set weights to zero for flagged data.

        Parameters
        ----------
        stream : containers.HFBData
            Container with HFB data and weights.

        mask : containers.HFBRFIMask
            Boolean mask indicating the samples that are contaminated by RFI.

        Returns
        -------
        stream : containers.HFBData
            Container with HFB data and weights, with weights of flagged data
            set to zero."""

        # flag = mask[:, :, np.newaxis, :]

        # Create a slice that will expand the mask to
        # the same dimensions as the weight array
        waxis = stream.weight.attrs["axis"]
        slc = [slice(None)] * len(waxis)
        for ww, name in enumerate(waxis):
            if name not in mask.mask.attrs["axis"]:
                slc[ww] = None

        # Extract mask
        flag = mask.mask[:].astype(stream.weight.dtype)

        # Convert mask from caput.mpiarray.MPIArray to regular numpy.ndarray
        flag = np.array(flag)

        # Expand mask to same dimension as weight array
        flag = flag[tuple(slc)]

        # Log how much data we're masking
        self.log.info(
            "%0.2f percent of data will be masked."
            % (100.0 * np.sum(flag) / float(flag.size),)
        )

        # Apply the mask
        if np.any(flag):

            # Apply the mask to the weights
            stream.weight[:] *= 1.0 - flag

            # If requested, apply the mask to the data
            if self.zero_data:
                stream.hfb[:] *= 1.0 - flag

        return stream
