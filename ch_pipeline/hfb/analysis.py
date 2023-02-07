"""Tasks for HFB analysis
"""
import numpy as np

from caput import config

from draco.core import task
from draco.util import tools

from . import containers


class HFBMakeTimeAverage(task.SingleTask):
    """Take average over time axis.

    Used for making sub-frequency band shape template and general time averaging.

    Attributes
    ----------
    weighting: str (default: "inverse_variance")
        The weighting to use in the stack.
        Either `uniform` or `inverse_variance`.
    """

    weighting = config.enum(["uniform", "inverse_variance"], default="inverse_variance")

    def process(self, stream):
        """Take average over time axis.

        Parameters
        ----------
        stream : HFBData, HFBHighResData
            Container with HFB data and weights.

        Returns
        -------
        out : HFBTimeAverage, HFBHighResTimeAverage
            Average of stream over time axis.
        """

        contmap = {
            containers.HFBData: containers.HFBTimeAverage,
            containers.HFBHighResData: containers.HFBHighResTimeAverage,
        }

        # Get the output container
        out_cont = contmap[stream.__class__]

        if self.weighting == "uniform":
            # Number of samples with non-zero weight in each time bin
            nsamples = np.count_nonzero(stream.weight[:], axis=-1)

            # For uniform weighting, the average of the variances is
            # sum( 1 / weight ) / nsamples,
            # and the averaged weight is the inverse of that
            variance = tools.invert_no_zero(stream.weight[:])
            weight = nsamples / np.sum(variance, axis=-1)

            # For uniform weighting, the averaged data is the average of all
            # non-zero data
            data = np.sum(stream.hfb[:], axis=-1) / nsamples

        else:
            # For inverse-variance weighting, the averaged weight turns out to
            # be equal to the sum of the weights
            weight = np.sum(stream.weight[:], axis=-1)

            # For inverse-variance weighting, the averaged data is the weighted
            # sum of the data, normalized by the sum of the weights
            data = np.sum(stream.weight[:] * stream.hfb[:], axis=-1) / weight

        # Create container to hold output
        out = out_cont(axes_from=stream, attrs_from=stream)

        # Save data to output container
        out.hfb[:] = data
        out.weight[:] = weight

        # Return output container
        return out


class HFBDivideByTemplate(task.SingleTask):
    """Divide HFB data by template of time-averaged HFB data.

    Used for flattening sub-frequency band shape by dividing on-source data by a template.
    """

    def process(self, stream, template):
        """Divide data by template and place in HFBData container.

        Parameters
        ----------
        stream : containers.HFBData
            Container with HFB data and weights; the numerator in the division.

        template : containers.HFBTimeAverage
            Container with time-averaged HFB data and weights; the denominator in the division.

        Returns
        -------
        out : containers.HFBData
            Container with HFB data and weights; the result of the division.
        """

        # Divide data by template
        data = stream.hfb[:] / template.hfb[:, :, :, np.newaxis]

        # Divide variance by square of template, which means to
        # multiply weight by square of template
        weight = stream.weight[:] * template.hfb[:, :, :, np.newaxis] ** 2

        # Create container to hold output
        out = containers.HFBData(axes_from=stream, attrs_from=stream)

        # Save data and weights to output container
        out.hfb[:] = data
        out.weight[:] = weight

        return out
