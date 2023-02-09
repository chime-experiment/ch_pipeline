"""Tasks for HFB analysis
"""
import numpy as np

from caput import config
from caput import mpiarray

from draco.core import task
from draco.util import tools
from draco.core import containers as dcontainers

import beam_model

from . import containers


class HFBAverageOverTime(task.SingleTask):
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

        # Average data over time, which corresponds to the final axis (index -1)
        data, weight = average_hfb(
            data=stream.hfb[:],
            weight=stream.weight[:],
            axis=-1,
            weighting=self.weighting,
        )

        # Create container to hold output
        out = out_cont(axes_from=stream, attrs_from=stream)

        # Save data and weights to output container
        out.hfb[:] = data
        out.weight[:] = weight

        return out


class MakeHighFreqRes(task.SingleTask):
    """Combine frequency and sub-frequency axes"""

    def process(self, stream):
        """Convert HFBData to HFBHighResData container

        Parameters
        ----------
        stream : HFBData
            Data with frequency and subfrequency axis

        Returns
        -------
        out : HFBHighResData
            Data with single high-resolution frequency axis
        """

        # Retrieve shape of data
        nfreq = len(stream._data["index_map"]["freq"]["centre"][:])
        nsubfreq = len(stream._data["index_map"]["subfreq"][:])
        nbeam = len(stream._data["index_map"]["beam"][:])

        # Retrieve data and weights
        data = stream.hfb[:]
        weight = stream.weight[:]

        # Change data and weights to numpy array, so that it can be reshaped
        if isinstance(data, mpiarray.MPIArray):
            data = data.local_array
            weight = weight.local_array

        # Make combined frequency axis
        cfreq = stream._data["index_map"]["freq"]["centre"]
        subfreq = stream._data["index_map"]["subfreq"][:]
        freq = (cfreq[:, np.newaxis] + subfreq).flatten()

        # Retrieve beam and time axes
        beam = stream._data["index_map"]["beam"][:]
        time = stream.time

        # Combine frequency and sub-frequency axes
        data = data.reshape(nfreq * nsubfreq, nbeam, -1)
        weight = weight.reshape(nfreq * nsubfreq, nbeam, -1)

        # Create container to hold output
        out = containers.HFBHighResData(
            freq=freq, beam=beam, time=time, attrs_from=stream
        )

        # Save data to output container
        out.hfb[:] = data
        out.weight[:] = weight

        # Return output container
        return out


class HFBDivideByTemplate(task.SingleTask):
    """Divide HFB data by template of time-averaged HFB data.

    Used for flattening sub-frequency band shape by dividing on-source data by a
    template.
    """

    def process(self, stream, template):
        """Divide data by template and place in HFBData container.

        Parameters
        ----------
        stream : containers.HFBData
            Container with HFB data and weights; the numerator in the division.

        template : containers.HFBTimeAverage
            Container with time-averaged HFB data and weights; the denominator in the
            division.

        Returns
        -------
        out : containers.HFBData
            Container with HFB data and weights; the result of the division.
        """

        template_array = template.hfb[:]

        # Divide data by template
        data = stream.hfb[:] * tools.invert_no_zero(template_array[:, :, :, np.newaxis])

        # Divide variance by square of template, which means to
        # multiply weight by square of template
        weight = stream.weight[:] * template_array[:, :, :, np.newaxis] ** 2

        # Create container to hold output
        out = containers.HFBData(axes_from=stream, attrs_from=stream)

        # Save data and weights to output container
        out.hfb[:] = data
        out.weight[:] = weight

        return out


class HFBStackDays(task.SingleTask):
    """Combine HFB data of multiple days.

    Attributes
    ----------
    weighting: str (default: "inverse_variance")
        The weighting to use in the stack.
        Either `uniform` or `inverse_variance`.
    """

    stack = None

    weighting = config.enum(["uniform", "inverse_variance"], default="inverse_variance")

    def process(self, sdata):
        """Add weights and data of one day to stack.

        Parameters
        ----------
        sdata : hfb.containers.HFBHighResSpectrum
            Individual (time-averaged) day to add to stack.
        """

        # If this is our first sidereal day, then initialize the
        # container that will hold the stack.
        if self.stack is None:
            self.stack = dcontainers.empty_like(sdata)

            self.stack.add_dataset("nsample")

            self.stack.redistribute("freq")

            # Initialize all datasets to zero
            for data in self.stack.datasets.values():
                data[:] = 0

        # Accumulate the total number of samples
        dtype = self.stack.nsample.dtype
        count = (sdata.weight[:] > 0.0).astype(dtype)
        self.stack.nsample[:] += count

        # Accumulate variances or inverse variances
        if self.weighting == "uniform":
            # Accumulate the variances in the stack.weight dataset.
            self.stack.weight[:] += tools.invert_no_zero(sdata.weight[:])
        else:
            # We are using inverse variance weights.  In this case,
            # we accumulate the inverse variances in the stack.weight
            # dataset.
            self.stack.weight[:] += sdata.weight[:]

        # Accumulate data
        if self.weighting == "uniform":
            self.stack.hfb[:] += sdata.hfb[:]
        else:
            self.stack.hfb[:] += sdata.weight[:] * sdata.hfb[:]

    def process_finish(self):
        """Normalize and return stacked weights and data."""

        # Compute weights and data
        if self.weighting == "uniform":
            # Number of days with non-zero weight
            norm = self.stack.nsample[:].astype(np.float32)

            # For uniform weighting, invert the accumulated variances and
            # multiply by number of days to finalize stack.weight.
            self.stack.weight[:] = norm * tools.invert_no_zero(self.stack.weight[:])

            # For uniform weighting, simply normalize accumulated data by number
            # of days to finalize stack.hfb.
            self.stack.hfb[:] /= norm

        else:
            # For inverse variance weighting, the weight dataset doesn't have to
            # be normalized (it is simply the sum of the weights). The accumulated
            # data have to be normalized by the sum of the weights.
            self.stack.hfb[:] *= tools.invert_no_zero(self.stack.weight[:])

        return self.stack


class HFBAverageOverBeams(task.SingleTask):
    """Taking the average of HFB data over beam axis.

    Attributes
    ----------
    weighting: str (default: "inverse_variance")
        The weighting to use in the averaging.
        Either `uniform` or `inverse_variance`.
    """

    weighting = config.enum(["uniform", "inverse_variance"], default="inverse_variance")

    def process(self, stream):
        """Take average over beam axis.

        Parameters
        ----------
        stream : containers.HFBHighResTimeAverage
            Container with time-averaged high-frequency resolution HFB data and weights.

        Returns
        -------
        out : containers.HFBHighResSpectrum
            Container with high-frequency resolution spectrum data and weights.
        """

        # Average data over beams, which corresponds to axis index 1
        data, weight = average_hfb(
            data=stream.hfb[:],
            weight=stream.weight[:],
            axis=1,
            weighting=self.weighting,
        )

        # Create container to hold output
        out = containers.HFBHighResSpectrum(axes_from=stream, attrs_from=stream)

        # Save data to output container
        out.hfb[:] = data
        out.weight[:] = weight

        # Return output container
        return out


def average_hfb(data, weight, axis, weighting="inverse_variance"):
    """Average HFB data

    Parameters
    ----------
    data : array
        Data to average.
    weight : array
        Weights of `data`.
    axis : int
        Index of axis to average over.
    weighting: str (default: "inverse_variance")
        The weighting to use in the averaging.
        Either `uniform` or `inverse_variance`.

    Output
    ------
    avg_data : array
        Averaged data.
    avg_weight : array
        Averaged weights.
    """

    if weighting == "uniform":
        # Binary weights for uniform weighting
        binary_weight = np.zeros(weight.shape)
        binary_weight[weight != 0] = 1

        # Number of samples with non-zero weight in each time bin
        nsamples = np.sum(binary_weight, axis=axis)

        # For uniform weighting, the average of the variances is
        # sum( 1 / weight ) / nsamples,
        # and the averaged weight is the inverse of that
        variance = tools.invert_no_zero(weight)
        avg_weight = nsamples * tools.invert_no_zero(np.sum(variance, axis=axis))

        # For uniform weighting, the averaged data is the average of all
        # non-zero data
        avg_data = np.sum(binary_weight * data, axis=axis) * tools.invert_no_zero(
            nsamples
        )

    else:
        # For inverse-variance weighting, the averaged weight turns out to
        # be equal to the sum of the weights
        avg_weight = np.sum(weight, axis=axis)

        # For inverse-variance weighting, the averaged data is the weighted
        # sum of the data, normalized by the sum of the weights
        avg_data = np.sum(weight * data, axis=axis) * tools.invert_no_zero(avg_weight)

    return avg_data, avg_weight


class HFBSelectTransit(task.SingleTask):
    """Find transit data through FRB beams.

    Task for selecting on- and off-source data.
    beam_width gives the width of the synthetic beam. To select off-source data, a window around the beam is chosen.
    Size of this window is determined by "w" parameter: window size = w * beam_width
    Normally, when w ~ 18, the source is outside of the sideloebes of the beam.  Data outside this window is taken as
    off-source data.
    On-source data selection is based on 2 criteria:

    1) The source must be inside the main lobe of synthetic beam.
    2) Sensitivity of the beam toward the source must be greater than some threshold.

    A few points about sensitivity for on-source data selection:

    i) Sensitivity of the last row of EW beams is higher in the side lobes than the main lobe. That is why So if we take maximum
    sensitivity of the beam to calculate the threshold needed for data selection, selected data won't be in the main lobe.
    Therefore, for setting sensitivity threshold, closest sample to the centre of the beam is chosen (this sample
    is closest to the transit time through the centre beam centre). Then, sensitivity corresponding to this sample is the
    maximum sensitivity we want for setting up the threshold for on-source data selection.

    ii) sensitivity

    Attributes
    ----------
    sensitivity_threshold: float

    beam_index : list
        List of beam indices for which find the transit time
    ra : float
        Right ascension of the source
    dec : float
        Declination of the source
    """

    sensitivity_threshold = config.Property(proptype=float)
    ra = config.Property(proptype=float)
    dec = config.Property(proptype=float)
    on_source = config.Property(proptype=bool, default=True)
    # s = 5

    def process(self, stream):
        """Find transit data for the given beams.

        Parameters
        ----------
        stream : containers.HFBData
            Container with HFB data and weights.


        Returns
        -------
        out : containers.HFBData
            Array consisting of transit data
        """
        formed_beam = beam_model.formed.FFTFormedBeamModel()
        tied = beam_model.composite.FutureMostAccurateCompositeBeamModel()

        # Extract beam indices, change format for sensitivity calculations
        beam = stream._data["index_map"]["beam"][:]
        nbeam = len(beam)
        if len(str(beam[0])) == 1:
            beam_index = [
                beam[0],
                int("100" + str(beam[0])),
                int("200" + str(beam[0])),
                int("300" + str(beam[0])),
            ]
        elif len(str(beam[0])) == 2:
            beam_index = [
                beam[0],
                int("10" + str(beam[0])),
                int("20" + str(beam[0])),
                int("30" + str(beam[0])),
            ]
        elif len(str(beam[0])) == 3:
            beam_index = [
                beam[0],
                int("1" + str(beam[0])),
                int("2" + str(beam[0])),
                int("3" + str(beam[0])),
            ]

        # Extract physical frequencies:
        nfreq = len(stream._data["index_map"]["freq"]["centre"][:])
        physical_freq = (
            stream._data["index_map"]["freq"]["centre"][:].reshape(nfreq, 1)
            + stream._data["index_map"]["subfreq"][:]
        ).flatten()
        # Extract time array, hfb data and weights for selected beams and frequencies:
        time = stream.time[:]  # Is it starting of acquistion time?
        data = stream.hfb[:, :, :, :]
        weight = stream.weight[:, :, :, :]

        # Convert equatorial position of the source to cartesian coordinates:

        pfe = []
        for j, time_sample in enumerate(time):
            pfe.append(
                formed_beam.get_position_from_equatorial(self.ra, self.dec, time[j])
            )
        pfe = np.asarray(pfe)
        # sens = actual_beam.get_sensitivity(self.beam_index, pfe, physical_freq)
        # sens_mask = np.copy(sens)
        sens = tied.get_sensitivity(beam_index, pfe, physical_freq)
        sens_on = np.copy(sens)
        # print(sens.shape)

        beam_width = formed_beam.get_beam_widths(beam_index, physical_freq)
        pos = formed_beam.get_beam_positions(beam_index, physical_freq)

        # off-source data
        if self.on_source is False:
            x_edge_upper = pos[:, :, 0] + 18 * beam_width[:, :, 0] / 2
            x_edge_lower = pos[:, :, 0] - 18 * beam_width[:, :, 0] / 2
            # Select data when the source is completely outside of the primary beam
            off = (pfe[:, 0][:, None, None] > x_edge_upper[None, :, :]) | (
                pfe[:, 0][:, None, None] < x_edge_lower[None, :, :]
            )
            sens[(off)] = 1
            sens[np.invert(off)] = 0
            # Swap axes to have (freq,beam,time), reshape it to have subfreq, and multiply sens by data, weight and time
            sens = np.swapaxes(sens, 0, 2)
            sens = sens.reshape(nfreq, 128, nbeam, len(time))

            # change the weights to select off-source data
            weight = weight * sens

        # on-source data:
        elif self.on_source is True:
            # Select only the main lobe using beam_width function

            x_edge_upper = pos[:, :, 0] + beam_width[:, :, 0] / 2
            x_edge_lower = pos[:, :, 0] - beam_width[:, :, 0] / 2

            main_lobe = (pfe[:, 0][:, None, None] < x_edge_upper[None, :, :]) & (
                pfe[:, 0][:, None, None] > x_edge_lower[None, :, :]
            )

            # Find sensitivity corresponding to the closest sample to transit wrt centre of the beam:
            beam_pos_x = formed_beam.get_beam_positions(beam_index, physical_freq)[
                :, :, 0
            ]  # shape (beam,freq,(x,y))
            transit_index = np.argmin(
                np.abs(pfe[:, 0][:, None, None] - beam_pos_x), axis=0
            )  # transit indices with shape (beam,freq)

            max_sens = np.zeros(transit_index.shape)

            for i in range(transit_index.shape[0]):
                for j in range(transit_index.shape[1]):
                    max_sens[i, j] = sens[:, i, j][transit_index[i, j]]

            threshold = self.sensitivity_threshold * max_sens

            sens[((sens_on > threshold[None, :, :]) & main_lobe)] = 1
            sens[((sens_on < threshold[None, :, :]) | np.invert(main_lobe))] = 0

            # Swap axes to have (freq,beam,time), reshape it to have subfreq, and multiply sens by data, weight and time
            sens = np.swapaxes(sens, 0, 2)
            sens = sens.reshape(nfreq, 128, nbeam, len(time))

            # change the weights to select on-source data
            weight = weight * sens

        # Create container to hold output
        out = containers.HFBData(axes_from=stream, attrs_from=stream)

        # Save weights and data to output container for on-source and off-source data
        out.hfb[:] = data
        out.weight[:] = weight

        # Return output container
        return out
