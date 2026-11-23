"""Tasks for transforming HFB data.

This covers combining the frequency axes, flattening the sub-frequency band
shape, removing systematics, and shifting data in RA or frequency.

Tasks
=====
- :py:class:`Average`
- :py:class:`StackDays`
- :py:class:`MakeHighFreqRes`
- :py:class:`MakeHighFreqResRingMap`
- :py:class:`FlattenPFB`
- :py:class:`FlattenRingMapPFB`
- :py:class:`DividePFB`
- :py:class:`DivideByTemplate`
- :py:class:`Difference`
- :py:class:`OnOffDifference`
- :py:class:`MedianSubtraction`
- :py:class:`AlignEWBeams`
- :py:class:`DopplerShift`
"""

import numpy as np
from caput import config, mpiarray
from caput.algorithms import median
from caput.containers import empty_like
from caput.pipeline import tasklib
from caput.util import mpitools
from caput.util.pfb import DeconvolvePFB
from ch_ephem.coord import bmxy_to_hadec
from ch_util.hfbcat import HFBCatalog, get_doppler_shifted_freq
from draco.util import tools
from skyfield.positionlib import Angle
from skyfield.starlib import Star

from . import containers
from .util import ensure_list, interpolation_linear


class Average(tasklib.base.ContainerTask):
    """Take average of HFB data over any axis.

    Used for making sub-frequency band shape template and general time averaging,
    and for taking the average of HFB data over beam axis.

    Attributes
    ----------
    axis: str (default: "time")
        Axis over which to take average.
        Options are `freq`, `subfreq`, `beam`, and `time`.
    weighting: str (default: "inverse_variance")
        The weighting to use in the stack.
        Either `uniform` or `inverse_variance`.
    """

    axis = config.enum(["freq", "subfreq", "beam", "beam_ew", "time"], default="time")
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
            "freq": {},
            "subfreq": {},
            "beam": {
                containers.HFBHighResTimeAverage: containers.HFBHighResSpectrum,
            },
            "beam_ew": {
                containers.HFBHighResRingMap: containers.HFBHighResBeamAvgRingMap,
            },
            "time": {
                containers.HFBData: containers.HFBTimeAverage,
                containers.HFBHighResData: containers.HFBHighResTimeAverage,
            },
        }

        # Check if necessary output container exists
        if stream.__class__ not in contmap[self.axis]:
            raise NotImplementedError(
                f"Averaging {stream.__class__} over {self.axis} is not implemented."
            )

        # Find index of axis over which to average
        axis_index = stream._dataset_spec["hfb"]["axes"].index(self.axis)

        # Extract hfb and weight datasets from input container
        data = stream.hfb[:]
        weight = stream.weight[:]

        # Average the data and compute the weight of the averaged data
        if self.weighting == "uniform":
            # Binary weights for uniform weighting
            binary_weight = np.zeros(weight.shape)
            binary_weight[weight != 0] = 1

            # Number of samples with non-zero weight in each time bin
            nsamples = np.sum(binary_weight, axis=axis_index)

            # For uniform weighting, the variance of averaged data is
            # sum( variance ) / nsamples**2,
            # and the weight is the inverse of that
            variance = tools.invert_no_zero(weight)
            avg_weight = nsamples**2 * tools.invert_no_zero(
                np.sum(variance, axis=axis_index)
            )

            # For uniform weighting, the averaged data is the average of all
            # non-zero data
            avg_data = np.sum(
                binary_weight * data, axis=axis_index
            ) * tools.invert_no_zero(nsamples)

        else:
            # For inverse-variance weighting, the averaged weight turns out to
            # be equal to the sum of the weights
            avg_weight = np.sum(weight, axis=axis_index)

            # For inverse-variance weighting, the averaged data is the weighted
            # sum of the data, normalized by the sum of the weights
            avg_data = np.sum(weight * data, axis=axis_index) * tools.invert_no_zero(
                avg_weight
            )

        # Get the output container
        out_cont_type = contmap[self.axis][stream.__class__]

        # Create container to hold output
        out = out_cont_type(axes_from=stream, attrs_from=stream)

        # Save data and weights to output container
        out.hfb[:] = avg_data
        out.weight[:] = avg_weight

        # Add index map of axis that was averaged over to attributes
        out.attrs[self.axis] = stream._data["index_map"][self.axis][:]

        return out


class StackDays(tasklib.base.ContainerTask):
    """Combine HFB data of multiple days.

    Attributes
    ----------
    tag : str (default: "stack")
        The tag to give the stack.
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
        sdata : any HFB data container
            Individual (time-averaged) day to add to stack.
        """
        sdata.redistribute("freq")

        # Get the LSD (or CSD) label out of the input's attributes.
        # If there is no label, use a placeholder.
        if "lsd" in sdata.attrs:
            input_lsd = sdata.attrs["lsd"]
        elif "csd" in sdata.attrs:
            input_lsd = sdata.attrs["csd"]
        else:
            input_lsd = -1

        input_lsd = ensure_list(input_lsd)

        # If this is our first sidereal day, then initialize the
        # container that will hold the stack.
        if self.stack is None:
            self.stack = empty_like(sdata)
            self.stack.hfb[:] = 0.0
            self.stack.weight[:] = 0.0

            # Add stack-specific dataset: count of samples, to be used as weight
            # for the uniform weighting case. Initialize this dataset to zero.
            if "nsample" not in self.stack.datasets:
                self.stack.add_dataset("nsample")
                self.stack.nsample[:] = 0

            self.stack.redistribute("freq")

            self.lsd_list = []

        # Accumulate
        self.log.info(f"Adding to stack LSD(s): {input_lsd}")

        self.lsd_list += input_lsd

        if "nsample" in sdata.datasets:
            # The input sidereal stream is already a stack
            # over multiple sidereal days. Use the nsample
            # dataset as the weight for the uniform case.
            count = sdata.nsample[:]
        else:
            # The input sidereal stream contains a single
            # sidereal day.  Use a boolean array that
            # indicates a non-zero weight dataset as
            # the weight for the uniform case.
            dtype = self.stack.nsample.dtype
            count = (sdata.weight[:] > 0.0).astype(dtype)

        # Accumulate the total number of samples.
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
        """Normalize and return stacked weights and data.

        Returns
        -------
        stack : same as sdata
            Stack of sidereal days.
        """
        self.stack.attrs["tag"] = self.tag
        self.stack.attrs["lsd"] = np.array(self.lsd_list)

        # Compute weights and data
        if self.weighting == "uniform":
            # Number of days with non-zero weight
            norm = self.stack.nsample[:].astype(np.float32)

            # For uniform weighting, invert the accumulated variances and
            # multiply by number of days squared to finalize stack.weight.
            self.stack.weight[:] = norm**2 * tools.invert_no_zero(self.stack.weight[:])

            # For uniform weighting, simply normalize accumulated data by number
            # of days to finalize stack.hfb.
            self.stack.hfb[:] *= tools.invert_no_zero(norm)

        else:
            # For inverse variance weighting, the weight dataset doesn't have to
            # be normalized (it is simply the sum of the weights). The accumulated
            # data have to be normalized by the sum of the weights.
            self.stack.hfb[:] *= tools.invert_no_zero(self.stack.weight[:])

        return self.stack


class MakeHighFreqRes(tasklib.base.ContainerTask):
    """Combine frequency and sub-frequency axes."""

    def process(self, stream):
        """Convert HFBData to HFBHighResData container.

        Parameters
        ----------
        stream : HFBData, HFBTimeAverage
            Data with frequency and subfrequency axis

        Returns
        -------
        out : HFBHighResData, HFBHighResTimeAverage
            Data with single high-resolution frequency axis
        """
        contmap = {
            containers.HFBData: containers.HFBHighResData,
            containers.HFBTimeAverage: containers.HFBHighResTimeAverage,
        }

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

        # Retrieve beam and time axes, if available
        beam = stream._data["index_map"]["beam"][:]
        time = stream.time if hasattr(stream, "time") else None

        # Determine new shape of data with combined frequency axis
        nfreq = len(cfreq)
        nsubfreq = len(subfreq)
        old_shape = data.shape
        new_shape = (nfreq * nsubfreq, *old_shape[2:])

        # Combine frequency and sub-frequency axes
        data = data.reshape(new_shape)
        weight = weight.reshape(new_shape)

        # Get the output container
        out_cont_type = contmap[stream.__class__]

        # Create container to hold output
        out = out_cont_type(freq=freq, beam=beam, time=time, attrs_from=stream)

        # Record sub-frequencies per coarse channel so downstream tasks can
        # recover the coarse-channel structure of the combined freq axis.
        out.attrs["nsubfreq"] = nsubfreq

        # Save data to output container
        out.hfb[:] = data
        out.weight[:] = weight

        # Return output container
        return out


class MakeHighFreqResRingMap(tasklib.base.ContainerTask):
    """Combine frequency and sub-frequency axes and reorder axes for HFB ringmaps."""

    def process(self, stream):
        """Convert HFBRingMap to HFBHighResRingMap container.

        Parameters
        ----------
        stream : HFBRingMap
            Ringmap with frequency and subfrequency axis

        Returns
        -------
        out : HFBHighResRingMap
            Ringmap with single high-resolution frequency axis
        """
        # Change distributed axis from freq to el
        stream.redistribute("el")

        # Retrieve data and weights
        data = stream.hfb[:]
        weight = stream.weight[:]

        # Make combined frequency axis
        cfreq = stream.index_map["freq"]["centre"]
        subfreq = stream.index_map["subfreq"]
        freq = (cfreq[:, np.newaxis] + subfreq).flatten()

        # Retrieve beam and RA axes
        beam = stream.beam_ew
        ra = stream.ra

        # Move frequency and subfrequency axes to the back
        axes = (2, 3, 4, 0, 1)
        data = data.transpose(axes)
        weight = weight.transpose(axes)

        # Determine new shape of data with combined frequency axis,
        # giving a length of None to the (distributed) el axis
        nbeam = len(beam)
        nra = len(ra)
        nfreq = len(cfreq)
        nsubfreq = len(subfreq)
        new_shape = (nbeam, None, nra, nfreq * nsubfreq)

        # Combine frequency and sub-frequency axes
        data = data.reshape(new_shape)
        weight = weight.reshape(new_shape)

        # Create container to hold output
        out = containers.HFBHighResRingMap(
            axes_from=stream, attrs_from=stream, freq=freq
        )

        # Record sub-frequencies per coarse channel so downstream tasks can
        # recover the coarse-channel structure of the combined freq axis.
        out.attrs["nsubfreq"] = nsubfreq

        # Save data to output container
        out.hfb[:] = data
        out.weight[:] = weight

        # HACK: If the `el` axis is contains fewer elements than there are MPI processes,
        # redistribute along new (longer) `freq` axis to avoid write-out bug
        if len(out.el) < mpitools.size:
            out.redistribute("freq")

        # Return output container
        return out


class FlattenPFB(tasklib.base.ContainerTask):
    """Flatten HFB data using PFB deconvolution."""

    def process(self, stream):
        """Flatten HFB data.

        Parameters
        ----------
        stream : containers.HFBData
            Container with HFB data.

        Returns
        -------
        out : containers.HFBData
            Container with flattened HFB data.
        """
        # Extract data and weight from container
        data = stream.hfb
        weight = stream.weight

        # Mask out DC bin (subfrequency bin 64) by setting weight to zero
        weight[:, 64, :, :] = 0

        # Instantiate PFB-deconvolution class with default parameters
        pfb_deconvolve = DeconvolvePFB()

        # Create output container
        out = containers.HFBData(axes_from=stream, attrs_from=stream)

        # Flatten one beam at a time
        for ibeam, _ in enumerate(stream._data["index_map"]["beam"][:]):
            # Select data and weights for beam
            data_beam = data[:, :, ibeam, :]
            weight_beam = weight[:, :, ibeam, :]

            # Apply PFB deconvolution
            flat_data, flat_weight = pfb_deconvolve.flatten(
                data_beam, weight_beam, centered=True
            )

            # Add flattened data and weights to output container
            out.hfb[:, :, ibeam, :] = flat_data
            out.weight[:, :, ibeam, :] = flat_weight

        return out


class FlattenRingMapPFB(tasklib.base.ContainerTask):
    """Flatten HFB ringmap using PFB deconvolution."""

    def process(self, stream):
        """Flatten HFB ringmap.

        Parameters
        ----------
        stream : containers.HFBRingMap
            Container with HFB ringmap.

        Returns
        -------
        out : containers.HFBRingMap
            Container with flattened HFB ringmap.
        """
        # Extract data and weight from container
        data = stream.hfb[:].local_array
        weight = stream.weight[:].local_array

        # Mask out DC bin (subfrequency bin 64) by setting weight to zero
        weight[:, 64, :, :] = 0

        # Instantiate PFB-deconvolution class with default parameters
        pfb_deconvolve = DeconvolvePFB()

        # Create output container
        out = containers.HFBRingMap(axes_from=stream, attrs_from=stream)

        # Flatten one EW beam and one elevation at a time
        for ibeam, _ in enumerate(stream.beam_ew):
            for iel, _ in enumerate(stream.el):
                # Select data and weights for beam
                data_beam = data[:, :, ibeam, iel, :]
                weight_beam = weight[:, :, ibeam, iel, :]

                # Apply PFB deconvolution
                flat_data, flat_weight = pfb_deconvolve.flatten(
                    data_beam, weight_beam, centered=True
                )

                # Add flattened data and weights to output container
                out.hfb[:, :, ibeam, iel, :] = flat_data
                out.weight[:, :, ibeam, iel, :] = flat_weight

        return out


class DividePFB(tasklib.base.ContainerTask):
    """Flatten HFB data by dividing out the PFB shape."""

    def process(self, stream):
        """Flatten HFB data.

        Parameters
        ----------
        stream : containers.HFBTimeAverage or containers.HFBRingMap
            Container with time-averaged HFB data.

        Returns
        -------
        out : same as stream
            Container with flattened time-averaged HFB data.
        """
        # Extract data and weight from container
        data = stream.hfb[:].local_array
        weight = stream.weight[:].local_array

        # Mask out DC bin (subfrequency bin 64) by setting weight to zero
        weight[:, 64, ...] = 0

        # Get PFB shape, instantiating PFB-deconvolution class with default parameters
        pfb_shape = DeconvolvePFB().Wt2.sum(axis=1)
        pfb_shape = np.roll(pfb_shape, 64)

        # Prepare pfb_shape for broadcasting
        if isinstance(stream, containers.HFBTimeAverage):
            pfb_shape = pfb_shape[:, np.newaxis]
        elif isinstance(stream, containers.HFBRingMap):
            pfb_shape = pfb_shape[:, np.newaxis, np.newaxis, np.newaxis]
        else:
            raise TypeError(
                "Task DividePFB expects HFBTimeAverage or HFBRingMap. "
                f"Got container of type {type(stream)}"
            )

        # Create container to hold output
        out = empty_like(stream)

        # Divide data by PFB shape and place in output container
        out.hfb[:] = data / pfb_shape

        # Divide uncertainties by PFB shape, hence multiply weights by square
        # of PFB shape, and place in output container.
        out.weight[:] = weight * pfb_shape**2

        return out


class DivideByTemplate(tasklib.base.ContainerTask):
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


class Difference(tasklib.base.ContainerTask):
    """Take the difference of two sets of HFB data.

    Used for flattening sub-frequency band shape by differencing on-source and
    off-source data.
    """

    def process(self, minuend, subtrahend):
        """Take difference and place in container.

        Parameters
        ----------
        minuend : HFBData or HFBTimeAverage
            Container with HFB data and weights; the data being subtracted from.

        subtrahend : HFBData or HFBTimeAverage
            Container with HFB data and weights; the data being subtracted.

        Returns
        -------
        out : HFBData or HFBTimeAverage
            Container with HFB data and weights; the result of the subtraction.
        """
        # Difference data
        data = minuend.hfb[:] - subtrahend.hfb[:]

        # Add standard deviations in quadrature, hence add variances, hence
        # take inverse of sum of reciprocals of weights
        weight = tools.invert_no_zero(
            tools.invert_no_zero(minuend.weight[:])
            + tools.invert_no_zero(subtrahend.weight[:])
        )

        # Create container to hold output
        out = empty_like(minuend)

        # Save data and weights to output container
        out.hfb[:] = data
        out.weight[:] = weight

        return out


class OnOffDifference(tasklib.base.ContainerTask):
    """Computes on-off differencing.

    Used for flattening sub-frequency band shape by differencing on-source and
    off-source data.

    Attributes
    ----------
    offset : int
        Number of samples on either side of the on-source sample to be ignored.
    nsamples : int
        Number of off-source samples on either side of the on-source sample to
        be averaged.
    """

    offset = config.Property(proptype=int, default=5)
    nsamples = config.Property(proptype=int, default=5)

    def process(self, stream):
        """Takes the average of off-source data and subtract it from on-source data.

        Parameters
        ----------
        stream : HFBHighResRingMap
            Container for HFB data and weights with a single high-resolution frequency
            axis.

        Returns
        -------
        out : HFBHighResRingMap
            Container with HFB data and weights; the result of the on-off differencing.
        """
        stream.redistribute("el")

        # Load data and weights
        data = stream.hfb[:]
        weight = stream.weight[:]

        # Change data and weights to numpy array, so that it can be reshaped
        if isinstance(data, mpiarray.MPIArray):
            data = data.local_array
            weight = weight.local_array

        ra = stream.ra[:]
        nra = len(ra)
        # Create a 1D kernel to select off-source data.
        kernel = np.zeros_like(ra)
        # Fill desired elements with one. nsample elements on either sides
        left = np.arange(self.offset + 1, self.offset + self.nsamples + 1)
        right = np.arange(nra - self.offset - self.nsamples, nra - self.offset)
        kernel[left] = 1
        kernel[right] = 1
        # We want the weighted average of off-source data, which is
        # sum(off-source data * weights) / sum(off-source weights)

        # Take the weighted sum of off-source data (numerator of the
        # weighted average). To do that, convolve the kernel above
        # with the data
        ker_fft = np.fft.fft(kernel)
        dat_fft = np.fft.fft(data * weight, axis=2)
        sum_data = np.fft.ifft(
            dat_fft * ker_fft[np.newaxis, np.newaxis, :, np.newaxis], axis=2
        ).real

        # Then, convolve the kernel above with the weights to find the sum
        # of off-source weights (denominator of the weighted average)
        weight_fft = np.fft.fft(weight, axis=2)

        # n is the sum of off-source weights over ra axis
        sum_weight = np.fft.ifft(
            weight_fft * ker_fft[np.newaxis, np.newaxis, :, np.newaxis], axis=2
        ).real

        # If the weight is zero, ifft above returns very small, but non-zero
        # value for n. This number goes to the denominator of weighted mean
        # and makes the weighted mean very large. To avoid such numerical
        # error (10^-16), zero n whereever weight is zero.
        sum_weight[weight == 0] = 0

        # Weighted average of off-source data
        off = sum_data * tools.invert_no_zero(sum_weight)

        # Weighted average of on-source data
        on = data * weight * tools.invert_no_zero(weight)

        # And on-off difference is:
        on_off = on - off

        # Now, evaluate the weight of on-off differenced data

        # Note that this only returns the weights corresponding to
        # inverse variance weighted data
        # TODO: Compute the weights corresponding to uniform average of data
        # On-source variance
        var = tools.invert_no_zero(weight)

        # Weight of average of off-source data is the sum over weights
        # That sum if given by n. So variance is 1/n
        var_off = tools.invert_no_zero(sum_weight)

        # variance of on-off data
        var_diff = var + var_off
        var_diff[var == 0] = 0

        # weight of on-off data
        weight_diff = tools.invert_no_zero(var_diff)

        # Create container to hold output
        out = containers.HFBHighResRingMap(axes_from=stream, attrs_from=stream)

        # Save data and weights to output container
        out.hfb[:] = on_off
        out.weight[:] = weight_diff

        return out


class MedianSubtraction(tasklib.base.ContainerTask):
    """Subtract weighted median along beam axis.

    This is to remove fluctuations in the data induced by temperature
    fluctuations in East and West receiver huts.
    """

    def process(self, stream):
        """Subtract weighted median of the data along beam axis from the data.

        A binary mask (0 when the weight is zero, and 1 for non-zero weights)
        is used in weighted median.

        Parameters
        ----------
        stream : containers.HFBData
            Container with HFB data and weights.

        Returns
        -------
        out : containers.HFBData
            Container with HFB data and weights.
        """
        # Extract data from container
        data = stream.hfb[:]
        weight = stream.weight[:]

        # Change data to numpy array, so that it can be reshaped
        if isinstance(data, mpiarray.MPIArray):
            data = data.local_array
            weight = weight.local_array

        # make a mask of non-zero weights
        mask = weight != 0

        # Generate binary weight
        binary_weight = mask.astype(np.float32)

        # Change the order of axes in data and mask arrays, as weighted median
        # is calculated along the last axis
        data_s = np.swapaxes(data, 2, 3)
        binary_weight = np.swapaxes(binary_weight, 2, 3)

        # Calculate weighted median (to exclude flagged data) along beam axis:
        med = median.weighted_median(data_s, binary_weight)

        # Sjubtract weighted median along all beams from the data
        diff = data - med[:, :, np.newaxis, :]

        # Create container to hold output
        out = containers.HFBData(stream)

        # Place diff in output container
        out.hfb[:] = diff
        out.weight[:] = weight

        return out


class AlignEWBeams(tasklib.base.ContainerTask):
    """Shift HFB ringmap data to true RA values in order to align EW beams."""

    def setup(self):
        """Load offsets and reference angles from CHIME/FRB beam model."""
        from beam_model.formed import FFTFormedActualBeamModel

        # Get the offsets in CHIME/FRB x coord (in deg) of the EW beams and the
        # reference zenith angles (CHIME/FRB y coord; in deg) of the NS beams
        # from the CHIME/FRB beam model
        beam_mdl = FFTFormedActualBeamModel()
        self.ew_beam_offset_deg = beam_mdl.config["ew_spacing"]
        self.ns_reference_angles_deg = beam_mdl.reference_angles

    def process(self, stream):
        """Align EW beams.

        Parameters
        ----------
        stream : containers.HFBHighResRingMap
            HFB ringmap container to align.

        Returns
        -------
        out : containers.HFBHighResRingMap
            HFB ringmap container with EW beams aligned in RA.
        """
        data = stream.hfb[:]
        weight = stream.weight[:]

        # Find CHIME/FRB XY coordinates of beams
        x_beam_list = self.ew_beam_offset_deg[stream.beam_ew]
        y_beam_list = self.ns_reference_angles_deg[stream.beam_ns]

        # Compute hour angles of beams
        x_beam_grid, y_beam_grid = np.meshgrid(x_beam_list, y_beam_list)
        ha_beam, _ = bmxy_to_hadec(x_beam_grid, y_beam_grid)

        # Initialize arrays to hold data and weights
        data_aligned = np.zeros_like(data)
        weight_aligned = np.zeros_like(weight)

        for insb, nsb in enumerate(stream.beam_ns):
            for iewb, ewb in enumerate(stream.beam_ew):
                # Compute actual RA of samples per beam
                ra_true = stream.ra - ha_beam[insb, ewb]

                # Do alignment by evaluating the data and weight at the RAs
                # given by the `ra` axis of the container. Do this using linear
                # interpolation with cyclic wrapping in RA, taking into account
                # uncertainties and flagged data.
                (
                    data_aligned[iewb, insb, :, :],
                    weight_aligned[iewb, insb, :, :],
                ) = interpolation_linear(
                    x=ra_true,
                    y=data[iewb, insb, :, :],
                    w=weight[iewb, insb, :, :],
                    xeval=stream.ra,
                    mode="wrap",
                    xperiod=360.0,
                )

        # Create output container; add data and weights
        out = containers.HFBHighResRingMap(copy_from=stream)
        out.data[:] = data_aligned
        out.weight[:] = weight_aligned

        return out


class DopplerShift(tasklib.base.ContainerTask):
    """Correct HFB data for Doppler shifts due to Earth's motion and rotation.

    Attributes
    ----------
    source_name : str
        Name of source, which should be in `ch_util.hfbcat.HFBCatalog`. If this is
        not provided, the task will look for it in the input container attributes.
    source_ra : float
        Right ascension of the source in degrees, in case the source is not in
        `ch_util.hfbcat.HFBCatalog`.
    source_dec : float
        Declination of the source in degrees, in case the source is not in
        `ch_util.hfbcat.HFBCatalog`.
    time_override : float
        Unix time used to calculate Doppler shift, to override the time calculated
    from the LSD and the RA of the source found in the container attributes.
    """

    source_name = config.Property(proptype=str, default=None)
    source_ra = config.Property(proptype=float, default=None)
    source_dec = config.Property(proptype=float, default=None)
    time_override = config.Property(proptype=float, default=None)

    source = None

    def setup(self, observer=None):
        """Setup the DopplerShift task.

        Parameters
        ----------
        observer : caput.time.Observer, optional
            Details of the observer, if not set default to CHIME.
        """
        # Set up the default Observer
        if observer is None:
            from ch_ephem.observers import chime as observer

        self.observer = observer

    def process(self, stream):
        """Doppler shift a container with high-resolution HFB data.

        Parameters
        ----------
        stream : HFBHighResData, HFBHighResTimeAverage, HFBHighResSpectrum
            Data with high-resolution frequency axis.

        Returns
        -------
        out : HFBHighResData, HFBHighResTimeAverage, HFBHighResSpectrum
            Doppler shifted data.
        """
        # On the first pass, obtain a skyfield Star object of the source. This can
        # either be retrieved from the HFB catalog (using the source name from the
        # container attributes, unless manually overridden via the task's source_name
        # attribute) or generated from the task's source_ra / source_dec attributes.
        if not self.source:
            if not self.source_ra and not self.source_dec:
                if not self.source_name:
                    self.source_name = stream.attrs["source_name"]
                self.source = HFBCatalog[self.source_name].skyfield
            elif self.source_ra and self.source_dec:
                self.source = Star(
                    ra=Angle(degrees=self.source_ra), dec=Angle(degrees=self.source_dec)
                )
            else:
                raise RuntimeError(
                    "Requires either a source name or both an RA and a Dec. "
                    f"Got source name {self.source_name}, RA {self.source_ra}, "
                    f"Dec {self.source_dec}"
                )

        # Obtain time used to compute Doppler correction from container LSD and source RA,
        # unless an override is provided.
        if self.time_override is not None:
            time = self.time_override
        else:
            if isinstance(stream.attrs["lsd"], list):
                raise TypeError(
                    f"Container includes multiple LSDs: {stream.attrs['lsd']} "
                    "Use time_override to force Doppler correction."
                )
            lsd_float = stream.attrs["lsd"] + self.source.ra.hours / 24.0
            time = self.observer.lsd_to_unix(lsd_float)

        # Extract frequencies, data, and weights from input container.
        freq_obs = stream.freq
        data_obs = stream.hfb[:]
        weight_obs = stream.weight[:]

        # Obtain Doppler shifted frequencies.
        freq_shifted = get_doppler_shifted_freq(
            source=self.source,
            date=time,
            freq_rest=freq_obs,
            obs=self.observer,
        ).squeeze()

        # Do Doppler shift by evaluating the data at the Doppler shifted frequencies
        # using linear interpolation and moving these to the observed frequencies.
        # Reverse frequency dimension, because the interpolation function requires
        # ascending x arrays. Zero out data points that need to be extrapolated.
        data_shifted, weight_shifted = interpolation_linear(
            x=freq_obs[::-1],
            y=data_obs[::-1, ...],
            w=weight_obs[::-1, ...],
            xeval=freq_shifted[::-1],
            mode="zero",
        )

        # Reverse back frequency dimension.
        data_shifted = data_shifted[::-1, ...]
        weight_shifted = weight_shifted[::-1, ...]

        # Create container to hold output.
        out = empty_like(stream)

        out.hfb[:] = data_shifted
        out.weight[:] = weight_shifted

        return out
