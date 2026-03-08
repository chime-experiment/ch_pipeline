"""Tasks for HFB analysis."""

import numpy as np
from beam_model.composite import FutureMostAccurateCompositeBeamModel
from caput import config, mpiarray
from caput.algorithms import median
from caput.containers import copy_datasets_filter, empty_like
from caput.pipeline import tasklib
from caput.util import mpitools
from ch_util.hfbcat import HFBCatalog, get_doppler_shifted_freq
from draco.util import tools
from skyfield.positionlib import Angle
from skyfield.starlib import Star

from . import containers
from .io import BeamSelectionMixin
from .pfb import DeconvolvePFB


class HFBAverage(tasklib.base.ContainerTask):
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

        # Save data to output container
        out.hfb[:] = data
        out.weight[:] = weight

        # HACK: If the `el` axis is contains fewer elements than there are MPI processes,
        # redistribute along new (longer) `freq` axis to avoid write-out bug
        if len(out.el) < mpitools.size:
            out.redistribute("freq")

        # Return output container
        return out


class HFBDivideByTemplate(tasklib.base.ContainerTask):
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


class HFBAlignEWBeams(tasklib.base.ContainerTask):
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
        from ch_ephem.coord import bmxy_to_hadec

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
                ) = _interpolation_linear(
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


class HFBDifference(tasklib.base.ContainerTask):
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


class HFBOnOffDifference(tasklib.base.ContainerTask):
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


class HFBStackDays(tasklib.base.ContainerTask):
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

        input_lsd = _ensure_list(input_lsd)

        # If this is our first sidereal day, then initialize the
        # container that will hold the stack.
        if self.stack is None:
            self.stack = empty_like(sdata)

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


class HFBSelectTransit(tasklib.base.ContainerTask):
    """Find transit data through FRB beams.

    Task for selecting on- and off-source data. The task can do both, but only
    one at a time (which one is set by the `selection` attribute). The selection
    of data is determined by the following criteria (a single one or both can be
    used, as specified by the `criteria` attribute):

    1) The offset in RA of the time samples with respect to the nominal centre of
       the synthetic beam being considered. This results in a window of data that
       is includes (in the case of on-source selection) or excluded (in the case
       of off-source selection). The size of the window is measured in units of
       the half-width of the synthetic beams in the EW direction, and set by the
       `on_source_include_window` and `off_source_exclude_window` attributes.
    2) The sensitivity of the synthetic beam toward the source. This must be greater
       than some threshold (in the case of on-source selection) or lower than some
       ceiling (in the case of off-source selection). The threshold/ceiling is set
       by the `on_source_sens_threshold` and `off_source_sens_ceiling` attributes,
       as a fraction of the maximum sensitivity in the beam's main lobe. (The
       sensitivity of the last row of EW beams, EW index 3, is higher in the side
       lobes than the main lobe. If we were to use the maximum sensitivity of the
       beam per se to calculate the threshold needed for data selection, selected
       data won't be in the main lobe.)

    Attributes
    ----------
    selection : str
        What part of the data to select. Options are: 'on-source', 'off-source'.
        Default is 'on-source'
    criteria : list of strings
        Criteria used to decide which time samples are selected. Possible entries:
        'offset' (i.e., the RA offset of the source to the centre of the beam),
        'sensitivity' (i.e., the synthetic beam sensitivity).
        Default is ['offset', 'sensitivity']
    on_source_include_window : float
        Half-width of the window to include when selecting on-source data in units
        of the EW half-width of the synthetic beams (beam and frequency dependent).
        Default is 1.0
    on_source_sens_threshold : float
        Fraction of maximum beam sensitivity above which samples are considered
        on-source.
        Default is 0.6
    off_source_exclude_window : float
        Half-width of the window to exclude when selecting off-source data in units
        of the EW half-width of the synthetic beams (beam and frequency dependent).
        A value of 10.0 excludes the first main sidelobe of the synthetic beams.
        A value of 18.0 excludes a further minor sidelobe, at the cost of requiring
        larger chunks of data around transit, especially at lower frequecies.
        Default is 18.0
    off_source_sens_ceiling : float
        Fraction of maximum beam sensitivity below which samples are considered
        off-source.
        Default is 0.005
    source_name : str
        Name of source, which should be in `ch_util.hfbcat.HFBCatalog`. If this is
        not provided, the task will look for it in the input container attributes.
    source_ra : float
        Right ascension of the source in degrees, in case this is not available in
        `ch_util.hfbcat.HFBCatalog` (or in case the value given there needs to be
        overridden manually).
    source_dec : float
        Declination of the source in degrees, in case this is not available in
        `ch_util.hfbcat.HFBCatalog` (or in case the value given there needs to be
        overridden manually).
    time_offset : float
        Time (is seconds) to add to the data times before comparison with the
        beam model. This is a temporary hack, to be removed when the data times
        are fully understood.
        Default is 0.0
    """

    selection = config.enum(["on-source", "off-source"], default="on-source")
    criteria = config.list_type(type_=str, default=["offset"])
    on_source_include_window = config.Property(proptype=float, default=1.0)
    on_source_sens_threshold = config.Property(proptype=float, default=0.6)
    off_source_exclude_window = config.Property(proptype=float, default=18.0)
    off_source_sens_ceiling = config.Property(proptype=float, default=0.005)
    source_name = config.Property(proptype=str, default=None)
    source_ra = config.Property(proptype=float, default=None)
    source_dec = config.Property(proptype=float, default=None)
    time_offset = config.Property(proptype=float, default=0.0)

    def setup(self):
        """Check criteria attribute."""
        if not self.criteria:
            raise ValueError("No selection criteria set.")

        available_criteria = ["offset", "sensitivity"]
        for criterion in self.criteria:
            if criterion not in available_criteria:
                raise ValueError(
                    f"Unknown selection criterion passed ({criterion})."
                    f"Available criteria are: {available_criteria}"
                )

    def process(self, stream):
        """Find transit data for the given beams.

        Parameters
        ----------
        stream : containers.HFBData
            Container with HFB data and weights.

        Returns
        -------
        out : containers.HFBData
            Container with same HFB data, but weights adjusted to make selection.
        """
        # On the first pass, obtain the source's coordinates from the HFB catalog,
        # unless they are provided manually via the task's source_ra / source_dec
        # attributes. To do this, get the source name from the container attributes,
        # unless manually overridden via the task's source_name attribute.
        if not self.source_ra or not self.source_dec:
            if not self.source_name:
                self.source_name = stream.attrs["source_name"]
            hfb_cat = HFBCatalog[self.source_name]
            if not self.source_ra:
                self.source_ra = hfb_cat.ra
            if not self.source_dec:
                self.source_dec = hfb_cat.dec

        # Extract axes lengths
        nfreq, nsubfreq, nbeam, ntime = stream.hfb.shape

        # Extract physical frequencies by combining CHIME frequency channels and
        # HFB subfrequencies into one vector
        freq = stream._data["index_map"]["freq"]["centre"][:]
        subfreq = stream._data["index_map"]["subfreq"]
        phys_freq = (freq.reshape(nfreq, 1) + subfreq).flatten()

        # Extract beam indices, then change format for sensitivity calculations.
        # The beam_model package uses a format (referred to here as `beam_number`)
        # where the EW beam index is indicated by thousands, and the NS beam index
        # by the remaining digits (a number ranging form 0 and 255). For example,
        # the indices [12, 268, 524, 780] will be changed to [12, 1012, 2012, 3012]
        beam_index = stream._data["index_map"]["beam"][:]
        dtype = beam_index.dtype
        nbeam_ns = 256
        beam_number = (
            beam_index % nbeam_ns + 1000 * np.floor(beam_index / nbeam_ns)
        ).astype(dtype)

        # Extract time array. HACK: Possibly add offset to time.
        time = stream.time[:] + self.time_offset

        # Load beam model used for sensitivity calculations, for looking up beam
        # positions and beam widths, and for finding source positions in beam-model
        # xy-coordinates. Set `interpolate_bad_freq=True` to avoid issues at
        # frequencies where the data-driven primary-beam model lacks data.
        beam_mdl = FutureMostAccurateCompositeBeamModel(interpolate_bad_freq=True)

        # Obtain the track of the source in beam-model xy-coordinates from its
        # equatorial position and time. The array has shape (ntime, 2).
        posxy_source = np.zeros((ntime, 2))
        for itime, time_sample in enumerate(time):
            posxy_source[itime, :] = beam_mdl.get_position_from_equatorial(
                self.source_ra, self.source_dec, time_sample
            )

        # Calculate the centre positions of the synthetic beams and their widths
        # (FWHM in the EW and NS directions) in beam-model xy-coordinates for all
        # combinations of beam and frequency. The shape of both resulting arrays
        # is (nbeam, nfreq, 2), with the final dimension giving x and y.
        posxy_beams = beam_mdl.get_beam_positions(beam_number, phys_freq)
        beam_width = beam_mdl.get_beam_widths(beam_number, phys_freq)

        # Compute the offset between the source position and the centres of the
        # synthetic beams in terms of beam model x coordinate. Make sure this
        # array gets axes (time, beam, phys_freq).
        x_offset = np.abs(
            posxy_source[:, 0][:, np.newaxis, np.newaxis] - posxy_beams[:, :, 0]
        )

        # Create a boolean array that selects the main lobe of the synthetic beam.
        # It is defined as time samples that are within `on-source-include-window`
        # synthetic beam EW half-widths of the centre of the beam. The array has axes
        # (time, beam, phys_freq).
        main_lobe = x_offset < self.on_source_include_window * beam_width[:, :, 0] / 2

        # Compute beam sensitivities only if necessary
        if "sensitivity" in self.criteria:
            # Calculate the (composite) beam sesitivities for all combinations of
            # source position (i.e., time), beam, and frequency, resulting in an
            # array of size (ntime, nbeam, nfreq * nsubfreq).
            sens = beam_mdl.get_sensitivity(beam_number, posxy_source, phys_freq)

            # Find the maximum sensitivity within the main lobe of the beam.
            # The resulting array has axes (beam, phys_freq)
            sens_only_main_lobe = sens * main_lobe.astype(float)
            max_sens = sens_only_main_lobe.max(axis=0)

        # Create a boolean mask that selects certain samples of the data:
        # True for selected samples, False for non-selected samples.
        # Initialize the array as all True, then set parts of it to False below
        # based on choosen criteria
        mask = np.full((ntime, nbeam, nfreq * nsubfreq), True)

        if self.selection == "on-source":
            # Make mask to select on-source data

            if "offset" in self.criteria:
                # Ensure that on-source data corresponds to time samples where the
                # source is inside the main lobe of the synthetic beam.
                mask &= main_lobe

            if "sensitivity" in self.criteria:
                # Ensure that on-source data corresponds to time samples where the
                # beam sensitivity is above the choosen threshold (i.e., at least
                # `on_source_sens_threshold` times the maximum sensitivity found
                # in the main lobe of the beam being considered).
                sens_threshold = self.on_source_sens_threshold * max_sens
                mask &= sens > sens_threshold[np.newaxis, :, :]

        elif self.selection == "off-source":
            # Make mask to select off-source data

            if "offset" in self.criteria:
                # Ensure that off-source data corresponds to time samples where
                # the source is far enough away from the synthetic beam (at least
                # `off_source_exclude_window` synthetic beam half-widths away from
                # the beam centre).
                x_offset_lim = self.off_source_exclude_window * beam_width[:, :, 0] / 2
                mask &= x_offset > x_offset_lim

            if "sensitivity" in self.criteria:
                # Ensure that off-source data corresponds to time samples where
                # the beam sensitivity is below the choosen threshold (at most
                # `off_source_sens_ceiling` times the maximum sensitivity found
                # in the main lobe of the beam being considered).
                sens_threshold = self.off_source_sens_ceiling * max_sens
                mask &= sens < sens_threshold[np.newaxis, :, :]

        # The mask created above has axes (time, beam, phys_freq). Convert this
        # back to the axes used in the hfb containers (freq, subfreq, beam, time):
        # First swap axes to get (phys_freq, beam, time). Then reshape the array
        # to convert phys_freq to freq and subfreq.
        mask = np.swapaxes(mask, 0, 2)
        mask = mask.reshape(nfreq, nsubfreq, nbeam, ntime)

        self.log.info(
            f"Average number of {self.selection} time samples selected: "
            f"{mask.sum(axis=3).mean():.1f}"
        )

        # Create container to hold output
        out = containers.HFBData(copy_from=stream)

        # Set weights of non-selected samples to zero
        weight = stream.weight[:] * mask.astype(float)
        out.weight[:] = weight

        # Return output container
        return out


class SelectBeam(BeamSelectionMixin, tasklib.base.ContainerTask):
    """Select a subset of EW and/or NS beams from a container.

    The selection is made by passing `beam_ew_include` and/or `beam_ns_index` or
    `beam_ns_range` attributes (see documentation in :class:`BeamSelectionsMixin`).
    """

    def setup(self):
        """Resolve the beam selection."""
        # Resolve the selections provided through the `beam_ew_include`,
        # `beam_ns_index`, and `beam_ns_range` attributes (via
        # `ch_pipeline.hfb.io.BeamSelectionsMixin`)
        self.beam_sel = self.resolve_beam_sel()

    def process(self, stream):
        """Select a subset of beams.

        Parameters
        ----------
        stream : containers.HFBBeamContainer
            An HFB container with a beam axis.

        Returns
        -------
        newstream : containers.HFBBeamContainer
            New container with a selection of beams.
        """
        # Create new container with subset of beams
        newstream = empty_like(stream, beam=self.beam_sel)

        # Make sure all datasets are initialised
        for name in stream.datasets.keys():
            if name not in newstream.datasets:
                newstream.add_dataset(name)

        # Find indices in current beam axis of selected subset of beams
        selindex = np.flatnonzero(np.isin(stream.beam, self.beam_sel)).tolist()

        # Copy over datasets
        copy_datasets_filter(stream, newstream, "beam", selindex)

        return newstream


class HFBFlattenPFB(tasklib.base.ContainerTask):
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


class HFBFlattenRingMapPFB(tasklib.base.ContainerTask):
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


class HFBDividePFB(tasklib.base.ContainerTask):
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
                "Task HFBDividePFB expects HFBTimeAverage or HFBRingMap. "
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


class HFBDopplerShift(tasklib.base.ContainerTask):
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
        """Setup the HFBDopplerShift task.

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
        data_shifted, weight_shifted = _interpolation_linear(
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


def _interpolation_linear(x, y, w, xeval, mode="zero", xperiod=None):
    """Linear interpolation with handling of uncertainties and flagged data.

    Approximates a 1-D function `y = f(x)` using linear interpolation, while
    taking into account the uncertainties on the known points of `y`.

    Parameters
    ----------
    x : array_like
        A 1-D array of real values.
    y : array_like
        An N-D array of real values. The first axis is the interpolation axis
        and must be the same length as `x`.
    w : array_like
        An N-D array of weights, the same shape as `y`, giving the inverse
        variance of each point in `y`.
    xeval : array_like
        The points at which to evaluate the interpolation.
    mode : {"zero", "extrapolate", "wrap"}, optional
        The `mode` parameter determines the behaviour for values of `xeval` that
        fall outside the range of `x`. Default is "zero". The behaviour for each
        valid value is as follows:
        "zero"
            Returns zeros outside the range of `x`.
        "extrapolate"
            Extrapolates (linearly) using the first two or last two input values.
        "wrap"
            Treats the input as circular, wrapping around to the opposite edge.
            This requires the `xperiod` parameter to be set.
    xperiod : float, optional
        The period of `x`, in case the "wrap" mode is used (otherwise ignored).
    """
    # Invert weights to obtain variances
    var = tools.invert_no_zero(w)

    if mode == "wrap":
        # Check if xperiod, needed in the "wrap" mode, was passed
        if not xperiod:
            raise ValueError("xperiod needed if mode is 'wrap'")

        # Wrap xeval points over period
        xeval %= xperiod

    # Find indices of x points left and right of xeval points
    index = np.searchsorted(x, xeval, side="left")
    ind1 = index - 1
    ind2 = index

    if mode == "wrap":
        # Find points below the range of x and overwrite left index with the index
        # at the end of x
        below = np.flatnonzero(ind1 == -1)
        if below.size > 0:
            ind1[below] = x.size - 1

        # Find points above the range of x and overwrite right index with the index
        # at the strart of x
        above = np.flatnonzero(ind2 == x.size)
        if above.size > 0:
            ind2[above] = 0

    else:
        # Find points below the range of x and overwrite left and right indices
        # with the two indices at the start of x for extrapolation
        below = np.flatnonzero(ind1 == -1)
        if below.size > 0:
            ind1[below] = 0
            ind2[below] = 1

        # Find points above the range of x and overwrite left and right indices
        # with the two indices at the end of x for extrapolation
        above = np.flatnonzero(ind2 == x.size)
        if above.size > 0:
            ind1[above] = x.size - 2
            ind2[above] = x.size - 1

    # Compute intervals
    adx1 = xeval - x[ind1]
    adx2 = x[ind2] - xeval

    if mode == "wrap":
        # For points below and above the range of x, overwrite the interval,
        # adjusting for the period
        if below.size > 0:
            adx1[below] = xeval[below] - x[-1] - xperiod
        if above.size > 0:
            adx2[above] = x[0] + xperiod - xeval[above]

    # Compute relative weights of left and right points
    norm = tools.invert_no_zero(adx1 + adx2)
    a1 = adx2 * norm
    a2 = adx1 * norm

    # Adjust shape of a1 and a2 to allow broadcasting with y and var
    new_axes_pos = tuple(range(1, len(y.shape)))
    a1 = np.expand_dims(a1, axis=new_axes_pos)
    a2 = np.expand_dims(a2, axis=new_axes_pos)

    # Do interpolation and extrapolation
    yeval = a1 * y[ind1, ...] + a2 * y[ind2, ...]
    weval = tools.invert_no_zero(a1**2 * var[ind1, ...] + a2**2 * var[ind2, ...])

    # Set interpolated weight to zero if either of the two weights going into
    # the interpolation is zero, indicated data flagged as bad
    flags = (w[ind1, ...] == 0.0) | (w[ind2, ...] == 0.0)
    weval[flags] = 0.0

    if mode == "zero":
        # Overwrite extrapolated points with zeros
        outside = np.concatenate((below, above))
        yeval[outside] = 0.0
        weval[outside] = 0.0

    return yeval, weval


def _ensure_list(x):
    if hasattr(x, "__iter__"):
        y = list(x)
    else:
        y = [x]

    return y


class HFBMedianSubtraction(tasklib.base.ContainerTask):
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
