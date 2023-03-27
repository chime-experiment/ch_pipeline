"""Tasks for HFB analysis
"""
import numpy as np

from caput import config
from caput import mpiarray

from draco.core import task
from draco.util import tools
from draco.core import containers as dcontainers

from beam_model.composite import FutureMostAccurateCompositeBeamModel

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
    tag : str (default: "stack")
        The tag to give the stack.
    weighting: str (default: "inverse_variance")
        The weighting to use in the stack.
        Either `uniform` or `inverse_variance`.
    """

    stack = None

    tag = config.Property(proptype=str, default="stack")
    weighting = config.enum(["uniform", "inverse_variance"], default="inverse_variance")

    def process(self, sdata):
        """Add weights and data of one day to stack.

        Parameters
        ----------
        sdata : hfb.containers.HFBHighResSpectrum
            Individual (time-averaged) day to add to stack.
        """

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
            self.stack = dcontainers.empty_like(sdata)

            # Add stack-specific dataset: count of samples, to be used as weight
            # for the uniform weighting case. Initialize this dataset to zero.
            if "nsample" not in self.stack.datasets:
                self.stack.add_dataset("nsample")
                self.stack.nsample[:] = 0

            self.stack.redistribute("freq")

            self.lsd_list = []

        # Accumulate
        self.log.info("Adding to stack LSD(s): %s" % input_lsd)

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
        stack : hfb.containers.HFBHighResSpectrum
            Stack of sidereal days.
        """
        self.stack.attrs["tag"] = self.tag
        self.stack.attrs["lsd"] = np.array(self.lsd_list)

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

        # Add list of beams to attributes
        out.attrs["beam"] = stream._data["index_map"]["beam"][:]

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
    ra : float
        Right ascension of the source.
    dec : float
        Declination of the source.
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
    ra = config.Property(proptype=float, default=None)
    dec = config.Property(proptype=float, default=None)
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
                self.ra, self.dec, time_sample
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


def _ensure_list(x):
    if hasattr(x, "__iter__"):
        y = [xx for xx in x]
    else:
        y = [x]

    return y
