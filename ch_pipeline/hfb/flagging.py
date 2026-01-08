"""HFB Tasks for flagging data."""

import beam_model.formed as fm
import numpy as np
from caput import config
from caput.algorithms import invert_no_zero
from caput.pipeline import tasklib
from draco.analysis.sidereal import _search_nearest
from draco.core.containers import LocalizedRFIMask
from scipy.spatial.distance import cdist

from .containers import HFBDirectionalRFIMaskBitmap, HFBRFIMask


class HFBRadiometerRFIFlagging(tasklib.base.ContainerTask):
    """Identify RFI in HFB data using the radiometer noise test, averaging over beams.

    Attributes
    ----------
    threshold : float
        The desired threshold for the RFI fliter. Data with sensitivity metric
        values above this threshold will be considered RFI. Default is 2.0
    keep_sens : bool
        Save the sensitivity metric data that were used to construct
        the mask in the output container.
    """

    threshold = config.Property(proptype=float, default=2.0)
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
        # Radiometer noise test: the sensitivity metric would be unity for
        # an ideal radiometer, it would be higher for data with RFI
        sensitivities = sensitivity_metric(stream=stream, average_beams=True)

        # Boolean mask indicating data that are contaminated by RFI
        mask = sensitivities > self.threshold

        # Create container to hold output
        out = HFBRFIMask(axes_from=stream, attrs_from=stream)

        if self.keep_sens:
            out.add_dataset("sens")
            out.sens[:] = sensitivities

        # Save mask to output container
        out.mask[:] = mask

        # Add beam selection to RFI mask attributes
        out.attrs["beam"] = stream._data["index_map"]["beam"][:]

        # Return output container
        return out


class ApplyHFBMask(tasklib.base.ContainerTask):
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
            set to zero.
        """
        # flag = mask[:, :, np.newaxis, :]

        # Create a slice that will expand the mask to
        # the same dimensions as the weight array
        waxis = stream.weight.attrs["axis"]
        slc = [slice(None)] * len(waxis)
        for ww, name in enumerate(waxis):
            if name not in mask.mask.attrs["axis"]:
                slc[ww] = None

        # Extract mask, transform to regular numpy array
        # TODO: Changes needed when distributed reading work for HFBData
        flag = mask.mask[:].local_array

        # Expand mask to same dimension as weight array
        flag = flag[tuple(slc)]

        # Log how much data we're masking
        self.log.info(f"{(100.0 * np.mean(flag)):.2f} percent of data will be masked.")

        # Apply the mask
        if np.any(flag):
            # Apply the mask to the weights
            stream.weight[:] *= 1.0 - flag

            # If requested, apply the mask to the data
            if self.zero_data:
                stream.hfb[:] *= 1.0 - flag

        return stream


class HFBDirectionalRFIFlagging(tasklib.base.ContainerTask):
    """Produce a RFI mask based on HFB sensitivity values.

    This task detects RFI signals based on deviations from expected radiometer noise
    in the second East-West beams (beam IDs: 256-512) for multiple values of std.
    It produces a directional RFI mask container recording the number of HFB subfrequency
    channels flagged for RFI under each std.

    Attributes
    ----------
    beam_ew_id : int
        The E-W beam index to inspect for RFI detection. Default is 1. (beam IDs: 256-512)
    std : list of float
        List of exactly four standard deviation values used as thresholds in RFI detection.
        Each value corresponds to one of the four 8-bit segments packed into the 32-bit mask.
        These thresholds are used to compare against the sensitivity metric, and each one
        produces a separate mask encoding the number of HFB subfrequency channels flagged as RFI.
        The order of values determines their byte position: the first maps to bits 0-7, the second
        to 8-15, and so on. Default is [0.25, 0.275, 0.30, 0.325].
    sigma_threshold : float
        This is to detect RFI in HFB sensitivity values. If it exceeds
        1 + treshold * std (or std_avg), then it indicates RFI events. Default is 5.
    """

    beam_ew_id = config.Property(proptype=int, default=1)
    std = config.Property(proptype=list, default=[0.25, 0.275, 0.30, 0.325])
    sigma_threshold = config.Property(proptype=float, default=5)

    def process(self, stream):
        """Produce a directional RFI mask from HFB data.

        Parameters
        ----------
        stream : containers.HFBData
            Container with HFB data and weights.

        Returns
        -------
        out : containers.HFBDirectionalRFIMaskBitmap
            Container holding the RFI masks across different std values.
        """
        # Extract HFB data shape so we can reshape sensitivities to separate E-W and N-S beam axes
        nfreq, nsubfreq, _, ntime = stream.hfb[:].shape
        nbeam_ew = len(stream.beam_ew)
        nbeam_ns = len(stream.beam_ns)

        # Radiometer noise test: the sensitivity metric would be unity for
        # an ideal radiometer, it would be higher for data with RFI
        sensitivities = sensitivity_metric(stream=stream, average_beams=False)
        sensitivities = sensitivities.reshape(
            nfreq, nsubfreq, nbeam_ew, nbeam_ns, ntime
        )[
            :, :, self.beam_ew_id, :, :
        ]  # (nfreq, nsubfreq, nbeam_ns, ntime)

        # Create output container with the same axes
        out = HFBDirectionalRFIMaskBitmap(
            std_key=self.std,
            freq=stream.freq[:],
            beam_ns=stream.beam_ns[:],
            time=stream.time[:],
        )
        out.attrs["bitmap"] = {std: i for i, std in enumerate(self.std[:])}

        # For each std, compute RFI flags and store counts
        for i in range(len(self.std)):

            # RFI detection for individual E-W beams
            flags = sensitivities > 1.0 + self.sigma_threshold * self.std[i]

            # Summing over subfrequency channels
            count = np.sum(flags, axis=1)

            # Store counts into appropriate directional masks
            out.set_subfreq_rfi(self.std[i], count)

        # Return output container
        return out


class RFIMaskHFBRegridderNearest(tasklib.base.ContainerTask):
    """Convert HFBDirectionalRFIMaskBitmap axis from beam_ns to el.

    This task takes an HFBDirectionalRFIMaskBitmap, selects the RFI mask corresponding
    to a specific std value used in the detection, then regrids the mask from
    the beam_ns axis to an el axis.

    Notes
    -----
    - Before CSD 3685, the time axes of HFB and cosmology data were not synchronized.

    Attributes
    ----------
    std : float
        Must match one of the std keys used when the HFBDirectionalRFIMaskBitmap was
        created. Default is 0.25.
    subfreq_threshold : int
        Subfrequency threshold for decoding the bitmap. A sample is flagged if
        the number of HFB subfrequency channels detecting RFI >= this threshold.
        Default is 2.
    keep_frac_rfi : bool
        Save the fraction of HFB subfrequency channels (Out of 128) detecting RFI
        that were used to construct the mask, storing this as an additional dataset
        in the output container. Default is False.
    spread_factor : float
        Spreading factor for conservative flagging. Each flagged mask sample is
        expanded to neighboring target el values within
        'spread_factor * resolution' of the axis. Default is 1.0.
    npix : int
        The number of pixels used to cover the full elevation range from -1 to 1.
        Default is 512.
    """

    std = config.Property(proptype=float, default=0.25)
    subfreq_threshold = config.Property(proptype=int, default=2)
    keep_frac_rfi = config.Property(proptype=bool, default=False)
    spread_factor = config.Property(proptype=float, default=1)
    npix = config.Property(proptype=int, default=512)

    def process(self, rfimaskbitmap):
        """Convert beam_ns axis of an HFBDIrectionalRFIMaskBitmap to el axis.

        Parameters
        ----------
        rfimaskbitmap : containers.HFBDirectionalRFIMaskBitmap
            Input RFI mask with axes (freq, beam_ns, time).

        Returns
        -------
        out : containers.LocalizedRFIMask
            Converted mask with axes (freq, el, time).
        """
        # Ensure data is distributed in frequency
        rfimaskbitmap.redistribute("freq")

        # Convert beam_ns indices to elevation angles (sin(theta))
        v = fm.FFTFormedBeamModel()
        angles = v.get_beam_positions(rfimaskbitmap.beam_ns[:], rfimaskbitmap.freq[:])
        el = np.sin(np.deg2rad(angles.T[1]))  # shape (freq, beam_ns)

        # Create new elevation axis, restricting to overlapping region with original beams
        to_ax = np.linspace(-1, 1, self.npix)
        start = _search_nearest(to_ax, np.min(el))
        end = _search_nearest(to_ax, np.max(el))
        valid_range = slice(start, end + 1)
        new_ax = to_ax[valid_range]

        # Create output container with the new axes
        out = LocalizedRFIMask(
            freq=rfimaskbitmap.freq[:], el=new_ax, time=rfimaskbitmap.time[:]
        )
        if self.keep_frac_rfi:
            out.add_dataset("frac_rfi")

        # Determine local frequency slice for this MPI rank
        freq_ax = list(rfimaskbitmap.subfreq_rfi.attrs["axis"]).index("freq")
        nfreq_local = rfimaskbitmap.subfreq_rfi[:].local_array.shape[freq_ax]
        sf = rfimaskbitmap.subfreq_rfi.local_offset[freq_ax]
        ef = sf + nfreq_local

        # Process each frequency separately since beam-to-el mapping varies with freq
        for i in range(sf, ef):

            # Extract mask for this std and threshold
            mask = rfimaskbitmap.get_mask(self.std, self.subfreq_threshold)[
                :
            ].local_array[i - sf]

            # Optionally carry over the number of HFB subrequency channels detecting RFI
            if self.keep_frac_rfi:
                frac_rfi = rfimaskbitmap.get_frac_rfi(self.std)[:].local_array[i - sf]

            # Locate slice in new elevation axis
            el_start = _search_nearest(new_ax, out.el[0])
            el_end = _search_nearest(new_ax, out.el[-1]) + 1

            # Estimate resolutions to determine mapping strategy
            from_ax = el[i, :]
            new_resolution = np.median(np.abs(np.diff(new_ax)))
            from_resolution = np.median(np.abs(np.diff(from_ax)))

            # Determine nearest-neighbor indices for mapping
            if new_resolution < from_resolution:
                nearest_indices = _search_nearest(from_ax, new_ax)
            else:
                nearest_indices = np.arange(len(from_ax))

            # Compute pairwise distances between each new axis point and nearest from_ax points
            dist = cdist(
                new_ax[:, np.newaxis],
                from_ax[nearest_indices, np.newaxis],
                metric="euclidean",
            )

            # Disable spreading if axes align exactly (diagonal distance is zero)
            if np.all(np.diag(dist) == 0):
                self.spread_factor = 0

            # Construct conservative spreading window
            resolution = np.median(np.abs(np.diff(from_ax)))
            window = np.abs(dist) <= self.spread_factor * resolution

            # Interpolate the boolean mask
            converted = np.tensordot(window, mask[nearest_indices], axes=([1], [0])) > 0
            out.mask[:].local_array[i - sf, el_start:el_end, :] = converted

            # Interpolate the float frac_rfi
            if self.keep_frac_rfi:
                window = window.astype(np.float32)
                numerator = np.tensordot(
                    window, frac_rfi[nearest_indices], axes=([1], [0])
                )
                denominator = np.sum(window, axis=-1).reshape(
                    (-1,) + (1,) * (numerator.ndim - 1)
                )
                converted = numerator * invert_no_zero(denominator)
                out.frac_rfi[:].local_array[i - sf, el_start:el_end, :] = converted

        # Return output container
        return out


def sensitivity_metric(stream, average_beams):
    """Compute the sensitivity metric for the given HFB data.

    This function calculates sensitivity values for each CHIME frequency,
    HFB subfrequency, beam ID, and time sample in the provided HFB data.

    Parameters
    ----------
    stream : HFBData
        HFB Data to compute sensitivity metric.
    average_beams : bool
        If True, average the data and weights across all beams before computing
        the sensitivity metric, resulting in a shape of [freq, subfreq, time].
        If False, compute the sensitivity per beam, resulting in
        [freq, subfreq, beam, time].

    Returns
    -------
    sensitivity metric : np.ndarray
        An array of computed sensitivity values with shape:
        - [freq, subfreq, time] if average_beams is True
        - [freq, subfreq, beam, time] if average_beams is False
    """
    # Extract the data and weight arrays
    data = stream.hfb[:].view(np.ndarray)
    weight = stream.weight[:].view(np.ndarray)

    if average_beams:
        data = np.mean(data, axis=2)
        weight = np.mean(weight, axis=2)

    # delta_nu is the frequency resolution of HFB data (390.625 kHz / 128)
    freq_width = stream.index_map["freq"]["width"][0] * 1e6
    nsubfreq = data.shape[1]
    delta_nu = freq_width / nsubfreq

    # delta_t is the integration time (~10.066 s)
    delta_t = np.median(np.diff(stream.time[:]))

    # frac_lost is the fraction of integration that was lost upstream,
    # but it is not distributed, so we need to cut out the frequencies
    # that are local for the stream
    ax = list(stream.hfb.attrs["axis"]).index("freq")
    nfreq_local = stream.hfb.local_shape[ax]
    sf = stream.hfb.local_offset[ax]
    ef = sf + nfreq_local
    frac_lost = stream["flags/frac_lost"][sf:ef]

    # Number of samples per data point in the HFB data
    n_samp = delta_nu * delta_t * (1.0 - frac_lost)

    if average_beams:
        inv_n_samp = invert_no_zero(n_samp)[:, None, :]
    else:
        inv_n_samp = invert_no_zero(n_samp)[:, None, None, :]

    # The ideal radiometer equation
    radiometer = data**2 * inv_n_samp

    # Compute and return the sensitivity metric
    return 2.0 * invert_no_zero(radiometer * weight)
