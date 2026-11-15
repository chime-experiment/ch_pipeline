"""HFB Tasks for flagging data."""

import os

import beam_model.formed as fm
import numpy as np
import scipy.sparse as ss
from caput import config, mpiarray
from caput.algorithms import invert_no_zero
from caput.pipeline import exceptions, tasklib
from draco.analysis.sidereal import _search_nearest
from draco.core.containers import LocalizedRFIMask, RFIMask
from draco.core.io import get_telescope
from draco.util import regrid
from scipy.spatial.distance import cdist

from ch_pipeline.hfb import containers


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
        out = containers.HFBRFIMask(axes_from=stream, attrs_from=stream)

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
    in the second East-West beams (beam IDs: 256-512) for multiple values of significance.
    It produces a directional RFI mask container recording the number of HFB subfrequency
    channels flagged for RFI under each significance.

    Attributes
    ----------
    beam_ew_id : int
        The E-W beam index to inspect for RFI detection. Default is 1. (beam IDs: 256-512)
    sigma : list of float
        List of exactly four significance values used as thresholds in RFI detection.
        Each value corresponds to one of the four 8-bit segments packed into the 32-bit mask.
        These thresholds are used to compare against the sensitivity metric, and each one
        produces a separate mask encoding the number of HFB subfrequency channels flagged as RFI.
        The order of values determines their byte position: the first maps to bits 0-7, the second
        to 8-15, and so on. Default is [4, 5, 6, 10].
    std : float
        This is to detect RFI in HFB sensitivity values. If it exceeds
        1 + sigma * std, then it indicates RFI events. Default is 0.25.

    Notes
    -----
    The RFI detection threshold is evaluated separately for each value in the sigma list using
    the expression 1 + sigma * std. The default values (std = 0.25 and sigma = [4, 5, 6, 10])
    correspond to effective thresholds of [2.0, 2.25, 2.5, 3.5]. These values were chosen to
    provide multiple level of sensitivity within a single packed mask.
    """

    beam_ew_id = config.Property(proptype=int, default=1)
    sigma = config.Property(proptype=list, default=[4, 5, 6, 10])
    std = config.Property(proptype=float, default=0.25)

    def process(self, stream):
        """Produce a directional RFI mask from HFB data.

        Parameters
        ----------
        stream : containers.HFBData
            Container with HFB data and weights.

        Returns
        -------
        out : containers.HFBDirectionalRFIMaskBitmap
            Container holding the RFI masks across different significance values.
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
        out = containers.HFBDirectionalRFIMaskBitmap(
            sigma_key=self.sigma,
            freq=stream.freq[:],
            beam_ns=stream.beam_ns[:],
            time=stream.time[:],
        )
        out.attrs["bitmap"] = {sigma: i for i, sigma in enumerate(self.sigma[:])}

        # For each significance, compute RFI flags and store counts
        for i in range(len(self.sigma)):

            # RFI detection for individual E-W beams
            flags = sensitivities > 1.0 + self.std * self.sigma[i]

            # Summing over subfrequency channels
            count = np.sum(flags, axis=1)

            # Store counts into appropriate directional masks
            out.set_subfreq_rfi(self.sigma[i], count)

        # Return output container
        return out


class RFIMaskHFBRegridderNearest(tasklib.base.ContainerTask):
    """Convert HFBDirectionalRFIMaskBitmap axis from beam_ns to el.

    This task takes an HFBDirectionalRFIMaskBitmap, selects the RFI mask corresponding
    to a specified significance value used in the detection and subfrequency threshold,
    then regrids the mask from the beam_ns axis to an el axis.

    Attributes
    ----------
    sigma : float
        Specify the value of significance used in RFI detection to create a RFI mask.
        Must match one of the sigma keys used when the HFBDirectionalRFIMaskBitmap was
        created. Default is 5.
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
    remove_persistent_beamns_frac: float
        If non-zero, remove beam_ns rows that are persistently flagged over time
        (i.e., flagged for more than the specified fraction of samples) because such persistent
        flagging is interpreted as an instrumental offset rather than genuine RFI.
    """

    sigma = config.Property(proptype=float, default=5)
    subfreq_threshold = config.Property(proptype=int, default=2)
    keep_frac_rfi = config.Property(proptype=bool, default=False)
    spread_factor = config.Property(proptype=float, default=1)
    npix = config.Property(proptype=int, default=512)
    remove_persistent_beamns_frac = config.Property(proptype=float, default=None)

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
            mask = rfimaskbitmap.get_mask(
                self.sigma,
                self.subfreq_threshold,
                remove_persistent_beamns_frac=self.remove_persistent_beamns_frac,
            )[:].local_array[i - sf]

            # Optionally carry over the number of HFB subrequency channels detecting RFI
            if self.keep_frac_rfi:
                frac_rfi = rfimaskbitmap.get_frac_rfi(self.sigma)[:].local_array[i - sf]

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


class RFIMaskReduceBeamNS(tasklib.base.ContainerTask):
    """Create RFIMask from HFBDirectionalRFIMaskBitmap.

    This task takes an HFBDirectionalRFIMaskBitmap(freq, beam_ns, time), selects the RFI
    mask corresponding to a specified significance value used in the detection and
    subfrequency threshold, then reduces the 'beam_ns' axis to create a RFIMask container
    (freq, time).

    Attributes
    ----------
    sigma : float
        Specify the value of significance used in RFI detection to create a RFI mask.
        Must match one of the sigma keys used when the HFBDirectionalRFIMaskBitmap was
        created. Default is 5.
    subfreq_threshold : int
        Subfrequency threshold for decoding the bitmap. A sample is flagged if
        the number of HFB subfrequency channels detecting RFI >= this threshold.
        Default is 2.
    beam_ns_threshold : int
        This number determines the minimum number of detected RFI events along the beam_ns
        axis required for a data point to be included in the reduced mask. Default is 1.
    remove_persistent_beamns_frac: float
        If non-zero, remove beam_ns rows that are persistently flagged over time
        (i.e., flagged for more than the specified fraction of samples) because such persistent
        flagging is interpreted as an instrumental offset rather than genuine RFI.
    """

    sigma = config.Property(proptype=float, default=5)
    subfreq_threshold = config.Property(proptype=int, default=2)
    beam_ns_threshold = config.Property(proptype=int, default=1)
    remove_persistent_beamns_frac = config.Property(proptype=float, default=None)

    def process(self, rfimaskbitmap):
        """Produce a RFI mask.

        Parameters
        ----------
        rfimaskbitmap : hfb.containers.HFBDirectionalRFIMaskBitmap
            beam_ns-specific RFI mask with axes (freq, beam_ns, time).

        Returns
        -------
        out : draco.containers.RFIMask(freq, time)
            Non beam_ns-specific RFI mask with axes (freq, time).

        """
        # Extract mask/frac and axes data
        mask = rfimaskbitmap.get_mask(
            self.sigma,
            self.subfreq_threshold,
            remove_persistent_beamns_frac=self.remove_persistent_beamns_frac,
        )
        beam_ns_axis = list(rfimaskbitmap.subfreq_rfi.attrs["axis"]).index("beam_ns")

        # Apply reduction condition
        reduced_mask = np.sum(mask, axis=beam_ns_axis) >= self.beam_ns_threshold

        # Create an output container
        output = RFIMask(axes_from=rfimaskbitmap, attrs_from=rfimaskbitmap)

        # The output RFI mask is not frequency distributed
        output.mask[:] = mpiarray.MPIArray.wrap(reduced_mask, axis=0).allgather()

        # Return output container
        return output


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


class HFBCosmologyRFIFlagging(tasklib.base.ContainerTask):
    """Make a stack of the daily intensity-mapping RFI masks.

    The output is on the CHIME coarse freq channels and the full RA range.
    For each CSD the selected masks are combined (OR), then rebinned from
    time to RA once, as in SiderealRebinner. An RA pixel is flagged if any
    time sample in it is flagged. Missing mask files are skipped.

    Attributes
    ----------
    csd_range : list
        Days to include, start <= csd < end. Either one range, [start, end],
        or several, [[start1, end1], [start2, end2], ...].
    nra : int
        Number of RA pixels over the full range, as in the ringmaps. Default is 4280.
    root : str
        Directory holding <csd>/rfi_mask_<name>_lsd_<csd>.h5.
    masks_to_include : list of str
        Masks to combine, from `all_masks`. Default: all.
    """

    all_masks = ("freq", "sensitivity", "transient", "static", "chisq")

    csd_range = config.Property(proptype=list)
    nra = config.Property(proptype=int, default=4280)
    root = config.Property(
        proptype=str, default="~/projects/rpp-chime/chime/chime_processed/daily/rev_09"
    )
    masks_to_include = config.Property(proptype=list, default=list(all_masks))

    def setup(self, manager):
        """Set up the observer.

        Parameters
        ----------
        manager : drift.core.telescope.TransitTelescope
            A Telescope object holding the geographic location of the telescope.
        """
        self.observer = get_telescope(manager)
        self._done = False

    def process(self):
        """Make the mask stack.

        Returns
        -------
        out : containers.CosmologyRFIMaskStack
            Combined mask of each day, on coarse freq and the full RA range.
        """
        if self._done:
            raise exceptions.PipelineStopIteration
        self._done = True

        unknown = set(self.masks_to_include) - set(self.all_masks)
        if unknown:
            raise config.CaputConfigError(
                f"Unknown masks {unknown}, use {self.all_masks}."
            )

        if self.csd_range is None:
            raise config.CaputConfigError("csd_range must be given.")

        # Days in any of the ranges
        ranges = np.atleast_2d(np.asarray(self.csd_range, dtype=float))
        csds = np.unique(
            np.concatenate([np.arange(np.ceil(s), np.ceil(e)) for s, e in ranges])
        ).astype(int)

        # CHIME coarse channel centres
        cfreq = np.linspace(
            self.observer.freq_start,
            self.observer.freq_end,
            self.observer.num_freq,
            endpoint=False,
        )

        out = containers.CosmologyRFIMaskStack(csd=csds, freq=cfreq, ra=self.nra)
        out.redistribute("csd")

        # Days held by this rank
        mask = out.mask[:]
        local = mask.local_array
        c0 = mask.local_offset[0]

        for ic, csd in enumerate(csds[c0 : c0 + local.shape[0]]):
            local[ic] = self._day_mask(csd, out.ra)

        return out

    def _day_mask(self, csd, ra):
        """Combined mask of one day on the (coarse freq, full RA) grid."""
        # Combine the masks that exist.
        mask, time = False, None
        for name in self.masks_to_include:
            path = os.path.join(
                os.path.expanduser(self.root), str(csd), f"rfi_mask_{name}_lsd_{csd}.h5"
            )
            if not os.path.exists(path):
                self.log.warning(f"CSD {csd}: {path} not found, skipping.")
                continue
            cont = RFIMask.from_file(path)
            mask = mask | cont.mask[:]
            time = cont.time

        if time is None:
            return False

        # Rebin from time to RA, as in SiderealRebinner. R[ra, time] is the
        # fraction of each time sample in each RA pixel.
        lsd = self.observer.unix_to_lsd(time)
        R = regrid.rebin_matrix(lsd, csd + ra / 360.0, width_t=np.median(np.diff(lsd)))

        # Return mask @ R.T > 0 which is where a flagged time sample falls in the pixel.
        return (mask @ ss.csr_array(R.T)) > 0


class HFBRollingMedianRFIFlagging(tasklib.base.ContainerTask):
    """Flag RFI from the deviation from a rolling median along CSD.

    For every day i and every pixel (beam_ew, el, ra, freq), take the days j
    with |csd_j - csd_i| <= window / 2, and only those with weight > 0. The
    deviations are in units of each day's noise, sigma_j = 1 / sqrt(w_j):

        med = median(data_j)
        r_j = (data_j - med) * sqrt(w_j)
        mad = median(|r_j|)
        mask_i = |r_i| > threshold * mad

    A pixel with fewer than `min_days` valid days in its window is not
    flagged (no estimate). Days with weight = 0 are not flagged either: the
    mask holds only the new RFI flags.

    Attributes
    ----------
    window : float
        Full width of the window, in CSD. Default 30.
    threshold : float
        Flag deviations larger than this many MADs. Default 6.
    min_days : int
        Minimum number of valid days in the window. Default 7.
    """

    window = config.Property(proptype=float, default=30.0)
    threshold = config.Property(proptype=float, default=6.0)
    min_days = config.Property(proptype=int, default=7)

    def process(self, stream):
        """Make the RFI mask.

        Parameters
        ----------
        stream : containers.HFBHighResRingMapStack
            Stack with axes (csd, beam_ew, el, ra, freq).

        Returns
        -------
        out : containers.HFBRingMapRFIMaskStack
            Mask with the same axes. True means RFI.
        """
        import warnings

        stream.redistribute("freq")
        out = containers.HFBRingMapRFIMaskStack(axes_from=stream, attrs_from=stream)
        out.redistribute("freq")

        # Names of the axes of the data
        axes = list(stream.hfb.attrs["axis"])
        # Position of the csd axis
        icsd = axes.index("csd")
        # Local parts of data, weight and mask, with csd moved to the front (views)
        d, w, m = (
            np.moveaxis(getattr(x, "local_array", x), icsd, 0)
            for x in (stream.hfb[:], stream.weight[:], out.mask[:])
        )
        # s = sqrt(w) = 1 / sigma of each sample; NaN where there are no data
        s = np.sqrt(np.where(w > 0, w, np.nan)).astype(np.float32)
        # Data with NaN where there are no data, so the medians ignore them
        d = np.where(w > 0, d, np.nan).astype(np.float32)
        # CSD of each day of the stack
        csd = np.asarray(stream.index_map["csd"], dtype=float)

        for i in range(len(csd)):
            # Days within half a window of day i (gaps in the data are not bridged)
            sel = np.abs(csd - csd[i]) <= self.window / 2
            dw = d[sel]

            # Pixels without any data in the window give NaN (and a warning)
            with warnings.catch_warnings():
                # Do not print the "All-NaN slice" warning
                warnings.simplefilter("ignore", RuntimeWarning)

                # Median over the days of the window, for every pixel
                med = np.nanmedian(dw, axis=0)
                # Robust scatter of the deviations in units of each day's noise
                mad = np.nanmedian(np.abs(dw - med) * s[sel], axis=0)

            # Number of days with data in the window, for every pixel
            nvalid = np.sum(np.isfinite(dw), axis=0)
            # Deviation of day i from the median, in units of its noise
            dev = np.abs(d[i] - med) * s[i]
            # Flag where there are enough days and the deviation is too large
            m[i] = (nvalid >= self.min_days) & (mad > 0) & (dev > self.threshold * mad)

        # Report the fraction of flagged samples on this rank
        self.log.info(
            f"Rolling median RFI mask: {m.mean() * 100:.2f}% of samples flagged "
            f"on this rank."
        )

        # Return the mask
        return out


class HFBApplyRFIMaskStack(tasklib.base.ContainerTask):
    """Apply an RFI mask stack to an HFB stack, in place.

    The mask can be

    - a CosmologyRFIMaskStack (csd, freq, ra), on coarse freq: each pixel of
      the stack takes the mask of its coarse channel and RA pixel, the same for
      every el and beam_ew.
    - an HFBRingMapRFIMaskStack (csd, beam_ew, el, ra, freq), on the same
      grid as the stack (e.g. from HFBRollingMedianRFIMask).

    The axes are matched by name. The RA pixels, and any el and beam_ew, must
    lie on the grid of the mask. The mask need not cover the whole stack. An error
    is raised if no pixel of the stack is in the mask along an axis.
    """

    def process(self, stream, mask):
        """Apply the mask.

        Parameters
        ----------
        stream : containers.HFBHighResRingMapStack or HFBHighResBeamAvgRingMapStack
            Stack to mask.
        mask : CosmologyRFIMaskStack or HFBRingMapRFIMaskStack
            Mask with a csd axis. Its other axes must be axes of the stack.

        Returns
        -------
        stream : containers.HFBHighResRingMapStack or HFBHighResBeamAvgRingMapStack
            Same as input, with flagged samples at zero weight.
        """
        # Distribute the stack over days, so each rank masks its own days
        stream.redistribute("csd")
        # Distribute the mask over days too
        mask.redistribute("csd")

        # Names of the axes of the stack
        axes = list(stream.weight.attrs["axis"])
        # Names of the axes of the mask
        maxes = list(mask.mask.attrs["axis"])
        if not set(maxes) <= set(axes):
            raise ValueError(f"Mask axes {maxes} are not all axes of the stack {axes}.")

        # The mask dataset
        m = mask.mask[:]
        # The days of the mask held by this rank
        local = m.local_array if isinstance(m, mpiarray.MPIArray) else m
        # Cut the mask to the pixels of the stack, one axis at a time
        for ax in maxes:
            # Days are matched later, by CSD
            if ax == "csd":
                continue
            # Coordinates of the stack along this axis (freq centres for freq)
            sv = np.asarray(
                stream.freq if ax == "freq" else stream.index_map[ax], dtype=float
            )
            # Coordinates of the mask along this axis
            mv = np.asarray(
                mask.freq if ax == "freq" else mask.index_map[ax], dtype=float
            )
            # Distance from every mask pixel (rows) to every stack pixel (columns)
            dist = mv[:, np.newaxis] - sv
            # For RA, take the shorter way round the circle (e.g. 359.9 to 0.1)
            if ax == "ra":
                dist = (dist + 180.0) % 360.0 - 180.0
            # Nearest mask pixel for each stack pixel
            ind = np.abs(dist).argmin(axis=0)

            # Pixel size of the mask along this axis
            step = np.median(np.abs(np.diff(mv))) if mv.size > 1 else np.inf
            # Within half a channel for freq (so coarse channels work), on the grid
            # of the mask for the other axes
            tol = 0.5 * step if ax == "freq" else 1e-3 * step
            # Stack pixels that are covered by the mask
            inside = np.abs(dist[ind, np.arange(sv.size)]) <= tol
            # None covered: the grids do not match
            if not inside.any():
                raise ValueError(f"No {ax} pixel of the stack is on the mask grid.")
            # Some not covered: report how many (they will not be flagged)
            if not inside.all():
                self.log.info(
                    f"{(~inside).sum()} of {sv.size} {ax} pixels are not in the mask."
                )

            # Position of this axis in the mask
            k = maxes.index(ax)
            # Shape to broadcast `inside` along this axis of the mask
            shape = [1] * local.ndim
            # Length of `inside` goes on this axis
            shape[k] = -1
            # Mask on the pixels of the stack; pixels not covered are not flagged
            local = np.take(local, ind, axis=k) & inside.reshape(shape)

        # Axes of the mask, in the order they appear in the stack
        order = [a for a in axes if a in maxes]
        # Put the axes of the mask in that order
        cut = np.transpose(local, [maxes.index(a) for a in order])
        # Add length-1 axes for the axes the mask does not have, so it broadcasts
        # against the weight of the stack
        cut = cut.reshape(
            [cut.shape[order.index(a)] if a in maxes else 1 for a in axes]
        )

        # Position of the csd axis in the stack
        icsd = axes.index("csd")
        # CSDs of the mask
        mcsd = np.asarray(mask.index_map["csd"], dtype=float)
        # CSDs of the stack
        scsd = np.asarray(stream.index_map["csd"], dtype=float)
        # CSD of each day of the cut
        if not isinstance(m, mpiarray.MPIArray):
            # Not distributed: the cut holds all days of the mask
            cut_csd = mcsd
        elif np.array_equal(mcsd, scsd):
            # Same days: each rank holds the same days of the stack and the mask,
            # so the local cut is enough (no need to gather a large mask)
            c0 = m.local_offset[maxes.index("csd")]
            # CSDs of the days of the mask held by this rank
            cut_csd = mcsd[c0 : c0 + cut.shape[icsd]]
        else:
            # Different days: all ranks share their (small) cuts
            cut = np.concatenate(mask.comm.allgather(cut), axis=icsd)
            # The gathered cut holds all days of the mask
            cut_csd = mcsd

        # Weight of the stack. Its local part is a view, so changing it in place
        # changes the stack
        weight = stream.weight[:]
        if isinstance(weight, mpiarray.MPIArray):
            # First day held by this rank, and the local part of the weight
            c0, weight = weight.local_offset[icsd], weight.local_array
        else:
            # Not distributed: all days
            c0 = 0
        # CSDs of the days of the stack held by this rank
        csd = scsd[c0 : c0 + weight.shape[icsd]]

        # Position of each CSD in the cut
        pos = {c: i for i, c in enumerate(cut_csd)}
        # Days of the stack that are in the mask (others are left as they are)
        has = np.array([c in pos for c in csd], dtype=bool)
        # Report the days that are not in the mask
        if not has.all():
            self.log.info(f"CSDs {csd[~has].astype(int).tolist()} are not in the mask.")

        # Select all pixels of the days that are in the mask
        sel = [slice(None)] * len(axes)
        # Only those days along csd
        sel[icsd] = np.flatnonzero(has)
        # Zero the weight where the mask of the same day is True
        weight[tuple(sel)] *= ~np.take(cut, [pos[c] for c in csd[has]], axis=icsd)

        # Return the stack, masked in place
        return stream
