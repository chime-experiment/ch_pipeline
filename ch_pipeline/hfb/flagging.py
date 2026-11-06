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

from .containers import HFBDirectionalRFIMaskBitmap, HFBRFIMask, SiderealRFIMaskStack


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
        out = HFBDirectionalRFIMaskBitmap(
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


class HFBMakeRFIMaskStack(tasklib.base.ContainerTask):
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
        out : SiderealRFIMaskStack
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

        out = SiderealRFIMaskStack(csd=csds, freq=cfreq, ra=self.nra)
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


class HFBApplyRFIMaskStack(tasklib.base.ContainerTask):
    """Apply a SiderealRFIMaskStack to an HFB stack, in place.

    The RA pixels of the stack must lie on the RA grid of the mask (the mask
    is made on the same grid), otherwise an error is raised.
    """

    def process(self, stream, mask):
        """Apply the mask.

        Parameters
        ----------
        stream : containers.HFBHighResRingMapStack or HFBHighResBeamAvgRingMapStack
            Stack to mask, on part of the high-resolution freq and RA ranges.
        mask : SiderealRFIMaskStack
            Mask on the same csd axis, on coarse freq and the full RA range from
            intensity mapping data.

        Returns
        -------
        stream : containers.HFBHighResRingMapStack or HFBHighResBeamAvgRingMapStack
            Same as input, with flagged samples at zero weight.
        """
        stream.redistribute("csd")
        mask.redistribute("csd")

        # Coarse channel of each HFB frequency: the nearest coarse centre
        ifreq = np.abs(mask.freq[:, np.newaxis] - stream.freq).argmin(axis=0)

        # Index of each RA pixel of the stream in the full RA grid of the mask
        ra = np.asarray(stream.index_map["ra"], dtype=float)
        ra_full = np.asarray(mask.index_map["ra"], dtype=float)
        nra = ra_full.size
        i0, i1 = _search_nearest(ra_full, np.array([ra[0], ra[-1]]) % 360.0)
        ira = np.arange(i0, i1 + 1 + (nra if i1 < i0 else 0)) % nra

        # The RA pixels must be on the grid of the mask
        dra = np.median(np.diff(ra_full))
        if ira.size != ra.size or np.any(
            np.abs((ra - ra_full[ira] + 180.0) % 360.0 - 180.0) > 1e-3 * dra
        ):
            raise ValueError(
                "The RA pixels of the stack are not on the RA grid of the mask."
            )

        # Mask of all days on the RA pixels of the stream
        m = mask.mask[:]
        if isinstance(m, mpiarray.MPIArray):
            # Each rank cuts its own days, then all ranks share them; the cut is small
            cut = np.concatenate(mask.comm.allgather(m.local_array[:, :, ira]), axis=0)
        else:
            cut = m[:, :, ira]  # not distributed

        # Weight with (csd, freq, ra) as the last three axes. This is a view of the
        # weight of the stream, so changing it in place changes the stream.
        axes = list(stream.weight.attrs["axis"])
        weight = stream.weight[:]
        if isinstance(weight, mpiarray.MPIArray):
            c0, weight = weight.local_offset[axes.index("csd")], weight.local_array
        else:
            c0 = 0  # not distributed
        w = np.moveaxis(
            weight, [axes.index(a) for a in ("csd", "freq", "ra")], [-3, -2, -1]
        )

        # Days of the stream held by this rank, and where they are in the mask.
        # Days that are not in the mask are left as they are.
        csd = np.asarray(stream.index_map["csd"], dtype=float)[c0 : c0 + w.shape[-3]]
        pos = {
            c: i for i, c in enumerate(np.asarray(mask.index_map["csd"], dtype=float))
        }
        has = np.array([c in pos for c in csd], dtype=bool)
        if not has.all():
            self.log.info(
                f"CSDs {csd[~has].astype(int).tolist()} are not in the mask, not masked."
            )

        # (csd, freq, ra) mask of these days on the freq and RA of the stream
        w[..., has, :, :] *= ~cut[[pos[c] for c in csd[has]]][:, ifreq]

        return stream
