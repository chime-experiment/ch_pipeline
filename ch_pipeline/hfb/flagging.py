"""HFB Tasks for flagging data."""

import beam_model.formed as fm
import numpy as np
from caput import config, tools
from draco.core import task
from draco.core.containers import LocalizedRFIMask

from .containers import HFBRFIMask, HFBSensitivityMask


class HFBRadiometerRFIFlagging(task.SingleTask):
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
        # Extract data and weight arrays, averaging over beams
        data = np.mean(stream.hfb[:], axis=2)
        weight = np.mean(stream.weight[:], axis=2)

        # Number of samples per data point in the HFB data:
        # n_samp = delta_nu * delta_t * (1 - frac_lost), where
        # delta_nu is the frequency resolution (390.625 kHz / 128),
        # delta_t the integration time (~10.066 s), and
        # frac_lost is the fraction of integration that was lost upstream
        freq_width = stream._data["index_map/freq/width"][0] * 1e6
        nsubfreq = len(stream._data["index_map/subfreq"])
        delta_nu = freq_width / nsubfreq
        delta_t = np.median(np.diff(stream._data["index_map/time/ctime"]))
        frac_lost = stream["flags/frac_lost"][0]
        n_samp = delta_nu * delta_t * (1.0 - frac_lost)

        # Ideal radiometer equation
        radiometer = data**2 * tools.invert_no_zero(n_samp)

        # Radiometer noise test: the sensitivity metric would be unity for
        # an ideal radiometer, it would be higher for data with RFI
        sensitivity_metric = 2.0 * tools.invert_no_zero(radiometer * weight)

        # Boolean mask indicating data that are contaminated by RFI
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


class HFBSensitivityFlagging(task.SingleTask):
    """Produce a RFI mask based on HFB sensitivity values.

    The mask is for each N-S beam positions averaged over every 4 E-W beam positions
    and for each chime frequency channel averaged over every 128 HFB subfrequencies.

    Attributes
    ----------
    std : float
        The estimated standard deviation of the Gaussian modeling the distribution
        of HFB sensitivity values. Default is 0.15.
    threshold : float
        This is to detect RFI in HFB sensitivity values. If it exceeds 1 + treshold * std,
        then it indicates RFI events. Default is 5.
    threshold_subfreq : int
        Flag out a CHIME frequency channel if the number of its subfrequency channels detecting
        RFI is above this value. Default is 1.
    keep_frac_rfi : bool
         Save the fraction of subfrequency channels detecting RFI events.
    """

    std = config.Property(proptype=float, default=0.15)
    threshold = config.Property(proptype=float, default=5)
    threshold_subfreq = config.Property(proptype=int, default=1)
    keep_frac_rfi = config.Property(proptype=bool, default=False)

    def process(self, stream):
        """Produce a RFI mask.

        Parameters
        ----------
        stream : containers.HFBData
            Container with HFB data and weights.

        Returns
        -------
        out : containers.HFBSensitivityMask
            Boolean mask that can be converted to a draco container `LocalizedRFIMask`
            with the task `HFBMaskConversion` to mask contaminated
            frequencies, beam/el, and time samples.

        """
        # Extract data and weight arrays
        data = stream.hfb[:].view(np.ndarray)
        weight = stream.weight[:].view(np.ndarray)
        freq = stream.freq[:]
        time = stream.time[:]

        # Number of samples per data point in the HFB data:
        # n_samp = delta_nu * delta_t * (1 - frac_lost), where
        # delta_nu is the frequency resolution (390.625 kHz / 128),
        # delta_t the integration time (~10.066 s), and
        # frac_lost is the fraction of integration that was lost upstream
        nfreq, nsubfreq, nbeam, ntime = data.shape
        freq_width = stream.index_map["freq"]["width"][0] * 1e6
        delta_nu = freq_width / nsubfreq
        delta_t = np.median(np.diff(time))
        nbeam_ew = 4
        nbeam_ns = nbeam // nbeam_ew
        beam_ns = np.arange(nbeam_ns)

        # frac_lost is not distributed, so we need to cut out the frequencies
        # that are local for the stream
        ax = list(stream.hfb.attrs["axis"]).index("freq")
        nfreq_local = stream.hfb.local_shape[ax]
        sf = stream.hfb.local_offset[ax]
        ef = sf + nfreq_local
        frac_lost = stream["flags/frac_lost"][sf:ef]
        n_samp = delta_nu * delta_t * (1.0 - frac_lost)

        # Ideal radiometer equation
        radiometer = data**2 * tools.invert_no_zero(n_samp)[:, None, None, :]

        # Radiometer noise test: the sensitivity metric would be unity for
        # an ideal radiometer, it would be higher for data with RFI
        sensitivity_metric = 2.0 * tools.invert_no_zero(radiometer * weight)

        # Averaging over E-W beams
        sensitivity_metric = sensitivity_metric.reshape(
            nfreq, nsubfreq, nbeam_ew, nbeam_ns, ntime
        )
        sensitivity_metric = np.mean(sensitivity_metric, axis=2)

        # Detecting RFI
        count = sensitivity_metric > 1.0 + self.threshold * self.std
        count = np.sum(count, axis=1)
        mask = count > self.threshold_subfreq

        # Create container to hold output
        out = HFBSensitivityMask(beam_ns=beam_ns, freq=freq, time=time)

        # Save mask to output container
        out.mask[:] = mask

        if self.keep_frac_rfi:
            out.add_dataset("frac_rfi")
            out.frac_rfi[:] = count / nsubfreq

        # Return output container
        return out


class HFBMaskConversion(task.SingleTask):
    """Convert axes from beam_ns to el.

    Note that before CSD = 3685, the time axes were not synchronized between cosmology and HFB data.

    Attributes
    ----------
    spread_size : int
        The number of cells to flag before and after a detected true value. This ensures
        conservative flagging, preventing missed detections due to axis alignment issues.
        Default is 1.
    npix : int
        The number of pixels used to cover the full el range from -1 to 1.
         Defualt is 512.
    """

    spread_size = config.Property(proptype=int, default=1)
    npix = config.Property(proptype=int, default=512)

    def process(self, rfimask):
        """Produce a RFI mask.

        Parameters
        ----------
        rfimask : containers.HFBSensitivityMask
            RFI mask whose axes are freq, beam_ns, and time.

        Returns
        -------
        out : containers.LocalizedRFIMask
            Boolean mask that can be converted to a draco container `LocalizedSiderealRFIMask`
            with the task `SiderealMaskConversion` to mask contaminated
            frequencies, el, and time/ra samples.
        """
        # Extract mask/frac and axes data
        mask = rfimask.mask[:]
        freq = rfimask.freq[:]
        time = rfimask.time[:]
        beam_ns = rfimask.beam_ns[:]

        nfreq, nbeam_ns, ntime = mask.shape

        # Convert beam IDs to el values
        v = fm.FFTFormedBeamModel()
        angles = v.get_beam_positions(beam_ns, freq)
        el = np.sin(np.deg2rad(angles.T[1]))

        # Find closest el indices
        full_el = np.linspace(-1, 1, self.npix)
        el_closest_indices = np.abs(el[:, :, None] - full_el).argmin(axis=2)

        # Make the new el axis
        min_index = np.min(el_closest_indices)
        max_index = np.max(el_closest_indices)
        new_el = full_el[min_index : max_index + 1]
        nel = len(new_el)

        ax = list(rfimask.mask.attrs["axis"]).index("freq")
        nfreq_local = rfimask.mask.local_shape[ax]
        sf = rfimask.mask.local_offset[ax]
        ef = sf + nfreq_local

        # Generate meshgrid indices for the freq and el axes
        n0, _, n2 = np.meshgrid(
            np.arange(nfreq_local) + sf,
            np.arange(nbeam_ns),
            np.arange(ntime),
            indexing="ij",
        )
        el_closest_indices = el_closest_indices - np.min(el_closest_indices)

        # Generate meshgrid indices for the freq and el axes
        n0, _, n2 = np.meshgrid(
            np.arange(nfreq), np.arange(nbeam_ns), np.arange(ntime), indexing="ij"
        )
        el_closest_indices = el_closest_indices - np.min(el_closest_indices)

        # Broadcast indices along the last axis
        index_tuple = (n0, el_closest_indices[sf:ef, :, None], n2)

        # Broadcast the mask data
        mask_adj = np.full((nfreq, nel, ntime), False)
        mask_adj[index_tuple] = mask

        # Falg before and after a given true value for conservative flagging
        for repeat in range(self.spread_size):
            mask_adj[:, :-1, :] |= mask_adj[:, 1:, :]
            mask_adj[:, 1:, :] |= mask_adj[:, :-1, :]
            mask_adj[:, :-2, :] |= mask_adj[:, 2:, :]
            mask_adj[:, 2:, :] |= mask_adj[:, :-2, :]

        # Create container to hold output
        out = LocalizedRFIMask(freq=freq, el=new_el, time=time)

        # Save the adjusted mask to output container
        out.mask[:] = mask_adj

        # If the input container has the frac_rfi dataset
        if rfimask.datasets.get("frac_rfi", None) is not None:
            frac_rfi = rfimask.frac_rfi[:]
            frac_rfi_adj = np.full((nfreq, nel, ntime), 0.0)
            frac_rfi_adj[index_tuple] = frac_rfi

            for repeat in range(self.spread_size):
                frac_rfi_adj[:, :-1, :] = np.maximum(
                    frac_rfi_adj[:, :-1, :], frac_rfi_adj[:, 1:, :]
                )
                frac_rfi_adj[:, 1:, :] = np.maximum(
                    frac_rfi_adj[:, 1:, :], frac_rfi_adj[:, :-1, :]
                )
                frac_rfi_adj[:, :-2, :] = np.maximum(
                    frac_rfi_adj[:, :-2, :], frac_rfi_adj[:, 2:, :]
                )
                frac_rfi_adj[:, 2:, :] = np.maximum(
                    frac_rfi_adj[:, 2:, :], frac_rfi_adj[:, :-2, :]
                )

            out.add_dataset("frac_rfi")
            out.frac_rfi[:] = frac_rfi_adj

        # Return output container
        return out
