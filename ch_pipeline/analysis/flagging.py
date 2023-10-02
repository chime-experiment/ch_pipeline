"""Tasks for Flagging Data

Tasks for calculating flagging out unwanted data. This includes RFI removal, and
data quality flagging on timestream data; sun excision on sidereal data; and
pre-map making flagging on m-modes.
"""
from typing import Union
import numpy as np

from caput import mpiutil, mpiarray, memh5, config, pipeline, tod
from ch_util import rfi, data_quality, tools, ephemeris, cal_utils, andata
from chimedb import dataflag as df
from chimedb.core import connect as connect_database

from draco.analysis import flagging as dflagging
from draco.core import task, io
from draco.core import containers as dcontainers

from ..core import containers


class RFIFilter(task.SingleTask):
    """Identify data contaminated by RFI.

    Attributes
    ----------
    stack: bool
        Average over all autocorrelations before constructing the mask.
    normalize : bool
        Normalize by the median value over time prior to stacking.
    flag1d : bool
        Only apply the MAD cut in the time direction.
        Useful if the frequency coverage is sparse.
    rolling : bool
        Use a rolling window instead of distinct blocks.
        This is slower, but recommended if stack is True
        or the number of feeds is small.
    apply_static_mask : bool
        Mask out frequencies known to be contaminated by persistent
        sources of RFI.  Mask is obtained from `ch_util.rfi.frequency_mask`.
        This is done before computing the median absolute deviation.
    keep_auto : bool
        Save the autocorrelations that were used to construct
        the mask in the output container.
    keep_ndev : bool
        Save the number of deviations that were used to construct
        the mask in the output container.
    freq_width : float
        Frequency interval in *MHz* to compare across.
    time_width : float
        Time interval in *seconds* to compare across.
    threshold_mad : float
        Threshold above which we mask the data.
    use_draco_container : bool
        If True, output container is a nondistributed draco RFIMask.
        Otherwise, return a distributed RFIMask from ch_pipeline
    """

    stack = config.Property(proptype=bool, default=False)
    normalize = config.Property(proptype=bool, default=False)
    flag1d = config.Property(proptype=bool, default=False)
    rolling = config.Property(proptype=bool, default=False)
    apply_static_mask = config.Property(proptype=bool, default=False)
    keep_auto = config.Property(proptype=bool, default=False)
    keep_ndev = config.Property(proptype=bool, default=False)
    freq_width = config.Property(proptype=float, default=10.0)
    time_width = config.Property(proptype=float, default=420.0)
    threshold_mad = config.Property(proptype=float, default=6.0)
    use_draco_container = config.Property(proptype=bool, default=False)

    def process(self, data) -> Union[containers.RFIMask, dcontainers.RFIMask]:
        """Creates a mask by identifying outliers in the
        autocorrelation data.  This mask can be used to zero out
        frequencies and time samples that are contaminated by RFI.

        Parameters
        ----------
        data : ch_util.andata.CorrData
            Generate the mask from the autocorrelation data
            in this container.

        Returns
        -------
        out : containers.RFIMask or dcontainers.RFIMask
            Boolean mask that can be applied to a timestream container
            with the task `ApplyCorrInputMask` to mask contaminated
            frequencies and time samples.
        """
        # Redistribute across frequency
        data.redistribute("freq")

        # Construct RFI mask
        auto_index, auto, ndev = rfi.number_deviations(
            data,
            apply_static_mask=self.apply_static_mask,
            freq_width=self.freq_width,
            time_width=self.time_width,
            flag1d=self.flag1d,
            rolling=self.rolling,
            stack=self.stack,
            normalize=self.normalize,
        )

        # Reorder output based on input chan_id
        minput = data.index_map["input"][auto_index]
        isort = np.argsort(minput["chan_id"])

        minput = minput[isort]
        auto = auto[:, isort, :]
        ndev = ndev[:, isort, :]

        # Place cut on the number of deviations.  Note that we are
        # only flagging positive excursions corresponding to an
        # increase in measured power relative to the local median.
        mask = ndev > self.threshold_mad

        # Create container to hold output
        if self.use_draco_container:
            # draco RFIMask container is not distributed
            out = dcontainers.RFIMask(axes_from=data, attrs_from=data)
            mask = mpiarray.MPIArray.wrap(mask, axis=0).allgather()[:, 0, :]
        else:
            out = containers.RFIMask(input=minput, axes_from=data, attrs_from=data)
            out.redistribute("freq")
            # Change flag convention
            mask = np.logical_not(mask)

            if self.keep_ndev:
                out.add_dataset("ndev")
                out.ndev[:] = ndev
            if self.keep_auto:
                out.add_dataset("auto")
                out.auto[:] = auto

        # Save mask to output container
        out.mask[:] = mask

        # Return output container
        return out


class RFISensitivityMask(dflagging.RFISensitivityMask):
    """CHIME version of RFISensitivityMask.

    This has a static mask for the local environment and will use the MAD
    algorithm (over SumThreshold) when bright sources are visible.
    """

    def _combine_st_mad_hook(self, times):
        """Use the MAD mask (over SumThreshold) whenever a bright source is overhead.

        Parameters
        ----------
        times : np.ndarray[ntime]
            Times of the data at floating point UNIX time.

        Returns
        -------
        combine : np.ndarray[ntime]
            Mixing array as a function of time. If `True` that sample will be
            filled from the MAD, if `False` use the SumThreshold algorithm.
        """
        # Switch for 4 deg each side of the source
        beam_window = np.radians(4.0)
        SIDEREAL_DAY = ephemeris.SIDEREAL_S * 24 * 3600

        # Select Sun transit times
        suntt = ephemeris.solar_transit(times[0])  # Sun transit time

        # Mask the sun for the maximum aparent dec of ~24 deg
        ra_window = beam_window / np.cos(np.deg2rad(24))
        time_window = ra_window / (2 * np.pi) * SIDEREAL_DAY

        # Time spans where we apply the MAD filter.
        madtimes = np.abs(times - suntt) < time_window

        # Select bright source transit times. Only CasA, CygA and TauA.
        sources = [ephemeris.CasA, ephemeris.CygA, ephemeris.TauA]

        ra_axis = np.radians(ephemeris.lsa(times))

        # Include bright point source transits in MAD times
        for src in sources:
            ra = src.ra.radians
            dec = src.dec.radians
            ra_window = beam_window / np.cos(dec)

            madtimes |= np.abs(ra_axis - ra) < ra_window

        return madtimes

    def _static_rfi_mask_hook(self, freq):
        """Use the static CHIME RFI mask.

        Parameters
        ----------
        freq : np.ndarray[nfreq]
            1D array of frequencies in the data (in MHz).

        Returns
        -------
        mask : np.ndarray[nfreq]
            Mask array. True will include a frequency channel, False masks it out.
        """
        return ~rfi.frequency_mask(freq)


class ChannelFlagger(task.SingleTask):
    """Mask out channels that appear weird in some way.

    Parameters
    ----------
    test_freq : list
        Frequencies to test the data at.
    """

    test_freq = config.Property(proptype=list, default=[610.0])

    ignore_fit = config.Property(proptype=bool, default=False)
    ignore_noise = config.Property(proptype=bool, default=False)
    ignore_gains = config.Property(proptype=bool, default=False)

    known_bad = config.Property(proptype=list, default=[])

    def process(self, timestream, inputmap):
        """Flag bad channels in timestream.

        Parameters
        ----------
        timestream : andata.CorrData
            Timestream to flag.

        Returns
        -------
        timestream : andata.CorrData
            Returns the same timestream object with a modified weight dataset.
        """

        # Redistribute over the frequency direction
        timestream.redistribute("freq")

        # Find the indices for frequencies in this timestream nearest
        # to the given physical frequencies
        freq_ind = [
            np.argmin(np.abs(timestream.freq - freq)) for freq in self.test_freq
        ]

        # Create a global channel weight (channels are bad by default)
        chan_mask = np.zeros(timestream.ninput, dtype=np.int)

        # Mark any powered CHIME channels as good
        chan_mask[:] = tools.is_chime_on(inputmap)

        # Calculate start and end frequencies
        sf = timestream.vis.local_offset[0]
        ef = sf + timestream.vis.local_shape[0]

        # Iterate over frequencies and find bad channels
        for fi in freq_ind:
            # Only run good_channels if frequency is local
            if fi >= sf and fi < ef:
                # Run good channels code and unpack arguments
                res = data_quality.good_channels(
                    timestream, test_freq=fi, inputs=inputmap, verbose=False
                )
                good_gains, good_noise, good_fit, test_channels = res

                self.log.info(
                    "Frequency %i bad channels: blank %i; gains %i; noise %i; fit %i %s",
                    fi,
                    np.sum(chan_mask == 0),
                    np.sum(good_gains == 0),
                    np.sum(good_noise == 0),
                    np.sum(good_fit == 0),
                    "[ignored]" if self.ignore_fit else "",
                )

                if good_noise is None:
                    good_noise = np.ones_like(test_channels)

                # Construct the overall channel mask for this
                # frequency (explicit cast to int or numpy complains,
                # this should really be done upstream).
                if not self.ignore_gains:
                    chan_mask[test_channels] *= good_gains.astype(np.int)
                if not self.ignore_noise:
                    chan_mask[test_channels] *= good_noise.astype(np.int)
                if not self.ignore_fit:
                    chan_mask[test_channels] *= good_fit.astype(np.int)

        # Gather the channel flags from all nodes, and combine into a
        # single flag (checking that all tests pass)
        chan_mask_all = np.zeros(
            (timestream.comm.size, timestream.ninput), dtype=np.int
        )
        timestream.comm.Allgather(chan_mask, chan_mask_all)
        chan_mask = np.prod(chan_mask_all, axis=0)

        # Mark already known bad channels
        for ch in self.known_bad:
            chan_mask[ch] = 0.0

        # Apply weights to files weight array
        chan_mask = chan_mask[np.newaxis, :, np.newaxis]
        weight = timestream.weight[:]
        tools.apply_gain(weight, chan_mask, out=weight)

        return timestream


class MonitorCorrInput(task.SingleTask):
    """Monitor good correlator inputs over several sidereal days.

    Parameters
    ----------
    n_day_min : int
        Do not apply a sidereal day flag if the number of days
        in the pass is less than n_day_min.  Default is 3.

    n_cut : int
        Flag a sidereal day as bad if the number of correlator
        inputs that are bad ONLY on this day is greater than n_cut.
        Default is 5.
    """

    n_day_min = config.Property(proptype=int, default=3)
    n_cut = config.Property(proptype=int, default=5)

    def setup(self, files):
        """Divide list of files up into sidereal days.

        Parameters
        ----------
        files : list
            List of filenames to monitor good correlator inputs.
        """

        from .sidereal import get_times, _days_in_csd

        self.files = np.array(files)

        # Initialize variables
        timemap = None
        input_map = None
        freq = None

        # If rank0, then create a map from csd to time range
        # and determine correlator inputs and frequencies
        if mpiutil.rank0:
            # Determine the days in each file and the days in all files
            se_times = get_times(files)
            se_csd = ephemeris.csd(se_times)
            days = np.unique(np.floor(se_csd).astype(np.int))

            # Determine the relevant files for each day
            filemap = [(day, _days_in_csd(day, se_csd, extra=0.005)) for day in days]

            # Determine the time range for each day
            timemap = [
                (day, ephemeris.csd_to_unix(np.array([day, day + 1]))) for day in days
            ]

            # Extract the frequency and inputs for the first day
            data_r = andata.Reader(self.files[filemap[0][1]])
            input_map = data_r.input
            freq = data_r.freq[:]

            ninput = len(input_map)
            nfreq = len(freq)

            # Loop through the rest of the days and make sure the
            # inputs and frequencies are the same
            for fmap in filemap[1:]:
                data_r = andata.Reader(self.files[fmap[1]])

                if len(data_r.input) != ninput:
                    ValueError(
                        "Differing number of corr inputs for csd %d and csd %d."
                        % (fmap[0], filemap[0][0])
                    )
                elif (
                    np.sum(
                        data_r.input["correlator_input"]
                        != input_map["correlator_input"]
                    )
                    > 0
                ):
                    ValueError(
                        "Different corr inputs for csd %d and csd %d."
                        % (fmap[0], filemap[0][0])
                    )

                if len(data_r.freq) != nfreq:
                    ValueError(
                        "Differing number of frequencies for csd %d and csd %d."
                        % (fmap[0], filemap[0][0])
                    )
                elif np.sum(data_r.freq["centre"] != freq["centre"]) > 0:
                    ValueError(
                        "Different frequencies for csd %d and csd %d."
                        % (fmap[0], filemap[0][0])
                    )

        # Broadcast results to all processes
        self.timemap = mpiutil.world.bcast(timemap, root=0)
        self.ndays = len(self.timemap)

        self.input_map = mpiutil.world.bcast(input_map, root=0)
        self.ninput = len(self.input_map)

        self.freq = mpiutil.world.bcast(freq, root=0)
        self.nfreq = len(self.freq)

    def process(self):
        """Calls ch_util.ChanMonitor for each sidereal day.

        Returns
        -------
        input_monitor : containers.CorrInputMonitor
            Saved for each sidereal day.  Contains the
            correlator input mask and frequency mask.
            Note that this is not output to the pipeline.  It is an
            ancillary data product that is saved when one sets the
            'save' parameter in the configuration file.
        csd_flag : containers.SiderealDayFlag
            Contains a mask that indicates bad sidereal days, determined as
            days that contribute a large number of unique bad correlator
            inputs.  Note that this is not output to the pipeline.
            It is ancillary data product that is saved when one sets the
            'save' parameter in the configuration file.
        input_monitor_all : containers.CorrInputMask
            Contains the correlator input mask obtained from taking AND
            of the masks from the (good) sidereal days.
        """

        from ch_util import chan_monitor

        # Check if we should stop
        if self.ndays == 0:
            raise pipeline.PipelineStopIteration

        # Get a range of days for this process to analyze
        n_local, i_day_start, i_day_end = mpiutil.split_local(self.ndays)
        i_day = np.arange(i_day_start, i_day_end)

        # Create local arrays to hold results
        input_mask = np.ones((n_local, self.ninput), dtype=np.bool)
        good_day_flag = np.zeros(n_local, dtype=np.bool)

        # Loop over days
        for i_local, i_dist in enumerate(i_day):
            csd, time_range = self.timemap[i_dist]

            # Print status
            self.log.info("Calling channel monitor for csd %d.", csd)

            # Create an instance of chan_monitor for this day
            cm = chan_monitor.ChanMonitor(*time_range)

            # Run the full test
            try:
                cm.full_check()
            except (RuntimeError, ValueError) as error:
                # No sources available for this csd
                self.log.info("    csd %d: %s", csd, error)
                continue

            # Accumulate flags over multiple days
            input_mask[i_local, :] = cm.good_ipts & cm.pwds
            good_day_flag[i_local] = True

            # If requested, write to disk
            if self.save:
                # Create a container to hold the results
                input_mon = containers.CorrInputMonitor(
                    freq=self.freq, input=self.input_map, distributed=False
                )

                # Place the results in the container
                input_mon.input_mask[:] = cm.good_ipts
                input_mon.input_powered[:] = cm.pwds
                input_mon.freq_mask[:] = cm.good_freqs
                input_mon.freq_powered[:] = cm.gpu_node_flag

                if hasattr(cm, "postns"):
                    input_mon.add_dataset("position")
                    input_mon.position[:] = cm.postns

                if hasattr(cm, "expostns"):
                    input_mon.add_dataset("expected_position")
                    input_mon.expected_position[:] = cm.expostns

                if cm.source1 is not None:
                    input_mon.attrs["source1"] = cm.source1.name

                if cm.source2 is not None:
                    input_mon.attrs["source2"] = cm.source2.name

                # Construct tag from csd
                tag = "csd_%d" % csd
                input_mon.attrs["tag"] = tag
                input_mon.attrs["csd"] = csd

                # Save results to disk
                self._save_output(input_mon)

        # Gather the flags from all nodes
        input_mask_all = np.zeros((self.ndays, self.ninput), dtype=np.bool)
        good_day_flag_all = np.zeros(self.ndays, dtype=np.bool)

        mpiutil.world.Allgather(input_mask, input_mask_all)
        mpiutil.world.Allgather(good_day_flag, good_day_flag_all)

        if not np.any(good_day_flag_all):
            ValueError("Channel monitor failed for all days.")

        # Find days where the number of correlator inputs that are bad
        # ONLY for this day is greater than some user specified threshold
        if np.sum(good_day_flag_all) >= max(2, self.n_day_min):
            n_uniq_bad = np.zeros(self.ndays, dtype=np.int)
            dindex = np.arange(self.ndays)[good_day_flag_all]

            for ii, day in enumerate(dindex):
                other_days = np.delete(dindex, ii)
                n_uniq_bad[day] = np.sum(
                    ~input_mask_all[day, :]
                    & np.all(input_mask_all[other_days, :], axis=0)
                )

            good_day_flag_all *= n_uniq_bad <= self.n_cut

            if not np.any(good_day_flag_all):
                ValueError(
                    "Significant number of new correlator inputs flagged bad each day."
                )

        # Write csd flag to file
        if self.save:
            # Create container
            csd_flag = containers.SiderealDayFlag(
                csd=np.array([tmap[0] for tmap in self.timemap])
            )

            # Save flags to container
            csd_flag.csd_flag[:] = good_day_flag_all

            csd_flag.attrs["tag"] = "flag_csd"

            # Write output to hdf5 file
            self._save_output(csd_flag)

        # Take the product of the input mask for all days that made threshold cut
        input_mask = np.all(input_mask_all[good_day_flag_all, :], axis=0)

        # Create a container to hold the results for the entire pass
        input_mon = containers.CorrInputMask(input=self.input_map)

        # Place the results for the entire pass in a container
        input_mon.input_mask[:] = input_mask

        input_mon.attrs["tag"] = "for_pass"

        # Ensure we stop on next iteration
        self.ndays = 0
        self.timemap = None

        # Return pass results
        return input_mon


class TestCorrInput(task.SingleTask):
    """Apply a series of tests to find the good correlator inputs.

    Parameters
    ----------
    test_freq : list
        Physical frequencies (in MHz) to examine.
    ignore_gain: bool
        Do not apply the test that checks for excessively high digital gains.
    ignore_noise: bool
        Do not apply the test that checks for compliance of noise
        to radiometer equation.
    ignore_fit: bool
        Do not apply the test that checks the goodness of the fit
        to a template Tsky.
    threshold: float
        Fraction of frequencies that must pass all requested
        tests in order for input to be flagged as good.
    """

    test_freq = config.Property(proptype=list, default=None)

    ignore_gains = config.Property(proptype=bool, default=False)
    ignore_noise = config.Property(proptype=bool, default=False)
    ignore_fit = config.Property(proptype=bool, default=False)

    threshold = config.Property(proptype=float, default=0.7)

    known_bad = config.Property(proptype=list, default=[])

    def __init__(self):
        """Set up variables that gives names to the various test
        and specify which tests will be applied.
        """

        # Gives names to the tests that will be run
        self.test = np.array(
            ["is_chime", "not_known_bad", "digital_gain", "radiometer", "sky_fit"]
        )
        self.ntest = len(self.test)

        # Determine what tests we will use
        self.use_test = ~np.array(
            [False, False, self.ignore_gains, self.ignore_noise, self.ignore_fit]
        )

    def process(self, timestream, inputmap):
        """Find good inputs using timestream.

        Parameters
        ----------
        timestream : andata.CorrData
            Apply series of tests to this timestream.
        inputmap : list of :class:`CorrInput`
            A list of describing the inputs as they are in timestream.

        Returns
        -------
        corr_input_test : containers.CorrInputTest
            Container with the results of all tests and a
            input mask that combines all tests and frequencies.
        """

        # Redistribute over the frequency direction
        timestream.redistribute("freq")

        # Extract the frequency map
        freqmap = timestream.index_map["freq"][:]

        # Find the indices for frequencies in this timestream nearest
        # to the requested test frequencies.
        if self.test_freq is None:
            freq_ind = np.arange(len(freqmap), dtype=np.int)
        else:
            freq_ind = [
                np.argmin(np.abs(freqmap["centre"] - freq)) for freq in self.test_freq
            ]

        # Calculate start and end frequencies
        nfreq = timestream.vis.local_shape[0]
        sfreq = timestream.vis.local_offset[0]
        efreq = sfreq + nfreq

        # Create local flag arrays (inputs are good by default)
        passed_test = np.ones((nfreq, timestream.ninput, self.ntest), dtype=np.int)
        is_test_freq = np.zeros(nfreq, dtype=np.bool)

        # Mark any non-CHIME inputs as bad
        for i in range(timestream.ninput):
            if not isinstance(inputmap[i], tools.CHIMEAntenna):
                passed_test[:, i, 0] = 0

        # Mark already known bad inputs
        for ch in self.known_bad:
            passed_test[:, ch, 1] = 0

        # Iterate over frequencies and find bad inputs
        for fi_local, fi_dist in enumerate(range(sfreq, efreq)):
            if fi_dist in freq_ind:
                # Check if vis_weight is zero for this frequency,
                # which would imply a bad GPU node.
                if "vis_weight" in timestream.flags:
                    if not np.any(timestream.weight[fi_dist]):
                        continue

                # Run good channels code and unpack arguments
                res = data_quality.good_channels(
                    timestream, test_freq=fi_dist, inputs=inputmap, verbose=False
                )
                good_gains, good_noise, good_fit, test_inputs = res

                # Save the results to local array
                if good_gains is not None:
                    passed_test[fi_local, test_inputs, 2] = good_gains

                if good_noise is not None:
                    passed_test[fi_local, test_inputs, 3] = good_noise

                if good_fit is not None:
                    passed_test[fi_local, test_inputs, 4] = good_fit

                is_test_freq[fi_local] = True

                # Print results for this frequency
                self.log.info(
                    "Frequency {} bad inputs: blank {}; gains {}{}; noise {}{}; fit {}{}".format(
                        fi_dist,
                        timestream.ninput - len(test_inputs),
                        np.sum(good_gains == 0) if good_gains is not None else "failed",
                        " [ignored]" if self.ignore_gains else "",
                        np.sum(good_noise == 0) if good_noise is not None else "failed",
                        " [ignored]" if self.ignore_noise else "",
                        np.sum(good_fit == 0) if good_fit is not None else "failed",
                        " [ignored]" if self.ignore_fit else "",
                    )
                )

        # Gather the input flags from all nodes
        passed_test_all = np.zeros(
            (timestream.comm.size, timestream.ninput, self.ntest), dtype=np.int
        )
        is_test_freq_all = np.zeros(timestream.comm.size, dtype=np.bool)

        timestream.comm.Allgather(passed_test, passed_test_all)
        timestream.comm.Allgather(is_test_freq, is_test_freq_all)

        # Keep only the test frequencies
        passed_test_all = passed_test_all[is_test_freq_all, ...]
        freqmap = freqmap[is_test_freq_all]

        # Average over frequencies
        input_mask_all = (
            np.sum(passed_test_all, axis=0) / float(passed_test_all.shape[0])
        ) >= self.threshold

        # Take the product along the test direction to determine good inputs for each frequency
        input_mask = np.prod(input_mask_all[:, self.use_test], axis=-1)

        # Create container to hold results
        corr_input_test = containers.CorrInputTest(
            freq=freqmap, test=self.test, axes_from=timestream, attrs_from=timestream
        )

        # Save flags to container, return container
        corr_input_test.input_mask[:] = input_mask

        corr_input_test.add_dataset("passed_test")
        corr_input_test.passed_test[:] = passed_test_all

        return corr_input_test


class AccumulateCorrInputMask(task.SingleTask):
    """Find good correlator inputs over multiple sidereal days.

    Parameters
    ----------
    n_day_min : int
        Do not apply a sidereal day flag if the number of days
        in the pass is less than n_day_min.  Default is 3.

    n_cut : int
        Flag a sidereal day as bad if the number of correlator
        inputs that are uniquely flagged bad on that day is
        greater than n_cut.  Default is 5.
    """

    n_day_min = config.Property(proptype=int, default=3)
    n_cut = config.Property(proptype=int, default=5)

    def __init__(self):
        """Create empty list.  As we iterate through
        sidereal days, we will append the corr_input_mask
        from each day to this list.
        """

        self._accumulated_input_mask = []
        self._csd = []

    def process(self, corr_input_mask):
        """Append corr_input_mask to list.

        Parameters
        ----------
        corr_input_mask : containers.CorrInputMask
        """

        if not self._accumulated_input_mask:
            self.input = corr_input_mask.input[:]

        self._accumulated_input_mask.append(corr_input_mask.input_mask[:])
        self._csd.append(corr_input_mask.attrs["csd"])

    def process_finish(self):
        """Determine good days as those where the fraction
        of good inputs is above some user specified
        threshold.  Then create accumulated input mask
        by taking the product of the input mask for all
        good days.

        Returns
        --------
        corr_input_mask : containers.CorrInputMask
        """
        ncsd = len(self._csd)

        input_mask_all = np.asarray(self._accumulated_input_mask)

        good_day_flag = np.ones(ncsd, dtype=np.bool)
        # Find days where the number of correlator inputs that are bad
        # ONLY for this day is greater than some user specified threshold
        if ncsd >= max(2, self.n_day_min):
            n_uniq_bad = np.zeros(ncsd, dtype=np.int)
            dindex = np.arange(ncsd)[good_day_flag]

            for ii, day in enumerate(dindex):
                other_days = np.delete(dindex, ii)
                n_uniq_bad[day] = np.sum(
                    ~input_mask_all[day, :]
                    & np.all(input_mask_all[other_days, :], axis=0)
                )

            good_day_flag *= n_uniq_bad <= self.n_cut

            if not np.any(good_day_flag):
                ValueError(
                    "Significant number of new correlator inputs flagged bad each day."
                )

        # Write csd flag to file
        if self.save:
            # Create container
            csd_flag = containers.SiderealDayFlag(csd=np.array(self._csd))

            # Save flags to container
            csd_flag.csd_flag[:] = good_day_flag

            csd_flag.attrs["tag"] = "flag_csd"

            # Write output to hdf5 file
            self._save_output(csd_flag)

        # Take the product of the input mask for all days that made threshold cut
        input_mask = np.all(input_mask_all[good_day_flag, :], axis=0)

        # Create container to hold results
        corr_input_mask = containers.CorrInputMask(input=self.input)

        corr_input_mask.attrs["tag"] = "for_pass"

        # Save input_mask to container, return container
        corr_input_mask.input_mask[:] = input_mask

        return corr_input_mask


class ApplyCorrInputMask(task.SingleTask):
    """Apply an input mask to a timestream."""

    def process(self, timestream, cmask):
        """Flag out events by giving them zero weight.

        Parameters
        ----------
        timestream : andata.CorrData or dcontainers.SiderealStream

        cmask : containers.RFIMask, containers.CorrInputMask, etc.

        Returns
        -------
        timestream : andata.CorrData or dcontainers.SiderealStream
        """

        # Make sure containers are distributed across frequency
        timestream.redistribute("freq")
        cmask.redistribute("freq")

        # Create a slice that will expand the mask to
        # the same dimensions as the weight array
        waxis = timestream.weight.attrs["axis"]
        slc = [slice(None)] * len(waxis)
        for ww, name in enumerate(waxis):
            if (name not in ["stack", "prod"]) and (
                name not in cmask.mask.attrs["axis"]
            ):
                slc[ww] = None

        # Extract input mask and weight array
        weight = timestream.weight[:]
        mask = cmask.mask[:].astype(weight.dtype)

        # Expand mask to same dimension as weight array
        mask = mask[tuple(slc)]

        # Determine mapping between inputs in the mask
        # and inputs in the timestream
        mask_input = cmask.index_map["input"][:]
        nminput = mask_input.size

        tstream_input = timestream.index_map["input"][:]
        ntinput = tstream_input.size

        if nminput == 1:
            # Apply same mask to all products
            weight *= mask

        else:
            # Map each input to a mask
            if nminput < ntinput:
                # The expression below will expand a mask constructed from the stacked
                # autocorrelation data for each cylinder/polarisation to the inputs from
                # that cylinder/polarisation.  However, we may want to make this more robust
                # and explicit by passing a telescope object or list of correlator inputs
                # and determining the cylinder/polarisation mapping from that.
                iexpand = (
                    np.digitize(
                        np.arange(ntinput), np.append(mask_input["chan_id"], ntinput)
                    )
                    - 1
                )

                mask = mask[:, iexpand]

            # Use apply_gain function to apply mask based on product map
            prod = timestream.index_map["prod"][timestream.index_map["stack"]["prod"]]
            tools.apply_gain(weight, mask, out=weight, prod_map=prod)

        # Return timestream
        return timestream


class ApplySiderealDayFlag(task.SingleTask):
    """Prevent certain sidereal days from progressing
    further in the pipeline processing (e.g.,
    exclude certain sidereal days from the sidereal stack).
    """

    def setup(self, csd_flag):
        """Create dictionary from input ."""

        self.csd_dict = {}
        for cc, csd in enumerate(csd_flag.csd[:]):
            self.csd_dict[csd] = csd_flag.csd_flag[cc]

    def process(self, timestream):
        """If this sidereal day is flagged as good or
        if no flag is specified for this sidereal day,
        then return the timestream.  If this sidereal day
        is flagged as bad, then return None.

        Parameters
        ----------
        timestream : andata.CorrData / dcontainers.SiderealStream

        Returns
        -------
        timestream : andata.CorrData / dcontainers.SiderealStream or None
        """

        # Fetch the csd from the timestream attributes
        if "lsd" in timestream.attrs:
            this_csd = timestream.attrs["lsd"]
        elif "csd" in timestream.attrs:
            this_csd = timestream.attrs["csd"]
        else:
            this_csd = None

        # Is this csd specified in the file?
        if this_csd not in self.csd_dict:
            output = timestream

            if this_csd is None:
                msg = (
                    "Warning: input timestream does not have 'csd'/'lsd' attribute.  "
                    + "Will continue pipeline processing."
                )
            else:
                msg = (
                    "Warning: status of CSD %d not given in %s.  "
                    + "Will continue pipeline processing."
                ) % (this_csd, self.file_name)

        else:
            # Is this csd flagged good?
            this_flag = self.csd_dict[this_csd]

            if this_flag:
                output = timestream

                msg = (
                    "CSD %d flagged good.  " + "Will continue pipeline processing."
                ) % this_csd
            else:
                output = None

                msg = (
                    "CSD %d flagged bad.  " + "Will halt pipeline processing."
                ) % this_csd

        # Print whether or not we will continue processing this csd
        self.log.info(msg)

        # Return input timestream or None
        return output


class NanToNum(task.SingleTask):
    """Finds NaN and replaces with 0."""

    dataset = config.Property(proptype=str, default="vis")

    def process(self, timestream):
        """Converts any NaN in the vis dataset and weight dataset
        to the value 0.0.

        Parameters
        ----------
        timestream : andata.CorrData or dcontainers.SiderealStream

        Returns
        --------
        timestream : andata.CorrData or dcontainers.SiderealStream
        """

        # Make sure we are distributed over frequency
        timestream.redistribute("freq")

        # Loop over frequencies to reduce memory usage
        for lfi, fi in timestream.datasets[self.dataset][:].enumerate(0):
            # Set non-finite values of the visibility equal to zero
            flag = ~np.isfinite(timestream.datasets[self.dataset][fi])
            if np.any(flag):
                timestream.datasets[self.dataset][fi][flag] = 0.0
                timestream.weight[fi][
                    flag
                ] = 0.0  # Also set weights to zero so we don't trust values
                self.log.info(
                    "%d visibilities are non finite for frequency=%i (%.2f %%)"
                    % (np.sum(flag), fi, np.sum(flag) * 100.0 / flag.size)
                )

            # Set non-finite values of the weight equal to zero
            flag = ~np.isfinite(timestream.weight[fi])
            if np.any(flag):
                timestream.weight[fi][flag] = 0
                self.log.info(
                    "%d weights are non finite for frequency=%i (%.2f %%)"
                    % (np.sum(flag), fi, np.sum(flag) * 100.0 / flag.size)
                )

        return timestream


class RadiometerWeight(task.SingleTask):
    """Update vis_weight according to the radiometer equation:

    vis_weight_ij = Nsamples / V_ii V_jj
    """

    def process(self, timestream):
        """Takes the input timestream.flags['vis_weight'], recasts it from uint8 to float32,
        multiplies by the total number of samples, and divides by the autocorrelations of the
        two feeds that form each baseline.

        Parameters
        ----------
        timestream : andata.CorrData

        Returns
        --------
        timestream : andata.CorrData
        """

        from .calibration import _extract_diagonal as diag

        # Redistribute over the frequency direction
        timestream.redistribute("freq")

        if isinstance(timestream, andata.CorrData):
            self.log.debug("Converting weights to effective number of samples.")

            # Extract number of samples per integration period
            max_nsamples = timestream.attrs["gpu.gpu_intergration_period"][0]

            # Extract the maximum possible value of vis_weight
            max_vis_weight = np.iinfo(timestream.flags["vis_weight"].dtype).max

            # Calculate the scaling factor that converts from vis_weight value
            # to number of samples
            vw_to_nsamp = max_nsamples / float(max_vis_weight)

            # Scale vis_weight by the effective number of samples
            vis_weight = (
                timestream.flags["vis_weight"][:].astype(np.float32) * vw_to_nsamp
            )

            # Recast vis_weight as float32
            # Wrap to produce MPIArray
            vis_weight = mpiarray.MPIArray.wrap(
                vis_weight, axis=0, comm=timestream.comm
            )

            # Extract attributes
            vis_weight_attrs = memh5.attrs2dict(timestream.flags["vis_weight"].attrs)

            # Delete current uint8 dataset
            timestream["flags"].__delitem__("vis_weight")

            # Create new float32 dataset
            vis_weight_dataset = timestream.create_flag(
                "vis_weight", data=vis_weight, distributed=True
            )

            # Copy attributes
            memh5.copyattrs(vis_weight_attrs, vis_weight_dataset.attrs)

        elif isinstance(timestream, dcontainers.SiderealStream):
            self.log.debug(
                "Scaling weights by outer product of inverse receiver temperature."
            )

            # Extract the autocorrelation
            Trec = diag(timestream.vis).real

            # Invert the autocorrelation
            inv_Trec = tools.invert_no_zero(Trec)

            # Scale the weights by the outerproduct of the inverse autocorrelations
            tools.apply_gain(timestream.weight[:], inv_Trec, out=timestream.weight[:])

        else:
            raise RuntimeError("Format of `timestream` argument is unknown.")

        # Return timestream with updated weights
        return timestream


class BadNodeFlagger(task.SingleTask):
    """Flag out bad GPU nodes by giving zero weight to their frequencies.

    Parameters
    ----------
    nodes : list of ints
        Indices of bad nodes to flag.
    nodes_by_acq : dict
        Dictionary whose entries have the name of the acquisition as keyword
        and a list of the nodes to flag as bad for that acquisition as value.
    flag_freq_zero : boolean, optional
        Whether to flag out frequency zero.
    """

    nodes = config.Property(proptype=list, default=[])
    nodes_by_acq = config.Property(proptype=dict, default={})

    flag_freq_zero = config.Property(proptype=bool, default=False)

    def process(self, timestream):
        """Flag out bad nodes by giving them zero weight.

        Parameters
        ----------
        timestream : andata.CorrData or dcontainers.SiderealStream

        Returns
        -------
        flagged_timestream : same type as timestream
        """

        # Redistribute over frequency
        timestream.redistribute("freq")

        # Determine local frequencies
        sf = timestream.vis.local_offset[0]
        ef = sf + timestream.vis.local_shape[0]

        # Extract autocorrelation indices
        auto_pi = np.array(
            [
                ii
                for (ii, pp) in enumerate(timestream.index_map["prod"])
                if pp[0] == pp[1]
            ]
        )

        # Create bad node flag by checking for frequencies/time samples where
        # the autocorrelations are all zero
        good_freq_flag = np.any(timestream.vis[:, auto_pi, :].real > 0.0, axis=1)

        # Apply bad node flag
        timestream.weight[:] *= good_freq_flag[:, np.newaxis, :]

        # If requested, flag the first frequency
        if self.flag_freq_zero and mpiutil.rank0:
            timestream.weight[0] = 0

        # Set up map from frequency to node
        basefreq = np.linspace(800.0, 400.0, 1024, endpoint=False)
        nodelist = np.array(
            [np.argmin(np.abs(ff - basefreq)) % 16 for ff in timestream.freq]
        )

        # Manually flag frequencies corresponding to specific GPU nodes
        for node in self.nodes:
            for nind in np.flatnonzero(nodelist == node):
                if nind >= sf and nind < ef:
                    timestream.weight[nind] = 0

                    self.log.info("Flagging node %d, freq %d.", node, nind)

        # Manually flag frequencies corresponding to specific GPU nodes on specific acquisitions
        this_acq = timestream.attrs.get("acquisition_name", None)
        if this_acq in self.nodes_by_acq:
            # Grab list of nodes from input dictionary
            nodes_to_flag = self.nodes_by_acq[this_acq]
            if not hasattr(nodes_to_flag, "__iter__"):
                nodes_to_flag = [nodes_to_flag]

            # Loop over nodes and perform flagging
            for node in nodes_to_flag:
                for nind in np.flatnonzero(nodelist == node):
                    if nind >= sf and nind < ef:
                        timestream.weight[nind] = 0

                        self.log.info(
                            "Flagging node %d, freq %d.", mpiutil.rank, node, nind
                        )

        # Return timestream with bad nodes flagged
        return timestream


def daytime_flag(time):
    """Return a flag that indicates if times occur during the day.

    Parameters
    ----------
    time : np.ndarray[ntime,]
        Unix timestamps.

    Returns
    -------
    flag : np.ndarray[ntime,]
        Boolean flag that is True if the time occured during the day and False otherwise.
    """
    time = np.atleast_1d(time)
    flag = np.zeros(time.size, dtype=np.bool)

    rise = ephemeris.solar_rising(time[0] - 24.0 * 3600.0, end_time=time[-1])
    for rr in rise:
        ss = ephemeris.solar_setting(rr)[0]
        flag |= (time >= rr) & (time <= ss)

    return flag


def transit_flag(body, time, nsigma=2.0):
    """Return a flag that indicates if times occured near transit of a celestial body.

    Parameters
    ----------
    body : skyfield.starlib.Star
        Skyfield representation of a celestial body.
    time : np.ndarray[ntime,]
        Unix timestamps.
    nsigma : float
        Number of sigma to flag on either side of transit.

    Returns
    -------
    flag : np.ndarray[ntime,]
        Boolean flag that is True if the times occur within nsigma of transit
        and False otherwise.
    """
    time = np.atleast_1d(time)
    obs = ephemeris.chime

    # Create boolean flag
    flag = np.zeros(time.size, dtype=np.bool)

    # Find transit times
    transit_times = obs.transit_times(
        body, time[0] - 24.0 * 3600.0, time[-1] + 24.0 * 3600.0
    )

    # Loop over transit times
    for ttrans in transit_times:
        # Compute source coordinates
        sf_time = ephemeris.unix_to_skyfield_time(ttrans)
        pos = obs.skyfield_obs().at(sf_time).observe(body)

        alt = pos.apparent().altaz()[0]
        dec = pos.cirs_radec(sf_time)[1]

        # Make sure body is above horizon
        if alt.radians > 0.0:
            # Estimate the amount of time the body is in the primary beam
            # as +/- nsigma sigma, where sigma denotes the width of the
            # primary beam.  We use the lowest frequency and E-W (or X) polarisation,
            # since this is the most conservative (largest sigma).
            window_deg = nsigma * cal_utils.guess_fwhm(
                400.0, pol="X", dec=dec.radians, sigma=True
            )
            window_sec = window_deg * 240.0 * ephemeris.SIDEREAL_S

            # Flag +/- window_sec around transit time
            begin = ttrans - window_sec
            end = ttrans + window_sec
            flag |= (time >= begin) & (time <= end)

    # Return boolean flag indicating times near transit
    return flag


def taper_mask(mask, nwidth, outer=False):
    num = len(mask)
    if outer:
        tapered_mask = 1.0 - mask.astype(np.float)
    else:
        tapered_mask = mask.astype(np.float)

    taper = np.hanning(2 * nwidth - 1)

    dmask = np.diff(tapered_mask)
    transition = np.where(dmask != 0)[0]

    for tt in transition:
        if dmask[tt] > 0:
            ind = np.arange(tt, tt + nwidth)
            tapered_mask[ind % num] *= taper[:nwidth]
        else:
            ind = np.arange(tt + 2 - nwidth, tt + 2)
            tapered_mask[ind % num] *= taper[-nwidth:]

    if outer:
        tapered_mask = 1.0 - tapered_mask

    return tapered_mask


class MaskDay(task.SingleTask):
    """Mask out the daytime data.

    This task can also act as a base class for applying an arbitrary
    mask to the data based on time.  This is achieved by redefining
    the `_flag` method.

    Attributes
    ----------
    zero_data : bool, optional
        Zero the data in addition to modifying the noise weights
    taper_width : float, optional
        Width (in seconds) of the taper applied to the mask.  Creates a smooth transition from
        masked to unmasked regions using a cosine function.  Set to 0.0 for no taper (default).
    outer_taper : bool, optional
        If outer_taper is True, then the taper occurs in the unmasked region.
        If outer_taper is False, then the taper occurs in the masked region.
    """

    zero_data = config.Property(proptype=bool, default=False)
    taper_width = config.Property(proptype=float, default=0.0)
    outer_taper = config.Property(proptype=bool, default=True)

    def process(self, sstream):
        """Set the weight to zero during day time.

        Parameters
        ----------
        sstream : dcontainers.SiderealStream or equivalent
            Unmasked sidereal stack.

        Returns
        -------
        mstream : dcontainers.SiderealStream or equivalent
            Masked sidereal stream.
        """
        # Redistribute over frequency
        sstream.redistribute("freq")

        # Get flag that indicates day times (RAs)
        if "time" in sstream.index_map:
            time = sstream.time[:]
            ntaper = int(self.taper_width / np.abs(np.median(np.diff(time))))

            flag = self._flag(time)

        else:
            # Fetch either the LSD or CSD attribute.  In the case of a SidrealStack,
            # there will be multiple LSDs and the flag will be the logical OR of the
            # flag from each individual LSD.
            csd = (
                sstream.attrs["lsd"] if "lsd" in sstream.attrs else sstream.attrs["csd"]
            )
            if not hasattr(csd, "__iter__"):
                csd = [csd]

            ra = sstream.ra[:]
            ntaper = int(
                self.taper_width
                / (np.abs(np.median(np.diff(ra))) * 240.0 * ephemeris.SIDEREAL_S)
            )

            flag = np.zeros(ra.size, dtype=np.bool)
            for cc in csd:
                time = ephemeris.csd_to_unix(cc + ra / 360.0)
                flag |= self._flag(time)

        # Log how much data were masking
        self.log.info(
            "%0.2f percent of data will be masked."
            % (100.0 * np.sum(flag) / float(flag.size),)
        )

        # Apply the mask
        if np.any(flag):
            # If requested, apply taper.
            if ntaper > 0:
                self.log.info("Applying taper over %d time samples." % ntaper)
                flag = taper_mask(flag, ntaper, outer=self.outer_taper)

            # Apply the mask to the weights
            sstream.weight[:] *= 1.0 - flag

            # If requested, apply the mask to the data
            if self.zero_data:
                sstream.vis[:] *= 1.0 - flag

        # Return masked sidereal stream
        return sstream

    def _flag(self, time):
        return daytime_flag(time)


# Alias DayMask to MaskDay for backwards compatibility
DayMask = MaskDay


class MaskSource(MaskDay):
    """Mask out data near source transits.

    Attributes
    ----------
    source : str or list of str
        Name of the source(s) in the same format as `ephemeris.source_dictionary`.
    nsigma : float
        Mask this number of sigma on either side of source transit.
        Here sigma is the exepected width of the primary beam for
        an E-W polarisation antenna at 400 MHz as defined by the
        `ch_util.cal_utils.guess_fwhm` function.
    """

    source = config.Property(default=None)
    nsigma = config.Property(proptype=float, default=2.0)

    def setup(self):
        """Save the skyfield bodies representing the sources to the `body` attribute."""
        if self.source is None:
            raise ValueError(
                "Must specify name of the source to mask as config property."
            )
        elif isinstance(self.source, list):
            source = self.source
        else:
            source = [self.source]

        self.body = [ephemeris.source_dictionary[src] for src in source]

    def _flag(self, time):
        flag = np.zeros(time.size, dtype=np.bool)
        for body in self.body:
            flag |= transit_flag(body, time, nsigma=self.nsigma)

        return flag


class MaskSun(MaskSource):
    """Mask out data near solar transit."""

    def setup(self):
        """Save the skyfield body for the sun."""
        planets = ephemeris.skyfield_wrapper.load("de421.bsp")
        self.body = [planets["sun"]]


class MaskMoon(MaskSource):
    """Mask out data near lunar transit."""

    def setup(self):
        """Save the skyfield body for the moon."""
        planets = ephemeris.skyfield_wrapper.load("de421.bsp")
        self.body = [planets["moon"]]


class MaskRA(task.SingleTask):
    """Mask out a range in right ascension.

    Attributes
    ----------
    start, end : float
        Start and end of masked out region.
    width : float
        Use a smooth transition of given width between the fully masked and
        unmasked data. This is interior to the region marked by start and end.
    zero_data : bool, optional
        Zero the data in addition to modifying the noise weights
        (default is True).
    remove_average : bool, optional
        Estimate and remove the mean level from each visibilty. This estimate
        does not use data from the masked region.
    """

    start = config.Property(proptype=float, default=90.0)
    end = config.Property(proptype=float, default=270.0)

    width = config.Property(proptype=float, default=60.0)

    zero_data = config.Property(proptype=bool, default=True)
    remove_average = config.Property(proptype=bool, default=True)

    def process(self, sstream):
        """Apply a day time mask.

        Parameters
        ----------
        sstream : dcontainers.SiderealStream
            Unmasked sidereal stack.

        Returns
        -------
        mstream : dcontainers.SiderealStream
            Masked sidereal stream.
        """

        sstream.redistribute("freq")

        ra_shift = (sstream.ra[:] - self.start) % 360.0
        end_shift = (self.end - self.start) % 360.0

        # Crudely mask the on and off regions
        mask_bool = ra_shift > end_shift

        # Put in the transition at the start of the day
        mask = np.where(
            ra_shift < self.width,
            0.5 * (1 + np.cos(np.pi * (ra_shift / self.width))),
            mask_bool,
        )

        # Put the transition at the end of the day
        mask = np.where(
            np.logical_and(ra_shift > end_shift - self.width, ra_shift <= end_shift),
            0.5 * (1 + np.cos(np.pi * ((ra_shift - end_shift) / self.width))),
            mask,
        )

        if self.remove_average:
            # Estimate the mean level from unmasked data
            import scipy.stats

            nanvis = (
                sstream.vis[:]
                * np.where(mask_bool, 1.0, np.nan)[np.newaxis, np.newaxis, :]
            )
            average = scipy.stats.nanmedian(nanvis, axis=-1)[:, :, np.newaxis]
            sstream.vis[:] -= average

        # Apply the mask to the data
        if self.zero_data:
            sstream.vis[:] *= mask

        # Modify the noise weights
        sstream.weight[:] *= tools.invert_no_zero(mask**2)

        return sstream


class MaskCHIMEData(task.SingleTask):
    """Mask out data ahead of map making.

    Attributes
    ----------
    intra_cylinder : bool
        Include baselines within the same cylinder (default=True).
    xx_pol : bool
        Include X-polarisation (default=True).
    yy no_pol : bool
        Include Y-polarisation (default=True).
    cross_pol : bool
        Include cross-polarisation (default=True).
    """

    intra_cylinder = config.Property(proptype=bool, default=True)

    xx_pol = config.Property(proptype=bool, default=True)
    yy_pol = config.Property(proptype=bool, default=True)
    cross_pol = config.Property(proptype=bool, default=True)

    def setup(self, tel):
        """Setup the task.

        Parameters
        ----------
        tel : :class:`ch_pipeline.core.pathfinder.CHIME`
            CHIME telescope class to use to get feed information.
        """
        self.telescope = io.get_telescope(tel)

    def process(self, mmodes):
        """Mask out unwanted datain the m-modes.

        Parameters
        ----------
        mmodes : dcontainers.MModes

        Returns
        -------
        mmodes : dcontainers.MModes
        """

        tel = self.telescope

        mmodes.redistribute("m")

        mw = mmodes.weight[:]

        for pi, (fi, fj) in enumerate(mmodes.prodstack):
            oi, oj = tel.feeds[fi], tel.feeds[fj]

            # Check if baseline is intra-cylinder
            if not self.intra_cylinder and (oi.cyl == oj.cyl):
                mw[..., pi] = 0.0

            # Check all the polarisation states
            is_xx = tools.is_array_x(oi) and tools.is_array_x(oj)
            is_yy = tools.is_array_y(oi) and tools.is_array_y(oj)

            if not self.xx_pol and is_xx:
                mw[..., pi] = 0.0

            if not self.yy_pol and is_yy:
                mw[..., pi] = 0.0

            if not self.cross_pol and not (is_xx or is_yy):
                mw[..., pi] = 0.0

        return mmodes


class MaskCHIMEMisc(task.SingleTask):
    """Some miscellaneous data masking routines."""

    mask_clock = config.Property(proptype=bool, default=True)

    mask_nodes = config.Property(proptype=list, default=None)

    mask_freq = config.Property(proptype=list, default=None)

    def process(self, ss):
        ss.redistribute("prod")

        # Mask out the 10 MHz lines
        if self.mask_clock:
            # Identify the frequency bins that contain the 10 MHz clock line
            m10 = (
                np.abs(((ss.freq["centre"] + 5) % 10.0) - 5.0) > 0.5 * ss.freq["width"]
            )

            ss.weight[:] *= m10[:, np.newaxis, np.newaxis]

        if self.mask_nodes is not None:
            node_index = ((800.0 - ss.freq["centre"]) / 400.0 * 1024) % 16

            for ni in self.mask_nodes:
                node_mask = node_index != ni
                ss.weight[:] *= node_mask[:, np.newaxis, np.newaxis]

        if self.mask_freq is not None:
            fmask = np.ones_like(ss.freq["centre"])
            fmask[self.mask_freq] = 0.0

            ss.weight[:] *= fmask[:, np.newaxis, np.newaxis]

        return ss


class MaskDecorrelatedCylinder(task.SingleTask):
    """Identify and mask frequencies and times where a cylinder decorrelated.

    If the error rate is high on a backplane link in the second stage shuffle
    of the F-engine corner turn, then on rare occassions (few times per year)
    the data streams being handled by that FPGA can become misaligned or
    desynchronized with the rest of the data streams.  The 512 correlator inputs
    on the cylinder corresponding to that pair of FPGA crates will have negligible
    correlation with all other inputs for the 64 frequencies received by that
    FPGA during the second stage shuffle.  This will persist until the data streams
    are re-synchronized with a correlator restart.

    This task identifies times and frequencies affected by these misalignment events
    by examining, for each cylinder, the ratio of 1-cylinder-separation, co-polar
    visibilities acquired by redundant baselines that do not contain the cylinder
    to those that do contain the cylinder.  This ratio will be close to 1 under
    normal operations since the baselines are largely redundant, however it will
    become very large after a cylinder becomes misaligned since there is no
    correlation and the denominator drops to near zero.

    Additionally, if provided the mapping between frequency channel and FPGA,
    this task will ensure that when there is evidence of a decorrelated cylinder,
    all 64 frequencies handled by the problematic FPGA are masked.

    Parameters
    ----------
    threshold: int
        Mask frequencies and times where the median ratio of the
        average-without-cylinder to average-with-cylinder visibility
        amplitude is greater than this threshold.
    max_frac_freq: float
        Mask any frequency that was transmitted by an FPGA motherboard slot
        with more than this fraction of frequencies masked by the above
        threshold.  Only relevant if the frequency map is provided at setup.
    """

    threshold = config.Property(proptype=float, default=5.0)
    max_frac_freq = config.Property(proptype=float, default=0.1)

    def setup(self, freq_map=None):
        """Determine the frequencies handled by each FPGA motherboard slot.

        If a motherboard decorrelates, then all frequencies transmitted by
        that slot will be affected.

        Parameters
        ----------
        freq_map : FrequencyMap
            The mapping between frequency bin and [crate, slot, link]
            as a function of time.
        """

        self._set_slot_freqs(freq_map)

    def process(self, data, inputmap):
        """Create the mask.

        Parameters
        ----------
        data : TimeStream
            Visibilites before averaging over cylinders.
        inputmap : list of :class:`CorrInput`
            A list describing the inputs in data.

        Returns
        -------
        out : RFIMask
            Mask with True indicating that a cylinder decorrelated
            at that frequency and time.
        """

        from draco.analysis.ringmapmaker import find_grid_indices

        # Distribute over frequencies
        data.redistribute("freq")

        # Get polarisations of feeds
        pol = tools.get_feed_polarisations(inputmap)

        # Get positions of feeds
        pos = tools.get_feed_positions(inputmap)

        # Get cylinder each feed is on
        cyl = np.array([inp.cyl if tools.is_chime(inp) else -1 for inp in inputmap])
        ucyl = np.unique(cyl[cyl >= 0])
        ncyl = ucyl.size

        # Make sure that none of our typical products reference non-array feeds
        stack_new, stack_flag = tools.redefine_stack_index_map(
            inputmap, data.prod, data.stack, data.reverse_map["stack"]
        )

        valid_stack = np.flatnonzero(stack_flag)
        ninvalid = stack_new.size - valid_stack.size

        if ninvalid > 0:
            stack_new = stack_new[valid_stack]
            self.log.info(
                "Could not find appropriate reference inputs for "
                f"{ninvalid:0.0f} stacked products.  Ignoring these "
                "products in decorrelated cylinder calculation."
            )

        t = data.prod[stack_new["prod"]]

        prodstack = t.copy()
        conj = stack_new["conjugate"]
        prodstack["input_a"] = np.where(conj, t["input_b"], t["input_a"])
        prodstack["input_b"] = np.where(conj, t["input_a"], t["input_b"])

        # Calculate baseline distance and polarisation pair
        index_a = prodstack["input_a"]
        index_b = prodstack["input_b"]

        bdist = pos[index_a] - pos[index_b]
        bpol = np.core.defchararray.add(pol[index_a], pol[index_b])

        # Find the grid indices
        xind, yind, dx, dy = find_grid_indices(bdist)

        # Only use the co-polar, 1-cylinder separation visibilities for this analysis
        ico = np.flatnonzero((pol[index_a] == pol[index_b]) & (np.abs(xind) == 1))

        # Create an identifier based on the polarisation pair and north-south baseline
        pol_map = {pstr: pp for pp, pstr in enumerate(np.unique(bpol[ico]))}

        idd = np.zeros((ico.size, 2), dtype=int)
        idd[:, 0] = [pol_map[bpol[ii]] for ii in ico]
        idd[:, 1] = yind[ico]

        # Find the unique pol/baselines and the inverse map
        uidd, index = np.unique(idd, return_inverse=True, axis=0)

        isort = np.argsort(index)

        # Determine boundaries of unique pol/baselines
        bnd = np.concatenate(
            ([0], np.flatnonzero(np.diff(index[isort]) > 0) + 1, [index.size])
        )

        # Ignore any unique pol/baseline that does not have ncyl - 1 redundant copies.
        # This can happen due to the valid_stack selection.
        ncopies = bnd[1:] - bnd[:-1]
        bflag = ncopies == (ncyl - 1)
        nmeas = np.sum(bflag)

        # Loop over the unique pol/baselines and for each one determine if each of the
        # cylinders is present in each of the redundant copies
        pindex = np.zeros((nmeas, ncyl - 1), dtype=int)
        flag_with = np.zeros((ncyl, nmeas, ncyl - 1), dtype=bool)

        cc = 0
        for bb in range(nmeas):
            if not bflag[bb]:
                continue

            pi = ico[isort[bnd[bb] : bnd[bb + 1]]]
            pindex[cc] = valid_stack[pi]

            flag_with[:, cc, :] = (
                ucyl[:, np.newaxis] == cyl[index_a[pi]][np.newaxis, :]
            ) | (ucyl[:, np.newaxis] == cyl[index_b[pi]][np.newaxis, :])

            cc += 1

        flag_without = ~flag_with

        # Extract the required data products
        vis = data.vis[:].local_array[:, pindex, :]
        flag = (data.weight[:].local_array[:] > 0.0)[:, pindex, :]

        # Define slices that will expand both the coefficients and data
        # to the correct shape for broadcasting against each other
        cslc = (slice(None), slice(None), slice(None), None)
        dslc = (None, slice(None), slice(None), slice(None))

        # Create an array to fill with the final mask
        mask = np.zeros((vis.shape[0], vis.shape[-1]), dtype=bool)

        # Loop over frequencies
        for ff in range(vis.shape[0]):
            # For each cylinder, average the magnitude of the visibilities for:
            #   all redundant baselines that contain that cylinder
            #   all redundant baselines that do not contain that cylinder
            fwith = flag_with[cslc] * flag[ff][dslc]
            fwithout = flag_without[cslc] * flag[ff][dslc]

            norm_with = np.sum(fwith, axis=2).astype(np.float32)
            norm_without = np.sum(fwithout, axis=2).astype(np.float32)

            avg_with = np.sum(
                fwith * np.abs(vis[ff][dslc]), axis=2
            ) * tools.invert_no_zero(norm_with)
            avg_without = np.sum(
                fwithout * np.abs(vis[ff][dslc]), axis=2
            ) * tools.invert_no_zero(norm_without)

            # Take the ratio of without the cylinder to with the cylinder
            ratio = avg_without * tools.invert_no_zero(avg_with)

            valid = (norm_with > 0.0) & (norm_without > 0.0)

            # If all entries along axis=1 are invalid then the following
            # nanmedian will throw a warning if we just fill the invalid
            # positions with NaN's. Instead fill samples where all axis=1 is
            # invalid with zeros which will still fail the following comparison
            # to generate the mask
            fill = np.where(valid.any(axis=1), np.nan, 0)[:, np.newaxis, ...]
            # Take the median of the ratio over all unique baselines
            med = np.nanmedian(np.where(valid, ratio, fill), axis=1)

            # Mask any time where the median was greater than some threshold
            mask[ff] = np.any(med > self.threshold, axis=0)

        # Gather the mask for all frequencies on all nodes
        mask = mpiarray.MPIArray.wrap(mask, axis=0).allgather()

        # If more than some (user specified) fraction of frequencies transmitted
        # by an FPGA motherboard slot have been masked, then mask all frequencies
        # transmitted by that motherboard slot.  The cylinder decorrelation is
        # expected to affect all of these frequencies.  This step is only possible
        # if the frequency map as a function of time has been provided on setup.
        if self.freq_map is not None and len(data.freq) == 1024:
            grouper, slot_index = self._get_slot_freqs(data.time[0])

            frac_freq_masked = np.sum(mask[grouper, :], axis=1) / grouper.shape[1]

            mask = mask | (frac_freq_masked > self.max_frac_freq)[slot_index, :]

        # Print the fraction of data that has been masked by this task
        self.log.info(
            "%0.2f percent of data was masked due to a decorrelated cylinder."
            % (100.0 * np.sum(mask) / np.prod(mask.shape),)
        )

        # Create output container and store final mask
        out = dcontainers.RFIMask(axes_from=data, attrs_from=data)

        out.mask[:] = mask

        return out

    def _set_slot_freqs(self, freq_map):
        """Determine the slot to frequency map as a function of time.

        Parameters
        ----------
        freq_map : FrequencyMap
            The mapping between frequency bin and [crate, slot, link]
            as a function of time.
        """

        if freq_map is not None:
            islot = list(freq_map.level).index("slot")
            slot = freq_map.stream[:, :, islot]

            nslot = np.max(slot) + 1
            ntime, nfreq = slot.shape
            nfreq_per_slot = nfreq // nslot

            grouper = np.zeros((ntime, nslot, nfreq_per_slot), dtype=int)
            for tt in range(ntime):
                grouper[tt] = np.argsort(slot[tt], kind="mergesort").reshape(
                    nslot, nfreq_per_slot
                )

            self.freq_map = freq_map
            self.grouper = grouper
            self.slot_index = slot

        else:
            self.freq_map = None

    def _get_slot_freqs(self, timestamp):
        """Look up the slot to frequency map that was used at a given time.

        Parameters
        ----------
        timestamp : float64
            Unix timestamp.

        Returns
        -------
        grouper : np.ndarray[nslot, nfreq_per_slot]
            Index into the frequency axis that will group
            frequency channels based on the FPGA motherboard slot
            that transmitted them.
        slot_index : np.ndarray[nfreq,]
            Index into the slot axis that will yield the
            FPGA motherboard slot that transmitted each
            frequency channel.
        """

        tindex = np.digitize(timestamp, self.freq_map.time) - 1

        if tindex < 0:
            tbefore = (self.freq_map.time.min() - timestamp) / (24 * 3600)
            raise RuntimeError(
                "Requested timestamp is before the earliest time "
                f"in the frequency map file by {tbefore:0.1f} days."
            )

        if timestamp > self.freq_map.attrs["end_time"]:
            tafter = (timestamp - self.freq_map.attrs["end_time"]) / (24 * 3600)
            self.log.warning(
                "Requested timestamp is after the end time covered "
                f"by the frequency map file by {tafter:0.1f} days.  "
                "Please ensure that the frequency map has not been "
                "updated since then."
            )

        return self.grouper[tindex], self.slot_index[tindex]


class ExpandMask(task.SingleTask):
    """Expand a mask along the time/RA axis.

    Used to mask the transitional regions between good and bad data.

    Parameters
    ----------
    nexpand : int
        If a time/RA is within nexpand samples from a masked time/RA,
        then it will be masked in the output.
    in_place : bool
        If True, then overwrite the raw mask with the expanded mask.
        If False, then create a new container with the expanded mask.
    """

    nexpand = config.Property(proptype=int, default=1)
    in_place = config.Property(proptype=bool, default=False)

    def process(self, raw_mask):
        """Mask any times/RAs that neighbor a masked time/RA.

        Parameters
        ----------
        raw_mask : RFIMask or SiderealRFIMask

        Returns
        -------
        exp_mask : RFIMask or SiderealRFIMask
        """

        nfreq, ntime = raw_mask.mask[:].shape

        mraw = np.zeros((nfreq, ntime + 2 * self.nexpand), dtype=bool)
        mraw[:, self.nexpand : -self.nexpand] = raw_mask.mask[:]

        window = 2 * self.nexpand + 1
        mexp = np.any(rfi._rolling_window_lastaxis(mraw, window), axis=-1)

        if self.in_place:
            exp_mask = raw_mask
        else:
            exp_mask = dcontainers.empty_like(raw_mask)

        exp_mask.mask[:] = mexp

        return exp_mask


class DataFlagger(task.SingleTask):
    """Flag data based on DataFlags in database.

    Parameters
    ----------
    flag_type : list
        List of DataFlagType names to apply. Defaults to the flags representing ranges of time known to be bad
        that may effect the delay spectrum estimate.
    """

    flag_type = config.list_type(
        type_=str,
        default=[
            "acjump",
            "bad_calibration_acquisition_restart",
            "bad_calibration_fpga_restart",
            "bad_calibration_gains",
            "decorrelated_cylinder",
            "globalflag",
            "rain1mm",
        ],
    )

    def setup(self):
        """Query the database for flags of the requested types."""
        flags = {}

        # Query flag database if on 0th node
        if self.comm.rank == 0:
            connect_database()
            flag_types = df.DataFlagType.select()
            possible_flags = []
            for ft in flag_types:
                possible_flags.append(ft.name)
                if ft.name in self.flag_type or "all" in self.flag_type:
                    self.log.info("Querying for %s Flags" % ft.name)
                    new_flags = df.DataFlag.select().where(df.DataFlag.type == ft)
                    flags[ft.name] = list(new_flags)

            # Check that user-proved flag names are valid
            for flag_name in self.flag_type:
                if flag_name != "all" and flag_name not in possible_flags:
                    self.log.warning("Warning: Unrecognized Flag %s" % flag_name)

        # Share flags with other nodes
        flags = self.comm.bcast(flags, root=0)

        # Save flags to class attribute
        self.log.info(
            "Found %d Flags in Total." % sum([len(flg) for flg in flags.values()])
        )
        self.flags = flags

    def process(self, timestream):
        """Set weight to zero for range of data covered by the database flags.

        Flags are applied based on time, frequency, and (for non-stacked data) input.

        Parameters
        ----------
        timestream : andata.CorrData or dcontainers.SiderealStream or dcontainers.TimeStream
            Timestream to flag.

        Returns
        -------
        timestream : andata.CorrData or dcontainers.SiderealStream or dcontainers.TimeStream
            Returns the same timestream object with a modified weight dataset.
        """
        # Redistribute over the frequency direction
        timestream.redistribute("freq")

        # Determine whether timestream is stacked data
        stacked = len(timestream.index_map["prod"]) > len(
            timestream.index_map["stack"]["prod"]
        )

        # If not stacked, determine which inputs are in the timestream.
        # If stacked, assume flags apply to all products.
        if not stacked:
            inputs = timestream.index_map["input"]["chan_id"][:]
            ninputs = len(inputs)
        else:
            ninputs = 1

        # Get time axis or convert RA axis
        if "ra" in timestream.index_map:
            ra = timestream.index_map["ra"][:]
            if "lsd" in timestream.attrs:
                csd = timestream.attrs["lsd"]
            else:
                csd = timestream.attrs["csd"]
            time = ephemeris.csd_to_unix(csd + ra / 360.0)
        else:
            time = timestream.time

        ntime = len(time)

        # Determine local frequencies
        sf = timestream.weight.local_offset[0]
        ef = sf + timestream.weight.local_shape[0]
        local_freq = timestream.freq[sf:ef]
        nfreq = len(local_freq)

        # Find the bin number of each local frequency
        basefreq = np.linspace(800.0, 400.0, 1024, endpoint=False)
        local_bin = np.array([np.argmin(np.abs(ff - basefreq)) for ff in local_freq])

        # Initiate weight mask (1 means not flagged)
        weight_mask = np.ones((nfreq, ninputs, ntime), dtype=bool)

        # Loop over flags of requested types
        for flag_type, flag_list in self.flags.items():
            for flag in flag_list:
                # Identify flagged times
                time_idx = (time >= flag.start_time) & (time <= flag.finish_time)
                if np.any(time_idx):
                    # Print info to log about why the data is being flagged
                    msg = (
                        "%d (of %d) samples flagged by a %s DataFlag covering %s to %s."
                        % (
                            np.sum(time_idx),
                            time_idx.size,
                            flag_type,
                            ephemeris.unix_to_datetime(flag.start_time).strftime(
                                "%Y%m%dT%H%M%SZ"
                            ),
                            ephemeris.unix_to_datetime(flag.finish_time).strftime(
                                "%Y%m%dT%H%M%SZ"
                            ),
                        )
                    )
                    self.log.info(msg)

                    # Refine the mask based on any frequency or input selection
                    flag_mask = time_idx[np.newaxis, np.newaxis, :]
                    if flag.freq is not None:
                        # `and` with flagged local frequencies
                        # By default, all frequencies are flagged
                        flag_mask = (
                            flag_mask
                            & flag.freq_mask[local_bin, np.newaxis, np.newaxis]
                        )

                    if flag.inputs is not None and not stacked:
                        # `and` with flagged inputs
                        # By default, all inputs are flagged
                        flag_mask = (
                            flag_mask & flag.input_mask[np.newaxis, inputs, np.newaxis]
                        )

                    # set weight=0 where flag=1
                    weight_mask = weight_mask & np.logical_not(flag_mask)

        # Multiply weight mask by existing weight dataset
        weight = timestream.weight[:]
        weight_mask = weight_mask.astype(weight.dtype)
        if stacked:
            # Apply same mask to all products
            weight.local_array[:] *= weight_mask
        else:
            # Use apply_gain function to apply mask based on product map
            products = timestream.index_map["prod"][
                timestream.index_map["stack"]["prod"]
            ]
            tools.apply_gain(weight, weight_mask, out=weight, prod_map=products)

        self.log.info(
            "%0.2f percent of data was flagged as bad."
            % (100.0 * (1.0 - (np.sum(weight_mask) / np.prod(weight_mask.shape))),)
        )

        return timestream


class ApplyInputFlag(task.SingleTask):
    """Flag bad inputs.

    Uses the flaginput acquisition generated by the real-time pipeline.
    """

    def setup(self, files, observer=None):
        """Load flaginput files that cover full span of time to be processed.

        Parameters
        ----------
        files: list of str
            List of paths to files containing the input flags.
        observer : caput.time.Observer, optional
            Details of the observer, if not set default to CHIME.
        """
        self.observer = ephemeris.chime if observer is None else observer

        self.input_flags = andata.FlagInputData.from_acq_h5(files, datasets=["flag"])

    def process(self, data):
        """Lookup and apply the relevant input flags.

        Parameters
        ----------
        data: TODContainer, SiderealStream, TrackBeam
            Must have `input` axis.

        Returns
        -------
        data: TODContainer, SiderealStream, TrackBeam
            The input container with the weight dataset
            set to zero for bad inputs.
        """
        axis = list(data.weight.attrs["axis"])
        daxis = {ax: ind for ind, ax in enumerate(axis)}

        if "input" not in daxis:
            raise RuntimeError("The weight dataset must have an input axis.")

        timestamp, time_axis = self._get_timestamp(data)

        sel = {time_axis: slice(None), "input": slice(None)}
        if data.distributed:
            avail = [ax for ax in axis if ax not in [time_axis, "input"]]

            if len(avail) > 0:
                data.redistribute(avail[0])
            else:
                data.redistribute("input")
                sinp = data.weight.local_offset[daxis["input"]]
                einp = sinp + data.weight.local_shape[daxis["input"]]

                sel["input"] = slice(sinp, einp)

        expand = tuple([sel.get(ax, None) for ax in axis])

        transpose = (time_axis is not None) and (daxis["input"] < daxis[time_axis])
        flag = self.input_flags.resample("flag", timestamp, transpose=transpose)

        if time_axis is None:
            flag = flag[0]

        data.weight[:] *= flag[expand].astype(np.float32)

        return data

    def _get_timestamp(self, data):
        """Determine the timestamp based on the container type."""

        if issubclass(type(data), tod.TOData):
            timestamp = data.time
            time_axis = "time"

        elif issubclass(type(data), dcontainers.SiderealStream):
            ra = data.ra
            lsd = data.attrs["lsd"] if "lsd" in data.attrs else data.attrs["csd"]
            timestamp = self.observer.lsd_to_unix(lsd + data.ra / 360.0)
            time_axis = "ra"

        elif issubclass(type(data), dcontainers.TrackBeam):
            ra = data.pix["phi"][:] + data.attrs["cirs_ra"]
            lsdf = self.observer.unix_to_lsd(data.attrs["transit_time"]) + ra / 360.0
            timestamp = self.observer.lsd_to_unix(lsdf)
            time_axis = "pix"

        elif issubclass(type(data), dcontainers.StaticGainData):
            timestamp = data.attrs["time"]
            time_axis = None

        else:
            raise RuntimeError(
                "Do not know how to calculate timestamp"
                f" for container type {type(data)}."
            )

        return timestamp, time_axis


class SetInputFlag(ApplyInputFlag):
    """Set input flags dataset based on values from the real-time pipeline at the time.

    This is useful for holographic observations or SiderealStreams, neither of which
    have their input_flags dataset populated.
    """

    def process(self, data):
        """Lookup and save the input flags based on the time.

        Parameters
        ----------
        data: TODContainer, SiderealStream, TrackBeam
            Must have `input_flags` dataset.

        Returns
        -------
        data: TODContainer, SiderealStream, TrackBeam
            The input container with the input_flags dataset
            set to 1.0 for good inputs and 0.0 for bad inputs.
        """
        daxis = {ax: ind for ind, ax in enumerate(data.input_flags.attrs["axis"])}

        timestamp, time_axis = self._get_timestamp(data)

        flag = self.input_flags.resample(
            "flag", timestamp, transpose=daxis["input"] < daxis[time_axis]
        )

        data.input_flags[:] = flag

        return data
