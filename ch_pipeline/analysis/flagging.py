"""
===============================================================
Tasks for Flagging Data (:mod:`~ch_pipeline.analysis.flagging`)
===============================================================

.. currentmodule:: ch_pipeline.analysis.flagging

Tasks for calculating flagging out unwanted data. This includes RFI removal, and
data quality flagging on timestream data; sun excision on sidereal data; and
pre-map making flagging on m-modes.

Tasks
=====

.. autosummary::
    :toctree: generated/

    RFIFilter
    ChannelFlagger
    MonitorCorrInput
    TestCorrInput
    AccumulateCorrInputMask
    ApplyCorrInputMask
    ApplySiderealDayFlag
    BadNodeFlagger
    DayMask
    MaskData
    MaskCHIMEData
"""
import os.path
import numpy as np

from caput import mpiutil, mpiarray, memh5, config, pipeline
from ch_util import rfi, data_quality, tools, ephemeris, cal_utils

from ..core import containers, task

class RFIFilter(task.SingleTask):
    """Filter RFI from a Timestream.

    This task works on the parallel
    :class:`~ch_pipeline.containers.TimeStream` objects.

    Attributes
    ----------
    threshold_mad : float
        Threshold above which we mask the data.
    """

    threshold_mad = config.Property(proptype=float, default=5.0)

    flag1d = config.Property(proptype=bool, default=False)

    def process(self, data):

        if mpiutil.rank0:
            print "RFI filtering %s" % data.attrs['tag']

        # Construct RFI mask
        mask = rfi.flag_dataset(data, only_autos=False, threshold=self.threshold_mad, flag1d=self.flag1d)

        data.weight[:] *= (1 - mask).astype(data.weight.dtype)  # Switch from mask to inverse noise weight

        # Redistribute across frequency
        data.redistribute('freq')

        return data


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
        timestream.redistribute('freq')

        # Find the indices for frequencies in this timestream nearest
        # to the given physical frequencies
        freq_ind = [np.argmin(np.abs(timestream.freq - freq)) for freq in self.test_freq]

        # Create a global channel weight (channels are bad by default)
        chan_mask = np.zeros(timestream.ninput, dtype=np.int)

        # Mark any CHIME channels as good
        for i in range(timestream.ninput):
            if isinstance(inputmap[i], tools.CHIMEAntenna):
                chan_mask[i] = 1

        # Calculate start and end frequencies
        sf = timestream.vis.local_offset[0]
        ef = sf + timestream.vis.local_shape[0]

        # Iterate over frequencies and find bad channels
        for fi in freq_ind:

            # Only run good_channels if frequency is local
            if fi >= sf and fi < ef:

                # Run good channels code and unpack arguments
                res = data_quality.good_channels(timestream, test_freq=fi, inputs=inputmap, verbose=False)
                good_gains, good_noise, good_fit, test_channels = res

                print ("Frequency %i bad channels: blank %i; gains %i; noise %i; fit %i %s" %
                       ( fi, np.sum(chan_mask == 0), np.sum(good_gains == 0), np.sum(good_noise == 0),
                         np.sum(good_fit == 0), '[ignored]' if self.ignore_fit else ''))

                if good_noise is None:
                    good_noise = np.ones_like(test_channels)

                # Construct the overall channel mask for this frequency
                if not self.ignore_gains:
                    chan_mask[test_channels] *= good_gains
                if not self.ignore_noise:
                    chan_mask[test_channels] *= good_noise
                if not self.ignore_fit:
                    chan_mask[test_channels] *= good_fit

        # Gather the channel flags from all nodes, and combine into a
        # single flag (checking that all tests pass)
        chan_mask_all = np.zeros((timestream.comm.size, timestream.ninput), dtype=np.int)
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
    """ Monitor good correlator inputs over several sidereal days.

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

        from sidereal import get_times, _days_in_csd
        from ch_util import andata

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
            filemap = [ (day, _days_in_csd(day, se_csd, extra=0.005)) for day in days ]

            # Determine the time range for each day
            timemap = [ (day, ephemeris.csd_to_unix(np.array([day, day+1])))
                         for day in days ]

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
                    ValueError("Differing number of corr inputs for csd %d and csd %d." % (fmap[0], filemap[0][0]))
                elif np.sum(data_r.input['correlator_input'] != input_map['correlator_input']) > 0:
                    ValueError("Different corr inputs for csd %d and csd %d." % (fmap[0], filemap[0][0]))

                if len(data_r.freq) != nfreq:
                    ValueError("Differing number of frequencies for csd %d and csd %d." % (fmap[0], filemap[0][0]))
                elif np.sum(data_r.freq['centre'] != freq['centre']) > 0:
                    ValueError("Different frequencies for csd %d and csd %d." % (fmap[0], filemap[0][0]))

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
        csd_flag : container.SiderealDayFlag
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
            print "Rank %d calling channel monitor for csd %d." % (mpiutil.rank, csd)

            # Create an instance of chan_monitor for this day
            cm = chan_monitor.ChanMonitor(*time_range)

            # Run the full test
            try:
                cm.full_check()
            except (RuntimeError, ValueError) as error:
                # No sources available for this csd
                print("    Rank %d, csd %d: %s" % (mpiutil.rank, csd, error))
                continue

            # Accumulate flags over multiple days
            input_mask[i_local, :] = cm.good_ipts & cm.pwds
            good_day_flag[i_local] = True

            # If requested, write to disk
            if self.save:

                # Create a container to hold the results
                input_mon = containers.CorrInputMonitor(freq=self.freq, input=self.input_map,
                                                        distributed=False)

                # Place the results in the container
                input_mon.input_mask[:] = cm.good_ipts
                input_mon.input_powered[:] = cm.pwds
                input_mon.freq_mask[:] = cm.good_freqs
                input_mon.freq_powered[:] = cm.gpu_node_flag

                if hasattr(cm, 'postns'):
                    input_mon.add_dataset('position')
                    input_mon.position[:] = cm.postns

                if hasattr(cm, 'expostns'):
                    input_mon.add_dataset('expected_position')
                    input_mon.expected_position[:] = cm.expostns

                if cm.source1 is not None:
                    input_mon.attrs['source1'] = cm.source1.name

                if cm.source2 is not None:
                    input_mon.attrs['source2'] = cm.source2.name

                # Construct tag from csd
                tag = 'csd_%d' % csd
                input_mon.attrs['tag'] = tag
                input_mon.attrs['csd'] = csd

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
                n_uniq_bad[day] = np.sum(~input_mask_all[day, :] & np.all(input_mask_all[other_days, :], axis=0))

            good_day_flag_all *= (n_uniq_bad <= self.n_cut)

            if not np.any(good_day_flag_all):
                ValueError("Significant number of new correlator inputs flagged bad each day.")

        # Write csd flag to file
        if self.save:

            # Create container
            csd_flag = containers.SiderealDayFlag(csd=np.array([ tmap[0] for tmap in self.timemap ]))

            # Save flags to container
            csd_flag.csd_flag[:] = good_day_flag_all

            csd_flag.attrs['tag'] = 'flag_csd'

            # Write output to hdf5 file
            self._save_output(csd_flag)

        # Take the product of the input mask for all days that made threshold cut
        input_mask = np.all(input_mask_all[good_day_flag_all, :], axis=0)

        # Create a container to hold the results for the entire pass
        input_mon = containers.CorrInputMask(input=self.input_map)

        # Place the results for the entire pass in a container
        input_mon.input_mask[:] = input_mask

        input_mon.attrs['tag'] = 'for_pass'

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
        """ Set up variables that gives names to the various test
            and specify which tests will be applied.
        """

        # Gives names to the tests that will be run
        self.test = np.array(['is_chime', 'not_known_bad', 'digital_gain', 'radiometer', 'sky_fit'])
        self.ntest = len(self.test)

        # Determine what tests we will use
        self.use_test = ~np.array([False, False, self.ignore_gains, self.ignore_noise, self.ignore_fit])


    def process(self, timestream, inputmap):
        """Find good inputs using timestream.

        Parameters
        ----------
        timestream : andata.CorrData
            Apply series of tests to this timestream.
        inputmap : list of :class:`CorrInput`s
            A list of describing the inputs as they are in timestream.

        Returns
        -------
        corr_input_test : container.CorrInputTest
            Container with the results of all tests and a
            input mask that combines all tests and frequencies.
        """

        # Redistribute over the frequency direction
        timestream.redistribute('freq')

        # Extract the frequency map
        freqmap = timestream.index_map['freq'][:]

        # Find the indices for frequencies in this timestream nearest
        # to the requested test frequencies.
        if self.test_freq is None:
            freq_ind = np.arange(len(freqmap), dtype=np.int)
        else:
            freq_ind = [np.argmin(np.abs(freqmap['centre'] - freq)) for freq in self.test_freq]

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
                if 'vis_weight' in timestream.flags:
                    if not np.any(timestream.weight[fi_dist]):
                        continue

                # Run good channels code and unpack arguments
                res = data_quality.good_channels(timestream, test_freq=fi_dist, inputs=inputmap, verbose=False)
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
                print ("Frequency {} bad inputs: blank {}; gains {}{}; noise {}{}; fit {}{}".format(
                         fi_dist, timestream.ninput - len(test_inputs),
                         np.sum(good_gains == 0) if good_gains is not None else 'failed', ' [ignored]' if self.ignore_gains else '',
                         np.sum(good_noise == 0) if good_noise is not None else 'failed', ' [ignored]' if self.ignore_noise else '',
                         np.sum(good_fit == 0) if good_fit is not None else 'failed', ' [ignored]' if self.ignore_fit else ''))

        # Gather the input flags from all nodes
        passed_test_all = np.zeros((timestream.comm.size, timestream.ninput, self.ntest), dtype=np.int)
        is_test_freq_all = np.zeros(timestream.comm.size, dtype=np.bool)

        timestream.comm.Allgather(passed_test, passed_test_all)
        timestream.comm.Allgather(is_test_freq, is_test_freq_all)

        # Keep only the test frequencies
        passed_test_all = passed_test_all[is_test_freq_all, ...]
        freqmap = freqmap[is_test_freq_all]

        # Average over frequencies
        input_mask_all = (np.sum(passed_test_all, axis=0) / float(passed_test_all.shape[0])) >= self.threshold

        # Take the product along the test direction to determine good inputs for each frequency
        input_mask = np.prod(input_mask_all[:, self.use_test], axis=-1)

        # Create container to hold results
        corr_input_test = containers.CorrInputTest(freq=freqmap, test=self.test,
                                              axes_from=timestream, attrs_from=timestream)

        # Save flags to container, return container
        corr_input_test.input_mask[:] = input_mask

        corr_input_test.add_dataset('passed_test')
        corr_input_test.passed_test[:] = passed_test_all

        return corr_input_test


class AccumulateCorrInputMask(task.SingleTask):
    """ Find good correlator inputs over multiple sidereal days.

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
        """ Create empty list.  As we iterate through
            sidereal days, we will append the corr_input_mask
            from each day to this list.
        """

        self._accumulated_input_mask = []
        self._csd = []


    def process(self, corr_input_mask):
        """ Append corr_input_mask to list.

        Parameters
        ----------
        corr_input_mask : container.CorrInputMask
        """

        if not self._accumulated_input_mask:
            self.input = corr_input_mask.input[:]

        self._accumulated_input_mask.append(corr_input_mask.input_mask[:])
        self._csd.append(corr_input_mask.attrs['csd'])


    def process_finish(self):
        """ Determine good days as those where the fraction
        of good inputs is above some user specified
        threshold.  Then create accumulated input mask
        by taking the product of the input mask for all
        good days.

        Returns
        --------
        corr_input_mask : container.CorrInputMask
        """

        ninput = len(self.input)
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
                n_uniq_bad[day] = np.sum(~input_mask_all[day, :] & np.all(input_mask_all[other_days, :], axis=0))

            good_day_flag *= (n_uniq_bad <= self.n_cut)

            if not np.any(good_day_flag):
                ValueError("Significant number of new correlator inputs flagged bad each day.")

        # Write csd flag to file
        if self.save:

            # Create container
            csd_flag = containers.SiderealDayFlag(csd=np.array(self._csd))

            # Save flags to container
            csd_flag.csd_flag[:] = good_day_flag

            csd_flag.attrs['tag'] = 'flag_csd'

            # Write output to hdf5 file
            self._save_output(csd_flag)

        # Take the product of the input mask for all days that made threshold cut
        input_mask = np.all(input_mask_all[good_day_flag, :], axis=0)

        # Create container to hold results
        corr_input_mask = containers.CorrInputMask(input=self.input)

        corr_input_mask.attrs['tag'] = 'for_pass'

        # Save input_mask to container, return container
        corr_input_mask.input_mask[:] = input_mask

        return corr_input_mask


class ApplyCorrInputMask(task.SingleTask):
    """ Flag out bad correlator inputs from a timestream or sidereal stack.
    """

    def process(self, timestream, inputmask):
        """Flag out bad correlator inputs by giving them zero weight.

        Parameters
        ----------
        timestream : andata.CorrData or containers.SiderealStream

        inputmask : containers.CorrInputMask

        Returns
        -------
        timestream : andata.CorrData or containers.SiderealStream
        """

        from ch_util import andata

        # Make sure that timestream is distributed over frequency
        timestream.redistribute('freq')

        # Extract mask
        mask = inputmask.datasets['input_mask'][:]

        # Apply mask to the vis_weight array
        weight = timestream.weight[:]
        tools.apply_gain(weight, mask[np.newaxis, :, np.newaxis], out=weight)

        # Return timestream
        return timestream


class ApplySiderealDayFlag(task.SingleTask):
    """ Prevent certain sidereal days from progressing
        further in the pipeline processing (e.g.,
        exclude certain sidereal days from the sidereal stack).
    """

    def setup(self, csd_flag):
        """ Create dictionary from input .
        """

        self.csd_dict = {}
        for cc, csd in enumerate(csd_flag.csd[:]):
            self.csd_dict[csd] = csd_flag.csd_flag[cc]

    def process(self, timestream):
        """ If this sidereal day is flagged as good or
        if no flag is specified for this sidereal day,
        then return the timestream.  If this sidereal day
        is flagged as bad, then return None.

        Parameters
        ----------
        timestream : andata.CorrData / containers.SiderealStream

        Returns
        -------
        timestream : andata.CorrData / containers.SiderealStream or None
        """

        # Fetch the csd from the timestream attributes
        this_csd = timestream.attrs.get('csd')

        # Is this csd specified in the file?
        if this_csd not in self.csd_dict:

            output = timestream

            if this_csd is None:
                msg = ("Warning: input timestream does not have 'csd' attribute.  " +
                       "Will continue pipeline processing.")
            else:
                msg = (("Warning: status of CSD %d not given in %s.  " +
                        "Will continue pipeline processing.") %
                        (this_csd, self.file_name))

        else:

            # Is this csd flagged good?
            this_flag = self.csd_dict[this_csd]

            if this_flag:
                output = timestream

                msg = (("CSD %d flagged good.  " +
                        "Will continue pipeline processing.") % this_csd)
            else:
                output = None

                msg = (("CSD %d flagged bad.  " +
                        "Will halt pipeline processing.") % this_csd)

        # Print whether or not we will continue processing this csd
        if mpiutil.rank0:
            print msg

        # Return input timestream or None
        return output


class NanToNum(task.SingleTask):
    """ Finds NaN and replaces with 0.
    """

    def process(self, timestream):
        """Converts any NaN in the vis dataset and weight dataset
        to the value 0.0.

        Parameters
        ----------
        timestream : andata.CorrData or containers.SiderealStream

        Returns
        --------
        timestream : andata.CorrData or containers.SiderealStream
        """

        # Make sure we are distributed over frequency
        timestream.redistribute('freq')

        # Loop over frequencies to reduce memory usage
        for lfi, fi in timestream.vis[:].enumerate(0):

            # Set non-finite values of the visibility equal to zero
            flag = ~np.isfinite(timestream.vis[fi])
            if np.any(flag):
                print "Rank %d: %d visibilities are non-finite." % (mpiutil.rank, np.sum(flag))
                timestream.vis[fi][flag] = 0.0

            # Set non-finite values of the weight equal to zero
            flag = ~np.isfinite(timestream.weight[fi])
            if np.any(flag):
                print "Rank %d: %d visibilities are non-finite." % (mpiutil.rank, np.sum(flag))
                timestream.weight[fi][flag] = 0

        return timestream


class RadiometerWeight(task.SingleTask):
    """ Update vis_weight according to the radiometer equation:

            vis_weight_ij = Nsamples / V_ii V_jj
    """

    def process(self, timestream):
        """ Takes the input timestream.flags['vis_weight'], recasts it from uint8 to float32,
        multiplies by the total number of samples, and divides by the autocorrelations of the
        two feeds that form each baseline.

        Parameters
        ----------
        timestream : andata.CorrData

        Returns
        --------
        timestream : andata.CorrData
        """

        from calibration import _extract_diagonal as diag

        # Redistribute over the frequency direction
        timestream.redistribute('freq')

        # Extract number of samples per integration period
        max_nsamples = timestream.attrs['gpu.gpu_intergration_period'][0]

        # Extract the maximum possible value of vis_weight
        max_vis_weight = np.iinfo(timestream.flags['vis_weight'].dtype).max

        # Calculate the scaling factor that converts from vis_weight value
        # to number of samples
        vw_to_nsamp = max_nsamples / float(max_vis_weight)

        # Extract the autocorrelation
        Trec = diag(timestream.vis).real

        # Calculate the inverse autocorrelation
        with np.errstate(divide='ignore', invalid='ignore'):
            inv_Trec = np.where(Trec > 0.0, 1.0 / Trec, 0.0)

        # Determine product map for loop
        nprod = timestream.nprod
        ninput = timestream.ninput
        prod_map = [tools.icmap(pp, ninput) for pp in range(nprod)]

        vis_weight = np.zeros(timestream.flags['vis_weight'].local_shape, dtype=np.float32)

        # Loop over products to save memory
        for pp, prod in enumerate(prod_map):

            # Determine the inputs.
            ii, jj = prod

            # Scale vis_weight by input autocorrelation and effective number of samples
            vis_weight[:,pp] = inv_Trec[:,ii]*inv_Trec[:,jj]*timestream.flags['vis_weight'][:, pp]*vw_to_nsamp

        # Recast vis_weight as float32
        # Wrap to produce MPIArray
        vis_weight = mpiarray.MPIArray.wrap(vis_weight, axis=0, comm=timestream.comm)

        # Extract attributes
        vis_weight_attrs = memh5.attrs2dict(timestream.flags['vis_weight'].attrs)

        # Delete current uint8 dataset
        timestream['flags'].__delitem__('vis_weight')

        # Create new float32 dataset
        vis_weight_dataset = timestream.create_flag('vis_weight', data=vis_weight, distributed=True)

        # Copy attributes
        memh5.copyattrs(vis_weight_attrs, vis_weight_dataset.attrs)

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
        timestream : andata.CorrData or containers.SiderealStream

        Returns
        -------
        flagged_timestream : same type as timestream
        """

        # Redistribute over frequency
        timestream.redistribute('freq')

        # Determine local frequencies
        sf = timestream.vis.local_offset[0]
        ef = sf + timestream.vis.local_shape[0]

        # Extract autocorrelation indices
        auto_pi = np.array([ ii for (ii, pp) in enumerate(timestream.index_map['prod']) if pp[0] == pp[1] ])

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
        nodelist = np.array([ np.argmin(np.abs(ff - basefreq)) % 16 for ff in timestream.freq ])

        # Manually flag frequencies corresponding to specific GPU nodes
        for node in self.nodes:
            for nind in np.flatnonzero(nodelist == node):
                if nind >= sf and nind < ef:
                    timestream.weight[nind] = 0

                    print "Rank %d is flagging node %d, freq %d." % (mpiutil.rank, node, nind)

        # Manually flag frequencies corresponding to specific GPU nodes on specific acquisitions
        this_acq = timestream.attrs.get('acquisition_name', None)
        if this_acq in self.nodes_by_acq:

            # Grab list of nodes from input dictionary
            nodes_to_flag = self.nodes_by_acq[this_acq]
            if not hasattr(nodes_to_flag, '__iter__'):
                nodes_to_flag = [nodes_to_flag]

            # Loop over nodes and perform flagging
            for node in nodes_to_flag:
                for nind in np.flatnonzero(nodelist == node):
                    if nind >= sf and nind < ef:
                        timestream.weight[nind] = 0

                        print "Rank %d is flagging node %d, freq %d." % (mpiutil.rank, node, nind)

        # Return timestream with bad nodes flagged
        return timestream


def daytime_flag(time):
    """ Return a flag that indicates if the input times
    occur during the day.

    Parameters
    ----------
    time : float (UNIX time)

    Returns
    -------
    flag : np.ndarray, dtype=np.bool
    """

    flag = np.zeros(len(time), dtype=np.bool)
    rise = ephemeris.solar_rising(time[0] - 24.0*3600.0, end_time=time[-1])
    for rr in rise:
        ss = ephemeris.solar_setting(rr)[0]
        flag |= ((time >= rr) & (time <= ss))

    return flag


def solar_transit_flag(time, nsig=5.0):
    """ Return a flag that indicates if the input times
    occur near sun transit.

    Parameters
    ----------
    time : float (UNIX time)

    Returns
    -------
    flag : np.ndarray, dtype=np.bool
    """

    import ephem

    deg_to_sec = 3600.0 * ephemeris.SIDEREAL_S / 15.0

    # Create boolean flag
    flag = np.zeros(len(time), dtype=np.bool)

    # Get position of sun at every time sample
    ra, dec = [], []
    obs, sun = ephemeris._get_chime(), ephem.Sun()
    for tt in time:
        obs.date = ephemeris.unix_to_ephem_time(tt)
        sun.compute(obs)

        ra.append(sun.ra)
        dec.append(sun.dec)

    # Estimate the amount of time the sun is in the primary beam
    # as +/- nsig sigma, where sigma denotes the width of the
    # primary beam.  We use the lowest frequency and E polarisation,
    # since this is the most conservative (largest sigma).
    window_sec = nsig*cal_utils.guess_fwhm(400.0, pol='X', dec=np.median(dec), sigma=True)*deg_to_sec

    # Sun transit
    transit_times = ephemeris.solar_transit(time[0] - window_sec, time[-1] + window_sec)
    for mid in transit_times:

        # Update peak location based on rotation of cylinder
        obs.date = ephemeris.unix_to_ephem_time(mid)
        sun.compute(obs)

        peak_ra = ephemeris.peak_RA(sun, deg=True)
        mid += (peak_ra - np.degrees(sun.ra))*deg_to_sec

        # Flag +/- window_sec around peak location
        begin = mid - window_sec
        end = mid + window_sec
        flag |= ((time >= begin) & (time <= end))

    return flag


def taper_mask(mask, nwidth, outer=False):

    num = len(mask)
    if outer:
        tapered_mask = 1.0 - mask.astype(np.float)
    else:
        tapered_mask = mask.astype(np.float)

    taper = np.hanning(2*nwidth - 1)

    dmask = np.diff(tapered_mask)
    transition = np.where(dmask != 0)[0]

    for tt in transition:
        if dmask[tt] > 0:
            ind = np.arange(tt, tt+nwidth)
            tapered_mask[ind % num] *= taper[:nwidth]
        else:
            ind = np.arange(tt+2-nwidth, tt+2)
            tapered_mask[ind % num] *= taper[-nwidth:]

    if outer:
        tapered_mask = 1.0 - tapered_mask

    return tapered_mask


class DayMask(task.SingleTask):
    """Mask out the daytime data.

    Attributes
    ----------
    zero_data : bool, optional
        Zero the data in addition to modifying the noise weights
        (default is False).
    remove_average : bool, optional
        Estimate and remove the mean level from each visibilty. This estimate
        does not use data from the masked region. (default is False)
    only_sun : bool, optional
        If only_sun is True, then a window of time around sun transit is flagged as bad.
        If only_sun is False, then all day time data is flagged as bad.  (default is False)
    taper_width : float, optional
        Width (in degrees) of the taper applied to the mask.  Creates a smooth transition from
        masked to unmasked regions using a cosine function.  Default is 0.0 (no taper).
    outer_taper : bool, optional
        If outer_taper is True, then the taper occurs in the unmasked region.
        If outer_taper is False, then the taper occurs in the masked region.
        Default is True.

    """

    zero_data = config.Property(proptype=bool, default=False)
    remove_average = config.Property(proptype=bool, default=False)
    only_sun = config.Property(proptype=bool, default=False)
    taper_width = config.Property(proptype=float, default=0.0)
    outer_taper = config.Property(proptype=bool, default=True)

    def process(self, sstream):
        """Apply a day time mask.

        Parameters
        ----------
        sstream : containers.SiderealStream
            Unmasked sidereal stack.

        Returns
        -------
        mstream : containers.SiderealStream
            Masked sidereal stream.
        """

        # Determine the flagging function to use
        if self.only_sun:
            flag_function = solar_transit_flag
        else:
            flag_function = daytime_flag

        # Redistribute over frequency
        sstream.redistribute('freq')

        # Get flag that indicates day times (RAs)
        if hasattr(sstream, 'time'):
            time = sstream.time
            flag = flag_function(time)

            ntaper = int(self.taper_width / np.abs(np.median(np.diff(time))))

        else:
            csd = sstream.attrs['csd']
            ra = sstream.index_map['ra'][:]

            if hasattr(csd, '__iter__'):
                flag = np.zeros(len(ra), dtype=np.bool)
                for cc in csd:
                    flag |= flag_function(ephemeris.csd_to_unix(cc + ra/360.0))
            else:
                flag = flag_function(ephemeris.csd_to_unix(csd + ra/360.0))

            ntaper = int(self.taper_width / np.abs(np.median(np.diff(ra))))

        # If requested, estimate and subtract the mean level
        if self.remove_average and not np.all(flag):
            if mpiutil.rank0:
                print("Subtracting mean visibility.")
            sstream.vis[:] -= np.mean(sstream.vis[..., ~flag], axis=-1)[..., np.newaxis]

        # Apply the mask
        if np.any(flag):

            # If requested, apply taper.
            if ntaper > 0:
                flag = taper_mask(flag, ntaper, outer=self.outer_taper)

            # Apply the mask to the weights
            sstream.weight[:] *= (1.0 - flag)

            # If requested, apply the mask to the data
            if self.zero_data:
                sstream.vis[:] *= (1.0 - flag)

        # Return masked sidereal stream
        return sstream


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
        sstream : containers.SiderealStream
            Unmasked sidereal stack.

        Returns
        -------
        mstream : containers.SiderealStream
            Masked sidereal stream.
        """

        sstream.redistribute('freq')

        ra_shift = (sstream.ra[:] - self.start) % 360.0
        end_shift = (self.end - self.start) % 360.0

        # Crudely mask the on and off regions
        mask_bool = ra_shift > end_shift

        # Put in the transition at the start of the day
        mask = np.where(ra_shift < self.width,
                        0.5 * (1 + np.cos(np.pi * (ra_shift / self.width))),
                        mask_bool)

        # Put the transition at the end of the day
        mask = np.where(np.logical_and(ra_shift > end_shift - self.width, ra_shift <= end_shift),
                        0.5 * (1 + np.cos(np.pi * ((ra_shift - end_shift) / self.width))),
                        mask)

        if self.remove_average:
            # Estimate the mean level from unmasked data
            import scipy.stats

            nanvis = sstream.vis[:] * np.where(mask_bool, 1.0, np.nan)[np.newaxis, np.newaxis, :]
            average = scipy.stats.nanmedian(nanvis, axis=-1)[:, :, np.newaxis]
            sstream.vis[:] -= average

        # Apply the mask to the data
        if self.zero_data:
            sstream.vis[:] *= mask

        # Modify the noise weights
        sstream.weight[:] *= mask**2

        return sstream


class MaskData(task.SingleTask):
    """Mask out data ahead of map making.

    Attributes
    ----------
    auto_correlations : bool
        Exclude auto correlations if set (default=False).
    m_zero : bool
        Ignore the m=0 mode (default=False).
    positive_m : bool
        Include positive m-modes (default=True).
    negative_m : bool
        Include negative m-modes (default=True).
    """

    auto_correlations = config.Property(proptype=bool, default=False)
    m_zero = config.Property(proptype=bool, default=False)
    positive_m = config.Property(proptype=bool, default=True)
    negative_m = config.Property(proptype=bool, default=True)

    def process(self, mmodes):
        """Mask out unwanted datain the m-modes.

        Parameters
        ----------
        mmodes : containers.MModes

        Returns
        -------
        mmodes : containers.MModes
        """

        # Exclude auto correlations if set
        if not self.auto_correlations:
            for pi, (fi, fj) in enumerate(mmodes.index_map['prod']):
                if fi == fj:
                    mmodes.weight[..., pi] = 0.0

        # Apply m based masks
        if not self.m_zero:
            mmodes.weight[0] = 0.0

        if not self.positive_m:
            mmodes.weight[1:, 0] = 0.0

        if not self.negative_m:
            mmodes.weight[1:, 1] = 0.0

        return mmodes


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
        tel : :class:`ch_pipeline.core.pathfinder.CHIMEPathfinder`
            CHIME telescope class to use to get feed information.
        """
        self.telescope = tel

    def process(self, mmodes):
        """Mask out unwanted datain the m-modes.

        Parameters
        ----------
        mmodes : containers.MModes

        Returns
        -------
        mmodes : containers.MModes
        """

        tel = self.telescope

        for pi, (fi, fj) in enumerate(mmodes.index_map['prod']):

            oi, oj = tel.feeds[fi], tel.feeds[fj]

            # Check if baseline is intra-cylinder
            if not self.intra_cylinder and (oi.cyl == oj.cyl):
                mmodes.weight[..., pi] = 0.0

            # Check all the polarisation states
            is_xx = tools.is_chime_x(oi) and tools.is_chime_x(oj)
            is_yy = tools.is_chime_y(oi) and tools.is_chime_y(oj)

            if not self.xx_pol and is_xx:
                mmodes.weight[..., pi] = 0.0

            if not self.yy_pol and is_yy:
                mmodes.weight[..., pi] = 0.0

            if not self.cross_pol and not (is_xx or is_yy):
                mmodes.weight[..., pi] = 0.0

        return mmodes
