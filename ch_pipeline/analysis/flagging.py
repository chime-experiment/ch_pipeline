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
    BadNodeFlagger
    DayMask
    SunClean
    MaskData
    MaskCHIMEData
"""
import os.path
import numpy as np

from caput import mpiutil, mpiarray, memh5, config, pipeline
from ch_util import rfi, data_quality, tools

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


class MonitorCorrInputs(task.SingleTask):
    """ Monitor good correlator inputs over several sidereal days.

    Parameters
    ----------
    threshold : float
        Flag a sidereal day as bad if the fraction of correlator
        inputs powered ON that pass the test is less than threshold.
    """

    threshold = config.Property(proptype=float, default=0.7)

    def setup(self, files):
        """Divide list of files up into sidereal days.

        Parameters
        ----------
        files : list
            List of filenames to monitor good correlator inputs.
        """

        from sidereal import get_times, _days_in_csd
        from ch_util import ephemeris, andata

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
            timemap = [ (day, ephemeris.CSD_ZERO + 24.0 * 3600.0 * ephemeris.SIDEREAL_S * np.array([day, day+1]))
                         for day in days ]

            # Extract the frequency and inputs for the first day
            data_r = andata.Reader(self.files[filemap[0][1]])
            input_map = data_r.input
            freq = data_r.freq['centre']

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

                if len(freq) != nfreq:
                    ValueError("Differing number of frequencies for csd %d and csd %d." % (fmap[0], filemap[0][0]))
                elif np.sum(data_r.freq['centre'] != freq) > 0:
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
            days when the fraction of good correlator inputs is less than
            'threshold'.  Note that this is not output to the pipeline.
            It is ancillary data product that is saved when one sets the
            'save' parameter in the configuration file.
        input_monitor_all : containers.CorrInputMonitor
            Contains the correlator input mask and frequency mask
            obtained from taking AND of the masks from the individual
            sidereal days.
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
        input_powered = np.ones((n_local, self.ninput), dtype=np.bool)
        freq_mask = np.ones((n_local, self.nfreq), dtype=np.bool)
        freq_powered = np.ones((n_local, self.nfreq), dtype=np.bool)

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
            except RuntimeError:
                # No sources available for this csd
                continue

            # Accumulate flags over multiple days
            input_mask[i_local, :] = cm.good_ipts
            freq_mask[i_local, :] = cm.good_freqs
            input_powered[i_local, :] = cm.pwds
            freq_powered[i_local, :] = cm.gpu_node_flag
            good_day_flag[i_local] = True

            # If requested, write to disk
            if self.save:

                # Create a container to hold the results
                input_mon = containers.CorrInputMonitor(freq=cm.freqs, input=cm.input_map,
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
        input_powered_all = np.zeros((self.ndays, self.ninput), dtype=np.bool)
        freq_mask_all = np.zeros((self.ndays, self.nfreq), dtype=np.bool)
        freq_powered_all = np.zeros((self.ndays, self.nfreq), dtype=np.bool)
        good_day_flag_all = np.zeros(self.ndays, dtype=np.bool)

        mpiutil.world.Allgather(input_mask, input_mask_all)
        mpiutil.world.Allgather(input_powered, input_powered_all)
        mpiutil.world.Allgather(freq_mask, freq_mask_all)
        mpiutil.world.Allgather(freq_powered, freq_powered_all)
        mpiutil.world.Allgather(good_day_flag, good_day_flag_all)

        # Look for bad days:
        # Calculate the fraction of inputs that are good each day
        n_good_input = np.sum(input_mask_all, axis=-1)
        n_powered_input = np.sum(input_powered_all, axis=-1)

        # Find days where the number of good correlator inputs is greater
        # than some user specified fraction of the number of
        # correlator inputs powered on
        good_day_flag_all *= (n_good_input > self.threshold*n_powered_input)

        if not np.any(good_day_flag_all):
            ValueError("More than %d%% of powered ON inputs flagged bad every day." % 100.0*(1.0 - self.threshold))

        # Write csd flag to file
        if self.save:

            # Create container
            csd_flag = containers.SiderealDayFlag(input=self.input_map, csd=np.array([ tmap[0] for tmap in self.timemap ]))

            # Save flags to container
            csd_flag.csd_flag[:] = good_day_flag_all

            csd_flag.add_dataset('input_mask')
            csd_flag.input_mask[:] = input_mask_all

            csd_flag.attrs['tag'] = 'flag_csd'

            # Write output to hdf5 file
            self._save_output(csd_flag)

        # Take the product of the input mask for all days that made threshold cut
        input_mask = np.all(input_mask_all[good_day_flag_all, :], axis=0)
        input_powered = np.all(input_powered_all[good_day_flag_all, :], axis=0)
        freq_mask = np.all(freq_mask_all[good_day_flag_all, :], axis=0)
        freq_powered = np.all(freq_powered_all[good_day_flag_all, :], axis=0)

        # Create a container to hold the results for the entire pass
        input_mon = containers.CorrInputMonitor(freq=self.freq, input=self.input_map)

        # Place the results for the entire pass in a container
        input_mon.input_mask[:] = input_mask
        input_mon.input_powered[:] = input_powered
        input_mon.freq_mask[:] = freq_mask
        input_mon.freq_powered[:] = freq_powered

        input_mon.attrs['tag'] = 'for_pass'

        # Ensure we stop on next iteration
        self.ndays = 0
        self.timemap = None

        # Return pass results
        return input_mon


class FindGoodCorrInputs(task.SingleTask):
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


class AccumulateGoodCorrInputs(task.SingleTask):
    """ Find good correlator inputs over multiple sidereal days.
        Also determine bad days as those with a lack of good
        correlator inputs.
    """

    threshold = config.Property(proptype=float, default=0.7)

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

        input_mask_all = np.asarray(self._accumulated_input_mask)

        # Calculate the fraction of inputs that are good each day
        frac_good_input = np.sum(input_mask_all, axis=-1) / float(ninput)

        # Find days where the fraction of good inputs
        # is greater than the user specified threshold
        good_day_flag = frac_good_input > self.threshold

        if not np.any(good_day_flag):
            ValueError("More than %d%% of inputs flagged bad every day." % 100.0*self.threshold)

        # Write csd flag to file
        if self.save:

            # Create container
            csd_flag = containers.SiderealDayFlag(input=self.input, csd=np.array(self._csd))

            # Save flags to container
            csd_flag.csd_flag[:] = good_day_flag

            csd_flag.add_dataset('input_mask')
            csd_flag.input_mask[:] = input_mask_all

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

    Attributes
    ----------
    file_name : str
        Path to the hdf5 file that contains the correlator input mask.
    """

    file_name = config.Property(proptype=str)

    def setup(self):
        """ Load correlator input mask from disk.
        """

        self.file_name = os.path.expandvars(os.path.expanduser(self.file_name))

        corr_input_mask = containers.CorrInputMask.from_file(self.file_name, distributed=False)

        self.input_mask = corr_input_mask.datasets['input_mask'][:]

    def process(self, timestream):
        """Flag out bad correlator inputs by giving them zero weight.

        Parameters
        ----------
        timestream : andata.CorrData or containers.SiderealStream

        Returns
        -------
        timestream : andata.CorrData or containers.SiderealStream
        """

        # Make sure that timestream is distributed over frequency
        timestream.redistribute('freq')

        # Apply mask to the vis_weight array
        weight = timestream.weight[:]
        tools.apply_gain(weight, self.input_mask[np.newaxis, :, np.newaxis], out=weight)

        # Add flag dataset
        flag_dataset = timestream.create_flag('input', data=self.input_mask, distributed=False)
        flag_dataset.attrs['axis'] = ('input', )

        # Return timestream
        return timestream


class ApplySiderealDayFlag(task.SingleTask):
    """ Prevent certain sidereal days from progressing
        further in the pipeline processing (e.g.,
        exclude certain sidereal days from the sidereal stack).

    Attributes
    ----------
    file_name : str
        Path to the hdf5 file that contains the sidereal day flag.
    """

    file_name = config.Property(proptype=str)

    def setup(self):
        """ Load sidereal day flag from disk.
        """

        self.file_name = os.path.expandvars(os.path.expanduser(self.file_name))

        csd_flag = containers.SiderealDayFlag.from_file(self.file_name, distributed=False)

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
        timestream : andata.CorrData or containers.SiderealStream

        Returns
        -------
        timestream : andata.CorrData or containers.SiderealStream
        """

        # Is this csd in the specified file?
        this_csd = timestream.attrs.get('csd')

        if this_csd not in self.csd_dict:

            if mpiutil.rank0:
                if this_csd is None:
                    print("Warning: input timestream does not have 'csd' attribute.  " +
                           "Continue pipeline processing.")
                else:
                    print(("Warning: status of CSD %d not given in %s.  " +
                           "Continue pipeline processing.") %
                           (this_csd, self.file_name))

            return timestream

        else:

            this_flag = self.csd_dict[this_csd]

            if this_flag:

                if mpiutil.rank0:
                    print(("CSD %d flagged good.  " +
                           "Continue pipeline processing.") % this_csd)

                return timestream

            else:

                if mpiutil.rank0:
                    print(("CSD %d flagged bad.  " +
                           "Halt pipeline processing.") % this_csd)

                return None



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
    flag_freq_zero : boolean, optional
        Whether to flag out frequency zero.
    """

    nodes = config.Property(proptype=list, default=[])

    flag_freq_zero = config.Property(proptype=bool, default=True)

    def process(self, timestream):
        """Flag out bad nodes by giving them zero weight.

        Parameters
        ----------
        timestream : andata.CorrData or containers.SiderealStream

        Returns
        -------
        flagged_timestream : same type as timestream
        """

        # Redistribute over time
        timestream.redistribute(['time', 'gated_time0'])

        # Extract autocorrelation indices
        auto_pi, _ = data_quality._get_autos_index(timestream.index_map['prod'])

        good_freq_flag = np.any(timestream.vis[:, auto_pi, :].real > 0.0, axis=1)
        timestream.weight[:] *= good_freq_flag[:, np.newaxis, :]

        # If requested, flag the first frequency
        if self.flag_freq_zero:
            timestream.weight[0] = 0

        # Manually flag frequencies corresponding to specific GPU nodes
        for node in self.nodes:
            if node < 0 or node >= 16:
                raise RuntimeError('Node index (%i) is invalid (should be 0-15).' % node)

            timestream.weight[node::16] = 0

        # Redistribute over frequency
        timestream.redistribute('freq')

        return timestream


class DayMask(task.SingleTask):
    """Crudely simulate a masking out of the daytime data.

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


class SunClean(task.SingleTask):
    """Clean the sun from data by projecting out signal from its location.

    Optionally flag out all data around transit, and sunrise/sunset.

    Attributes
    ----------
    flag_time : float, optional
        Flag out time around sun rise/transit/set. Should be set in degrees. If
        :obj:`None` (default), then don't flag at all.
    """

    flag_time = config.Property(proptype=float, default=None)

    def setup(self, inputmap):
        self.inputmap = inputmap

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

        inputmap = self.inputmap

        from ch_util import ephemeris
        import ephem

        sstream.redistribute('freq')

        def ra_dec_of(body, time):
            obs = ephemeris._get_chime()
            obs.date = ephemeris.unix_to_ephem_time(time)

            body.compute(obs)

            return body.ra, body.dec, body.alt

        # Get array of CSDs for each sample
        ra = sstream.index_map['ra'][:]
        csd = sstream.attrs['csd'] + ra / 360.0

        # Get position of sun at every time sample
        times = ephemeris.csd_to_unix(csd)
        sun_pos = np.array([ra_dec_of(ephem.Sun(), t) for t in times])

        # Get hour angle and dec of sun, in radians
        ha = 2 * np.pi * (ra / 360.0) - sun_pos[:, 0]
        dec = sun_pos[:, 1]
        el = sun_pos[:, 2]

        # Construct lengths for each visibility and determine what polarisation combination they are
        feed_pos = tools.get_feed_positions(inputmap)
        vis_pos = np.array([ feed_pos[ii] - feed_pos[ij] for ii, ij in sstream.index_map['prod'][:]])

        feed_list = [ (inputmap[fi], inputmap[fj]) for fi, fj in sstream.index_map['prod'][:]]
        pol_ind = np.array([ 2 * tools.is_chime_y(fi) + tools.is_chime_y(fj) for fi, fj in feed_list])

        # Initialise new container
        sscut = sstream.__class__(axes_from=sstream, attrs_from=sstream)
        sscut.redistribute('freq')

        wv = 3e2 / sstream.index_map['freq']['centre']

        # Iterate over frequencies and polarisations to null out the sun
        for lfi, fi in sstream.vis[:].enumerate(0):

            # Get the baselines in wavelengths
            u = vis_pos[:, 0] / wv[fi]
            v = vis_pos[:, 1] / wv[fi]

            # Calculate the phase that the sun would have using the fringestop routine
            fsphase = tools.fringestop_phase(ha[np.newaxis, :], np.radians(ephemeris.CHIMELATITUDE),
                                             dec[np.newaxis, :], u[:, np.newaxis], v[:, np.newaxis])

            # Calculate the visibility vector for the sun
            sun_vis = fsphase.conj() * (el > 0.0)

            # Mask out the auto-correlations
            sun_vis *= np.logical_or(u != 0.0, v != 0.0)[:, np.newaxis]

            # Copy over the visiblities and weights
            vis = sstream.vis[fi]
            weight = sstream.weight[fi]
            sscut.vis[fi] = vis
            sscut.weight[fi] = weight

            # Iterate over polarisations to do projection independently for each.
            # This is needed because of the different beams for each pol.
            for pol in range(4):

                # Mask out other polarisations in the visibility vector
                sun_vis_pol = sun_vis * (pol_ind == pol)[:, np.newaxis]

                # Calculate various projections
                vds = (vis * sun_vis_pol.conj() * weight).sum(axis=0)
                sds = (sun_vis_pol * sun_vis_pol.conj() * weight).sum(axis=0)
                isds = tools.invert_no_zero(sds)

                # Subtract sun contribution from visibilities and place in new array
                sscut.vis[fi] -= sun_vis_pol * vds * isds

        # If needed mask out the regions around sun rise, set and transit
        if self.flag_time is not None:

            # Find the RAs of each event
            transit_ra = ephemeris.transit_RA(ephemeris.solar_transit(times[0], times[-1]))
            rise_ra = ephemeris.transit_RA(ephemeris.solar_rising(times[0], times[-1]))
            set_ra = ephemeris.transit_RA(ephemeris.solar_setting(times[0], times[-1]))

            # Construct a mask for each
            rise_mask = ((ra - rise_ra) % 360.0) > self.flag_time
            set_mask = ((ra - set_ra + self.flag_time) % 360.0) > self.flag_time
            transit_mask = ((ra - transit_ra + self.flag_time / 2) % 360.0) > self.flag_time

            # Combine the masks and apply to data
            mask = np.logical_and(rise_mask, np.logical_and(set_mask, transit_mask))
            sscut.weight[:] *= mask

        return sscut


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
