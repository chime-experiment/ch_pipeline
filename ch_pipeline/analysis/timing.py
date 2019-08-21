"""Tasks for constructing timing corrections and applying them to data.

.. currentmodule:: ch_pipeline.analysis.timing

Tasks
=====

.. autosummary::
    :toctree: generated/

    ApplyTimingCorrection
    ConstructTimingCorrection
"""
# === Start Python 2/3 compatibility
from __future__ import (absolute_import, division,
                        print_function, unicode_literals)
from future.builtins import *  # noqa  pylint: disable=W0401, W0614
from future.builtins.disabled import *  # noqa  pylint: disable=W0401, W0614
# === End Python 2/3 compatibility


import os

import numpy as np

from ch_util import timing, ephemeris
from caput import config, pipeline
from draco.core import task


class ApplyTimingCorrection(task.SingleTask):
    """Apply a timing correction to the visibilities.

    Parameters
    ----------
    use_input_flags : bool
        Account for bad inputs when determining a stacked timing correction.
        Takes roughly twice as long.
    refer_to_transit : bool
        Reference the timing correction to the time of source transit.
        Useful for applying the timing correction to holography data that has
        already been calibrated to have zero phase at transit.
    """

    use_input_flags = config.Property(proptype=bool, default=False)
    refer_to_transit = config.Property(proptype=bool, default=False)

    def setup(self, tcorr):
        """Set the timing correction to use.

        Parameters
        ----------
        tcorr : ch_util.timing.TimingCorrection or list of
            Timing correction that is relevant for the range of time being processed.
            Note that the timing correction must *already* be referenced with respect
            to the times that were used to calibrate if processing stacked data.
        """
        if not isinstance(tcorr, list):
            tcorr = [tcorr]
        self.tcorr = tcorr

    def process(self, tstream):
        """Apply the timing correction to the input timestream.

        Parameters
        ----------
        tstream : andata.CorrData, containers.TimeStream, or containers.SiderealStream
            Apply the timing correction to the visibilities stored in this container.

        Returns
        -------
        tstream_corr : same as `tstream`
            Timestream with corrected visibilities.
        """
        # Determine times
        if 'time' in tstream.index_map:
            timestamp = tstream.time
        else:
            csd = tstream.attrs['lsd'] if 'lsd' in tstream.attrs else tstream.attrs['csd']
            timestamp = ephemeris.csd_to_unix(csd + tstream.ra / 360.0)

        # Extract local frequencies
        tstream.redistribute('freq')

        nfreq = tstream.vis.local_shape[0]
        sfreq = tstream.vis.local_offset[0]
        efreq = sfreq + nfreq

        freq = tstream.freq[sfreq:efreq]

        # If requested, extract the input flags
        input_flags = tstream.input_flags[:] if self.use_input_flags else None

        # Find the right timing correction
        for tcorr in self.tcorr:
            if timestamp[0] > tcorr.time[0] and timestamp[-1] < tcorr.time[-1]:
                break
        else:
            raise RuntimeError(
                "Could not find timing correction file covering "
                "range of timestream data (%s to %s)" %
                tuple(ephemeris.unix_to_datetime([timestamp[0], timestamp[-1]]))
            )

        self.log.info("Using correction file %s" % tcorr.attrs["tag"])

        # If requested, reference the timing correct with respect to source transit time
        if self.refer_to_transit:
            # First check for transit_time attribute in file
            ttrans = tstream.attrs.get('transit_time', None)
            if ttrans is None:
                source = tstream.attrs['source_name']
                ttrans = ephemeris.transit_times(ephemeris.source_dictionary[source],
                                                 tstream.time[0], tstream.time[-1])
                if ttrans.size != 1:
                    raise RuntimeError("Found %d transits of %s in timestream.  "
                                       "Require single transit." % (ttrans.size, source))
                else:
                    ttrans = ttrans[0]

            self.log.info("Referencing timing correction to %s (RA=%0.1f deg)." %
                          (ephemeris.unix_to_datetime(ttrans).strftime("%Y%m%dT%H%M%SZ"),
                           ephemeris.lsa(ttrans)))

            tcorr.set_global_reference_time(ttrans, interpolate=True, interp='linear')

        # Apply the timing correction
        tcorr.apply_timing_correction(tstream, time=timestamp, freq=freq,
                                      input_flags=input_flags, copy=False)

        return tstream


class ConstructTimingCorrection(task.SingleTask):
    """Generate a timing correction from the cross correlation of noise source inputs.

    Parameters
    ----------
    check_amp: bool
        Do not include frequencies and times where the
        square root of the autocorrelations is an outlier.
    check_sig: bool
        Do not include frequencies and times where the
        square root of the inverse weight is an outlier.
    nsigma: float
        Number of median absolute deviations to consider
        a data point an outlier in the checks specified above.
    threshold: float
        A (frequency, input) must pass the checks specified above
        more than this fraction of the time,  otherwise it will be
        flaged as bad for all times.
    nparam: int
        Number of parameters for polynomial fit to the
        time averaged phase versus frequency.
    min_freq: float
        Minimum frequency in MHz to include in the fit.
    max_freq: float
        Maximum frequency in MHz to include in the fit.
    max_iter_weight: int
        The weight for each frequency is estimated from the variance of the
        residuals of the template fit from the previous iteration.  This
        is the total number of times to iterate.  Setting to 0 corresponds
        to linear least squares.
    input_sel : list
        Generate the timing correction from inputs with these chan_id's.
    output_suffix: str
        The suffix to append to the end of the name of the output files.
    """

    check_amp = config.Property(proptype=bool, default=True)
    check_sig = config.Property(proptype=bool, default=True)
    nsigma = config.Property(proptype=float, default=5.0)
    threshold = config.Property(proptype=float, default=0.5)
    nparam = config.Property(proptype=int, default=2)
    min_freq = config.Property(proptype=float, default=420.0)
    max_freq = config.Property(proptype=float, default=780.0)
    max_iter_weight = config.Property(proptype=int, default=2)
    input_sel = config.Property(proptype=(lambda val: val if val is None else list(val)),
                                default=None)
    output_suffix = config.Property(proptype=str, default='chimetiming_delay')

    _parameters = ['check_amp', 'check_sig', 'nsigma', 'threshold', 'nparam',
                   'min_freq', 'max_freq', 'max_iter_weight', 'input_sel']

    _datasets_fixed_for_acq = ['static_phi', 'weight_static_phi', 'static_phi_fit',
                               'static_amp', 'weight_static_amp']

    def setup(self):
        """Get ready to generate timing corrections."""
        self.kwargs = {}
        for param in self._parameters:
            self.kwargs[param] = getattr(self, param)

        self.current_acq = None

    def process(self, filelist):
        """Generate timing correction from an input list of files.

        Parameters
        ----------
        filelist : list of files

        Returns
        -------
        tcorr : ch_util.timing.TimingCorrection
            Timing correction derived from noise source data.
        """
        # Determine the acquisition
        new_acq = np.unique([os.path.basename(os.path.dirname(ff)) for ff in filelist])
        if new_acq.size > 1:
            raise RuntimeError("Cannot process multiple acquisitions.  Received %d." %
                               new_acq.size)
        else:
            new_acq = new_acq[0]

        # If this is a new acquisition, then ensure the
        # static phase and amplitude are recalculated
        if new_acq != self.current_acq:
            self.current_acq = new_acq

            for key in self._datasets_fixed_for_acq:
                self.kwargs[key] = None

        # Process the chimetiming data
        self.log.info("Processing %d files from %s." % (len(filelist), self.current_acq))

        tcorr = timing.TimingData.from_acq_h5(filelist,
                                              only_correction=True,
                                              distributed=self.comm.size > 1,
                                              comm=self.comm,
                                              **self.kwargs)

        # Save the static phase and amplitude to be used on subsequent iterations
        # within this acquisition
        for key in self._datasets_fixed_for_acq:
            if key in tcorr.datasets:
                self.kwargs[key] = tcorr.datasets[key][:]
            elif key in tcorr.flags:
                self.kwargs[key] = tcorr.flags[key][:]
            else:
                msg = "Dataset %s could not be found in timing correction object." % key
                raise RuntimeError(msg)

        # Create a tag indicating the range of time processed
        tfmt = "%Y%m%dT%H%M%SZ"
        start_time = ephemeris.unix_to_datetime(tcorr.time[0]).strftime(tfmt)
        end_time = ephemeris.unix_to_datetime(tcorr.time[-1]).strftime(tfmt)
        tag = [start_time, 'to', end_time]
        if self.output_suffix:
            tag.append(self.output_suffix)

        tcorr.attrs['tag'] = '_'.join(tag)

        # Return timing correction
        return tcorr
