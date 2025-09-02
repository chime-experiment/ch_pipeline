"""Tasks for constructing timing corrections and applying them to data."""

import os
from typing import ClassVar

import caput.time as ctime
import numpy as np
from caput import config
from caput.pipeline import PipelineRuntimeError
from ch_ephem import sources
from ch_ephem.observers import chime
from ch_util import timing

# For querying DataFlag database
from chimedb import dataflag as df
from chimedb.core import connect as connect_database
from draco.core import task


class ApplyTimingCorrection(task.SingleTask):
    """Apply a timing correction to the visibilities.

    Only runs on days flagged by needs_timing_correction.

    Parameters
    ----------
    use_input_flags : bool
        Account for bad inputs when determining a stacked timing correction.
        Takes roughly twice as long.
    refer_to_transit : bool
        Reference the timing correction to the time of source transit.
        Useful for applying the timing correction to holography data that has
        already been calibrated to have zero phase at transit.
    transit_window: float
        Reference the timing correction to the median value over a window (in seconds)
        around the time of source transit.
    pass_if_missing: bool (default False)
        If a timing correction cannot be found for the time range do not raise an exception
        and just pass the data on.
    """

    use_input_flags = config.Property(proptype=bool, default=False)
    refer_to_transit = config.Property(proptype=bool, default=False)
    transit_window = config.Property(proptype=float, default=0.0)
    pass_if_missing = config.Property(proptype=bool, default=False)

    def setup(self, tcorr):
        """Set the timing correction to use.

        Parameters
        ----------
        tcorr : ch_util.timing.TimingCorrection or list of
            Timing correction that is relevant for the range of time being processed.
            Note that the timing correction must *already* be referenced with respect
            to the times that were used to calibrate if processing stacked data.
        """
        flags = None

        # Query flag database for date ranges that need timing correction
        if self.comm.rank == 0:
            connect_database()

            needs_timing_correction_type = df.DataFlagType.get(
                name="needs_timing_correction"
            )

            if needs_timing_correction_type is None:
                raise RuntimeError(
                    "Could not find data flag type `needs_timing_correction`"
                )

            # We are extracting the time ranges here from the DataFlag objects
            # to avoid potential issues with non-0 ranks accidentally doing database queries later on
            flags = list(
                df.DataFlag.select(df.DataFlag.start_time, df.DataFlag.finish_time)
                .where(df.DataFlag.type == needs_timing_correction_type)
                .dicts()
            )

            self.log.debug(
                f"Queried database and received {len(flags)} needs_timing_correction flags."
            )

        # Share timing correction flags with other nodes
        # Save flags to class attribute
        self.flags = self.comm.bcast(flags, root=0)

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
        if "time" in tstream.index_map:
            timestamp = tstream.time
        else:
            csd = (
                tstream.attrs["lsd"] if "lsd" in tstream.attrs else tstream.attrs["csd"]
            )
            timestamp = chime.lsd_to_unix(csd + tstream.ra / 360.0)

        # Extract local frequencies
        tstream.redistribute("freq")

        nfreq = tstream.vis.local_shape[0]
        sfreq = tstream.vis.local_offset[0]
        efreq = sfreq + nfreq

        freq = tstream.freq[sfreq:efreq]

        # If requested, extract the input flags
        input_flags = tstream.input_flags[:] if self.use_input_flags else None

        # Check needs_timing_correction flags time ranges to see if input timestream
        # falls within the interval
        needs_timing_correction = False
        for flag in self.flags:
            if (
                timestamp[0] >= flag["start_time"]
                and timestamp[0] <= flag["finish_time"]
            ):
                if timestamp[-1] >= flag["finish_time"]:
                    raise PipelineRuntimeError(
                        f"Data covering {timestamp[0]} to {timestamp[-1]} partially overlaps "
                        f"needs_timing_correction DataFlag covering {ctime.unix_to_datetime(flag['start_time']).strftime('%Y%m%dT%H%M%SZ')} "
                        f"to {ctime.unix_to_datetime(flag['finish_time']).strftime('%Y%m%dT%H%M%SZ')}."
                    )

                self.log.info(
                    f"Data covering {timestamp[0]} to {timestamp[-1]} flagged by "
                    f"needs_timing_correction DataFlag covering {ctime.unix_to_datetime(flag['start_time']).strftime('%Y%m%dT%H%M%SZ')} "
                    f"to {ctime.unix_to_datetime(flag['finish_time']).strftime('%Y%m%dT%H%M%SZ')}. Timing correction will be applied."
                )
                needs_timing_correction = True
                break

        if not needs_timing_correction:
            self.log.info(
                f"Data in span {ctime.unix_to_datetime(timestamp[0]).strftime('%Y%m%dT%H%M%SZ')} to {ctime.unix_to_datetime(timestamp[-1]).strftime('%Y%m%dT%H%M%SZ')} does not need timing correction"
            )
            return tstream

        # Find the right timing correction
        for tcorr in self.tcorr:
            if timestamp[0] >= tcorr.time[0] and timestamp[-1] <= tcorr.time[-1]:
                break
        else:
            ts_ = tuple(ctime.unix_to_datetime([timestamp[0], timestamp[-1]]))
            msg = (
                "Could not find timing correction file covering "
                f"range of timestream data ({ts_[0]} to {ts_[1]})."
            )

            if self.pass_if_missing:
                self.log.warning(msg + " Doing nothing.")
                return tstream

            raise RuntimeError(msg)

        self.log.info(f"Using correction file {tcorr.attrs['tag']}")

        # If requested, reference the timing correct with respect to source transit time
        if self.refer_to_transit:
            # First check for transit_time attribute in file
            ttrans = tstream.attrs.get("transit_time", None)
            if ttrans is None:
                source = tstream.attrs["source_name"]
                ttrans = chime.transit_times(
                    sources.source_dictionary[source],
                    tstream.time[0],
                    tstream.time[-1],
                )
                if ttrans.size != 1:
                    raise RuntimeError(
                        f"Found {ttrans.size:d} transits of {source} in timestream.  "
                        "Require single transit."
                    )

                ttrans = ttrans[0]

            self.log.info(
                "Referencing timing correction to "
                f"{ctime.unix_to_datetime(ttrans).strftime('%Y%m%dT%H%M%SZ')} "
                f"(RA={chime.unix_to_lsa(ttrans):.1f} deg)."
            )

            tcorr.set_global_reference_time(
                ttrans, window=self.transit_window, interpolate=True, interp="linear"
            )

        # Apply the timing correction
        tcorr.apply_timing_correction(
            tstream, time=timestamp, freq=freq, input_flags=input_flags, copy=False
        )

        return tstream


class ConstructTimingCorrection(task.SingleTask):
    """Generate a timing correction from the cross correlation of noise source inputs.

    Parameters
    ----------
    min_frac_kept: float
        Do not include frequencies and times where the fraction
        of data that remains is less than this threshold.
    threshold: float
        A (frequency, input) must pass the checks specified above
        more than this fraction of the time,  otherwise it will be
        flaged as bad for all times.
    min_freq: float
        Minimum frequency in MHz to include in the fit.
    max_freq: float
        Maximum frequency in MHz to include in the fit.
    mask_rfi: bool
        Mask frequencies that occur within known RFI bands.  Note that the
        noise source data does not contain RFI, however the real-time pipeline
        does not distinguish between noise source inputs and sky inputs, and as
        a result will discard large amounts of data in these bands.
    max_iter_weight: int
        The weight for each frequency is estimated from the variance of the
        residuals of the template fit from the previous iteration.  Outliers
        are also flagged at each iteration with an increasingly aggresive threshold.
        This is the total number of times to iterate.  Setting to 1 corresponds
        to linear least squares.  Default is 1, unless check_amp or check_phi is True,
        in which case this defaults to the maximum number of thresholds provided.
    check_amp: bool
        Do not fit frequencies and times where the residual amplitude is an outlier.
    nsigma_amp: list of float
        If check_amp is True, then residuals greater than this number of sigma
        will be considered an outlier.  Provide a list containing the value to be used
        at each iteration.  If the length of the list is less than max_iter_weight,
        then the last value in the list will be repeated for the remaining iterations.
    check_phi: bool
        Do not fit frequencies and times where the residual phase is an outlier.
    nsigma_phi: list of float
        If check_phi is True, then residuals greater than this number of sigma
        will be considered an outlier.  Provide a list containing the value to be used
        at each iteration.  If the length of the list is less than max_iter_weight,
        then the last value in the list will be repeated for the remaining iterations.
    input_sel : list
        Generate the timing correction from inputs with these chan_id's.
    output_suffix: str
        The suffix to append to the end of the name of the output files.
    """

    min_frac_kept = config.Property(proptype=float, default=0.85)
    threshold = config.Property(proptype=float, default=0.5)
    min_freq = config.Property(proptype=float, default=420.0)
    max_freq = config.Property(proptype=float, default=600.0)
    mask_rfi = config.Property(proptype=bool, default=True)
    max_iter_weight = config.Property(
        proptype=(lambda val: val if val is None else int(val)), default=None
    )
    check_amp = config.Property(proptype=bool, default=False)
    nsigma_amp = config.Property(
        default=[1000.0, 500.0, 200.0, 100.0, 50.0, 20.0, 10.0, 5.0]
    )
    check_phi = config.Property(proptype=bool, default=True)
    nsigma_phi = config.Property(
        default=[1000.0, 500.0, 200.0, 100.0, 50.0, 20.0, 10.0, 5.0]
    )
    nparam = config.Property(proptype=int, default=2)
    input_sel = config.Property(
        proptype=(lambda val: val if val is None else list(val)), default=None
    )
    output_suffix = config.Property(proptype=str, default="chimetiming_delay")

    _parameters: ClassVar[str] = [
        "min_frac_kept",
        "threshold",
        "min_freq",
        "max_freq",
        "mask_rfi",
        "max_iter_weight",
        "check_amp",
        "nsigma_amp",
        "check_phi",
        "nsigma_phi",
        "nparam",
        "input_sel",
    ]

    _datasets_fixed_for_acq: ClassVar[str] = [
        "static_phi",
        "weight_static_phi",
        "static_phi_fit",
        "static_amp",
        "weight_static_amp",
    ]

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
            Path to file(s) containing the timing data.

        Returns
        -------
        tcorr : ch_util.timing.TimingCorrection
            Timing correction derived from noise source data.
        """
        # Determine the acquisition
        new_acq = np.unique([os.path.basename(os.path.dirname(ff)) for ff in filelist])
        if new_acq.size > 1:
            raise RuntimeError(
                f"Cannot process multiple acquisitions.  Received {new_acq.size:d}."
            )

        new_acq = new_acq[0]

        # If this is a new acquisition, then ensure the
        # static phase and amplitude are recalculated
        if new_acq != self.current_acq:
            self.current_acq = new_acq

            for key in self._datasets_fixed_for_acq:
                self.kwargs[key] = None

        # Process the chimetiming data
        self.log.info(f"Processing {len(filelist):d} files from {self.current_acq}.")

        tcorr = timing.TimingData.from_acq_h5(
            filelist,
            only_correction=True,
            distributed=self.comm.size > 1,
            comm=self.comm,
            **self.kwargs,
        )

        self.log.info(
            f"Finished processing {len(filelist):d} files from {self.current_acq}."
        )

        # Save the static phase and amplitude to be used on subsequent iterations
        # within this acquisition
        for key in self._datasets_fixed_for_acq:
            if key in tcorr.datasets:
                self.kwargs[key] = tcorr.datasets[key][:]
            elif key in tcorr.flags:
                self.kwargs[key] = tcorr.flags[key][:]
            else:
                raise RuntimeError(
                    f"Dataset {key} could not be found in timing correction object."
                )

        # Save the names of the files used to construct the correction
        tcorr.attrs["archive_files"] = np.array(filelist)

        # Create a tag indicating the range of time processed
        tfmt = "%Y%m%dT%H%M%SZ"
        start_time = ctime.unix_to_datetime(tcorr.time[0]).strftime(tfmt)
        end_time = ctime.unix_to_datetime(tcorr.time[-1]).strftime(tfmt)
        tag = [start_time, "to", end_time]
        if self.output_suffix:
            tag.append(self.output_suffix)

        tcorr.attrs["tag"] = "_".join(tag)

        # Return timing correction
        return tcorr
