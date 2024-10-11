"""Tasks for sidereal regridding.

Tasks for taking the timestream data and regridding it into sidereal days
which can be stacked.

Usage
=====

Generally you would want to use these tasks together. Starting with a
:class:`LoadTimeStreamSidereal`, then feeding that into
:class:`SiderealRegridder` to grid onto each sidereal day, and then into
:class:`draco.analysis.SiderealStacker` if you want to combine the different days.
"""

import gc
from collections import Counter
from typing import ClassVar

import caput.time as ctime
import numpy as np
from caput import config, pipeline, tod
from caput.weighted_median import weighted_median
from ch_ephem import sources
from ch_ephem.observers import chime
from ch_util import andata, tools
from draco.analysis import sidereal
from draco.core import containers, task
from mpi4py import MPI


class LoadTimeStreamSidereal(task.SingleTask):
    """Load data in sidereal days.

    This task takes an input list of data, and loads in a sidereal day at a
    time, and passes it on.

    .. deprecated:: pass1
        The preferred option now is to load a whole range of files one at a time
        and feed them into the :class:`SiderealGrouper`.

    Attributes
    ----------
    padding : float
        Extra amount of a sidereal day to pad each timestream by. Useful for
        getting rid of interpolation artifacts.
    freq_physical : list
        List of physical frequencies in MHz.
        Given first priority.
    channel_range : list
        Range of frequency channel indices, either
        [start, stop, step], [start, stop], or [stop]
        is acceptable.  Given second priority.
    channel_index : list
        List of frequency channel indices.
        Given third priority.
    only_autos : bool
        Only load the autocorrelations.
    """

    padding = config.Property(proptype=float, default=0.005)

    freq_physical = config.Property(proptype=list, default=[])
    channel_range = config.Property(proptype=list, default=[])
    channel_index = config.Property(proptype=list, default=[])

    only_autos = config.Property(proptype=bool, default=False)

    def setup(self, files):
        """Divide the list of files up into sidereal days.

        Parameters
        ----------
        files : list
            List of files to load.
        """
        self.files = files

        filemap = None
        if self.comm.rank == 0:
            se_times = get_times(self.files)
            se_csd = chime.unix_to_lsd(se_times)
            days = np.unique(np.floor(se_csd).astype(np.int64))

            # Construct list of files in each day
            filemap = [
                (day, _days_in_csd(day, se_csd, extra=self.padding)) for day in days
            ]

            # Filter our days with only a few files in them.
            filemap = [(day, dmap) for day, dmap in filemap if dmap.size > 1]
            filemap.sort()

        self.filemap = self.comm.bcast(filemap, root=0)

        # Set up frequency selection.
        if self.freq_physical:
            basefreq = np.linspace(800.0, 400.0, 1024, endpoint=False)
            self.freq_sel = sorted(
                {np.argmin(np.abs(basefreq - freq)) for freq in self.freq_physical}
            )

        elif self.channel_range and (len(self.channel_range) <= 3):
            self.freq_sel = list(range(*self.channel_range))

        elif self.channel_index:
            self.freq_sel = self.channel_index

        else:
            self.freq_sel = None

    def process(self):
        """Load in each sidereal day.

        Returns
        -------
        ts : andata.CorrData
            The timestream of each sidereal day.
        """
        if len(self.filemap) == 0:
            raise pipeline.PipelineStopIteration

        # Extract filelist for this CSD
        csd, fmap = self.filemap.pop(0)
        dfiles = sorted([self.files[fi] for fi in fmap])

        self.log.debug("Starting read of CSD:%i [%i files]", csd, len(fmap))

        # Set up product selection
        # NOTE: this probably doesn't work with stacked data
        prod_sel = None
        if self.only_autos:
            rd = andata.CorrReader(dfiles)
            prod_sel = np.array(
                [ii for (ii, pp) in enumerate(rd.prod) if pp[0] == pp[1]]
            )

        # Load files
        ts = andata.CorrData.from_acq_h5(
            dfiles, distributed=True, freq_sel=self.freq_sel, prod_sel=prod_sel
        )

        # Add attributes for the CSD and a tag for labelling saved files
        ts.attrs["tag"] = "csd_%i" % csd
        ts.attrs["lsd"] = csd

        # Add a weight dataset if needed
        if "vis_weight" not in ts.flags:
            weight_dset = ts.create_flag(
                "vis_weight",
                shape=ts.vis.shape,
                dtype=np.uint8,
                distributed=True,
                distributed_axis=0,
            )
            weight_dset.attrs["axis"] = ts.vis.attrs["axis"]

            # Set weight to maximum value (255), unless the vis value is
            # zero which presumably came from missing data. NOTE: this may have
            # a small bias
            weight_dset[:] = np.where(ts.vis[:] == 0.0, 0, 255)

        gc.collect()

        return ts


class SiderealGrouper(sidereal.SiderealGrouper):
    """SiderealGrouper that automatically uses the location of CHIME.

    See `draco.analysis.sidereal.SiderealGrouper` for extended documentation.
    """

    def setup(self, observer=None):
        """Setup the SiderealGrouper task.

        Parameters
        ----------
        observer : caput.time.Observer, optional
            Details of the observer, if not set default to CHIME.
        """
        # Set up the default Observer
        observer = chime if observer is None else observer

        sidereal.SiderealGrouper.setup(self, observer)


class WeatherGrouper(SiderealGrouper):
    """Group Weather files together within the sidereal day.

    This is just a weather file specific version of `SiderealGrouper`.
    """

    def _process_current_lsd(self):
        # Override with a weather file specific version. It's not clear *why* exactly
        # this is needed

        # Check if we have weather data for this day.
        if len(self._timestream_list) == 0:
            self.log.info("No weather data for this sidereal day")
            return None

        # Check if there is data missing
        # Calculate the length of data in this current LSD
        start = self._timestream_list[0].time[0]
        end = self._timestream_list[-1].time[-1]
        sid_seconds = 86400.0 / ctime.SIDEREAL_S

        if (end - start) < (sid_seconds + 2 * self.padding):
            self.log.info("Not enough weather data - skipping this day")
            return None

        lsd = self._current_lsd

        # Convert the current lsd day to unix time and pad it.
        unix_start = self.observer.lsd_to_unix(lsd)
        unix_end = self.observer.lsd_to_unix(lsd + 1)
        self.pad_start = unix_start - self.padding
        self.pad_end = unix_end + self.padding

        times = np.concatenate([ts.time for ts in self._timestream_list])
        start_ind = int(np.argmin(np.abs(times - self.pad_start)))
        stop_ind = int(np.argmin(np.abs(times - self.pad_end)))

        self.log.info("Constructing LSD:%i [%i files]", lsd, len(self._timestream_list))

        # Concatenate timestreams
        ts = tod.concatenate(self._timestream_list, start=start_ind, stop=stop_ind)

        # Make sure that our timestamps of the concatenated files don't fall
        # out of the requested lsd time span
        if (ts.time[0] > unix_start) or (ts.time[-1] < unix_end):
            return None

        ts.attrs["tag"] = "lsd_%i" % lsd
        ts.attrs["lsd"] = lsd

        return ts


class SiderealRegridder(sidereal.SiderealRegridder):
    """SiderealRegridder that automatically uses the location of CHIME.

    See `draco.analysis.sidereal.SiderealRegridder` for extended documentation.
    """

    def setup(self, observer=None):
        """Setup the SiderealRegridder task.

        Parameters
        ----------
        observer : caput.time.Observer, optional
            Details of the observer, if not set default to CHIME.
        """
        # Down mix requires the baseline distribution to work and so this simple
        # wrapper around the draco regridder will not work if it is turned on
        if self.down_mix and observer is None:
            raise ValueError("A Telescope object must be supplied if down_mix=True.")

        # Set up the default Observer
        observer = chime if observer is None else observer

        sidereal.SiderealRegridder.setup(self, observer)


class SiderealMean(task.SingleTask):
    """Calculate the weighted mean(median) over time.

    Parameters
    ----------
    median : bool
        Calculate weighted median instead of the weighted mean.
    inverse_variance : bool
        Use inverse variance weights.  If this is False, then use uniform
        weighting of timesamples where the `weight` dataset is greater than zero.
    mask_day : bool
        Mask out daytime data prior to calculating the mean(median).
        Note that timestamps where the weight dataset is equal to zero will be excluded
        from the calculation regardless.
    mask_ra : list of two element lists
        Only use data within these ranges of right ascension (in degrees) to calculate
        the mean(median).
    mask_sources : bool
        Mask out the transit of bright sources prior to calculating the mean(median).
        The set of sources that are masked is determined by the `flux_threshold` and
        `dec_threshold` parameters, with values of 400 Jy and 5 deg masking out
        the big 4 (CygA, CasA, TauA, VirA).
    flux_threshold : float
        Only mask sources with flux above this threshold in Jansky.
    dec_threshold : float
        Only mask sources with declination above this threshold in degrees.
    nsigma : float
        Mask this number of sigma on either side of the source transits.
        Here sigma is the expected width of the primary beam for an E-W
        polarisation antenna at 400 MHz as defined by
        :py:func:`ch_util.cal_utils.guess_fwhm`.
    missing_threshold : float
        If less than this fraction of data remains within the regions included by the
        specified masks, then all the data is counted as missing (per baseline and
        frequency). Default is 0.0, i.e. this is not applied.
    """

    median = config.Property(proptype=bool, default=False)
    inverse_variance = config.Property(proptype=bool, default=False)
    mask_day = config.Property(proptype=bool, default=False)
    mask_ra = config.Property(proptype=list, default=[])
    mask_sources = config.Property(proptype=bool, default=False)
    flux_threshold = config.Property(proptype=float, default=400.0)
    dec_threshold = config.Property(proptype=bool, default=5.0)
    nsigma = config.Property(proptype=float, default=2.0)
    missing_threshold = config.Property(proptype=float, default=0.0)
    use_default_range_for_quarter = config.Property(proptype=bool, default=False)

    _reference_ra_range: ClassVar[dict[str, list]] = {
        "q1": [[150.0, 165.0]],
        "q2": [[240.0, 255.0]],
        "q3": [[315.0, 330.0]],
        "q4": [[15.0, 30.0]],
    }

    def setup(self):
        """Determine which sources will be masked, if any."""
        from ch_util import fluxcat

        self._name_of_statistic = "median" if self.median else "mean"

        self.body = []
        if self.mask_sources:
            for src, body in sources.source_dictionary.items():
                if src in fluxcat.FluxCatalog:
                    if (
                        fluxcat.FluxCatalog[src].predict_flux(fluxcat.FREQ_NOMINAL)
                        > self.flux_threshold
                    ) and (body.dec.degrees > self.dec_threshold):
                        self.log.info(
                            f"Will mask {src} prior to calculating sidereal {self._name_of_statistic}."
                        )
                        self.body.append(body)

    def process(self, sstream):
        """Calculate the mean (median) over the sidereal day.

        Parameters
        ----------
        sstream : andata.CorrData, containers.TimeStream, containers.SiderealStream,
                  containers.HybridVisStream, containers.RingMap

        Returns
        -------
        mustream : same as sstream
            Same type of container as the input but with a singleton time axis
            that contains the mean (or median) value.
        """
        from .flagging import daytime_flag, transit_flag

        # Make sure we are distributed over frequency
        sstream.redistribute("freq")

        # Extract lsd
        lsd = sstream.attrs["lsd"] if "lsd" in sstream.attrs else sstream.attrs["csd"]
        lsd_list = lsd if hasattr(lsd, "__iter__") else [lsd]

        # Calculate the right ascension, method differs depending on input container
        if "ra" in sstream.index_map:
            ra = sstream.ra
            timestamp = {dd: chime.lsd_to_unix(dd + ra / 360.0) for dd in lsd_list}
            flag_quiet = np.ones(ra.size, dtype=bool)

        elif "time" in sstream.index_map:
            ra = chime.unix_to_lsa(sstream.time)
            timestamp = {lsd: sstream.time}
            flag_quiet = np.fix(chime.unix_to_lsd(sstream.time)) == lsd

        else:
            raise RuntimeError("Format of `sstream` argument is unknown.")

        # If requested, determine "quiet" region of sky.
        # In the case of a SiderealStack, there will be multiple LSDs and the
        # mask will be the logical AND of the mask from each individual LSDs.
        if self.mask_day:
            for dd, time_dd in timestamp.items():
                # Mask daytime data
                flag_quiet &= ~daytime_flag(time_dd)

        if self.mask_sources:
            for dd, time_dd in timestamp.items():
                # Mask data near bright source transits
                for body in self.body:
                    flag_quiet &= ~transit_flag(body, time_dd, nsigma=self.nsigma)

        # Only use data within user specified ranges of RA
        if self.use_default_range_for_quarter:
            dates = ctime.unix_to_datetime(chime.lsd_to_unix(np.array(lsd_list)))
            quarter = Counter([(d.month - 1) // 3 + 1 for d in dates]).most_common(1)[0]
            ra_ranges = self._reference_ra_range[f"q{quarter[0]}"]
            self.log.info(
                f"Using default RA range for quarter {quarter[0]} ({quarter[1]} days)."
            )

        elif self.mask_ra:
            ra_ranges = self.mask_ra

        else:
            ra_ranges = []

        if ra_ranges:
            mask_ra = np.zeros(ra.size, dtype=bool)
            for ra_range in ra_ranges:
                self.log.info(
                    f"Using data between RA = [{ra_range[0]:.2f}, {ra_range[1]:.2f}] deg"
                )
                mask_ra |= (ra >= ra_range[0]) & (ra <= ra_range[1])
            flag_quiet &= mask_ra

        # Create output container
        newra = np.mean(ra[flag_quiet], keepdims=True)
        mustream = sstream.__class__(
            ra=newra,
            axes_from=sstream,
            attrs_from=sstream,
            distributed=True,
            comm=sstream.comm,
        )
        mustream.redistribute("freq")
        mustream.attrs["statistic"] = self._name_of_statistic

        # Dereference visibilities
        all_vis = sstream.vis[:].local_array
        mu_vis = mustream.vis[:].local_array

        # Combine the visibility weights with the quiet flag
        all_weight = sstream.weight[:].local_array * flag_quiet.astype(np.float32)
        if not self.inverse_variance:
            all_weight = (all_weight > 0.0).astype(np.float32)

        # Only include freqs/baselines where enough data is actually present
        frac_present = (all_weight > 0.0).sum(axis=-1) / flag_quiet.sum(axis=-1)
        all_weight *= (frac_present > self.missing_threshold)[..., np.newaxis]

        # Log number of frequencies that do not have enough data
        waxis = sstream.weight.attrs["axis"][:-1]
        caxind = tuple([ii for ii, ax in enumerate(waxis) if ax != "freq"])

        num_freq_missing_local = int(
            (frac_present <= self.missing_threshold).all(axis=caxind).sum()
        )
        num_freq_missing = self.comm.allreduce(num_freq_missing_local, op=MPI.SUM)

        self.log.info(
            "Cannot estimate a sidereal mean for "
            f"{100.0 * num_freq_missing / len(mustream.freq):.2f}% of all frequencies."
        )

        # Save the total number of nonzero samples as the weight dataset of the output
        # container
        mustream.weight[:] = np.sum(all_weight, axis=-1, keepdims=True)

        # Identify any axis not contained in weight
        if isinstance(sstream, containers.HybridVisStream):
            vslc = [np.s_[:, :, :, ee] for ee in range(all_vis.shape[3])]

        elif isinstance(sstream, containers.Ringmap):
            vslc = list(range(all_vis.shape[0]))

        else:
            vslc = [slice(None)]

        # If requested, compute median
        if self.median:

            missing = ~(all_weight.any(axis=-1))

            # Loop over all axes not shared with weight dataset
            for slc in vslc:

                mu_vis[slc][..., 0].real = weighted_median(
                    np.ascontiguousarray(all_vis[slc].real, dtype=np.float32),
                    all_weight,
                )
                mu_vis[slc][..., 0].imag = weighted_median(
                    np.ascontiguousarray(all_vis[slc].imag, dtype=np.float32),
                    all_weight,
                )

                # Where all the weights are zero explicitly set the median to zero
                mu_vis[slc][..., 0][missing] = 0.0

        else:
            # Otherwise calculate the mean
            # Again loop over all axes not shared with weight dataset
            for slc in vslc:
                mu_vis[slc] = np.sum(all_weight * all_vis[slc], axis=-1, keepdims=True)
                mu_vis[slc] *= tools.invert_no_zero(mustream.weight[:])

        # Return container with singleton time axis containing the mean value
        return mustream


class ChangeSiderealMean(task.SingleTask):
    """Subtract or add an overall offset (over time) to each visibility.

    Parameters
    ----------
    add : bool
        Add the value instead of subtracting.
    """

    add = config.Property(proptype=bool, default=False)

    def process(self, sstream, mustream):
        """Add or subtract mustream from the sidereal stream.

        Parameters
        ----------
        sstream : andata.CorrData or containers.SiderealStream
            Timestream or sidereal stream.

        mustream : andata.CorrData or containers.SiderealStream
            Timestream or sidereal stream with 1 element in the time axis
            that contains the value to add or subtract.

        Returns
        -------
        sstream : same as input
            Timestream or sidereal stream with value added or subtracted.
        """
        # Check that input visibilities have consistent shapes
        sshp, mshp = sstream.vis.shape, mustream.vis.shape

        if np.any(sshp[0:-1] != mshp[0:-1]):
            ValueError("Frequency or product axis differ between inputs.")

        if mshp[-1] != 1:
            ValueError("Mean value has incorrect shape, must be (..., 1).")

        # Ensure both inputs are distributed over frequency
        sstream.redistribute("freq")
        mustream.redistribute("freq")

        # Determine indices of autocorrelations
        if "prod" in mustream.index_map:
            prod = mustream.index_map["prod"][mustream.index_map["stack"]["prod"]]
            not_auto = prod["input_a"] != prod["input_b"]

            pslc = tuple(
                [
                    slice(None) if ax in ["prod", "stack"] else np.newaxis
                    for ax in mustream.vis.attrs["axis"]
                ]
            )

            mu = np.where(not_auto[pslc], mustream.vis[:].local_array, 0.0)
        else:
            mu = mustream.vis[:].local_array

        # Add or subtract value to the cross-correlations
        if self.add:
            sstream.vis[:].local_array[:] += mu
        else:
            sstream.vis[:].local_array[:] -= mu

        # Set weights to zero if there was no mean
        sstream.weight[:].local_array[:] *= (
            mustream.weight[:].local_array > 0.0
        ).astype(sstream.weight.dtype)

        # Return sidereal stream with modified offset
        return sstream


def get_times(acq_files):
    """Extract the start and end times of a list of acquisition files.

    Parameters
    ----------
    acq_files : list
        List of filenames.

    Returns
    -------
    times : np.ndarray[nfiles, 2]
        Start and end times.
    """
    if isinstance(acq_files, list):
        return np.array([get_times(acq_file) for acq_file in acq_files])

    if isinstance(acq_files, str):
        # Load in file (but ignore all datasets)
        ad_empty = andata.AnData.from_acq_h5(acq_files, datasets=())
        start = ad_empty.timestamp[0]
        end = ad_empty.timestamp[-1]

        return start, end

    raise TypeError(f"Input {acq_files!r}, not understood")


def _days_in_csd(day, se_csd, extra=0.005):
    # Find which days are in each CSD
    stest = se_csd[:, 1] > day - extra
    etest = se_csd[:, 0] < day + 1 - extra

    return np.where(np.logical_and(stest, etest))[0]
