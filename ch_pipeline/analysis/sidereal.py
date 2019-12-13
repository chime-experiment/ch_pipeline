"""
=====================================================================
Tasks for sidereal regridding (:mod:`~ch_pipeline.analysis.sidereal`)
=====================================================================

.. currentmodule:: ch_pipeline.analysis.sidereal

Tasks for taking the timestream data and regridding it into sidereal days
which can be stacked.

Tasks
=====

.. autosummary::
    :toctree: generated/

    LoadTimeStreamSidereal
    SiderealGrouper
    SiderealRegridder
    SiderealMean
    ChangeSiderealMean

Usage
=====

Generally you would want to use these tasks together. Starting with a
:class:`LoadTimeStreamSidereal`, then feeding that into
:class:`SiderealRegridder` to grid onto each sidereal day, and then into
:class:`SiderealStacker` if you want to combine the different days.
"""
# === Start Python 2/3 compatibility
from __future__ import absolute_import, division, print_function, unicode_literals
from future.builtins import *  # noqa  pylint: disable=W0401, W0614
from future.builtins.disabled import *  # noqa  pylint: disable=W0401, W0614

# === End Python 2/3 compatibility

from past.builtins import basestring
import gc
import numpy as np

from caput import pipeline, config
from caput import mpiutil
from ch_util import andata, ephemeris, tools
from draco.core import task
from draco.analysis import sidereal

from ..core import containers


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
        if mpiutil.rank0:

            se_times = get_times(self.files)
            se_csd = ephemeris.csd(se_times)
            days = np.unique(np.floor(se_csd).astype(np.int))

            # Construct list of files in each day
            filemap = [
                (day, _days_in_csd(day, se_csd, extra=self.padding)) for day in days
            ]

            # Filter our days with only a few files in them.
            filemap = [(day, dmap) for day, dmap in filemap if dmap.size > 1]
            filemap.sort()

        self.filemap = mpiutil.world.bcast(filemap, root=0)

        # Set up frequency selection.
        if self.freq_physical:
            basefreq = np.linspace(800.0, 400.0, 1024, endpoint=False)
            self.freq_sel = sorted(
                set([np.argmin(np.abs(basefreq - freq)) for freq in self.freq_physical])
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

        if mpiutil.rank0:
            print("Starting read of CSD:%i [%i files]" % (csd, len(fmap)))

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
        observer = ephemeris.chime_observer() if observer is None else observer

        sidereal.SiderealGrouper.setup(self, observer)


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

        # Set up the default Observer
        observer = ephemeris.chime_observer() if observer is None else observer

        sidereal.SiderealRegridder.setup(self, observer)


class SiderealMean(task.SingleTask):
    """Calculate the weighted mean(median) over time for each visibility.

    Parameters
    ----------
    median : bool
        Calculate weighted median instead of the weighted mean.
    inverse_variance : bool
        Use inverse variance weights.  If this is False, then use uniform
        weighting of timesamples where the `weight` dataset is greater than zero.
    apply_mask : bool
        Mask out daytime data and (optionally) the transit of bright sources
        prior to calculating the mean(median). Note that timestamps where the
        weight dataset is equal to zero will be excluded from the calculation regardless.
    mask_sources : bool
        Mask out the transit of bright sources prior to calculating the mean(median).
        Only relevant if `apply_mask` is True.  The set of sources that are masked
        is determined by the `flux_threshold` and `dec_threshold` parameters, with
        values of 400 Jy and 5 deg masking out the big 4 (CygA, CasA, TauA, VirA).
    flux_threshold : float
        Only mask sources with flux above this threshold in Jansky.
    dec_threshold : float
        Only mask sources with declination above this threshold in degrees.
    nsigma : float
        Mask this number of sigma on either side of the source transits.
        Here sigma is the expected width of the primary beam for an E-W
        polarisation antenna at 400 MHz as defined by the
       `ch_util.cal_utils.guess_fwhm` function.
    """

    median = config.Property(proptype=bool, default=False)
    inverse_variance = config.Property(proptype=bool, default=False)
    apply_mask = config.Property(proptype=bool, default=False)

    mask_sources = config.Property(proptype=bool, default=True)
    flux_threshold = config.Property(proptype=float, default=400.0)
    dec_threshold = config.Property(proptype=bool, default=5.0)
    nsigma = config.Property(proptype=float, default=2.0)

    def setup(self):
        """Determine which sources will be masked, if any."""
        from ch_util import fluxcat

        self._name_of_statistic = "median" if self.median else "mean"

        self.body = []
        if self.apply_mask and self.mask_sources:
            for src, body in ephemeris.source_dictionary.items():
                if (
                    fluxcat.FluxCatalog[src].predict_flux(fluxcat.FREQ_NOMINAL)
                    > self.flux_threshold
                ) and (body.dec.degrees > self.dec_threshold):

                    self.log.info(
                        "Will mask %s prior to calculating sidereal %s."
                        % (src, self._name_of_statistic)
                    )

                    self.body.append(body)

    def process(self, sstream):
        """Calculate the mean(median) over the sidereal day.

        Parameters
        ----------
        sstream : andata.CorrData or containers.SiderealStream
            Timestream or sidereal stream.

        Returns
        -------
        mustream : same as sstream
            Sidereal stream containing only the mean(median) value.
        """
        from .flagging import daytime_flag, transit_flag
        import weighted as wq

        # Make sure we are distributed over frequency
        sstream.redistribute("freq")

        # Extract lsd
        lsd = sstream.attrs["lsd"] if "lsd" in sstream.attrs else sstream.attrs["csd"]
        lsd_list = lsd if hasattr(lsd, "__iter__") else [lsd]

        # Calculate the right ascension, method differs depending on input container
        if "ra" in sstream.index_map:
            ra = sstream.ra
            timestamp = {dd: ephemeris.csd_to_unix(dd + ra / 360.0) for dd in lsd_list}
            flag_quiet = np.ones(ra.size, dtype=np.bool)

        elif "time" in sstream.index_map:

            ra = ephemeris.lsa(sstream.time)
            timestamp = {lsd: sstream.time}
            flag_quiet = np.fix(ephemeris.unix_to_csd(sstream.time)) == lsd

        else:
            raise RuntimeError("Format of `sstream` argument is unknown.")

        # If requested, determine "quiet" region of sky
        if self.apply_mask:

            # In the case of a SiderealStack, there will be multiple LSDs and the
            # mask will be the logical AND of the mask from each individual LSDs.
            for dd, time_dd in timestamp.items():

                # Mask daytime data
                flag_quiet &= ~daytime_flag(time_dd)

                # Mask data near bright source transits
                for body in self.body:
                    flag_quiet &= ~transit_flag(body, time_dd, nsigma=self.nsigma)

        # Create output container
        newra = np.mean(ra[flag_quiet], keepdims=True)
        mustream = containers.SiderealStream(
            ra=newra,
            axes_from=sstream,
            attrs_from=sstream,
            distributed=True,
            comm=sstream.comm,
        )
        mustream.redistribute("freq")
        mustream.attrs["statistic"] = self._name_of_statistic

        # Dereference visibilities
        all_vis = sstream.vis[:].view(np.ndarray)
        mu_vis = mustream.vis[:].view(np.ndarray)

        # Combine the visibility weights with the quiet flag
        all_weight = sstream.weight[:].view(np.ndarray) * flag_quiet.astype(np.float32)
        if not self.inverse_variance:
            all_weight = (all_weight > 0.0).astype(np.float32)

        # Save the total number of nonzero samples as the weight dataset of the output container
        mustream.weight[:] = np.sum(all_weight, axis=-1, keepdims=True)

        # If requested, compute median (requires loop over frequencies and baselines)
        if self.median:
            nfreq, nbaseline, _ = all_vis.shape
            for ff in range(nfreq):
                for bb in range(nbaseline):

                    vis = all_vis[ff, bb]
                    weight = all_weight[ff, bb]

                    # wq.median will generate warnings and return NaN if the weights are all zero.
                    # Check for this case and set the mean visibility to 0+0j.
                    if np.any(weight):
                        mu_vis[ff, bb, 0] = wq.median(
                            vis.real, weight
                        ) + 1.0j * wq.median(vis.imag, weight)
                    else:
                        mu_vis[ff, bb, 0] = 0.0 + 0.0j

        else:
            # Otherwise calculate the mean
            mu_vis[:] = np.sum(all_weight * all_vis, axis=-1, keepdims=True)
            mu_vis[:] *= tools.invert_no_zero(mustream.weight[:])

        # Return sidereal stream containing the mean value
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
        """
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

        if np.any(sshp[0:2] != mshp[0:2]):
            ValueError("Frequency or product axis differ between inputs.")

        if (len(mshp) != 3) or (mshp[-1] != 1):
            ValueError("Mean value has incorrect shape, must be (nfreq, nprod, 1).")

        # Ensure both inputs are distributed over frequency
        sstream.redistribute("freq")
        mustream.redistribute("freq")

        # Determine indices of autocorrelations
        prod = mustream.index_map["prod"][mustream.index_map["stack"]["prod"]]
        not_auto = (prod["input_a"] != prod["input_b"]).astype(np.float32)
        not_auto = not_auto[np.newaxis, :, np.newaxis]

        # Add or subtract value to the cross-correlations
        if self.add:
            sstream.vis[:] += mustream.vis[:].view(np.ndarray) * not_auto
        else:
            sstream.vis[:] -= mustream.vis[:].view(np.ndarray) * not_auto

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
    elif isinstance(acq_files, basestring):
        # Load in file (but ignore all datasets)
        ad_empty = andata.AnData.from_acq_h5(acq_files, datasets=())
        start = ad_empty.timestamp[0]
        end = ad_empty.timestamp[-1]
        return start, end
    else:
        raise Exception("Input %s, not understood" % repr(acq_files))


def _days_in_csd(day, se_csd, extra=0.005):
    # Find which days are in each CSD
    stest = se_csd[:, 1] > day - extra
    etest = se_csd[:, 0] < day + 1 - extra

    return np.where(np.logical_and(stest, etest))[0]
