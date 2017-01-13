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
    SiderealStacker
    MeanSubtract

Usage
=====

Generally you would want to use these tasks together. Starting with a
:class:`LoadTimeStreamSidereal`, then feeding that into
:class:`SiderealRegridder` to grid onto each sidereal day, and then into
:class:`SiderealStacker` if you want to combine the different days.
"""

import gc
import numpy as np

from caput import pipeline, config
from caput import mpiutil
from ch_util import andata, ephemeris
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
            filemap = [ (day, _days_in_csd(day, se_csd, extra=self.padding)) for day in days ]

            # Filter our days with only a few files in them.
            filemap = [ (day, dmap) for day, dmap in filemap if dmap.size > 1 ]
            filemap.sort()

        self.filemap = mpiutil.world.bcast(filemap, root=0)

        # Set up frequency selection.
        if self.freq_physical:
            basefreq = np.linspace(800.0, 400.0, 1024, endpoint=False)
            self.freq_sel = sorted(set([ np.argmin(np.abs(basefreq - freq)) for freq in self.freq_physical ]))

        elif self.channel_range and (len(self.channel_range) <= 3):
            self.freq_sel = range(*self.channel_range)

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
        dfiles = sorted([ self.files[fi] for fi in fmap ])

        if mpiutil.rank0:
            print "Starting read of CSD:%i [%i files]" % (csd, len(fmap))

        # Set up product selection
        prod_sel = None
        if self.only_autos:
            rd = andata.CorrReader(dfiles)
            prod_sel = np.array([ ii for (ii, pp) in enumerate(rd.prod) if pp[0] == pp[1] ])

        # Load files
        ts = andata.CorrData.from_acq_h5(dfiles, distributed=True,
                                         freq_sel=self.freq_sel, prod_sel=prod_sel)

        # Add attributes for the CSD and a tag for labelling saved files
        ts.attrs['tag'] = ('csd_%i' % csd)
        ts.attrs['lsd'] = csd

        # Add a weight dataset if needed
        if 'vis_weight' not in ts.flags:
            weight_dset = ts.create_flag('vis_weight', shape=ts.vis.shape, dtype=np.uint8,
                                         distributed=True, distributed_axis=0)
            weight_dset.attrs['axis'] = ts.vis.attrs['axis']

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


class MeanSubtract(task.SingleTask):
    """Subtract the weighted mean (over time) of every element of the
    visibility matrix excluding the auto-correlations.

    Parameters
    ----------
    all_data : bool
        If this is True, then use all of the input data to
        calculate the weighted mean.  If False, then use only
        night-time data away from the transit of bright
        point sources.  Default is False.
    median : bool
        Subtracted weighted median instead of weighted mean.
    """

    all_data = config.Property(proptype=bool, default=False)
    daytime = config.Property(proptype=bool, default=False)
    median = config.Property(proptype=bool, default=True)

    def process(self, sstream):
        """
        Parameters
        ----------
        sstream : andata.CorrData or containers.SiderealStream
            Timestream or sidereal stream.

        Returns
        -------
        sstream : same as input
            Timestream or sidereal stream with mean subtracted.
        """

        from flagging import daytime_flag
        from ch_util import cal_utils, tools
        import weighted as wq

        # Make sure we are distributed over frequency
        sstream.redistribute('freq')

        # Fetch the CSD (preferring it to be labelled the LSD)
        csd = sstream.attrs['lsd'] if 'lsd' in sstream.attrs else sstream.attrs['csd']

        # Extract product map
        prod = sstream.index_map['prod']

        # Check if we are using all of the data to calculate the mean,
        # or only "quiet" periods.
        if self.all_data:

            if isinstance(sstream, andata.CorrData):
                flag_quiet = np.ones(len(sstream.time), dtype=np.bool)

            elif isinstance(sstream, containers.SiderealStream):
                flag_quiet = np.ones(len(sstream.index_map['ra'][:]), dtype=np.bool)

            else:
                raise RuntimeError('Format of `sstream` argument is unknown.')

        else:

            # Check if we are dealing with CorrData or SiderealStream
            if isinstance(sstream, andata.CorrData):
                # Extract ra
                ra = ephemeris.transit_RA(sstream.time)

                # Find night time data
                if self.daytime:
                    flag_quiet = daytime_flag(sstream.time)
                else:
                    flag_quiet = ~daytime_flag(sstream.time)

                flag_quiet &= (np.fix(ephemeris.csd(sstream.time)) == csd)

            elif isinstance(sstream, containers.SiderealStream):
                # Extract csd and ra
                if hasattr(csd, '__iter__'):
                    csd_list = csd
                else:
                    csd_list = [csd]

                ra = sstream.index_map['ra']

                # Find night time data
                flag_quiet = np.ones(len(ra), dtype=np.bool)
                for cc in csd_list:
                    if self.daytime:
                        flag_quiet &= daytime_flag(ephemeris.csd_to_unix(cc + ra / 360.0))
                    else:
                        flag_quiet &= ~daytime_flag(ephemeris.csd_to_unix(cc + ra / 360.0))

            else:
                raise RuntimeError('Format of `sstream` argument is unknown.')

            # Find data free of bright point sources
            for src_name, src_ephem in ephemeris.source_dictionary.iteritems():

                peak_ra = ephemeris.peak_RA(src_ephem, deg=True)
                src_window = 3.0 * cal_utils.guess_fwhm(400.0, pol='X', dec=src_ephem._dec, sigma=True)

                dra = (ra - peak_ra) % 360.0
                dra -= (dra > 180.0) * 360.0

                flag_quiet &= np.abs(dra) > src_window

        # Loop over frequencies and baselines to reduce memory usage
        for lfi, fi in sstream.vis[:].enumerate(0):
            for lbi, bi in sstream.vis[:].enumerate(1):

                # Do not subtract mean of autocorrelation
                if prod[bi][0] == prod[bi][1]:
                    continue

                # Extract visibility and weight
                data = sstream.vis[fi, bi]
                weight = sstream.weight[fi, bi] * flag_quiet

                # Subtract weighted median or weighted mean value
                if self.median:
                    if np.any(weight > 0.0):
                        med = wq.median(data.real, weight) + 1.0J * wq.median(data.imag, weight)
                        sstream.vis[fi, bi] -= med

                else:
                    norm = tools.invert_no_zero(np.sum(weight))
                    sstream.vis[fi, bi] -= norm * np.sum(weight * data)

        # Return mean subtracted map
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
        raise Exception('Input %s, not understood' % repr(acq_files))


def _days_in_csd(day, se_csd, extra=0.005):
    # Find which days are in each CSD
    stest = se_csd[:, 1] > day - extra
    etest = se_csd[:, 0] < day + 1 - extra

    return np.where(np.logical_and(stest, etest))[0]


def _ensure_list(x):

    if hasattr(x, '__iter__'):
        y = [xx for xx in x]
    else:
        y = [x]

    return y
