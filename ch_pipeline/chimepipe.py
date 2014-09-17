from caput import pipeline
from caput import config

from caput import mpiutil, mpidataset

import glob
import numpy as np

from ch_util import andata, ephemeris
from ch_util import rfi_pipeline

import containers


def get_times(acq_files):

    if isinstance(acq_files, list):
        return np.array([get_times(acq_file) for acq_file in acq_files])
    elif isinstance(acq_files, str):
        # Load in file (but ignore all datasets)
        ad_empty = andata.AnData.from_acq_h5(acq_files, datasets=())
        start = ad_empty.timestamp[0]
        end = ad_empty.timestamp[-1]
        return start, end
    else:
        raise Exception('Input %s, not understood' % repr(acq_files))


def _days_in_csd(day, se_csd, extra=0.005):

    stest = se_csd[:, 1] > day - extra
    etest = se_csd[:, 0] < day + 1 - extra

    return np.where(np.logical_and(stest, etest))[0]


class LoadTimeStreamSidereal(pipeline.TaskBase):
    """Load data in sidereal days.

    This task takes an input list of data, and loads in a sidereal day at a
    time, and passes it on.

    Attributes
    ----------
    files : glob pattern
        List of filenames as a glob pattern.
    padding : float
        Extra amount of a sidereal day to pad each timestream by. Useful for
        getting rid of interpolation artifacts.
    """

    filepat = config.Property(proptype=str)
    padding = config.Property(proptype=float, default=0.005)

    def setup(self):
        """Divide the list of files up into sidereal days.
        """

        self.files = glob.glob(self.filepat)

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

    def next(self):
        """Load in each sidereal day.

        Returns
        -------
        ts : containers.TimeStream
            The timestream of each sidereal day.
        """

        if len(self.filemap) == 0:
            raise pipeline.PipelineStopIteration

        csd, fmap = self.filemap.pop(0)
        dfiles = [ self.files[fi] for fi in fmap ]

        if mpiutil.rank0:
            print "Starting read of CSD:%i [%i files]" % (csd, len(fmap))

        ts = containers.TimeStream.from_acq_files(dfiles)
        ts.attrs['tag'] = ('csd_%i' % csd)
        ts.attrs['csd'] = csd

        return ts


class RFIFilter(pipeline.TaskBase):
    """Filter RFI from a Timestream.

    Attributes
    ----------
    threshold_mad : float
        Threshold above which we mask the data.
    """

    threshold_mad = config.Property(proptype=float, default=5.0)

    def next(self, data):

        data.redistribute(axis=2)

        # Construct RFI mask
        mask = rfi_pipeline.flag_dataset_with_mad(data, only_autos=False,
                                                  threshold=self.threshold_mad)

        # Turn mask into MPIArray
        mask = mpidataset.MPIArray.wrap(mask.view(np.uint8), axis=2)

        # Create MaskedTimeStream instance and redistribute back over frequency
        mts = containers.MaskedTimeStream.from_timestream_and_mask(data, mask)
        mts.redistribute(0)

        return mts


class SideralRegridder(pipeline.TaskBase):

    offset = config.Property(proptype=float)

    def next(self, data, mask):

        pass

    def finish(self):

        pass


class SaveOutput(pipeline.TaskBase):
    """Save out the input, and pass it on.

    Assumes that the input has a `to_hdf5` method. Appends a *tag* if there is
    a `tag` entry in the attributes, otherwise just uses a count.

    Attributes
    ----------
    root : str
        Root of the file name to output to.
    """

    root = config.Property(proptype=str)

    count = 0

    def next(self, data):

        if 'tag' not in data.attrs:
            tag = self.count
            self.count += 1
        else:
            tag = data.attrs['tag']

        fname = '%s_%s.h5' % (self.root, str(tag))

        data.to_hdf5(fname)

        return data
