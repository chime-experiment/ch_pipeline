"""
==========================================
Tasks for IO (:mod:`~ch_pipeline.core.io`)
==========================================

.. currentmodule:: ch_pipeline.core.io

Tasks for calculating IO. Notably a task which will write out the parallel
MPIDataset classes.

Tasks
=====

.. autosummary::
    :toctree: generated/

    LoadCorrDataFiles
    LoadSetupFile
    LoadFileFromTag

File Groups
===========

Several tasks accept groups of files as arguments. These are specified in the YAML file as a dictionary like below.

.. codeblock:: yaml

    list_of_file_groups:
        -   tag: first_group  # An optional tag naming the group
            files:
                -   'file1.h5'
                -   'file[3-4].h5'  # Globs are processed
                -   'file7.h5'

        -   files:  # No tag specified, implicitly gets the tag 'group_2'
                -   'another_file1.h5'
                -   'another_file2.h5'


    single_group:
        files: ['file1.h5', 'file2.h5']
"""

import os.path
import gc
import numpy as np

from caput import pipeline, mpiutil, memh5
from caput import config

from ch_util import andata

from draco.core import task


def _list_of_filelists(files):
    # Take in a list of lists/glob patterns of filenames
    import glob

    f2 = []

    for filelist in files:

        if isinstance(filelist, str):
            filelist = glob.glob(filelist)
        elif isinstance(filelist, list):
            pass
        else:
            raise Exception('Must be list or glob pattern.')
        f2.append(filelist)

    return f2


def _list_or_glob(files):
    # Take in a list of lists/glob patterns of filenames
    import glob

    if isinstance(files, str):
        files = sorted(glob.glob(files))
    elif isinstance(files, list):
        pass
    else:
        raise RuntimeError('Must be list or glob pattern.')

    return files


def _list_of_filegroups(groups):
    # Process a file group/groups
    import glob

    # Convert to list if the group was not included in a list
    if not isinstance(groups, list):
        groups = [groups]

    # Iterate over groups, set the tag if needed, and process the file list
    # through glob
    for gi, group in enumerate(groups):

        files = group['files']

        if 'tag' not in group:
            group['tag'] = 'group_%i' % gi

        flist = []

        for fname in files:
            flist += glob.glob(fname)

        group['files'] = flist

    return groups


class LoadCorrDataFiles(task.SingleTask):
    """Load data from files passed into the setup routine.

    File must be a serialised subclass of :class:`memh5.BasicCont`.

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

    files = None

    _file_ptr = 0

    freq_physical = config.Property(proptype=list, default=[])
    channel_range = config.Property(proptype=list, default=[])
    channel_index = config.Property(proptype=list, default=[])

    only_autos = config.Property(proptype=bool, default=False)

    def setup(self, files):
        """Set the list of files to load.

        Parameters
        ----------
        files : list
        """
        if not isinstance(files, (list, tuple)):
            raise RuntimeError('Argument must be list of files.')

        self.files = files

        # Set up frequency selection.
        if self.freq_physical:
            basefreq = np.linspace(800.0, 400.0, 1024, endpoint=False)
            self.freq_sel = sorted(set([ np.argmin(np.abs(basefreq - freq))
                                         for freq in self.freq_physical ]))

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

        if len(self.files) == self._file_ptr:
            raise pipeline.PipelineStopIteration

        # Collect garbage to remove any prior CorrData objects
        gc.collect()

        # Fetch and remove the first item in the list
        file_ = self.files[self._file_ptr]
        self._file_ptr += 1

        # Set up product selection
        prod_sel = None
        if self.only_autos:
            rd = andata.CorrReader(file_)
            prod_sel = np.array([ ii for (ii, pp) in enumerate(rd.prod) if pp[0] == pp[1] ])

        # Load file
        self.log.info("Reading file %i of %i. (%s)", self._file_ptr, len(self.files), file_)

        ts = andata.CorrData.from_acq_h5(file_, distributed=True,
                                         freq_sel=self.freq_sel, prod_sel=prod_sel)

        # Use a simple incrementing string as the tag
        if 'tag' not in ts.attrs:
            tag = 'file%03i' % self._file_ptr
            ts.attrs['tag'] = tag

        # Add a weight dataset if needed
        if 'vis_weight' not in ts.flags:
            weight_dset = ts.create_flag('vis_weight', shape=ts.vis.shape, dtype=np.uint8,
                                         distributed=True, distributed_axis=0)
            weight_dset.attrs['axis'] = ts.vis.attrs['axis']

            # Set weight to maximum value (255), unless the vis value is
            # zero which presumably came from missing data. NOTE: this may have
            # a small bias
            weight_dset[:] = np.where(ts.vis[:] == 0.0, 0, 255)

        # Return timestream
        return ts


class LoadSetupFile(pipeline.TaskBase):
    """Loads a file from disk into a memh5 container
    during setup.

    Attributes
    ----------
    filename : str
        Path to a saved container.
    """

    filename = config.Property(proptype=str)

    def setup(self):
        """Load the file into a container.

        Returns
        -------
        cont : subclass of `memh5.BasicCont`
        """

        from caput import memh5

        cont = None

        # Check that the file exists
        if not os.path.exists(self.filename):
            raise RuntimeError('File does not exist: %s' % self.filename)

        self.log.info("Loading file: %s", self.filename)

        if mpiutil.rank0:

            # Load into container
            cont = memh5.BasicCont.from_file(self.filename, distributed=False)

        # Broadcast to other nodes
        cont = mpiutil.world.bcast(cont, root=0)

        # Make sure all nodes have container before return
        mpiutil.world.Barrier()

        # Return container
        return cont


class LoadFileFromTag(task.SingleTask):
    """Loads a file from disk into a memh5 container.
    The suffix of the filename is extracted from the
    tag of the input.

    Attributes
    ----------
    prefix : str
        Filename is assumed to have the format:
            prefix + incont.attrs['tag'] + '.h5'

    only_prefix : bool
        If True, then the class will return the same
        container at each iteration.  The filename
        is assumed to have the format:
            prefix + '.h5'

    distributed : bool
        Whether or not the memh5 container should be
        distributed.
    """

    prefix = config.Property(proptype=str)

    only_prefix = config.Property(proptype=bool, default=False)

    distributed = config.Property(proptype=bool, default=False)

    def setup(self):
        """Determine filename convention.  If only_prefix is True,
        then load the file into a container.
        """

        from caput import memh5

        self.outcont = None

        if self.only_prefix:

            filename = self.prefix

            split_ext = os.path.splitext(filename)
            if split_ext[1] not in [".h5", ".hdf5"]:
                filename = split_ext[0] + ".h5"

            # Check that the file exists
            if not os.path.exists(filename):
                raise RuntimeError('File does not exist: %s' % filename)

            self.log.info("Loading file: %s", filename)

            self.outcont = memh5.BasicCont.from_file(filename, distributed=self.distributed)

        else:

            self.prefix = os.path.splitext(self.prefix)[0]

        return

    def process(self, incont):
        """ Determine filename from the input container.
        Load file into the output container.

        Parameters
        ----------
        incont : subclass of `memh5.BasicCont`

        Returns
        -------
        outcont : subclass of `memh5.BasicCont`
        """

        if not self.only_prefix:

            filename = self.prefix + incont.attrs['tag'] + '.h5'

            # Check that the file exists
            if not os.path.exists(filename):
                raise RuntimeError('File does not exist: %s' % filename)

            self.log.info("Loading file: %s", filename)

            # Load into container
            self.outcont = memh5.BasicCont.from_file(filename, distributed=self.distributed)

        return self.outcont
