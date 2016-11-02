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

    LoadFiles
    LoadMaps
    LoadFilesFromParams
    LoadSetupFile
    LoadFileFromTag
    Save
    Print
    LoadBeamTransfer

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

from caput import pipeline, mpiutil
from caput import config

from ch_util import andata

from . import task


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


class LoadMaps(pipeline.TaskBase):
    """Load a series of maps from files given in the tasks parameters.

    Maps are given as one, or a list of `File Groups` (see
    :mod:`ch_pipeline.core.io`). Maps within the same group are added together
    before being passed on.

    Attributes
    ----------
    maps : list or dict
        A dictionary specifying a file group, or a list of them.
    """

    maps = config.Property(proptype=_list_of_filegroups)

    def next(self):
        """Load the groups of maps from disk and pass them on.

        Returns
        -------
        map : :class:`containers.Map`
        """

        from . import containers

        # Exit this task if we have eaten all the file groups
        if len(self.maps) == 0:
            raise pipeline.PipelineStopIteration

        group = self.maps.pop(0)

        map_stack = None

        # Iterate over all the files in the group, load them into a Map
        # container and add them all together
        for mfile in group['files']:

            current_map = containers.Map.from_file(mfile, distributed=True)
            current_map.redistribute('freq')

            # Start the stack if needed
            if map_stack is None:
                map_stack = current_map

            # Otherwise, check that the new map has consistent frequencies,
            # nside and pol and stack up.
            else:

                if (current_map.freq != map_stack.freq).all():
                    raise RuntimeError('Maps do not have consistent frequencies.')

                if (current_map.index_map['pol'] != map_stack.index_map['pol']).all():
                    raise RuntimeError('Maps do not have the same polarisations.')

                if (current_map.index_map['pixel'] != map_stack.index_map['pixel']).all():
                    raise RuntimeError('Maps do not have the same pixelisation.')

                map_stack.map[:] += current_map.map[:]

        # Assign a tag to the stack of maps
        map_stack.attrs['tag'] = group['tag']

        return map_stack


class LoadFilesFromParams(pipeline.TaskBase):
    """Load data from files given in the tasks parameters.

    Attributes
    ----------
    files : glob pattern, or list
        Can either be a glob pattern, or lists of actual files.
    """

    files = config.Property(proptype=_list_or_glob)

    def next(self):
        """Load the given files in turn and pass on.

        Returns
        -------
        cont : subclass of `memh5.BasicCont`
        """

        from caput import memh5

        if len(self.files) == 0:
            raise pipeline.PipelineStopIteration

        # Fetch and remove the first item in the list
        file_ = self.files.pop(0)

        print "Loading file %s" % file_

        cont = memh5.BasicCont.from_file(file_, distributed=True)

        if 'tag' not in cont.attrs:
            # Get the first part of the actual filename and use it as the tag
            tag = os.path.splitext(os.path.basename(file_))[0]

            cont.attrs['tag'] = tag

        return cont


# Define alias for old code
LoadBasicCont = LoadFilesFromParams


class LoadFiles(LoadFilesFromParams):
    """Load data from files passed into the setup routine.

    File must be a serialised subclass of :class:`memh5.BasicCont`.
    """

    files = None

    def setup(self, files):
        """Set the list of files to load.

        Parameters
        ----------
        files : list
        """
        if not isinstance(files, (list, tuple)):
            raise RuntimeError('Argument must be list of files.')

        self.files = files


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

        if mpiutil.rank0:
            print "Reading file %i of %i." % (self._file_ptr, len(self.files))

        # Set up product selection
        prod_sel = None
        if self.only_autos:
            rd = andata.CorrReader(file_)
            prod_sel = np.array([ ii for (ii, pp) in enumerate(rd.prod) if pp[0] == pp[1] ])

        # Load file
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


class Save(pipeline.TaskBase):
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
        """Write out the data file.

        Assumes it has an MPIDataset interface.

        Parameters
        ----------
        data : mpidataset.MPIDataset
            Data to write out.
        """

        if 'tag' not in data.attrs:
            tag = self.count
            self.count += 1
        else:
            tag = data.attrs['tag']

        fname = '%s_%s.h5' % (self.root, str(tag))

        data.to_hdf5(fname)

        return data


class Print(pipeline.TaskBase):
    """Stupid module which just prints whatever it gets. Good for debugging.
    """

    def next(self, input_):

        print input_

        return input_


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

        # Check that the file exists
        if not os.path.exists(self.filename):
            raise RuntimeError('File does not exist: %s' % self.filename)

        print "Loading file: %s" % self.filename

        # Load into container
        cont = memh5.BasicCont.from_file(self.filename, distributed=True)

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
            prefix + incont.attrs['tag] + '.h5'

    only_prefix : bool
        If True, then the class will return the same
        container at each iteration.  The filename
        is assumed to have the format:
            prefix + '.h5'
    """

    prefix = config.Property(proptype=str)

    only_prefix = config.Property(proptype=bool, default=False)

    def setup(self):
        """Determine filename convention.  Load the file into
        a container if only_prefix is True.
        """

        from caput import memh5

        if self.only_prefix:

            filename = self.prefix

            extension = os.path.splitext(filename)[1]
            if extension not in [".h5", ".hdf5"]:
                filename += ".h5"

            # Check that the file exists
            if not os.path.exists(filename):
                raise RuntimeError('File does not exist: %s' % filename)

            if mpiutil.rank0:
                print "Loading file: %s" % filename

            # Load into container
            self.outcont = memh5.BasicCont.from_file(filename, distributed=True)

        else:

            self.prefix = os.path.splitext(self.prefix)[0]


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

            if mpiutil.rank0:
                print "Loading file: %s" % filename

            # Load into container
            self.outcont = memh5.BasicCont.from_file(self.filename, distributed=True)

        return self.outcont


class LoadBeamTransfer(pipeline.TaskBase):
    """Loads a beam transfer manager from disk.

    Attributes
    ----------
    product_directory : str
        Path to the saved Beam Transfer products.
    """

    product_directory = config.Property(proptype=str)

    def setup(self):
        """Load the beam transfer matrices.

        Returns
        -------
        tel : TransitTelescope
            Object describing the telescope.
        bt : BeamTransfer
            BeamTransfer manager.
        feed_info : list, optional
            Optional list providing additional information about each feed.
        """

        import os

        from drift.core import beamtransfer

        if not os.path.exists(self.product_directory):
            raise RuntimeError('BeamTransfers do not exist.')

        bt = beamtransfer.BeamTransfer(self.product_directory)

        tel = bt.telescope

        try:
            return tel, bt, tel.feeds
        except AttributeError:
            return tel, bt
