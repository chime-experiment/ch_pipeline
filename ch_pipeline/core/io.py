"""
Tasks for IO

Tasks for calculating IO. Notably a task which will write out the parallel
MPIDataset classes.

File Groups
===========

Several tasks accept groups of files as arguments. These are specified in the YAML file as a dictionary like below.

.. code-block:: yaml

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


import re
import os.path
import gc
import numpy as np

from caput import pipeline, memh5
from caput import config

from ch_util import andata

from draco.core import task


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
    datasets : list
        List of datasets to load.  Defaults to all
        available datasets.
    only_autos : bool
        Only load the autocorrelations.
    """

    files = None

    _file_ptr = 0

    freq_physical = config.Property(proptype=list, default=[])
    channel_range = config.Property(proptype=list, default=[])
    channel_index = config.Property(proptype=list, default=[])

    datasets = config.Property(default=None)

    only_autos = config.Property(proptype=bool, default=False)

    def setup(self, files):
        """Set the list of files to load.

        Parameters
        ----------
        files : list
        """
        if not isinstance(files, (list, tuple)):
            raise RuntimeError("Argument must be list of files.")

        self.files = files

        # Set up frequency selection.
        if self.freq_physical:
            basefreq = np.linspace(800.0, 400.0, 1024, endpoint=False)
            self.freq_sel = sorted(
                set([np.argmin(np.abs(basefreq - freq)) for freq in self.freq_physical])
            )

        elif self.channel_range and (len(self.channel_range) <= 3):
            self.freq_sel = slice(*self.channel_range)

        elif self.channel_index:
            self.freq_sel = self.channel_index

        else:
            self.freq_sel = slice(None)

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
        # NOTE: this probably doesn't work with stacked data
        prod_sel = None
        if self.only_autos:
            rd = andata.CorrReader(file_)
            prod_sel = np.array(
                [ii for (ii, pp) in enumerate(rd.prod) if pp[0] == pp[1]]
            )

        # Load file
        if (
            isinstance(self.freq_sel, slice)
            and (prod_sel is None)
            and (self.datasets is None)
        ):
            self.log.info(
                "Reading file %i of %i. (%s) [fast io]",
                self._file_ptr,
                len(self.files),
                file_,
            )
            ts = andata.CorrData.from_acq_h5_fast(
                file_, freq_sel=self.freq_sel, comm=self.comm
            )
        else:
            self.log.info(
                "Reading file %i of %i. (%s) [slow io]",
                self._file_ptr,
                len(self.files),
                file_,
            )
            ts = andata.CorrData.from_acq_h5(
                file_,
                datasets=self.datasets,
                distributed=True,
                comm=self.comm,
                freq_sel=self.freq_sel,
                prod_sel=prod_sel,
            )

        # Store file name
        ts.attrs["filename"] = file_

        # Use a simple incrementing string as the tag
        if "tag" not in ts.attrs:
            tag = "file%03i" % self._file_ptr
            ts.attrs["tag"] = tag

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

        # Return timestream
        return ts


class LoadDataFiles(task.SingleTask):
    """Load general CHIME data from files passed into the setup routine.

    This does *not* support correlator data. Use `LoadCorrDataFiles` instead.
    """

    files = None

    _file_ptr = 0

    acqtype = config.Property(proptype=str, default="weather")

    _acqtype_reader = {
        "hk": andata.HKReader,
        "hkp": andata.HKPReader,
        "weather": andata.WeatherReader,
        "rawadc": andata.RawADCReader,
        "gain": andata.CalibrationGainReader,
        "digitalgain": andata.DigitalGainReader,
        "flaginput": andata.FlagInputReader,
    }

    def setup(self, files):
        """Set the list of files to load.

        Parameters
        ----------
        files : list
        """
        if self.acqtype not in self._acqtype_reader:
            raise ValueError(f'Specified acqtype "{self.acqtype}" is not supported.')

        if not isinstance(files, (list, tuple)):
            raise ValueError("Argument must be list of files.")

        self.files = files

    def process(self):
        """Load in each sidereal day.

        Returns
        -------
        ts : andata.CorrData
            The timestream of each sidereal day.
        """

        if len(self.files) == self._file_ptr:
            raise pipeline.PipelineStopIteration

        # Collect garbage to remove any prior data objects
        gc.collect()

        # Fetch and remove the first item in the list
        file_ = self.files[self._file_ptr]
        self._file_ptr += 1

        # Set up a Reader class
        rd = self._acqtype_reader[self.acqtype](file_)

        self.log.info(f"Reading file {self._file_ptr} of {len(self.files)}. ({file_})")
        ts = rd.read()

        # Return timestream
        return ts


class LoadSetupFile(task.MPILoggedTask):
    """Loads a file from disk into a memh5 container during setup.

    Attributes
    ----------
    filename : str
        Path to a saved container.
    distributed : bool, optional
        Whether the file should be loaded distributed across ranks.
    convert_strings : bool, optional
        Convert strings to unicode when loading.
    """

    filename = config.Property(proptype=str)
    distributed = config.Property(proptype=bool, default=True)
    convert_strings = config.Property(proptype=bool, default=True)

    def setup(self):
        """Load the file into a container.

        Returns
        -------
        cont : subclass of `memh5.BasicCont`
        """
        # Check that the file exists
        if not os.path.exists(self.filename):
            raise RuntimeError("File does not exist: %s" % self.filename)

        self.log.info("Loading file: %s", self.filename)

        # Load into container
        cont = memh5.BasicCont.from_file(
            self.filename,
            distributed=self.distributed,
            comm=self.comm,
            convert_attribute_strings=self.convert_strings,
            convert_dataset_strings=self.convert_strings,
        )

        # Make sure all nodes have container before return
        self.comm.Barrier()

        # Return container
        return cont


class LoadFileFromTag(task.SingleTask):
    """Loads a file from disk into a memh5 container.
    The suffix of the filename is extracted from the
    tag of the input.

    Attributes
    ----------
    prefix : str
        Filename is assumed to have the format: prefix + incont.attrs['tag'] + '.h5'

    only_prefix : bool
        If True, then the class will return the same
        container at each iteration.  The filename
        is assumed to have the format: prefix + '.h5'

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
                raise RuntimeError("File does not exist: %s" % filename)

            self.log.info("Loading file: %s", filename)

            self.outcont = memh5.BasicCont.from_file(
                filename, distributed=self.distributed
            )

        else:

            self.prefix = os.path.splitext(self.prefix)[0]

        return

    def process(self, incont):
        """Determine filename from the input container.
        Load file into the output container.

        Parameters
        ----------
        incont : subclass of `memh5.BasicCont`

        Returns
        -------
        outcont : subclass of `memh5.BasicCont`
        """

        if not self.only_prefix:

            filename = self.prefix + incont.attrs["tag"] + ".h5"

            # Check that the file exists
            if not os.path.exists(filename):
                raise RuntimeError("File does not exist: %s" % filename)

            self.log.info("Loading file: %s", filename)

            # Load into container
            self.outcont = memh5.BasicCont.from_file(
                filename, distributed=self.distributed
            )

        return self.outcont


class FilterExisting(task.MPILoggedTask):
    """Filter out files from any list that have already been processed.

    Each file is found in the database and is compared against given
    criteria to see if has already been processed.

    Attributes
    ----------
    regex_csd : str
        A regular expression to find processed CSDs. Compared to filenames
        in the current directory.
    min_files_csd : int
        The minimum number of files on a CSD for it to be processed.
    """

    existing_csd_regex = config.Property(proptype=str, default=None)
    skip_csd = config.Property(proptype=list, default=[])
    min_files_in_csd = config.Property(proptype=int, default=6)

    def __init__(self):

        super(FilterExisting, self).__init__()

        self.csd_list = []
        self.corr_files = {}

        if mpiutil.rank0:
            # Look for CSDs in the current directory
            import glob

            files = glob.glob("*")
            if self.existing_csd_regex:
                for file_ in files:
                    mo = re.search(self.existing_csd_regex, file_)
                    if mo is not None:
                        self.csd_list.append(int(mo.group(1)))

            # Search the database to get the start and end times of all correlation files
            from chimedb import data_index as di
            from chimedb.core import connect
            from ch_util import ephemeris

            connect()
            query = (
                di.ArchiveFile.select(
                    di.ArchiveAcq.name,
                    di.ArchiveFile.name,
                    di.CorrFileInfo.start_time,
                    di.CorrFileInfo.finish_time,
                )
                .join(di.ArchiveAcq)
                .switch(di.ArchiveFile)
                .join(di.CorrFileInfo)
            )

            for acq, fname, start, finish in query.tuples():

                if start is None or finish is None:
                    continue

                start_csd = ephemeris.csd(start)
                finish_csd = ephemeris.csd(finish)

                name = os.path.join(acq, fname)
                self.corr_files[name] = (start_csd, finish_csd)

            self.log.debug("Skipping existing CSDs %s", repr(self.csd_list))

        # Broadcast results to other ranks
        self.corr_files = mpiutil.world.bcast(self.corr_files, root=0)
        self.csd_list = mpiutil.world.bcast(self.csd_list, root=0)

    def next(self, files):
        """Filter the incoming file lists."""

        csd_list = {}

        for path in files:

            acq, fname = path.split("/")[-2:]
            name = os.path.join(acq, fname)

            # Always include non corr files
            if name not in self.corr_files:
                self.log.debug("Non time stream file encountered %s.", name)
                continue

            # Figure out which CSD the file starts and ends on
            start, end = [int(t) for t in self.corr_files[name]]

            # Add this file to the set of files for the relevant days
            csd_list.setdefault(start, set()).add(path)
            csd_list.setdefault(end, set()).add(path)

        new_files = set()

        for csd, csd_files in sorted(csd_list.items()):

            if csd in self.csd_list:
                self.log.debug("Skipping existing CSD=%i, files: %s", csd, csd_files)
                continue

            if csd in self.skip_csd:
                self.log.debug("Skipping specified CSD=%i, files: %s", csd, csd_files)
                continue

            if len(csd_files) < self.min_files_in_csd:
                self.log.debug("Skipping CSD=%i with too few files: %s", csd, csd_files)
                continue

            # Great, we passed the cut, add to the final set
            new_files.update(csd_files)

        self.log.debug(
            "Input list %i files, after filtering %i files.", len(files), len(new_files)
        )

        return sorted(list(new_files))
