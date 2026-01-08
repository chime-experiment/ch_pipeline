"""Tasks for IO.

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

import gc
import os.path
import re
from typing import ClassVar

import numpy as np
from caput import config
from caput.pipeline import exceptions, tasklib
from ch_util import andata

from . import containers


class LoadCorrDataFiles(tasklib.base.ContainerTask, tasklib.io.SelectionsMixin):
    """Load CHIME correlator data from a file list passed into the setup routine.

    File must be a serialised subclass of :class:`ch_util.andata.CorrData`.

    Attributes
    ----------
    freq_physical : list
        List of physical frequencies in MHz. Given highest priority.
    channel_range : list
        Range of frequency channel indices, either `[start, stop, step]`, `[start,
        stop]`, or `[stop]` is acceptable. Given second priority.
    channel_index : list
        List of frequency channel indices. Given third priority.
    datasets : list
        List of datasets to load. Defaults to all available datasets.
    only_autos : bool
        Only load the autocorrelations.
    use_draco_container : bool
        Load the data into a draco compatible container rather than CorrData. Defaults
        to True.
    """

    files: list[str] = None

    _file_ptr = 0

    freq_physical = config.Property(proptype=list, default=[])
    channel_range = config.Property(proptype=list, default=[])
    channel_index = config.Property(proptype=list, default=[])

    datasets = config.Property(default=None)

    only_autos = config.Property(proptype=bool, default=False)

    use_draco_container = config.Property(proptype=bool, default=True)

    def setup(self, files: list[str]):
        """Set the list of files to load.

        Parameters
        ----------
        files : list
            list of correlator data file paths
        """
        if not isinstance(files, list | tuple):
            raise RuntimeError("Argument must be list of files.")

        self.files = files

        self._sel = self._resolve_sel()

        # Set up frequency selection.
        if self.freq_physical:
            basefreq = np.linspace(800.0, 400.0, 1024, endpoint=False)
            freq_sel = sorted(
                {np.argmin(np.abs(basefreq - freq)) for freq in self.freq_physical}
            )

        elif self.channel_range and (len(self.channel_range) <= 3):
            freq_sel = slice(*self.channel_range)

        elif self.channel_index:
            freq_sel = self.channel_index

        else:
            freq_sel = slice(None)

        self._sel["freq_sel"] = freq_sel

    def process(self):
        """Load in each sidereal day.

        Returns
        -------
        ts : andata.CorrData or containers.CHIMETimeStream
            The timestream file. Return type depends on the value of
            `use_draco_container`.
        """
        if len(self.files) == self._file_ptr:
            raise exceptions.PipelineStopIteration

        # Collect garbage to remove any prior CorrData objects
        gc.collect()

        # Fetch and remove the first item in the list
        file_ = self.files[self._file_ptr]
        self._file_ptr += 1

        # Set up product selection
        # NOTE: this probably doesn't work with stacked data
        if self.only_autos:
            rd = andata.CorrReader(file_)
            self._sel["prod_sel"] = np.array(
                [ii for (ii, pp) in enumerate(rd.prod) if pp[0] == pp[1]]
            )

        # Load file
        fast_sel = all(
            (sel is None or (axis == "freq_sel" and isinstance(sel, slice)))
            for axis, sel in self._sel.items()
        )

        if fast_sel and (self.datasets is None):
            self.log.info(
                "Reading file %i of %i. (%s) [fast io]",
                self._file_ptr,
                len(self.files),
                file_,
            )
            ts = andata.CorrData.from_acq_h5_fast(
                file_,
                comm=self.comm,
                **self._sel,
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
                **self._sel,
            )

        # Store file name
        ts.attrs["filename"] = file_

        # Use a simple incrementing string as the tag
        if "tag" not in ts.attrs:
            tag = f"file{self._file_ptr:03d}"
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
        if self.use_draco_container:
            ts = containers.CHIMETimeStream.from_corrdata(ts)

        return ts


class LoadDataFiles(tasklib.base.ContainerTask):
    """Load general CHIME data from files passed into the setup routine.

    This does *not* support correlator data. Use `LoadCorrDataFiles` instead.
    """

    files = None

    _file_ptr = 0

    acqtype = config.Property(proptype=str, default="weather")

    _acqtype_reader: ClassVar = {
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
            list of chime data file paths to load, EXCLUDING correlator data
        """
        if self.acqtype not in self._acqtype_reader:
            raise ValueError(f'Specified acqtype "{self.acqtype}" is not supported.')

        if not isinstance(files, list | tuple):
            raise ValueError("Argument must be list of files.")

        self.files = files

    def process(self):
        """Load and return the next available file.

        Raises a PipelineStopIteration if there are no more files to load.

        Returns
        -------
        data : subclass of andata.BaseData
        """
        return self._load_next_file()

    def _load_next_file(self):
        """Load the next available file into memory."""
        if self._file_ptr == len(self.files):
            raise exceptions.PipelineStopIteration

        # Collect garbage to remove any prior data objects
        gc.collect()

        # Fetch and remove the next item in the list
        file_ = self.files[self._file_ptr]
        self._file_ptr += 1

        # Set up a Reader class
        rd = self._acqtype_reader[self.acqtype](file_)

        self.log.info(f"Reading file {self._file_ptr} of {len(self.files)}. ({file_})")

        return rd.read()


class LoadGainUpdates(LoadDataFiles):
    """Iterate over gain updates.

    Attributes
    ----------
    acqtype: {"gain"|"digitalgain"}
        Type of acquisition.
    keep_transition: bool
        If this is True, then gain updates that were transitional
        in nature -- i.e., they executed a smooth transition to
        new gains -- will be loaded. By default, transitional
        gain updates are ignored.
    """

    gains = None

    acqtype = config.enum(["gain", "digitalgain"], default="gain")
    keep_transition = config.Property(proptype=bool, default=False)

    def process(self):
        """Load the next available gain update.

        Returns
        -------
        out: StaticGainUpdate
            The next gain update, packaged into a pipeline container.
        """
        # If there are no gains available, then load the next file.
        if self.gains is None:
            self.gains = self._load_next_file()

        # Make sure we are not dealing with a transitional gain update
        if not self.keep_transition:
            while "transition" in self.gains.update_id[self._time_ptr].decode():
                self._time_ptr += 1

                if self._time_ptr == self.gains.ntime:
                    self.gains = self._load_next_file()

        # Create output container
        out = containers.StaticGainData(
            axes_from=self.gains,
            attrs_from=self.gains,
            distributed=True,
            comm=self.comm,
        )

        out.add_dataset("weight")
        out.redistribute("freq")

        # Save the update_time and update_id as attributes
        out.attrs["time"] = self.gains.time[self._time_ptr]
        out.attrs["update_id"] = self.gains.update_id[self._time_ptr].decode()
        out.attrs["tag"] = out.attrs["update_id"]

        # Find the local frequencies
        sfreq = out.gain.local_offset[0]
        efreq = sfreq + out.gain.local_shape[0]

        fsel = slice(sfreq, efreq)

        # Transfer over the gains and weights for the local frequencies
        out.gain[:] = self.gains.gain[self._time_ptr][fsel]

        if "weight" in self.gains:
            out.weight[:] = self.gains.weight[self._time_ptr][fsel]
        else:
            out.weight[:] = 1.0

        # Increment the time pointer
        self._time_ptr += 1

        # Determine if we need to load a new file on the next iteration
        if self._time_ptr == self.gains.ntime:
            self.gains = None

        # Output the static gain container
        return out

    def _load_next_file(self):
        """Load the next available file into memory."""
        gains = super()._load_next_file()
        self._time_ptr = 0

        return gains


class LoadSetupFile(tasklib.io.BaseLoadFiles):
    """Loads a file from disk into a memh5 container during setup.

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
        # Call the baseclass setup to resolve any selections
        super().setup()

        # Load the requested file
        cont = self._load_file(self.filename)

        # Set the done attribute so the pipeline recognizes this task is finished
        self.done = True

        return cont

    def process(self):
        """Override parent process method to do nothing."""
        pass


class LoadFileFromTag(tasklib.io.BaseLoadFiles):
    """Loads a file from disk into a memh5 container.

    The suffix of the filename is extracted from the tag of the input container.

    Attributes
    ----------
    prefix : str
        Filename is assumed to have the format: "{prefix}{incont.attrs['tag']}.h5"
    only_prefix : bool
        If True, then the class will return the same container at each iteration.
        The filename is assumed to have the format: "{prefix}.h5"
    """

    prefix = config.Property(proptype=str)
    only_prefix = config.Property(proptype=bool, default=False)

    def setup(self):
        """Determine filename convention.

        If only_prefix is True, then load the file into a container.
        """
        # Call the baseclass setup to resolve any selections
        super().setup()

        self.outcont = None

        # If we are returning the same file for every iteration,
        # then load that file now.
        if self.only_prefix:
            filename = self.prefix

            split_ext = os.path.splitext(filename)
            if split_ext[1] not in [".h5", ".hdf5"]:
                filename = split_ext[0] + ".h5"

            # Load file into outcont attribute
            self.outcont = self._load_file(filename)

        else:
            self.prefix = os.path.splitext(self.prefix)[0]

    def process(self, incont):
        """Determine filename from input container.  Load file into output container.

        Parameters
        ----------
        incont : subclass of `memh5.BasicCont`
            get the `tag` attribute from this container

        Returns
        -------
        outcont : subclass of `memh5.BasicCont`
        """
        if not self.only_prefix:
            filename = self.prefix + incont.attrs["tag"] + ".h5"

            # Load file into outcont attribute
            self.outcont = self._load_file(filename)

        # Return the outcont attribute
        return self.outcont


class FilterExisting(tasklib.base.MPILoggedTask):
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
        super().__init__()

        from caput.util import mpitools

        self.csd_list = []
        self.corr_files = {}

        if mpitools.rank0:
            # Look for CSDs in the current directory
            import glob

            files = glob.glob("*")
            if self.existing_csd_regex:
                for file_ in files:
                    mo = re.search(self.existing_csd_regex, file_)
                    if mo is not None:
                        self.csd_list.append(int(mo.group(1)))

            # Search the database to get the start and end times of all correlation files
            from ch_ephem.observers import chime
            from chimedb import data_index as di
            from chimedb.core import connect

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

                start_csd = chime.unix_to_lsd(start)
                finish_csd = chime.unix_to_lsd(finish)

                name = os.path.join(acq, fname)
                self.corr_files[name] = (start_csd, finish_csd)

            self.log.debug("Skipping existing CSDs %s", repr(self.csd_list))

        # Broadcast results to other ranks
        self.corr_files = mpitools.world.bcast(self.corr_files, root=0)
        self.csd_list = mpitools.world.bcast(self.csd_list, root=0)

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
            start, end = (int(t) for t in self.corr_files[name])

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

        return sorted(new_files)
