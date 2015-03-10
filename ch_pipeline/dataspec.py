"""
====================================================
Dataset Specification (:mod:`~ch_pipeline.dataspec`)
====================================================

.. currentmodule:: ch_pipeline.dataspec

Defines a standard way of specifying sets of data to process.

Tasks
=====

.. autosummary::
    :toctree: generated/

    LoadDataspec
    CreateInputMap

Routines
========

.. autosummary::
    :toctree: generated/

    finder_from_spec
    files_from_spec

Dataspec Format
===============

A dataspec is just a dictionary with three required keys.

:name:
    A short name given to the dataset.
:instrument:
    The name of the instrument that took the data.
:timerange:
    A specification of the time range, or ranges that make up the dataset.

The time range is specified either as a dictionary with `start` and `end`
keys, containing datetime objects (in UTC). Or it can be a list of such
ditionaries, to specify multiple time ranges to include. This can be contained
in a dataspec YAML file, and loaded using :class:`LoadDataspec`. Example:

.. codeblock:: yaml

    datasets:
        -   name:       A
            instrument: blanchard
            timerange:
                -   start:  2014-07-26 03:00:00
                    end:    2014-07-28 01:00:00

                -   start:  2014-07-28 11:00:00
                    end:    2014-07-31 00:00:00
"""

import os

from caput import mpiutil, pipeline, config
from ch_util import andata, tools, ephemeris

_DEFAULT_NODE_SPOOF = {'scinet_scratch': '/scratch/k/krs/jrs65/chime_archive/'}


def finder_from_spec(spec, node_spoof=None):
    """Get a `Finder` object from the dataspec.

    Parameters
    ----------
    dspec : dict
        Dataspec dictionary.

    Returns
    -------
    fi : data_index.Finder
    """

    instrument = spec['instrument']
    timerange = spec['timerange']

    fi = None
    if mpiutil.rank0:

        from ch_util import data_index as di

        # Get instrument
        inst_obj = di.ArchiveInst.select().where(di.ArchiveInst.name == instrument).get()

        # Ensure timerange is a list
        if not isinstance(timerange, list):
            timerange = [timerange]

        # Find the earliest and latest times
        earliest = min([ tr['start'] for tr in timerange ])
        latest   = max([ tr['end']   for tr in timerange ])

        # Set the archive_root
        if node_spoof is None and 'node_spoof' in spec:
            node_spoof = spec['node_spoof']

        # Create a finder object limited to the relevant time
        fi = di.Finder(node_spoof=node_spoof)
        #fi.only_corr()

        # Set the time range that encapsulates all the intervals
        fi.set_time_range(earliest, latest)

        # Add in all the time ranges
        for ti in timerange:
            fi.include_time_interval(ti['start'], ti['end'])

        # Only include the required instrument
        fi.filter_acqs(di.ArchiveAcq.inst == inst_obj)

    return fi


def files_from_spec(spec, node_spoof=None):
    """Get the names of files in a dataset.

    Parameters
    ----------
    dspec : dict
        Dataspec dictionary.

    Returns
    -------
    files : list
    """
    # Get a list of files in a dataset from an instrument name and timerange.

    files = None

    if mpiutil.rank0:

        # Get the finder object
        fi = finder_from_spec(spec, node_spoof)

        # Pull out the results and extract all the files
        results = fi.get_results()
        files = [ fname for result in results for fname in result[0] ]
        files.sort()

    files = mpiutil.world.bcast(files, root=0)

    return files


class LoadDataspec(pipeline.TaskBase):
    """Load a dataspec from a file.

    Attributes
    ----------
    dataset_file : str
        YAML file containing dataset specification. If not specified, use the
        one contained within the ch_pipeline repository.
    dataset_name : str
        Name of dataset to use.
    archive_root : str
        Root of archive to add to file paths.
    """

    dataset_file = config.Property(proptype=str, default='')
    dataset_name = config.Property(proptype=str, default='')
    node_spoof = config.Property(proptype=dict, default=_DEFAULT_NODE_SPOOF)

    def setup(self):
        """Fetch the dataspec from the parameters.

        Returns
        -------
        dspec : dict
            Dataspec as dictionary.
        """

        import yaml

        # Set to default datasets file
        if self.dataset_file == '':
            self.dataset_file = os.path.dirname(__file__) + '/data/datasets.yaml'

        # Check existense and read yaml datasets file
        if not os.path.exists(self.dataset_file):
            raise Exception("Dataset file not found.")

        with open(self.dataset_file, 'r') as f:
            dconf = yaml.safe_load(f)

        if 'datasets' not in dconf:
            raise Exception("No datasets.")

        dsets = dconf['datasets']

        # Find the correct dataset
        dspec = None
        for ds in dsets:
            if ds['name'] == self.dataset_name:
                dspec = ds
                break

        # Raise exception if it's not found
        if dspec is None:
            raise Exception("Dataset %s not found in %s." % (self.dataset_name, self.dataset_file))

        if ('instrument' not in dspec) or ('timerange' not in dspec):
            raise Exception("Invalid dataset.")

        # Add archive root if exists
        if self.node_spoof is not None:
            dspec['node_spoof'] = self.node_spoof

        return dspec


class CreateInputMap(pipeline.TaskBase):
    """From a dataspec describing the data create a list of objects describing
    the inputs in the files.
    """

    def setup(self, dspec):
        """Generate an input description from the dataspec.

        Parameters
        ----------
        dspec : dict
            Dataspec as dictionary.

        Returns
        -------
        inputs : list of :class:`CorrInput`s
            A list of describing the inputs as they are in the file.
        """
        file0 = files_from_spec(dspec)[0]
        inputs = None

        if mpiutil.rank0:
            print "Reading file to construct inputmap...",
            # Open up the first file to extract metadata
            ad0 = andata.CorrData.from_acq_h5(file0, datasets=())

            print "done."

            # Get the start time of the data
            time0 = ephemeris.unix_to_datetime(ad0.timestamp[0])

            # Fetch the serial numbers of the inputs in the file
            input_serials = list(ad0.index_map['input']['correlator_input'])

            # Fetch input description from database, and turn into dict for querying
            inputs = tools.get_correlator_inputs(time0, correlator=dspec['instrument'])
            input_dict = { input_.input_sn: input_ for input_ in inputs }

            # Use a dict lookup to get the inputs as they are arranged in this file
            inputs = [ input_dict[serial] for serial in input_serials ]

            print "Finished input map."

        # Broadcast input description to all ranks
        inputs = mpiutil.world.bcast(inputs, root=0)

        return inputs
