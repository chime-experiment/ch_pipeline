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

Routines
========

.. autosummary::
    :toctree: generated/

    finder_from_spec
    files_from_spec

Format
======

A dataspec is just a dictionary with three required keys.

:name:
    A short name given to the dataset.
:instrument:
    The name of the instrument that took the data.
:timerange:
    A specification of the time range, or ranges that make up the dataset.

The time range is specified either as a dictionary with `start` and `end`
keys, containing datetime objects (in UTC). Or it can be a list of such
ditionaries, to specify multiple time ranges to include.
"""

import os

from caput import mpiutil, pipeline, config


def finder_from_spec(spec, archive_root=None):
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

        # Create a finder object limited to the relevant time
        fi = di.Finder()

        # Set the archive_root
        if archive_root is None:
            archive_root = spec['archive_root'] if 'archive_root' in spec else ''
        fi.archive_root = archive_root

        # Set the time range that encapsulates all the intervals
        fi.set_time_range(earliest, latest)

        # Add in all the time ranges
        for ti in timerange:
            fi.include_time_interval(ti['start'], ti['end'])

        # Only include the required instrument
        fi.filter_acqs(di.ArchiveAcq.inst == inst_obj)

    return fi


def files_from_spec(spec, archive_root=None):
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
        fi = finder_from_spec(spec, archive_root)

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
    archive_root = config.Property(proptype=str, default='')

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
        if self.archive_root != '':
            dspec['archive_root'] = self.archive_root

        return dspec
