"""
=====================================================
Tasks for fringestopping CHIME data to a given source
=====================================================

.. currentmodule:: ch_pipeline.analysis.fringestop

Tasks for taking the timestream data and fringestop it to a given source

Tasks
=====
    Fringestop

Usage
=====
Use this task together with
:ch_pipeline.core.dataquery.QueryDatabase to query the database to generate a file list
:ch_pipeline.core.io.LoadCorrDataFiles to load the timestream data from the files in the previous file list
:ch_pipeline.core.dataquery.QueryInputs to query the inputmap of the timestream data
"""

from datetime import datetime
from caput import config, mpiutil
from ch_util import tools, ephemeris
from draco.core import containers, task


class FringeStop(task.SingleTask):
    """Fringe stop CHIME data to a given source

    Parameters
    ----------
    source : string
        The source to fringe stop data to, for example, 'CAS_A', 'CYG_A'
    overwrite : bool
        Whether overwrite the original timestream data with the fringestopped timestream data, default is False
    """

    source = config.Property(proptype=str)
    overwrite = config.Property(proptype=bool, default=False)

    def process(self, tstream, inputmap):
        """Apply the fringe stop of CHIME data to a given source

        Parameters
        ----------
        tstream : andata.CorrData
            timestream data to be fringestoped
        inputmap : list of :class:`CorrInput`s
            A list of describing the inputs as they are in the file, output from the tools.get_correlator_inputs()

        Returns
        ----------
        tstream : andata.CorrData
            returns the same timestream object but fringestopped
        """

        tstream.redistribute('freq')

        start_freq = tstream.vis.local_offset[0]
        nfreq = tstream.vis.local_shape[0]
        end_freq = start_freq + nfreq
        freq = tstream.freq[start_freq:end_freq]
        prod_map = tstream.index_map['prod']
        src = ephemeris.source_dictionary[self.source]
        feeds = [inputmap[tstream.input[i][0]]
                 for i in range(len(tstream.input))]

        fs_vis = tools.fringestop_time(
            tstream.vis,
            times=tstream.time,
            freq=freq,
            feeds=feeds,
            src=src,
            prod_map=prod_map)

        if self.overwrite:
            tstream.vis[:] = fs_vis
            return tstream
        else:
            tstream_fs = containers.empty_like(tstream)
            tstream_fs.vis[:] = fs_vis
            return tstream_fs
