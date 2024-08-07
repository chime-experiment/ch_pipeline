"""Tasks for fringestopping CHIME data.

Tasks for taking the timestream data and fringestop it to a given source

Usage
=====

Use this task together with:

* :class:`~ch_pipeline.core.dataquery.QueryDatabase` to query the database
  and generate a file list.
* :class:`~ch_pipeline.core.io.LoadCorrDataFiles` to load the timestream
  from the files in the previous file list
* :class:`~ch_pipeline.core.dataquery.QueryInputs` to query the inputmap
  of the timestream data
"""

from caput import config
from ch_ephem import sources
from ch_ephem.observers import chime
from ch_util import tools
from draco.core import containers, task


class FringeStop(task.SingleTask):
    """Fringe stop CHIME data to a given source.

    Parameters
    ----------
    source : string
        The source to fringe stop data to, for example, 'CAS_A', 'CYG_A'
    overwrite : bool
        Whether overwrite the original timestream data with the fringestopped
        timestream data, default is False
    telescope_rotation : float
        Rotation of the telescope from true north in degrees.  A positive rotation is
        anti-clockwise when looking down at the telescope from the sky.
    wterm : bool (default False)
        Include the w term (vertical displacement) in the fringestop phase calculation.
    """

    source = config.Property(proptype=str)
    overwrite = config.Property(proptype=bool, default=False)
    telescope_rotation = config.Property(proptype=float, default=chime.rotation)
    wterm = config.Property(proptype=bool, default=False)

    def process(self, tstream, inputmap):
        """Apply the fringe stop of CHIME data to a given source.

        Parameters
        ----------
        tstream : andata.CorrData
            timestream data to be fringestoped
        inputmap : list of :class:`CorrInput`
            A list describing the inputs as they are in the file, output from
            `ch_pipeline.core.dataquery.QueryInputs`

        Returns
        -------
        tstream : andata.CorrData
            Returns the same timestream object but fringestopped
        """
        tstream.redistribute("freq")

        start_freq = tstream.vis.local_offset[0]
        nfreq = tstream.vis.local_shape[0]
        end_freq = start_freq + nfreq
        freq = tstream.freq[start_freq:end_freq]
        prod_map = tstream.index_map["prod"][tstream.index_map["stack"]["prod"]]
        src = sources.source_dictionary[self.source]

        # Rotate the telescope
        tools.change_chime_location(rotation=self.telescope_rotation)

        # Fringestop
        fs_vis = tools.fringestop_time(
            tstream.vis[:],
            times=tstream.time,
            freq=freq,
            feeds=inputmap,
            src=src,
            prod_map=prod_map,
            wterm=self.wterm,
            inplace=self.overwrite,
        )

        # Return telescope to default rotation
        tools.change_chime_location(default=True)

        if self.overwrite:
            return tstream

        tstream_fs = containers.empty_like(tstream)
        tstream_fs.vis[:] = fs_vis
        tstream_fs.weight[:] = tstream.weight

        return tstream_fs
