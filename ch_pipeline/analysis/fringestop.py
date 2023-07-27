"""Tasks for fringestopping CHIME data

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

import numpy as np
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
    telescope_rotation = config.Property(proptype=float, default=tools._CHIME_ROT)
    wterm = config.Property(proptype=bool, default=False)

    def process(self, tstream, inputmap):
        """Apply the fringe stop of CHIME data to a given source

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
        src = ephemeris.source_dictionary[self.source]

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
        else:
            tstream_fs = containers.empty_like(tstream)
            tstream_fs.vis[:] = fs_vis
            tstream_fs.weight[:] = tstream.weight
            return tstream_fs


class ApplyBTermCorrection(task.SingleTask):
    overwrite = config.Property(proptype=bool, default=True)

    def process(self, track_in, feeds):
        if self.overwrite:
            track = track_in
        else:
            track = containers.TrackBeam(
                axes_from=track_in,
                attrs_from=track_in,
                distributed=track_in.distributed,
                comm=track_in.comm,
            )
            track["beam"] = track_in["beam"][:]
            track["weight"] = track_in["weight"][:]

        track.redistribute("freq")

        src_dec = np.radians(track.attrs["dec"])

        self.log.warning("The number of feeds is %d" % len(feeds))
        self.log.warning(f"The feeds are {feeds}")
        prod_map = _construct_holography_prod_map(feeds)

        self.log.warning(f"The shape of the product map is {prod_map.shape}")

        nfreq = track.beam.local_shape[0]
        local_slice = slice(track.beam.local_offset[0], track.beam.local_offset[0] + nfreq)

        freq = track.freq[local_slice]

        bterm_delay = tools.bterm(src_dec, feeds, prod_map)
        self.log.warning(f"The shape of the delay term is {bterm_delay.shape}")
        bterm_phase = np.exp(2.j * np.pi * bterm_delay * freq * 1e6)
        self.log.warning(f"The shape of the delay phase is {bterm_delay.shape}")

        track["beam"].local_data[:] *= (bterm_phase.T.reshape((nfreq, 2, 2048)))[..., np.newaxis]

        return track


def _construct_holography_prod_map(feeds):
    nfeeds = len(feeds)
    holo_indices = tools.get_holographic_index(feeds)

    input_pols = tools.get_feed_polarisations(feeds)

    prod_map_dtype = np.dtype([("input_a", np.int32), ("input_b", np.int32)])

    prod_map = np.zeros(2 * nfeeds, dtype=prod_map_dtype)

    for pp in range(prod_map.shape[0]):
        ii = pp % nfeeds
        copol = ~(pp // nfeeds)
        if copol:
            holo_input = holo_indices[0] if input_pols[ii] == "S" else holo_indices[1]
        else:
            holo_input = holo_indices[1] if input_pols[ii] == "S" else holo_indices[0]

        input_pair = (ii, holo_input)

        prod_map[pp] = (min(input_pair), max(input_pair))

    return prod_map