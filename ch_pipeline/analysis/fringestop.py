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
    """Correct the phase of a holography transit for the pointing geometry of the
    26m.

    The z-component of the holographic baseline changes with the declination
    pointed to by the 26m and is proportional to the distance labeled CD in the
    memo by John Galt; see DocLib #703.

    Parameters
    ----------
    overwrite : bool
        Sets whether the phase correction is applied in-place (overwriting
        the input transit) or if a new track is returned instead. Default
        is True.
    """
    overwrite = config.Property(proptype=bool, default=True)

    def process(self, track_in, feeds):
        """Apply the phase correction to the input transit.

        Parameters
        ----------
        track_in : draco.core.containers.TrackBeam
            The input holography transit. The correction will be applied
            in-place to this transit if `overwrite` is True.
        feeds : list[ch_util.tools.CorrInput]
            The list of correlator inputs used in the corresponding
            transit. Obtained from `ch_pipeline.core.dataquery.QueryInputs`

        Returns
        -------
        track : draco.core.containers.TrackBeam
            The phase-corrected track.
        """

        # If we're overwriting, the output `track` is just the input
        if self.overwrite:
            track = track_in
        else:
            # Else, create a new container for the output and copy over
            # attributes and data from the input
            track = containers.TrackBeam(
                axes_from=track_in,
                attrs_from=track_in,
                distributed=track_in.distributed,
                comm=track_in.comm,
            )
            track["beam"] = track_in["beam"][:]
            track["weight"] = track_in["weight"][:]

        # Redistribute in frequency
        track.redistribute("freq")

        # Fetch the declination of the track
        src_dec = np.radians(track.attrs["dec"])

        # Construct a full product map from the list of input feeds
        prod_map = _construct_holography_prod_map(feeds)

        # Get the shape of the local data
        nfreq = track.beam.local_shape[0]
        npol = track.beam.local_shape[1]
        ninput = track.beam.local_shape[2]

        # Create a slice for the local portion of the frequency axis
        local_slice = slice(track.beam.local_offset[0], track.beam.local_offset[0] + nfreq)

        freq = track.freq[local_slice]

        # Fetch the geometric delay and turn it into a phase
        bterm_delay = tools.bterm(src_dec, feeds, prod_map)
        bterm_phase = np.exp(
            2.j * np.pi * np.outer(freq, bterm_delay) * 1e6
        )

        # Reshape to multiply into data
        bterm_phase = bterm_phase.reshape(nfreq, npol, ninput, 1)

        # Apply the phase
        track["beam"].local_data[:] *= bterm_phase

        return track


def _construct_holography_prod_map(feeds):
    """Create a product map for holography from a list of feeds.

    Parameters
    ----------
    feeds : list[ch_util.tools.CorrInput]
        A list of correlator inputs to be paired with the Galt inputs.

    Returns
    -------
    prod_map : np.ndarray[2 * nfeeds]
        An array of holographic input produts.
    """
    nfeeds = len(feeds)

    # Fetch the polarizations of the inputs
    input_pols = tools.get_feed_polarisations(feeds)

    # Figure out where in the input list the Galt inputs are
    holo_indices = tools.get_holographic_index(feeds)

    # Initialize a numpy array to contain the product map
    prod_map_dtype = np.dtype([("input_a", np.int32), ("input_b", np.int32)])

    prod_map = np.zeros(2 * nfeeds, dtype=prod_map_dtype)

    # Loop over pols and feeds and fill the product map
    pols = ["co", "cross"]
    for pp, po in enumerate(pols):

        for ii in range(nfeeds):

            # TODO: the layout database queried using the `ch_util.tools`
            # functions does not actually specify the polarisations of
            # the Galt inputs, so for now they are hard-coded here, with
            # the lower Galt index being Y pol and the higher being X
            if po == "co":
                holo_input  = holo_indices[0] if input_pols[ii] == "S" else holo_indices[1]
            else:
                holo_input  = holo_indices[1] if input_pols[ii] == "S" else holo_indices[0]

            pr = nfeeds * pp + ii

            input_pair = (ii, holo_input)

            prod_map[pr] = (min(input_pair), max(input_pair))

    return prod_map
