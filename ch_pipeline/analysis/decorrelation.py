"""Task for correcting decorrelation due to delays between signals

Tasks
=====

.. autosummary::
    :toctree: generated/

    CorrectDecorrelation

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
# === Start Python 2/3 compatibility
from __future__ import absolute_import, division, print_function, unicode_literals
from future.builtins import *  # noqa  pylint: disable=W0401, W0614
from future.builtins.disabled import *  # noqa  pylint: disable=W0401, W0614

# === End Python 2/3 compatibility

import numpy as np
from datetime import datetime
from caput import config
from ch_util import tools, ephemeris
from draco.core import containers, task
from draco.util.tools import invert_no_zero


class CorrectDecorrelation(task.SingleTask):
    """Correct decorrelation in the PFB due to delays between signals.

    The delays and therefore the correction are direction dependent, so
    the source being observed must be specified. This is mainly useful for
    holography observations.

    Parameters
    ----------
    source : string
        The source being observed, for example, 'CAS_A', 'CYG_A'
    overwrite : bool
        Whether overwrite the original timestream data with the corrected
        timestream data, default is False
    telescope_rotation : float
        Rotation of the telescope from true north in degrees.  A positive rotation is
        anti-clockwise when looking down at the telescope from the sky.
    wterm : bool (default False)
        Include the w term (vertical displacement) in the delay calculation.
    bterm : bool (default False)
        Include a correction for the geometry of the 26m Galt telescope.
    """

    source = config.Property(proptype=str)
    overwrite = config.Property(proptype=bool, default=False)
    telescope_rotation = config.Property(proptype=float, default=tools._CHIME_ROT)
    wterm = config.Property(proptype=bool, default=False)
    bterm = config.Property(proptype=bool, default=False)

    def process(self, tstream, inputmap):
        """Apply the decorrelation correction for a given source.

        Parameters
        ----------
        tstream : andata.CorrData
            timestream data
        inputmap : list of :class:`CorrInput`
            A list of describing the inputs as they are in the file, output from
            `tools.get_correlator_inputs()`.

        Returns
        -------
        tstream : andata.CorrData
            Returns the corrected timestream.
        """

        tstream.redistribute("freq")

        prod_map = tstream.prodstack
        src = ephemeris.source_dictionary[self.source]

        # Rotate the telescope
        tools.change_chime_location(rotation=self.telescope_rotation)

        # correct visibilities
        corr_vis = tools.decorrelation(
            tstream.vis[:],
            times=tstream.time,
            feeds=inputmap,
            src=src,
            prod_map=prod_map,
            wterm=self.wterm,
            bterm=self.bterm,
            inplace=self.overwrite,
        )

        # weights are inverse variance
        weight = np.sqrt(invert_no_zero(tstream.weight[:]))
        weight = tools.decorrelation(
            weight,
            times=tstream.time,
            feeds=inputmap,
            src=src,
            prod_map=prod_map,
            wterm=self.wterm,
            bterm=self.bterm,
            inplace=True,
        )
        weight = invert_no_zero(weight) ** 2

        # Return telescope to default rotation
        tools.change_chime_location(default=True)

        if self.overwrite:
            # visibilities were corrected in place
            tstream.weight[:] = weight
            return tstream
        else:
            tstream_corr = containers.empty_like(tstream)
            tstream_corr.vis[:] = corr_vis
            tstream_corr.weight[:] = weight
            return tstream_corr
