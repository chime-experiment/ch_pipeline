"""
========================================================
Jackknife tasks (:mod:`~ch_pipeline.analysis.jackknife`)
========================================================

.. currentmodule:: ch_pipeline.analysis.jackknife

Statistical analysis of differenced datasets.

Tasks
=====

.. autosummary::
    :toctree: generated/

    Difference
"""
import numpy as np

from caput import config

from draco.core import task
from draco.util import tools

from ..core import containers


class Difference(task.SingleTask):
    """ Difference two visibilities of the same shape.

    Attributes
    ----------
    in_place : bool
        Perform the difference in place.
        Default is True.
    """

    in_place = config.Property(proptype=bool, default=True)

    def process(self, sstream1, sstream2):
        """Differences the visibilities in the two
        input sidereal streams.

        Parameters
        ----------
        sstream1 : containers.SiderealStream
            Input sidereal stream.

        sstream2 : containers.SiderealStream
            Input sidereal stream.

        Returns
        -------
        dstream : containers.SiderealStream
            sstream1.vis - sstream2.vis
        """

        # Make sure input sidereal streams are distributed over the same axis
        sstream1.redistribute('freq')
        sstream2.redistribute('freq')

        # Make sure input sidereal streams have the same size
        if sstream1.vis.shape != sstream2.vis.shape:
            InputError("Cannot difference visibilities, incompatible shapes.")

        for ax in sstream1.vis.attrs['axis']:
            if np.any(sstream1.index_map[ax] != sstream2.index_map[ax]):
                InputError("Cannot difference visibilities, incompatible axis.")

        # Either subtract sstream2 from sstream1, or create a new container
        # that contains sstream1 - sstream2.
        if self.in_place:
            dstream = sstream1
        else:
            dstream = containers.empty_like(sstream1)

        # Redefine the tag
        tag1 = sstream1.attrs.get('tag', None)
        tag2 = sstream2.attrs.get('tag', None)

        if (tag1 is not None) and (tag2 is not None):
            dstream.attrs['tag'] = tag1 + '_minus_' + tag2
        elif tag1 is not None:
            dstream.attrs['tag'] = tag1 + '_diff'
        elif tag2 is not None:
            dstream.attrs['tag'] = tag2 + '_diff'
        else:
            dstream.attrs['tag'] = 'diff'

        # Redefine the lsd
        lsd = []
        lsd1 = sstream1.attrs.get('lsd', None)
        if lsd1 is not None:
            if hasattr(lsd1, '__iter__'):
                lsd += [xx for xx in lsd1]
            else:
                lsd.append(lsd1)

        lsd2 = sstream2.attrs.get('lsd', None)
        if lsd2 is not None:
            if hasattr(lsd2, '__iter__'):
                lsd += [xx for xx in lsd2]
            else:
                lsd.append(lsd2)

        dstream.attrs['lsd'] = np.array(lsd)

        # Loop over frequencies and baselines to reduce memory usage
        for lfi, fi in sstream1.vis[:].enumerate(0):
            for lbi, bi in sstream1.vis[:].enumerate(1):

                dstream.vis[fi, bi, :] = (sstream1.vis[fi, bi, :].view(np.ndarray) -
                                          sstream2.vis[fi, bi, :].view(np.ndarray))

                var1 = tools.invert_no_zero(sstream1.weight[fi, bi, :].view(np.ndarray))
                var2 = tools.invert_no_zero(sstream2.weight[fi, bi, :].view(np.ndarray))
                flag = (var1 > 0.0) & (var2 > 0.0)

                dstream.weight[fi, bi, :] = tools.invert_no_zero(var1 + var2) * flag

        # Return the difference
        return dstream




