import numpy as np

from cora.util import hputil
from caput import mpiutil, pipeline, config, mpiarray

from draco.core import containers, task, io


class SimulateCrossTalk(task.SingleTask):
    """Add cross talk to a sidereal stream.
    """

    alpha_file = config.Property(proptype=str, default='./alphafile.txt')

    def setup(self, bt):
        """Setup the simulation.

        Parameters
        ----------
        bt : ProductManager or BeamTransfer
            Beam Transfer maanger.
        """
        self.beamtransfer = io.get_beamtransfer(bt)
        self.telescope = io.get_telescope(bt)

    def process(self, sstream):
        """Some useful comments about stuff.

           Parameters
           ----------
           sstream : SiderealStream object such that sstream.vis[:]
               is a [nfreq, nbaseline, ntime] array.
        """

        # Redistribute over time
        sstream.redistribute('time')

        # Read in telescope system
        bt = self.beamtransfer
        tel = self.telescope

        alphafile = self.alpha_file

        # Construct container and set visibility data
        xstream = containers.SiderealStream(freq=freqmap, ra=ntime, input=feed_index,
                                            prod=tel.uniquepairs, distributed=True, comm=map_.comm)
        xstream.vis[:] = mpiarray.MPIArray.wrap(vis_stream, axis=0)
        xstream.weight[:] = 1.0

        return xstream
