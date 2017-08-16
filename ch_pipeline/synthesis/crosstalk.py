import numpy as np

from cora.util import hputil
from caput import mpiutil, pipeline, config, mpiarray

from draco.core import containers, task, io

from ch_util import tools

class SimulateCrossTalk(task.SingleTask):
    """Add cross talk to a sidereal stream.
    """

    alpha_file = config.Property(proptype=str, default='./alphafile.txt')
    delay_file = config.Property(proptype=str, default='./delayfile.txt')

    def setup(self, bt):
        """Setup the simulation.

        Parameters
        ----------
        bt : ProductManager or BeamTransfer
            Beam Transfer maanger.
        """
# I don't believe these are needed
        
#        self.beamtransfer = io.get_beamtransfer(bt)
        self.telescope = io.get_telescope(bt)

    def process(self, vis_stream):
        """Some useful comments about stuff.

           Parameters
           ----------
           sstream : SiderealStream object such that sstream.vis[:]
               is a [nfreq, nbaseline, ntime] array.
        """

        # Redistribute over time
        sstream.redistribute('time')

        # Read in telescope system
#        bt = self.beamtransfer
        tel = self.telescope
#        self._tsys = tel.tsys()
#        self._nfreq = tel.nfreq
        nfeeds = tel.nfeed
#       self._baselines = np.sqrt(tel.baselines[:,0]**2 + tel.baselines[:,1]**2) 
#        self._vis = sstream.vis
#        self._freq = sstream.freq['centre'] * 1e6
        
        alphafile = self.alpha_file
        delayfile = self.delay_file
        
        alpha_mat = np.ones_like(vis_stream.vis[0,:,0])
        delay_mat = np.ones_like(vis_stream.vis[0,:,0])
        
        numpy.savetxt(alphafile, alpha_mat.view(float).reshape(-1, 2))
        alpha_mat = numpy.loadtxt(alphafile).view(complex).reshape(-1)
        
        numpy.savetxt(delayfile, delay_mat.view(float).reshape(-1, 2))
        delay_mat = numpy.loadtxt(delayfile).view(complex).reshape(-1)
    
        sky_xtalk = np.zeros_like(vis_stream.vis)
        
        #FOR FULL CHIME, WE'LL ONLY WANT TO STORE ALPHA AND DELAY FOR A SINGLE CYLINDER, 256X256 MATRIX.

        # Iterate over the products to find the auto-correlations and add the noise into them
        for visidx, prod in enumerate(data.index_map['prod']):
            # Great an auto!
            #if prod[0] == prod[1]:
            #    data.vis[:, pi] += self.recv_temp
            chan_i = prod[0]
            chan_j = prod[1]

            for l in range(chan_i, nfeeds): #by row <upper triangle>
                idx2 = tools.cmap(chan_i, l, nfeeds)
                idx3 = tools.cmap(chan_j, l, nfeeds)

                if chan_i > l:
                    temp_vis = np.conjugate(vis[:, idx2, :]
                else:
                    temp_vis = vis[:, idx2, :]

                if chan_j > l:
                    alpha_val = np.conjugate(alpha_mat[idx3])
                    delay_val = np.conjugate(delay_mat[idx3])
                else:
                    alpha_val = alpha_mat[idx3]
                    delay_val = delay_mat[idx3]

                sky_xtalk[:, visidx, :] += np.conjugate(alpha_val * delay_val) * temp_vis

            for k in range(chan_j): #by column <upper triangle>
                idx2 = tools.cmap(k, chan_j, nfeeds)
                idx3 = tools.cmap(chan_i, k, nfeeds)

                if k > chan_j:
                    temp_vis = np.conjugate(vis[:, idx2, :]
                else:
                    temp_vis = vis[:, idx2, :]

                if chan_i > k:
                    alpha_val = np.conjugate(alpha_mat[idx3])
                    delay_val = np.conjugate(delay_mat[idx3])
                else:
                    alpha_val = alpha_mat[idx3]
                    delay_val = delay_mat[idx3]

                sky_xtalk[:, visidx, :] += np.conjugate(alpha_val * delay_val) * temp_vis
   
        #vis_stream['vis'][:] = sky_xtalk[:] 
                                        
        vis_stream.vis = sky_xtalk
                                        
        # Construct container and set visibility data
        #xstream = containers.SiderealStream(freq=freqmap, ra=ntime, input=feed_index,
        #                                    prod=tel.uniquepairs, distributed=True, comm=map_.comm)
        #vis_stream.vis[:] = mpiarray.MPIArray.wrap(sky_xtalk, axis=0)
        vis_stream.weight[:] = 1.0

        return vis_stream

