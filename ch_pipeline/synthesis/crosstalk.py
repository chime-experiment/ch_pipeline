import numpy as np
import pickle
from cora.util import hputil
from caput import mpiutil, pipeline, config, mpiarray

from ..core import containers, task

from ch_util import tools

class SimulateCrossTalk(task.SingleTask):
    """Add cross talk to a sidereal stream.
    """

    alpha_file = config.Property(proptype=str, default='./alphafile.pkl')
    delay_file = config.Property(proptype=str, default='./delayfile.pkl')

    def setup(self, bt):
        """Setup the simulation.

        Parameters
        ----------
        bt : ProductManager or BeamTransfer
            Beam Transfer maanger.
        """
        self.telescope = bt.telescope

    def process(self, vis_stream):
        """Some useful comments about stuff.

           Parameters
           ----------
           sstream : SiderealStream object such that sstream.vis[:]
               is a [nfreq, nbaseline, ntime] array.
        """

        # Redistribute over time
        vis_stream.redistribute('ra')

        # Read in telescope system
#        bt = self.beamtransfer
        tel = self.telescope
#        self._tsys = tel.tsys()
        nfreq = tel.nfreq
        nfeeds = tel.nfeed
    
        full_arr = nfeeds*(nfeeds+1)/2
    
        #print nfeeds
#       self._baselines = np.sqrt(tel.baselines[:,0]**2 + tel.baselines[:,1]**2) 
#        self._vis = sstream.vis
#        self._freq = sstream.freq['centre'] * 1e6
        
        alphafile = self.alpha_file
        delayfile = self.delay_file
        
        #alpha_mat = np.ones_like(vis_stream.vis[:,:,0])
        #delay_mat = np.ones_like(vis_stream.vis[:,:,0])
         
        tmp = np.arange(nfreq*full_arr, dtype=np.float)
        tmp = tmp.reshape((nfreq, full_arr))
        
        #print tmp.shape

        alpha_mat = 0.1*np.ones_like(tmp)
        delay_mat = 0.33*np.ones_like(tmp)*10**(-7)
        
    #COMMENT OUT FOR NOW 
        output = open(alphafile, 'wb')
        pickle.dump(alpha_mat, output, -1)
        output.close()

        pkl_file = open(alphafile, 'rb')

        alpha_mat = pickle.load(pkl_file)
    
        pkl_file.close()
        
        output = open(delayfile, 'wb')
        pickle.dump(delay_mat, output, -1)
        output.close()

        pkl_file = open(delayfile, 'rb')

        delay_mat = pickle.load(pkl_file)
    
        pkl_file.close()
        
        #np.savetxt(alphafile, alpha_mat.view(float).reshape(-1, 2))
        #alpha_mat = np.loadtxt(alphafile).view(complex).reshape(-1)
        
        #np.savetxt(delayfile, delay_mat.view(float).reshape(-1, 2))
        #delay_mat = np.loadtxt(delayfile).view(complex).reshape(-1)    
    
        sky_xtalk = np.zeros_like(vis_stream.vis[:,:,:])
        
        #FOR FULL CHIME, WE'LL ONLY WANT TO STORE ALPHA AND DELAY FOR A SINGLE CYLINDER, 256X256 MATRIX.
        
        #print 'start prod loop'

        for visidx, prod in enumerate(vis_stream.index_map['prod']):
            # Great an auto!
            #if prod[0] == prod[1]:
            #    data.vis[:, pi] += self.recv_temp
            
            chan_i = prod[0]
            chan_j = prod[1]
            
            prod_trans = np.transpose(vis_stream.index_map['prod'])
            prod_list = vis_stream.index_map['prod'].tolist()
            
            #print alpha_mat.shape
            
            for l in prod_trans[1,:]: #by row <upper triangle> (j values of baselines Vij)
                
                if ([chan_i, l] in prod_list):
                    idx2 = prod_list.index([chan_i,l])
                    #idx3 = prod_list.index([chan_j,l])
                    #idx2 = tools.cmap(chan_i, l, nfeeds)
                    idx3 = tools.cmap(chan_j, l, nfeeds)
             
                    #print chan_i
                    #print l
                    #print idx2
            
                    #print chan_j
                    #print l
                    #print idx3

                    if (chan_i > l):
                        temp_vis = np.conjugate(vis_stream.vis[:, idx2, :])
                    else:
                        temp_vis = vis_stream.vis[:, idx2, :]

                    if (chan_j > l):
                        alpha_val = np.conjugate(alpha_mat[:,idx3])
                        delay_val = np.conjugate(delay_mat[:,idx3])
                    else:
                        alpha_val = alpha_mat[:,idx3]
                        delay_val = delay_mat[:,idx3]
                        
                    alpha_val = alpha_val[:, np.newaxis]
                    delay_val = delay_val[:, np.newaxis]

                    sky_xtalk[:, visidx, :] += np.conjugate(alpha_val * delay_val) * temp_vis
                    
            #print visidx

            for k in prod_trans[0,:]: #by column <upper triangle> (i values of baselines Vij)
                
                if ([k, chan_j] in prod_list):
                    idx2 = prod_list.index([k, chan_j])
                    #idx3 = prod_list.index([chan_i,k])
                    #idx2 = tools.cmap(k, chan_j, nfeeds)
                    idx3 = tools.cmap(chan_i, k, nfeeds)

                    if k > chan_j:
                        temp_vis = np.conjugate(vis_stream.vis[:, idx2, :])
                    else:
                        temp_vis = vis_stream.vis[:, idx2, :]

                    if chan_i > k:
                        alpha_val = np.conjugate(alpha_mat[:,idx3])
                        delay_val = np.conjugate(delay_mat[:,idx3])
                    else:
                        alpha_val = alpha_mat[:,idx3]
                        delay_val = delay_mat[:,idx3]
                        
                    alpha_val = alpha_val[:, np.newaxis]
                    delay_val = delay_val[:, np.newaxis]

                    sky_xtalk[:, visidx, :] += np.conjugate(alpha_val * delay_val) * temp_vis
            
            #print visidx
   
        #vis_stream['vis'][:] = sky_xtalk[:] 
                                        
        vis_stream.vis[:] = sky_xtalk[:]
                                        
        # Construct container and set visibility data
        #xstream = containers.SiderealStream(freq=freqmap, ra=ntime, input=feed_index,
        #                                    prod=tel.uniquepairs, distributed=True, comm=map_.comm)
        #vis_stream.vis[:] = mpiarray.MPIArray.wrap(sky_xtalk, axis=0)
        vis_stream.weight[:] = 1.0

        return vis_stream