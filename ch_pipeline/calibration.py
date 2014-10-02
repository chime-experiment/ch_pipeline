"""
=======================================================
Tasks for Calibration (:mod:`~ch_pipeline.calibration`)
=======================================================

.. currentmodule:: ch_pipeline.calibration

Tasks for calibrating the data.

Tasks
=====

.. autosummary::
    :toctree: generated/

    PointSourceCalibration
"""
import os

import numpy as np
from scipy import interpolate
#import matplotlib.pyplot as plt

from caput import pipeline
from caput import config

from ch_util import andata
from ch_util import tools 
from ch_util import ephemeris

import misc_data_io as misc
#import gain_sol_transit as gsol # Temp until the eigensolver is in ch_util. 
# Or should the gainsol code go here?

def gen_corr_matrix(data, nfeed, feed_loc=False): # Should probably be in ch_util.tools 
     """Generates Hermitian (nfeed, nfeed) correlation matrix from unique correlations

     Parameters
     ----------
     data: (nfreq, ncorr, ntimes) np.complex128 arr
          Visibility array to be decomposed
     nfeed: int 
          Number of feeds (duhhh)

     Returns
     -------
     Hermitian correlation matrix
     """

     if not feed_loc:
          feed_loc=range(nfeed)
          
     corr_mat = np.zeros((len(feed_loc), len(feed_loc)), np.complex128)

     for ii in range(len(feed_loc)):
          for jj in range(ii, len(feed_loc)):
               corr_mat[ii, jj] = data[tools.cmap(feed_loc[ii], feed_loc[jj], nfeed)]
               corr_mat[jj, ii] = np.conj(data[tools.cmap(feed_loc[ii], feed_loc[jj], nfeed)])

     return corr_mat

def solve_gain(data, nfeed, feed_loc=False):
     """
     Steps through each time/freq pixel, generates a Hermitian (nfeed,nfeed)
     matrix and calculates gains from its largest eigenvector.

     Parameters
     ----------
     data: (nfreq, ncorr, ntimes) np.complex128 arr
          Visibility array to be decomposed
     nfeed: int 
          Number of feeds in configuration
     feed_loc: list
          Which feeds to include 

     Returns
     -------
     gain_arr: (nfreq, nfeed, ntimes) np.complex128
          Gain solution for each feed, time, and frequency

     """
     if not feed_loc:
          feed_loc=range(nfeed)     

     gain_arr = np.zeros([data.shape[0], len(feed_loc), data.shape[-1]], np.complex128)

     for nu in range(data.shape[0]):
          if (nu%64)==0:
               print "Freq %d" % nu
          for tt in range(data.shape[-1]):
               corr_arr = gen_corr_matrix(data[nu, :, tt], nfeed, feed_loc=feed_loc)
               corr_arr[np.diag_indices(len(corr_arr))] = 0.0

               evl, evec = np.linalg.eigh(corr_arr) 
               gain_arr[nu, :, tt] = evl[-1]**0.5 * evec[:, -1]

     return gain_arr

def interp_gains(trans_times, gain_mat, times, axis=-1):
     """ Linearly interpolates gain solutions in 
     sidereal day.
     """
     f = interpolate.interp1d(trans_times, gain_mat, kind='quadratic', axis=axis)
     return f(times)

"""def gen_waterfall(arr):
     plt.figure()
     plt.subplot(311)
     plt.imshow(arr.real[:, 0], aspect='auto', interpolation='nearest', vmax=10*np.std(arr.real[:,0]))
     plt.subplot(312)
     [plt.plot(arr[nu])[0] for nu in np.round(np.linspace(100,1020, 20))]
     plt.subplot(313)
     plt.plot(arr[205, 5])
     plt.savefig('outimage.png')
     pass
"""
def get_uv(corrs):
     freq = np.linspace(800, 400, 1024)  
     feed_loc = np.loadtxt('/home/k/krs/connor/feed_loc_layout42.txt')

     d_EW, d_NS = misc.calc_baseline(feed_loc)[:2]
     u = d_EW[np.newaxis, corrs, np.newaxis] * freq[:, np.newaxis, np.newaxis] * 1e6 / (3e8)
     v = d_NS[np.newaxis, corrs, np.newaxis] * freq[:, np.newaxis, np.newaxis] * 1e6 / (3e8)

     return u, v

def fringestop(data, corrs, RA, obj=ephemeris.CasA):
     data = data - np.median(data, axis=-1)[:, :, np.newaxis]
      
     ha = np.deg2rad(RA[np.newaxis, np.newaxis, :]) - obj.ra
     u, v = get_uv(corrs)
     phase = fringestop_phase(ha, np.deg2rad(chime_lat), obj.dec, u, v)
     return data * phase

def fringestop_phase(ha, lat, dec, u, v):
     """Return the phase required to fringestop. All angle inputs are radians. 

     Parameter
        ---------
     ha : array_like
          The Hour Angle of the source to fringestop too.
     lat : array_like
          The latitude of the observatory.
     dec : array_like
          The declination of the source.
     u : array_like
          The EW separation in wavelengths (increases to the E)
     v : array_like
          The NS separation in wavelengths (increases to the N)
      
     Returns
     -------
     phase : np.ndarray
        The phase required to *correct* the fringeing. Shape is
         given by the broadcast of the arguments together.
     """
        
     uhdotn = np.cos(dec) * np.sin(-ha)
     vhdotn = np.cos(lat) * np.sin(dec) - np.sin(lat) * np.cos(dec) * np.cos(-ha)
     phase = uhdotn * u + vhdotn * v

     return np.exp(2.0J * np.pi * phase)

     

class PointSourceCalibration(pipeline.TaskBase):

#    files = config.Property(proptype=list)

    def setup(self, files):
        """ 
        
        """
        xfeeds = [0,1,2,3,12,13,14,15] # Assumed to be layout 42
        yfeeds = [4,5,6,7,8,9,10,11]
        self.nfeed = 16
        self.gain_mat = []
        self.trans_times = []

        k=0
        for f in files:
          print "Starting with", f
          data = andata.AnData.from_acq_h5(f)
          times = data.timestamp
          self.nfreq = data.nfreq
          self.ntime = data.ntime
          self.trans_times.append(ephemeris.transit_times(
                    ephemeris.CasA, data.timestamp[0]))
          
          #take only 5 minutes around transit
          times_ind = np.where((times > self.trans_times[k] - 100.) & (times < self.trans_times[k] + 100.))
          vis = data.vis[..., times_ind[0]]
          del data

          print "Starting x-decomp on corrs:", xfeeds
          gain_arr_x = solve_gain(vis, self.nfeed, feed_loc=xfeeds)

          print "Starting y-decomp on corrs:", yfeeds
          gain_arr_y = solve_gain(vis, self.nfeed, feed_loc=yfeeds)

          # Use ones since we'll be dividing in PointSourceCalibration.next
          gains = np.ones([self.nfreq, self.nfeed], np.complex128) 

          gains[:, xfeeds] = np.median(gain_arr_x, axis=-1) # Take time avg of gains solution
          gains[:, yfeeds] = np.median(gain_arr_y, axis=-1)
          print gains[gains < 0.0]

          gains *= np.sign(gains[:, 0, np.newaxis])

          print "Computing gain matrix for sidereal day %d" % k
          self.gain_mat.append((gains[:, :, np.newaxis] * np.conj(gains[:, np.newaxis]))[...,np.newaxis])
          k+=1

        self.gain_mat = np.concatenate(self.gain_mat, axis=-1)
        self.gain_mat = self.gain_mat[:, np.triu_indices(self.nfeed)[0], np.triu_indices(self.nfeed)[1]]
        self.trans_times = np.concatenate(self.trans_times)
        

    def next(self, data):

        times = data.timestamp
        trans_cent = ephemeris.transit_times(ephemeris.CasA, times[0])

        # Now try to get 3 transit calibrations on which to do quadratic interp
        ct = np.where(self.trans_times==trans_cent)[0]
        trans_times = self.trans_times[ct-1:ct+2]

        print trans_times
        gain_mat_full = interp_gains(trans_times, self.gain_mat, times)
     
        calibrated_data = data.vis / gain_mat_full

        return calibrated_data, gain_mat_full


if __name__ == '__main__':
     import h5py
     flpass0 = [u'/scratch/k/krs/jrs65/chime_archive/20140822T193501Z_blanchard_corr/00206207_0000.h5',
                u'/scratch/k/krs/jrs65/chime_archive/20140822T193501Z_blanchard_corr/00288671_0000.h5',
                u'/scratch/k/krs/jrs65/chime_archive/20140827T174947Z_blanchard_corr/00042126_0000.h5']
     
     data_file = flpass0[1]
     
     P = PointSourceCalibration()
     P.setup(flpass0)
     data_obj = andata.AnData.from_acq_h5(data_file)
     datacal, solution = P.next(data_obj)
     del data_obj

     os.system('rm -f /scratch/k/krs/connor/calpass0_out.hdf5')
     ff = h5py.File('/scratch/k/krs/connor/calpass0_out.hdf5','w')
     ff.create_dataset('datacal', data=datacal)
     ff.create_dataset('solution', data=solution)
     ff.close()

"""
Layout 42 Pass 0 Conf B
-----------------------

1948.24 CSS016C0 P1 CXA0202B CANBH0B 29821-0000-0029-C12
1948.24 CSS016C0 P2 CXA0233B CANBH4B 29821-0000-0029-C08
1978.72 CSS016C1 P1 CXA0236C CANBH1B 29821-0000-0029-C13
1978.72 CSS016C1 P2 CXA0010B CANBH5B 29821-0000-0029-C09
2009.20 CSS016C2 P1 CXA0239C CANBH2B 29821-0000-0029-C14
2009.20 CSS016C2 P2 CXA0067B CANBH6B 29821-0000-0029-C10
2039.68 CSS016C3 P1 CXA0006B CANBH3B 29821-0000-0029-C15
2039.68 CSS016C3 P2 CXA0113B CANBH7B 29821-0000-0029-C11
2069.24 CSS019C0 P2 CXA0171B CANBG0B 29821-0000-0029-C04
2069.24 CSS019C0 P1 CXA0200B CANBG4B 29821-0000-0029-C00
2099.72 CSS019C1 P2 CXA0189B CANBG1B 29821-0000-0029-C05
2099.72 CSS019C1 P1 CXA0208B CANBG5B 29821-0000-0029-C01
2130.20 CSS019C2 P2 CXA0185B CANBG2B 29821-0000-0029-C06
2130.20 CSS019C2 P1 CXA0240C CANBG6B 29821-0000-0029-C02
2160.68 CSS019C3 P2 CXA0169B CANBG3B 29821-0000-0029-C07
2160.68 CSS019C3 P1 CXA0063B CANBG7B 29821-0000-0029-C03

"""
