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
import numpy as np

from caput import pipeline
from caput import config
from ch_util import andata
from ch_util import tools 

import gain_sol_transit as gsol # Temp until the eigensolver is in ch_util. 
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

def interp_gains(transit_times, gain_mat):
     """ Linearly interpolates gain solutions in 
     sidereal day.
     """
     f = interpolate.interp1d(transit_times, gain_mat, axis=-1)
     return f(times)
     

class PointSourceCalibration(pipeline.TaskBase):

    def setup(self, files):
        """ Derive calibration solution from input
        start with the calibration solution itself
        How much time do I want from each transit? 
        Enough to actually get the transit at all frequencies 
        Do I want to fringestop? Yes probably yes.
        Really do need the layout information. 

        Need to know feed layout. Which are P0 which are P1? 

        This is parallelized over frequency, right?

        Should the linear interpolation go in here? I'll have 
        one gain per feed per frequency per sidereal day. 
        """
        #ts = containers.TimeStream.from_acq_files(files)
        data = andata.AnData.from_acq_h5(files)
        # Need to select subset of this data with only, say, 10 minutes of data. 
        # Need to figure out a way to select pols
        # Will this handle an MPIdataset? 

        xfeeds = [0,1,2,12,13,14,15]
        yfeeds = [4,5,6,8,9,10,11]
        nfeed = 16

        print "Starting x-decomp on corrs:", xfeeds
        gain_arr_x = solve_gain(data.vis, nfeed, feed_loc=xfeeds)

        print "Starting y-decomp on corrs:", yfeeds
        gain_arr_y = solve_gain(data.vis, nfeed, feed_loc=yfeeds)

        # Use ones since we'll be dividing in PointSourceCalibration.next
        gains = np.ones([data.nfreq, nfeed, data.ntime], np.complex128) 

        gains[:, xfeeds] = gain_arr_x
        gains[:, yfeeds] = gain_arr_y
        
        print "Computing gain matrix for sidereal day"
        self.gain_mat = gains[:, :, np.newaxis] * np.conj(gains[:, np.newaxis])

    def next(self, data):
        # Should already have N_sidereal gain matrices, doens't do the linear interpolation 
        # until here. 
        # Calibrate data as it comes in.
         

        return calibrated_data
