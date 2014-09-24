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

from caput import pipeline
from caput import config
from ch_util import andata
from ch_util import tools 

import gain_sol_transit as gsol # Temp until the eigensolver is in ch_util. 
# Or should the gainsol code go here?

def gen_corr_matrix(data, nfeed, feed_loc=False):
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
        # Need to select subset of this data
        data_xpol, data_ypol = data.vis, data.vis # Need to figure out a way to select pols
        # Will this handle an MPIdataset? 
        nfeed = 16
        gain_arr_x = gsol.solve_gain(data_xpol, nfeed)
        gain_arr_y = gsol.solve_gain(data_ypol, nfeed)

        gain_sol = np.concatenate([gain_arr_x[np.newaxis], gain_arr_y[np.newaxis]])
        return gain_sol

    def next(self, data):

        # Calibrate data as it comes in.

        return calibrated_data
