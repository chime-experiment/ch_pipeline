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
import datetime

from caput import pipeline
from caput import config

from ch_util import andata
from ch_util import tools 
from ch_util import ephemeris
from ch_util import data_index

import misc_data_io as misc

def gen_corr_matrix(data, nfeed, feed_loc=False): 
     """Generates Hermitian (nfeed, nfeed) correlation matrix from unique correlations
     (should probably be in ch_util.tools, or at least not in here)

     Parameters
     ----------
     data : array_like
          Visibility array to be decomposed
     nfeed : int 
          Number of feeds (duhhh)

     Returns
     -------
     corr_mat : array_like
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
     data : array_like
          Visibility array to be decomposed
     nfeed : number 
          Number of feeds in configuration
     feed_loc : list
          Which feeds to include 

     Returns
     -------
     gain_arr : array_like
          Gain solution for each feed, time, and frequency

     """
     if not feed_loc:
          feed_loc=range(nfeed)     

     gain_arr = np.zeros([data.shape[0], len(feed_loc), data.shape[-1]], np.complex128)

     for nu in range(data.shape[0]):

          for tt in range(data.shape[-1]):

               corr_arr = gen_corr_matrix(data[nu, :, tt], nfeed, feed_loc=feed_loc)
               corr_arr[np.diag_indices(len(corr_arr))] = 0.0

               evl, evec = np.linalg.eigh(corr_arr) 
               gain_arr[nu, :, tt] = evl[-1]**0.5 * evec[:, -1]

     return gain_arr

def interp_gains(trans_times, gain_mat, times, axis=-1):
     """ Linearly interpolates gain solutions in sidereal day.
     
     Parameter
     ---------
     trans_times : array_like 
           Unix time of object transit
     gain_mat : array_like
           Array of gains shaped (freq, ncorr, ndays)
     times : array_like
           Timestamps onto which gain solution is interpolated

     Returns
     -------
     Array of interpolated gains
     """
     f = interpolate.interp1d(trans_times, gain_mat, kind='linear', axis=axis)
  
     return f(times)

def list_transits(start_time, end_time, obj=ephemeris.CasA, tdel=600):
     """ Get a list of files that contain point source transits
     
     Parameter
     ---------
     start_time : float or :class:'datetime.datetime'
          Unix/POSIX time or UTC start of desired time range. Optional.
     end_time : float or :class:'datetime.datetime'
          Unix/POSIX time or UTC start of desired time range. Optional.
     obj : ephem.Body. Optional
          Body for which to find transits. 
     tdel : float. Optional
          Total amount of time to include surrounding the transit in sec.
     
     Returns
     -------
     interval_list : :class:'DataIntervalList'
          Search results.      
     """
     f = data_index.Finder()
     f.set_time_range(start_time, end_time)
     f.filter_acqs(data_index.ArchiveInst.name == 'blanchard')
     f.include_transits(obj, time_delta=tdel)

     return f.get_results()

class PointSourceCalibration(pipeline.TaskBase):

    files = config.Property(proptype=list)
    start_time = config.Property(proptype=float)
    end_time = config.Property(proptype=float)

    def setup(self, start_time, end_time):
        """ 
        
        """
        data_dir = '/scratch/k/krs/jrs65/chime_archive/'
        transit_list = list_transits(start_time, end_time, obj=ephemeris.CasA)

        ndays = len(transit_list)

        xfeeds = [0,1,2,3,12,13,14,15] # Assumed to be layout 42
        yfeeds = [4,5,6,7,8,9,10,11]

        self.nfeed = 16
        self.gain_mat = []
        self.trans_times = []

        k=0
        for files in transit_list:

             
             if len(files[0]) > 3:
                  continue # Skip large acqquisitions. Usually high-cadence observations

             flist = []

             for f in files[0]:
                  flist.append(data_dir + f)

             print "Reading in:", flist

             reader = andata.Reader(flist)
             reader.freq_sel = range(1024)
             data = reader.read()
             
             times = data.timestamp
             self.nfreq = data.nfreq
             self.ntime = data.ntime
             self.trans_times.append(ephemeris.transit_times(
                                       ephemeris.CasA, times[0]))
          
             times_ind = np.where((times > self.trans_times[k] - 60.) & (times < self.trans_times[k] + 60.))
             vis = data.vis[..., times_ind[0]]

             del data

             gain_arr_x = solve_gain(vis, self.nfeed, feed_loc=xfeeds)
             gain_arr_y = solve_gain(vis, self.nfeed, feed_loc=yfeeds)
             
             print "Finished eigendecomposition" 
             print ""

             # Use ones since we'll be dividing in PointSourceCalibration.next
             gains = np.ones([self.nfreq, self.nfeed], np.complex128) 

             gains[:, xfeeds] = np.median(gain_arr_x, axis=-1) # Take time avg of gains solution
             gains[:, yfeeds] = np.median(gain_arr_y, axis=-1)

             print "Computing gain matrix for sidereal day %d of %d" % (k+1, ndays)
             self.gain_mat.append((gains[:, :, np.newaxis] * np.conj(gains[:, np.newaxis]))[...,np.newaxis])
             k+=1

        self.gain_mat = np.concatenate(self.gain_mat, axis=-1)
        self.gain_mat = self.gain_mat[:, np.triu_indices(self.nfeed)[0], np.triu_indices(self.nfeed)[1]]
        self.trans_times = np.concatenate(self.trans_times)
                  

    def next(self, data):

        times = data.timestamp
        trans_cent = ephemeris.transit_times(ephemeris.CasA, times[0])
        print trans_cent

        ct = np.where(self.trans_times==trans_cent)[0]

        trans_times = self.trans_times

        ind_cal = np.where((times > trans_times[0]) & (times < trans_times[-1]))[0]
        gain_mat_full = interp_gains(trans_times, self.gain_mat, times[ind_cal])
     
        calibrated_data = data.vis[..., ind_cal] / gain_mat_full

        return calibrated_data, gain_mat_full

    def next_parallel(self, files):
        ts = containers.TimeStream.from_acq_files(files)
        ts.redistribute(0)
        
        freq_low = ts.vis.local_offset[0]
        freq_up = freq_low + ts.vis.local_shape[0]

        times = ts.timestamp 

        trans_cent = ephemeris.transit_times(ephemeris.CasA, times[0])

        ct = np.where(self.trans_times==trans_cent)[0]
        trans_times = self.trans_times[ct-1:ct+1]
        
        ind_cal = np.where((times > trans_times[0]) & (times < trans_times[-1]))[0]
        gain_mat_full = interp_gains(trans_times, self.gain_mat[freq_low:freq_up], times[ind_cal])
        
        calibrated_data = data.vis[..., ind_cal] / gain_mat_full

        return calibrated_data, gain_mat_full
        
        

if __name__ == '__main__':
     import h5py
     flpass0 = [u'/scratch/k/krs/jrs65/chime_archive/20140822T193501Z_blanchard_corr/00206207_0000.h5',
                u'/scratch/k/krs/jrs65/chime_archive/20140822T193501Z_blanchard_corr/00288671_0000.h5',
                u'/scratch/k/krs/jrs65/chime_archive/20140827T174947Z_blanchard_corr/00042126_0000.h5']
     
     start_time, end_time = datetime.datetime(2014, 8, 24), datetime.datetime(2014,8,28)

     data_file = flpass0[1]
     
     print "Attempting to generate cal solutions between", start_time, "and", end_time
     print ""

     P = PointSourceCalibration()
     P.setup(start_time, end_time)

     os.system('rm -f /scratch/k/krs/connor/calpass0_out.hdf5')
     
     print ""
     print "Writing cal sol to file"
     print ""

     ff = h5py.File('/scratch/k/krs/connor/calpass0_out.hdf5','w')     
     ff.create_dataset('solution', data=P.gain_mat)
     
     data_obj = andata.AnData.from_acq_h5(data_file)
     datacal, solution = P.next(data_obj)

     del data_obj

     ff.create_dataset('datacal', data=datacal)
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
