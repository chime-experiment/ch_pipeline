"""
=================================================
Tasks for Cross Talk Model (:mod:`~ch_pipeline.xtalk`)
=================================================

.. currentmodule:: ch_pipeline.xtalk

Tasks for adding cross talk to the data. 

Tasks
=====

.. autosummary::
    :toctree: generated/

    CrossTalkSameCylinder
"""

import numpy as np

from caput import pipeline
from caput import config
from caput import mpidataset, mpiutil

from ch_util import tools
import scipy.constants as c
import datetime


class CrossTalkSameCylinder(pipeline.TaskBase):
    """Simulating Cross Talk models

    Attributes
    ----------
    t_receiver : bool, optional
        Cross talk due to receiver temperature (default = True)
    t_sky : bool, optional
        Cross talk due to sky tempereature (default = True)
    bounce : int, optional
        Number of bounces (default 0)
    """
    set_50K = config.Property(proptype=bool, default=True)
    t_receiver = config.Property(proptype=bool, default=True)
    t_sky = config.Property(proptype=bool, default=True)
    bounce = config.Property(proptype=int, default=0)

    def setup(self, telescope, inputmap):
        """ Setup the simulation for crosstalk.
 
        Parameters
        ----------
        telescope : TransitTelescope
        	Telescope object.
        inputmap : Feedmap
        	List of correlator inputs.
        """
        self.telescope = telescope
        pol = tools.get_feed_polarisations(inputmap)
        self._pol_pairs = get_pol_pairs(pol)

    def next(self, sstream):    
	""" Simulate a SiderealStream (or Timestream) with Cross talk.
	
	Returns
	-------
	sstream_xtalk : SiderealStream (or Timestream) with Cross talk.
	"""
        tel = self.telescope
        self._tsys = tel.tsys()
        self._nfreq = tel.nfreq
        self._nfeeds = tel.nfeed
       	self._baselines = np.sqrt(tel.baselines[:,0]**2 + tel.baselines[:,1]**2) 
	self._vis = sstream.vis
        self._freq = sstream.freq['centre'] * 1e6

        xtalk = self.get_xtalk()

        sstream['vis'][:] = (sstream['vis'][:] + xtalk)

        return sstream

    def get_coeffs(self, alpha_list):
        """Figure out which visibilities are nearest neighbor, 2 feeds
        apart and setting coupling coefficient for each baseline seperation.
	(crosstalk for cross polarization not yet implemented.)
	
	Parameters
	----------
	alpha_list : List of coupling coefficients.
		[Same feed, nearest neighbor, 2 feeds apart, ...]
	Returns
	-------
	alpha_arr : numpy.array of dtype ('index', 'alpha')
		index: tuple of integers, 
			prod_map indices of coupling factor between feeds i, j
		alpha: float / or complex, coupling factor. 
			(currently hard coded in methode get_xtalk.) 
	"""

        alpha_struct = []

	nprod = self._nfeeds * (self._nfeeds + 1) / 2
     	self._prod = np.empty(nprod, dtype=[('input_a', np.uint16),
                                     ('input_b', np.uint16)])
     	for i in range(nprod):
            self._prod[i]['input_a'] = tools.icmap(i, self._nfeeds)[0]         
	    self._prod[i]['input_b'] = tools.icmap(i, self._nfeeds)[1]

        for i in range (len(alpha_list)):
            if alpha_list[i] == 0.0:
                continue
            else:
                for j in range(len(self._baselines)):
                    if np.round(abs(self._baselines[j] / 0.3048)).astype(int) == i and self._pol_pairs[j,0] == self._pol_pairs[j,1]:
                        alpha_struct.append({'index': j, 'alpha': alpha_list[i]})

        alpha_arr = np.zeros(len(alpha_struct), np.dtype([('index', np.int16,
            (2,)), ('alpha', np.complex64)]))
        
        for i in range(alpha_arr.shape[0]):
            idx = alpha_struct[i]['index']
            chan_i, chan_j = self._prod[idx][0], self._prod[idx][1]
            alpha_arr[i]['index'][0], alpha_arr[i]['index'][1] = chan_i, chan_j
            alpha_arr[i]['alpha'] = alpha_struct[i]['alpha']
        
        return alpha_arr


    def get_xtalk(self):
        """ Get the receiver and/or sky Cross talk given an input model.
	Parameters
	----------
	nbounce : integer, number of bounces to consider.
		(options 0, 1, 2; default:0)
	t_sky : bool, if sky cross talk is considered. (default: True).
	t_receiver : bool, if t_receiver cross talk is considered. (default: True).
	set_50K : bool, if 50K are added to autocorrelations. (default: True).
	
	Returns
	-------
	xtalk : numpy.array in shape of visibilties. 
		contains all crosstalk terms due to input model.
	"""
        nbounce = self.bounce
	t_sky = self.t_sky
	t_receiver = self.t_receiver
	set_50K = self.set_50K

        if nbounce == 0:
            
            alpha_list = [0, 0.15]
            alpha = self.get_coeffs(alpha_list)
            path = self._baselines
            phase_factor = _get_phase(self._freq, path)
            alpha_dot_phase = _get_alpha_dot_phase(alpha, phase_factor,
                    self._nfeeds)
            print "path is no bounce"

        elif nbounce == 1:
            
            # feed to focal line assuming 5m
            alpha_list = [0.15 for i in range(self._nfeeds)]
            alpha = self.get_coeffs(alpha_list)
            path = 2 * np.sqrt((self._baselines/2)**2 + 5**2) 
            phase_factor = _get_phase(self._freq, path)
            alpha_dot_phase = _get_alpha_dot_phase(alpha, phase_factor,
                    self._nfeeds)
            print "path is one bounce"

        elif nbounce == 2:
            
            alpha_list = [0.08 for i in range(self._nfeeds)]
            alpha = self.get_coeffs(alpha_list)
            path = 4 * np.sqrt((self._baselines/2)**2 + 5**2)
            phase_factor = _get_phase(self._freq, path)
            alpha_dot_phase = _get_alpha_dot_phase(alpha, phase_factor,
                    self._nfeeds)
            print "path is two bounces" 
       

        if (t_sky and t_receiver and set_50K) is True:
            trec = _set_treceiver(self._vis, self._tsys, self._nfeeds)
            vis_and_tsys = self._vis + trec
            xtalk = _sparse_mult(alpha_dot_phase, vis_and_tsys, self._nfeeds)
       	    xtalk = xtalk + trec
	    print "Special case: added Tsys to Vis and then calculated overall crosstalk"

        elif (t_sky and t_receiver) is True and set_50K is False:
            xtalk_sky = _sparse_mult(alpha_dot_phase, self._vis, self._nfeeds)
            xtalk_trec =  _sparse_mult_tsys(alpha_dot_phase, self._tsys,
                    self._vis, self._nfeeds)
            xtalk = xtalk_sky + xtalk_trec
        
        elif (t_sky and set_50K) is True and t_receiver is False:
            trec = _set_treceiver(self._vis, self._tsys, self._nfeeds)
            xtalk_sky = _sparse_mult(alpha_dot_phase, self._vis, self._nfeeds)
            xtalk = trec + xtalk_sky

        elif (t_receiver and set_50K) is True and t_sky is False:
            trec = _set_treceiver(self._vis, self._tsys, self._nfeeds)
            xtalk_trec =  _sparse_mult_tsys(alpha_dot_phase, self._tsys,
                    self._vis, self._nfeeds)
            xtalk = trec + xtalk_trec

        elif t_receiver is True and (t_sky and set_50K) is False:
            xtalk = _sparse_mult_tsys(alpha_dot_phase, self._tsys, self._vis,
                    self._nfeeds)

        elif t_sky is True and (t_receiver and set_50K) is False:
            xtalk = _sparse_mult(alpha_dot_phase, self._vis, self._nfeeds)
        
        
        print "DONE!"


        return xtalk

def _sparse_mult(alpha_dot_phase, vis, nfeeds):
    """ Sparse multiplication of alpha_dot_phase with visibilties"""
 
    sky_xtalk = np.zeros_like(vis) 
    print "Calculating sky xtalk"
    n_alpha = alpha_dot_phase.shape[1]
    
    for i in range(nfeeds):
        for j in range(i, nfeeds):
            visidx = tools.cmap(i, j, nfeeds)
             
            for a in range(n_alpha):
                chan_i, chan_j = alpha_dot_phase[0,a]['index'][0], alpha_dot_phase[0, a]['index'][1]
                alpha_iter = alpha_dot_phase[:, a, np.newaxis]['alpha_dot_phase'] 
                
                if chan_i == i:
                    idx2 = tools.cmap(chan_j, j, nfeeds)
                    if chan_j > j:
                        temp_vis = np.conjugate(vis[:,idx2,:])
                    else: 
                        temp_vis = vis[:,idx2,:]
                    sky_xtalk[:, visidx, :] += alpha_iter * temp_vis
                
                if chan_i == j:
                    idx2 = tools.cmap(i, chan_j, nfeeds)
                    if i > chan_j:
                        temp_vis = np.conjugate(vis[:, idx2, :])
                    else: 
                        temp_vis = vis[:,idx2,:]
                    sky_xtalk[:, visidx, :] += np.conjugate(alpha_iter) * temp_vis
                
                if chan_j == j:
                    idx2 = tools.cmap(i, chan_i, nfeeds)
                    if i > chan_i:
                        temp_vis = np.conjugate(vis[:,idx2,:])
                    else: 
                         temp_vis = vis[:,idx2,:]
                    sky_xtalk[:, visidx, :] += np.conjugate(alpha_iter) * temp_vis
                
                if chan_j == i:
                    idx2 = tools.cmap(chan_i, j, nfeeds)
                    if chan_i > j:
                        tmp_vis = np.conjugate(vis[:,idx2,:])
                    else: 
                        temp_vis = vis[:,idx2,:]
                    sky_xtalk[:, visidx, :] += alpha_iter * temp_vis
    
    return sky_xtalk  

def _set_treceiver(vis, tsys, nfeeds):
    """ Setting temperature of receivers in autocorrelations."""
    
    print "Setting T receiver" 
    treceiver = np.zeros_like(vis)
    
    for i in range(nfeeds):
        for j in range(i, nfeeds):
            if (i == j):    
                idx = tools.cmap(i, i, nfeeds)
                treceiver[:,idx,:] = tsys[:,np.newaxis]
    
    return treceiver

def _sparse_mult_tsys(alpha_dot_phase, tsys, vis, nfeeds):
    """ 
    Parameters
    ----------
    alpha_dot_phase: 2D np.array of dtype ('index', 'alpha_dot_phase') and 
	shape (freq,prod). Is sparse.
    tsys: 2D array of shape (freq,prod). Is sparse.

    Returns
    -------
    xtalk: 2D np.array, xtalk due to receiver temperature. Full matrix of shape
    (freq, prod)
    """
    xtalk = np.zeros_like(vis)
    for a in range(alpha_dot_phase.shape[1]):
        chan_i, chan_j = alpha_dot_phase['index'][0,a][0], alpha_dot_phase['index'][0,a][1]
	idx = tools.cmap(chan_i, chan_j, nfeeds)
        xtalk[:, idx, :] = (alpha_dot_phase['alpha_dot_phase'][:, a, np.newaxis] * 
                tsys[:, np.newaxis] + 
                np.conjugate(alpha_dot_phase['alpha_dot_phase'][:, a, np.newaxis]) *
                tsys[:,np.newaxis])

    print "Calculating cross talk due to t_receiver"""
    
    return xtalk 

def _get_phase(nu, dij):
    """ Return  phase due to a path length difference between 2 feeds
    as a function of frequency spectrum.

    Parameters
    ----------
    nu : array_like
         The input frequencies.
    dij : array_like
         The path length difference.

    Returns
    -------
    phase : 2D np.ndarray
            The phase due to the path length difference.
    """

    phase = np.zeros((nu.shape[0], dij.shape[0]), dtype=np.complex64)
    time_lag = dij / c.c

    for i in range(nu.shape[0]):
        phase[i, :] = np.exp(2.0J * np.pi * nu[i] * time_lag)

    return phase

def _get_alpha_dot_phase(alpha, phase, nfeeds):
    """ Return the product between alpha and phase as a function of 
    frequency. Sparse matrix multiplication

    Parameter
    ---------
    alpha : np.array of dtype ('index', 'alpha') 
	'index': tuple of integers, 
    		prod_map indices of coupling factor between feeds i, j        
	'alpha' : complex, coupling factor. Is sparse.
    phase_factor: 2D np.array of (freq, prod). Full matrix. 
    
    Returns
    ------
    alpha_dot_phase: np.array of dtype ('index', 'alpha')
	'index' : tuple of integers,
		prod_map indices of coupling factor between feeds i, j
        'alpha_dot_phase' : complex, coupling factor including phase
                factor. Is sparse.
    """
    alpha_dot_phase = np.zeros((phase.shape[0], len(alpha)), np.dtype([('index',
                        np.int16, (2,)),('alpha_dot_phase', np.complex64)]))
    for v in range(phase.shape[0]):
        for i in range(len(alpha)):
            chan_i, chan_j = alpha[i]['index'][0],alpha[i]['index'][1]
            idx = tools.cmap(chan_i, chan_j, nfeeds)
            alpha_dot_phase[v, i]['index'][0], alpha_dot_phase[v, i]['index'][1] = chan_i, chan_j
            alpha_dot_phase[v, i]['alpha_dot_phase'] = (alpha[i]['alpha'] *
                    phase[v, idx])

    return alpha_dot_phase

def get_pol_pairs(pol):
    """
    Gets the polarization pair for each correlation product.
    Can be one of 'S''S', 'E''E', 'S''E', 'E''S'.
    
    Parameters
    ----------
    pol: 1D array of type 'string' with the polarizations as returned from
            tools.get_feed_polarisation from database.

    Returns
    -------
    pack_pol: the polarization pairs for each correlation product.
                pol_pairs is same shape as product axis.
    """

    npol = pol.shape[0]
    pol_pairs = np.zeros((npol,npol,2), dtype=np.str)
    
    for i in range(npol):
        for j in range(npol):
            pol_pairs[i,j,0] = pol[i]
            pol_pairs[i,j,1] = pol[j]

    pack_pol = tools.pack_product_array(pol_pairs[np.newaxis,:,:,:])
    pack_pol = pack_pol[0,:,:]

    return pack_pol

