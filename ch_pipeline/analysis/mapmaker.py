"""
========================================================
Map making tasks (:mod:`~ch_pipeline.analysis.mapmaker`)
========================================================

.. currentmodule:: ch_pipeline.analysis.mapmaker

Tools for making maps from CHIME data.

Tasks
=====

.. autosummary::
    :toctree: generated/

    RingMapMaker
"""
import numpy as np
import scipy.constants

from caput import config

from draco.core import task
from draco.core import io
from draco.util import tools

from ch_util import ephemeris
from ch_util import tools as ch_tools

from ..core import containers


class RingMapMaker(task.SingleTask):
    """A simple and quick map-maker that forms a series of beams on the meridian.

    This is designed to run on data after it has been collapsed down to
    non-redundant baselines only.

    Attributes
    ----------
    npix : int
        Number of map pixels in the declination dimension.  Default is 512.

    span : float
        Span of map in the declination dimension. Value of 1.0 generates a map
        that spans from horizon-to-horizon.  Default is 1.0.

    weight : string ('natural', 'uniform', or 'inverse_variance')
        How to weight the non-redundant baselines:
            'natural' - each baseline weighted by its redundancy (default)
            'uniform' - each baseline given equal weight
            'inverse_variance' - each baseline weighted by the weight attribute

    exclude_intracyl : bool
        Exclude intracylinder baselines from the calculation.  Default is False.

    include_auto: bool
        Include autocorrelations in the calculation.  Default is False.
    """

    npix = config.Property(proptype=int, default=512)

    span = config.Property(proptype=float, default=1.0)

    weight = config.Property(proptype=str, default='natural')

    exclude_intracyl = config.Property(proptype=bool, default=False)

    include_auto = config.Property(proptype=bool, default=False)

    def setup(self, tel):
        """Set the Telescope instance to use.

        Parameters
        ----------
        tel : TransitTelescope
        """

        if self.weight not in ['natural', 'uniform', 'inverse_variance']:
            KeyError("Do not recognize weight = %s" % self.weight)

        self.telescope = io.get_telescope(tel)

    def process(self, sstream):
        """Computes the ringmap.

        Parameters
        ----------
        sstream : containers.SiderealStream
            The input sidereal stream.

        Returns
        -------
        rm : containers.RingMap
        """

        # Redistribute over frequency
        sstream.redistribute('freq')
        nfreq = sstream.vis.local_shape[0]

        # Extract the right ascension (or calculate from timestamp)
        ra = sstream.ra if 'ra' in sstream.index_map else ephemeris.lsa(sstream.time)
        nra = ra.size

        # Construct mapping from vis array to unpacked 2D grid
        nprod = sstream.prod.shape[0]
        pind = np.zeros(nprod, dtype=np.int)
        xind = np.zeros(nprod, dtype=np.int)
        ysep = np.zeros(nprod, dtype=np.float)

        for pp, (ii, jj) in enumerate(sstream.prod):

            if self.telescope.feedconj[ii, jj]:
                ii, jj = jj, ii

            fi = self.telescope.feeds[ii]
            fj = self.telescope.feeds[jj]

            pind[pp] = 2 * int(fi.pol == 'S') + int(fj.pol == 'S')
            xind[pp] = np.abs(fi.cyl - fj.cyl)
            ysep[pp] = fi.pos[1] - fj.pos[1]

        abs_ysep = np.abs(ysep)
        min_ysep, max_ysep = np.percentile(abs_ysep[abs_ysep > 0.0], [0, 100])

        yind = np.round(ysep / min_ysep).astype(np.int)

        grid_index = list(zip(pind, xind, yind))

        # Define several variables describing the baseline configuration.
        nfeed = int(np.round(max_ysep / min_ysep)) + 1
        nvis_1d = 2 * nfeed - 1
        ncyl = np.max(xind) + 1
        nbeam = 2 * ncyl - 1

        # Define polarisation axis
        pol = np.array([x + y for x in ['X', 'Y'] for y in ['X', 'Y']])
        npol = len(pol)

        # Create empty array for output
        vis = np.zeros((nfreq, npol, nra, ncyl, nvis_1d), dtype=np.complex128)
        invvar = np.zeros((nfreq, npol, nra, ncyl, nvis_1d), dtype=np.float64)
        weight = np.zeros((nfreq, npol, nra, ncyl, nvis_1d), dtype=np.float64)

        # If natural or uniform weighting was choosen, then calculate the
        # redundancy of the collated visibilities.
        if self.weight != 'inverse_variance':
            redundancy = tools.calculate_redundancy(sstream.input_flags[:],
                                                    sstream.index_map['prod'][:],
                                                    sstream.reverse_map['stack']['stack'][:],
                                                    sstream.vis.shape[1])

            if self.weight == 'uniform':
                redundancy = (redundancy > 0).astype(np.float32)

        # Unpack visibilities into new array
        for vis_ind, (p_ind, x_ind, y_ind) in enumerate(grid_index):

            # Handle different options for weighting baselines
            if self.weight == 'inverse_variance':
                w = sstream.weight[:, vis_ind]

            else:
                w = (sstream.weight[:, vis_ind] > 0.0).astype(np.float32)
                w *= redundancy[np.newaxis, vis_ind]

            # Different behavior for intracylinder and intercylinder baselines.
            if (x_ind == 0):

                if not self.exclude_intracyl:

                    vis[:, p_ind, :, x_ind, y_ind] = sstream.vis[:, vis_ind]
                    vis[:, p_ind, :, x_ind, -y_ind] = sstream.vis[:, vis_ind].conj()

                    invvar[:, p_ind, :, x_ind, y_ind] = sstream.weight[:, vis_ind]
                    invvar[:, p_ind, :, x_ind, -y_ind] = sstream.weight[:, vis_ind]

                    weight[:, p_ind, :, x_ind, y_ind] = w
                    weight[:, p_ind, :, x_ind, -y_ind] = w

            else:

                vis[:, p_ind, :, x_ind, y_ind] = sstream.vis[:, vis_ind]

                invvar[:, p_ind, :, x_ind, y_ind] = sstream.weight[:, vis_ind]

                weight[:, p_ind, :, x_ind, y_ind] = w

        # Remove auto-correlations
        if not self.include_auto:
            weight[..., 0, 0] = 0.0

        # Normalize the weighting function
        coeff = np.full(ncyl, 2.0, dtype=np.float)
        coeff[0] = 1.0
        norm = np.sum(np.dot(coeff, weight), axis=-1)

        weight *= tools.invert_no_zero(norm)[..., np.newaxis, np.newaxis]

        # Construct phase array
        el = self.span * np.linspace(-1.0, 1.0, self.npix)

        vis_pos_1d = np.fft.fftfreq(nvis_1d, d=(1.0 / (nvis_1d * min_ysep)))

        # Create empty ring map
        rm = containers.RingMap(beam=nbeam, el=el, pol=pol, ra=ra,
                                axes_from=sstream, attrs_from=sstream)
        # Add datasets
        rm.add_dataset('rms')
        rm.add_dataset('dirty_beam')

        # Make sure ring map is distributed over frequency
        rm.redistribute('freq')

        # Estimate rms noise in the ring map by propagating estimates
        # of the variance in the visibilities
        rm.rms[:] = np.sqrt(np.sum(np.dot(coeff,
                    tools.invert_no_zero(invvar) * weight**2.0), axis=-1))

        # Loop over local frequencies and fill ring map
        for lfi, fi in sstream.vis[:].enumerate(0):

            # Get the current frequency and wavelength
            fr = sstream.freq[fi]

            wv = scipy.constants.c * 1e-6 / fr

            # Create array that will be used for the inverse
            # discrete Fourier transform in y-direction
            pa = np.exp(-2.0J * np.pi * vis_pos_1d[:, np.newaxis] * el[np.newaxis, :] / wv)

            # Perform inverse discrete fourier transform in y-direction
            # and inverse fast fourier transform in x-direction
            bfm = np.fft.irfft(np.dot(weight[lfi] * vis[lfi], pa), nbeam, axis=2) * nbeam
            sb = np.fft.irfft(np.dot(weight[lfi], pa), nbeam, axis=2) * nbeam

            # Save to container
            rm.map[fi] = bfm
            rm.dirty_beam[fi] = sb

        return rm



class CombinedRingMapMaker(task.SingleTask):
    """A map-maker that forms a single map by combining different beams along the meridian.

    This is designed to run on data after it has been collapsed down to
    non-redundant baselines only.

    Attributes
    ----------
    npix_dec : int
        Number of map pixels in the declination dimension.  Default is 512.

    span_dec : float
        Span of map in the declination dimension. Default is 80. degrees.

    weight : string ('natural', 'uniform', or 'inverse_variance')
        How to weight the non-redundant baselines:
            'natural' - each baseline weighted by its redundancy (default)
            'uniform' - each baseline given equal weight
            'inverse_variance' - each baseline weighted by the weight attribute

    exclude_intracyl : bool
        Exclude intracylinder baselines from the calculation.  Default is False.

    include_auto: bool
        Include autocorrelations in the calculation.  Default is False.
    
    time_window_s: float
        Time window in seconds. For a given transit time, a map is formed in 
        a window of time_window_s seconds (time_window_s/2. at each side of 
        the transit time) and added to the map stack. Default is 900. (15 min 
        at each side of a given transit time)
    
    beam: string ('uniform', 'gaussian')
        Beam profile to weight samples from different RAs into a single pixel
        'uniform' - Uniform beam (boxcar)
        'gaussian' - Gaussian beam
    """

    npix_dec = config.Property(proptype=int, default=512)

    span_dec = config.Property(proptype=float, default=80.0)

    weight = config.Property(proptype=str, default='natural')

    exclude_intracyl = config.Property(proptype=bool, default=False)

    include_auto = config.Property(proptype=bool, default=False)

    time_window_s = config.Property(proptype=float, default=300.)

    beam = config.Property(proptype=str, default='gaussian')

    def setup(self, tel):
        """Set the Telescope instance to use.

        Parameters
        ----------
        tel : TransitTelescope
        """

        if self.weight not in ['natural', 'uniform', 'inverse_variance']:
            KeyError("Do not recognize weight = %s" % self.weight)

        self.telescope = io.get_telescope(tel)

    def process(self, sstream):
        """Computes the ringmap.

        Parameters
        ----------
        sstream : containers.SiderealStream
            The input sidereal stream.

        Returns
        -------
        rm : containers.RingMap
        """

        # Redistribute over frequency
        sstream.redistribute('freq')
        nfreq = sstream.vis.local_shape[0]

        # Extract the right ascension (or calculate from timestamp)
        ra = sstream.ra if 'ra' in sstream.index_map else ephemeris.lsa(sstream.time)
        nra = ra.size

        # Get baseline of each stack product
        baselines = self.telescope.baselines.copy()
        autos_stack_prods = np.sqrt(baselines[:, 0]**2 + baselines[:, 1]**2)==0.0
        intracyl_stack_prods = baselines[:, 0]==0.0
        
        # Get polarization of each stack product. Taken from the RingMapMaker class code
        nstack = sstream.prod.shape[0]
        pol = np.array([x + y for x in ['X', 'Y'] for y in ['X', 'Y']])
        npol = len(pol)
        pol_index = np.zeros(nstack, dtype=np.int) #pol of each vis from to convention above
        
        for sp, (ii, jj) in enumerate(sstream.prod):

            if self.telescope.feedconj[ii, jj]:
                ii, jj = jj, ii

            fi = self.telescope.feeds[ii]
            fj = self.telescope.feeds[jj]

            pol_index[sp] = 2 * int(fi.pol == 'S') + int(fj.pol == 'S')

        # Get window size in number of RA pixels
        delta_ra = 360. / float(nra) # RA spacing in degrees. Assumes unifom spacing
        delta_t = delta_ra  * 240.  # RA spacing in seconds (~240 sec/degree)
        npix_w_side = int((self.time_window_s/2.) / delta_t) # N win pixels at each side of transit
        ha_w_index = np.arange(-npix_w_side, npix_w_side + 1, dtype=np.int32) 
        ha_w = delta_ra * ha_w_index # Window hour angle
        npix_w = len(ha_w) # Number of pixels of time window

        # Declination array
        lat = ephemeris.CHIMELATITUDE # latitude of CHIME array
        dec = np.linspace(lat-self.span_dec/2., lat+self.span_dec/2., self.npix_dec, 
                          endpoint=False)

        # Construct phase array
        # Get local frequencies. Taken from solar module
        sfreq = sstream.vis.local_offset[0]
        efreq = sfreq + nfreq
        f_MHz = sstream.freq[sfreq:efreq]
        # Calculate wavelengths
        wv = scipy.constants.c * 1e-6 / f_MHz
        # u and v arrays, each has dimensions [freq bin, stack prod, time]
        u = baselines[np.newaxis, :, 0, np.newaxis] / wv[:, np.newaxis, np.newaxis]
        v = baselines[np.newaxis, :, 1, np.newaxis] / wv[:, np.newaxis, np.newaxis]
        # Construct window phase array and beam profile
        npix = npix_w * self.npix_dec
        fs_phase = np.zeros((nfreq, nstack, npix), dtype=np.complex128)
        for i, DEC in enumerate(dec): # Fringestop phase at every ra for given DEC
            fs_phase[:, :, i*npix_w:(i+1)*npix_w] = ch_tools.fringestop_phase(
                                                    np.radians(ha_w)[np.newaxis, np.newaxis, :], 
                                                    np.radians(lat), np.radians(DEC), u, v)
        
        # Get Beam profile.
        # beam dimensions according to RingMap container
        if self.beam == 'uniform': # Uniform beam
            beam = np.ones((nfreq, npol, npix_w, self.npix_dec), dtype=np.float64)
        else:#Gaussian beam. Taken from Mateus' Quasar Stack code.
            beam = np.zeros((nfreq, npol, npix_w, self.npix_dec), dtype=np.float64)
            for pp in range(npol):
                beam[:, pp] = self._beamfunc(np.radians(ha_w)[np.newaxis, :, np.newaxis], pp, 
                                             f_MHz[:, np.newaxis, np.newaxis], 
                                             np.radians(dec)[np.newaxis, np.newaxis, :])
        # Normalize beam. Currently the beam is normalized by the sum of the squares of the
        # beam pixel values at each declination (each dec separately). With this normalization,
        # and assuming the true beam, the flux of a point source at transit time should be 
        # correct after map stacking (note I normalize the beam squared because the data 
        # already has one factor of the beam in it)
        beam /= np.sum(beam**2, axis=2)[:, :, np.newaxis, :]

        # Construct weight function.
        # Handle different options for weighting baselines
        if self.weight == 'inverse_variance':
            weight = sstream.weight[:]
        else: # natural or uniform weighting. Need to calculate redundancy
            redundancy = tools.calculate_redundancy(sstream.input_flags[:],
                                                    sstream.index_map['prod'][:],
                                                    sstream.reverse_map['stack']['stack'][:],
                                                    sstream.vis.shape[1])

            if self.weight == 'uniform':
                redundancy = (redundancy > 0).astype(np.float32)

            weight = (sstream.weight[:] > 0.0).astype(np.float32)
            # redundancy has shape [stack prod, time]
            weight *= redundancy[np.newaxis, :, :]
        # Remove intracylinder baselines
        if self.exclude_intracyl:
            weight[:, intracyl_stack_prods] = 0.0
        # Remove auto-correlations
        if not self.include_auto:
            weight[:, autos_stack_prods] = 0.0
        # Normalize weight function. Each polarization separately
        pol_stack_prods = np.zeros((npol, nstack), dtype=bool) # I need this later
        for pp in range(npol): 
            pol_stack_prods[pp] = pol_index==pp # stack products of polarization pp
            norm_pol = np.sum(weight[:, pol_stack_prods[pp]], axis=1)
            weight[:, pol_stack_prods[pp]] *= tools.invert_no_zero(norm_pol)[:, np.newaxis, :]

        # Create empty ring map. Note that the declination goes into el parameter
        rm = containers.RingMap(beam=1, el=dec, pol=pol, ra=ra,
                                axes_from=sstream, attrs_from=sstream)
        # Add datasets
        rm.add_dataset('rms')
        rm.add_dataset('dirty_beam')
        # Make sure ring map is distributed over frequency
        rm.redistribute('freq')

        # Fill map
        # Loop over polarization
        for pp in range(npol): 
            # Estimate rms noise in the ring map by propagating estimates
            # of the variance in the visibilities. Does not include the beam yet
            rm.rms[:, pp] = np.sum(tools.invert_no_zero(sstream.weight[:][:, pol_stack_prods[pp]]) 
                                   * weight[:, pol_stack_prods[pp]]**2.0, axis=1)
            # Loop over local frequencies 
            for lfi, fi in sstream.vis[:].enumerate(0):
                vis = sstream.vis[:][lfi, pol_stack_prods[pp]] # vis data [stack_prod, ra]
                z = fs_phase[lfi, pol_stack_prods[pp]] # fs phase [stack_prod, pixel]
                w = weight[lfi, pol_stack_prods[pp]] # weight [stack_prod, ra]
                # Get map for local field for every ra
                m = np.dot(z.T, vis * w).real # local map [pixel, ra]
                db = np.dot(z.T, w).real # local dirty beam [pixel, ra]
                # place local map into rm.map and multiply by beam
                for ra_index, RA in enumerate(ra):
                    m_ha = m[:, ra_index].reshape(self.npix_dec, npix_w) # local map [dec, ha]
                    db_ha = db[:, ra_index].reshape(self.npix_dec, npix_w)
                    # Find position of local map in rm.map
                    ra_index_range = ra_index - ha_w_index #-ve ha index means RApixel > RA
                    # Wrap ra indices around edges
                    ra_index_range[ra_index_range < 0] += nra
                    ra_index_range[ra_index_range >= nra] -= nra
                    # Add local map to map stack
                    rm.map[fi, pp, ra_index_range, 0, :] += m_ha.T * beam[lfi, pp]
                    rm.dirty_beam[fi, pp, ra_index_range, 0, :] += db_ha.T * beam[lfi, pp]

        return rm


    def _beamfunc(self, ha, pol, freq, dec, zenith=0.70999994):
        """Beam model. Taken from Mateus' Quasar stack code which only works for XX and YY. 
        For cross-polarizations returns the geometric mean of XX and YY beams

        Parameters
        ----------
        ha : array or float
            Hour angle (in radians) to compute beam at.
        freq : array or float
            Frequency in MHz
        dec : array or float
            Declination in radians
        pol : int
            Polarization index. XX : 0, XY: 1, YX: 2, YY : 3
        zenith : float
            Polar angle of the telescope zenith in radians. 
            Equal to pi/2 - latitude
        
        Returns
        -------
        b: array
            Power beam model. Its shape is given by the broadcast of the arguments together.
        """
        def _sig(pp, freq, dec):
            """
            """
            sig_amps = {0:14.87857614, 3:9.95746878}
            return sig_amps[pp]/freq/np.cos(dec)
    
        def _amp(pp, dec, zenith):
            """
            """
            def _flat_top_gauss6(x, A, sig, x0):
                """Flat-top gaussian. Power of 6."""
                return A*np.exp(-abs((x-x0)/sig)**6)
            def _flat_top_gauss3(x, A, sig, x0):
                """Flat-top gaussian. Power of 3."""
                return A*np.exp(-abs((x-x0)/sig)**3)

            prm_ns_x = np.array([9.97981768e-01, 1.29544939e+00, 0.])
            prm_ns_y = np.array([9.86421047e-01, 8.10213326e-01, 0.])

            if pp==0:
                return _flat_top_gauss6(dec - (0.5 * np.pi - zenith), *prm_ns_x)
            elif pp==3:
                return _flat_top_gauss3(dec - (0.5 * np.pi - zenith), *prm_ns_y)

        ha0 = 0.
        if pol in [0, 3]: #co-pol. Use quasar beam model
            return _amp(pol, dec, zenith)*np.exp(-((ha-ha0)/_sig(pol, freq, dec))**2)
        else: # cross-pol. Return geometric mean of XX and YY beams
            return np.sqrt(_amp(0, dec, zenith)*np.exp(-((ha-ha0)/_sig(0, freq, dec))**2) * 
                           _amp(3, dec, zenith)*np.exp(-((ha-ha0)/_sig(3, freq, dec))**2))
            