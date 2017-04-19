"""
===============================================================
Pathfinder Telescope Model (:mod:`ch_pipeline.core.pathfinder`)
===============================================================

.. currentmodule:: ch_pipeline.core.pathfinder

A model for the CHIME Pathfinder telescope. This attempts to query the
configuration db (:mod:`~ch_analysis.pathfinder.configdb`) for the details of
the feeds and their positions.

Classes
=======

.. autosummary::
    :toctree: generated/

    CHIMEPathfinder
"""

import numpy as np
import h5py
import healpy
import warnings

from caput import config, mpiutil

from drift.core import telescope
from drift.telescope import cylbeam

from cora.util import coord, hputil

from ch_util import ephemeris, tools

from scipy.interpolate import griddata

class CHIMEPathfinder(telescope.PolarisedTelescope):
    """Model telescope for the CHIME Pathfinder.

    This class currently uses a simple Gaussian model for the primary beams.

    Attributes
    ----------
    layout : datetime or int
        Specify which layout to use.
    correlator : string
        Restrict to a specific correlator.
    skip_non_chime : boolean
        Ignore non CHIME feeds in the BeamTransfers.
    redundant : boolean
        Use only redundant baselines (default is False).
    use_pathfinder_freq: boolean
        Use the pathfinder channelization of 1024 frequencies between 400 and
        800 MHz.  Setting this to True also enables the specification of a
        subset of these frequencies through the four attributes below.  Default
        is True.
    channel_bin : int, optional
        Number of channels to bin together. Must exactly divide the total
        number. Binning is performed prior to selection of any subset. Default
        is 1.
    freq_physical : list, optional
        Select subset of frequencies using a list of physical frequencies in
        MHz. Finds the closests pathfinder channel.
    channel_range : list, optional
        Select subset of frequencies using a range of frequency channel indices,
        either [start, stop, step], [start, stop], or [stop] is acceptable.
    channel_index : list, optional
        Select subset of frequencies using a list of frequency channel indices.
    input_sel : list, optional
        Select a reduced set of feeds to use. Useful for generating small
        subsets of the data.
    """

    # Configure which feeds and layout to use
    layout = config.Property(default=None)
    correlator = config.Property(proptype=str, default=None)
    skip_non_chime = config.Property(proptype=bool, default=False)

    # Redundancy settings
    redundant = config.Property(proptype=bool, default=False)

    # Configure frequency properties
    use_pathfinder_freq = config.Property(proptype=bool, default=True)
    channel_bin = config.Property(proptype=int, default=1)
    freq_physical = config.Property(proptype=list, default=[])
    channel_range = config.Property(proptype=list, default=[])
    channel_index = config.Property(proptype=list, default=[])

    # Input selection
    input_sel = config.Property(proptype=list, default=None)

    # Fix base properties
    num_cylinders = 2
    cylinder_width = 20.0
    cylinder_spacing = tools._PF_SPACE
    cylinder_length = 40.0

    rotation_angle = tools._PF_ROT

    zenith = np.array([np.pi/2 - np.deg2rad(ephemeris.CHIMELATITUDE), 
                       0.0])

    auto_correlations = True

    _pickle_keys = ['_feeds']

    #
    # === Initialisation routines ===
    #

    def __init__(self, feeds=None):
        import datetime

        self._feeds = feeds

        # Set location properties
        self.latitude = ephemeris.CHIMELATITUDE
        self.longitude = ephemeris.CHIMELONGITUDE
        self.altitude = ephemeris.CHIMEALTITUDE

        # Set the LSD start epoch (i.e. CHIME Pathfinder first light)
        self.lsd_start_day = datetime.datetime(2013, 11, 15)

    @classmethod
    def from_layout(cls, layout, correlator=None, skip=False):
        """Create a Pathfinder telescope description for the specified layout.

        Parameters
        ----------
        layout : integer or datetime
            Layout id number (corresponding to one in database), or datetime
        correlator : string, optional
            Name of the specific correlator. Needed to return a unique config
            in some cases.
        skip : boolean, optional
            Whether to skip non-CHIME antennas. If False, leave them in but
            set them to infinite noise (unsupported at the moment).

        Returns
        -------
        tel : CHIMEPathfinder
        """

        pf = cls()

        pf.layout = layout
        pf.correlator = correlator
        pf.skip_non_chime = skip
        pf._load_layout()

        return pf

    def _load_layout(self):
        """Load the Pathfinder layout from the database.

        Generally this routine shouldn't be called directly. Use
        :method:`CHIMEPathfinder.from_layout` or configure from a YAML file.
        """
        if self.layout is None:
            raise Exception("Layout attributes not set.")

        # Fetch feed layout from database
        feeds = tools.get_correlator_inputs(self.layout, self.correlator)

        if mpiutil.size > 1:
            feeds = mpiutil.world.bcast(feeds, root=0)

        if self.skip_non_chime:
            raise Exception("Not supported.")

        self._feeds = feeds

    def _finalise_config(self):
        # Override base method to implement automatic loading of layout when
        # configuring from YAML.

        if self.layout is not None:
            print "Loading layout: %s" % str(self.layout)
            self._load_layout()

    #
    # === Redefine properties of the base class ===
    #

    # Tweak the following two properties to change the beam width
    @property
    def fwhm_e(self):
        """Full width half max of the E-plane antenna beam."""
        return 2.0 * np.pi / 3.0 * 0.7

    @property
    def fwhm_h(self):
        """Full width half max of the H-plane antenna beam."""
        return 2.0 * np.pi / 3.0 * 1.2

    # Set the approximate uv feed sizes
    @property
    def u_width(self):
        return self.cylinder_width

    # v-width property override
    @property
    def v_width(self):
        return 1.0

    def calculate_frequencies(self):
        """Override default version to give support for specifying by frequency
        channel number.
        """
        if self.use_pathfinder_freq:
            # Use pathfinder channelization of 1024 bins between 400 and 800 MHz.
            basefreq = np.linspace(800.0, 400.0, 1024, endpoint=False)

            # Bin the channels together
            if len(basefreq) % self.channel_bin != 0:
                raise Exception("Channel binning must exactly divide the total number of channels")

            basefreq = basefreq.reshape(-1, self.channel_bin).mean(axis=-1)

            # If requested, select subset of frequencies.
            if self.freq_physical:
                basefreq = basefreq[ [ np.argmin(np.abs(basefreq - freq)) for freq in self.freq_physical ] ]

            elif self.channel_range and (len(self.channel_range) <= 3):
                basefreq = basefreq[slice(*self.channel_range)]

            elif self.channel_index:
                basefreq = basefreq[self.channel_index]

            # Save to object
            self._frequencies = np.unique(basefreq)[::-1]

        else:
            # Otherwise use the standard method
            telescope.TransitTelescope.calculate_frequencies(self)

    @property
    def feeds(self):
        """Return a description of the feeds as a list of :class:`tools.CorrInput` instances.
        """

        if self.input_sel is None:
            feeds = self._feeds
        else:
            feeds = [self._feeds[fi] for fi in self.input_sel]

        return feeds

    @property
    def input_index(self):
        """An index_map describing the inputs known to the telescope. Useful
        for generating synthetic datasets.
        """
        # Extract lists of channel ID and serial numbers
        channels, feed_sn = zip(*[(feed.id, feed.input_sn) for feed in self.feeds])

        # Create an input index map and return it.
        from ch_util import andata
        return andata._generate_input_map(feed_sn, channels)

    @property
    def feedpositions(self):
        """The set of feed positions on *all* cylinders.

        This is constructed for the given layout and includes all rotations of
        the cylinder axis.

        Returns
        -------
        feedpositions : np.ndarray
            The positions in the telescope plane of the receivers. Packed as
            [[u1, v1], [u2, v2], ...].
        """

        # Fetch cylinder relative positions
        pos = tools.get_feed_positions(self.feeds, rotation=self.rotation_angle)

        # The above routine returns NaNs for non CHIME feeds. This is a bit
        # messy, so turn them into zeros.
        pos = np.nan_to_num(pos)

        return pos

    @property
    def beamclass(self):
        """Beam class definition for the Pathfinder.

        When `self.redundant` is set, the X-polarisation feeds get
        `beamclass = 0`, and the Y-polarisation gets `beamclass = 1`.
        If `self.redundant` is `False`, then the feeds are just give an
        increasing unique class. In both cases, any other type of feed gets set
        to `-1` and should be ignored.
        """
        # Make beam class just channel number.

        def _feedclass(f):
            if tools.is_chime_x(f):
                return 0
            if tools.is_chime_y(f):
                return 1
            return -1

        if self.redundant:
            return np.array([_feedclass(f) for f in self.feeds])
        else:
            beamclass = [
                fi if tools.is_chime(feed) else -1
                for fi, feed in enumerate(self.feeds)
            ]
            return np.array(beamclass)

    #
    # === Setup the primary beams ===
    #

    def beam(self, feed, freq):
        """Primary beam implementation for the CHIME Pathfinder.

        This only supports normal CHIME cylinder antennas. Asking for the beams
        for other types of inputs will cause an exception to be thrown. The
        beams from this routine are rotated by `self.rotation_angle` to account
        for the Pathfinder rotation.
        """
        # # Fetch beam parameters out of config database.

        feed_obj = self.feeds[feed]

        # Check that feed exists and is a CHIME cylinder antenna
        if feed_obj is None:
            raise ValueError("Craziness. The requested feed doesn't seem to exist.")

        if not tools.is_chime(feed_obj):
            raise ValueError('Requested feed is not a CHIME antenna.')

        # Get the beam rotation parameters.
        yaw = -self.rotation_angle
        pitch = 0.0
        roll = 0.0

        rot = np.radians([yaw, pitch, roll])

        # We can only support feeds angled parallel or perp to the cylinder
        # axis. Check for these and throw exception for anything else.
        if tools.is_chime_y(feed_obj):
            return cylbeam.beam_y(self._angpos, self.zenith,
                                  self.cylinder_width / self.wavelengths[freq],
                                  self.fwhm_e, self.fwhm_h, rot=rot)
        elif tools.is_chime_x(feed_obj):
            return cylbeam.beam_x(self._angpos, self.zenith,
                                  self.cylinder_width / self.wavelengths[freq],
                                  self.fwhm_e, self.fwhm_h, rot=rot)
        else:
            raise RuntimeError("Given polarisation (feed.pol=%s) not supported." % feed_obj.pol)

    #
    # === Override methods determining the feed pairs we should calculate ===
    #
    # These should probably get ported back into `driftscan` as options.

    def _sort_pairs(self):
        # Reimplemented sort pairs to ensure that returned array is in
        # channel order.

        # Create mask of included pairs, that are not conjugated
        tmask = np.logical_and(self._feedmask, np.logical_not(self._feedconj))
        uniq = telescope._get_indices(self._feedmap, mask=tmask)

        # Get channel id for each feed in the pair, this will be used for the sort
        ci, cj = np.array([(self.feeds[fi].id, self.feeds[fj].id) for fi, fj in uniq]).T

        # # Sort by constructing a numpy array with the keys as fields, and use
        # # np.argsort to get the indices

        # Create array of keys to sort
        dt = np.dtype('i4,i4')
        sort_arr = np.zeros(ci.size, dtype=dt)
        sort_arr['f0'] = ci
        sort_arr['f1'] = cj

        # Get map which sorts
        sort_ind = np.argsort(sort_arr)

        # Invert mapping
        tmp_sort_ind = sort_ind.copy()
        sort_ind[tmp_sort_ind] = np.arange(sort_ind.size)

        # Remap feedmap entries
        fm_copy = self._feedmap.copy()
        wmask = np.where(self._feedmask)
        fm_copy[wmask] = sort_ind[self._feedmap[wmask]]

        self._feedmap = fm_copy

    def _make_ew(self):
        # # Reimplemented to make sure entries we always pick the upper
        # # triangle (and do not reorder to make EW baselines)
        if self.redundant:
            super(CHIMEPathfinder, self)._make_ew()

    def _unique_beams(self):
        # Override to mask out any feed where the beamclass is less than zero.
        # This is used to get exclude feeds which are not normal CHIME cylinder
        # feeds

        beam_map, beam_mask = telescope.TransitTelescope._unique_beams(self)

        # Construct a mask including only the feeds where the beam class is
        # greater than zero
        bc_mask = (self.beamclass >= 0)
        bc_mask = np.logical_and(bc_mask[:, np.newaxis], bc_mask[np.newaxis, :])

        beam_mask = np.logical_and(beam_mask, bc_mask)

        return beam_map, beam_mask

class CHIMEPathfinderExternalBeam(CHIMEPathfinder):
    """Model telescope for the CHIME Pathfinder.

        This class uses an external beam model that is read in from a file.

        Attributes
        ----------
        primary_beamx_filename : str
            The file name for the EW polarisation.

        primary_beamy_filename : str
            The file name for the NS polarisation.
        
        cylinder_id : bool
            Are the beams different for the different cylinders?
            Default False.

        vector_file : bool
            Is the beam vector in the file (or only its amplitude)?
            Default True.

        normalize : bool
            Normalize the beam amplitude to the largest pixel?
            Default False.
    """

    primary_beamx_filename = config.Property(
        proptype=str,
        default='/project/k/krs/cahofer/pass1/beams/beamx_400_800_nfreq200.hdf5'
    )

    primary_beamy_filename = config.Property(
        proptype=str,
        default='/project/k/krs/cahofer/pass1/beams/beamy_400_800_nfreq200.hdf5'
    )

    vector_file = config.Property(proptype=bool, default=True)
    normalize = config.Property(proptype=bool, default=False)
    rotate = config.Property(proptype=bool, default=False)
    cylinder_id = config.Property(proptype=bool, default=False)

    def __init__(self, *args, **kwargs):
        """ Instantiates a CHIMEPathfinderExternalBeam object.
        """
        
        self._beam_initialized = False

    def beam(self, feed, freq_id):
        """
        Overloads the beam of the Pathfinder class.

        """

        if not self._beam_initialized:
            self._init_beam()
            self._beam_initialized = True

        ## Fetch beam parameters out of config database.
        feed_obj = self.feeds[feed]
        tel_freq = self.frequencies

        if feed_obj is None:
            raise Exception("Craziness. The requested feed doesn't seem to exist.")
        
        #Parse polarisation id
        if tools.is_chime_y(feed_obj):
            polstr = "NS"
        elif tools.is_chime_x(feed_obj):
            polstr = "EW"
        else:
            raise ValueError("Polarisation %s not supported by this feed" % str(feed_obj.pol),
                             feed_obj)
        self._polstr = polstr

        #Parse cylinder id
        cylnum = feed_obj.cyl
        if cylnum == 0:
            cylstr = "W"
        elif cylnum == 1:
            cylstr = "E"
        else:
            raise ValueError("Cylinder %s not supported." % str(cylnum))
        self._cylstr = cylstr
        
        #Get the beam
        map_out = self._get_beam(freq_id, cylstr, polstr)
    
        return map_out

    def _get_beam(self, freq_id, cylstr, polstr):
        """
        Get the beam from file if it is not in memory.

        """
        
        #Get pol cyl info
        pci = self._pol_cyl_map()

        if freq_id in self._loaded_freq[pci]:
            freq_idl = self._loaded_freq[pci].index(freq_id)
            map_out = self._beam_arr[pci][freq_idl]

        else:
            #Get from file
            map_out = self._beam_io(freq_id, cylstr, polstr)

            if len(self._beam_arr[pci]) == 0:
                #First time we are loading a map of this pol cyl type
                self._loaded_freq[pci].append(freq_id)
                self._beam_arr[pci] = map_out[np.newaxis, ...]

            else:
                #Find array index
                largerfreq = np.where(np.array(self._loaded_freq[pci]) > freq_id)[0]
                if len(largerfreq) == 0:
                    fi = len(self._loaded_freq[pci])
                else:
                    fi = largerfreq[0]
                #Add to the array and list
                self._beam_arr[pci] = np.insert(self._beam_arr[pci], fi, map_out, axis=0)
                self._loaded_freq[pci].insert(fi, freq_id)
            
        if not np.all(map_out == 0.0):
            if self.normalize:
                maxpix = np.max(np.absolute(map_out), axis=0)
                map_out = map_out/maxpix[np.newaxis, :]
        else:
            warning.warn("Returning a zero beam map.", RuntimeWarning)

        #Up- or down-grade the map if required
        map_out = healpy.pixelfunc.ud_grade(map_out.swapaxes(0, 1), self._nside)
        map_out = np.array(map_out).swapaxes(0, 1)

        return map_out

    def _beam_io(self, freq_id, cylstr, polstr):
        """
        Modify this routine for a specific instance of beam file type.
        """

        if polstr == 'NS':
            infile = self.primary_beamy_filename
        elif polstr == 'EW':
            infile = self.primary_beamx_filename

        if cylstr == "W":
            cylnum = 0
        elif cylstr == 'E':
            cylnum = 1

        #Get telescope frequencies
        tel_freq = self.frequencies

        try:
            print "Attempting to read beam "+cylstr+"-"+polstr+" from disk..."

            with h5py.File(infile, 'r') as f:

                if self.cylinder_id:
                    map_freq = f['freq'][:]
                    good_freq = f['good_freq'][:]
                    gmap_freq = map_freq[good_freq]
                    gfreq_sel = np.arange(len(map_freq))[good_freq]
                    freq_sel = _nearest_freq(tel_freq, gmap_freq, freq_id)
                    freq_sel = gfreq_sel[freq_sel]

                    beam_map = f['abeam'][freq_sel, cylnum, :]

                    npix = f['abeam'][0, 0].shape[0]

                else:
                    map_freq = f['freq'][:]
                    freq_sel = _nearest_freq(tel_freq, map_freq, freq_id)
                    beam_map = f['beam'][freq_sel, :]
                    beam_map = np.concatenate((beam_map['Et'][..., np.newaxis], 
                                               beam_map['Ep'][..., np.newaxis]),
                                              axis=-1)
                    
                    npix = f['beam']['Et'][0].shape[0]

        except IOError:
            raise IOError("Could not load beams from disk [path: %s]." % infile)

        #Get map nside
        nside = healpy.npix2nside(npix)

        if self.rotate:                
            #Get telescope coords
            thetai, phii = healpy.pix2ang(nside, np.arange(npix))
            phii = np.where(np.logical_and(phii > np.pi, phii <= 2*np.pi), -(2*np.pi - phii), phii)
            #Rotate from telescope coords to ra, dec.
            rot = [0.0, ephemeris.CHIMELATITUDE, self.rotation_angle]
            if len(beam_map.shape) == 2:
                beam_map[0] = _rotate(beam_map[0], thetai, phii, rot=rot)
                beam_map[1] = _rotate(beam_map[1], thetai, phii, rot=rot)
            else:
                beam_map = _rotate(beam_map, thetai, phii, rot=rot)

        if not self.vector_file:
            #Get the polarisation vector
            pvec = self.pol_vector(polstr, nside)
            beam_map, pvec = np.broadcast_arrays(beam_map[..., np.newaxis], pvec)
            beam_map = beam_map*pvec

        if len(freq_sel) == 2:
            allzero = np.all(np.all(beam_map == 0.0, axis=-1), axis=-1)
            nonzero = np.logical_not(allzero)
            if np.all(allzero):
                beam_map = beam_map[0]

            elif np.all(nonzero):                
                #Interpolate
                freq_high = map_freq[freq_sel[1]]
                freq_low = map_freq[freq_sel[0]]
                freq_int = tel_freq[freq_id]

                alpha = (freq_high - freq_int) / (freq_high - freq_low)
                beta = (freq_int - freq_low) / (freq_high - freq_low)

                map_t = beam_map[0, :, 0] * alpha + beam_map[1, :, 0] * beta
                map_p = beam_map[0, :, 1] * alpha + beam_map[1, :, 1] * beta
                
                beam_map = np.column_stack((map_t, map_p))

            else:
                #Just use the non zero one
                beam_map = beam_map[nonzero]

        return beam_map


    def pol_vector(self, polstr, nside):
        """
        Get the polarisation vector assuming no leakage.
        """

        zenith = self.zenith

        that, phat = coord.thetaphi_plane_cart(zenith)
        rot = np.deg2rad([-self.rotation_angle, 0.0, 0.0])
        xhat, yhat, zhat = cylbeam.rotate_ypr(rot, phat, -that, coord.sph_to_cart(zenith))

        angpos = hputil.ang_positions(nside)

        if polstr == "EW":
            pvec = cylbeam.polpattern(angpos, xhat)
        elif polstr == "NS":
            pvec = cylbeam.polpattern(angpos, yhat)

        return pvec

    def _pol_cyl_map(self):

        if self._polstr == 'NS':
            if self.cylinder_id:
                if self._cylstr == 'W':
                    return 0
                elif self._cylstr == 'E':
                    return 2
            else:
                return 0
        elif self._polstr == 'EW':
            if self.cylinder_id:
                if self._cylstr == 'W':
                    return 1
                elif self._cylstr == 'E':
                    return 3
            else:
                return 1

    def _init_beam(self):

        if self.cylinder_id:
            self._npolcyl = 4
            self._loaded_freq = [[], [], [], []]
            self._beam_arr = [[], [], [], []]
        else:
            self._npolcyl = 2
            self._loaded_freq = [[], []]
            self._beam_arr = [[], []]

def _nearest_freq(tel_freq, map_freq, freq_id):

    """Find nearest neighbor frequencies.

    Parameters
    ----------
    tel_freq : float
        frequencies from telescope object.
    map_freq : float
        frequencies from beam map file.
    freq_id : int
        frequency selection.

    Returns
    -------
    freq_ind : list of neighboring map frequencies matched to tel_freq.

    """

    diff_freq = abs(map_freq - tel_freq[freq_id])
    map_freq_width = abs(map_freq[1] - map_freq[0])
    match_mask = diff_freq < map_freq_width

    freq_ind = np.nonzero(match_mask)[0]

    return freq_ind

def _rotate(bmap, theta, phi, rot=[0.0, 49.32, 1.9]):
    
    yaw = np.deg2rad(rot[0])
    pitch = np.deg2rad(rot[1])
    roll = np.deg2rad(rot[2])
    
    #Convert to cartesian                           
    x = np.cos(phi)*np.sin(theta)
    y = np.sin(phi)*np.sin(theta)
    z = np.cos(theta)

    cart = np.array([x, y, z])

    #Apply yaw rotation
    rot = [[np.cos(yaw), np.sin(yaw),  0],
           [-np.sin(yaw), np.cos(yaw), 0],
           [0.0,         0.0,        1.0]]
    
    cartp = np.dot(rot, cart)
    
    #Apply pitch rotation
    rot = [[np.cos(pitch),    0, np.sin(pitch)],
           [0,            1.0,           0],
           [-np.sin(pitch), 0.0, np.cos(pitch)]]

    cartp = np.dot(rot, cartp)

    #Apply 1.9 degree rotation              
    rot = [[1.0,            0.0,         0.0],
            [0, np.cos(roll),  np.sin(roll)],
            [0, -np.sin(roll), np.cos(roll)]]

    cartp = np.dot(rot, cartp)

    #Get the new angular coordinates                                                                   
    phip = np.arctan2(cartp[1], cartp[0])
    thetap = np.arccos(cartp[2])
    
    #Interpolate on to rotated grid
    amap = griddata((theta, phi), bmap, (thetap, phip), method='linear', fill_value=0.0)
    
    return amap

"""
def _nearest_freq(tel_freq, map_freq, freq_id):
	
    """ """Find nearest neighbor frequencies.
	
    Parameters
    ----------
    tel_freq : float
        Frequencies from telescope object
    map_freq : float
        Frequencies for which beams exist.
    freq_id : int
        Frequency selection.
	
    Returns
    -------
    freq_ind : int
        List of closest frequencies matched to tel_freq.

    """ """
	
    diff_freq = abs(map_freq - tel_freq)

    freq_width = 400.0/1024/2
	
    match_mask = diff_freq < freq_width

    if np.any(match_mask):
        freq_ind = [np.argmin(diff_freq)]
    else:
        freq_ext = np.argmin(diff_freq)
        if map_freq[freq_ext] < tel_freq:
            freq_ind = np.append(freq_ext-1, freq_ext).tolist()
        else:
            freq_ind = np.append(freq_ext, freq_ext+1).tolist()
        
    return freq_ind
"""
