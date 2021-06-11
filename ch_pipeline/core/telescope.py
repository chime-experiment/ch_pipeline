"""
Pathfinder/CHIME Telescope Model

A model for both CHIME and Pathfinder telescopes. This attempts to query the
configuration db (:mod:`~ch_analysis.pathfinder.configdb`) for the details of
the feeds and their positions.
"""

import logging

import numpy as np
import h5py
import healpy

from caput import config, mpiutil

from drift.core import telescope
from drift.telescope import cylbeam

from draco.core.containers import ContainerBase

from ch_util import ephemeris, tools


# Get the logger for the module
logger = logging.getLogger(__name__)


class CHIME(telescope.PolarisedTelescope):
    """Model telescope for the CHIME/Pathfinder.

    This class currently uses a simple Gaussian model for the primary beams.

    Attributes
    ----------
    layout : datetime or int
        Specify which layout to use.
    correlator : string
        Restrict to a specific correlator.
    skip_non_chime : boolean
        Ignore non CHIME feeds in the BeamTransfers.
    stack_type : string, optional
        Stacking type.
        `redundant`: feeds of same polarization have same beam class (default).
        `redundant_cyl`: feeds of same polarization and cylinder have same beam
        class.
        `unique`: Each feed has a unique beam class.
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
    baseline_masking_type : string, optional
        Select a subset of baselines.
        `total_length` selects baselines according to their total length. Need to
        specifiy `minlength` and `maxlength` properties (defined in baseclass).
        `individual_length` selects baselines according to their seperation
        in the North-South (specify `minlength_ns` and `maxlength_ns`) or
        the East-West (specify `minlength_ew` and `maxlength_ew`).
    minlength_ns, maxlength_ns : scalar
        Minimum and maximum North-South baseline lengths to include (in metres)
    minlength_ew, maxlength_ew:
        Minimum and maximum East-West baseline lengths to include (in metres)
    dec_normalized: float, optional
        Normalize the beam by its magnitude at transit at this declination
        in degrees.
    """

    # Configure which feeds and layout to use
    layout = config.Property(default=None)
    correlator = config.Property(proptype=str, default=None)
    skip_non_chime = config.Property(proptype=bool, default=False)

    # Redundancy settings
    stack_type = config.enum(
        ["redundant", "redundant_cyl", "unique"], default="redundant"
    )

    # Configure frequency properties
    use_pathfinder_freq = config.Property(proptype=bool, default=True)
    channel_bin = config.Property(proptype=int, default=1)
    freq_physical = config.Property(proptype=list, default=[])
    channel_range = config.Property(proptype=list, default=[])
    channel_index = config.Property(proptype=list, default=[])

    # Input selection
    input_sel = config.Property(proptype=list, default=None)

    # Baseline masking options
    baseline_masking_type = config.enum(
        ["total_length", "individual_length"], default="individual_length"
    )
    minlength_ew = config.Property(proptype=float, default=0.0)
    maxlength_ew = config.Property(proptype=float, default=1.0e7)
    minlength_ns = config.Property(proptype=float, default=0.0)
    maxlength_ns = config.Property(proptype=float, default=1.0e7)

    # Auto-correlations setting (overriding default in baseclass)
    auto_correlations = config.Property(proptype=bool, default=True)

    # Beam normalization
    dec_normalized = config.Property(proptype=float, default=None)

    # Fix base properties
    cylinder_width = 20.0
    cylinder_spacing = tools._PF_SPACE

    _pickle_keys = ["_feeds"]

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

        # Set the LSD start epoch (i.e. CHIME/Pathfinder first light)
        self.lsd_start_day = datetime.datetime(2013, 11, 15)

        # Set the overall normalization of the beam
        self._set_beam_normalization()

    @classmethod
    def from_layout(cls, layout, correlator=None, skip=False):
        """Create a CHIME/Pathfinder telescope description for the specified layout.

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
        tel : CHIME
        """

        tel = cls()

        tel.layout = layout
        tel.correlator = correlator
        tel.skip_non_chime = skip
        tel._load_layout()

        return tel

    def _load_layout(self):
        """Load the CHIME/Pathfinder layout from the database.

        Generally this routine shouldn't be called directly. Use
        :method:`CHIME.from_layout` or configure from a YAML file.
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
            logger.debug("Loading layout: %s", str(self.layout))
            self._load_layout()

        # Set the overall normalization of the beam
        self._set_beam_normalization()

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

    # Set non-zero rotation angle for pathfinder and chime
    @property
    def rotation_angle(self):
        if self.correlator == "pathfinder":
            return tools._PF_ROT
        elif self.correlator == "chime":
            return tools._CHIME_ROT
        else:
            return 0.0

    def calculate_frequencies(self):
        """Override default version to give support for specifying by frequency
        channel number.
        """
        if self.use_pathfinder_freq:
            # Use pathfinder channelization of 1024 bins between 400 and 800 MHz.
            basefreq = np.linspace(800.0, 400.0, 1024, endpoint=False)

            # Bin the channels together
            if len(basefreq) % self.channel_bin != 0:
                raise Exception(
                    "Channel binning must exactly divide the total number of channels"
                )

            basefreq = basefreq.reshape(-1, self.channel_bin).mean(axis=-1)

            # If requested, select subset of frequencies.
            if self.freq_physical:
                basefreq = basefreq[
                    [np.argmin(np.abs(basefreq - freq)) for freq in self.freq_physical]
                ]

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
        """Return a description of the feeds as a list of :class:`tools.CorrInput` instances."""

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
        channels, feed_sn = list(
            zip(*[(feed.id, feed.input_sn) for feed in self.feeds])
        )

        # Create an input index map and return it.
        from ch_util import andata

        return andata._generate_input_map(feed_sn, channels)

    _pos = None

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

        if self._pos is None:
            # Fetch cylinder relative positions
            pos = tools.get_feed_positions(self.feeds)

            # The above routine returns NaNs for non CHIME feeds. This is a bit
            # messy, so turn them into zeros.
            self._pos = np.nan_to_num(pos)

        return self._pos

    @property
    def beamclass(self):
        """Beam class definition for the CHIME/Pathfinder.

        When `self.stack_type` is `redundant`, the X-polarisation feeds get
        `beamclass = 0`, and the Y-polarisation gets `beamclass = 1`.
        When `self.stack_type` is `redundant_cyl`, feeds of same polarisation
        and cylinder have same beam class. The beam class is given by
        `beamclass = 2*cyl + pol` where `cyl` is the cylinder number according to
        `ch_util.tools` convention and `pol` is the polarisation (0 for X and 1
        for Y polarisation)
        When `self.stack_type` is `unique`, then the feeds are just given an
        increasing unique class.
        In all cases, any other type of feed gets set to `-1` and should be
        ignored.
        """
        # Make beam class just channel number.

        def _feedclass(f, redundant_cyl=False):
            if tools.is_array(f):
                if tools.is_array_x(f):  # feed is X polarisation
                    pol = 0
                else:  # feed is Y polarisation
                    pol = 1

                if redundant_cyl:
                    return 2 * f.cyl + pol
                else:
                    return pol
            return -1

        if self.stack_type == "redundant":
            return np.array([_feedclass(f) for f in self.feeds])
        elif self.stack_type == "redundant_cyl":
            return np.array([_feedclass(f, redundant_cyl=True) for f in self.feeds])
        else:
            beamclass = [
                fi if tools.is_array(feed) else -1 for fi, feed in enumerate(self.feeds)
            ]
            return np.array(beamclass)

    @property
    def polarisation(self):
        """
        Polarisation map.

        Returns
        -------
        pol : np.ndarray
            One-dimensional array with the polarization for each feed ('X' or 'Y').
        """

        def _pol(f):
            if tools.is_array(f):
                if tools.is_array_x(f):  # feed is X polarisation
                    return "X"
                else:  # feed is Y polarisation
                    return "Y"
            return "N"

        return np.asarray([_pol(f) for f in self.feeds], dtype=np.str)

    #
    # === Setup the primary beams ===
    #

    def beam(self, feed, freq, angpos=None):
        """Primary beam implementation for the CHIME/Pathfinder.

        This only supports normal CHIME cylinder antennas. Asking for the beams
        for other types of inputs will cause an exception to be thrown. The
        beams from this routine are rotated by `self.rotation_angle` to account
        for the CHIME/Pathfinder rotation.

        Parameters
        ----------
        feed : int
            Index for the feed.
        freq : int
            Index for the frequency.
        angpos : np.ndarray[nposition, 2], optional
            Angular position on the sky (in radians).
            If not provided, default to the _angpos
            class attribute.

        Returns
        -------
        beam : np.ndarray[nposition, 2]
            Amplitude vector of beam at each position on the sky.
        """
        # # Fetch beam parameters out of config database.

        feed_obj = self.feeds[feed]

        # Check that feed exists and is a CHIME cylinder antenna
        if feed_obj is None:
            raise ValueError("Craziness. The requested feed doesn't seem to exist.")

        if not tools.is_array(feed_obj):
            raise ValueError("Requested feed is not a CHIME antenna.")

        # If the angular position was not provided, then use the values in the
        # class attribute.
        if angpos is None:
            angpos = self._angpos

        # Get the beam rotation parameters.
        yaw = -self.rotation_angle
        pitch = 0.0
        roll = 0.0

        rot = np.radians([yaw, pitch, roll])

        # We can only support feeds angled parallel or perp to the cylinder
        # axis. Check for these and throw exception for anything else.
        if tools.is_array_y(feed_obj):
            beam = cylbeam.beam_y(
                angpos,
                self.zenith,
                self.cylinder_width / self.wavelengths[freq],
                self.fwhm_e,
                self.fwhm_h,
                rot=rot,
            )
        elif tools.is_array_x(feed_obj):
            beam = cylbeam.beam_x(
                angpos,
                self.zenith,
                self.cylinder_width / self.wavelengths[freq],
                self.fwhm_e,
                self.fwhm_h,
                rot=rot,
            )
        else:
            raise RuntimeError(
                "Given polarisation (feed.pol=%s) not supported." % feed_obj.pol
            )

        # Normalize the beam
        if self._beam_normalization is not None:
            beam *= self._beam_normalization[freq, feed, np.newaxis, :]

        return beam

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
        dt = np.dtype("i4,i4")
        sort_arr = np.zeros(ci.size, dtype=dt)
        sort_arr["f0"] = ci
        sort_arr["f1"] = cj

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
        if self.stack_type != "unique":
            super(CHIME, self)._make_ew()

    def _unique_baselines(self):
        # Reimplement unique baselines in order to mask out either according to total
        # baseline length or maximum North-South and East-West baseline seperation.

        from drift.core import telescope

        # Construct array of indices
        fshape = [self.nfeed, self.nfeed]
        f_ind = np.indices(fshape)

        # Construct array of baseline separations
        bl1 = self.feedpositions[f_ind[0]] - self.feedpositions[f_ind[1]]
        bl2 = np.around(bl1[..., 0] + 1.0j * bl1[..., 1], self._bl_tol)

        # Construct array of baseline lengths
        blen = np.sum(bl1 ** 2, axis=-1) ** 0.5

        if self.baseline_masking_type == "total_length":
            # Create mask of included baselines
            mask = np.logical_and(blen >= self.minlength, blen <= self.maxlength)
        else:
            mask_ew = np.logical_and(
                abs(bl1[..., 0]) >= self.minlength_ew,
                abs(bl1[..., 0]) <= self.maxlength_ew,
            )
            mask_ns = np.logical_and(
                abs(bl1[..., 1]) >= self.minlength_ns,
                abs(bl1[..., 1]) <= self.maxlength_ns,
            )
            mask = np.logical_and(mask_ew, mask_ns)

        # Remove the auto correlated baselines between all polarisations
        if not self.auto_correlations:
            mask = np.logical_and(blen > 0.0, mask)

        return telescope._remap_keyarray(bl2, mask), mask

    def _unique_beams(self):
        # Override to mask out any feed where the beamclass is less than zero.
        # This is used to get exclude feeds which are not normal CHIME cylinder
        # feeds

        beam_map, beam_mask = telescope.TransitTelescope._unique_beams(self)

        # Construct a mask including only the feeds where the beam class is
        # greater than zero
        bc_mask = self.beamclass >= 0
        bc_mask = np.logical_and(bc_mask[:, np.newaxis], bc_mask[np.newaxis, :])

        beam_mask = np.logical_and(beam_mask, bc_mask)

        return beam_map, beam_mask

    def _set_beam_normalization(self):
        """Determine the beam normalization for each feed and frequency.

        The beam will be normalized by its value at transit at the declination
        provided in the dec_normalized config parameter.  If this config parameter
        is set to None, then there is no additional normalization applied.
        """

        self._beam_normalization = None

        if self.dec_normalized is not None:

            angpos = np.array(
                [(0.5 * np.pi - np.radians(self.dec_normalized)), 0.0]
            ).reshape(1, -1)

            beam = np.ones((self.nfreq, self.nfeed, 2), dtype=np.float64)

            beam_lookup = {}

            for fe, feed in enumerate(self.feeds):

                if not tools.is_array(feed):
                    continue

                beamclass = self.beamclass[fe]

                if beamclass not in beam_lookup:

                    beam_lookup[beamclass] = np.ones((self.nfreq, 2), dtype=np.float64)
                    for fr in range(self.nfreq):
                        beam_lookup[beamclass][fr] = self.beam(fe, fr, angpos)[0]

                beam[:, fe, :] = beam_lookup[beamclass]

            self._beam_normalization = tools.invert_no_zero(
                np.sqrt(np.sum(beam ** 2, axis=-1))
            )


def _flat_top_gauss6(x, A, sig, x0):
    """Flat-top gaussian. Power of 6."""
    return A * np.exp(-abs((x - x0) / sig) ** 6)


def _flat_top_gauss3(x, A, sig, x0):
    """Flat-top gaussian. Power of 3."""
    return A * np.exp(-abs((x - x0) / sig) ** 3)


class CHIMEParameterizedBeam(CHIME):
    """CHIME telescope that uses a parameterized fit to the driftscan beam.

    This speeds up evaluation of the beam model.
    """

    SIGMA_EW = [14.87857614, 9.95746878]

    FUNC_NS = [_flat_top_gauss6, _flat_top_gauss3]
    PARAM_NS = np.array(
        [[9.97981768e-01, 1.29544939e00, 0.0], [9.86421047e-01, 8.10213326e-01, 0.0]]
    )

    def _sigma(self, pol_index, freq_index, dec):
        """Width of the power beam in the EW direction."""
        return self.SIGMA_EW[pol_index] / self.frequencies[freq_index] / np.cos(dec)

    def _beam_amplitude(self, pol_index, dec):
        """Amplitude of the power beam at meridian."""
        return self.FUNC_NS[pol_index](
            dec - np.radians(self.latitude), *self.PARAM_NS[pol_index]
        )

    def beam(self, feed, freq, angpos=None):
        """Parameterized fit to driftscan cylinder beam model for CHIME telescope.

        Parameters
        ----------
        feed : int
            Index for the feed.
        freq : int
            Index for the frequency.
        angpos : np.ndarray[nposition, 2], optional
            Angular position on the sky (in radians).
            If not provided, default to the _angpos
            class attribute.

        Returns
        -------
        beam : np.ndarray[nposition, 2]
            Amplitude vector of beam at each position on the sky.
        """

        feed_obj = self.feeds[feed]

        # Check that feed exists and is a CHIME cylinder antenna
        if feed_obj is None:
            raise ValueError("Craziness. The requested feed doesn't seem to exist.")

        if not tools.is_array(feed_obj):
            raise ValueError("Requested feed is not a CHIME antenna.")

        # If the angular position was not provided, then use the values in the
        # class attribute.
        if angpos is None:
            angpos = self._angpos

        dec = 0.5 * np.pi - angpos[:, 0]
        ha = angpos[:, 1]

        # We can only support feeds angled parallel or perp to the cylinder
        # axis. Check for these and throw exception for anything else.
        if tools.is_array_x(feed_obj):
            pol = 0
        elif tools.is_array_y(feed_obj):
            pol = 1
        else:
            raise RuntimeError(
                "Given polarisation (feed.pol=%s) not supported." % feed_obj.pol
            )

        beam = np.zeros((angpos.shape[0], 2), dtype=np.float64)
        beam[:, 0] = np.sqrt(
            self._beam_amplitude(pol, dec)
            * np.exp(-((ha / self._sigma(pol, freq, dec)) ** 2))
        )

        # Normalize the beam
        if self._beam_normalization is not None:
            beam *= self._beam_normalization[freq, feed, np.newaxis, :]

        return beam


class CHIMEExternalBeam(CHIME):
    """Model telescope for the CHIME.

    This class uses an external beam model that is read in from a file.
    """

    primary_beam_filename = config.Property(proptype=str)
    freq_interp_beam = config.Property(proptype=bool, default=False)

    def _finalise_config(self):
        """Get the beam file object."""
        logger.debug("Reading beam model from {}...".format(self.primary_beam_filename))
        self._primary_beam = ContainerBase.from_file(
            self.primary_beam_filename, mode="r", distributed=False, ondisk=True
        )

        # cache axes
        self._beam_freq = self._primary_beam.freq[:]
        self._beam_nside = self._primary_beam.nside
        # TODO must use bytestring here because conversion doesn't work with ondisk=True
        self._beam_pol_map = {
            "X": list(self._primary_beam.pol[:]).index(b"X"),
            "Y": list(self._primary_beam.pol[:]).index(b"Y"),
        }

        if len(self._primary_beam.input) > 1:
            raise ValueError("Per-feed beam model not supported for now.")

        super()._finalise_config()

    def beam(self, feed, freq_id):
        feed_obj = self.feeds[feed]
        tel_freq = self.frequencies
        nside = self._nside
        npix = healpy.nside2npix(nside)

        if feed_obj is None:
            raise ValueError("The requested feed doesn't seem to exist.")

        if tools.is_array_x(feed_obj):
            pol_ind = self._beam_pol_map["X"]
        elif tools.is_array_y(feed_obj):
            pol_ind = self._beam_pol_map["Y"]
        else:
            raise ValueError("Polarisation not supported by this feed", feed_obj)

        # find nearest frequency
        freq_sel = _nearest_freq(
            tel_freq, self._beam_freq, freq_id, single=(not self.freq_interp_beam)
        )
        beam_map = self._primary_beam.beam[freq_sel, pol_ind, 0, :]

        # check resolution
        if nside != self._beam_nside:
            logger.debug(
                "Resampling external beam from nside {:d} to {:d}".format(
                    self._beam_nside,
                    nside,
                )
            )
            beam_map_new = np.zeros((len(freq_sel), npix), dtype=beam_map.dtype)
            beam_map_new["Et"] = healpy.ud_grade(beam_map["Et"], nside)
            beam_map_new["Ep"] = healpy.ud_grade(beam_map["Ep"], nside)
            beam_map = beam_map_new

        if len(freq_sel) == 1:
            # exact match
            map_out = np.empty((npix, 2), dtype=beam_map.dtype["Et"])
            map_out[:, 0] = beam_map["Et"][0]
            map_out[:, 1] = beam_map["Ep"][0]
            return map_out
        elif len(freq_sel) == 0:
            raise ValueError(
                "No beam model spans frequency {:.2f}.".format(tel_freq[freq_id])
            )
        else:
            # interpolate between pair of frequencies
            freq_high = self._beam_freq[freq_sel[1]]
            freq_low = self._beam_freq[freq_sel[0]]
            freq_int = tel_freq[freq_id]

            alpha = (freq_high - freq_int) / (freq_high - freq_low)
            beta = (freq_int - freq_low) / (freq_high - freq_low)

            map_t = beam_map["Et"][0] * alpha + beam_map["Et"][1] * beta
            map_p = beam_map["Ep"][0] * alpha + beam_map["Ep"][1] * beta

            map_out = np.empty((npix, 2), dtype=beam_map.dtype["Et"])
            map_out[:, 0] = healpy.pixelfunc.ud_grade(map_t, nside)
            map_out[:, 1] = healpy.pixelfunc.ud_grade(map_p, nside)

            return map_out


def _nearest_freq(tel_freq, map_freq, freq_id, single=False):
    """Find nearest neighbor frequencies. Assumes map frequencies
    are uniformly spaced.

    Parameters
    ----------
    tel_freq : float
        frequencies from telescope object.
    map_freq : float
        frequencies from beam map file.
    freq_id : int
        frequency selection.
    single : bool
        Only return the single nearest neighbour.

    Returns
    -------
    freq_ind : list of neighboring map frequencies matched to tel_freq.

    """

    diff_freq = abs(map_freq - tel_freq[freq_id])
    if single:
        return np.array([np.argmin(diff_freq)])

    map_freq_width = abs(map_freq[1] - map_freq[0])
    match_mask = diff_freq < map_freq_width

    freq_ind = np.nonzero(match_mask)[0]

    return freq_ind
