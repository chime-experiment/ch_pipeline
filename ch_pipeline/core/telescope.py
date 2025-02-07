"""Pathfinder/CHIME Telescope Model.

A model for both CHIME and Pathfinder telescopes. This attempts to query the
configuration db (:mod:`~ch_analysis.pathfinder.configdb`) for the details of
the feeds and their positions.
"""

import logging
from functools import cached_property
from typing import ClassVar
import pickle
import datetime


import healpy
import numpy as np
from caput import config, misc, mpiutil
from ch_util import tools
from cora.util import coord, hputil
from draco.core import task
from draco.core.containers import ContainerBase, GridBeam, HEALPixBeam
from drift.core import telescope
from drift.telescope import cylbeam
from scipy.interpolate import RectBivariateSpline

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
    read_local_layout : boolean
        Read in the layout of the CHIME telescope from a file, saved to this
        repository, rather than attempting to connect and read the layout from
        the database. Default is True.
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
        Select a subset of baselines.  `total_length` selects baselines according to
        their total length. Need to specify `minlength` and `maxlength` properties
        (defined in baseclass).  `individual_length` selects baselines according to
        their seperation in the North-South (specify `minlength_ns` and `maxlength_ns`)
        or the East-West (specify `minlength_ew` and `maxlength_ew`).
    minlength_ns, maxlength_ns : float
        Minimum and maximum North-South baseline lengths to include (in metres)
    minlength_ew, maxlength_ew: float
        Minimum and maximum East-West baseline lengths to include (in metres)
    dec_normalized: float, optional
        Normalize the beam by its magnitude at transit at this declination
        in degrees.
    skip_pol_pair : list
        List of antenna polarisation pairs to skip. Valid entries are "XX", "XY", "YX"
        or "YY". Like the skipped frequencies these pol pairs will have entries
        generated but their beam transfer matrices are implicitly zero and thus not
        calculated.
    """

    # Configure which feeds and layout to use
    layout = config.Property(default=None)
    correlator = config.Property(proptype=str, default=None)
    skip_non_chime = config.Property(proptype=bool, default=False)
    read_local_layout = config.Property(proptype=bool, default=True)

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
    # Skipping frequency/baseline parameters
    skip_pol_pair = config.list_type(type_=str, maxlength=4, default=[])

    # Fix base properties
    cylinder_width = 20.0
    # XXX CHECK: Should CHIME be using the Pathfinder antenna spacing?
    cylinder_spacing = tools.PF_SPACE

    _exwidth: ClassVar[float] = [0.7]
    _eywidth = _exwidth

    _hxwidth: ClassVar[float] = [1.2]
    _hywidth = _hxwidth

    _pickle_keys: ClassVar[str] = ["_feeds"]

    #
    # === Initialisation routines ===
    #

    def __init__(self, feeds=None):
        from ch_ephem.observers import chime

        self._feeds = feeds

        # Set location properties
        self.latitude = chime.latitude
        self.longitude = chime.longitude
        self.altitude = chime.altitude

        # Set the LSD start epoch (i.e. CHIME/Pathfinder first light)
        self.lsd_start_day = datetime.datetime(2013, 11, 15)

        # Set the overall normalization of the beam
        self._set_beam_normalization()

    @classmethod
    def from_layout(cls, layout, read_local_layout=True, correlator=None, skip=False):
        """Create a CHIME/Pathfinder telescope description for the specified layout.

        Parameters
        ----------
        layout : integer or datetime
            Layout id number (corresponding to one in database), or datetime
        read_local_layout: boolean, optional
            Read the feed layout from a file saved to this repository, rather
            than connecting to the database. Default is True.
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
        tel.read_local_layout = read_local_layout
        tel.skip_non_chime = skip
        tel._load_layout()

        return tel

    def _load_layout(self):
        """Load the CHIME/Pathfinder layout.

        Will use a routine to read in a layout from a file saved
        to this repository if `read_local_layout` is set to True.
        Otherwise, will attempt to query the layout database.

        Generally this routine shouldn't be called directly. Use
        :method:`CHIME.from_layout` or configure from a YAML file.
        """
        if self.layout is None:
            raise Exception("Layout attributes not set.")

        if self.read_local_layout:
            # If we're reading locally, use the pickled layouts file
            feeds = self._read_local_layout()
        else:
            # If not, fetch feed layout from database
            feeds = tools.get_correlator_inputs(self.layout, self.correlator)

        if mpiutil.size > 1:
            feeds = mpiutil.world.bcast(feeds, root=0)

        if self.skip_non_chime:
            raise Exception("Not supported.")

        self._feeds = feeds

    def _read_local_layout(self):
        """Load the telescope layout from a file saved to this repository.

        Note that this function will fallback to a database query if the
        requested layout is earlier than September 1, 2018. As with
        :method:`CHIME._load_layout`, this routine should not be called
        directly.
        """
        # Do I/O, and resolve the layout, only on rank 0
        if mpiutil.rank == 0:

            # Get the path of the layout file
            from importlib.resources import files

            layout_path = files("ch_pipeline/core/telescope_files/layouts.pkl")

            with layout_path.open("rb") as layout_f:
                layouts = pickle.load(layout_f)

            # Load layout start and end times in arrays for comparisons
            lay_start = np.array([lay["start"] for lay in layouts])
            lay_end = np.array([lay["end"] for lay in layouts])

            # Handle edge cases, i.e. where requested layout is earlier
            # than the earliest saved layout...
            if self.layout < lay_start[0]:
                logger.info(
                    "You have requested a layout from before the earliest "
                    "recorded layout. Will attempt to query the "
                    "database. To avoid this behavior, select a layout "
                    "after September 1, 2018."
                )

                feeds = tools.get_correlator_inputs(self.layout, self.correlator)

            # ... or later than the latest
            elif self.layout > lay_start[-1]:
                logger.warning(
                    "You have requested the latest locally recorded layout, "
                    "from October 13, 2020. There is no guarantee that this "
                    "is the latest layout available from the database. "
                    "Attempt a database connection if you are certain you "
                    "need the latest available layout."
                )

                feeds = layouts[-1]["inputs"]

            # If not, find which layout was in use at the requested time
            else:

                # The last 'end' element is None, change it to today
                lay_end[-1] = datetime.datetime.today()

                lay_in_use = np.where(
                        (self.layout > lay_start) & (self.layout < lay_end)
                )[0][0]

                feeds = layouts[lay_in_use]["inputs"]

        else:
            feeds = None

        return feeds

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
    @cached_property
    def fwhm_ex(self):
        """Full width half max of the E-plane antenna beam for X polarization."""
        return np.polyval(np.array(self._exwidth) * 2.0 * np.pi / 3.0, self.frequencies)

    @cached_property
    def fwhm_hx(self):
        """Full width half max of the H-plane antenna beam for X polarization."""
        return np.polyval(np.array(self._hxwidth) * 2.0 * np.pi / 3.0, self.frequencies)

    @cached_property
    def fwhm_ey(self):
        """Full width half max of the E-plane antenna beam for Y polarization."""
        return np.polyval(np.array(self._eywidth) * 2.0 * np.pi / 3.0, self.frequencies)

    @cached_property
    def fwhm_hy(self):
        """Full width half max of the H-plane antenna beam for Y polarization."""
        return np.polyval(np.array(self._hywidth) * 2.0 * np.pi / 3.0, self.frequencies)

    # Set the approximate uv feed sizes
    @property
    def u_width(self):
        """The approximate physical width (in the u-direction) of the telescope."""
        return self.cylinder_width

    # v-width property override
    @property
    def v_width(self):
        """The approximate physical length (in the v-direction) of the telescope."""
        return 1.0

    # Set non-zero rotation angle for pathfinder and chime
    @property
    def rotation_angle(self):
        """Rotation from north towards west, in degrees."""
        if self.correlator == "pathfinder":
            from ch_ephem.observers import pathfinder

            return pathfinder.rotation

        if self.correlator == "chime":
            from ch_ephem.observers import chime

            return chime.rotation

        return 0.0

    def calculate_frequencies(self):
        """Override default version support specifying by frequency channel number."""
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
        """An index_map describing the inputs known to the telescope.

        Useful for generating synthetic datasets.
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

    @cached_property
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

                return pol

            return -1

        if self.stack_type == "redundant":
            return np.array([_feedclass(f) for f in self.feeds])

        if self.stack_type == "redundant_cyl":
            return np.array([_feedclass(f, redundant_cyl=True) for f in self.feeds])

        return np.array(
            [fi if tools.is_array(feed) else -1 for fi, feed in enumerate(self.feeds)]
        )

    @cached_property
    def polarisation(self):
        """Polarisation map.

        Returns
        -------
        pol : np.ndarray
            One-dimensional array with the polarization for each feed ('X' or 'Y').
        """

        def _pol(f):
            if tools.is_array(f):
                if tools.is_array_x(f):  # feed is X polarisation
                    return "X"
                # feed is Y polarisation
                return "Y"
            return "N"

        return np.asarray([_pol(f) for f in self.feeds], dtype=str)

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
                self.fwhm_ey[freq],
                self.fwhm_hy[freq],
                rot=rot,
            )
        elif tools.is_array_x(feed_obj):
            beam = cylbeam.beam_x(
                angpos,
                self.zenith,
                self.cylinder_width / self.wavelengths[freq],
                self.fwhm_ex[freq],
                self.fwhm_hx[freq],
                rot=rot,
            )
        else:
            raise RuntimeError(
                f"Given polarisation (feed.pol={feed_obj.pol}) not supported."
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
            super()._make_ew()

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
        blen = np.sum(bl1**2, axis=-1) ** 0.5

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
                np.sqrt(np.sum(beam**2, axis=-1))
            )

    def _skip_baseline(self, bl_ind):
        """Override to skip baselines based on which polarisation pair they are."""
        # Pull in baseline skip choice from parent class
        skip_bl = super()._skip_baseline(bl_ind)

        pol_i, pol_j = self.polarisation[self.uniquepairs[bl_ind]]
        pol_pair = pol_i + pol_j

        skip_pol = pol_pair in self.skip_pol_pair

        return skip_bl or skip_pol


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

    SIGMA_EW: ClassVar[float] = [14.87857614, 9.95746878]

    FUNC_NS: ClassVar = [_flat_top_gauss6, _flat_top_gauss3]
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
                f"Given polarisation (feed.pol={feed_obj.pol}) not supported."
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


class CHIMEFitBeam(CHIME):
    """Driftscan model with revised FWHM for north-south beam.

    Point source beam model was fit to a flat Gaussian at each frequency.
    Best-fit FWHM as a function of frequency was fit with a cubic
    polynomial. This class revises coefficients of FWHM from the
    fit. Detailed comparisons are documented in:
    https://bao.chimenet.ca/doc/documents/1448

    Note that the optimization was carried out in 585-800 MHz.
    This model will produce large extrapolation errors if used below that.
    """

    _eywidth = (
        3.0
        / (2 * np.pi)
        * np.array([1.15310483e-07, -2.30462590e-04, 1.50451290e-01, -3.07440520e01])
    )

    _hxwidth = (
        3.0
        / (2 * np.pi)
        * np.array([2.97495306e-07, -6.00582101e-04, 3.99949759e-01, -8.66733249e01])
    )


class CHIMEExternalBeam(CHIME):
    """Model telescope for the CHIME.

    This class uses an external beam model that is read in from a file.

    Attributes
    ----------
    primary_beam_filename : str
        Path to the file containing the primary beam. Can either be a Healpix beam or a
        GridBeam.
    freq_interp_beam : bool, optional
        Interpolate between neighbouring frequencies if we don't have a beam for every
        frequency channel.
    force_real_beam : bool, optional
        Ensure the output beam is real, regardless of what the datatype of the beam file
        is. This can help save memory if the saved beam is complex but you know the
        imaginary part is zero.
    """

    primary_beam_filename = config.Property(proptype=str)
    freq_interp_beam = config.Property(proptype=bool, default=False)
    force_real_beam = config.Property(proptype=bool, default=False)

    def _finalise_config(self):
        """Get the beam file object."""
        logger.debug(f"Reading beam model from {self.primary_beam_filename}...")
        self._primary_beam = ContainerBase.from_file(
            self.primary_beam_filename, mode="r", distributed=False, ondisk=True
        )

        self._is_grid_beam = isinstance(self._primary_beam, GridBeam)
        self._is_healpix_beam = isinstance(self._primary_beam, HEALPixBeam)

        # cache axes
        self._beam_freq = self._primary_beam.freq[:]
        self._beam_nside = None if self._is_grid_beam else self._primary_beam.nside

        # TODO must use bytestring here because conversion doesn't work with ondisk=True
        if self._is_grid_beam:
            self._beam_pol_map = {
                "X": list(self._primary_beam.pol[:]).index(b"XX"),
                "Y": list(self._primary_beam.pol[:]).index(b"YY"),
            }
        else:
            self._beam_pol_map = {
                "X": list(self._primary_beam.pol[:]).index(b"X"),
                "Y": list(self._primary_beam.pol[:]).index(b"Y"),
            }

        if len(self._primary_beam.input) > 1:
            raise ValueError("Per-feed beam model not supported for now.")

        # If a HEALPixBeam, must check types of theta and phi fields
        if self._is_healpix_beam:
            hpb_types = [
                v[0].type for v in self._primary_beam.beam.dtype.fields.values()
            ]

            complex_beam = np.all(
                [np.issubclass_(hpbt, np.complexfloating) for hpbt in hpb_types]
            )
        else:
            complex_beam = np.issubclass_(
                self._primary_beam.beam.dtype.type, np.complexfloating
            )

        self._output_dtype = (
            np.complex128 if complex_beam and not self.force_real_beam else np.float64
        )

        super()._finalise_config()

    def beam(self, feed, freq_id):
        """Get the beam pattern.

        Parameters
        ----------
        feed : int
            Feed index.
        freq_id : int
            Frequency ID.

        Returns
        -------
        beam : np.ndarray[pixel, pol]
            Return the vector beam response at each point in the Healpix grid. This
            array is of type `np.float64` if the input beam pattern is real, of
            `force_real_beam` is set, otherwise is is on type `np.complex128`.
        """
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
        # Raise an error if we can't find any suitable frequency
        if len(freq_sel) == 0:
            raise ValueError(f"No beam model spans frequency {tel_freq[freq_id]}.")

        if self._is_grid_beam:
            # Either we haven't set up interpolation coords yet, or the nside has
            # changed
            if (self._beam_nside is None) or (self._beam_nside != nside):
                self._beam_nside = nside
                self._setup_gridbeam_interpolation()

            # interpolate gridbeam onto HEALPix
            beam_map = self._interpolate_gridbeam(freq_sel, pol_ind)

        else:  # Healpix input beam just need to change to the required resolution
            beam_map = self._primary_beam.beam[freq_sel, pol_ind, 0, :]

            # Check resolution and resample to a better resolution if needed
            if nside != self._beam_nside:
                if nside > self._beam_nside:
                    logger.warning(
                        f"Requested nside={nside} higher than that of "
                        f"beam {self._beam_nside}"
                    )

                logger.debug(
                    f"Resampling external beam from nside {self._beam_nside:d} to {nside:d}"
                )
                beam_map_new = np.zeros((len(freq_sel), npix), dtype=beam_map.dtype)
                beam_map_new["Et"] = healpy.ud_grade(beam_map["Et"], nside)
                beam_map_new["Ep"] = healpy.ud_grade(beam_map["Ep"], nside)
                beam_map = beam_map_new

        map_out = np.empty((npix, 2), dtype=self._output_dtype)

        # Pull out the real part of the beam if we are forcing a conversion. This should
        # do nothing if the array is already real
        def _conv_real(x):
            if self.force_real_beam:
                return x.real
            return x

        if len(freq_sel) == 1:
            # exact match
            map_out[:, 0] = _conv_real(beam_map["Et"][0])
            map_out[:, 1] = _conv_real(beam_map["Ep"][0])
        else:
            # interpolate between pair of frequencies
            freq_high = self._beam_freq[freq_sel[1]]
            freq_low = self._beam_freq[freq_sel[0]]
            freq_int = tel_freq[freq_id]

            alpha = (freq_high - freq_int) / (freq_high - freq_low)
            beta = (freq_int - freq_low) / (freq_high - freq_low)

            map_out[:, 0] = _conv_real(
                beam_map["Et"][0] * alpha + beam_map["Et"][1] * beta
            )
            map_out[:, 1] = _conv_real(
                beam_map["Ep"][0] * alpha + beam_map["Ep"][1] * beta
            )

        return map_out

    def _setup_gridbeam_interpolation(self):
        # grid beam coordinates
        self._x_grid = self._primary_beam.phi[:]
        self._y_grid = self._primary_beam.theta[:]

        # celestial coordinates
        angpos = hputil.ang_positions(self._nside)
        x_cel = coord.sph_to_cart(angpos).T

        # rotate to telescope coords
        # first align y with N, then polar axis with NCP
        self._x_tel = cylbeam.rotate_ypr(
            (1.5 * np.pi, np.radians(90.0 - self.latitude), 0), *x_cel
        )

        # mask any pixels outside grid
        x_t, y_t, z_t = self._x_tel
        self._pix_mask = (
            (z_t > 0)
            & (np.abs(x_t) < np.abs(self._x_grid.max()))
            & (np.abs(y_t) < np.abs(self._y_grid.max()))
        )

        # pre-compute polarisation pattern
        # taken from driftscan
        zenith = np.array([np.pi / 2.0 - np.radians(self.latitude), 0.0])
        that, phat = coord.thetaphi_plane_cart(zenith)
        xhat, yhat, zhat = cylbeam.rotate_ypr(
            [-self.rotation_angle, 0.0, 0.0], phat, -that, coord.sph_to_cart(zenith)
        )

        self._pvec_x = cylbeam.polpattern(angpos, xhat)
        self._pvec_y = cylbeam.polpattern(angpos, yhat)

    def _interpolate_gridbeam(self, f_sel, p_ind):
        x, y = self._x_grid, self._y_grid
        x_t, y_t, z_t = self._x_tel
        mask = self._pix_mask

        # interpolation routine requires increasing axes
        reverse_x = (np.diff(self._x_grid) < 0).any()
        if reverse_x:
            x = x[::-1]

        npix = healpy.nside2npix(self._nside)
        beam_out = np.zeros(
            (len(f_sel), npix), dtype=HEALPixBeam._dataset_spec["beam"]["dtype"]
        )
        for i, fi in enumerate(f_sel):
            # For now we just use the magnitude. Assumes input is power beam
            beam = self._primary_beam.beam[fi, p_ind, 0]
            if reverse_x:
                beam = beam[:, ::-1]
            beam_spline = RectBivariateSpline(y, x, np.sqrt(np.abs(beam)))

            # beam amplitude
            amp = np.zeros(npix, dtype=beam.real.dtype)
            amp[mask] = beam_spline(y_t[mask], x_t[mask], grid=False)

            # polarisation projection
            pvec = self._pvec_x if self._beam_pol_map["X"] == p_ind else self._pvec_y

            beam_out[i]["Et"] = amp * pvec[:, 0]
            beam_out[i]["Ep"] = amp * pvec[:, 1]

        return beam_out


def _nearest_freq(tel_freq, map_freq, freq_id, single=False):
    """Find nearest neighbor frequencies. Assumes map frequencies are uniformly spaced.

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

    return np.nonzero(match_mask)[0]


class MakeTelescope(task.MPILoggedTask):
    r"""A simple task to construct a telescope object.

    This removes the need to use driftscan to create a saved beam transfer manager
    solely for running pipelines that only need the telescope object.

    Attributes
    ----------
    telescope_type : str
        The type of telescope object to create. Generally this must be a fully qualified
        Python class name, however, \"chime\" (default) can be used to create a
        `ch_pipeline.core.telescope.CHIME` instance.
    telescope_config : dict
        Configuration passed straight to the telescope class via it's `from_config`
        method.
    """

    telescope_type = config.Property(proptype=str, default="chime")
    telescope_config = config.Property(proptype=dict)

    def setup(self) -> telescope.TransitTelescope:
        """Create and return the telescope object."""
        _type_map = {
            "chime": CHIME,
        }

        if self.telescope_type in _type_map:
            tel_class = _type_map[self.telescope_type]
        else:
            tel_class = misc.import_class(self.telescope_type)

        return tel_class.from_config(self.telescope_config)
