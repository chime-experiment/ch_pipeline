"""Predict the tracks of satellites from their orbital elements and flag them.

Orbital elements are kept in an HDF5 file of two-line element sets (TLEs), one
row per element set, written by :func:`write_tle_file` (for example from a
Space-Track ``gp_history`` download, with :func:`parse_tle_text`).  The file has
the datasets, each with one entry per element set and sorted by satellite and
epoch:

- ``norad_id`` (int64): the NORAD catalogue number
- ``name`` (string): the name of the object
- ``epoch_jd`` (float64): the epoch of the element set, as a UTC Julian date
- ``line1``, ``line2`` (string): the two lines of the element set

and the attributes ``source`` (where the elements came from) and ``created``.

The position of a satellite at a time is computed with SGP4 (through skyfield)
from the element set nearest in epoch, as seen from CHIME, and expressed in
telescope coordinates ``x = cos(alt) sin(az)`` (east) and ``y = cos(alt) cos(az)``
(north), the coordinates of the hybrid beamformed maps.  The small rotation of
the cylinders (0.07 degrees) is ignored.
"""

import re
from datetime import datetime, timezone

import h5py
import numpy as np
import scipy.ndimage
from caput import config
from caput.pipeline import tasklib
from draco.core import containers as draco_containers

from .transients import (
    _band_power,
    _BandMixin,
    _is_full_day,
    _power_scale,
    _power_terms,
    _stream_lsd,
)

TLE_DTYPE = [
    ("norad_id", np.int64),
    ("name", "U32"),
    ("epoch_jd", np.float64),
    ("line1", "U69"),
    ("line2", "U69"),
]


def parse_tle_text(text):
    """Parse element sets in two-line or three-line format.

    Lines may be separated by newlines, carriage returns or apostrophes (as in some
    Space-Track downloads).  A line starting with "0 " gives the name of the next
    element set.  Element sets of the same satellite with the same epoch are kept
    once (the first).

    Returns
    -------
    tles : np.ndarray
        Structured array with the fields of ``TLE_DTYPE``, sorted by satellite and
        epoch.
    """
    from sgp4.api import Satrec

    lines = [ln.strip() for ln in re.split(r"\r\n|\r|\n|'", text) if ln.strip()]
    rows, seen, name = [], set(), ""
    ii = 0
    while ii < len(lines):
        ln = lines[ii]
        if ln.startswith("0 "):
            name = ln[2:].strip()
            ii += 1
            continue
        if (
            ln.startswith("1 ")
            and ii + 1 < len(lines)
            and lines[ii + 1].startswith("2 ")
        ):
            l1, l2 = ln, lines[ii + 1]
            sat = Satrec.twoline2rv(l1, l2)
            epoch = sat.jdsatepoch + sat.jdsatepochF
            key = (sat.satnum, round(epoch, 8))
            if key not in seen:
                seen.add(key)
                rows.append((sat.satnum, name, epoch, l1, l2))
            name = ""
            ii += 2
            continue
        raise ValueError(f"Unexpected line {ii}: {ln[:40]}")

    tles = np.array(rows, dtype=TLE_DTYPE)
    return tles[np.lexsort((tles["epoch_jd"], tles["norad_id"]))]


def write_tle_file(path, tles, source=""):
    """Write element sets (from :func:`parse_tle_text`) to an HDF5 file."""
    with h5py.File(path, "w") as f:
        f["norad_id"] = tles["norad_id"]
        f["epoch_jd"] = tles["epoch_jd"]
        str_type = h5py.string_dtype()
        for name in ("name", "line1", "line2"):
            f.create_dataset(name, data=tles[name].astype(object), dtype=str_type)
        f.attrs["source"] = source
        f.attrs["created"] = datetime.now(timezone.utc).isoformat()


def read_tle_file(path):
    """Read element sets written by :func:`write_tle_file`."""
    with h5py.File(path, "r") as f:
        tles = np.zeros(f["norad_id"].shape[0], dtype=TLE_DTYPE)
        tles["norad_id"] = f["norad_id"][:]
        tles["epoch_jd"] = f["epoch_jd"][:]
        for name in ("name", "line1", "line2"):
            tles[name] = f[name].asstr()[:]
    return tles


def nearest_tle(tles, norad_id, jd):
    """The element set of a satellite nearest in epoch to the Julian date ``jd``.

    Returns None if the file has none for the satellite.
    """
    sel = np.flatnonzero(tles["norad_id"] == norad_id)
    if sel.size == 0:
        return None
    return tles[sel[np.argmin(np.abs(tles["epoch_jd"][sel] - jd))]]


def satellite_tracks(tles, unix, norad_ids=None, max_epoch_offset=3.0):
    """Telescope coordinates of satellites as seen from CHIME.

    Parameters
    ----------
    tles : np.ndarray
        Element sets, from :func:`read_tle_file`.
    unix : np.ndarray
        Unix times.
    norad_ids : list of int, optional
        The satellites.  Default is all in ``tles``.
    max_epoch_offset : float
        Leave out satellites whose nearest element set is further than this many
        days from the middle of the times.

    Returns
    -------
    tracks : dict
        {norad_id: (x, y, alt, element_set)}, with ``x``, ``y`` and the altitude
        ``alt`` (degrees) at each time.
    """
    from ch_ephem.observers import chime
    from skyfield.api import EarthSatellite, load, wgs84

    ts = load.timescale()
    site = wgs84.latlon(chime.latitude, chime.longitude, elevation_m=chime.altitude)
    unix = np.atleast_1d(np.asarray(unix, dtype=np.float64))
    t = ts.from_datetimes([datetime.fromtimestamp(u, timezone.utc) for u in unix])
    jd_mid = 2440587.5 + np.median(unix) / 86400.0

    ids = np.unique(tles["norad_id"]) if norad_ids is None else list(norad_ids)
    tracks = {}
    for nid in ids:
        el = nearest_tle(tles, int(nid), jd_mid)
        if el is None or abs(el["epoch_jd"] - jd_mid) > max_epoch_offset:
            continue
        sat = EarthSatellite(str(el["line1"]), str(el["line2"]), str(el["name"]), ts)
        alt, az, _ = (sat - site).at(t).altaz()
        a, z = alt.radians, az.radians
        tracks[int(nid)] = (
            np.cos(a) * np.sin(z),
            np.cos(a) * np.cos(z),
            alt.degrees,
            el,
        )
    return tracks


class FlagSatellites(_BandMixin, tasklib.base.ContainerTask):
    """Flag the tracks of satellites predicted from their orbital elements.

    For each satellite in ``norad_ids`` with an element set within
    ``max_epoch_offset`` days of the data (in ``tle_file``, see
    :func:`write_tle_file`), its telescope coordinates are computed at every sample
    (see :func:`satellite_tracks`).  Where it is above ``min_alt`` and within
    ``x_max`` of the meridian in telescope-x, the strip of rows within ``half_dy``
    of its telescope-y is its track.

    With ``detect`` (the default) the track is flagged only in the bands and at the
    samples where the satellite is seen in the data: the band power of the
    filtered visibilities (as in :class:`~ch_pipeline.analysis.transients.
    FlagMovingRFI`, in units of the thermal noise and of its running median along
    RA over ``local_window`` samples on either side) is summed over the strip at
    each sample, and a band is flagged at the samples where the sum exceeds its
    expectation for noise by ``snr_detect`` standard deviations, and at the
    ``detect_dilate`` samples on either side.  Without ``detect`` the whole track
    is flagged in every band.  The flags are then grown by ``grow_dt`` samples.

    The output is a mask with one entry per band, like that of
    :class:`~ch_pipeline.analysis.transients.FlagMovingRFI`.  If another such mask
    is given as an optional requirement (for example the output of
    ``FlagMovingRFI``), the output is the union of the two.

    Attributes
    ----------
    tle_file : str
        HDF5 file of element sets (see :func:`write_tle_file`).
    norad_ids : list of int
        The satellites to flag.  Default is all in the file.
    max_epoch_offset : float
        Largest difference in days between the element set and the data.  Default
        is 3.
    min_alt : float
        Lowest altitude in degrees of a satellite to flag.  Default is 0.
    x_max : float
        Flag a satellite only within this of the meridian in telescope-x.  Default is
        0.2.
    half_dy : float
        Half-width of the track in telescope-y.  Default is 0.015.
    detect : bool
        Flag only the bands and samples where the satellite is seen.  Default is
        True.
    snr_detect : float
        Significance of a detection.  Default is 5.
    detect_dilate : int
        Samples on either side of a detection also flagged.  Default is 5.
    grow_dt : int
        Growth of the flags in samples.  Default is 1.
    pol : list of str
        Polarisations to use.  Default is ["XX", "YY"].
    ew_index : list of int
        Indices of the east-west baselines to use.  Default is all except 0 m.
    local_window : int
        Half-width in samples of the running median.  Default is 30.
    nsample_chunk : int
        Approximate number of samples averaged at once.  Default is 65536.
    """

    tle_file = config.Property(proptype=str)
    norad_ids = config.list_type(int, default=None)
    max_epoch_offset = config.Property(proptype=float, default=3.0)
    min_alt = config.Property(proptype=float, default=0.0)
    x_max = config.Property(proptype=float, default=0.2)
    half_dy = config.Property(proptype=float, default=0.015)
    detect = config.Property(proptype=bool, default=True)
    snr_detect = config.Property(proptype=float, default=5.0)
    detect_dilate = config.Property(proptype=int, default=5)
    grow_dt = config.Property(proptype=int, default=1)
    pol = config.list_type(str, default=["XX", "YY"])
    ew_index = config.list_type(int, default=None)
    local_window = config.Property(proptype=int, default=30)
    nsample_chunk = config.Property(proptype=int, default=65536)

    rfi_mask = None

    def setup(self, mask=None):
        """Set a mask to combine with the satellite flags.

        Parameters
        ----------
        mask : draco.core.containers.LocalizedSiderealRFIMask, optional
            Mask of bands, elevations and samples, on the same bands and grid.
        """
        self.rfi_mask = mask

    def process(self, stream):
        """Flag the satellites.

        Parameters
        ----------
        stream : ch_pipeline.core.containers.MFilteredHybridVisStream
            m-mode filtered visibilities, with the ``noise_scale`` dataset if the
            noise is to be measured relative to it.

        Returns
        -------
        mask : draco.core.containers.LocalizedSiderealRFIMask
            True for the flagged bands, elevations and samples.
        """
        from ch_ephem.observers import chime
        from scipy.stats import gamma

        if "elevation_vis_weight" in stream.datasets:
            raise ValueError("Elevation-dependent weights are not supported.")

        freq = stream.freq[:]
        band_index, band_map = self._assign_bands(freq)
        nband = band_map.size
        chans = [np.flatnonzero(band_index == bb) for bb in range(nband)]
        el = stream.index_map["el"][:]
        ra = stream.ra[:]
        nra = ra.size
        periodic = _is_full_day(ra) and bool(stream.attrs.get("ra_periodic", True))

        # Predicted tracks
        lsd = _stream_lsd(stream)
        unix = chime.lsd_to_unix(lsd + ra / 360.0)
        tracks = self._tracks(unix)
        self.log.info(
            f"Tracks of {len(tracks)} satellites with element sets within "
            f"{self.max_epoch_offset} days."
        )

        terms = _power_terms(stream, self.pol, self.ew_index, self.log)
        Z, nz, el_slc = _band_power(stream, terms, chans, self.nsample_chunk, self.log)
        el_local = el[el_slc]
        z_med = np.concatenate([[1.0], gamma(np.arange(1, len(terms) + 1)).median()])

        zs = None
        flag = np.zeros(Z.shape, dtype=bool)  # band, el_local, ra
        for nid, (x, y, alt, tle) in tracks.items():
            seen = (alt > self.min_alt) & (np.abs(x) < self.x_max)
            nseen = int(seen.sum())
            if nseen == 0:
                self.log.info(f"{tle['name']} ({nid}): not near the meridian.")
                continue
            strip = (
                np.abs(el_local[:, np.newaxis] - y[np.newaxis]) <= self.half_dy
            ) & (
                seen[np.newaxis]
            )  # el_local, ra

            if self.detect:
                if zs is None:
                    zs = np.zeros(Z.shape, dtype=np.float32)
                    for bb in range(nband):
                        s = _power_scale(
                            Z[bb], nz[bb], z_med, self.local_window, periodic
                        )
                        zs[bb] = np.where(nz[bb] > 0, Z[bb] / s, 0.0)
                S = np.sum(np.where(strip[np.newaxis], zs, 0.0), axis=1)  # band, ra
                N = np.sum(np.where(strip[np.newaxis], nz, 0), axis=1).astype(
                    np.float64
                )
                S = self.comm.allreduce(S)
                N = self.comm.allreduce(N)
                sig = np.where(N > 0, (S - N) / np.sqrt(np.maximum(N, 1.0)), 0.0)
                on = sig > self.snr_detect
                if self.detect_dilate > 0:
                    on = scipy.ndimage.binary_dilation(
                        on, structure=np.ones((1, 2 * self.detect_dilate + 1), bool)
                    )
                on &= seen[np.newaxis]
            else:
                on = np.broadcast_to(seen[np.newaxis], (nband, nra))

            flag |= strip[np.newaxis] & on[:, np.newaxis, :]
            bands_on = np.flatnonzero(on.any(axis=1))
            centres = ", ".join(f"{c:.0f}" for c in band_map["centre"][bands_on])
            self.log.info(
                f"{tle['name']} ({nid}): {nseen} samples near the meridian, flagged at "
                f"{int(on.any(axis=0).sum())} samples in {bands_on.size} bands"
                + (f" ({centres} MHz)." if bands_on.size else ".")
            )

        if self.grow_dt > 0 and flag.any():
            flag = scipy.ndimage.binary_dilation(
                flag, structure=np.ones((1, 1, 2 * self.grow_dt + 1), dtype=bool)
            )

        if self.rfi_mask is not None:
            flag |= self._other_mask(band_map, ra, el, el_slc)

        out = draco_containers.LocalizedSiderealRFIMask(
            freq=band_map,
            ra=ra,
            el=el,
            attrs_from=stream,
            distributed=stream.distributed,
            comm=stream.comm,
        )
        out.redistribute("el")
        out.mask[:].local_array[:] = flag.transpose(0, 2, 1)
        return out

    def _tracks(self, unix):
        """Predicted tracks {norad_id: (x, y, alt, element_set)} at the times."""
        tles = read_tle_file(self.tle_file)
        return satellite_tracks(tles, unix, self.norad_ids, self.max_epoch_offset)

    def _other_mask(self, band_map, ra, el, el_slc):
        """The local part of the mask given to setup, as [band, el, ra]."""
        m = self.rfi_mask
        fmap = m.index_map["freq"][:]
        if fmap.size != band_map.size or not np.allclose(
            fmap["centre"], band_map["centre"]
        ):
            raise ValueError("The bands of the given mask differ.")
        for ax, a in (("ra", ra), ("el", el)):
            b = m.index_map[ax][:]
            if a.shape != b.shape or not np.allclose(a, b):
                raise ValueError(f"The {ax} axis of the given mask differs.")
        m.redistribute("el")
        mm = m.mask[:]
        if mm.local_bounds != el_slc:
            raise RuntimeError("The given mask is not distributed like the data.")
        return mm.local_array.transpose(0, 2, 1)
