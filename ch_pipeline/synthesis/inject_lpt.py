"""Inject simulated long-period transients (LPTs) into sidereal streams.

Three tasks, used around the transient search:

- :class:`GenerateLPTCatalog` draws a random catalogue of LPT pulses for one
  sidereal day (reproducible: the random seed is set by ``seed`` and the LSD) and
  records where each should appear (the truth).
- :class:`InjectLPT` adds them to the stacked visibilities of the day, through the
  primary beam, before any processing.  It builds on the injection tasks of
  :mod:`ch_pipeline.synthesis.inject_modified` (catalogue as input, external beam,
  ICRS to CIRS), but only touches the samples of each pulse.
- :class:`MatchLPTInjections` matches the candidate table of the search to the
  catalogue.

Conventions for the polarisation.  The Stokes parameters are defined in the frame
of the feeds: Q > 0 along the X (east-west) feeds, so that for a source at the
centre of the beam XX = I + Q, YY = I - Q, XY = U + iV and YX = U - iV (the
uncalibrated phase between the X and Y arrays is not applied).  The polarisation
angle is psi = pol_angle + RM lambda^2 + q(HA), with pol_angle the angle from the
X feed at transit and lambda = 0, and q the parallactic angle (zero at transit),
which rotates the plane of polarisation relative to the feeds as the source
drifts.  The co-polar products take the beam of their own polarisation, and the
cross-polar products the geometric mean of the XX and YY beams (as the matched
filter does with ``cross_beam: copol``).
"""

import numpy as np
import scipy.constants
import scipy.ndimage
from caput import config
from caput.astro.coordinates import spherical
from caput.pipeline import tasklib
from draco.core import io
from draco.util import interferometry, tools
from scipy.special import erf

from ..core import containers
from .inject_modified import SIGMA, BaseInject, BeamExternal, _get_baseline_info, _icrs_to_cirs

# Angular speed of the sky in rad / s
_OMEGA = 2.0 * np.pi / 86164.0905


def _pulse_profile(t, dt, t0, width, profile):
    """Mean of the pulse profile (peak 1) over samples centred at t of length dt.

    ``tophat`` is 1 for |t - t0| < width / 2; ``gaussian`` has FWHM ``width``.
    """
    lo, hi = t - 0.5 * dt, t + 0.5 * dt
    if profile == "tophat":
        a, b = t0 - 0.5 * width, t0 + 0.5 * width
        return np.clip(np.minimum(hi, b) - np.maximum(lo, a), 0.0, None) / dt
    if profile == "gaussian":
        sig = width / (2.0 * np.sqrt(2.0 * np.log(2.0)))
        s2 = np.sqrt(2.0) * sig
        frac = 0.5 * (erf((hi - t0) / s2) - erf((lo - t0) / s2))
        return frac * np.sqrt(2.0 * np.pi) * sig / dt
    raise ValueError(f"Unknown profile {profile}.")


def _parallactic_angle(ha, lat, dec):
    """Parallactic angle (radians) at hour angle ``ha``; all angles in radians."""
    return np.arctan2(
        np.sin(ha), np.tan(lat) * np.cos(dec) - np.sin(dec) * np.cos(ha)
    )


def _scintillation(freq, index, bw, seed):
    """Multiplicative scintillation gain on the frequencies ``freq`` (MHz).

    The intensity of a complex Gaussian field correlated over ``bw`` MHz (an
    exponentially distributed gain with unit mean when fully modulated), mixed with
    a constant: (1 - index) + index |z|^2.  The same on every rank for a seed.
    """
    if index <= 0.0 or bw <= 0.0:
        return np.ones(freq.size)
    order = np.argsort(freq)
    f = freq[order]
    # A fixed grid (a quarter of the CHIME channel width from 380 MHz), so the field
    # of a seed does not depend on which channels are present
    df = 0.390625 / 4
    lo = 380.0 - df * np.ceil(max(380.0 - f[0], 0.0) / df)
    grid = np.arange(lo, max(f[-1], 820.0) + df, df)
    rng = np.random.default_rng(seed)
    z = rng.standard_normal(grid.size) + 1j * rng.standard_normal(grid.size)
    sig = bw / df / (2.0 * np.sqrt(2.0 * np.log(2.0)))
    zr = scipy.ndimage.gaussian_filter1d(z.real, sig, mode="wrap")
    zi = scipy.ndimage.gaussian_filter1d(z.imag, sig, mode="wrap")
    p = zr**2 + zi**2
    p /= p.mean()
    g = np.interp(f, grid, p)
    out = np.empty(freq.size)
    out[order] = (1.0 - index) + index * g
    return out


class GenerateLPTCatalog(tasklib.base.ContainerTask):
    """Draw a random catalogue of long-period transient pulses for one day.

    The sources transit at RAs spread over ``ra_range`` (at least ``min_sep``
    degrees apart), at declinations uniform in sin(dec) over ``dec_range``.  Each
    has one pulse, centred ``t_offset`` seconds from its transit (uniform within
    +-``t_offset_sigma`` times the width sigma of the XX primary beam in hour angle
    at 600 MHz), of the given ``profile`` and ``duration`` (log-uniform), with a
    peak flux (log-uniform, Jy at ``reference_freq``) and a spectrum
    ``(nu / nu_ref)^alpha exp(beta ln^2(nu / nu_ref))``.  Linear and circular
    polarisation fractions, rotation measure and polarisation angle are uniform
    (the fractions scaled down together if L^2 + V^2 > 1), and, if
    ``scint_index_range`` is not zero, a scintillation with that modulation index
    and a bandwidth log-uniform in ``scint_bw_range`` (MHz).

    The seed of the random draws is (``seed``, LSD), so a day always gets the same
    catalogue.  The truth on the day (CIRS position, telescope-y at transit, and the
    telescope-x and meridian RA at the start of the pulse) is computed with the
    telescope's ICRS to CIRS conversion, as :class:`InjectLPT` does.

    Attributes
    ----------
    n_source : int
        Number of pulses.  Default is 20.
    seed : int
        Base random seed.  Default is 0.
    profile : {"tophat", "gaussian"}
        Pulse profile (``duration`` is its width or FWHM).  Default is "tophat".
    reference_freq : float
        Frequency (MHz) of the flux and spectrum.  Default is 600.
    ra_range, dec_range : list of float
        Ranges in degrees.  Defaults are [0, 360] and [-10, 89].
    min_sep : float
        Smallest separation in RA (degrees) between transits.  Default is 3.
    flux_range : list of float
        Peak flux in Jy (log-uniform).  Default is [0.02, 2].
    duration_range : list of float
        Duration in s (log-uniform).  Default is [10, 300].
    spectral_index_range, curvature_range : list of float
        Uniform ranges.  Defaults are [-3, 1] and [0, 0].
    t_offset_sigma : float
        Half-range of the pulse time about transit, in beam sigmas.  Default is 1.
    frac_lin_range, frac_circ_range : list of float
        Defaults are [0, 1] and [-0.5, 0.5].
    rm_range : list of float
        Rotation measure (rad / m^2).  Default is [-300, 300].
    scint_index_range, scint_bw_range : list of float
        Defaults are [0, 0] (none) and [0.5, 20] MHz.
    """

    n_source = config.Property(proptype=int, default=20)
    seed = config.Property(proptype=int, default=0)
    profile = config.enum(["tophat", "gaussian"], default="tophat")
    reference_freq = config.Property(proptype=float, default=600.0)
    ra_range = config.list_type(type_=float, length=2, default=[0.0, 360.0])
    dec_range = config.list_type(type_=float, length=2, default=[-10.0, 89.0])
    min_sep = config.Property(proptype=float, default=3.0)
    flux_range = config.list_type(type_=float, length=2, default=[0.02, 2.0])
    duration_range = config.list_type(type_=float, length=2, default=[10.0, 300.0])
    spectral_index_range = config.list_type(type_=float, length=2, default=[-3.0, 1.0])
    curvature_range = config.list_type(type_=float, length=2, default=[0.0, 0.0])
    t_offset_sigma = config.Property(proptype=float, default=1.0)
    frac_lin_range = config.list_type(type_=float, length=2, default=[0.0, 1.0])
    frac_circ_range = config.list_type(type_=float, length=2, default=[-0.5, 0.5])
    rm_range = config.list_type(type_=float, length=2, default=[-300.0, 300.0])
    scint_index_range = config.list_type(type_=float, length=2, default=[0.0, 0.0])
    scint_bw_range = config.list_type(type_=float, length=2, default=[0.5, 20.0])

    def setup(self, manager):
        """Get the telescope.

        Parameters
        ----------
        manager : ProductManager, BeamTransfer or TransitTelescope
        """
        self.telescope = io.get_telescope(manager)
        self.latitude = np.radians(self.telescope.latitude)

    def process(self, data):
        """Draw the catalogue for the day of ``data``.

        Parameters
        ----------
        data : SiderealStream (or any container with an ``lsd`` attribute)

        Returns
        -------
        catalog : containers.LPTCatalog
        """
        lsd = int(np.atleast_1d(data.attrs["lsd"])[0])
        rng = np.random.default_rng([self.seed, lsd])
        n = self.n_source

        def uniform(r):
            return rng.uniform(r[0], r[1], n) if r[1] > r[0] else np.full(n, r[0])

        def loguniform(r):
            return np.exp(uniform(np.log(r)))

        # RAs at least min_sep apart (on the circle)
        lo, hi = self.ra_range
        ra = []
        for _ in range(100 * n):
            if len(ra) == n:
                break
            r = rng.uniform(lo, hi)
            if all(abs((r - q + 180.0) % 360.0 - 180.0) >= self.min_sep for q in ra):
                ra.append(r)
        if len(ra) < n:
            raise RuntimeError(f"Could not place {n} sources {self.min_sep} deg apart.")
        ra = np.array(ra) % 360.0
        sdec = rng.uniform(*np.sin(np.radians(self.dec_range)), n)
        dec = np.degrees(np.arcsin(sdec))

        flux = loguniform(self.flux_range)
        duration = loguniform(self.duration_range)
        alpha = uniform(self.spectral_index_range)
        beta = uniform(self.curvature_range)
        sig_ha = SIGMA["XX"] / 600.0 / np.cos(np.radians(dec))  # rad
        t_off = rng.uniform(-1.0, 1.0, n) * self.t_offset_sigma * sig_ha / _OMEGA
        lin = uniform(self.frac_lin_range)
        circ = uniform(self.frac_circ_range)
        tot = np.hypot(lin, circ)
        lin, circ = lin / np.maximum(tot, 1.0), circ / np.maximum(tot, 1.0)
        rm = uniform(self.rm_range)
        angle = rng.uniform(0.0, 180.0, n)
        sindex = uniform(self.scint_index_range)
        sbw = loguniform(self.scint_bw_range)
        sseed = rng.integers(0, 2**31 - 1, n)

        # The truth on this day
        epoch = self.telescope.lsd_to_unix(lsd)
        ra_c, dec_c = _icrs_to_cirs(ra, dec, epoch, self.telescope)
        ra_c, dec_c = np.atleast_1d(ra_c) % 360.0, np.atleast_1d(dec_c)
        dec_r = np.radians(dec_c)
        _, y, _ = spherical.sph_to_ground(0.0, self.latitude, dec_r)
        # start of the pulse (for the Gaussian, half the FWHM before the peak)
        ha_start = _OMEGA * (t_off - 0.5 * duration)
        x_start, _, _ = spherical.sph_to_ground(ha_start, self.latitude, dec_r)
        ra_start = (ra_c + np.degrees(ha_start)) % 360.0
        unix_center = np.array(
            [self.telescope.lsd_to_unix(lsd + r / 360.0) for r in ra_c]
        ) + t_off

        cat = containers.LPTCatalog(object_id=np.arange(n))
        cat["position"]["ra"][:] = ra
        cat["position"]["dec"][:] = dec
        cat["position"].attrs["coords"] = "icrs"
        tab = cat["lpt"]
        for name, val in (
            ("flux", flux),
            ("spectral_index", alpha),
            ("spectral_curvature", beta),
            ("duration", duration),
            ("t_offset", t_off),
            ("frac_lin", lin),
            ("frac_circ", circ),
            ("rm", rm),
            ("pol_angle", angle),
            ("scint_index", sindex),
            ("scint_bw", sbw),
            ("scint_seed", sseed),
            ("ra_cirs", ra_c),
            ("dec_cirs", dec_c),
            ("y", y),
            ("x_start", x_start),
            ("ra_start", ra_start),
            ("unix_center", unix_center),
        ):
            tab[name][:] = val
        cat.attrs["tag"] = f"lpt_lsd_{lsd}"
        cat.attrs["lsd"] = lsd
        cat.attrs["seed"] = self.seed
        cat.attrs["profile"] = self.profile
        cat.attrs["reference_freq"] = self.reference_freq
        self.log.info(
            f"Drew {n} LPT pulses for LSD {lsd}: flux {flux.min():.3f}-{flux.max():.3f} "
            f"Jy, duration {duration.min():.0f}-{duration.max():.0f} s."
        )
        return cat


class InjectLPT(BeamExternal):
    """Inject the pulses of an :class:`GenerateLPTCatalog` catalogue into a stream.

    For each source, the samples where its pulse profile (averaged over the sample)
    exceeds ``min_profile`` and it is above the horizon get

        V_pp(nu, t) = A_pp(nu, t) B_pp(nu, HA) exp(2 pi i b . s / lambda)

    with the visibility phase of :mod:`inject_modified` (the conjugate of the
    fringe-stopping phase), the beam ``B`` from the external beam (the geometric
    mean of XX and YY for the cross-polar products) and ``A`` the Stokes
    combination of the module docstring times the spectrum, the scintillation and
    the profile.  Only the samples of each pulse are touched.

    Attributes
    ----------
    polarization : {"copol", "full", "single"}
        Products to inject into (see :class:`BaseInject`).  Default is "full".
    inject : bool
        Add to the input stream (default True) or to a copy with zero visibilities.
    min_profile : float
        Smallest mean profile of a sample to inject.  Default is 1e-4.
    parallactic : bool
        Rotate the polarisation by the parallactic angle.  Default is True.
    """

    polarization = config.enum(["full", "copol", "single"], default="full")
    inject = config.Property(proptype=bool, default=True)
    min_profile = config.Property(proptype=float, default=1e-4)
    parallactic = config.Property(proptype=bool, default=True)

    def setup(self, manager, catalog, beam):
        """Get the telescope and the catalogue, and set up the co-polar beams.

        Parameters
        ----------
        manager : ProductManager, BeamTransfer or TransitTelescope
        catalog : containers.LPTCatalog
        beam : GridBeam
            Primary beam (celestial or telescope coordinates) with XX and YY.
        """
        BaseInject.setup(self, manager, catalog)
        inject_pol = list(self.process_pol)
        self.process_pol = ["XX", "YY"]
        self._initialize_beam(beam)
        self.process_pol = inject_pol
        self.profile = catalog.attrs.get("profile", "tophat")
        self.reference_freq = float(catalog.attrs.get("reference_freq", 600.0))

    def process(self, data):
        """Inject the pulses.

        Parameters
        ----------
        data : SiderealStream
            Stacked visibilities of one day, with an ``lsd`` attribute.

        Returns
        -------
        out : SiderealStream
            The input (``inject``) or a copy with only the pulses.
        """
        if "ra" not in data.index_map:
            raise ValueError("InjectLPT needs a stream with an RA axis.")
        ra = np.radians(data.index_map["ra"][:])
        lsd = np.mean(data.attrs["lsd"]) if "lsd" in data.attrs else None
        self.epoch = self.telescope.lsd_to_unix(lsd) if lsd is not None else None

        data.redistribute("freq")
        if self.inject:
            out = data
        else:
            out = data.__class__(
                axes_from=data, attrs_from=data, distributed=data.distributed, comm=data.comm
            )
            out.redistribute("freq")
            out.vis[:] = 0.0
            out.weight[:] = data.weight[:].copy() if self.copy_weight else 1.0
        out.attrs["lpt_catalog"] = self.catalog_tag

        vis = out.vis[:]
        lo, nf = vis.local_offset[0], vis.local_shape[0]
        freq_all = out.index_map["freq"]["centre"][:]
        self.freq_local = out.index_map["freq"][lo : lo + nf]
        nu = self.freq_local["centre"]
        lam = scipy.constants.c / (nu * 1e6)

        self._process_catalog()
        self._initialize_beam_with_data()

        dist, pol = _get_baseline_info(out, self.telescope)
        pol_index = {p: np.flatnonzero(pol == p) for p in self.process_pol}
        pol_index = {p: v for p, v in pol_index.items() if v.size}
        ib = np.sort(np.concatenate(list(pol_index.values())))
        pos = {p: np.searchsorted(ib, v) for p, v in pol_index.items()}
        uv = dist[:, ib][:, np.newaxis, :, np.newaxis] / lam[np.newaxis, :, np.newaxis, np.newaxis]

        dra = np.median(np.diff(ra))
        dt = dra / _OMEGA
        tab = self.catalog["lpt"][:]
        x_nu = nu / self.reference_freq
        ninj = 0
        for ss in range(self.nsource):
            sra, sdec = self.sra[ss], self.sdec[ss]
            ha = tools.correct_phase_wrap(ra - sra)
            sin_alt = np.sin(sdec) * np.sin(self.latitude) + np.cos(sdec) * np.cos(
                self.latitude
            ) * np.cos(ha)
            w = _pulse_profile(
                ha / _OMEGA, dt, tab["t_offset"][ss], tab["duration"][ss], self.profile
            )
            idx = np.flatnonzero((w > self.min_profile) & (sin_alt > 0.0))
            if idx.size == 0:
                self.log.info(f"Source {ss} has no samples to inject.")
                continue
            ninj += 1

            spec = (
                tab["flux"][ss]
                * x_nu ** tab["spectral_index"][ss]
                * np.exp(tab["spectral_curvature"][ss] * np.log(x_nu) ** 2)
            )
            gain = _scintillation(
                freq_all, tab["scint_index"][ss], tab["scint_bw"][ss], tab["scint_seed"][ss]
            )[lo : lo + nf]
            spec = spec * gain

            for slc in tools.find_contiguous_slices(idx):
                sha = ha[slc]
                stokes_i = spec[:, np.newaxis] * w[slc][np.newaxis]  # freq, time
                psi = np.radians(tab["pol_angle"][ss]) + tab["rm"][ss] * lam[:, np.newaxis] ** 2
                if self.parallactic:
                    psi = psi + _parallactic_angle(sha, self.latitude, sdec)[np.newaxis]
                lin = tab["frac_lin"][ss] * stokes_i
                q, u = lin * np.cos(2 * psi), lin * np.sin(2 * psi)
                v = tab["frac_circ"][ss] * stokes_i

                bxx = self._beamfunc(nu, "XX", sdec, sha)
                byy = self._beamfunc(nu, "YY", sdec, sha)
                bxy = np.sqrt(np.abs(bxx * byy))
                amp = {
                    "XX": (stokes_i + q) * bxx,
                    "YY": (stokes_i - q) * byy,
                    "XY": (u + 1j * v) * bxy,
                    "YX": (u - 1j * v) * bxy,
                }
                phase = interferometry.fringestop_phase(
                    sha[np.newaxis, np.newaxis, :], self.latitude, sdec, *uv
                ).conj()  # freq, baseline, time
                for p, ii in pos.items():
                    phase[:, ii] *= amp[p][:, np.newaxis, :]
                vis.local_array[:, ib, slc] += phase.astype(vis.local_array.dtype)

        self.log.info(f"Injected {ninj} of {self.nsource} LPT pulses.")
        out.attrs["lpt_ninjected"] = ninj
        return out


class MatchLPTInjections(tasklib.base.ContainerTask):
    """Match the candidates of the search to the injected pulses.

    An injection is found if a candidate lies within ``match_dy`` of its
    telescope-y at transit and starts within the pulse, widened by ``match_dt``
    seconds on each side (start times in RA of the meridian).  Of several, the one
    with the highest S/N is taken.

    Attributes
    ----------
    match_dy : float
        Half-width in telescope-y.  Default is 0.01.
    match_dt : float
        Extra time in seconds on each side of the pulse.  Default is 30.
    """

    match_dy = config.Property(proptype=float, default=0.01)
    match_dt = config.Property(proptype=float, default=30.0)

    def process(self, table, catalog):
        """Match.

        Parameters
        ----------
        table : containers.TransientCandidateTable
            Sifted candidates (the ``sift`` table is used if present).
        catalog : containers.LPTCatalog
            The injected pulses.

        Returns
        -------
        out : containers.LPTInjectionResults
        """
        rec = table.candidate[:]
        sift = table.datasets["sift"][:] if "sift" in table.datasets else None
        lpt = catalog["lpt"][:]
        n = lpt.size
        out = containers.LPTInjectionResults(object_id=n, attrs_from=catalog)
        res = out.injection[:].copy()
        res["ra"] = catalog["position"]["ra"][:]
        res["dec"] = catalog["position"]["dec"][:]
        for name in ("flux", "spectral_index", "duration", "t_offset", "frac_lin",
                     "frac_circ", "y", "x_start", "ra_start"):  # fmt: skip
            res[name] = lpt[name]
        res["candidate"] = -1
        for name in ("snr", "amplitude", "cand_duration", "cand_spectral_index",
                     "cand_y", "cand_x", "cand_ra", "ns_scale", "ns_dchi2"):  # fmt: skip
            res[name] = np.nan

        deg_s = np.degrees(_OMEGA)
        for k in range(n):
            d_ra = (rec["ra"] - lpt["ra_start"][k] + 180.0) % 360.0 - 180.0
            win = (d_ra >= -self.match_dt * deg_s) & (
                d_ra <= (lpt["duration"][k] + self.match_dt) * deg_s
            )
            near = win & (np.abs(rec["y"] - lpt["y"][k]) <= self.match_dy)
            res["n_match"][k] = near.sum()
            if not near.any():
                continue
            j = np.flatnonzero(near)[np.argmax(rec["snr"][near])]
            res["found"][k] = True
            res["candidate"][k] = j
            res["snr"][k] = rec["snr"][j]
            res["amplitude"][k] = rec["amplitude"][j]
            res["cand_duration"][k] = rec["duration"][j]
            res["cand_spectral_index"][k] = rec["spectral_index"][j]
            res["cand_y"][k] = rec["y"][j]
            res["cand_x"][k] = rec["x"][j]
            res["cand_ra"][k] = rec["ra"][j]
            res["ns_scale"][k] = rec["ns_scale"][j]
            res["ns_dchi2"][k] = rec["ns_dchi2"][j]
            if sift is not None:
                res["keep"][k] = sift["keep"][j]
                res["flags"][k] = sift["flags"][j]
        out.injection[:] = res
        if "sift_flags" in table.attrs:
            out.attrs["sift_flags"] = table.attrs["sift_flags"]
        out.attrs["match_dy"] = self.match_dy
        out.attrs["match_dt"] = self.match_dt
        self.log.info(
            f"Found {res['found'].sum()} of {n} injected pulses; "
            f"{(res['found'] & res['keep']).sum()} kept by the sifter."
        )
        return out
