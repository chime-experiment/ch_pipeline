"""Inject simulated long-period transients (LPTs) into the transient search.

The catalogue holds the intrinsic properties of the sources only
(:class:`ch_pipeline.core.containers.LPTCatalog`).  Two ways to make one:

- :class:`GenerateLPTCatalog` draws a fixed catalogue once (to be saved and used
  for every day): each source pulses with a period, at a fixed position, so the
  same sources reappear on the days their pulses fall in the main lobe.  This tests
  the combination of days and folds in CHIME's observing window.
- :class:`GenerateDailyLPTCatalog` draws a new catalogue every day (seeded by the
  LSD), each source with one pulse near its transit.  This samples sky position and
  day for the selection function of the single-day search.

The properties are drawn with :mod:`ch_pipeline.synthesis.lpt_population`.

Each day, :class:`InjectLPT` finds the pulses that fall within ``window_sigma``
primary-beam widths of transit (periodic in barycentric time if ``barycentric``),
writes them to an :class:`~ch_pipeline.core.containers.LPTInjectionLog` (where and
when each appears: the truth), and adds them to a copy of the stacked visibilities
restricted to the samples they touch, with the masked weights of the day.  The
search config grids and beamforms that sparse stream as it does the data (both
are linear and act sample by sample) and, once the real search is done,
:class:`AddInjectedHybrid` adds the result in place to the unfiltered hybrid
stream, from which the search runs again.  :class:`MatchLPTInjections` then
matches the injected pulses to the candidates of that second search.

Conventions for the polarisation.  The Stokes parameters are defined in the frame
of the feeds: Q > 0 along the X (east-west) feeds, so that for a source at the
centre of the beam XX = I + Q, YY = I - Q, XY = U + iV and YX = U - iV (the
uncalibrated phase between the X and Y arrays is not applied).  The polarisation
angle is psi = pol_angle + RM lambda^2 + q(HA), with pol_angle the angle from the
X feed at transit and lambda = 0, and q the parallactic angle (zero at transit).
The co-polar products take the beam of their own polarisation, and the cross-polar
products the geometric mean of the XX and YY beams (as the matched filter does with
``cross_beam: copol``).
"""

import json

import numpy as np
import scipy.constants
import scipy.ndimage
from caput import config
from caput.astro.coordinates import spherical
from caput.pipeline import exceptions, tasklib
from draco.core import io
from draco.util import interferometry, tools
from scipy.special import erf

from ..core import containers
from .inject_modified import SIGMA, BaseInject, BeamExternal, _get_baseline_info
from .lpt_population import draw_population, merge_population

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




def _beam_sigma_time(dec_deg):
    """Width sigma (s) of the XX primary beam at 600 MHz, in time, at a declination."""
    return SIGMA["XX"] / 600.0 / np.cos(np.radians(dec_deg)) / _OMEGA


def _roemer_delay(unix, ra, dec):
    """Delay (s) of a pulse at the solar-system barycentre after its arrival at the Earth.

    The projection of the Earth's barycentric position on the direction of the
    source (ICRS degrees), divided by c: a pulse that reaches the barycentre at t
    reaches the Earth at t - delay.
    """
    from caput.astro import skyfield as skf

    t = skf.unix_to_skyfield_time(np.atleast_1d(np.asarray(unix, dtype=float)))
    r = skf.skyfield_wrapper.ephemeris["earth"].at(t).position.m  # (3, n), ICRS
    ra, dec = np.radians(ra), np.radians(dec)
    s = np.array([np.cos(dec) * np.cos(ra), np.cos(dec) * np.sin(ra), np.sin(dec) * np.ones_like(ra)])
    return np.sum(r * s.reshape(3, -1), axis=0) / scipy.constants.c


def _make_catalog(pop, seed_rng, epoch_ref, attrs):
    n = pop["ra"].size
    cat = containers.LPTCatalog(object_id=np.arange(n))
    cat["position"]["ra"][:] = pop["ra"]
    cat["position"]["dec"][:] = pop["dec"]
    cat["position"].attrs["coords"] = "icrs"
    tab = cat["lpt"]
    period = pop["period"]
    epoch = np.where(period > 0, epoch_ref + seed_rng.uniform(0.0, 1.0, n) * period, np.nan)
    for name in ("flux", "flux_scatter", "null_fraction", "period", "t_offset_beam", "duration",
                 "spectral_index", "spectral_curvature", "frac_lin", "frac_circ", "pol_angle", "rm",
                 "scint_index", "scint_bw", "distance"):  # fmt: skip
        tab[name][:] = pop[name]
    tab["epoch"][:] = epoch
    tab["seed"][:] = seed_rng.integers(0, 2**62, n)
    for k, v in attrs.items():
        cat.attrs[k] = v
    return cat


class GenerateLPTCatalog(tasklib.base.ContainerTask):
    """Draw a fixed catalogue of periodic LPTs (once; save it and use it every day).

    Attributes
    ----------
    n_source : int
        Number of sources.  Default is 80 (with the default periods and a window of
        +-2 beam sigma, ~60-100 pulses fall in the main lobe each day).
    seed : int
        Random seed.  Default is 0.
    profile : {"tophat", "gaussian"}
        Pulse profile (``duration`` is its width or FWHM).  Default is "tophat".
    reference_freq : float
        Frequency (MHz) of the flux and spectrum.  Default is 600.
    epoch_ref : float
        The epoch of each source is ``epoch_ref`` plus a random fraction of its
        period (unix time at the barycentre).  Default is 2020-01-01 00:00 UTC.
    population : dict
        Distributions of the properties, updating
        :data:`ch_pipeline.synthesis.lpt_population.DEFAULT_POPULATION` (see
        :func:`~ch_pipeline.synthesis.lpt_population.draw_population`).
    """

    n_source = config.Property(proptype=int, default=80)
    seed = config.Property(proptype=int, default=0)
    profile = config.enum(["tophat", "gaussian"], default="tophat")
    reference_freq = config.Property(proptype=float, default=600.0)
    epoch_ref = config.Property(proptype=float, default=1577836800.0)
    population = config.Property(proptype=dict, default={})

    _done = False

    def _attrs(self, tag, spec):
        return {
            "tag": tag,
            "seed": self.seed,
            "profile": self.profile,
            "reference_freq": self.reference_freq,
            "population": json.dumps(spec),
        }

    def process(self):
        """Draw the catalogue (the first call only).

        Returns
        -------
        catalog : containers.LPTCatalog
        """
        if self._done:
            raise exceptions.PipelineStopIteration
        self._done = True
        spec = merge_population(self.population)
        rng = np.random.default_rng(self.seed)
        pop = draw_population(rng, self.n_source, spec)
        cat = _make_catalog(pop, rng, self.epoch_ref, self._attrs(f"lpt_seed{self.seed}", spec))
        self.log.info(
            f"Drew {self.n_source} LPTs: periods {np.min(pop['period']):.0f}-{np.max(pop['period']):.0f} s, "
            f"flux {pop['flux'].min():.4f}-{pop['flux'].max():.3f} Jy."
        )
        return cat


class GenerateDailyLPTCatalog(GenerateLPTCatalog):
    """Draw a new catalogue for each day, each source pulsing once near its transit.

    As :class:`GenerateLPTCatalog`, but seeded by (``seed``, LSD of the input), with
    the period set to 0: one pulse a day, ``t_offset_beam`` (from the population)
    primary-beam widths from transit.  For the selection function of the single-day
    search, marginalised over sky position and day.  Default ``n_source`` is 30.
    """

    n_source = config.Property(proptype=int, default=30)

    def process(self, data):
        """Draw the catalogue for the day of ``data`` (any container with ``lsd``).

        Returns
        -------
        catalog : containers.LPTCatalog
        """
        lsd = int(np.atleast_1d(data.attrs["lsd"])[0])
        spec = merge_population(self.population)
        spec["period"] = 0.0
        rng = np.random.default_rng([self.seed, lsd])
        pop = draw_population(rng, self.n_source, spec)
        attrs = self._attrs(f"lpt_daily_lsd_{lsd}", spec)
        attrs["lsd"] = lsd
        cat = _make_catalog(pop, rng, self.epoch_ref, attrs)
        self.log.info(f"Drew {self.n_source} single LPT pulses for LSD {lsd}.")
        return cat


class InjectLPT(BeamExternal):
    """Find the pulses of a catalogue on the day and inject them (sparse stream).

    The transit of each source on the day is at the meridian RA of its CIRS
    position (in [0, 360) for the LSD of the data).  A source with a period pulses
    at ``epoch + k period`` at the barycentre (at the Earth earlier by the Roemer
    delay if ``barycentric``); those pulses whose centre is within ``window_sigma``
    widths of the XX primary beam at 600 MHz (plus half the duration) of transit are
    taken.  A source with period 0 has one pulse at ``t_offset_beam`` beam widths
    from transit.  Each pulse gets its own random draws (seeded by the source and
    the pulse number, so the same on any day that sees it): nulled with probability
    ``null_fraction``, a flux scattered by exp(``flux_scatter`` N(0, 1)), and the
    seed of its scintillation.

    The pulses are added to a copy of the input restricted to the samples where
    any has a mean profile above ``min_profile`` (and is above the horizon), with
    the input's weights there and zero visibilities otherwise:

        V_pp(nu, t) = A_pp(nu, t) B_pp(nu, HA) exp(2 pi i b . s / lambda)

    with the phase of :mod:`inject_modified` (the conjugate of the fringe-stopping
    phase), the beam of the external beam model (the geometric mean of XX and YY for
    the cross-polar products) and ``A`` the Stokes combination of the module
    docstring times the spectrum, the scintillation and the profile.  The input is
    not modified.

    Attributes
    ----------
    polarization : {"copol", "full", "single"}
        Products to inject into (see :class:`BaseInject`).  Default is "full".
    window_sigma : float
        Half-width of the window around transit in beam widths.  Default is 2.
    barycentric : bool
        Periodic in barycentric time.  Default is True.
    min_profile : float
        Smallest mean profile of a sample to inject.  Default is 1e-4.
    parallactic : bool
        Rotate the polarisation by the parallactic angle.  Default is True.
    """

    polarization = config.enum(["full", "copol", "single"], default="full")
    window_sigma = config.Property(proptype=float, default=2.0)
    barycentric = config.Property(proptype=bool, default=True)
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

    def _schedule(self, lsd):
        """The pulses of the day: a dict of arrays, one entry per pulse."""
        tab = self.catalog["lpt"][:]
        ra_i, dec_i = self.catalog["position"]["ra"][:], self.catalog["position"]["dec"][:]
        ra_c, dec_c = np.degrees(self.sra) % 360.0, np.degrees(self.sdec)
        t_tr = self.telescope.lsd_to_unix(lsd + ra_c / 360.0)
        sig_t = _beam_sigma_time(dec_c)
        half = self.window_sigma * sig_t + 0.5 * tab["duration"]
        rows = []
        for s in range(tab.size):
            per = tab["period"][s]
            if per > 0:
                delay = _roemer_delay(t_tr[s], ra_i[s], dec_i[s])[0] if self.barycentric else 0.0
                ep = tab["epoch"][s]
                k0 = int(np.ceil((t_tr[s] - half[s] + delay - ep) / per))
                k1 = int(np.floor((t_tr[s] + half[s] + delay - ep) / per))
                pulses = [(k, ep + k * per - delay) for k in range(k0, k1 + 1)]
            else:
                pulses = [(int(lsd), t_tr[s] + tab["t_offset_beam"][s] * sig_t[s])]
            for k, tc in pulses:
                rng = np.random.default_rng([int(tab["seed"][s]), int(k) + 2**40])
                nulled = rng.uniform() < tab["null_fraction"][s]
                flux = tab["flux"][s] * np.exp(tab["flux_scatter"][s] * rng.standard_normal())
                rows.append((s, k, nulled, tc, tc - t_tr[s], flux, int(rng.integers(0, 2**31 - 1))))
        out = {
            name: np.array([r[i] for r in rows])
            for i, name in enumerate(("source", "pulse", "nulled", "unix_center", "t_offset", "flux", "scint_seed"))
        }
        if not rows:
            out = {k: np.zeros(0) for k in out}
        return out, ra_c, dec_c, t_tr

    def process(self, data):
        """Inject the pulses of the day.

        Parameters
        ----------
        data : SiderealStream
            Stacked visibilities of one day (masked), with an ``lsd`` attribute.

        Returns
        -------
        sparse : SiderealStream
            The pulses alone, on the samples they touch, with the input's weights.
        log : containers.LPTInjectionLog
            The pulses of the day.
        """
        if "ra" not in data.index_map:
            raise ValueError("InjectLPT needs a stream with an RA axis.")
        lsd = int(np.atleast_1d(data.attrs["lsd"])[0])
        self.epoch = self.telescope.lsd_to_unix(lsd)
        data.redistribute("freq")
        ra_deg = data.index_map["ra"][:]
        t_all = self.telescope.lsd_to_unix(lsd + ra_deg / 360.0)
        dt = np.median(np.diff(t_all))

        self._process_catalog()
        sched, ra_c, dec_c, t_tr = self._schedule(lsd)
        tab = self.catalog["lpt"][:]
        npulse = sched["source"].size
        src = sched["source"].astype(int)

        # Samples touched by each pulse, on the full RA axis
        ha_all = tools.correct_phase_wrap(np.radians(ra_deg)[np.newaxis] - self.sra[src][:, np.newaxis])
        sin_alt = np.sin(self.sdec[src])[:, None] * np.sin(self.latitude) + np.cos(self.sdec[src])[:, None] * np.cos(
            self.latitude
        ) * np.cos(ha_all)
        prof = np.zeros((npulse, ra_deg.size))
        for p in range(npulse):
            prof[p] = _pulse_profile(t_all, dt, sched["unix_center"][p], tab["duration"][src[p]], self.profile)
        use = (prof > self.min_profile) & (sin_alt > 0.0) & ~sched["nulled"].astype(bool)[:, None]
        injected = use.any(axis=1)
        isel = np.flatnonzero(use.any(axis=0))
        if isel.size == 0:
            isel = np.array([0])

        # The sparse copy: the input's weights on those samples, no visibilities
        sparse = data.__class__(
            axes_from=data, attrs_from=data, ra=ra_deg[isel], distributed=data.distributed, comm=data.comm
        )
        sparse.redistribute("freq")
        sparse.vis[:] = 0.0
        sparse.weight[:].local_array[:] = data.weight[:].local_array[..., isel]
        sparse.attrs["lpt_catalog"] = self.catalog_tag

        vis = sparse.vis[:]
        lo, nf = vis.local_offset[0], vis.local_shape[0]
        freq_all = sparse.index_map["freq"]["centre"][:]
        self.freq_local = sparse.index_map["freq"][lo : lo + nf]
        nu = self.freq_local["centre"]
        lam = scipy.constants.c / (nu * 1e6)
        self._initialize_beam_with_data()

        dist, pol = _get_baseline_info(sparse, self.telescope)
        pol_index = {p: np.flatnonzero(pol == p) for p in self.process_pol}
        pol_index = {p: v for p, v in pol_index.items() if v.size}
        ib = np.sort(np.concatenate(list(pol_index.values())))
        pos = {p: np.searchsorted(ib, v) for p, v in pol_index.items()}
        uv = dist[:, ib][:, np.newaxis, :, np.newaxis] / lam[np.newaxis, :, np.newaxis, np.newaxis]
        x_nu = nu / self.reference_freq

        bsum = np.zeros(npulse)
        wsum = np.zeros(npulse)
        for p in range(npulse):
            if not injected[p]:
                continue
            s = src[p]
            sdec = self.sdec[s]
            cols = np.flatnonzero(use[p, isel])
            w = prof[p, isel][cols]
            sha = ha_all[p, isel][cols]
            spec = (
                sched["flux"][p]
                * x_nu ** tab["spectral_index"][s]
                * np.exp(tab["spectral_curvature"][s] * np.log(x_nu) ** 2)
            )
            spec = spec * _scintillation(
                freq_all, tab["scint_index"][s], tab["scint_bw"][s], int(sched["scint_seed"][p])
            )[lo : lo + nf]
            stokes_i = spec[:, np.newaxis] * w[np.newaxis]  # freq, time
            psi = np.radians(tab["pol_angle"][s]) + tab["rm"][s] * lam[:, np.newaxis] ** 2
            if self.parallactic:
                psi = psi + _parallactic_angle(sha, self.latitude, sdec)[np.newaxis]
            lin = tab["frac_lin"][s] * stokes_i
            q, u = lin * np.cos(2 * psi), lin * np.sin(2 * psi)
            v = tab["frac_circ"][s] * stokes_i
            bxx = self._beamfunc(nu, "XX", sdec, sha)
            byy = self._beamfunc(nu, "YY", sdec, sha)
            bxy = np.sqrt(np.abs(bxx * byy))
            bsum[p] = np.sum(bxx * w[np.newaxis])
            wsum[p] = nf * np.sum(w)
            amp = {
                "XX": (stokes_i + q) * bxx,
                "YY": (stokes_i - q) * byy,
                "XY": (u + 1j * v) * bxy,
                "YX": (u - 1j * v) * bxy,
            }
            phase = interferometry.fringestop_phase(
                sha[np.newaxis, np.newaxis, :], self.latitude, sdec, *uv
            ).conj()  # freq, baseline, time
            for pp, ii in pos.items():
                phase[:, ii] *= amp[pp][:, np.newaxis, :]
            vis.local_array[np.ix_(np.arange(nf), ib, cols)] += phase.astype(vis.local_array.dtype)

        from mpi4py import MPI

        sparse.comm.Allreduce(MPI.IN_PLACE, bsum, op=MPI.SUM)
        sparse.comm.Allreduce(MPI.IN_PLACE, wsum, op=MPI.SUM)

        # The log: where and when each pulse appears
        log = containers.LPTInjectionLog(pulse=npulse, attrs_from=self.catalog)
        rec = log.pulse[:].copy()
        if npulse:
            dec_r = self.sdec[src]
            _, y, _ = spherical.sph_to_ground(0.0, self.latitude, dec_r)
            t_start = sched["unix_center"] - 0.5 * tab["duration"][src]
            ha_start = _OMEGA * (t_start - t_tr[src])
            x_start, _, _ = spherical.sph_to_ground(ha_start, self.latitude, dec_r)
            rec["source"] = src
            rec["pulse"] = sched["pulse"]
            rec["injected"] = injected
            rec["nulled"] = sched["nulled"].astype(bool)
            rec["unix_center"] = sched["unix_center"]
            rec["t_offset"] = sched["t_offset"]
            rec["flux"] = sched["flux"]
            rec["duration"] = tab["duration"][src]
            rec["spectral_index"] = tab["spectral_index"][src]
            rec["frac_lin"] = tab["frac_lin"][src]
            rec["frac_circ"] = tab["frac_circ"][src]
            rec["ra"] = self.catalog["position"]["ra"][:][src]
            rec["dec"] = self.catalog["position"]["dec"][:][src]
            rec["ra_cirs"] = ra_c[src]
            rec["dec_cirs"] = dec_c[src]
            rec["y"] = y
            rec["x_start"] = x_start
            rec["ra_start"] = (ra_c[src] + np.degrees(ha_start)) % 360.0
            rec["beam_mean"] = bsum * tools.invert_no_zero(wsum)
            rec["scint_seed"] = sched["scint_seed"]
        log.pulse[:] = rec
        log.attrs["lsd"] = lsd
        log.attrs["tag"] = f"lsd_{lsd}"
        log.attrs["window_sigma"] = self.window_sigma
        log.attrs["barycentric"] = self.barycentric
        self.log.info(
            f"LSD {lsd}: {npulse} pulses in the main lobe, {int(injected.sum())} injected "
            f"({int(sched['nulled'].astype(bool).sum()) if npulse else 0} nulled), on {isel.size} samples."
        )
        return sparse, log


class AddInjectedHybrid(tasklib.base.ContainerTask):
    """Add the beamformed injections to the hybrid stream, in place.

    ``setup`` takes the hybrid stream of the injections (the sparse stream of
    :class:`InjectLPT`, gridded and beamformed as the data) and any number of
    products of the real search that must exist before the data are changed: list
    the last products of every branch of the search that reads the hybrid stream,
    so that nothing of the real search sees the injections.  ``process`` adds the
    injections to the matching samples (by RA) of the hybrid stream, in place.
    """

    def setup(self, injection, *done):
        """Keep the injections.

        Parameters
        ----------
        injection : HybridVisStream
            The injections alone, on their samples.
        *done
            Products of the real search, only to order the tasks.
        """
        self.injection = injection
        self.log.info(f"The real search is done ({len(done)} products); injecting.")

    def process(self, data):
        """Add the injections.

        Parameters
        ----------
        data : HybridVisStream

        Returns
        -------
        data : HybridVisStream
            The same container, with the injections added.
        """
        inj = self.injection
        data.redistribute("freq")
        inj.redistribute("freq")
        for ax in ("pol", "ew", "el"):
            a, b = data.index_map[ax][:], inj.index_map[ax][:]
            if a.shape != b.shape or not np.array_equal(a, b):
                raise ValueError(f"The {ax} axes of the data and the injections differ.")
        if not np.allclose(data.freq, inj.freq):
            raise ValueError("The frequency axes of the data and the injections differ.")
        ra_d, ra_i = data.index_map["ra"][:], inj.index_map["ra"][:]
        dra = np.median(np.diff(ra_d))
        idx = np.clip(np.searchsorted(ra_d, ra_i - 0.5 * dra), 0, ra_d.size - 1)
        ok = np.abs(ra_d[idx] - ra_i) < 0.01 * abs(dra)
        if not ok.all():
            self.log.warning(f"{np.sum(~ok)} injected samples are not in the data.")
        data.vis[:].local_array[..., idx[ok]] += inj.vis[:].local_array[..., ok]
        data.attrs["lpt_injected"] = True
        self.log.info(f"Added the injections on {int(ok.sum())} samples.")
        return data


class DiscardInput(tasklib.base.ContainerTask):
    """Take an input and output nothing (to switch off a branch of a config)."""

    def process(self, data):
        return None


class MatchLPTInjections(tasklib.base.ContainerTask):
    """Match the candidates of the search to the injected pulses of the day.

    An injected pulse is found if a candidate lies within ``match_dy`` of its
    telescope-y at transit and starts within the pulse, widened by ``match_dt``
    seconds on each side (start times in RA of the meridian).  Of several, the one
    with the highest S/N is taken.  Pulses that were not injected (nulled, below the
    horizon) are kept, with ``found`` False.

    Attributes
    ----------
    match_dy : float
        Half-width in telescope-y.  Default is 0.01.
    match_dt : float
        Extra time in seconds on each side of the pulse.  Default is 30.
    """

    match_dy = config.Property(proptype=float, default=0.01)
    match_dt = config.Property(proptype=float, default=30.0)

    def process(self, table, log):
        """Match.

        Parameters
        ----------
        table : containers.TransientCandidateTable
            Sifted candidates of the search with the injections.
        log : containers.LPTInjectionLog

        Returns
        -------
        out : containers.LPTInjectionResults
        """
        rec = table.candidate[:]
        sift = table.datasets["sift"][:] if "sift" in table.datasets else None
        pl = log.pulse[:]
        n = pl.size
        out = containers.LPTInjectionResults(pulse=n, attrs_from=log)
        res = out.injection[:].copy()
        for name in pl.dtype.names:
            res[name] = pl[name]
        res["candidate"] = -1
        for name in ("snr", "amplitude", "cand_duration", "cand_spectral_index",
                     "cand_y", "cand_x", "cand_ra", "ns_scale", "ns_dchi2"):  # fmt: skip
            res[name] = np.nan

        deg_s = np.degrees(_OMEGA)
        for k in range(n):
            if not pl["injected"][k]:
                continue
            d_ra = (rec["ra"] - pl["ra_start"][k] + 180.0) % 360.0 - 180.0
            win = (d_ra >= -self.match_dt * deg_s) & (
                d_ra <= (pl["duration"][k] + self.match_dt) * deg_s
            )
            near = win & (np.abs(rec["y"] - pl["y"][k]) <= self.match_dy)
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
        ninj = int(pl["injected"].sum())
        self.log.info(
            f"Found {res['found'].sum()} of {ninj} injected pulses; "
            f"{(res['found'] & res['keep']).sum()} kept by the sifter."
        )
        return out
