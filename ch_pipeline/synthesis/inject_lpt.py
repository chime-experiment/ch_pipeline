"""Inject simulated long-period transients (LPTs) into the transient search.

The catalogue holds the intrinsic properties of the sources only
(:class:`ch_pipeline.core.containers.LPTCatalog`).  Two ways to make one:

- :class:`GenerateLPTCatalog` draws a fixed catalogue once (to be saved and used
  for every day): each source pulses with a period, at a fixed position, so the
  same sources reappear on the days their pulses fall in the main lobe.  This tests
  the combination of days and folds in CHIME's observing window.
- :class:`GenerateDailyLPTCatalog` draws a new catalogue every day (seeded by the
  LSD), each source with one pulse near its transit.  This samples sky position and
  day for the selection function of the single-day search.  With ``periodic`` the
  sources keep their periods and pulse every period around that pulse: trains of
  pulses through the main lobe, as an LPT gives, for tuning the sifter.

The properties are drawn with :mod:`ch_pipeline.synthesis.lpt_population`.

Each day, :class:`InjectLPT` finds the pulses that fall where the beam model covers
the source (periodic in barycentric time if ``barycentric``),
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
centre of the beam XX = I + Q, YY = I - Q, XY = (U + iV) e^{i phi} and
YX = (U - iV) e^{-i phi}, with phi = xy_offset + 2 pi nu xy_delay the phase between
the X and Y arrays (zero by default; the Stokes search of
:class:`ch_pipeline.analysis.transients.TransientMatchedFilterSkyNoise` removes the
same phase).  The polarisation
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
from .inject import SIGMA, BaseInject, BeamExternal, _get_baseline_info
from .lpt_population import SUBPULSE_COLUMNS, draw_population, merge_population

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


# Scatter of ln(width) and ln(amplitude) of the sub-pulses of a pulse
_SUB_WIDTH_SIGMA = 0.3
_SUB_AMP_SIGMA = 0.5


def _subpulses(rng, duration, rate, width, period, jitter):
    """Times (s from the centre), FWHMs (s) and amplitudes of the sub-pulses of one pulse.

    The envelope is a Gaussian of FWHM ``duration``.  With ``period`` 0 there are
    Poisson(``rate`` x ``duration`` / 60) sub-pulses (``rate`` per minute of FWHM; at
    least 1, at most as many as fill half the envelope's FWHM), placed at random under the envelope (within 2 sigma), each at
    least 1.5 FWHM from the others; else one every ``period`` s across the envelope
    from a random phase, each moved by ``jitter`` periods (normal).  Widths scatter
    by a factor exp(0.3 N), amplitudes are the envelope at the sub-pulse times
    exp(0.5 N).
    """
    sig = duration / (2.0 * np.sqrt(2.0 * np.log(2.0)))
    if period > 0:
        t = -2.0 * sig + rng.uniform(0.0, period) + period * np.arange(int(4.0 * sig / period) + 1)
        t = t[t <= 2.0 * sig]
        if t.size == 0:
            t = np.array([rng.uniform(-sig, sig)])
        t = t + jitter * period * rng.standard_normal(t.size)
        w = width * np.exp(_SUB_WIDTH_SIGMA * rng.standard_normal(t.size))
    else:
        nmax = max(1, int(0.5 * duration / width))
        n = int(np.clip(rng.poisson(rate * duration / 60.0), 1, nmax))
        t, w = [], []
        for _ in range(50 * n):
            if len(t) == n:
                break
            tk = rng.normal(0.0, sig)
            wk = width * np.exp(_SUB_WIDTH_SIGMA * rng.standard_normal())
            if abs(tk) > 2.0 * sig:
                continue
            if all(abs(tk - tj) >= 1.5 * max(wk, wj) for tj, wj in zip(t, w)):
                t.append(tk)
                w.append(wk)
        t, w = np.array(t), np.array(w)
    a = np.exp(-0.5 * (t / sig) ** 2) * np.exp(_SUB_AMP_SIGMA * rng.standard_normal(t.size))
    return t, w, a


def _subpulse_profile(t, dt, t0, duration, floor, sub):
    """Mean over samples (centred at ``t``, length ``dt``) of a pulse with sub-pulses.

    ``floor`` x the Gaussian envelope (FWHM ``duration``, centre ``t0``) plus
    ``1 - floor`` x the sum of the Gaussian sub-pulses ``sub`` (times from the
    centre, FWHMs, amplitudes; see :func:`_subpulses`) scaled to a peak of 1, the
    whole scaled so that its peak (instantaneous, not sample-averaged) is 1.
    """
    ts, ws, a = sub
    rel = np.asarray(t, dtype=np.float64) - t0  # relative times keep the precision

    def mean_sum(x, step):
        out = floor * _pulse_profile(x, step, 0.0, duration, "gaussian")
        if a.size:
            s = sum(ak * _pulse_profile(x, step, tk, wk, "gaussian") for tk, wk, ak in zip(ts, ws, a))
            out = out + (1.0 - floor) * s / sub_peak
        return out

    # the peak of the sum of the sub-pulses alone, then of the whole profile
    fine = np.arange(-1.5 * duration, 1.5 * duration, max(min(ws.min() if ws.size else duration, duration) / 20.0, 0.05))
    sub_peak = 1.0
    if a.size:
        g = sum(ak * np.exp(-4.0 * np.log(2.0) * ((fine - tk) / wk) ** 2) for tk, wk, ak in zip(ts, ws, a))
        sub_peak = float(g.max()) or 1.0
        inst = floor * np.exp(-4.0 * np.log(2.0) * (fine / duration) ** 2) + (1.0 - floor) * g / sub_peak
    else:
        inst = floor * np.exp(-4.0 * np.log(2.0) * (fine / duration) ** 2)
    peak = float(inst.max()) or 1.0
    return mean_sum(rel, dt) / peak


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




def _subpulse_column(tab, name):
    """A sub-pulse column of the catalogue table, zeros for a catalogue without them."""
    if name in tab.dtype.names:
        return np.nan_to_num(np.asarray(tab[name], dtype=np.float64))
    return np.zeros(tab.size)


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
                 "scint_index", "scint_bw", "distance", *SUBPULSE_COLUMNS):  # fmt: skip
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
        Number of sources.  Default is 80 (with the default periods, ~0.35 pulses
        per source per day fall where the XX beam at 600 MHz is above half its peak,
        and ~300 pulses a day where the beam model is defined).
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

    With ``periodic``, the sources keep the periods of the population and their
    epochs are left NaN: :class:`InjectLPT` then puts a pulse at ``t_offset_beam``
    beam widths from the transit of the day and others every period before and
    after it (in Earth time), so each source gives a train of pulses through the
    main lobe.

    Attributes
    ----------
    periodic : bool
        Keep the periods.  Default is False.
    """

    n_source = config.Property(proptype=int, default=30)
    periodic = config.Property(proptype=bool, default=False)

    def process(self, data):
        """Draw the catalogue for the day of ``data`` (any container with ``lsd``).

        Returns
        -------
        catalog : containers.LPTCatalog
        """
        lsd = int(np.atleast_1d(data.attrs["lsd"])[0])
        spec = merge_population(self.population)
        if not self.periodic:
            spec["period"] = 0.0
        rng = np.random.default_rng([self.seed, lsd])
        pop = draw_population(rng, self.n_source, spec)
        attrs = self._attrs(f"lpt_daily_lsd_{lsd}", spec)
        attrs["lsd"] = lsd
        attrs["periodic"] = self.periodic
        cat = _make_catalog(pop, rng, self.epoch_ref, attrs)
        if self.periodic:
            # anchored at the transit of the day by InjectLPT
            cat["lpt"]["epoch"][:] = np.nan
            self.log.info(
                f"Drew {self.n_source} LPTs for LSD {lsd} pulsing around transit "
                f"(periods {np.min(pop['period']):.0f}-{np.max(pop['period']):.0f} s)."
            )
        else:
            self.log.info(f"Drew {self.n_source} single LPT pulses for LSD {lsd}.")
        return cat


class InjectLPT(BeamExternal):
    """Find the pulses of a catalogue in the day's data and inject them (sparse stream).

    Every pulse that falls on a sample of the input where the source is above the
    horizon and within the beam model (for a beam in telescope coordinates, within
    its range of telescope-x; in celestial coordinates, its range of hour angle) is
    injected; the primary beam itself sets how much signal it gives, which is small
    far from transit.  Outside the beam model nothing is injected (the model would
    have to be extrapolated).  A circumpolar source is also seen below the pole.

    A source with a period pulses at ``epoch + k period`` at the barycentre (at the
    Earth earlier by the Roemer delay at that time if ``barycentric``).  A source
    with period 0 has one pulse at ``t_offset_beam`` widths of a Gaussian XX beam at
    600 MHz from its upper transit on the day (in [0, 360) of meridian RA).  A
    source with a period and a NaN epoch (:class:`GenerateDailyLPTCatalog` with
    ``periodic``) has that pulse and others every period before and after it, at
    the Earth.  Each
    pulse gets its own random draws (seeded by the source and the pulse number, so
    the same on any day whose data contain it): nulled with probability
    ``null_fraction``, a flux scattered by exp(``flux_scatter`` N(0, 1)), and the
    seed of its scintillation.  The input of a day extends beyond 0-360 degrees of
    RA, so a pulse near the ends can be in the data of two days; it is the same
    pulse in both (``in_day`` marks the day whose candidate maps contain it).

    The pulses are added to a copy of the input restricted to the samples where
    any has a mean profile above ``min_profile`` (and is covered by the beam), with
    the input's weights there and zero visibilities otherwise:

        V_pp(nu, t) = A_pp(nu, t) B_pp(nu, HA) exp(2 pi i b . s / lambda)

    with the phase of :mod:`inject` (the conjugate of the fringe-stopping
    phase), the beam of the external beam model (the geometric mean of XX and YY for
    the cross-polar products) and ``A`` the Stokes combination of the module
    docstring times the spectrum, the scintillation and the profile.  The input is
    not modified.

    Attributes
    ----------
    polarization : {"copol", "full", "single"}
        Products to inject into (see :class:`BaseInject`).  Default is "full".
    barycentric : bool
        Periodic in barycentric time.  Default is True.
    min_profile : float
        Smallest mean profile of a sample to inject.  Default is 1e-4.
    parallactic : bool
        Rotate the polarisation by the parallactic angle.  Default is True.
    xy_delay : float
        Delay of the X-Y phase in ns (see the module docstring).  Default is 0.
    xy_offset : float
        Offset of the X-Y phase in degrees.  Default is 0.
    """

    polarization = config.enum(["full", "copol", "single"], default="full")
    barycentric = config.Property(proptype=bool, default=True)
    min_profile = config.Property(proptype=float, default=1e-4)
    parallactic = config.Property(proptype=bool, default=True)
    xy_delay = config.Property(proptype=float, default=0.0)
    xy_offset = config.Property(proptype=float, default=0.0)

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
        phi = np.asarray(beam.phi, dtype=float)
        if beam.coords == "telescope":
            self._cover = ("x", phi.min(), phi.max())
        else:
            ha = np.radians((phi + 180.0) % 360.0 - 180.0)
            self._cover = ("ha", ha.min(), ha.max())

    def _covered(self, ha, dec):
        """True where a source at ``dec`` and hour angle ``ha`` (radians) is above the
        horizon and within the beam model."""
        sin_alt = np.sin(dec) * np.sin(self.latitude) + np.cos(dec) * np.cos(self.latitude) * np.cos(ha)
        kind, lo, hi = self._cover
        if kind == "x":
            q = -np.cos(dec) * np.sin(ha)
        else:
            podal = np.abs(ha) <= 0.5 * np.pi
            q = np.where(podal, ha, ((ha + 2.0 * np.pi) % (2.0 * np.pi)) - np.pi)
        return (sin_alt > 0.0) & (q >= lo) & (q <= hi)

    def _schedule(self, lsd, t_all, covered):
        """The pulses in the data: a dict of arrays, one entry per pulse.

        ``covered`` [source, sample] marks the samples within the beam model.
        """
        tab = self.catalog["lpt"][:]
        ra_i, dec_i = self.catalog["position"]["ra"][:], self.catalog["position"]["dec"][:]
        ra_c, dec_c = np.degrees(self.sra) % 360.0, np.degrees(self.sdec)
        t_tr = self.telescope.lsd_to_unix(lsd + ra_c / 360.0)
        sig_t = _beam_sigma_time(dec_c)
        dt = np.median(np.diff(t_all))
        rows = []
        nsub = _subpulse_column(tab, "subpulse_rate")
        for s in range(tab.size):
            cov = covered[s]
            if not cov.any():
                continue
            dur, per = tab["duration"][s], tab["period"][s]
            # half the span of a pulse: a top hat's half width, or 2.35 sigma of an
            # envelope of sub-pulses (they are placed within 2 sigma)
            half = dur if nsub[s] > 0 else 0.5 * dur
            if per > 0 and not np.isfinite(tab["epoch"][s]):
                # a train through the transit of the day, at the Earth
                ep = t_tr[s] + tab["t_offset_beam"][s] * sig_t[s]
                t0 = t_all[cov].min() - half - dt
                t1 = t_all[cov].max() + half + dt
                k = np.arange(np.ceil((t0 - ep) / per), np.floor((t1 - ep) / per) + 1)
                pulses = list(zip(k.astype(np.int64), ep + k * per))
            elif per > 0:
                ep = tab["epoch"][s]
                t0 = t_all[cov].min() - half - dt
                t1 = t_all[cov].max() + half + dt
                pad = 600.0 if self.barycentric else 0.0  # > the largest Roemer delay
                k = np.arange(np.ceil((t0 - pad - ep) / per), np.floor((t1 + pad - ep) / per) + 1)
                tc = ep + k * per
                if self.barycentric and tc.size:
                    # arrival at the Earth: the delay at the (approximate) arrival time
                    tc = tc - _roemer_delay(tc - _roemer_delay(tc, ra_i[s], dec_i[s]), ra_i[s], dec_i[s])
                pulses = list(zip(k.astype(np.int64), tc))
            else:
                pulses = [(int(lsd), t_tr[s] + tab["t_offset_beam"][s] * sig_t[s])]
            for kk, tc in pulses:
                on = np.abs(t_all - tc) <= half + 0.5 * dt
                if not (on & cov).any():
                    continue
                rng = np.random.default_rng([int(tab["seed"][s]), int(kk) + 2**40])
                nulled = rng.uniform() < tab["null_fraction"][s]
                flux = tab["flux"][s] * np.exp(tab["flux_scatter"][s] * rng.standard_normal())
                scint_seed = int(rng.integers(0, 2**31 - 1))
                # drawn after the others, so a smooth pulse's draws are as before
                sub_seed = int(rng.integers(0, 2**31 - 1)) if nsub[s] > 0 else 0
                rows.append((s, kk, nulled, tc, flux, scint_seed, sub_seed))
        names = ("source", "pulse", "nulled", "unix_center", "flux", "scint_seed", "sub_seed")
        out = {name: np.array([r[i] for r in rows]) for i, name in enumerate(names)}
        if not rows:
            out = {k: np.zeros(0) for k in names}
        return out, ra_c, dec_c

    def process(self, data):
        """Inject the pulses in the day's data.

        Parameters
        ----------
        data : SiderealStream or TimeStream
            Stacked visibilities of one day (masked), with an ``lsd`` attribute.  A
            TimeStream (the daily ``tstream`` files) has its RA computed from the
            times as in MakeVisGrid.

        Returns
        -------
        sparse : SiderealStream
            The pulses alone, on the samples they touch, with the input's weights.
        log : containers.LPTInjectionLog
            The pulses of the day.
        """
        lsd = int(np.atleast_1d(data.attrs["lsd"])[0])
        self.epoch = self.telescope.lsd_to_unix(lsd)
        data.redistribute("freq")
        if "ra" in data.index_map:
            ra_deg = data.index_map["ra"][:]
            t_all = self.telescope.lsd_to_unix(lsd + ra_deg / 360.0)
            axis = {"ra": None}
        elif "time" in data.index_map:
            t_all = np.asarray(data.time, dtype=np.float64)
            ra_deg = 360.0 * (self.telescope.unix_to_lsd(t_all) - lsd)
            axis = {"time": data.index_map["time"][:]}
        else:
            raise ValueError("InjectLPT needs a stream with an RA or a time axis.")
        dt = np.median(np.diff(t_all))

        self._process_catalog()
        ha_src = tools.correct_phase_wrap(np.radians(ra_deg)[np.newaxis] - self.sra[:, np.newaxis])
        covered = self._covered(ha_src, self.sdec[:, np.newaxis])
        sched, ra_c, dec_c = self._schedule(lsd, t_all, covered)
        tab = self.catalog["lpt"][:]
        npulse = sched["source"].size
        src = sched["source"].astype(int)

        # Samples touched by each pulse, on the full RA axis
        prof = np.zeros((npulse, ra_deg.size))
        nsub = _subpulse_column(tab, "subpulse_rate")
        n_drawn = np.zeros(npulse, dtype=np.int32)
        support = np.zeros(npulse)
        for p in range(npulse):
            s = src[p]
            dur = tab["duration"][s]
            if nsub[s] > 0:
                sub = _subpulses(
                    np.random.default_rng(int(sched["sub_seed"][p])), dur, nsub[s],
                    _subpulse_column(tab, "subpulse_width")[s], _subpulse_column(tab, "subpulse_period")[s],
                    _subpulse_column(tab, "subpulse_jitter")[s],
                )  # fmt: skip
                prof[p] = _subpulse_profile(
                    t_all, dt, sched["unix_center"][p], dur, _subpulse_column(tab, "subpulse_floor")[s], sub
                )
                n_drawn[p] = sub[0].size
                support[p] = 2.0 * dur
            else:
                prof[p] = _pulse_profile(t_all, dt, sched["unix_center"][p], dur, self.profile)
                support[p] = dur
        use = (prof > self.min_profile) & covered[src] & ~sched["nulled"].astype(bool)[:, None]
        injected = use.any(axis=1)
        isel = np.flatnonzero(use.any(axis=0))
        if isel.size == 0:
            isel = np.array([0])

        # The sparse copy: the input's weights on those samples, no visibilities
        name, full = next(iter(axis.items()))
        sub = ra_deg[isel] if full is None else full[isel]
        sparse = data.__class__(
            axes_from=data, attrs_from=data, distributed=data.distributed, comm=data.comm, **{name: sub}
        )
        # A new axis loses its attributes; for the time axis ``alignment`` says
        # whether the times are the start, centre or end of each sample, and
        # ``.time`` (and so the RA of MakeVisGrid) depends on it
        for k, v in data.index_attrs[name].items():
            if not str(k).startswith("__memh5"):
                sparse.index_attrs[name][k] = v
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
        xy_phase = np.exp(
            1j * (np.radians(self.xy_offset) + 2 * np.pi * nu * 1e6 * self.xy_delay * 1e-9)
        )[:, np.newaxis]

        bsum = np.zeros(npulse)
        wsum = np.zeros(npulse)
        for p in range(npulse):
            if not injected[p]:
                continue
            s = src[p]
            sdec = self.sdec[s]
            cols = np.flatnonzero(use[p, isel])
            w = prof[p, isel][cols]
            sha = ha_src[s, isel][cols]
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
                "XY": (u + 1j * v) * bxy * xy_phase,
                "YX": (u - 1j * v) * bxy * xy_phase.conj(),
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
            tc = sched["unix_center"]
            lsd_c = self.telescope.unix_to_lsd(tc)
            # the nearest upper transit, and the hour angle at the centre and start
            n_tr = np.rint(lsd_c - lsd - ra_c[src] / 360.0)
            t_up = self.telescope.lsd_to_unix(lsd + n_tr + ra_c[src] / 360.0)
            ha_c = tools.correct_phase_wrap(_OMEGA * (tc - t_up))
            t_start = tc - 0.5 * support
            ha_start = tools.correct_phase_wrap(_OMEGA * (t_start - t_up))
            _, y, _ = spherical.sph_to_ground(ha_c, self.latitude, dec_r)
            x_start, _, _ = spherical.sph_to_ground(ha_start, self.latitude, dec_r)
            rec["source"] = src
            rec["pulse"] = sched["pulse"]
            rec["injected"] = injected
            rec["nulled"] = sched["nulled"].astype(bool)
            rec["in_day"] = (lsd_c >= lsd) & (lsd_c < lsd + 1)
            rec["unix_center"] = tc
            rec["t_offset"] = tc - t_up
            rec["ha_center"] = np.degrees(ha_c)
            rec["flux"] = sched["flux"]
            rec["duration"] = tab["duration"][src]
            rec["spectral_index"] = tab["spectral_index"][src]
            rec["frac_lin"] = tab["frac_lin"][src]
            rec["frac_circ"] = tab["frac_circ"][src]
            rec["pol_angle"] = tab["pol_angle"][src]
            rec["rm"] = tab["rm"][src]
            rec["ra"] = self.catalog["position"]["ra"][:][src]
            rec["dec"] = self.catalog["position"]["dec"][:][src]
            rec["ra_cirs"] = ra_c[src]
            rec["dec_cirs"] = dec_c[src]
            rec["y"] = y
            rec["x_start"] = x_start
            rec["ra_start"] = 360.0 * (self.telescope.unix_to_lsd(t_start) - lsd) % 360.0
            rec["beam_mean"] = bsum * tools.invert_no_zero(wsum)
            rec["scint_seed"] = sched["scint_seed"]
            rec["n_subpulse"] = n_drawn
            rec["support"] = support
        log.pulse[:] = rec
        log.attrs["lsd"] = lsd
        log.attrs["tag"] = f"lsd_{lsd}"
        log.attrs["beam_cover"] = json.dumps([self._cover[0], float(self._cover[1]), float(self._cover[2])])
        log.attrs["barycentric"] = self.barycentric
        self.log.info(
            f"LSD {lsd}: {npulse} pulses within the beam model, {int(injected.sum())} injected "
            f"({int(sched['nulled'].astype(bool).sum()) if npulse else 0} nulled), "
            f"{int(rec['in_day'].sum()) if npulse else 0} in the day, on {isel.size} of {ra_deg.size} samples."
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
        if ra_i.size and not ok.any():
            raise ValueError(
                f"None of the {ra_i.size} injected samples is on the RA grid of the data "
                f"(offsets {np.min(np.abs(ra_d[idx] - ra_i)):.4g} deg or more)."
            )
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
                     "cand_y", "cand_x", "cand_ra", "ns_scale", "ns_dchi2",
                     "cand_imag"):  # fmt: skip
            res[name] = np.nan

        deg_s = np.degrees(_OMEGA)
        for k in range(n):
            if not pl["injected"][k]:
                continue
            d_ra = (rec["ra"] - pl["ra_start"][k] + 180.0) % 360.0 - 180.0
            span = pl["duration"][k]
            if "support" in pl.dtype.names and pl["support"][k] > 0:
                span = pl["support"][k]
            win = (d_ra >= -self.match_dt * deg_s) & (d_ra <= (span + self.match_dt) * deg_s)
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
            if "amplitude_imag" in rec.dtype.names:
                res["cand_imag"][k] = rec["amplitude_imag"][j]
            if sift is not None:
                res["keep"][k] = sift["keep"][j]
                res["flags"][k] = sift["flags"][j]
        out.injection[:] = res
        if "sift_flags" in table.attrs:
            out.attrs["sift_flags"] = table.attrs["sift_flags"]
        out.attrs["match_dy"] = self.match_dy
        out.attrs["match_dt"] = self.match_dt
        pol = str(table.attrs.get("pol", "co"))
        out.attrs["pol"] = pol
        ninj = int(pl["injected"].sum())
        self.log.info(
            f"{pol}: found {res['found'].sum()} of {ninj} injected pulses; "
            f"{(res['found'] & res['keep']).sum()} kept by the sifter."
        )
        return out
