"""Random draws of simulated long-period transient (LPT) populations.

A light version of the distributions of ``searchopt.population``: each property is
drawn from a distribution given by a spec, either a number (fixed) or a dict
``{"distribution": name, **parameters}``, and only random draws are supported.

Distributions: ``Fixed(value)``, ``Uniform(lower, upper)``,
``LogUniform(lower, upper)``, ``PowerLaw(index, lower, upper)`` (pdf proportional
to x^index on [lower, upper]; the CDF of LPT periods of Suvorov et al. 2026,
F(P) ~ P^alpha, is ``index = alpha - 1``), ``Normal(mean, sigma)``,
``TruncatedNormal(mean, sigma, lower, upper)`` (absolute bounds),
``LogNormal(median, sigma)`` (sigma of ln x) and ``Choice(values, weights)``.

Sky models: ``Isotropic`` (uniform on the sphere), ``GalacticDisk`` (an
exponential disk, as used for Galactic compact binaries by Korol et al. 2022 and
for LPTs by Suvorov et al. 2026: surface density ~ exp(-R / scale_length), density
~ exp(-|z| / scale_height), the Sun at r_sun from the centre), both limited to a
range of declination, and ``Mixture`` (each source from one of several models).

Dispersion measures: ``DispersionMeasure``, from the DMs of the known pulsars near
each line of sight or from an electron-density model (NE2025, NE2001, YMW16) and a distance.

Polarisation: a point on the Poincare sphere, isotropic in direction (optionally
squashed towards the equator, i.e. towards linear polarisation, by
``circular_scale`` < 1) times a degree of polarisation from a distribution; this
gives the linear and circular fractions and the angle together.
"""

import copy

import numpy as np
import scipy.special


class Distribution:
    """Base class: ``draw(rng, number)`` returns an array of random values."""

    @staticmethod
    def from_spec(spec):
        """A distribution from a spec: a number (fixed) or a dict with ``distribution``."""
        if isinstance(spec, Distribution):
            return spec
        if np.isscalar(spec):
            return Fixed(float(spec))
        if not isinstance(spec, dict) or "distribution" not in spec:
            raise ValueError(f"Invalid distribution spec {spec!r}.")
        kw = dict(spec)
        name = kw.pop("distribution")
        classes = {c.__name__.lower(): c for c in _subclasses(Distribution)}
        key = str(name).replace("_", "").lower()
        if key not in classes:
            raise ValueError(f"Unknown distribution {name!r}; have {sorted(classes)}.")
        return classes[key](**kw)

    def draw(self, rng, number):
        raise NotImplementedError


class Fixed(Distribution):
    def __init__(self, value):
        self.value = value

    def draw(self, rng, number):
        return np.full(number, self.value, dtype=float)


class Uniform(Distribution):
    def __init__(self, lower, upper):
        self.lower, self.upper = float(lower), float(upper)

    def draw(self, rng, number):
        return rng.uniform(self.lower, self.upper, number)


class LogUniform(Distribution):
    def __init__(self, lower, upper):
        if lower <= 0 or upper <= 0:
            raise ValueError("LogUniform needs positive bounds.")
        self.lower, self.upper = float(lower), float(upper)

    def draw(self, rng, number):
        return np.exp(rng.uniform(np.log(self.lower), np.log(self.upper), number))


class PowerLaw(Distribution):
    """pdf ~ x^index on [lower, upper]."""

    def __init__(self, index, lower, upper):
        self.index, self.lower, self.upper = float(index), float(lower), float(upper)

    def draw(self, rng, number):
        u = rng.uniform(size=number)
        a = self.index + 1.0
        if abs(a) < 1e-12:
            return self.lower * (self.upper / self.lower) ** u
        return (self.lower**a + u * (self.upper**a - self.lower**a)) ** (1.0 / a)


class Normal(Distribution):
    def __init__(self, mean, sigma):
        self.mean, self.sigma = float(mean), float(sigma)

    def draw(self, rng, number):
        return self.mean + self.sigma * rng.standard_normal(number)


class TruncatedNormal(Distribution):
    def __init__(self, mean, sigma, lower, upper):
        self.mean, self.sigma = float(mean), float(sigma)
        self.lo = scipy.special.ndtr((lower - mean) / sigma)
        self.hi = scipy.special.ndtr((upper - mean) / sigma)

    def draw(self, rng, number):
        q = rng.uniform(self.lo, self.hi, number)
        return self.mean + self.sigma * scipy.special.ndtri(q)


class LogNormal(Distribution):
    def __init__(self, median, sigma):
        self.median, self.sigma = float(median), float(sigma)

    def draw(self, rng, number):
        return self.median * np.exp(self.sigma * rng.standard_normal(number))


class Choice(Distribution):
    def __init__(self, values, weights=None):
        self.values = np.asarray(values, dtype=float)
        w = np.ones(self.values.size) if weights is None else np.asarray(weights, dtype=float)
        self.weights = w / w.sum()

    def draw(self, rng, number):
        return rng.choice(self.values, size=number, p=self.weights)


def _subclasses(cls):
    out = set(cls.__subclasses__())
    for c in cls.__subclasses__():
        out |= _subclasses(c)
    return out


# ---- sky positions ------------------------------------------------------------


def _unit_vectors(rng, number):
    v = rng.standard_normal((number, 3))
    return v / np.linalg.norm(v, axis=1, keepdims=True)


class Isotropic:
    """Uniform on the sphere (ICRS), within ``dec_range`` (degrees)."""

    def __init__(self, dec_range=(-90.0, 90.0)):
        self.dec_range = tuple(dec_range)

    def draw(self, rng, number):
        """Returns ra, dec (degrees, ICRS) and distance (NaN)."""
        lo, hi = np.sin(np.radians(self.dec_range))
        dec = np.degrees(np.arcsin(rng.uniform(lo, hi, number)))
        ra = rng.uniform(0.0, 360.0, number)
        return ra, dec, np.full(number, np.nan)


class GalacticDisk:
    """An exponential Galactic disk, viewed from the Sun.

    Galactocentric radius from a Gamma(2, scale_length) distribution (surface
    density ~ exp(-R / scale_length)) up to ``r_max``, azimuth uniform, height from
    a Laplace distribution of scale ``scale_height`` (density ~ exp(-|z| / h)); the
    Sun at (-r_sun, 0, z_sun) kpc.  Positions outside ``dec_range`` are redrawn.
    """

    def __init__(self, scale_length=2.5, scale_height=0.3, r_sun=8.1, z_sun=0.0,
                 r_max=20.0, dec_range=(-90.0, 90.0)):  # fmt: skip
        self.scale_length, self.scale_height = scale_length, scale_height
        self.r_sun, self.z_sun, self.r_max = r_sun, z_sun, r_max
        self.dec_range = tuple(dec_range)

    def draw(self, rng, number):
        """Returns ra, dec (degrees, ICRS) and distance (kpc)."""
        import astropy.units as u
        from astropy.coordinates import SkyCoord

        ra, dec, dist = [], [], []
        need = number
        for _ in range(1000):
            m = max(4 * need, 64)
            r = rng.gamma(2.0, self.scale_length, m)
            r = r[r < self.r_max]
            k = r.size
            phi = rng.uniform(0.0, 2 * np.pi, k)
            z = rng.laplace(0.0, self.scale_height, k)
            x, y = r * np.cos(phi) + self.r_sun, r * np.sin(phi)
            zz = z - self.z_sun
            d = np.sqrt(x**2 + y**2 + zz**2)
            gl = np.degrees(np.arctan2(y, x)) % 360.0
            gb = np.degrees(np.arcsin(zz / d))
            c = SkyCoord(l=gl * u.deg, b=gb * u.deg, frame="galactic").icrs
            ok = (c.dec.deg >= self.dec_range[0]) & (c.dec.deg <= self.dec_range[1])
            ra.append(c.ra.deg[ok])
            dec.append(c.dec.deg[ok])
            dist.append(d[ok])
            need = number - sum(a.size for a in ra)
            if need <= 0:
                break
        else:
            raise RuntimeError("Could not draw enough positions in the declination range.")
        return (np.concatenate(ra)[:number], np.concatenate(dec)[:number],
                np.concatenate(dist)[:number])  # fmt: skip


class Mixture:
    """A mixture of sky models: each source is drawn from one of ``components``.

    ``components`` is a list of sky specs (each a dict with ``model``), ``weights``
    their probabilities (equal by default).  A ``dec_range`` given here is the
    default of the components that have none.
    """

    def __init__(self, components, weights=None, dec_range=None):
        if not components:
            raise ValueError("Mixture needs at least one component.")
        specs = [dict(c) for c in components]
        if dec_range is not None:
            for c in specs:
                c.setdefault("dec_range", dec_range)
        self.components = [sky_from_spec(c) for c in specs]
        w = np.ones(len(specs)) if weights is None else np.asarray(weights, dtype=float)
        if w.size != len(specs) or np.any(w < 0) or w.sum() <= 0:
            raise ValueError("Mixture weights must be non-negative, one per component.")
        self.weights = w / w.sum()

    def draw(self, rng, number):
        """Returns ra, dec (degrees, ICRS) and distance (kpc, NaN where a component has none)."""
        which = rng.choice(len(self.components), size=number, p=self.weights)
        ra, dec, dist = np.empty(number), np.empty(number), np.full(number, np.nan)
        for i, comp in enumerate(self.components):
            m = which == i
            if m.any():
                ra[m], dec[m], dist[m] = comp.draw(rng, int(m.sum()))
        return ra, dec, dist


SKY_MODELS = {"isotropic": Isotropic, "galacticdisk": GalacticDisk, "galactic_disk": GalacticDisk,
              "mixture": Mixture}  # fmt: skip


def sky_from_spec(spec):
    kw = dict(spec)
    name = str(kw.pop("model", "isotropic")).lower()
    if name not in SKY_MODELS:
        raise ValueError(f"Unknown sky model {name!r}; have {sorted(SKY_MODELS)}.")
    return SKY_MODELS[name](**kw)


# ---- dispersion measure ---------------------------------------------------------

#: Dispersion constant: the delay at frequency nu (MHz) is K_DM x DM / nu^2 seconds
K_DM = 4.148808e3


def dispersion_delay(dm, freq, reference_freq):
    """Delay (s) at ``freq`` (MHz) relative to ``reference_freq`` for a DM (pc cm^-3)."""
    return K_DM * dm * (np.asarray(freq, dtype=np.float64) ** -2 - float(reference_freq) ** -2)


class DispersionMeasure:
    """Dispersion measures (pc cm^-3) of sources from their positions.

    ``model``:

    - ``"pulsars"``: the DM of a known pulsar near the line of sight, one of the
      ``n_neighbour`` nearest on the sky in ``catalogue`` (an HDF5 file with a
      ``pulsar`` table of ``ra``, ``dec`` in degrees and ``dm``, as
      scripts/sift/make_pulsar_catalog.py writes from the ATNF catalogue), picked
      at random.  This samples the measured DMs along each direction, with the
      spread in distance of the pulsars there, and needs no distance.
    - ``"ne2025"`` or ``"ne2001"``: the electron-density model of Ocker & Cordes
      (2026) or Cordes & Lazio (2002) integrated to the source's distance, with the
      pure-Python ``mwprop`` package (about 0.15 s per source).  ``"ymw16"``: the
      model of Yao, Manchester & Wang (2017), with the ``pygedm`` package (compiled;
      needs libf2c).  Sources without a distance (NaN, e.g. from the isotropic sky
      model) get one drawn from ``distance`` (a distribution spec in kpc; default
      log-uniform 0.2-5).

    ``distance_max`` (kpc) limits the distance the model is integrated to (a sky
    model whose distances are not tied to the fluxes, such as a flattened disc, puts
    sources further than a detectable one would be).  ``scale`` multiplies the
    result; ``dm_max`` clips it.
    """

    def __init__(self, model="pulsars", catalogue=None, n_neighbour=10, distance=None,
                 distance_max=None, scale=1.0, dm_max=None):  # fmt: skip
        self.model = str(model).lower()
        if self.model not in ("pulsars", "ne2025", "ne2001", "ymw16"):
            raise ValueError(f"Unknown DM model {model!r}; have pulsars, ne2025, ne2001, ymw16.")
        if self.model == "pulsars" and catalogue is None:
            raise ValueError("The pulsars DM model needs a catalogue file.")
        self.catalogue = catalogue
        self.n_neighbour = int(n_neighbour)
        self.distance = Distribution.from_spec(
            distance if distance is not None else {"distribution": "loguniform", "lower": 0.2, "upper": 5.0}
        )
        self.distance_max = distance_max
        self.scale = float(scale)
        self.dm_max = dm_max
        self._psr = None

    def _pulsars(self):
        """Unit vectors and DMs of the catalogue's pulsars with a DM."""
        if self._psr is None:
            import h5py

            with h5py.File(self.catalogue, "r") as f:
                tab = f["pulsar"][:]
            ok = np.isfinite(tab["dm"]) & (tab["dm"] > 0) & np.isfinite(tab["ra"]) & np.isfinite(tab["dec"])
            if ok.sum() < self.n_neighbour:
                raise ValueError(f"Only {int(ok.sum())} pulsars with a DM in {self.catalogue}.")
            self._psr = (_radec_vectors(tab["ra"][ok], tab["dec"][ok]), np.asarray(tab["dm"][ok], dtype=float))
        return self._psr

    def draw(self, rng, ra, dec, distance=None):
        """DMs for sources at ``ra``, ``dec`` (ICRS degrees) and ``distance`` (kpc or NaN)."""
        ra, dec = np.atleast_1d(ra).astype(float), np.atleast_1d(dec).astype(float)
        n = ra.size
        if self.model == "pulsars":
            vec, dms = self._pulsars()
            cosd = _radec_vectors(ra, dec) @ vec.T  # source, pulsar
            near = np.argpartition(-cosd, self.n_neighbour - 1, axis=1)[:, : self.n_neighbour]
            dm = dms[near[np.arange(n), rng.integers(0, self.n_neighbour, n)]]
        else:
            import astropy.units as u
            from astropy.coordinates import SkyCoord

            dist = np.full(n, np.nan) if distance is None else np.atleast_1d(distance).astype(float).copy()
            drawn = self.distance.draw(rng, n)
            dist = np.where(np.isfinite(dist) & (dist > 0), dist, drawn)
            if self.distance_max is not None:
                dist = np.minimum(dist, float(self.distance_max))
            gal = SkyCoord(ra=ra * u.deg, dec=dec * u.deg).galactic
            gl, gb = gal.l.deg, gal.b.deg
            dm = np.empty(n)
            if self.model == "ymw16":
                try:
                    import pygedm
                except ImportError as exc:
                    raise ImportError("The ymw16 DM model needs the pygedm package.") from exc
                for i in range(n):
                    val, _ = pygedm.dist_to_dm(gl[i], gb[i], dist[i] * 1e3, method="ymw16")
                    dm[i] = getattr(val, "value", val)
            else:
                import contextlib
                import io

                try:
                    if self.model == "ne2025":
                        from mwprop.nemod.NE2025 import ne2025 as nemod
                    else:
                        from mwprop.nemod.NE2001 import ne2001 as nemod
                except ImportError as exc:
                    raise ImportError(f"The {self.model} DM model needs the mwprop package.") from exc
                # mwprop reads the parameters of whichever of its two models is imported
                # first in a process, and the other then silently evaluates the same one
                import mwprop.nemod.config_nemod as nemod_config

                flag = "eval_NE2025" if self.model == "ne2025" else "eval_NE2001"
                if not getattr(nemod_config, flag, False):
                    raise RuntimeError(
                        f"mwprop is set up for its other model in this process; {self.model} "
                        "needs a process in which it is the first mwprop model imported."
                    )
                for i in range(n):
                    # distance (kpc) to DM (ndir < 0); the model prints its warnings
                    with contextlib.redirect_stdout(io.StringIO()):
                        values = nemod(float(gl[i]), float(gb[i]), float(dist[i]), -1, classic=False,
                                       dmd_only=True, do_analysis=False, plotting=False, verbose=False)[1]  # fmt: skip
                    dm[i] = float(values["DM"])
        dm = self.scale * dm
        if self.dm_max is not None:
            dm = np.minimum(dm, float(self.dm_max))
        return dm


def _radec_vectors(ra, dec):
    ra, dec = np.radians(ra), np.radians(dec)
    return np.stack([np.cos(dec) * np.cos(ra), np.cos(dec) * np.sin(ra), np.sin(dec)], axis=-1)


# ---- polarisation -------------------------------------------------------------


class Poincare:
    """Polarisation state: a direction on the Poincare sphere times a degree.

    The direction is isotropic, with the circular (third) component scaled by
    ``circular_scale`` before normalising (< 1 favours linear polarisation); the
    degree of polarisation is drawn from ``degree``.
    """

    def __init__(self, degree=None, circular_scale=1.0):
        self.degree = Distribution.from_spec(
            degree if degree is not None else {"distribution": "uniform", "lower": 0.0, "upper": 1.0}
        )
        self.circular_scale = float(circular_scale)

    def draw(self, rng, number):
        """Returns the linear and circular fractions and the angle (degrees, 0-180)."""
        s = _unit_vectors(rng, number)
        s[:, 2] *= self.circular_scale
        s /= np.linalg.norm(s, axis=1, keepdims=True)
        p = np.clip(self.degree.draw(rng, number), 0.0, 1.0)
        lin = p * np.hypot(s[:, 0], s[:, 1])
        circ = p * s[:, 2]
        angle = np.degrees(0.5 * np.arctan2(s[:, 1], s[:, 0])) % 180.0
        return lin, circ, angle


# ---- the population -----------------------------------------------------------

#: Defaults: broad, flat (in log for scale parameters) distributions spanning the
#: detection threshold, for measuring a selection function rather than a prior.
#: Periods follow the power-law CDF fitted to the known LPTs (Suvorov et al. 2026,
#: alpha = 1/2, 7 min to 4 h); period 0 means one pulse a day (see GenerateLPTCatalog).
DEFAULT_POPULATION = {
    "sky": {"model": "isotropic", "dec_range": [-11.0, 90.0]},
    "flux": {"distribution": "loguniform", "lower": 0.003, "upper": 0.3},
    "period": {"distribution": "powerlaw", "index": -0.5, "lower": 420.0, "upper": 14400.0},
    "duration": {"distribution": "loguniform", "lower": 10.0, "upper": 300.0},
    "duty_cycle": None,
    "spectral_index": {"distribution": "uniform", "lower": -3.0, "upper": 1.0},
    "spectral_curvature": 0.0,
    "rm": {"distribution": "uniform", "lower": -300.0, "upper": 300.0},
    "polarisation": {"degree": {"distribution": "uniform", "lower": 0.0, "upper": 1.0}, "circular_scale": 1.0},
    "flux_scatter": 0.0,
    "null_fraction": 0.0,
    "scint_index": 0.0,
    "scint_bw": {"distribution": "loguniform", "lower": 0.5, "upper": 20.0},
    "t_offset_beam": {"distribution": "uniform", "lower": -1.0, "upper": 1.0},
    # Pulses made of sub-pulses (see SUBPULSE_DEFAULTS); none by default
    "subpulse": None,
    # Dispersion measure: None (no dispersion) or the arguments of DispersionMeasure
    "dm": None,
}

#: The sub-pulse model, for a ``fraction`` of the sources (see :func:`draw_population`).
#: Each such pulse is a Gaussian envelope of FWHM ``duration`` (s; drawn from
#: ``duration`` here, replacing the population's, then limited to half the period) with
#: sub-pulses in it: ``rate`` per minute of the FWHM (a Poisson mean), each a Gaussian
#: of FWHM about ``width`` (s), placed at random under the envelope without overlapping,
#: or, for a ``periodic_fraction`` of the sources, every ``period`` s with a jitter of
#: ``jitter`` periods; a smooth part of the envelope, ``floor``, lies under them.  The
#: defaults follow ASKAP J1839-0756 on CHIME (LSD 3749: ~6 features of 1-2 samples
#: over ~10 min, so ~1 a minute of FWHM) and GPM J1839-10 (30-300 s pulses with quasi-periodic substructure).
SUBPULSE_DEFAULTS = {
    "fraction": 0.0,
    "duration": {"distribution": "loguniform", "lower": 60.0, "upper": 1200.0},
    "rate": {"distribution": "loguniform", "lower": 0.2, "upper": 2.0},
    "width": {"distribution": "loguniform", "lower": 10.0, "upper": 30.0},
    "floor": {"distribution": "uniform", "lower": 0.0, "upper": 0.5},
    "periodic_fraction": 0.0,
    "period": {"distribution": "loguniform", "lower": 20.0, "upper": 120.0},
    "jitter": {"distribution": "uniform", "lower": 0.0, "upper": 0.3},
}

SUBPULSE_COLUMNS = ("subpulse_rate", "subpulse_width", "subpulse_floor", "subpulse_period", "subpulse_jitter")


def merge_population(spec=None):
    """The default population updated by ``spec`` (a dict of property specs)."""
    out = copy.deepcopy(DEFAULT_POPULATION)
    for k, v in (spec or {}).items():
        if k not in out:
            raise ValueError(f"Unknown population property {k!r}; have {sorted(out)}.")
        out[k] = copy.deepcopy(v)
    if out["subpulse"] is not None:
        sub = copy.deepcopy(SUBPULSE_DEFAULTS)
        for k, v in out["subpulse"].items():
            if k not in sub:
                raise ValueError(f"Unknown sub-pulse property {k!r}; have {sorted(sub)}.")
            sub[k] = copy.deepcopy(v)
        out["subpulse"] = sub
    return out


def draw_population(rng, number, spec=None):
    """Draw ``number`` sources.

    Returns a dict of arrays: ra, dec (ICRS degrees), distance (kpc or NaN), flux,
    period (s; 0 for one pulse a day), duration (s; at most half the period),
    spectral_index, spectral_curvature, rm, frac_lin, frac_circ, pol_angle,
    flux_scatter, null_fraction, scint_index, scint_bw and t_offset_beam, and the
    sub-pulse model (``SUBPULSE_COLUMNS``, all 0 for a smooth pulse; see
    :data:`SUBPULSE_DEFAULTS`): the rate of sub-pulses per minute of FWHM (0: none), their FWHM
    (s), the smooth floor, the spacing of quasi-periodic ones (s; 0: at random) and
    its jitter (periods), and the dispersion measure ``dm`` (pc cm^-3; 0 without a
    ``dm`` model, see :class:`DispersionMeasure`).  The sub-pulse draws come after all the others and are
    made only with a ``fraction`` above 0, so a population without them is drawn as
    before for a seed.
    """
    pop = merge_population(spec)
    out = {}
    out["ra"], out["dec"], out["distance"] = sky_from_spec(pop["sky"]).draw(rng, number)
    for key in ("flux", "period", "spectral_index", "spectral_curvature", "rm",
                "flux_scatter", "null_fraction", "scint_index", "scint_bw", "t_offset_beam"):  # fmt: skip
        out[key] = Distribution.from_spec(pop[key]).draw(rng, number)
    if pop["duty_cycle"] is not None:
        duty = Distribution.from_spec(pop["duty_cycle"]).draw(rng, number)
        out["duration"] = np.where(out["period"] > 0, duty * out["period"], np.nan)
        if np.any(out["period"] <= 0):
            fallback = Distribution.from_spec(pop["duration"]).draw(rng, number)
            out["duration"] = np.where(out["period"] > 0, out["duration"], fallback)
    else:
        out["duration"] = Distribution.from_spec(pop["duration"]).draw(rng, number)
    out["duration"] = np.where(
        out["period"] > 0, np.minimum(out["duration"], 0.5 * out["period"]), out["duration"]
    )
    pol = pop["polarisation"] or {}
    out["frac_lin"], out["frac_circ"], out["pol_angle"] = Poincare(**pol).draw(rng, number)
    out["null_fraction"] = np.clip(out["null_fraction"], 0.0, 1.0)
    for key in SUBPULSE_COLUMNS:
        out[key] = np.zeros(number)
    sub = pop["subpulse"]
    # drawn last, and only if used, so that a population without sub-pulses draws as before
    if sub is not None and float(sub["fraction"]) > 0.0:
        structured = rng.uniform(size=number) < float(sub["fraction"])
        dur = Distribution.from_spec(sub["duration"]).draw(rng, number)
        out["duration"] = np.where(structured, dur, out["duration"])
        out["duration"] = np.where(
            out["period"] > 0, np.minimum(out["duration"], 0.5 * out["period"]), out["duration"]
        )
        rate = Distribution.from_spec(sub["rate"]).draw(rng, number)
        width = Distribution.from_spec(sub["width"]).draw(rng, number)
        floor = np.clip(Distribution.from_spec(sub["floor"]).draw(rng, number), 0.0, 1.0)
        periodic = rng.uniform(size=number) < float(sub["periodic_fraction"])
        sub_period = Distribution.from_spec(sub["period"]).draw(rng, number)
        jitter = Distribution.from_spec(sub["jitter"]).draw(rng, number)
        out["subpulse_rate"] = np.where(structured, rate, 0.0)
        out["subpulse_width"] = np.where(structured, width, 0.0)
        out["subpulse_floor"] = np.where(structured, floor, 0.0)
        out["subpulse_period"] = np.where(structured & periodic, sub_period, 0.0)
        out["subpulse_jitter"] = np.where(structured & periodic, jitter, 0.0)
    # after everything else, and only if used, so that a population without it draws as before
    out["dm"] = np.zeros(number)
    if pop["dm"] is not None:
        out["dm"] = DispersionMeasure(**pop["dm"]).draw(rng, out["ra"], out["dec"], out["distance"])
    return out
