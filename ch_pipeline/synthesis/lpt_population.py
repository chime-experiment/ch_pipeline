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

Sky models: ``Isotropic`` (uniform on the sphere) and ``GalacticDisk`` (an
exponential disk, as used for Galactic compact binaries by Korol et al. 2022 and
for LPTs by Suvorov et al. 2026: surface density ~ exp(-R / scale_length), density
~ exp(-|z| / scale_height), the Sun at r_sun from the centre), both limited to a
range of declination.

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


SKY_MODELS = {"isotropic": Isotropic, "galacticdisk": GalacticDisk, "galactic_disk": GalacticDisk}


def sky_from_spec(spec):
    kw = dict(spec)
    name = str(kw.pop("model", "isotropic")).lower()
    if name not in SKY_MODELS:
        raise ValueError(f"Unknown sky model {name!r}; have {sorted(SKY_MODELS)}.")
    return SKY_MODELS[name](**kw)


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
}


def merge_population(spec=None):
    """The default population updated by ``spec`` (a dict of property specs)."""
    out = copy.deepcopy(DEFAULT_POPULATION)
    for k, v in (spec or {}).items():
        if k not in out:
            raise ValueError(f"Unknown population property {k!r}; have {sorted(out)}.")
        out[k] = copy.deepcopy(v)
    return out


def draw_population(rng, number, spec=None):
    """Draw ``number`` sources.

    Returns a dict of arrays: ra, dec (ICRS degrees), distance (kpc or NaN), flux,
    period (s; 0 for one pulse a day), duration (s; at most half the period),
    spectral_index, spectral_curvature, rm, frac_lin, frac_circ, pol_angle,
    flux_scatter, null_fraction, scint_index, scint_bw and t_offset_beam.
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
    return out
