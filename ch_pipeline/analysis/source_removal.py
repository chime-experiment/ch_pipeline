"""Tasks for removing bright sources from the data.

Tasks for constructing models for bright sources and subtracting them from the data.
"""

import json
import numpy as np

from scipy.constants import c as speed_of_light
import scipy.signal
import scipy.linalg

from caput import config
from ch_util import andata, ephemeris, tools
from ch_util.fluxcat import FluxCatalog
from draco.core import task, io
from draco.core import containers as dcontainers

from ..core import containers


def _correct_phase_wrap(phi):
    return ((phi + np.pi) % (2.0 * np.pi)) - np.pi


def model_extended_sources(
    freq,
    distance,
    timestamp,
    bodies,
    nstream=1,
    min_altitude=5.0,
    min_ha=0.0,
    max_ha=0.0,
    flag_lsd=None,
    avg_phi=False,
    **kwargs,
):
    """Generate a model for the visibilities.

    Model consists of the sum of the signal from multiple (possibly extended) sources.
    For extended sources, the dependence of the signal on (W-E, S-N) baseline is described
    by a 2D hermite polynomial.

    Parameters
    ----------
    freq : np.ndarray[nfreq,]
        Frequencies in MHz.
    distance : np.ndarray[2, nbaseline]
        The S-N and W-E distance for each baseline.
    timestamp : np.ndarray[nlsd, ntime]
        Unix timestamp.
    bodies : nsource element list
        List of `skyfield.api.Star` (or equivalent) for the sources being modelled.
    min_altitude : float
        Do not include a source in the model if it has an altitude less than
        this value in degrees.
    min_ha : float
        Do not include a source in the model if it has an hour angle less than
        this value in degrees.
    max_ha : float
        Do not include a source in the model if it has an hour angle greater than
        this value in degrees.
    scale_x : nsource element list
        Angular extent of each source in arcmin in the W-E direction
    scale_y : nsource element list
        Angular extent of each source in arcmin in the S-N direction
    degree_x : nsource element list
        The degree of the polynomial used to model the extend emission
        of each source in the W-E direction.
    degree_y : nsource element list
        The degree of the polynomial used to model the extend emission
        of each source in the S-N direction.
    degree_t : nsource element list
        The degree of the polynomial used to model time-dependent,
        common-mode variations in the gain.

    Returns
    -------
    S : np.ndarray[nfreq, nbaseline, ntime, nparam]
        Model for the visibilities.
    source_bound : np.ndarray[nsource+1,]
        Indices into the param axis of the source model.
        The model for source `i` is given by the parameters
        between `source_bound[i]` and `source_bound[i+1]`.
    """

    def interleave(x, n):
        return np.array([val for tup in zip(*([x] * n)) for val in tup])

    nsource = len(bodies)

    # Parse input parameters
    degree_x = np.array(kwargs["degree_x"])
    degree_y = np.array(kwargs["degree_y"])
    degree_t = np.array(kwargs.get("degree_t", [0] * nsource))

    expand_x = kwargs.get("expand_x", False)
    if expand_x:
        scale_x = np.ones(nsource)
    else:
        scale_x = np.radians(np.array(kwargs["scale_x"]) / 60.0)

    expand_y = kwargs.get("expand_y", False)
    if expand_y:
        scale_y = np.ones(nsource)
    else:
        scale_y = np.radians(np.array(kwargs["scale_y"]) / 60.0)

    scale_t = np.array(kwargs.get("scale_t", [0.66] * nsource))

    degree_x = np.array(kwargs["degree_x"])
    degree_y = np.array(kwargs["degree_y"])
    degree_t = np.array(kwargs.get("degree_t", [0] * nsource))

    scale = np.vstack((scale_x, scale_y, scale_t)).T
    poly_deg = np.vstack((degree_x, degree_y, degree_t)).T

    freq = np.atleast_1d(freq)
    nfreq = freq.size

    timestamp = np.atleast_1d(timestamp)
    if timestamp.ndim == 1:
        timestamp = timestamp.reshape(1, -1)
    nlsd, ntime = timestamp.shape

    nbaseline = distance.shape[-1]

    # Calculate baseline distances in wavelengths
    lmbda = speed_of_light * 1e-6 / freq
    u, v = (
        distance[:, np.newaxis, :, np.newaxis]
        / lmbda[np.newaxis, :, np.newaxis, np.newaxis]
    )

    # Setup for calculating source coordinates
    lat = np.radians(ephemeris.CHIMELATITUDE)
    date = ephemeris.unix_to_skyfield_time(timestamp.reshape(-1))

    observer = ephemeris.chime.skyfield_obs().at(date)

    # Generate polynomials
    ncoeff_x = degree_x + 1
    ncoeff_y = degree_y + 1
    ncoeff_t = degree_t + 1
    ncoeff = ncoeff_x * ncoeff_y * ncoeff_t
    nparam = np.sum(nstream * ncoeff)

    source_bound = np.concatenate(([0], np.cumsum(interleave(ncoeff, nstream))))

    S = np.zeros((nfreq, nbaseline, ntime, nparam), dtype=np.complex64)

    for ss, body in enumerate(bodies):

        # Calculate the source coordinates
        obs = observer.observe(body).apparent()
        src_radec = obs.cirs_radec(date)
        src_altaz = obs.altaz()

        src_ra, src_dec = src_radec[0], src_radec[1]
        src_alt, src_az = src_altaz[0], src_altaz[1]

        ha = _correct_phase_wrap(np.radians(ephemeris.lsa(timestamp)) - src_ra.radians.reshape(nlsd, ntime))
        dec = src_dec.radians.reshape(nlsd, ntime)

        weight = src_alt.radians.reshape(nlsd, ntime) > np.radians(min_altitude)

        if max_ha > 0.0:
            weight &= np.abs(ha) <= (np.radians(max_ha) / np.cos(dec))

        if min_ha > 0.0:
            weight &= np.abs(ha) >= (np.radians(min_ha) / np.cos(dec))

        # Construct a weight array to average over sidereal days
        if flag_lsd is None:
            w = weight.astype(float)
        else:
            w = (weight & flag_lsd).astype(float)

        wnorm = np.sum(w, axis=0)
        valid = np.flatnonzero(wnorm > 0.0)
        nvalid = valid.size

        if nvalid == 0:
            continue

        ha = ha[:, valid]
        dec = dec[:, valid]

        w = w[:, valid] * tools.invert_no_zero(wnorm[np.newaxis, valid])

        # Calculate the average over sidereal days
        avg_ha = np.sum(w * ha, axis=0)
        avg_dec = np.sum(w * dec, axis=0)

        # Evaluate polynomial
        ones = np.ones((nfreq, nbaseline, nvalid), dtype=np.float64)
        coords = [ax * scale[ss, ii] * ones for ii, ax in enumerate([u, v, avg_ha])]

        if expand_x:
            eta = -np.pi * np.cos(avg_dec) * np.sin(avg_ha)
            coords[0] = eta[np.newaxis, np.newaxis, :] * coords[0]

        if expand_y:
            eta = np.pi * (
                np.cos(lat) * np.sin(avg_dec) - np.sin(lat) * np.cos(avg_dec) * np.cos(avg_ha)
            )
            coords[1] = eta[np.newaxis, np.newaxis, :] * coords[1]

        H = np.polynomial.hermite.hermvander3d(*coords, poly_deg[ss])

        # Calculate the fringestop phase
        if avg_phi:

            phi = np.zeros((nfreq, nbaseline, nvalid), dtype=complex)
            for tt in range(nlsd):

                phi += w[tt, np.newaxis, np.newaxis, :] * tools.fringestop_phase(
                    ha[tt, np.newaxis, np.newaxis, :], lat, dec[tt, np.newaxis, np.newaxis, :], u, v
                ).conj()

        else:

            phi = tools.fringestop_phase(
                avg_ha[np.newaxis, np.newaxis, :], lat, avg_dec[np.newaxis, np.newaxis, :], u, v
            ).conj()

        model = H * phi[..., np.newaxis]

        for st in range(nstream):
            aa, bb = (
                source_bound[ss * nstream + st],
                source_bound[ss * nstream + st + 1],
            )
            S[..., valid, aa:bb] = model

    return S, source_bound


def solve_single_time(vis, weight, source_model):
    """Fit source model to the visibilities, treating each time independently.

    Parameters
    ----------
    vis : np.ndarray[nbaseline, ntime]
        Measured visibilities.
    weight : np.ndarray[nbaseline, ntime]
        1/sigma^2 uncertainty on the visibilities.
    source_model : np.ndarray[nbaseline, ntime, nparam]
        Model for the visibilities generated by the
        `model_extended_sources` method.

    Returns
    -------
    coeff : np.ndarray[ntime, nparam]
        Best-fit coefficients of the model for each time.
    """
    nbaseline, ntime, nparam = source_model.shape

    coeff = np.zeros((ntime, nparam), dtype=np.complex64)

    flag = np.any(np.abs(source_model) > 0.0, axis=0)

    for tt in range(ntime):

        valid = np.flatnonzero(flag[tt])

        if valid.size == 0:
            continue

        isigma = np.sqrt(weight[:, tt])
        S = source_model[:, tt, valid]

        coeff[tt, valid] = scipy.linalg.lstsq(isigma[:, np.newaxis] * S,
                                              isigma * vis[:, tt],
                                              lapack_driver='gelsy',
                                              check_finite=False)[0]

    return coeff


def solve_multiple_times(vis, weight, source_model):
    """Fit source model to the visibilities at all times.

    Parameters
    ----------
    vis : np.ndarray[nbaseline, ntime]
        Measured visibilities.
    weight : np.ndarray[nbaseline, ntime]
        1/sigma^2 uncertainty on the visibilities.
    source_model : np.ndarray[nbaseline, ntime, nparam]
        Model for the visibilities generated by the
        `model_extended_sources` method.

    Returns
    -------
    coeff : np.ndarray[nparam,]
        Best-fit coefficients of the model.
    """
    nbaseline, ntime, nparam = source_model.shape

    isigma = np.sqrt(weight.reshape(-1))
    S = source_model.reshape(-1, nparam)

    coeff = scipy.linalg.lstsq(isigma[:, np.newaxis] * S,
                               isigma * vis.reshape(-1),
                               lapack_driver='gelsy',
                               check_finite=False)[0]

    return coeff


class SolveSources(task.SingleTask):
    """Fit source model to the visibilities.

    Model consists of the sum of the signal from multiple (possibly extended) sources.
    Note that the extended source option is still in development.

    Attributes
    ----------
    sources : nsource element list
        Names of the sources that will be included in the model.
    degree_x : nsource element list
        The degree of the polynomial used to model the extended emission
        of each source in the W-E direction.  Set to zero for point source (default).
    degree_y : nsource element list
        The degree of the polynomial used to model the extended emission
        of each source in the S-N direction.  Set to zero for point source (default).
    extent : nsource element list
        Angular extent of each source in arcmin.
    min_altitude : float
        Do not include a source in the model if its altitude is less than
        this value in degrees.
    min_ha : float
        Do not include a source in the model if it has an hour angle less than
        this value in degrees.
    max_ha : float
        Do not include a source in the model if it has an hour angle greater than
        this value in degrees.
    min_distance : list
        Do not include baselines in the fit with a distance less than
        this value in meters.  If the list contains a single element,
        then the cut is placed on the total baseline distance.  If the list
        contains two or more elements, then the cut is placed on the
        [W-E, N-S, ...] component.
    telescope_rotation : float
        Rotation of the telescope from true north in degrees.  A positive rotation is
        anti-clockwise when looking down at the telescope from the sky.
    max_iter : int
        Maximum number of iterations to perform for extended sources.
    """

    sources = config.Property(
        proptype=list, default=["CYG_A", "CAS_A", "TAU_A", "VIR_A"]
    )
    degree_x = config.Property(proptype=list, default=[0, 0, 0, 0])
    degree_y = config.Property(proptype=list, default=[0, 0, 0, 0])
    degree_t = config.Property(proptype=list, default=[0, 0, 0, 0])

    extent = config.Property(proptype=list, default=[1.0, 4.0, 4.0, 7.0])
    scale_t = config.Property(proptype=list, default=[0.66, 0.66, 0.66, 0.66])

    expand_x = config.Property(proptype=bool, default=False)
    expand_y = config.Property(proptype=bool, default=False)

    min_altitude = config.Property(proptype=float, default=5.0)
    max_ha = config.Property(proptype=float, default=0.0)
    min_ha = config.Property(proptype=float, default=0.0)

    min_distance = config.Property(proptype=list, default=[10.0, 0.0])
    telescope_rotation = config.Property(proptype=float, default=tools._CHIME_ROT)
    max_iter = config.Property(proptype=int, default=4)
    avg_phi = config.Property(proptype=bool, default=False)

    flag_lsd = None

    def setup(self, tel):
        """Set up the source model.

        Parameters
        ----------
        tel : analysis.telescope.CHIMETelescope
        """
        telescope = io.get_telescope(tel)
        self.inputmap = telescope.feeds

        self.bodies = [
            ephemeris.source_dictionary[src]
            if src in ephemeris.source_dictionary
            else ephemeris.skyfield_wrapper.ephemeris[src]
            for src in self.sources
        ]

        self.nsources = len(self.sources)

        # Set up kwargs for various source models
        self.point_source_kwargs = {
            "degree_x": [0] * self.nsources,
            "degree_y": [0] * self.nsources,
            "degree_t": [0] * self.nsources,
            "scale_x": [1] * self.nsources,
            "scale_y": [1] * self.nsources,
            "scale_t": [1] * self.nsources,
            "expand_x": self.expand_x,
            "expand_y": self.expand_y,
            "min_altitude": self.min_altitude,
            "max_ha": self.max_ha,
            "min_ha": self.min_ha,
            "avg_phi": self.avg_phi,
        }

        if any(self.degree_x) or any(self.degree_y):
            self.extended_source_kwargs = {
                "degree_x": self.degree_x,
                "degree_y": self.degree_y,
                "degree_t": self.degree_t,
                "scale_x": self.extent,
                "scale_y": self.extent,
                "scale_t": self.scale_t,
                "expand_x": self.expand_x,
                "expand_y": self.expand_y,
                "min_altitude": self.min_altitude,
                "max_ha": self.max_ha,
                "min_ha": self.min_ha,
                "avg_phi": self.avg_phi,
            }
        else:
            self.extended_source_kwargs = {}

    def process(self, data):
        """Fit source model to visibilities.

        Parameters
        ----------
        data : andata.CorrData, core.containers.SiderealStream, or equivalent

        Returns
        -------
        out : core.containers.SourceModel
            Best-fit parameters of the source model.
        """
        # Distribute over frequencies
        data.redistribute("freq")

        # Determine local dimensions
        nfreq, nstack, ntime = data.vis.local_shape

        # Find the local frequencies
        sfreq = data.vis.local_offset[0]
        efreq = sfreq + nfreq

        freq = data.freq[sfreq:efreq]

        # Calculate time
        if "ra" in data.index_map:

            if self.flag_lsd is None:

                lsd = np.atleast_1d(data.attrs["lsd"] if "lsd" in data.attrs else data.attrs["csd"])
                timestamp = ephemeris.csd_to_unix(lsd[:, np.newaxis] + data.ra[np.newaxis, :] / 360.0)
                flag_lsd = None

            else:

                self.log.info(f"Using timestamps and flags for {self.flag_lsd.lsd.size} sidereal days.")
                lsd = self.flag_lsd.lsd[:]
                timestamp = self.flag_lsd.timestamp[:]
                flag_lsd = self.flag_lsd.flag[:]

                if not np.array_equal(self.flag_lsd.ra, data.ra):
                    raise RuntimeError("The sidereal day flag is not sampled at the correct RAs.")

        elif "time" in data.index_map:

            timestamp = data.time.reshape(1, ntime)
            lsd = np.atleast_1d(np.fix(np.mean(ephemeris.unix_to_csd(timestamp))))
            flag_lsd = None

        else:
            raise RuntimeError("Unable to extract time from input container.")

        nlsd = timestamp.shape[0]

        # Redefine stack axis so that it only contains chime antennas
        stack_new, stack_flag = tools.redefine_stack_index_map(
            self.inputmap,
            data.index_map["prod"],
            data.index_map["stack"],
            data.reverse_map["stack"],
        )

        prod_new = data.index_map["prod"][stack_new["prod"]]

        # Swap the product pair order for conjugated stack indices
        cj = np.flatnonzero(stack_new["conjugate"].astype(bool))
        if cj.size > 0:
            prod_new["input_a"][cj], prod_new["input_b"][cj] = (
                prod_new["input_b"][cj],
                prod_new["input_a"][cj],
            )

        # Calculate baseline distances
        tools.change_chime_location(rotation=self.telescope_rotation)
        feedpos = tools.get_feed_positions(self.inputmap).T
        distance = feedpos[:, prod_new["input_a"]] - feedpos[:, prod_new["input_b"]]
        self.log.info("Rotation set to %0.4f deg" % self.inputmap[0]._rotation)
        tools.change_chime_location(default=True)

        # Flag out short baselines
        min_distance = np.array(self.min_distance)
        sep = np.sqrt(np.sum(distance**2, axis=0))
        if min_distance.size == 1:
            baseline_weight = (sep >= min_distance[0]).astype(np.float32)
        else:
            baseline_weight = np.all(
                distance >= min_distance[:, np.newaxis], axis=0
            ).astype(np.float32)

        # Calculate polarisation products, determine unique values
        feedpol = tools.get_feed_polarisations(self.inputmap)
        pol = np.core.defchararray.add(
            feedpol[prod_new["input_a"]], feedpol[prod_new["input_b"]]
        )

        upol = np.unique(pol)
        npol = len(upol)

        # Determine parameter names
        param_name = []
        for ss, src in enumerate(self.sources):
            npar = (
                (self.degree_x[ss] + 1)
                * (self.degree_y[ss] + 1)
                * (self.degree_t[ss] + 1)
            )
            for ii in range(npar):
                param_name.append((src, 0, ii))
        param_name = np.array(
            param_name, dtype=[("source", "U32"), ("stream", "u2"), ("coeff", "u2")]
        )

        # Create output container
        out = containers.SourceModel(
            lsd=lsd,
            pol=upol,
            time=timestamp[0],
            source=np.array(self.sources),
            param=param_name,
            axes_from=data,
            attrs_from=data,
            distributed=data.distributed,
            comm=data.comm,
        )

        # Determine extended source model
        if self.extended_source_kwargs:
            out.add_dataset("amplitude")
            out.add_dataset("coeff")

            out.attrs["source_model_kwargs"] = json.dumps(self.extended_source_kwargs)

        else:
            out.add_dataset("amplitude")
            out.attrs["source_model_kwargs"] = json.dumps(self.point_source_kwargs)

        # Dereference datasets
        all_vis = data.vis[:].view(np.ndarray)
        all_weight = data.weight[:].view(np.ndarray)

        out.redistribute("freq")
        out_amplitude = out.amplitude[:].view(np.ndarray)
        out_amplitude[:] = 0.0
        if self.extended_source_kwargs:
            out_coeff = out.coeff[:].view(np.ndarray)
            out_coeff[:] = 0.0

        if nlsd > 1:
            out.add_dataset("timestamp")
            out.datasets["timestamp"][:] = timestamp

            if flag_lsd is not None:
                out.add_dataset("flag_lsd")
                out.datasets["flag_lsd"][:] = flag_lsd

        # Loop over polarisations
        for pp, upp in enumerate(upol):

            this_pol = np.flatnonzero(pol == upp)

            dist_pol = distance[:, this_pol]
            bweight_pol = baseline_weight[this_pol, np.newaxis]

            # Loop over frequencies
            for ff, nu in enumerate(freq):

                # Extract datasets for this polarisation and frequency
                vis = all_vis[ff, this_pol, :]
                weight = all_weight[ff, this_pol, :] * bweight_pol

                # Determine the initial source model, assuming all sources are point sources
                psrc_model, _ = model_extended_sources(
                    nu, dist_pol, timestamp, self.bodies, flag_lsd=flag_lsd, **self.point_source_kwargs
                )
                psrc_model = psrc_model[0]

                # Obtain initial estimate of each source assuming point source
                amplitude = solve_single_time(vis, weight, psrc_model)

                # If modeling extended sources, iterate over time-dependent normalization
                # and baseline dependent response.  Assumes the description of the extended
                # emission is constant in time.
                if self.extended_source_kwargs:

                    ext_model, sedge = model_extended_sources(
                        nu,
                        dist_pol,
                        timestamp,
                        self.bodies,
                        flag_lsd=flag_lsd,
                        **self.extended_source_kwargs,
                    )
                    ext_model = ext_model[0]

                    iters = 0
                    while iters < self.max_iter:

                        model = ext_model.copy()
                        for ss in range(self.nsources):
                            model[:, :, sedge[ss] : sedge[ss + 1]] *= amplitude[
                                np.newaxis, :, ss, np.newaxis
                            ]

                        coeff = solve_multiple_times(vis, weight, model)

                        # Correct for extended structure
                        model = np.zeros_like(psrc_model)
                        for ss in range(self.nsources):
                            model[:, :, ss] = np.sum(
                                ext_model[..., sedge[ss] : sedge[ss + 1]]
                                * coeff[sedge[ss] : sedge[ss + 1]],
                                axis=-1,
                            )

                        # Obtain initial estimate of coefficients assuming point source
                        amplitude = solve_single_time(vis, weight, model)

                        # Increment counter
                        iters += 1

                    # Save results to container
                    out_coeff[ff, pp, :] = coeff

                out_amplitude[ff, pp, :, :] = amplitude

        # Save a few attributes necessary to interpret the data
        out.attrs["min_distance"] = self.min_distance
        out.attrs["telescope_rotation"] = self.telescope_rotation

        return out


class SolveSourcesStack(SolveSources):

    def process(self, data, flag_lsd):

        self.flag_lsd = flag_lsd

        return super().process(data)


class LPFSourceAmplitude(task.SingleTask):
    """Apply a 2D low-pass filter in (freq, time) to the measured source amplitude.

    Attributes
    ----------
    window : list
        The size of the moving average window along the
        [freq, time] axis.  This sets the cutoff scale of
        the low-pass filter.
    niter : list
        Number of iterations of the moving average filter
        along the [freq, time] axis.  The peak-to-sidelobe
        ratio of the filter's transfer function scales
        roughly as 0.05^niter.
    frac_required : float
        The fraction of samples within a window that must be
        valid (unmasked) in order for the filtered data point
        to be considered valid.
    ignore_main_lobe : boolean
        Do not apply the filter within the main lobe of the
        primary beam.
    main_lobe_threshold : float
        If the source amplitude is greater than this fraction
        of the source flux than than [freq, RA] is considered
        within main lobe.  Only relevant when ignore_main_lobe
        is True.
    """

    window = config.Property(proptype=list, default=[3, 5])
    niter = config.Property(proptype=list, default=[12, 8])
    frac_required = config.Property(proptype=float, default=0.40)

    ignore_main_lobe = config.Property(proptype=bool, default=True)
    main_lobe_threshold = config.Property(proptype=float, default=0.05)

    def process(self, model):
        """Low-pass filter the provided source amplitudes.

        Parameters
        ----------
        model : containers.SourceModel
            Best-fit parameters of the source model.

        Returns
        -------
        model : containers.SourceModel
            The input container with the amplitude dataset
            filtered.  Note that amplitudes that were previously
            zero will be interpolated to a non-zero value if more than
            frac_required of the nearby data points are non-zero.
        """

        model.redistribute("pol")
        npol = model.amplitude.local_shape[1]

        amp = model.amplitude[:].view(np.ndarray)

        for ss, src in enumerate(model.source):

            flux = FluxCatalog[src].predict_flux(model.freq)
            inv_flux = tools.invert_no_zero(flux)[:, np.newaxis]

            for pp in range(npol):

                a = amp[:, pp, :, ss]
                flag = np.abs(a) > 0

                alpf = apply_kz_lpf_2d(
                    a,
                    flag,
                    window=self.window,
                    niter=self.niter,
                    mode=["reflect", "wrap"],
                    frac_required=self.frac_required,
                )

                if self.ignore_main_lobe:
                    use_lpf = np.abs(alpf * inv_flux) < self.main_lobe_threshold
                    alpf = np.where(use_lpf, alpf, a)

                amp[:, pp, :, ss] = alpf

        return model


class SubtractSources(task.SingleTask):
    """Subtract a source model from the visibilities."""

    def setup(self, tel):
        """Extract inputmap from the telescope instance provided.

        Parameters
        ----------
        tel : analysis.telescope.CHIMETelescope
        """
        telescope = io.get_telescope(tel)
        self.inputmap = telescope.feeds

    def process(self, data, model):
        """Subtract a source model from the visibilities.

        Parameters
        ----------
        data : andata.CorrData, core.containers.SiderealStream, or equivalent

        model : core.containers.SourceModel
            Best-fit parameters of the source model.

        Returns
        -------
        data : andata.CorrData, core.containers.SiderealStream, or equivalent
        """
        # Extract various arguments describing the model from the attributes of the model container
        sources = model.index_map["source"]
        telescope_rotation = model.attrs["telescope_rotation"]
        source_model_kwargs = json.loads(model.attrs["source_model_kwargs"])

        bodies = [
            ephemeris.source_dictionary[src]
            if src in ephemeris.source_dictionary
            else ephemeris.skyfield_wrapper.ephemeris[src]
            for src in sources
        ]

        # Distribute over frequencies
        data.redistribute("freq")
        model.redistribute("freq")

        # Determine local dimensions
        nfreq, nstack, ntime = data.vis.local_shape

        # Find the local frequencies
        sfreq = data.vis.local_offset[0]
        efreq = sfreq + nfreq

        freq = data.freq[sfreq:efreq]

        # Calculate time
        if "ra" in data.index_map:

            if "timestamp" in model.datasets:
                timestamp = model.datasets["timestamp"][:]
                flag_lsd = model.datasets["flag_lsd"][:]
            else:
                timestamp = model.index_map["time"][:].reshape(1, -1)
                flag_lsd = None

        elif "time" in data.index_map:
            timestamp = data.time.reshape(1, -1)
            flag_lsd = None

        else:
            raise RuntimeError("Unable to extract time from input container.")

        # Redefine stack axis so that it only contains chime antennas
        stack_new, stack_flag = tools.redefine_stack_index_map(
            self.inputmap,
            data.index_map["prod"],
            data.index_map["stack"],
            data.reverse_map["stack"],
        )

        prod_new = data.index_map["prod"][stack_new["prod"]]

        # Swap the product pair order for conjugated stack indices
        cj = np.flatnonzero(stack_new["conjugate"].astype(bool))
        if cj.size > 0:
            prod_new["input_a"][cj], prod_new["input_b"][cj] = (
                prod_new["input_b"][cj],
                prod_new["input_a"][cj],
            )

        # Calculate baseline distances
        tools.change_chime_location(rotation=telescope_rotation)
        feedpos = tools.get_feed_positions(self.inputmap).T
        distance = feedpos[:, prod_new["input_a"]] - feedpos[:, prod_new["input_b"]]
        self.log.info("Rotation set to %0.4f deg" % self.inputmap[0]._rotation)
        tools.change_chime_location(default=True)

        # Calculate polarisation products, determine unique values
        feedpol = tools.get_feed_polarisations(self.inputmap)
        pol = np.core.defchararray.add(
            feedpol[prod_new["input_a"]], feedpol[prod_new["input_b"]]
        )

        # Dereference dataset
        vis = data.vis[:].view(np.ndarray)
        amp = model.amplitude[:].view(np.ndarray)

        coeff = model.coeff[:].view(np.ndarray) if "coeff" in model.datasets else None

        # Subtract source model
        for pp, upp in enumerate(model.index_map["pol"]):

            this_pol = np.flatnonzero(pol == upp)
            dist_pol = distance[:, this_pol]

            for ff, nu in enumerate(freq):

                # Calculate source model
                source_model, sedge = model_extended_sources(
                    nu, dist_pol, timestamp, bodies, flag_lsd=flag_lsd, **source_model_kwargs
                )
                source_model = source_model[0]

                # Sum over coefficients of source model
                if coeff is not None:
                    mdl = np.sum(
                        amp[ff, pp, np.newaxis, :, :][..., model.source_index]
                        * coeff[ff, pp, np.newaxis, np.newaxis, :]
                        * source_model,
                        axis=-1,
                    )
                else:
                    mdl = np.sum(
                        amp[ff, pp, np.newaxis, :, :] * source_model,
                        axis=-1,
                    )

                vis[ff, this_pol, :] -= mdl

        return data


class AccumulateBeam(task.SingleTask):
    """Accumulate the stacked beam for each source."""

    def setup(self):
        """Create a class dictionary to hold the beam for each source."""

        self.beam_stack = {}

    def process(self, beam_stack):
        """Add the beam for this source to the class dictionary."""

        self.beam_stack[beam_stack.attrs["source_name"]] = beam_stack

        return None

    def process_finish(self):
        """Return the class dictionary containing the beam for all sources."""

        return self.beam_stack


class SolveSourcesWithBeam(SolveSources):
    """Fit a source model to the visibilities using external measurements of the beam.

    Model consists of the sum of the signal from multiple (possibly extended) sources.
    The signal from each source is modelled as a static 2D hermite polynomial in (u, v)
    modulated by three factors: the external beam measurement, the geometric phase,
    and a hermite polynomial in time to account for common mode drift in the gain.
    """

    time_variable = config.Property(proptype=bool, default=False)
    joint_pol = config.Property(proptype=bool, default=False)

    flag_lsd = None

    def setup(self, tel):
        """Set up the source model.

        Parameters
        ----------
        tel : analysis.telescope.CHIMETelescope
        """
        telescope = io.get_telescope(tel)
        self.inputmap = telescope.feeds

        self.bodies = [
            ephemeris.source_dictionary[src]
            if src in ephemeris.source_dictionary
            else ephemeris.skyfield_wrapper.ephemeris[src]
            for src in self.sources
        ]

        self.nsources = len(self.sources)

        # Set up kwargs for various source models
        self.source_kwargs = {
            "degree_x": self.degree_x,
            "degree_y": self.degree_y,
            "degree_t": self.degree_t,
            "scale_x": self.extent,
            "scale_y": self.extent,
            "scale_t": self.scale_t,
            "expand_x": self.expand_x,
            "expand_y": self.expand_y,
            "min_altitude": self.min_altitude,
            "max_ha": self.max_ha,
            "min_ha": self.min_ha,
            "avg_phi": self.avg_phi,
        }

        # Check if we are allowing the amplitude to vary as a function of time
        # and define the appropriate solver
        if self.time_variable:
            assert not any(self.degree_x)
            assert not any(self.degree_y)
            assert not any(self.degree_t)
            self.solver = solve_single_time
        else:
            self.solver = solve_multiple_times

    def process(self, data, beams):
        """Fit source model to visibilities.

        Parameters
        ----------
        data : andata.CorrData, core.containers.SiderealStream, or equivalent

        beams : dict of andata.CorrData, core.containers.SiderealSteam, or equivalent
            Dictionary containing the beam measurements.  The keys must be
            the souce names and the values must be containers of the same type as the
            input data, which contain the holographic measurements of that source
            averaged over all pairs of feeds in the stacked baselines.

        Returns
        -------
        out : core.containers.SourceModel
            Best-fit parameters of the source model.
        """
        # Distribute over frequencies
        data.redistribute("freq")

        # Determine local dimensions
        nfreq, nstack, ntime = data.vis.local_shape

        nstream = np.unique([len(val.index_map["stream"][:]) for val in beams.values()])
        if len(nstream) > 1:
            raise RuntimeError("All sources must have the same number of streams.")
        else:
            nstream = int(nstream[0])

        # Find the local frequencies
        sfreq = data.vis.local_offset[0]
        efreq = sfreq + nfreq

        freq = data.freq[sfreq:efreq]

        # Calculate time
        bmatch = {}
        if "ra" in data.index_map:

            for key, val in beams.items():
                imatch = np.searchsorted(data.ra, val.ra)
                if np.array_equal(data.ra[imatch], val.ra):
                    bmatch[key] = imatch
                else:
                    raise RuntimeError(
                        f"The beam for {key} is not sampled at the correct RAs."
                    )

            if self.flag_lsd is None:

                lsd = np.atleast_1d(data.attrs["lsd"] if "lsd" in data.attrs else data.attrs["csd"])
                timestamp = ephemeris.csd_to_unix(lsd[:, np.newaxis] + data.ra[np.newaxis, :] / 360.0)
                flag_lsd = None

            else:

                self.log.info(f"Using timestamps and flags for {self.flag_lsd.lsd.size} sidereal days.")
                lsd = self.flag_lsd.lsd[:]
                timestamp = self.flag_lsd.timestamp[:]
                flag_lsd = self.flag_lsd.flag[:]

                if not np.array_equal(self.flag_lsd.ra, data.ra):
                    raise RuntimeError("The sidereal day flag is not sampled at the correct RAs.")

        elif "time" in data.index_map:

            for key, val in beams.items():
                imatch = np.searchsorted(data.time, val.time)
                if np.array_equal(data.time[imatch], val.time):
                    bmatch[key] = imatch
                else:
                    raise RuntimeError(
                        f"The beam for {key} is not sampled at the correct RAs."
                    )

            timestamp = data.time.reshape(1, ntime)
            lsd = np.atleast_1d(np.fix(np.mean(ephemeris.unix_to_csd(timestamp))))
            flag_lsd = None

        else:
            raise RuntimeError("Unable to extract time from input container.")

        nlsd = timestamp.shape[0]

        # Redefine stack axis so that it only contains chime antennas
        stack_new, stack_flag = tools.redefine_stack_index_map(
            self.inputmap,
            data.index_map["prod"],
            data.index_map["stack"],
            data.reverse_map["stack"],
        )

        prod_new = data.index_map["prod"][stack_new["prod"]]

        # Swap the product pair order for conjugated stack indices
        cj = np.flatnonzero(stack_new["conjugate"].astype(bool))
        if cj.size > 0:
            prod_new["input_a"][cj], prod_new["input_b"][cj] = (
                prod_new["input_b"][cj],
                prod_new["input_a"][cj],
            )

        # Calculate baseline distances
        tools.change_chime_location(rotation=self.telescope_rotation)
        feedpos = tools.get_feed_positions(self.inputmap).T
        distance = feedpos[:, prod_new["input_a"]] - feedpos[:, prod_new["input_b"]]
        self.log.info("Rotation set to %0.4f deg" % self.inputmap[0]._rotation)
        tools.change_chime_location(default=True)

        # Flag out short baselines
        min_distance = np.array(self.min_distance)
        sep = np.sqrt(np.sum(distance**2, axis=0))
        if min_distance.size == 1:
            baseline_weight = (sep >= min_distance[0]).astype(np.float32)
        else:
            baseline_weight = np.all(
                distance >= min_distance[:, np.newaxis], axis=0
            ).astype(np.float32)

        # Calculate polarisation products, determine unique values
        feedpol = tools.get_feed_polarisations(self.inputmap)

        # Determine polarisation axis
        if self.joint_pol:
            # Jointly fit all polarisation products
            upol = np.array(["co", "cross"], dtype="<U8")
            apol, bpol = feedpol[prod_new["input_a"]], feedpol[prod_new["input_b"]]
            pol_index = [np.flatnonzero(apol == bpol), np.flatnonzero(apol != bpol)]
            pol_conj = [np.flatnonzero(apol[pind] > bpol[pind]) for pind in pol_index]

            self.log.info("Fitting all polarisations jointly.")

        else:
            pol = np.core.defchararray.add(
                feedpol[prod_new["input_a"]], feedpol[prod_new["input_b"]]
            )

            upol = np.unique(pol)
            pol_index = [np.flatnonzero(pol == upp) for upp in upol]
            pol_conj = [np.array([]) for upp in upol]

            self.log.info(f"Fitting each polarisation ({', '.join(upol)}) separately.")

        # Determine parameter names
        param_name = []
        for ss, src in enumerate(self.sources):
            npar = (
                (self.degree_x[ss] + 1)
                * (self.degree_y[ss] + 1)
                * (self.degree_t[ss] + 1)
            )
            for st in range(nstream):
                for ii in range(npar):
                    param_name.append((src, st, ii))
        param_name = np.array(
            param_name, dtype=[("source", "U32"), ("stream", "u2"), ("coeff", "u2")]
        )

        # Create output container
        out = containers.SourceModel(
            lsd=lsd,
            pol=upol,
            time=timestamp[0],
            source=np.array(self.sources),
            param=param_name,
            axes_from=data,
            attrs_from=data,
            distributed=data.distributed,
            comm=data.comm,
        )

        source_kwargs = self.source_kwargs.copy()
        source_kwargs["nstream"] = nstream
        out.attrs["source_model_kwargs"] = json.dumps(source_kwargs)

        if self.time_variable:
            out.add_dataset("amplitude")
            out.redistribute("freq")
            out_dset = out.amplitude[:].view(np.ndarray)
        else:
            out.add_dataset("coeff")
            out.redistribute("freq")
            out_dset = out.coeff[:].view(np.ndarray)

        if nlsd > 1:
            out.add_dataset("timestamp")
            out.datasets["timestamp"][:] = timestamp

            if flag_lsd is not None:
                out.add_dataset("flag_lsd")
                out.datasets["flag_lsd"][:] = flag_lsd

        out_dset[:] = 0.0

        # Dereference datasets
        all_vis = data.vis[:].view(np.ndarray)
        all_weight = data.weight[:].view(np.ndarray)

        bvis, bweight = {}, {}
        for key, val in beams.items():
            val.redistribute("freq")
            bvis[key] = val.vis[:].view(np.ndarray)
            bweight[key] = val.weight[:].view(np.ndarray)

        # Loop over polarisations
        for pp, this_pol in enumerate(pol_index):

            dist_pol = distance[:, this_pol]
            bweight_pol = baseline_weight[this_pol, np.newaxis]

            # Loop over frequencies
            for ff, nu in enumerate(freq):

                # Extract datasets for this polarisation and frequency
                vis = all_vis[ff, this_pol, :].copy()
                weight = all_weight[ff, this_pol, :] * bweight_pol

                # Only process times where we have a beam
                flag_beam = np.zeros((self.nsources, ntime), dtype=bool)
                for ss, src in enumerate(self.sources):
                    flag_beam[ss, bmatch[src]] = np.any(
                        bweight[src][ff] > 0.0, axis=(0, 1)
                    )

                valid_time = np.flatnonzero(np.any(flag_beam, axis=0))
                self.log.info(
                    f"There are {valid_time.size} valid times for frequency {ff}."
                )
                if valid_time.size == 0:
                    continue

                vis = vis[:, valid_time]
                weight = weight[:, valid_time]
                flg = flag_lsd[:, valid_time] if flag_lsd is not None else None

                # Determine extended source model
                source_model, sedge = model_extended_sources(
                    nu,
                    dist_pol,
                    timestamp[:, valid_time],
                    self.bodies,
                    nstream=nstream,
                    flag_lsd=flg,
                    **self.source_kwargs,
                )
                source_model = source_model[0]

                # Multipy source model by the effective beam
                for ss, src in enumerate(self.sources):

                    valid_in = np.flatnonzero(flag_beam[ss, bmatch[src]])
                    valid_out = np.flatnonzero(flag_beam[ss, valid_time])

                    for st in range(nstream):

                        aa, bb = sedge[ss * nstream + st], sedge[ss * nstream + st + 1]

                        this_flag = (
                            bweight[src][ff, :, st, :][this_pol][:, valid_in] > 0.0
                        ).astype(np.float32)
                        this_beam = (
                            bvis[src][ff, :, st, :][this_pol][:, valid_in] * this_flag
                        )
                        source_model[..., valid_out, aa:bb] *= this_beam[
                            ..., np.newaxis
                        ]

                # Conjugate if necessary.
                to_conj = pol_conj[pp]
                if to_conj.size > 0:
                    vis[to_conj] = vis[to_conj].conj()
                    source_model[to_conj] = source_model[to_conj].conj()

                # Solve for the coefficients
                out_dset[ff, pp] = self.solver(vis, weight, source_model)

        # Save a few attributes necessary to interpret the data
        out.attrs["min_distance"] = min_distance
        out.attrs["telescope_rotation"] = self.telescope_rotation

        return out


class SolveSourcesWithBeamStack(SolveSourcesWithBeam):

    def process(self, data, beams, flag_lsd):

        self.flag_lsd = flag_lsd

        return super().process(data, beams)


class SubtractSourcesWithBeam(task.SingleTask):
    """Subtract a source model from the visibilities."""

    save_model = config.Property(proptype=bool, default=False)
    min_ha = config.Property(proptype=float, default=None)
    max_ha = config.Property(proptype=float, default=None)

    def setup(self, tel):
        """Extract inputmap from the telescope instance provided.

        Parameters
        ----------
        tel : analysis.telescope.CHIMETelescope
        """
        telescope = io.get_telescope(tel)
        self.inputmap = telescope.feeds

    def process(self, data, model, beams):
        """Subtract a source model from the visibilities.

        Parameters
        ----------
        data : andata.CorrData, core.containers.SiderealStream, or equivalent

        model : core.containers.SourceModel
            Best-fit parameters of the source model.

        beams : dict of andata.CorrData, core.containers.SiderealSteam, or equivalent
            Dictionary containing the beam measurements.  The keys must be
            the souce names and the values must be containers of the same type as the
            input data, which contain the holographic measurements of that source
            averaged over all pairs of feeds in the stacked baselines.

        Returns
        -------
        data : andata.CorrData, core.containers.SiderealStream, or equivalent
        """
        # Extract various arguments describing the model from the attributes of the model container
        sources = model.index_map["source"]
        nsources = sources.size
        telescope_rotation = model.attrs["telescope_rotation"]
        source_model_kwargs = json.loads(model.attrs["source_model_kwargs"])
        nstream = source_model_kwargs["nstream"]

        if self.min_ha is not None:
            source_model_kwargs["min_ha"] = self.min_ha
            self.log.info(f"Resetting min_ha to {self.min_ha:0.1f}.")

        if self.max_ha is not None:
            source_model_kwargs["max_ha"] = self.max_ha
            self.log.info(f"Resetting max_ha to {self.max_ha:0.1f}.")

        bodies = [
            ephemeris.source_dictionary[src]
            if src in ephemeris.source_dictionary
            else ephemeris.skyfield_wrapper.ephemeris[src]
            for src in sources
        ]

        # Distribute over frequencies
        data.redistribute("freq")
        model.redistribute("freq")

        # Determine local dimensions
        nfreq, nstack, ntime = data.vis.local_shape

        # Find the local frequencies
        sfreq = data.vis.local_offset[0]
        efreq = sfreq + nfreq

        freq = data.freq[sfreq:efreq]

        # Calculate time
        bmatch = {}
        if "ra" in data.index_map:

            for key, val in beams.items():
                imatch = np.searchsorted(data.ra, val.ra)
                if np.array_equal(data.ra[imatch], val.ra):
                    bmatch[key] = imatch
                else:
                    raise RuntimeError(
                        f"The beam for {key} is not sampled at the correct RAs."
                    )

            if "timestamp" in model.datasets:
                timestamp = model.datasets["timestamp"][:]
                flag_lsd = model.datasets["flag_lsd"][:]
            else:
                timestamp = model.index_map["time"][:].reshape(1, -1)
                flag_lsd = None

        elif "time" in data.index_map:

            for key, val in beams.items():
                imatch = np.searchsorted(data.time, val.time)
                if np.array_equal(data.time[imatch], val.time):
                    bmatch[key] = imatch
                else:
                    raise RuntimeError(
                        f"The beam for {key} is not sampled at the correct RAs."
                    )

            timestamp = data.time.reshape(1, -1)
            flag_lsd = None

        else:
            raise RuntimeError("Unable to extract time from input container.")

        # Redefine stack axis so that it only contains chime antennas
        stack_new, stack_flag = tools.redefine_stack_index_map(
            self.inputmap,
            data.index_map["prod"],
            data.index_map["stack"],
            data.reverse_map["stack"],
        )

        prod_new = data.index_map["prod"][stack_new["prod"]]

        # Swap the product pair order for conjugated stack indices
        cj = np.flatnonzero(stack_new["conjugate"].astype(bool))
        if cj.size > 0:
            prod_new["input_a"][cj], prod_new["input_b"][cj] = (
                prod_new["input_b"][cj],
                prod_new["input_a"][cj],
            )

        # Calculate baseline distances
        tools.change_chime_location(rotation=telescope_rotation)
        feedpos = tools.get_feed_positions(self.inputmap).T
        distance = feedpos[:, prod_new["input_a"]] - feedpos[:, prod_new["input_b"]]
        self.log.info("Rotation set to %0.4f deg" % self.inputmap[0]._rotation)
        tools.change_chime_location(default=True)

        # Calculate polarisation products, determine unique values
        feedpol = tools.get_feed_polarisations(self.inputmap)
        upol = model.index_map["pol"]
        if "co" in upol:
            # Jointly fit all polarisation products
            apol, bpol = feedpol[prod_new["input_a"]], feedpol[prod_new["input_b"]]
            pol_index = [np.flatnonzero(apol == bpol), np.flatnonzero(apol != bpol)]
            pol_conj = [np.flatnonzero(apol[pind] > bpol[pind]) for pind in pol_index]

        else:
            # Calculate polarisation products, determine unique values
            pol = np.core.defchararray.add(
                feedpol[prod_new["input_a"]], feedpol[prod_new["input_b"]]
            )
            pol_index = [np.flatnonzero(pol == upp) for upp in upol]
            pol_conj = [np.array([]) for upp in upol]

        # If requested, create output container.
        if self.save_model:
            out = dcontainers.empty_like(data)
            out.redistribute("freq")
            out.vis[:] = 0.0
            out.weight[:] = data.weight[:].copy()
        else:
            out = data

        ovis = out.vis[:].view(np.ndarray)

        # Dereference dataset
        try:
            coeff = model.coeff[:].view(np.ndarray)
        except KeyError:
            coeff = model.amplitude[:].view(np.ndarray)
        else:
            coeff = coeff[:, :, np.newaxis, :]

        bvis, bweight = {}, {}
        for key, val in beams.items():
            val.redistribute("freq")
            bvis[key] = val.vis[:].view(np.ndarray)
            bweight[key] = val.weight[:].view(np.ndarray)

        # Subtract source model
        for pp, this_pol in enumerate(pol_index):

            dist_pol = distance[:, this_pol]

            # Loop over frequencies
            for ff, nu in enumerate(freq):

                # Only process times where we have a beam
                flag_beam = np.zeros((nsources, ntime), dtype=bool)
                for ss, src in enumerate(sources):
                    flag_beam[ss, bmatch[src]] = np.any(
                        bweight[src][ff] > 0.0, axis=(0, 1)
                    )

                valid_time = np.flatnonzero(np.any(flag_beam, axis=0))
                self.log.info(
                    f"There are {valid_time.size} valid times for frequency {ff}."
                )
                if valid_time.size == 0:
                    continue

                flg = flag_lsd[:, valid_time] if flag_lsd is not None else None

                # Calculate source model
                source_model, sedge = model_extended_sources(
                    nu, dist_pol, timestamp[:, valid_time], bodies, flag_lsd=flg, **source_model_kwargs
                )
                source_model = source_model[0]

                # Multipy source model by the effective beam
                for ss, src in enumerate(sources):

                    valid_in = np.flatnonzero(flag_beam[ss, bmatch[src]])
                    valid_out = np.flatnonzero(flag_beam[ss, valid_time])

                    for st in range(nstream):

                        aa, bb = sedge[ss * nstream + st], sedge[ss * nstream + st + 1]

                        this_flag = (
                            bweight[src][ff, :, st, :][this_pol][:, valid_in] > 0.0
                        ).astype(np.float32)
                        this_beam = (
                            bvis[src][ff, :, st, :][this_pol][:, valid_in] * this_flag
                        )
                        source_model[..., valid_out, aa:bb] *= this_beam[
                            ..., np.newaxis
                        ]

                to_conj = pol_conj[pp]
                if to_conj.size > 0:
                    source_model[to_conj] = source_model[to_conj].conj()

                mdl = np.sum(
                    coeff[ff, pp, np.newaxis, :, :] * source_model,
                    axis=-1,
                )

                if to_conj.size > 0:
                    mdl[to_conj] = mdl[to_conj].conj()

                if self.save_model:
                    for ii, vt in enumerate(valid_time):
                        ovis[ff, this_pol, vt] = mdl[..., ii]
                else:
                    for ii, vt in enumerate(valid_time):
                        ovis[ff, this_pol, vt] -= mdl[..., ii]

        return out


def kz_coeffs(m, k):
    """Compute the coefficients for a Kolmogorov-Zurbenko (KZ) filter.

    Parameters
    ----------
    m : int
        Size of the moving average window.
    k : int
        Number of iterations.

    Returns
    -------
    coeff : np.ndarray
        Array of size k * (m - 1) + 1 containing the filter coefficients.
    """

    # Coefficients at degree one
    coef = np.ones(m, dtype=np.float64)

    # Iterate k-1 times over coefficients
    for i in range(1, k):

        t = np.zeros((m, m + i * (m - 1)))
        for km in range(m):
            t[km, km : km + coef.size] = coef

        coef = np.sum(t, axis=0)

    assert coef.size == k * (m - 1) + 1

    return coef / m**k


def apply_kz_lpf_2d(y, flag, window=3, niter=8, mode="wrap", frac_required=0.80):
    """Apply a 2D Kolmogorov-Zurbenko (KZ) filter.

    The KZ filter is an FIR filter that is equivalent to repeated application
    of a moving average.  The "window" parameter is the size of the moving
    average window and determines the cut off for the low-pass filter.  The
    "niter" parameter is the number of times that the moving average is applied
    and controls the peak-to-sidelobe ratio of the transfer function of the filter.
    This method pre-computes the coefficients for a given "window" and "niter",
    and then convolves once with the data.

    Parameters
    ----------
    y : np.ndarray
        The data to filter.  Must be two dimensional.
    flag : np.ndarray
        Boolean array with the same shape as y where True
        indicates valid data and False indicates invalid data
    window : int or list of int
        The size of the moving average window.  This can either be
        a 2 element list, in which case a different window size
        will be used for each dimension, or a single number, in which
        case the same value will be used for both dimensions.
    niter : int or list of int
        Number of iterations of the moving average filter.  This can
        either be a 2 element list, in which case a  different number of
        iterations will be used for each dimension, or a single number,
        in which case the same value will be used for both dimensions.
    mode : str or list of str
        The method used to pad the edges of the array.  This can
        either be a 2 element list, in which case a  different method
        will be used for each dimension, or a single string, in which
        case the same method will be used for both dimensions.
    frac_required : float
        The fraction of samples within a window that must be valid in
        order for the filtered data point to be considered valid.

    Returns
    -------
    y_lpf : np.ndarray
        The low-pass filtered data.  The value of the array is set to zero
        if the data is determined invalid based on frac_required argument.
    """

    # Parse inputs
    if np.isscalar(window):
        window = [window] * 2

    if np.isscalar(niter):
        niter = [niter] * 2

    window = [w + (not (w % 2)) for w in window]
    total = [k * (w - 1) + 1 for (w, k) in zip(window, niter)]
    hwidth = [tt // 2 for tt in total]

    pad_width = tuple([(hw, hw) for hw in hwidth])

    # Get filter coefficients and construct the 2D kernel
    coeff = [kz_coeffs(w, k) for (w, k) in zip(window, niter)]
    kernel = np.outer(coeff[0], coeff[1])

    # Pad the array using the requested method
    y = np.where(flag, y, 0.0)

    if np.isscalar(mode):

        y_extended = np.pad(y, pad_width, mode=mode)
        flag_extended = np.pad(flag.astype(np.float64), pad_width, mode=mode)

    else:
        y_extended = y
        flag_extended = flag.astype(np.float64)

        for dd, (pw, md) in enumerate(zip(pad_width, mode)):

            pws = tuple([pw if ii == dd else (0, 0) for ii in range(2)])

            y_extended = np.pad(y_extended, pws, mode=md)
            flag_extended = np.pad(flag_extended, pws, mode=md)

    # Filter the array
    y_lpf = scipy.signal.convolve(y_extended, kernel, mode="valid")

    # Filter the flags
    flag_lpf = scipy.signal.convolve(flag_extended, kernel, mode="valid")
    flag_lpf = flag_lpf * (flag_lpf >= frac_required)

    # Renormalize the low pass filtered data and return
    y_lpf *= tools.invert_no_zero(flag_lpf)

    return y_lpf
