"""Tasks for removing bright sources from the data.

Tasks for constructing models for bright sources and subtracting them from the data.
"""

import json
import numpy as np

from scipy.constants import c as speed_of_light
import scipy.signal

from caput import config
import caput.time as ctime
from ch_ephem.observers import chime
from ch_ephem.sources import source_dictionary
from ch_util import andata, tools
from ch_util.fluxcat import FluxCatalog
from draco.core import task, io

from ..core import containers


def _correct_phase_wrap(phi):
    return ((phi + np.pi) % (2.0 * np.pi)) - np.pi


def model_extended_sources(
    freq,
    distance,
    timestamp,
    bodies,
    min_altitude=5.0,
    min_ha=0.0,
    max_ha=0.0,
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
    timestamp : np.ndarray[ntime,]
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
    nsource = len(bodies)

    # Parse input parameters
    scale_x = np.radians(np.array(kwargs["scale_x"]) / 60.0)
    scale_y = np.radians(np.array(kwargs["scale_y"]) / 60.0)
    scale_t = np.array(kwargs.get("scale_t", [0.66] * nsource))

    degree_x = np.array(kwargs["degree_x"])
    degree_y = np.array(kwargs["degree_y"])
    degree_t = np.array(kwargs.get("degree_t", [0] * nsource))

    scale = np.vstack((scale_x, scale_y, scale_t)).T
    poly_deg = np.vstack((degree_x, degree_y, degree_t)).T

    freq = np.atleast_1d(freq)
    timestamp = np.atleast_1d(timestamp)

    nfreq = freq.size
    ntime = timestamp.size
    nbaseline = distance.shape[-1]

    ones = np.ones((nfreq, nbaseline, ntime), dtype=np.float64)

    # Calculate baseline distances in wavelengths
    lmbda = speed_of_light * 1e-6 / freq
    u, v = (
        distance[:, np.newaxis, :, np.newaxis]
        / lmbda[np.newaxis, :, np.newaxis, np.newaxis]
    )

    # Setup for calculating source coordinates
    lat = np.radians(chime.latitude)
    date = ctime.unix_to_skyfield_time(timestamp)

    observer = chime.skyfield_obs().at(date)

    # Generate polynomials
    ncoeff_x = degree_x + 1
    ncoeff_y = degree_y + 1
    ncoeff_t = degree_t + 1
    ncoeff = ncoeff_x * ncoeff_y * ncoeff_t
    nparam = np.sum(ncoeff)

    source_bound = np.concatenate(([0], np.cumsum(ncoeff)))

    S = np.zeros((nfreq, nbaseline, ntime, nparam), dtype=np.complex64)

    for ss, body in enumerate(bodies):
        # Calculate the source coordinates
        obs = observer.observe(body).apparent()
        src_radec = obs.cirs_radec(date)
        src_altaz = obs.altaz()

        src_ra, src_dec = src_radec[0], src_radec[1]
        src_alt, src_az = src_altaz[0], src_altaz[1]

        ha = _correct_phase_wrap(
            np.radians(chime.unix_to_lsa(timestamp)) - src_ra.radians
        )
        dec = src_dec.radians

        weight = src_alt.radians > np.radians(min_altitude)

        if max_ha > 0.0:
            weight &= np.abs(ha) <= (np.radians(max_ha) / np.cos(dec))

        if min_ha > 0.0:
            weight &= np.abs(ha) >= (np.radians(min_ha) / np.cos(dec))

        # Evaluate polynomial
        aa, bb = source_bound[ss], source_bound[ss + 1]

        coords = [ax * scale[ss, ii] * ones for ii, ax in enumerate([u, v, ha])]

        H = np.polynomial.hermite.hermvander3d(*coords, poly_deg[ss])

        # Calculate the fringestop phase
        phi = tools.fringestop_phase(
            ha[np.newaxis, np.newaxis, :], lat, dec[np.newaxis, np.newaxis, :], u, v
        ).conj()

        S[..., aa:bb] = (
            H * phi[..., np.newaxis] * weight[np.newaxis, np.newaxis, :, np.newaxis]
        )

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

    for tt in range(ntime):
        S = source_model[:, tt, :]

        # Calculate covariance of model coefficients
        C = np.dot(S.T.conj(), weight[:, tt, np.newaxis] * S)

        # Solve for model coefficients
        coeff[tt, :] = np.linalg.lstsq(
            C, np.dot(S.T.conj(), weight[:, tt] * vis[:, tt]), rcond=None
        )[0]

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

    weight = weight.flatten()
    vis = vis.flatten()

    S = source_model.reshape(-1, nparam)

    # Calculate covariance of model coefficients
    C = np.dot(S.T.conj(), weight[:, np.newaxis] * S)

    # Solve for model coefficients
    coeff = np.linalg.lstsq(C, np.dot(S.T.conj(), weight * vis), rcond=None)[0]

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

    min_altitude = config.Property(proptype=float, default=5.0)
    max_ha = config.Property(proptype=float, default=0.0)
    min_ha = config.Property(proptype=float, default=0.0)

    min_distance = config.Property(proptype=list, default=[10.0, 0.0])
    telescope_rotation = config.Property(proptype=float, default=chime.rotation)
    max_iter = config.Property(proptype=int, default=4)

    def setup(self, tel):
        """Set up the source model.

        Parameters
        ----------
        tel : analysis.telescope.CHIMETelescope
        """
        telescope = io.get_telescope(tel)
        self.inputmap = telescope.feeds

        self.bodies = [
            (
                source_dictionary[src]
                if src in source_dictionary
                else ctime.skyfield_wrapper.ephemeris[src]
            )
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
            "min_altitude": self.min_altitude,
            "max_ha": self.max_ha,
            "min_ha": self.min_ha,
        }

        if any(self.degree_x) or any(self.degree_y):
            self.extended_source_kwargs = {
                "degree_x": self.degree_x,
                "degree_y": self.degree_y,
                "degree_t": self.degree_t,
                "scale_x": self.extent,
                "scale_y": self.extent,
                "scale_t": self.scale_t,
                "min_altitude": self.min_altitude,
                "max_ha": self.max_ha,
                "min_ha": self.min_ha,
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
            csd = data.attrs["lsd"] if "lsd" in data.attrs else data.attrs["csd"]
            csd = np.fix(np.mean(csd))
            timestamp = chime.lsd_to_unix(csd + data.ra / 360.0)
            output_kwargs = {"time": timestamp}
        elif "time" in data.index_map:
            timestamp = data.time
            output_kwargs = {}
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
                param_name.append((src, ii))
        param_name = np.array(param_name, dtype=[("source", "U32"), ("coeff", "u2")])

        # Create output container
        out = containers.SourceModel(
            pol=upol,
            source=np.array(self.sources),
            param=param_name,
            axes_from=data,
            attrs_from=data,
            **output_kwargs,
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
        if self.extended_source_kwargs:
            out_coeff = out.coeff[:].view(np.ndarray)

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
                    nu, dist_pol, timestamp, self.bodies, **self.point_source_kwargs
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
            (
                source_dictionary[src]
                if src in source_dictionary
                else ctime.skyfield_wrapper.ephemeris[src]
            )
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
            csd = data.attrs["lsd"] if "lsd" in data.attrs else data.attrs["csd"]
            csd = np.fix(np.mean(csd))
            timestamp = chime.lsd_to_unix(csd + data.ra / 360.0)
        elif "time" in data.index_map:
            timestamp = data.time
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
                    nu, dist_pol, timestamp, bodies, **source_model_kwargs
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

    def setup(self, tel):
        """Set up the source model.

        Parameters
        ----------
        tel : analysis.telescope.CHIMETelescope
        """
        telescope = io.get_telescope(tel)
        self.inputmap = telescope.feeds

        self.bodies = [
            (
                source_dictionary[src]
                if src in source_dictionary
                else ctime.skyfield_wrapper.ephemeris[src]
            )
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
            "min_altitude": self.min_altitude,
            "max_ha": self.max_ha,
            "min_ha": self.min_ha,
        }

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

        # Find the local frequencies
        sfreq = data.vis.local_offset[0]
        efreq = sfreq + nfreq

        freq = data.freq[sfreq:efreq]

        # Calculate time
        if "ra" in data.index_map:
            csd = data.attrs["lsd"] if "lsd" in data.attrs else data.attrs["csd"]
            csd = np.fix(np.mean(csd))
            timestamp = chime.lsd_to_unix(csd + data.ra / 360.0)
            output_kwargs = {"time": timestamp}
        elif "time" in data.index_map:
            timestamp = data.time
            output_kwargs = {}
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
                param_name.append((src, ii))
        param_name = np.array(param_name, dtype=[("source", "U32"), ("coeff", "u2")])

        # Create output container
        out = containers.SourceModel(
            pol=upol,
            source=np.array(self.sources),
            param=param_name,
            axes_from=data,
            attrs_from=data,
            **output_kwargs,
        )

        out.add_dataset("coeff")

        out.attrs["source_model_kwargs"] = json.dumps(self.source_kwargs)

        self.log.info("Beams contain following sources: %s" % str(list(beams.keys())))

        # Dereference datasets
        all_vis = data.vis[:].view(np.ndarray)
        all_weight = data.weight[:].view(np.ndarray)

        out.redistribute("freq")
        out_coeff = out.coeff[:].view(np.ndarray)

        # Loop over polarisations
        for pp, upp in enumerate(upol):
            this_pol = np.flatnonzero(pol == upp)

            dist_pol = distance[:, this_pol]
            bweight_pol = baseline_weight[this_pol, np.newaxis]

            # Loop over frequencies
            for ff, nu in enumerate(freq):
                # Determine extended source model
                source_model, sedge = model_extended_sources(
                    nu, dist_pol, timestamp, self.bodies, **self.source_kwargs
                )
                source_model = source_model[0]

                # Multipy source model by the effective beam
                for ss, src in enumerate(self.sources):
                    this_beam = beams[src].vis[ff][this_pol].view(np.ndarray) * (
                        beams[src].weight[ff][this_pol].view(np.ndarray) > 0.0
                    ).astype(np.float32)
                    source_model[..., sedge[ss] : sedge[ss + 1]] *= this_beam[
                        ..., np.newaxis
                    ]

                # Extract datasets for this polarisation and frequency
                vis = all_vis[ff, this_pol, :]
                weight = all_weight[ff, this_pol, :] * bweight_pol

                out_coeff[ff, pp, :] = solve_multiple_times(vis, weight, source_model)

        # Save a few attributes necessary to interpret the data
        out.attrs["min_distance"] = min_distance
        out.attrs["telescope_rotation"] = self.telescope_rotation

        return out


class SubtractSourcesWithBeam(task.SingleTask):
    """Subtract a source model from the visibilities."""

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
        telescope_rotation = model.attrs["telescope_rotation"]
        source_model_kwargs = json.loads(model.attrs["source_model_kwargs"])

        bodies = [
            (
                source_dictionary[src]
                if src in source_dictionary
                else ctime.skyfield_wrapper.ephemeris[src]
            )
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
            csd = data.attrs["lsd"] if "lsd" in data.attrs else data.attrs["csd"]
            csd = np.fix(np.mean(csd))
            timestamp = chime.lsd_to_unix(csd + data.ra / 360.0)
        elif "time" in data.index_map:
            timestamp = data.time
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
        pol = np.array(
            [feedpol[pn["input_a"]] + feedpol[pn["input_b"]] for pn in prod_new]
        )

        # Dereference dataset
        vis = data.vis[:].view(np.ndarray)
        coeff = model.coeff[:].view(np.ndarray)

        # Subtract source model
        for pp, upp in enumerate(model.index_map["pol"]):
            this_pol = np.flatnonzero(pol == upp)
            dist_pol = distance[:, this_pol]

            # Loop over frequencies
            for ff, nu in enumerate(freq):
                # Calculate source model
                source_model, sedge = model_extended_sources(
                    nu, dist_pol, timestamp, bodies, **source_model_kwargs
                )
                source_model = source_model[0]

                # Multipy source model by the effective beam
                for ss, src in enumerate(sources):
                    this_beam = beams[src].vis[ff][this_pol].view(np.ndarray) * (
                        beams[src].weight[ff][this_pol].view(np.ndarray) > 0.0
                    ).astype(np.float32)
                    source_model[..., sedge[ss] : sedge[ss + 1]] *= this_beam[
                        ..., np.newaxis
                    ]

                mdl = np.sum(
                    coeff[ff, pp, np.newaxis, np.newaxis, :] * source_model,
                    axis=-1,
                )

                vis[ff, this_pol, :] -= mdl

        return data


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
