"""Tasks for removing bright sources from the data.

.. currentmodule:: ch_pipeline.analysis.source_removal

Tasks for constructing models for bright sources and subtracting them from the data.

Tasks
=====

.. autosummary::
    :toctree: generated/

    SolveSources
    SubtractSources

"""

import json
import numpy as np

from scipy.constants import c as speed_of_light

from caput import config
from ch_util import andata, ephemeris, tools
from draco.core import task, io

from ..core import containers


def model_extended_sources(
    freq, distance, timestamp, bodies, min_altitude=10.0, **kwargs
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
        Do not include a source in the model if it has an altitude less than this value.
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

    # Calculate baseline distances in wavelengths
    lmbda = speed_of_light * 1e-6 / freq
    u, v = (
        distance[:, np.newaxis, :, np.newaxis]
        / lmbda[np.newaxis, :, np.newaxis, np.newaxis]
    )

    # Setup for calculating source coordinates
    lat = np.radians(ephemeris.CHIMELATITUDE)
    observer = ephemeris._get_chime()
    observer.date = timestamp

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
        src_ra, src_dec = observer.cirs_radec(body)
        src_alt, src_az = observer.altaz(body)

        ha = np.radians(ephemeris.lsa(timestamp)) - src_ra.radians
        dec = src_dec.radians
        weight = src_alt.radians > np.radians(min_altitude)

        # Evaluate polynomial
        aa, bb = source_bound[ss], source_bound[ss + 1]

        H = np.polynomial.hermite.hermvander3d(
            u * scale[ss, 0], v * scale[ss, 1], ha * scale[ss, 2], poly_deg[ss]
        )

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
        Do not include a source in the model if its altitude is less than this value.
    min_distance : float
        Do not include baselines in the fit with a total distance less than this value.
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

    min_altitude = config.Property(proptype=float, default=10.0)
    min_distance = config.Property(proptype=float, default=20.0)
    telescope_rotation = config.Property(proptype=float, default=tools._CHIME_ROT)
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
            "min_altitude": self.min_altitude,
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
            timestamp = ephemeris.csd_to_unix(csd + data.ra / 360.0)
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
        cj = np.flatnonzero(stack_new["conjugate"].astype(np.bool))
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
        sep = np.sqrt(np.sum(distance ** 2, axis=0))
        baseline_weight = (sep > self.min_distance).astype(np.float32)

        # Calculate polarisation products, determine unique values
        feedpol = tools.get_feed_polarisations(self.inputmap)
        pol = np.array(
            [feedpol[pn["input_a"]] + feedpol[pn["input_b"]] for pn in prod_new]
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
                param_name.append(
                    "%s_3d_hermite_polynomial_coefficient_%02d" % (src.lower(), ii)
                )
        param_name = np.array(param_name)

        # Create output container
        out = containers.SourceModel(
            pol=upol,
            time=timestamp,
            source=np.array(self.sources),
            param=param_name,
            axes_from=data,
            attrs_from=data,
        )

        # Determine the initial source model, assuming all sources are point sources
        source_model, _ = model_extended_sources(
            freq, distance, timestamp, self.bodies, **self.point_source_kwargs
        )

        # Determine extended source model
        if self.extended_source_kwargs:
            ext_source_model, sedge = model_extended_sources(
                freq, distance, timestamp, self.bodies, **self.extended_source_kwargs
            )

            out.add_dataset("amplitude")
            out.add_dataset("coeff")

            out.add_dataset("source_index")
            for ss in range(self.nsources):
                out.source_index[sedge[ss] : sedge[ss + 1]] = ss

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

            # Loop over frequencies
            for ff in range(nfreq):

                # Extract datasets for this polarisation and frequency
                vis = all_vis[ff, this_pol, :]
                weight = (
                    all_weight[ff, this_pol, :] * baseline_weight[this_pol, np.newaxis]
                )
                psrc_model = source_model[ff, this_pol, :, :]

                # Obtain initial estimate of each source assuming point source
                amplitude = solve_single_time(vis, weight, psrc_model)

                # If modeling extended sources, iterate over time-dependent normalization
                # and baseline dependent response.  Assumes the description of the extended
                # emission is constant in time.
                if self.extended_source_kwargs:

                    ext_model = ext_source_model[ff, this_pol, :, :]

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
            csd = data.attrs["lsd"] if "lsd" in data.attrs else data.attrs["csd"]
            csd = np.fix(np.mean(csd))
            timestamp = ephemeris.csd_to_unix(csd + data.ra / 360.0)
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
        cj = np.flatnonzero(stack_new["conjugate"].astype(np.bool))
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

        # Calculate source model
        source_model, sedge = model_extended_sources(
            freq, distance, timestamp, bodies, **source_model_kwargs
        )

        # Dereference dataset
        vis = data.vis[:].view(np.ndarray)
        amp = model.amplitude[:].view(np.ndarray)

        coeff = model.coeff[:].view(np.ndarray) if "coeff" in model.datasets else None

        # Subtract source model
        for pp, upp in enumerate(model.index_map["pol"]):

            this_pol = np.flatnonzero(pol == upp)

            if coeff is not None:
                mdl = np.sum(
                    amp[:, pp, np.newaxis, :, :][..., model.source_index]
                    * coeff[:, pp, np.newaxis, np.newaxis, :]
                    * source_model[:, this_pol, :, :],
                    axis=-1,
                )
            else:
                mdl = np.sum(
                    amp[:, pp, np.newaxis, :, :] * source_model[:, this_pol, :, :],
                    axis=-1,
                )

            vis[:, this_pol, :] -= mdl

        return data


class AccumulateBeam(task.SingleTask):
    """Accumulate the stacked beam for each source."""

    def setup(self):

        self.beam_stack = {}

    def process(self, beam_stack):

        self.beam_stack[beam_stack.attrs["source_name"]] = beam_stack

        return None

    def process_finish(self):

        return self.beam_stack


class SolveSourcesWithBeam(SolveSources):
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
            "min_altitude": self.min_altitude,
        }

    def process(self, data, beams):
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
            timestamp = ephemeris.csd_to_unix(csd + data.ra / 360.0)
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
        cj = np.flatnonzero(stack_new["conjugate"].astype(np.bool))
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
        sep = np.sqrt(np.sum(distance ** 2, axis=0))
        baseline_weight = (sep > self.min_distance).astype(np.float32)

        # Calculate polarisation products, determine unique values
        feedpol = tools.get_feed_polarisations(self.inputmap)
        pol = np.array(
            [feedpol[pn["input_a"]] + feedpol[pn["input_b"]] for pn in prod_new]
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
                param_name.append(
                    "%s_3d_hermite_polynomial_coefficient_%02d" % (src.lower(), ii)
                )
        param_name = np.array(param_name)

        # Create output container
        out = containers.SourceModel(
            pol=upol,
            time=timestamp,
            source=np.array(self.sources),
            param=param_name,
            axes_from=data,
            attrs_from=data,
        )

        # Determine extended source model
        source_model, sedge = model_extended_sources(
            freq, distance, timestamp, self.bodies, **self.source_kwargs
        )

        out.add_dataset("coeff")

        out.add_dataset("source_index")
        for ss in range(self.nsources):
            out.source_index[sedge[ss] : sedge[ss + 1]] = ss

        out.attrs["source_model_kwargs"] = json.dumps(self.source_kwargs)

        # Multipy source model by the effective beam
        for ss, src in enumerate(self.sources):
            this_beam = beams[src].vis[:].view(np.ndarray) * (
                beams[src].weight[:].view(np.ndarray) > 0.0
            ).astype(np.float32)
            source_model[..., sedge[ss] : sedge[ss + 1]] *= this_beam[..., np.newaxis]

        # Dereference datasets
        all_vis = data.vis[:].view(np.ndarray)
        all_weight = data.weight[:].view(np.ndarray)

        out.redistribute("freq")
        out_coeff = out.coeff[:].view(np.ndarray)

        # Loop over polarisations
        for pp, upp in enumerate(upol):

            this_pol = np.flatnonzero(pol == upp)

            # Loop over frequencies
            for ff in range(nfreq):

                # Extract datasets for this polarisation and frequency
                vis = all_vis[ff, this_pol, :]
                weight = (
                    all_weight[ff, this_pol, :] * baseline_weight[this_pol, np.newaxis]
                )
                model = source_model[ff, this_pol, :, :]

                out_coeff[ff, pp, :] = solve_multiple_times(vis, weight, model)

        # Save a few attributes necessary to interpret the data
        out.attrs["min_distance"] = self.min_distance
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
            csd = data.attrs["lsd"] if "lsd" in data.attrs else data.attrs["csd"]
            csd = np.fix(np.mean(csd))
            timestamp = ephemeris.csd_to_unix(csd + data.ra / 360.0)
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
        cj = np.flatnonzero(stack_new["conjugate"].astype(np.bool))
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

        # Calculate source model
        source_model, sedge = model_extended_sources(
            freq, distance, timestamp, bodies, **source_model_kwargs
        )

        # Multipy source model by the effective beam
        for ss, src in enumerate(sources):
            this_beam = beams[src].vis[:].view(np.ndarray) * (
                beams[src].weight[:].view(np.ndarray) > 0.0
            ).astype(np.float32)
            source_model[..., sedge[ss] : sedge[ss + 1]] *= this_beam[..., np.newaxis]

        # Dereference dataset
        vis = data.vis[:].view(np.ndarray)
        coeff = model.coeff[:].view(np.ndarray)

        # Subtract source model
        for pp, upp in enumerate(model.index_map["pol"]):

            this_pol = np.flatnonzero(pol == upp)

            mdl = np.sum(
                coeff[:, pp, np.newaxis, np.newaxis, :]
                * source_model[:, this_pol, :, :],
                axis=-1,
            )

            vis[:, this_pol, :] -= mdl

        return data
