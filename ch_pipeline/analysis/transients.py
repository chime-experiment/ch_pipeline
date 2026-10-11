"""Tasks for the search for long period transients in hybrid beamformed data."""

import inspect
import json
import time

import numpy as np
import scipy.constants
import scipy.interpolate
from caput import config, mpiarray
from caput.astro.coordinates import spherical
from caput.pipeline import tasklib
from draco.analysis.beamform import BeamFormExternalMixin
from draco.analysis.ringmapmaker import find_grid_indices
from draco.core import containers as draco_containers
from draco.core import io
from draco.util import interferometry, tools
from scipy.linalg import cho_factor, cho_solve

from ch_pipeline.core import containers

MMODE_FILTER_CONFIG_ATTR = "convolution_m_filter"


class FitTemplate(tasklib.base.ContainerTask):
    """Fit a template to hybrid beamformed visibilities and subtract it.

    The model ``a * template + b`` is fit by weighted least squares in blocks of
    ``window`` samples along RA (or a rolling window), separately for each
    polarisation, frequency, baseline and elevation.

    Attributes
    ----------
    eps : float
        Regularisation added to the least-squares fit.  Default is 0.
    window : int
        Number of RA samples in each fit.  Default is the full axis.
    rolling : bool
        Fit in a rolling window centred on each sample.  Default is False.
    wrap : bool
        Wrap the rolling window around the ends of the RA axis.  Default is False.
    offset_only : bool
        Only subtract the offset ``b``.  Default is False.
    in_place : bool
        Modify the input container.  Default is False.
    """

    eps = config.Property(proptype=float, default=0.0)
    window = config.Property(proptype=int, default=0)
    rolling = config.Property(proptype=bool, default=False)
    wrap = config.Property(proptype=bool, default=False)
    offset_only = config.Property(proptype=bool, default=False)
    in_place = config.Property(proptype=bool, default=False)

    def process(self, data, template):
        """Subtract the template from the data.

        Parameters
        ----------
        data, template : draco.core.containers.HybridVisStream
            Visibilities to fit, and the template to fit to them.

        Returns
        -------
        out : draco.core.containers.HybridVisStream
            Residual visibilities.
        """
        data.redistribute("freq")
        template.redistribute("freq")

        self.log.info(
            f"Fitting template from lsd {template.attrs['lsd']} "
            f"to data from lsd {data.attrs['lsd']}."
        )

        # Dereference the required datasets
        dvis = data.vis[:].local_array
        dweight = data.weight[:].local_array[..., np.newaxis, :]

        tvis = template.vis[:].local_array
        tweight = template.weight[:].local_array[..., np.newaxis, :]

        shp = tvis.shape

        flag = (dweight > 0.0) & (tweight > 0.0)
        weight = np.ones(shp, dtype=float) * dweight * flag

        # Determine how many data points to average
        window = (
            self.window if (self.window > 0) and (self.window < shp[-1]) else shp[-1]
        )

        if self.rolling:
            nroll = window + int(not (window % 2))
            window = shp[-1]
            nsel = shp[-1]

        else:
            nroll = 0
            nblock = shp[-1] // window
            nsel = window * nblock

        # Reshape
        y = dvis[..., :nsel].reshape(-1, window)
        t = tvis[..., :nsel].reshape(-1, window)
        w = weight[..., :nsel].reshape(-1, window)

        # Perform fit
        a, b = fit_amp_offset(
            y, t, w, eps=self.eps, nroll=nroll, wrap=self.wrap, with_cov=False
        )

        # If we did not perform a rolling estimate, then we need to expand
        # dimensions of fit parameters so they can broadcast against the data.
        if not self.rolling:
            a = a[:, np.newaxis]
            b = b[:, np.newaxis]

        # Subtract model from data
        if self.offset_only:
            rvis = y - b
            rweight = dweight

        else:
            rvis = y - (a * t + b)
            rweight = (
                tools.invert_no_zero(
                    tools.invert_no_zero(dweight) + tools.invert_no_zero(tweight)
                )
                * flag
            )

        # Create output container
        if self.in_place:
            out = data
        else:
            out = data.copy()
            out.redistribute("freq")

        out.vis[:].local_array[..., :nsel] = rvis.reshape(shp)

        out.weight[:].local_array[..., :nsel] = rweight[..., 0, :nsel]
        out.weight[:].local_array[..., nsel:] = 0.0

        return out


class FitYesterday(FitTemplate):
    """Fit and subtract the previous input of the pipeline (e.g. the previous day)."""

    def setup(self):
        """Initialise with no template."""
        self.template = None

    def process(self, data):
        """Subtract the previous input, which is then replaced by this one.

        Returns nothing for the first input.
        """
        template = self.template
        self.template = data

        if template is not None:
            return super().process(data, template)

        return None


class FitStaticTemplate(FitTemplate):
    """Fit and subtract a fixed template (e.g. a stack) from each input."""

    def setup(self, template):
        """Set the template.

        Parameters
        ----------
        template : draco.core.containers.HybridVisStream
            Template to fit to each input.
        """
        self.template = template

    def process(self, data):
        """Subtract the template from the data."""
        return super().process(data, self.template)


class MaskEWBaselines(tasklib.base.ContainerTask):
    """Set the weights of selected east-west baselines to zero.

    Attributes
    ----------
    index : list of int
        Indices of the east-west baselines to mask.  Default is [0].
    in_place : bool
        Modify the input container.  Default is True.
    """

    index = config.list_type(int, default=[0])
    in_place = config.Property(proptype=bool, default=True)

    def process(self, data):
        """Mask the baselines."""
        out = data.copy() if not self.in_place else data

        weight = out.weight[:].local_array

        waxes = list(out.weight.attrs["axis"])
        iew = waxes.index("ew")

        for x in self.index:
            slc = (slice(None),) * iew + (x,)
            weight[slc] = 0.0

        return out


class DayenuMFilterHybridVis(tasklib.base.ContainerTask):
    """High-pass filter a HybridVisStream along RA with a DAYENU m-mode filter.

    For each frequency and east-west baseline, the m-modes of the sky, its
    north-south aliases, and the noise cross-talk are removed using the
    pseudo-inverse of the DAYENU covariance for the unmasked samples.

    Attributes
    ----------
    epsilon : float
        Stop-band rejection.  Default is 1e-12.
    num_width : float
        Half-width of the sky stop band in units of the half cylinder spacing.
        Default is 1.
    num_width_alias : float
        Half-width of the stop band for the aliased sky.  Zero (default) does not
        filter the aliased sky.
    m_cut_cross_talk : float
        Remove ``|m|`` below this to suppress noise cross-talk.  Default is 0.
    common_el : bool
        Use one stop band for all elevations.  Must be True.  Default is False.
    single_filter : bool
        Merge the stop bands into a single band.  Default is False.
    min_m_sep : float
        Merge stop bands separated by less than this.  Default is 0.
    atten_threshold : float
        Mask samples where the filter diagonal is below this fraction of its
        median.  Default is 0.
    abs_atten_threshold : float
        Mask samples where the filter diagonal is below this.  Default is 0.
    save_filter : bool
        Save the filter.  Default is False.
    calculate_cov : bool
        Save the covariance of the filtered visibilities.  Default is False.
    """

    epsilon = config.Property(proptype=float, default=1e-12)
    num_width = config.Property(proptype=float, default=1.0)
    num_width_alias = config.Property(proptype=float, default=0.0)
    m_cut_cross_talk = config.Property(proptype=float, default=0.0)

    common_el = config.Property(proptype=bool, default=False)
    single_filter = config.Property(proptype=bool, default=False)
    min_m_sep = config.Property(proptype=float, default=0.0)

    atten_threshold = config.Property(proptype=float, default=0.0)
    abs_atten_threshold = config.Property(proptype=float, default=0.0)

    save_filter = config.Property(proptype=bool, default=False)
    calculate_cov = config.Property(proptype=bool, default=False)

    def setup(self, manager):
        """Get the telescope geometry used to define the stop bands.

        Parameters
        ----------
        manager : draco.core.io.TelescopeConvertible
            Object holding the telescope.
        """
        self.telescope = io.get_telescope(manager)
        self.lat = np.radians(self.telescope.latitude)
        self.width = 0.5 * self.telescope.cylinder_spacing

        _xind, _yind, _min_xsep, min_ysep = find_grid_indices(self.telescope.baselines)
        self.min_ysep = min_ysep

    def process(self, stream):
        """Filter the stream in place.

        Parameters
        ----------
        stream : draco.core.containers.HybridVisStream
            Hybrid beamformed visibilities.

        Returns
        -------
        stream : draco.core.containers.HybridVisStream
            Filtered visibilities.
        """
        if not self.common_el:
            raise RuntimeError("El-dependent filter not currently supported.")

        is_fringestopped = stream.attrs.get("fringestopped", False)
        is_real = (
            is_fringestopped
            and (self.m_cut_cross_talk == 0.0)
            and (self.num_width_alias == 0.0)
        )

        # Create a filter dataset
        if self.save_filter:
            if not is_real:
                if "complex_ra_filter" not in stream.datasets:
                    stream.add_dataset("complex_ra_filter")
            else:
                if "ra_filter" not in stream.datasets:
                    stream.add_dataset("ra_filter")
            stream.ra_filter[:] = 0.0

        if self.calculate_cov:
            if not is_real:
                if "complex_ra_cov" not in stream.datasets:
                    stream.add_dataset("complex_ra_cov")
            else:
                if "ra_cov" not in stream.datasets:
                    stream.add_dataset("ra_cov")
            stream.ra_cov[:] = 0.0

        # Distribute over products
        stream.redistribute("freq")

        npol, _nfreq, _new, _nel, _nra = stream.vis.local_shape

        fslc = stream.vis[:].local_bounds

        # Get the required axes
        ra = np.radians(stream.ra[:])
        ew = stream.index_map["ew"]
        el = stream.index_map["el"]

        freq = stream.freq[fslc]

        # Dereference the required datasets
        vis = stream.vis[:].local_array
        weight = stream.weight[:].local_array
        if self.save_filter:
            filt = stream.ra_filter[:].local_array
        if self.calculate_cov:
            ra_cov = stream.ra_cov[:].local_array

        # Loop over products
        for ff, nu in enumerate(freq):

            for xx, bx in enumerate(ew):

                flag = np.all(weight[:, ff, xx] > 0.0, axis=0)

                if not np.any(flag):
                    weight[:, ff, xx] = 0.0
                    continue

                cuts = self._get_cut(nu, bx, el, is_fringestopped)

                for ee, mc, mw in zip(*cuts):

                    self.log.info(
                        f"{100 * np.sum(flag) / flag.size:0.2f} percent of data available."
                    )
                    for c, w in zip(mc, mw):
                        self.log.info(f"mc is {c:0.2f} and mw is {w:0.2f}")

                    # Construct the filter
                    t0 = time.time()
                    try:
                        NF = mmode_filter(
                            ra,
                            flag,
                            m_width=mw,
                            m_centre=mc,
                            epsilon=self.epsilon,
                        )

                    except np.linalg.LinAlgError as exc:
                        self.log.error(
                            "Failed to converge while generating filter "
                            f"for freq {ff} and ew {xx}: {exc}"
                        )
                        weight[:, ff, xx] = 0.0
                        continue

                    tvis = np.ascontiguousarray(vis[:, ff, xx, ee])
                    tvar = tools.invert_no_zero(weight[:, ff, xx])

                    vis[:, ff, xx, ee] = np.matmul(tvis, NF)
                    weight[:, ff, xx] = tools.invert_no_zero(
                        np.matmul(tvar, np.abs(NF) ** 2)
                    )

                    if self.save_filter:
                        filt[ff, xx] = NF

                    if self.calculate_cov:
                        for pp in range(npol):
                            ra_cov[pp, ff, xx] = np.matmul(
                                NF.T.conj(), tvar[pp, :, np.newaxis] * NF
                            )

                    if (self.atten_threshold > 0.0) or (self.abs_atten_threshold > 0.0):
                        diag = np.abs(np.diag(NF))
                        med_diag = np.median(diag[diag > 0.0])
                        th = np.maximum(
                            self.atten_threshold * med_diag, self.abs_atten_threshold
                        )

                        self.log.info(
                            f"Median diagonal is {med_diag:0.2f}, threshold is {th:0.3f}."
                        )

                        flag_low = diag > th

                        weight[:, ff, xx] *= flag_low.astype(weight.dtype)

                    self.log.info(
                        f"Took {time.time() - t0:0.2f} seconds to apply filter."
                    )
                    t0 = time.time()

        return stream

    def _get_cut(self, freq, xsep, el, is_fringestopped):
        return get_mmode_cuts(
            freq,
            xsep,
            el,
            is_fringestopped=is_fringestopped,
            lat=self.lat,
            width=self.width,
            min_ysep=self.min_ysep,
            num_width=self.num_width,
            num_width_alias=self.num_width_alias,
            m_cut_cross_talk=self.m_cut_cross_talk,
            common_el=self.common_el,
            single_filter=self.single_filter,
            min_m_sep=self.min_m_sep,
        )


class ConvolutionMFilterHybridVis(DayenuMFilterHybridVis):
    """High-pass a HybridVisStream using a fast convolution along right ascension.

    This task uses the same foreground m-mode cuts as
    :class:`DayenuMFilterHybridVis`, but applies them with FFT convolutions rather
    than constructing and inverting a covariance matrix.  A raised-cosine
    transition outside the stop band can be used to reduce ringing in right
    ascension.

    Zero-weight samples are always excluded using a Boolean validity mask.  The
    foreground estimate is calculated as

    ``LPF(valid * vis) / LPF(valid)``

    and subtracted from the visibility.  The input inverse-variance weights are
    used only to identify valid samples and to propagate the output variance.

    The diagonal noise variance is propagated through the normalized convolution.
    In addition to the variance of the low-pass estimate, this includes its
    covariance with the original sample because the output is ``vis - lowpass``.

    Notes
    -----
    The FFT convolution is circular and assumes a uniformly sampled, periodic RA
    axis.  For a full sidereal day this avoids introducing artificial boundaries
    at the beginning and end of the stream.

    Any number of disjoint stop bands can be used.  The primary sky, alias, and
    cross-talk cuts are combined into a single foreground response in m-space,
    FFT'd to obtain the RA-domain kernel, and estimated jointly with one mask
    normalization.  The inherited ``common_el`` and ``single_filter`` options
    remain available, but ``common_el`` defaults to false so that each elevation
    uses its own response.

    The normalized convolution requires the combined foreground response to
    include m=0.  In the intended configuration this is supplied by the
    cross-talk cut (or by a sky cut that contains m=0).  The task creates the
    elevation-dependent weight dataset when necessary and propagates a distinct
    output variance for every elevation.  It also returns a
    :class:`draco.core.containers.RingMapMask` containing the original validity
    mask used by every east-west baseline.  Both outputs store the filter
    configuration in the ``convolution_m_filter`` attribute.
    """

    transition_width = config.Property(proptype=float, default=5.0)
    in_place = config.Property(proptype=bool, default=True)

    def process(self, stream):
        """Apply the convolutional m-mode high-pass filter.

        Returns
        -------
        filtered_stream : containers.HybridVisStream
            Filtered visibilities with elevation-dependent propagated weights.
        input_mask : draco.core.containers.RingMapMask
            Samples excluded from the filter, where ``True`` indicates missing
            data.  The mask is common to all east-west baselines.
        """
        if self.transition_width < 0.0:
            raise ValueError("transition_width must be non-negative.")
        if self.save_filter or self.calculate_cov:
            self.log.warning(
                "save_filter and calculate_cov are DAYENU-only options and are "
                "ignored by ConvolutionMFilterHybridVis."
            )

        out = stream if self.in_place else stream.copy()
        out.redistribute("freq")

        # The filter response, normalization, and propagated variance can all
        # differ with elevation.  Upgrade an older HybridVisStream weight dataset
        # by broadcasting its values before filtering.
        if "elevation_vis_weight" not in out:
            input_weight = out["vis_weight"][:].local_array.copy()
            del out["vis_weight"]
            out.add_dataset("elevation_vis_weight")
            out.weight[:].local_array[:] = input_weight[..., np.newaxis, :]

        npol = out.vis.local_shape[0]

        # Get the required axes.  A HybridVisStream has a uniform RA axis.  The
        # cut calculation expects RA-independent elevation and east-west axes,
        # as in DayenuMFilterHybridVis.
        ra = np.radians(out.ra[:])
        ew = out.index_map["ew"]
        el = out.index_map["el"]

        fslc = out.vis[:].local_bounds
        freq = out.freq[fslc]

        vis = out.vis[:].local_array
        stream_weight = out.weight[:].local_array

        is_fringestopped = out.attrs.get("fringestopped", False)

        response_kwargs = {
            name: getattr(self, name)
            for name, parameter in inspect.signature(
                make_mmode_response
            ).parameters.items()
            if parameter.kind == inspect.Parameter.KEYWORD_ONLY
        }
        filter_config = {**response_kwargs, "ew": ew.tolist()}

        # Use the union of samples missing from any east-west baseline.  This
        # gives every baseline the same input mask, which can be represented by
        # a RingMapMask and later reused to filter a pulse template exactly.
        input_flag = np.all(stream_weight > 0.0, axis=2)

        for ff, nu in enumerate(freq):
            for xx, bx in enumerate(ew):
                mmode_response = make_mmode_response(
                    ra,
                    nu,
                    bx,
                    el,
                    is_fringestopped,
                    **response_kwargs,
                )
                input_weight = stream_weight[:, ff, xx].copy()

                for pp in range(npol):
                    pweight = input_weight[pp]
                    pflag = input_flag[pp, ff]

                    tvis = np.ascontiguousarray(vis[pp, ff, xx])

                    tvis, valid, tvar = apply_mmode_filter(
                        tvis,
                        mmode_response,
                        pflag,
                        tools.invert_no_zero(pweight),
                    )

                    vis[pp, ff, xx] = tvis
                    stream_weight[pp, ff, xx] = tools.invert_no_zero(
                        tvar
                    ) * valid.astype(stream_weight.dtype)

        filter_config_json = json.dumps(filter_config, sort_keys=True)
        out.attrs[MMODE_FILTER_CONFIG_ATTR] = filter_config_json

        input_mask = draco_containers.RingMapMask(
            axes_from=out,
            attrs_from=out,
            distributed=out.distributed,
            comm=out.comm,
        )
        input_mask.redistribute("freq")
        input_mask.mask[:].local_array[:] = np.swapaxes(~input_flag, -2, -1)
        input_mask.attrs[MMODE_FILTER_CONFIG_ATTR] = filter_config_json

        return out, input_mask


# class LowPassDelayFilter(tasklib.base.ContainerTask):
#
#     tauw = config.Property(proptype=float, default=0.02)
#     threshold = config.Property(proptype=float, default=1e-12)
#     eps = config.Property(proptype=float, default=1e-3)
#     in_place = config.Property(proptype=bool, default=False)
#
#     def process(self, data):
#
#         if self.in_place:
#             out = data
#         else:
#             out = data.copy()
#
#         data.redistribute("ha")
#         out.redistribute("ha")
#
#         beam = data.beam[:].local_array
#         weight = data.weight[:].local_array
#
#         obeam = out.beam[:].local_array
#         oweight = out.weight[:].local_array
#
#         vobs, vaxind = interpolate._flatten_axes(data.beam, ("object_id", "pol", "freq"))
#         wobs, waxind = interpolate._flatten_axes(data.weight, ("object_id", "pol", "freq"))
#
#         cov = dpss.make_covariance(data.freq, [self.tauw], [0.0])
#         basis = dpss.get_basis(cov, threshold=self.threshold)
#
#         ishp = tuple([vobs.shape[i] for i in vaxind[:-1]])
#
#         for ind in np.ndindex(*ishp):
#
#             Ni = wobs[ind]
#             W = Ni > 0.0
#
#             vp = dpss.project(vobs[ind], Ni, basis)
#
#             vfilt, wfilt = dpss.solve(vp, Ni, basis, self.eps)
#
#             obeam[ind] = vfilt.reshape(shp)
#             oweight[ind] = (wfilt * W).reshape(shp)
#
#         return out


class LowPassDelayFilter(tasklib.base.ContainerTask):
    """Low-pass filter beamformed data along frequency, keeping delays below tauw.

    Applied to the ``beam`` dataset with axes [object, pol, freq, ew, ha].

    Attributes
    ----------
    tauw : float
        Delay cutoff in microseconds.  Default is 0.02.
    threshold : float
        Unused.
    eps : float
        Unused.
    in_place : bool
        Modify the input container.  Default is False.
    """

    tauw = config.Property(proptype=float, default=0.02)
    threshold = config.Property(proptype=float, default=1e-12)
    eps = config.Property(proptype=float, default=1e-3)
    in_place = config.Property(proptype=bool, default=False)

    def process(self, data):
        """Filter each object, polarisation, baseline and hour angle."""
        out = data if self.in_place else data.copy()

        data.redistribute("ha")
        out.redistribute("ha")

        freq = data.freq

        beam = data.beam[:].local_array
        weight = data.weight[:].local_array

        obeam = out.beam[:].local_array
        oweight = out.weight[:].local_array

        nsource, npol, _nfreq, new, nha = beam.shape

        for ss in range(nsource):
            for hh in range(nha):
                for pp in range(npol):
                    for ee in range(new):

                        slc = (ss, pp, slice(None), ee, hh)

                        flag = weight[slc] > 0.0

                        # Construct the filter
                        t0 = time.time()
                        try:
                            NF = lowpass_delay_filter(
                                freq,
                                flag,
                                self.tauw,
                                epsilon=self.epsilon,
                            )

                        except np.linalg.LinAlgError as exc:
                            self.log.error(
                                "Failed to converge while generating filter "
                                f"for index ({ss}, {hh}, {pp}, {ee}): {exc}"
                            )
                            oweight[slc] = 0.0
                            continue

                        tbeam = np.ascontiguousarray(beam[slc])
                        tvar = tools.invert_no_zero(weight[slc])

                        obeam[slc] = np.matmul(NF, tbeam)
                        oweight[slc] = tools.invert_no_zero(
                            np.matmul(np.abs(NF) ** 2, tvar)
                        )

                        if (self.atten_threshold > 0.0) or (
                            self.abs_atten_threshold > 0.0
                        ):
                            diag = np.abs(np.diag(NF))
                            med_diag = np.median(diag[diag > 0.0])
                            th = np.maximum(
                                self.atten_threshold * med_diag,
                                self.abs_atten_threshold,
                            )

                            self.log.info(
                                f"Median diagonal is {med_diag:0.2f}, threshold is {th:0.3f}."
                            )

                            flag_low = diag > th

                            oweight[slc] *= flag_low.astype(weight.dtype)

                        self.log.info(
                            f"Took {time.time() - t0:0.2f} seconds to construct filter."
                        )

        return out


class TransientSearch(BeamFormExternalMixin, tasklib.base.ContainerTask):
    """Beamform on a catalog of sources using the HybridVisStream data product.

    Attributes
    ----------
    window : float
        Window size in degrees.  For each source, right ascensions corresponding to
        abs(ra - source_ra) <= window are extracted from the hybrid beamformed
        visibility at the declination closest to the sources location.  Default is
        5 degrees.
    ignore_rot : bool
        Ignore the telescope rotation_angle when calculating the baseline distances
        used to beamform in the east-west direction.  Defaults to False.
    """

    window = config.Property(proptype=float, default=5.0)
    ignore_rot = config.Property(proptype=bool, default=False)
    weight = config.enum(["uniform", "inverse_variance"], default="uniform")

    def setup(self, manager, beam):
        """Define the observer and the catalog of sources.

        Parameters
        ----------
        manager : draco.core.io.TelescopeConvertible
            Observer object holding the geographic location of the telescope.
            Note that if ignore_rot is False and this object has a non-zero
            rotation_angle, then the beamforming will account for the phase
            due to the north-south component of the rotation.

        beam : draco.core.containers.GridBeam
            Primary beam model used to weight the east-west beamforming.
        """
        super().setup(beam)

        self.telescope = io.get_telescope(manager)
        self.latitude = np.radians(self.telescope.latitude)
        if not self.ignore_rot and hasattr(self.telescope, "rotation_angle"):
            self.log.info(
                "Correcting for phase due to north-south component of a "
                f"{self.telescope.rotation_angle:0.2f} degree rotation."
            )
            self.rot = np.radians(self.telescope.rotation_angle)
        else:
            self.rot = 0.0

    def process(self, hvis):
        """Finish beamforming in the east-west direction.

        Parameters
        ----------
        hvis : draco.core.containers.HybridVisStream
            Visibilities beamformed in the north-south direction to
            a grid of declinations along the meridian.

        Returns
        -------
        out : draco.core.containers.HybridFormedBeamHA
            Visibilities beamformed to the location of sources
            in a catalog.
        """
        lsd = hvis.attrs.get("lsd", hvis.attrs.get("csd"))

        # Distribute over frequency, identify local frequencies
        hvis.redistribute("el")

        _npol, _nfreq, _new, _nel, nra = hvis.vis.local_shape

        el = hvis.index_map["el"][hvis.vis[:].local_bounds]
        pol = hvis.index_map["pol"]
        freq = hvis.freq
        ra = hvis.ra

        # Find hour angle window
        dra = np.median(np.abs(np.diff(ra)))

        nwin = int(np.floor(self.window / dra))
        nha = 2 * nwin + 1

        offset = np.arange(nha, dtype=int) - nwin

        ha_deg = offset * dra
        ha = np.radians(ha_deg)

        decs = np.arcsin(el) + self.latitude

        timestamp = self.telescope.lsd_to_unix(lsd + ra / 360.0)

        # Calculate baseline distances
        lmbda = scipy.constants.c / (freq * 1e6)

        ew = hvis.index_map["ew"]
        u = (
            ew[np.newaxis, :, np.newaxis] / lmbda[:, np.newaxis, np.newaxis]
        )  # freq, ew, ha
        v = np.sin(self.rot) * u

        # Dereference input datasets
        vis = hvis.vis[:].local_array  # pol, freq, ew, el, ra
        weight = hvis.weight[:].local_array  # pol, freq, ew, ra

        # Create the output container
        out = containers.BandAveragedBeamformedData(
            ha=ha_deg,
            axes_from=hvis,
            attrs_from=hvis,
            distributed=hvis.distributed,
            comm=hvis.comm,
        )

        out.redistribute("el")

        # Dereference output datasets
        ovis = out.vis[:].local_array
        ovis[:] = 0.0

        oweight = out.weight[:].local_array
        oweight[:] = 0.0

        otime = out.time[:].local_array
        otime[:] = 0.0

        ax = (0, 1)

        # Loop over polarisations and elevations
        for pp, pstr in enumerate(pol):

            for dd, dec in enumerate(decs):

                # Calculate the template
                beam = self._beamfunc(pstr, dec, ha)  # freq_local, ha
                beam = mpiarray.MPIArray.wrap(beam, axis=0).allgather()  # freq, ha

                phi = interferometry.fringestop_phase(
                    ha, self.latitude, dec, u, v
                )  # freq, ew, ha

                template = beam[:, np.newaxis, :].conj() * phi  # freq, ew, ha

                for ii, oo in enumerate(offset):

                    r_min = max(0, oo)
                    r_max = min(nra, nra + oo)
                    if r_min >= r_max:
                        continue

                    r_slc = slice(r_min, r_max)
                    s_min, s_max = r_min - oo, r_max - oo
                    s_slc = slice(s_min, s_max)

                    otime[pp, dd, s_slc, ii] = timestamp[r_slc]

                    t = template[:, :, ii, np.newaxis]
                    y = vis[pp, :, :, dd, r_slc]

                    w = weight[pp, :, :, r_slc]
                    var = tools.invert_no_zero(w)

                    if self.weight == "uniform":
                        w = (w > 0.0).astype(float)

                    tt = np.abs(t) ** 2

                    inv_sum_wtt = tools.invert_no_zero(np.sum(w * tt, axis=ax))
                    sum_wty = np.sum(w * t * y, axis=ax)

                    sum_wwttv = np.sum(w**2 * tt * var, axis=ax)

                    ovis[pp, dd, s_slc, ii] = sum_wty * inv_sum_wtt
                    oweight[pp, dd, s_slc, ii] = tools.invert_no_zero(
                        sum_wwttv * inv_sum_wtt**2
                    )

        return out


def circular_rolling_sum(a, window, wrap=False):
    """Sum over a window centred on each sample along the last axis.

    Parameters
    ----------
    a : np.ndarray
        Array to sum.
    window : int
        Number of samples in the window.  Must be odd.
    wrap : bool
        Wrap around the ends of the axis.  Otherwise pad with zeros.

    Returns
    -------
    out : np.ndarray
        Rolling sum with the same shape as ``a``.
    """
    if not (window % 2):
        raise RuntimeError(
            f"Requested a window of size {window}, but window must be odd."
        )

    if window == 1:
        return a

    edge = window // 2

    tail = a[..., -edge:] if wrap else np.zeros((*a.shape[:-1], edge), dtype=a.dtype)
    head = a[..., :edge] if wrap else np.zeros((*a.shape[:-1], edge), dtype=a.dtype)
    e = np.concatenate((tail, a, head), axis=-1)

    shape = (*e.shape[:-1], e.shape[-1] - window + 1, window)
    strides = (*e.strides, e.strides[-1])

    eroll = np.lib.stride_tricks.as_strided(e, shape=shape, strides=strides)

    return np.sum(eroll, axis=-1)


def fit_amp_offset(y, t, w, eps=0.0, nroll=0, with_cov=False, wrap=False):
    """Weighted least-squares fit of y = a * t + b, row-wise.

    Parameters
    ----------
    y, t, w : np.ndarray with shape (nrecord, nsample)
        Data, template, and inverse-variance weights.
        Set w=0 where a sample should be ignored.
    eps : float
        Small ridge added to the normal matrix to guard against singular rows
        (use 0.0 if you prefer hard failure via NaNs).
    nroll : int
        If positive, fit in a rolling window of this many samples centred on each
        sample instead of the whole record.
    with_cov : bool
        Also return the covariance of the parameters.
    wrap : bool
        Wrap the rolling window around the ends of each record.

    Returns
    -------
    a, b : np.ndarray with shape (nrecord,)
        Best-fit amplitude and offset per record.
    cov : np.ndarray with shape (nrecord, 2, 2)
        (X^T W X)^{-1} per record. If noise variances are correctly scaled,
        this is the parameter covariance.
    """
    if nroll > 0:

        def fsum(x):
            return circular_rolling_sum(x, nroll, wrap=wrap)

    else:

        def fsum(x):
            return np.sum(x, axis=-1)

    # Weighted sums per record
    Sw = fsum(w)
    St = fsum(w * t)
    Sy = fsum(w * y)
    Stt = fsum(w * np.abs(t) ** 2)
    Sty = fsum(w * t.conj() * y)

    # Normal matrix and RHS per record:
    # [[Stt, St], [St, Sw]] @ [a, b] = [Sty, Sy]
    D = (Stt + eps) * (Sw + eps) - np.abs(St) ** 2

    invD = tools.invert_no_zero(D)

    # Solve 2x2 systems in closed form, vectorized
    a = (Sty * Sw - St.conj() * Sy) * invD
    b = (Stt * Sy - St * Sty) * invD

    # Covariance
    if not with_cov:

        return a, b

    cov = np.empty((*a.shape, 2, 2), dtype=y.dtype)
    cov[..., 0, 0] = Sw * invD
    cov[..., 0, 1] = -St.conj() * invD
    cov[..., 1, 0] = -St * invD
    cov[..., 1, 1] = Stt * invD

    return a, b, cov


def lowpass_delay_filter(freq, flag, tau, epsilon=1e-12):
    """Construct a filter along frequency that keeps delays below ``tau``.

    Parameters
    ----------
    freq : np.ndarray[nfreq]
        Frequency in MHz.
    flag : np.ndarray[nfreq] of bool
        Valid frequencies.
    tau : float
        Delay cutoff in microseconds.
    epsilon : float
        Rejection of delays above the cutoff.

    Returns
    -------
    filt : np.ndarray[nfreq, nfreq]
        Filter, zero for invalid frequencies.
    """
    # Make sure the flag array is properly sized
    ishp = flag.shape
    nfreq = freq.size
    assert ishp[0] == nfreq
    assert len(ishp) == 1

    # Construct the covariance matrix
    dfreq = freq[:, np.newaxis] - freq[np.newaxis, :]

    delta_freq = np.median(np.abs(np.diff(freq)))
    a = 2.0 * delta_freq * tau  # = tau / taum
    aeps = a * epsilon  # (tau / taum) * epsilon

    cov = np.eye(nfreq, dtype=np.float64) / aeps
    cov += a * (1.0 - 1.0 / aeps) * np.sinc(2.0 * tau * dfreq)
    print("using lpf3")
    #
    # taum = 1.0 / (2.0 * np.median(np.abs(np.diff(freq))))
    #
    # cov = np.eye(nfreq, dtype=np.float64) / epsilon
    # cov += (1.0 - tau / (taum * epsilon)) * np.sinc(2.0 * tau * dfreq)

    # cov = np.eye(nfreq, dtype=np.float64) / epsilon + np.sinc(2.0 * tau * dfreq)

    valid = np.flatnonzero(flag)
    valid_2d = np.ix_(valid, valid)

    cho = cho_factor(cov[valid_2d], lower=True, check_finite=False, overwrite_a=True)

    I = np.eye(valid.size, dtype=np.float64)

    NF = np.zeros_like(cov)
    NF[valid_2d] = cho_solve(cho, I, check_finite=False, overwrite_b=True)

    return NF


def mmode_filter(ra, flag, m_width, m_centre=0.0, epsilon=1e-12):
    """Construct an m-mode filter.

    The filter will attenuate signals with m ranging from
    [m_centre - m_width, m_centre + m_width].  If more
    than one value of m_centre and m_width are provided,
    then the filter will have multiple stop bands.

    Parameters
    ----------
    ra : np.ndarray[nra,]
        Transiting Right Ascension in radians.
    flag : np.ndarray[nra,]
        Boolean flag that indicates what data are valid.
    m_width : float or np.ndarray[nstopband,]
        The half width of the stop-band region in micro-seconds.
    m_centre : float or np.ndarray[nstopband,]
        The centre of the stop-band region in micro-seconds.
        Defaults to 0.
    epsilon : float or np.ndarray[nstopband,]
        The stop-band rejection.  Defaults to 1e-12.

    Returns
    -------
    pinv : np.ndarray[nvalid, nvalid]
        Pseudo-inverse of the analytical covariance matrix with flag applied.
    """

    # Ensure consistent size for parameter values
    def _ensure_consistent(param, nstopband):
        if np.isscalar(param):
            return [param] * nstopband
        if len(param) == 1:
            return [param[0]] * nstopband
        assert len(param) == nstopband
        return param

    args = [m_width, m_centre, epsilon]
    nstopband = np.max([np.atleast_1d(param).size for param in args])
    args = [np.array(_ensure_consistent(param, nstopband)) for param in args]

    # Determine datatype
    dtype = np.complex128 if np.any(np.abs(args[1]) > 0.0) else np.float64

    # Make sure the flag array is properly sized
    ishp = flag.shape
    nra = ra.size
    assert ishp[0] == nra
    assert len(ishp) == 1

    # Construct the covariance matrix
    dra = ra[:, np.newaxis] - ra[np.newaxis, :]

    cov = np.eye(nra, dtype=dtype)
    for mw, mc, eps in zip(*args):

        term = np.sinc(mw * dra / np.pi) / eps
        if np.abs(mc) > 0.0:
            term = term * np.exp(1.0j * mc * dra)

        cov += term

    valid = np.flatnonzero(flag)
    valid_2d = np.ix_(valid, valid)

    cho = cho_factor(cov[valid_2d], lower=True, check_finite=False, overwrite_a=True)

    I = np.eye(valid.size, dtype=dtype)

    NF = np.zeros_like(cov)
    NF[valid_2d] = cho_solve(cho, I, check_finite=False, overwrite_b=True)

    return NF.T


def apply_mmode_filter(
    data,
    mmode_response,
    flag,
    variance=None,
):
    """Apply a normalized convolutional m-mode high-pass filter.

    Parameters
    ----------
    data : np.ndarray[..., nra]
        Data to filter.  The final axis must correspond to right ascension.
    mmode_response : np.ndarray[..., nra]
        Low-pass transfer function in m-space.  This may be the union of any
        number of disjoint foreground bands.  Its m=0 sample must be one so that
        the mask normalization is well defined.
    flag : np.ndarray[..., nra] of bool
        Valid input samples.  This is applied in both the numerator and
        denominator of the foreground estimate.
    variance : np.ndarray[..., nra], optional
        Independent input-sample variances to propagate through the filter.  If
        omitted, only the filtered data and validity flag are returned.  This is
        useful for applying the response to a pulse template.

    Returns
    -------
    filtered : np.ndarray[..., nra]
        The high-pass filtered data.
    valid : np.ndarray[..., nra] of bool
        Valid input samples with a reliable mask normalization.
    filtered_variance : np.ndarray[..., nra]
        Diagonal variance of the filtered data.  Only returned when ``variance``
        is provided.

    Notes
    -----
    The low-pass variance is the input variance convolved with the squared
    combined kernel.  The output variance also includes the covariance between
    the original sample and its low-pass estimate.  The calculation is valid for
    complex kernels produced by asymmetric or non-contiguous foreground bands.
    """
    numerator = np.fft.ifft(
        np.fft.fft(data * flag, axis=-1) * mmode_response,
        axis=-1,
    )
    denominator = np.fft.ifft(
        np.fft.fft(flag, axis=-1) * mmode_response,
        axis=-1,
    )

    valid = flag & (denominator != 0.0)

    inv_denominator = np.zeros_like(denominator, dtype=np.complex128)
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        np.divide(1.0, denominator, out=inv_denominator, where=valid)
        foreground = numerator * inv_denominator

    filtered = data - foreground
    valid &= np.isfinite(filtered)
    filtered = np.where(valid, filtered, 0.0)

    if variance is None:
        return filtered, valid

    # The variance of the normalized low-pass estimate is
    #
    #   conv(|kernel|^2, flag * variance) / |denominator|^2.
    #
    # Since the final result is data - lowpass(data), subtract twice the
    # covariance of the input sample with its low-pass estimate.
    kernel = np.fft.ifft(mmode_response, axis=-1)
    lowpass_variance = np.fft.ifft(
        np.fft.fft(flag * variance, axis=-1) * np.fft.fft(np.abs(kernel) ** 2, axis=-1),
        axis=-1,
    ).real

    with np.errstate(invalid="ignore", over="ignore"):
        filtered_variance = np.where(
            valid,
            variance
            + lowpass_variance * np.abs(inv_denominator) ** 2
            - 2.0
            * (kernel[..., 0, np.newaxis] * flag * variance * inv_denominator).real,
            0.0,
        )

    # Guard against tiny negative values from cancellation at machine precision.
    filtered_variance = np.maximum(filtered_variance, 0.0)
    valid &= np.isfinite(filtered_variance)
    filtered = np.where(valid, filtered, 0.0)
    filtered_variance = np.where(valid, filtered_variance, 0.0)

    return filtered, valid, filtered_variance


def _mmode_axis_from_ra(ra):
    """Return the discrete angular m-mode axis for a uniform RA grid."""
    dra = np.median(np.diff(ra))
    return 2.0 * np.pi * np.fft.fftfreq(ra.size, d=dra)


def foreground_mmode_response(mmode, m_width, m_centre=0.0, transition_width=0.0):
    """Construct the union of tapered foreground bands in m-space."""
    m_width, m_centre = np.broadcast_arrays(
        np.atleast_1d(m_width),
        np.atleast_1d(m_centre),
    )

    # A DFT response is periodic.  Use the wrapped distance so that bands and
    # their transitions behave correctly if they cross the Nyquist boundary.
    dm = np.abs(mmode[1] - mmode[0])
    m_period = dm * mmode.size
    delta = mmode[np.newaxis, :] - m_centre.ravel()[:, np.newaxis]
    distance = np.abs((delta + 0.5 * m_period) % m_period - 0.5 * m_period)
    width = m_width.ravel()[:, np.newaxis]

    response = (distance <= width).astype(float)

    if transition_width > 0.0:
        transition = (distance > width) & (distance < (width + transition_width))
        x = (distance - width) / transition_width
        response = np.where(
            transition,
            0.5 * (1.0 + np.cos(np.pi * x)),
            response,
        )

    # Taking the maximum forms the union of the sky, cross-talk, and alias
    # components without double-counting regions where their tapers overlap.
    return np.max(response, axis=0)


def get_mmode_cuts(
    freq,
    xsep,
    el,
    *,
    is_fringestopped,
    lat,
    width,
    min_ysep,
    num_width,
    num_width_alias,
    m_cut_cross_talk,
    common_el,
    single_filter,
    min_m_sep,
):
    """Calculate the sky, alias, and cross-talk m-mode stop bands."""
    nel = el.size
    mc = np.zeros((3, nel), dtype=float)
    mw = np.zeros((3, nel), dtype=float)
    cut_flag = np.zeros((3, nel), dtype=bool)

    lmbda = scipy.constants.c / (freq * 1e6)

    # Primary sky cut.
    uc = xsep / lmbda
    uw = num_width * width / lmbda
    scale = -2.0 * np.pi * np.cos(np.arcsin(el) + lat)

    cut_flag[0] = True
    mc[0] = scale * uc
    mw[0] = np.abs(scale * uw)

    offset = mc[0].copy() if is_fringestopped else 0.0

    # North-south beam alias cut.
    if num_width_alias > 0.0:
        el_alias = el + (1.0 - 2.0 * (el > 0.0)) * lmbda / min_ysep
        flag_alias = np.abs(el_alias) < 1.0

        if np.any(flag_alias):
            uwa = num_width_alias * width / lmbda
            scale_alias = -2.0 * np.pi * np.cos(np.arcsin(el_alias[flag_alias]) + lat)

            cut_flag[1] = flag_alias
            mc[1, flag_alias] = scale_alias * uc
            mw[1, flag_alias] = np.abs(scale_alias * uwa)

    # Cross-talk cut around m=0.
    if m_cut_cross_talk > 0.0:
        cut_flag[2] = True
        mw[2] = np.abs(m_cut_cross_talk)

    mc = mc - offset * cut_flag

    omc, omw = [], []
    if common_el:
        el_index = [slice(None)]

        mlower = np.min(np.where(cut_flag, mc - mw, np.inf), axis=-1)
        mupper = np.max(np.where(cut_flag, mc + mw, -np.inf), axis=-1)
        valid_cut = np.isfinite(mlower) & np.isfinite(mupper)

        centre = 0.5 * (mlower[valid_cut] + mupper[valid_cut])
        half_width = 0.5 * (mupper[valid_cut] - mlower[valid_cut])
        centre, half_width = merge_ranges(
            centre,
            half_width,
            eps=min_m_sep,
        )
        omc.append(centre)
        omw.append(half_width)

    else:
        el_index = list(range(nel))
        for ee in el_index:
            valid_cut = np.flatnonzero(cut_flag[:, ee])
            centre, half_width = merge_ranges(
                mc[valid_cut, ee],
                mw[valid_cut, ee],
                eps=min_m_sep,
            )
            omc.append(centre)
            omw.append(half_width)

    if single_filter:
        smc, smw = [], []
        for centre, half_width in zip(omc, omw):
            lower = np.min(centre - half_width)
            upper = np.max(centre + half_width)
            smc.append(np.atleast_1d(0.5 * (lower + upper)))
            smw.append(np.atleast_1d(0.5 * (upper - lower)))
        omc, omw = smc, smw

    return el_index, omc, omw


def make_mmode_response(
    ra,
    freq,
    xsep,
    el,
    is_fringestopped,
    *,
    num_width,
    num_width_alias,
    m_cut_cross_talk,
    common_el,
    single_filter,
    min_m_sep,
    transition_width,
    lat,
    width,
    min_ysep,
):
    """Reconstruct the elevation-dependent foreground response in m-space.

    Parameters
    ----------
    ra : np.ndarray[nra]
        Right ascension in radians.
    freq : float
        Frequency in MHz.
    xsep : float
        East-west baseline separation in metres.
    el : np.ndarray[nel]
        Sine of the elevation coordinate.
    is_fringestopped : bool
        Whether the input visibilities have been fringe stopped.
    num_width, num_width_alias, m_cut_cross_talk : float
        Widths of the primary sky, aliased sky, and cross-talk cuts.
    common_el, single_filter : bool
        Options controlling whether responses are shared or cuts are merged.
    min_m_sep, transition_width : float
        Minimum separation used when merging cuts and the width of their cosine
        transitions.
    lat : float
        Telescope latitude in radians.
    width, min_ysep : float
        Telescope dimensions used to calculate the sky and alias cuts.

    Returns
    -------
    response : np.ndarray[nra] or np.ndarray[nel, nra]
        Foreground response on the discrete m-mode FFT grid.  A one-dimensional
        response is returned when ``common_el`` is true.
    """
    _, m_centre, m_width = get_mmode_cuts(
        freq,
        xsep,
        el,
        is_fringestopped=is_fringestopped,
        lat=lat,
        width=width,
        min_ysep=min_ysep,
        num_width=num_width,
        num_width_alias=num_width_alias,
        m_cut_cross_talk=m_cut_cross_talk,
        common_el=common_el,
        single_filter=single_filter,
        min_m_sep=min_m_sep,
    )

    mmode = _mmode_axis_from_ra(ra)
    response = [
        foreground_mmode_response(
            mmode,
            half_width,
            centre,
            transition_width,
        )
        for centre, half_width in zip(m_centre, m_width)
    ]
    response = response[0] if common_el else np.stack(response)

    if not np.all(response[..., 0] == 1.0):
        raise RuntimeError(
            "The combined foreground response must include m=0 in its "
            "unit-response region for normalized convolution."
        )

    return response


def merge_ranges(c, w, eps=0.0):
    """Merge overlapping (or nearly-overlapping) ranges given centers `c` and half-widths `w`.

    Interval i is [c[i] - w[i], c[i] + w[i]].
    After sorting by left edge, a new merged group starts only when
        next_left > current_max_right + eps
    so:
      - Touching intervals (next_left == current_max_right) MERGE when eps >= 0 (default 0.0).
      - Small gaps up to size `eps` are bridged (use larger eps to remove tiny slivers).

    Parameters
    ----------
    c : array-like of float
        Centres of the ranges.
    w : array-like of float
        Half-widths of the ranges (must be >= 0).
    eps : float, optional
        Nonnegative tolerance for merging across small gaps. Default 0.0.

    Returns
    -------
    merged_c : np.ndarray
        Centers of merged ranges.
    merged_w : np.ndarray
        Half-widths of merged ranges.
    """
    c = np.asarray(c, dtype=float)
    w = np.asarray(w, dtype=float)

    if c.size != w.size:
        raise ValueError("c and w must have the same length")
    if c.size == 0:
        return np.array([]), np.array([])
    if not np.all(np.isfinite(c)) or not np.all(np.isfinite(w)):
        raise ValueError("centers and widths must be finite")
    if np.any(w < 0):
        raise ValueError("half-widths must be nonnegative")
    if eps < 0:
        raise ValueError("eps must be nonnegative")

    # Convert to [L, R], sort by L
    L = c - w
    R = c + w
    order = np.argsort(L)
    Ls = L[order]
    Rs = R[order]

    # Sweep with cumulative max of right edges
    cumR = np.maximum.accumulate(Rs)

    # Start a new group when the gap exceeds eps
    new_group = np.empty(Ls.size, dtype=bool)
    new_group[0] = True
    new_group[1:] = Ls[1:] >= (cumR[:-1] + eps)

    starts = np.flatnonzero(new_group)
    ends = np.r_[starts[1:] - 1, Ls.size - 1]

    merged_L = Ls[starts]
    merged_R = cumR[ends]

    merged_c = (merged_L + merged_R) / 2.0
    merged_w = (merged_R - merged_L) / 2.0
    return merged_c, merged_w


# -----------------------------------------------------------------------------
# Masked least-squares m-mode filter
# -----------------------------------------------------------------------------


class LSQMFilterHybridVis(DayenuMFilterHybridVis):
    """High-pass filter a HybridVisStream by least-squares fitting of m-modes.

    For every frequency, east-west baseline, and polarisation, the m-modes in the
    foreground stop band are fit to the data and subtracted.  The stop band is
    defined as in :class:`DayenuMFilterHybridVis`.  By default the fit uses a full
    sidereal day and is weighted by the inverse variance; with uniform weights this
    is equivalent to DAYENU.  If ``fir_length`` is set, the fit is instead done in a
    window of that many samples centred on each sample, so that each sample only
    affects the output within ``fir_length // 2`` samples.

    The output contains the filtered visibilities, their inverse variance, the
    inverse variance of the input visibilities, and the elements of the filter near
    the diagonal, which are used by :class:`TransientMatchedFilter`.

    Attributes
    ----------
    fir_length : int
        Length of the sliding window in samples.  Zero (default) fits the full day.
    fir_margin : float
        Increase the half-width of each stop band by this much when using a sliding
        window, which cannot cut as sharply in m.  Default is 0.
    weighted : bool
        Weight the full-day fit by the inverse variance.  Default is True.
    pol : list of str
        Polarisations to filter.  Default is all.
    ew_index : list of int
        Indices of the east-west baselines to filter.  Default is all.
    response_max_lag : int
        Maximum lag of the saved filter response.  Default is 31.
    nrow_chunk : int
        Number of elevations to filter at once.  Default is 256.
    """

    fir_length = config.Property(proptype=int, default=0)
    fir_margin = config.Property(proptype=float, default=0.0)
    weighted = config.Property(proptype=bool, default=True)
    pol = config.list_type(str, default=None)
    ew_index = config.list_type(int, default=None)
    response_max_lag = config.Property(proptype=int, default=31)
    nrow_chunk = config.Property(proptype=int, default=256)

    def process(self, stream):
        """Filter the stream.

        Parameters
        ----------
        stream : draco.core.containers.HybridVisStream
            Hybrid beamformed visibilities with elevation-independent weights.

        Returns
        -------
        out : ch_pipeline.core.containers.MFilteredHybridVisStream
            Filtered visibilities, their weights, the input noise weights, and the
            near-diagonal filter response.
        """
        if not self.common_el:
            raise ValueError("LSQMFilterHybridVis requires common_el = True.")
        if self.fir_length < 0 or self.response_max_lag < 0:
            raise ValueError("fir_length and response_max_lag must be non-negative.")
        if self.save_filter or self.calculate_cov:
            self.log.warning(
                "save_filter and calculate_cov are DAYENU-only options and are "
                "ignored by LSQMFilterHybridVis."
            )

        stream.redistribute("freq")

        if "elevation_vis_weight" in stream.datasets:
            raise ValueError(
                "LSQMFilterHybridVis requires elevation-independent weights."
            )

        # Check the RA axis
        ra = stream.ra[:]
        nra = ra.size
        dra = np.median(np.diff(ra))
        if not np.allclose(np.diff(ra), dra, rtol=1e-4):
            raise ValueError("LSQMFilterHybridVis requires a uniform RA axis.")
        periodic = _is_full_day(ra)
        if self.fir_length == 0 and not periodic:
            raise ValueError(
                "The global filter requires a full sidereal day. "
                "Set fir_length to filter a partial day."
            )
        dra_rad = np.radians(dra)

        fir_length = (
            self.fir_length + (1 - self.fir_length % 2) if self.fir_length else 0
        )
        if fir_length > nra:
            raise ValueError(f"fir_length ({fir_length}) exceeds the RA axis ({nra}).")
        max_lag = self.response_max_lag
        if fir_length and max_lag > fir_length // 2:
            raise ValueError("response_max_lag cannot exceed fir_length // 2.")

        # Select the polarisations and baselines to process
        pol = list(stream.index_map["pol"])
        ipol = (
            list(range(len(pol)))
            if self.pol is None
            else [pol.index(p) for p in self.pol]
        )

        ew = stream.index_map["ew"][:]
        iew = list(range(ew.size)) if self.ew_index is None else list(self.ew_index)

        el = stream.index_map["el"][:]
        nel = el.size

        is_fringestopped = stream.attrs.get("fringestopped", False)

        out = containers.MFilteredHybridVisStream(
            pol=np.array([pol[ii] for ii in ipol]),
            ew=ew[iew],
            lag=np.arange(-max_lag, max_lag + 1),
            axes_from=stream,
            attrs_from=stream,
            distributed=stream.distributed,
            comm=stream.comm,
        )
        out.add_dataset("ra_response")
        out.add_dataset("noise_weight")
        out.redistribute("freq")

        fslc = stream.vis[:].local_bounds
        freq = stream.freq[fslc]

        vis_in = stream.vis[:].local_array
        weight_in = stream.weight[:].local_array

        vis = out.vis[:].local_array
        weight = out.weight[:].local_array
        noise_weight = out.noise_weight[:].local_array
        response = out.ra_response[:].local_array
        vis[:] = 0.0
        weight[:] = 0.0
        noise_weight[:] = 0.0
        response[:] = 0.0

        for ff, nu in enumerate(freq):
            for oxx, xx in enumerate(iew):
                _, mc, mw = self._get_cut(nu, ew[xx], el, is_fringestopped)
                mc = np.atleast_1d(mc[0])
                mw = np.atleast_1d(mw[0])

                t0 = time.time()

                # The FIR filter uses one flag for all polarisations
                if fir_length:
                    flag = np.all(weight_in[ipol, ff, xx] > 0.0, axis=0)
                    if not np.any(flag):
                        continue
                    filt = self._make_filter(
                        lambda: MaskedLSQFIRFilter(
                            flag,
                            dra_rad,
                            mc,
                            mw + self.fir_margin,
                            fir_length,
                            epsilon=self.epsilon,
                            periodic=periodic,
                        ),
                        nu,
                        ew[xx],
                    )
                    if filt is None:
                        continue
                else:
                    mmode_index = stopband_mmode_index(nra, mc, mw)

                for opp, pp in enumerate(ipol):
                    w = weight_in[pp, ff, xx].astype(np.float64)

                    if fir_length:
                        w = w * flag
                    else:
                        if not np.any(w > 0.0):
                            continue
                        fit_weight = (
                            w if self.weighted else (w > 0.0).astype(np.float64)
                        )
                        filt = self._make_filter(
                            lambda: MaskedLSQMModeFilter(
                                fit_weight, mmode_index, epsilon=self.epsilon
                            ),
                            nu,
                            ew[xx],
                        )
                        if filt is None:
                            continue

                    valid = self._valid_samples(filt.gain)

                    for e0 in range(0, nel, self.nrow_chunk):
                        esl = slice(e0, min(e0 + self.nrow_chunk, nel))
                        yfilt = filt.apply(vis_in[pp, ff, xx, esl])
                        vis[opp, ff, oxx, esl] = np.where(valid, yfilt, 0.0)

                    var = filt.propagate_variance(tools.invert_no_zero(w))
                    weight[opp, ff, oxx] = tools.invert_no_zero(var) * valid
                    noise_weight[opp, ff, oxx] = w * valid
                    response[opp, ff, oxx] = filt.response(max_lag)

                self.log.debug(
                    f"Filtered freq {nu:0.2f} MHz, ew {ew[xx]:0.1f} m "
                    f"({filt.nmode} modes) in {time.time() - t0:0.2f} seconds."
                )

        return out

    def _make_filter(self, factory, nu, bx):
        try:
            return factory()
        except np.linalg.LinAlgError as exc:
            self.log.error(
                f"Failed to construct filter for freq {nu:0.2f} MHz and ew {bx:0.1f} m: {exc}"
            )
            return None

    def _valid_samples(self, gain):
        valid = gain > 0.0
        th = 0.0
        if self.atten_threshold > 0.0 and np.any(valid):
            th = self.atten_threshold * np.median(gain[valid])
        th = max(th, self.abs_atten_threshold)
        if th > 0.0:
            valid &= gain > th
        return valid


class CorrectFilterGain(tasklib.base.ContainerTask):
    """Divide m-mode filtered visibilities by the zero-lag response of the filter.

    After this, a transient lasting one sample has its full amplitude.

    Attributes
    ----------
    min_gain : float
        Give zero weight to samples with a lower zero-lag response.  Default is 0.
    in_place : bool
        Correct the input in place.  The filter response is removed, so this is only
        safe if no other task uses the input.  Default is False.
    """

    min_gain = config.Property(proptype=float, default=0.0)
    in_place = config.Property(proptype=bool, default=False)

    def process(self, stream):
        """Apply the gain correction.

        Parameters
        ----------
        stream : ch_pipeline.core.containers.MFilteredHybridVisStream
            Output of :class:`LSQMFilterHybridVis`.

        Returns
        -------
        stream : ch_pipeline.core.containers.MFilteredHybridVisStream
            Gain-corrected visibilities and weights.
        """
        if "ra_response" not in stream.datasets:
            raise ValueError("The stream has no filter response to correct for.")

        if not self.in_place:
            stream = stream.copy()

        stream.redistribute("freq")

        lag = list(stream.index_map["lag"])
        gain = 1.0 - stream.ra_response[:].local_array[..., lag.index(0)].real

        valid = gain > max(self.min_gain, 0.0)
        inv_gain = np.where(valid, tools.invert_no_zero(gain), 0.0)

        stream.vis[:].local_array[:] *= inv_gain[:, :, :, np.newaxis, :]
        stream.weight[:].local_array[:] *= (gain**2 * valid).astype(np.float32)

        del stream["ra_response"]
        del stream["noise_weight"]
        stream.attrs["filter_gain_corrected"] = True

        return stream


def _is_full_day(ra):
    """Whether a uniform RA axis in degrees spans one sidereal day."""
    dra = np.median(np.diff(ra))
    return bool(np.abs(ra.size * dra - 360.0) < 1e-3 * dra)


def stopband_mmode_index(nra, m_centre, m_width):
    """Return the FFT bins of the integer m-modes inside a set of stop bands.

    Parameters
    ----------
    nra : int
        Number of samples in a full, periodic sidereal day.
    m_centre, m_width : np.ndarray[nband]
        Centre and half-width of each stop band.

    Returns
    -------
    index : np.ndarray[nmode]
        Indices into a length ``nra`` FFT of the m-modes inside any stop band.
    """
    m = np.fft.fftfreq(nra, d=1.0 / nra)

    sel = np.zeros(nra, dtype=bool)
    for mc, mw in zip(np.atleast_1d(m_centre), np.atleast_1d(m_width)):
        distance = np.abs((m - mc + 0.5 * nra) % nra - 0.5 * nra)
        sel |= distance <= mw

    return np.flatnonzero(sel)


class MaskedLSQMModeFilter:
    """Remove a set of m-modes from periodic data by weighted least squares.

    Computes ``(I - P) y`` with ``P = F (F^H W F + r)^-1 F^H W``, where
    ``F[t, k] = exp(2 pi i m_k t / N)``, ``W`` holds the weights, and ``r`` is a
    small regularisation, ``epsilon`` times the diagonal of ``F^H W F``.

    Parameters
    ----------
    weight : np.ndarray[N]
        Non-negative weights.
    mmode_index : np.ndarray[nmode] of int
        FFT bins of the m-modes to remove.
    epsilon : float
        Relative regularisation.
    """

    def __init__(self, weight, mmode_index, epsilon=1e-12):
        w = np.asarray(weight, dtype=np.float64)
        self.flag = w > 0.0
        self.N = N = w.size
        self.m = m = np.unique(np.asarray(mmode_index) % N)
        self.nmode = m.size

        # The projection does not depend on the overall scale of the weights
        self.w = w / np.mean(w[self.flag]) if np.any(self.flag) else w

        # F^H W F depends only on m_k - m_l, so it is given by FFT(w)
        self._dm = (m[:, np.newaxis] - m[np.newaxis, :]) % N

        fwf = np.fft.fft(self.w)[self._dm]
        fwf[np.diag_indices_from(fwf)] += epsilon * self.w.sum()

        self._cho = cho_factor(fwf, lower=True, check_finite=False)
        self.inv_fwf = cho_solve(
            self._cho, np.eye(m.size, dtype=np.complex128), check_finite=False
        )

        # P_tt = w_t Q_tt with Q = F (F^H W F + r)^-1 F^H
        self.p_diag = np.where(self.flag, self.w * self._quadform_lag(0).real, 0.0)
        self.gain = np.where(self.flag, 1.0 - self.p_diag, 0.0)

    def _quadform_lag(self, lag):
        """Return ``Q[t, t + lag]`` for every ``t``, where ``Q = F (F^H W F + r)^-1 F^H``."""
        mat = self.inv_fwf
        if lag:
            mat = mat * np.exp(-2.0j * np.pi * self.m * lag / self.N)[np.newaxis, :]
        dm = self._dm.ravel()
        c = np.bincount(dm, weights=mat.real.ravel(), minlength=self.N) + 1.0j * (
            np.bincount(dm, weights=mat.imag.ravel(), minlength=self.N)
        )
        return self.N * np.fft.ifft(c)

    def apply(self, data):
        """Apply the filter along the last axis of ``data``."""
        y = np.asarray(data, dtype=np.complex128) * self.flag

        proj = np.fft.fft(y * self.w, axis=-1)[..., self.m]
        coeff = np.zeros(y.shape, dtype=np.complex128)
        coeff[..., self.m] = cho_solve(
            self._cho, proj.reshape(-1, self.nmode).T, check_finite=False
        ).T.reshape(proj.shape)

        foreground = self.N * np.fft.ifft(coeff, axis=-1)

        return (y - foreground) * self.flag

    def propagate_variance(self, variance):
        """Return the variance of the filtered data given the input variance."""
        return np.where(self.flag, self.gain * np.asarray(variance), 0.0)

    def response(self, max_lag):
        """Return the filter elements ``P[t, t + lag]`` for ``|lag| <= max_lag``.

        Returns
        -------
        resp : np.ndarray[N, 2 * max_lag + 1]
            Rows of samples with zero weight are zero.
        """
        N = self.N
        resp = np.zeros((N, 2 * max_lag + 1), dtype=np.complex128)

        for d in range(max_lag + 1):
            q = self._quadform_lag(d)
            # P[t, t + d] = Q[t, t + d] w[t + d]
            resp[:, max_lag + d] = q * np.roll(self.w, -d)
            if d:
                # Q is Hermitian: Q[t, t - d] = conj(Q[t - d, t])
                resp[:, max_lag - d] = np.roll(q, d).conj() * np.roll(self.w, d)

        resp[~self.flag] = 0.0
        return resp


class MaskedLSQFIRFilter:
    """Remove a set of m-modes by least squares in a sliding window.

    For each sample, the m-modes are fit to the valid data in a window of ``length``
    samples centred on it, using the eigenvectors of the DAYENU covariance over the
    window.  This is equivalent to applying DAYENU to each window.

    Parameters
    ----------
    flag : np.ndarray[N] of bool
        Valid samples.
    dra : float
        Spacing of the samples in radians.
    m_centre, m_width : np.ndarray[nband]
        Centre and half-width of each stop band.
    length : int
        Window length in samples.  Must be odd.
    epsilon : float
        Stop-band rejection, as in DAYENU.
    periodic : bool
        Whether the samples span a full sidereal day.  If not, windows are
        truncated at the edges.
    """

    def __init__(
        self, flag, dra, m_centre, m_width, length, epsilon=1e-12, periodic=True
    ):
        if not length % 2:
            raise ValueError("length must be odd.")

        self.flag = np.asarray(flag, dtype=bool)
        self.N = N = self.flag.size
        self.h = h = length // 2
        self.offset = offset = np.arange(-h, h + 1)
        self.L = length

        # Pad with zeros if the data are not periodic, so windows never wrap
        self.Np = Np = N if periodic else N + 2 * h

        # Eigenmodes of the windowed DAYENU kernel
        delta = (offset[:, np.newaxis] - offset[np.newaxis, :]) * dra
        kernel = np.zeros((length, length), dtype=np.complex128)
        for mc, mw in zip(np.atleast_1d(m_centre), np.atleast_1d(m_width)):
            kernel += np.sinc(mw * delta / np.pi) * np.exp(1.0j * mc * delta)

        lam, U = np.linalg.eigh(kernel)
        keep = lam > 1e-3 * epsilon
        U = U[:, keep]
        ridge = epsilon / lam[keep]
        self.nmode = U.shape[1]

        fpad = np.zeros(Np, dtype=np.float64)
        fpad[:N] = self.flag

        # Index of every sample in every window
        self._index = (np.arange(N)[:, np.newaxis] + offset[np.newaxis, :]) % Np
        fwin = fpad[self._index]

        # Taps for windows without masked samples.  The basis is orthonormal, so
        # the least-squares solution is simple.
        uc = U[h]
        taps_clean = U.conj() @ (uc / (1.0 + ridge))

        self.dirty = np.flatnonzero(self.flag & (fwin.sum(axis=1) < length))

        self.taps = np.zeros((N, length), dtype=np.complex128)
        self.taps[self.flag] = taps_clean
        self.taps_clean = taps_clean

        failed = np.zeros(N, dtype=bool)
        if self.dirty.size:
            z, ok = self._dirty_window_weights(
                U, uc, 1.0 + ridge, fwin[self.dirty] == 0
            )
            self.taps[self.dirty] = (z @ U.conj().T) * fwin[self.dirty]
            self.taps[self.dirty[~ok]] = 0.0
            failed[self.dirty[~ok]] = True

        self.p_diag = np.where(self.flag, self.taps[:, h].real, 0.0)
        self.gain = np.where(self.flag & ~failed, 1.0 - self.p_diag, 0.0)

        # FFT of the clean-window convolution kernel, kappa[-offset] = taps
        kappa = np.zeros(Np, dtype=np.complex128)
        kappa[(-offset) % Np] = taps_clean
        self._kappa_fft = np.fft.fft(kappa)

    @staticmethod
    def _dirty_window_weights(U, uc, d0, masked):
        """Solve for the filter in windows that contain masked samples.

        Returns
        -------
        z : np.ndarray[nwindow, nmode]
            Coefficients of the filter at the centre of each window.
        ok : np.ndarray[nwindow] of bool
            False where the solve failed.
        """
        K = U.shape[1]
        nd = masked.shape[0]

        z = np.zeros((nd, K), dtype=np.complex128)
        ok = np.ones(nd, dtype=bool)

        a = uc / d0
        for ii in range(nd):
            V = U[masked[ii]]
            m = V.shape[0]
            try:
                if m < K:
                    VD = V / d0
                    S = np.eye(m) - VD @ V.conj().T
                    # Fewer masked samples than modes: solve an m x m system instead,
                    # z = a + (a V^H) S^{-1} (V D0^{-1}), using S^T = conj(S)
                    c = np.linalg.solve(S.conj(), a @ V.conj().T)
                    z[ii] = a + c @ VD
                else:
                    M = np.diag(d0).astype(np.complex128) - V.conj().T @ V
                    # z = u_c M^{-1}, solved as conj(M) z = u_c
                    z[ii] = np.linalg.solve(M.conj(), uc)
            except np.linalg.LinAlgError:
                ok[ii] = False

        ok &= np.all(np.isfinite(z), axis=1)
        return z, ok

    def apply(self, data, tile=None):
        """Apply the filter along the last axis of ``data``."""
        y = np.asarray(data, dtype=np.complex128) * self.flag
        N, Np, L, h = self.N, self.Np, self.L, self.h

        ypad = np.zeros((*y.shape[:-1], Np), dtype=np.complex128)
        ypad[..., :N] = y

        foreground = np.fft.ifft(np.fft.fft(ypad, axis=-1) * self._kappa_fft, axis=-1)[
            ..., :N
        ]

        if self.dirty.size:
            tile = L if tile is None else tile
            lead = y.shape[:-1]
            y2 = ypad.reshape(-1, Np)
            fg2 = np.ascontiguousarray(foreground).reshape(-1, N)

            jj = np.arange(L)
            for tt in np.unique(self.dirty // tile):
                t0 = tt * tile
                t1 = min(t0 + tile, N)
                nt = t1 - t0
                cols = np.arange(nt)

                # H[s, t] = taps[t, s - t] maps the input span to the tile
                hmat = np.zeros((nt + L - 1, nt), dtype=np.complex128)
                hmat[cols[:, np.newaxis] + jj, cols[:, np.newaxis]] = self.taps[t0:t1]

                span = np.arange(t0 - h, t1 + h) % Np
                fg2[:, t0:t1] = y2[:, span] @ hmat

            foreground = fg2.reshape((*lead, N))

        return (y - foreground) * self.flag

    def propagate_variance(self, variance):
        """Return the variance of the filtered data given the input variance."""
        vpad = np.zeros(self.Np, dtype=np.float64)
        vpad[: self.N] = np.where(self.flag, variance, 0.0)

        quad = np.sum(np.abs(self.taps) ** 2 * vpad[self._index], axis=1)
        out = vpad[: self.N] * (1.0 - 2.0 * self.p_diag) + quad

        return np.where(self.flag, np.maximum(out, 0.0), 0.0)

    def response(self, max_lag):
        """Return the filter elements ``P[t, t + lag]`` for ``|lag| <= max_lag``.

        Returns
        -------
        resp : np.ndarray[N, 2 * max_lag + 1]
            Rows of samples with zero weight are zero.
        """
        if max_lag > self.h:
            raise ValueError("max_lag cannot exceed the half-width of the window.")
        return self.taps[:, self.h - max_lag : self.h + max_lag + 1].copy()


# -----------------------------------------------------------------------------
# Transient matched filter
# -----------------------------------------------------------------------------


class TransientMatchedFilter(tasklib.base.ContainerTask):
    """Matched filter for transients in m-mode filtered hybrid beamformed visibilities.

    Searches over the start time, duration, telescope-x, and spectral index of a
    transient in each elevation.  The template follows the source as it drifts
    through the primary beam, using the nearest x on the grid at each sample.  For
    visibilities filtered as ``(I - P) y``, the amplitude is estimated as
    ``Re(b^H W (I - P) y) / b^H W (I - P) b``, where ``b`` is the template and ``W``
    the inverse noise variance, using the elements of ``P`` near the diagonal from
    the input.  Frequencies are combined within bands, by default the 6 MHz
    television channels starting at 398 MHz, and the bands are combined for each
    spectral index.

    Attributes
    ----------
    durations : list of int
        Durations in samples.  Default is [1, 2, 4, 8, 16].
    spectral_indices : list of float
        Spectral indices.  Default is [-3, -1.5, 0, 1.5, 3].
    reference_freq : float
        Reference frequency for the spectral index in MHz.  Default is 600.
    band_durations : list of int
        Durations for which the result for each band is also returned.
        Default is [1].
    band_start : float
        Lower edge of the first band in MHz.  Default is 398.
    band_width : float
        Width of each band in MHz.  Default is 6.
    band_edges : list of float
        Explicit band edges in MHz.  Overrides band_start and band_width.
    single_band : bool
        Combine all frequencies into a single band.  Default is False.
    pol : list of str
        Polarisations to use.  Default is ["XX", "YY"].
    combine_pol : bool
        Combine the polarisations.  Default is True.
    ew_index : list of int
        Indices of the east-west baselines to use.  Default is all.
    x_max : float
        Half-width of the telescope-x grid.  Default is ``x_nfwhm`` times the
        half width at half maximum of the primary beam at the lowest frequency.
    x_nfwhm : float
        See ``x_max``.  Default is 1.
    x_spacing : float
        Spacing of the telescope-x grid.  Default is the smaller of a quarter of the
        narrowest primary beam FWHM and ``lambda_min / (x_oversample * max(ew))``.
    x_oversample : float
        See ``x_spacing``.  Default is 8.
    save_imag : bool
        Also save the imaginary part as a null test.  Default is False.
    ignore_rot : bool
        Ignore the telescope rotation.  Default is False.
    nel_chunk : int
        Number of elevations to process at once.  Default is 8.
    """

    durations = config.list_type(int, default=[1, 2, 4, 8, 16])
    spectral_indices = config.list_type(
        (int, float), default=[-3.0, -1.5, 0.0, 1.5, 3.0]
    )
    reference_freq = config.Property(proptype=float, default=600.0)
    band_durations = config.list_type(int, default=[1])
    band_start = config.Property(proptype=float, default=398.0)
    band_width = config.Property(proptype=float, default=6.0)
    band_edges = config.list_type((int, float), default=None)
    single_band = config.Property(proptype=bool, default=False)
    pol = config.list_type(str, default=["XX", "YY"])
    combine_pol = config.Property(proptype=bool, default=True)
    ew_index = config.list_type(int, default=None)
    x_max = config.Property(proptype=float, default=None)
    x_nfwhm = config.Property(proptype=float, default=1.0)
    x_spacing = config.Property(proptype=float, default=None)
    x_oversample = config.Property(proptype=float, default=8.0)
    save_imag = config.Property(proptype=bool, default=False)
    ignore_rot = config.Property(proptype=bool, default=False)
    nel_chunk = config.Property(proptype=int, default=8)

    def setup(self, manager, beam):
        """Set the telescope and primary beam model.

        Parameters
        ----------
        manager : ProductManager, BeamTransfer, or TransitTelescope
            Object from which a telescope instance is extracted.
        beam : draco.core.containers.GridBeam
            Primary beam model in telescope coordinates.
        """
        self.telescope = io.get_telescope(manager)
        self.latitude = np.radians(self.telescope.latitude)

        if self.ignore_rot or not hasattr(self.telescope, "rotation_angle"):
            self.rot = 0.0
        else:
            self.rot = np.radians(self.telescope.rotation_angle)

        if beam.coords != "telescope":
            raise NotImplementedError(
                "TransientMatchedFilter requires a GridBeam in telescope coordinates."
            )
        if beam.input.size > 1:
            raise NotImplementedError("Input-dependent beam models are not supported.")

        self.beam = beam

    def process(self, hvis):
        """Apply the matched filter.

        Parameters
        ----------
        hvis : ch_pipeline.core.containers.MFilteredHybridVisStream
            Output of :class:`LSQMFilterHybridVis`.

        Returns
        -------
        trials : ch_pipeline.core.containers.TransientMatchedFilterTrials
            Result for each duration and spectral index.
        bands : ch_pipeline.core.containers.TransientMatchedFilter
            Result for each band, for the durations in ``band_durations``.
        """
        durations = sorted(set(self.durations))
        band_durations = sorted(set(self.band_durations))
        if not band_durations or not set(band_durations) <= set(durations):
            raise ValueError("band_durations must be a non-empty subset of durations.")
        if min(durations) < 1:
            raise ValueError("Durations must be at least one sample.")
        wmax = max(durations)
        alphas = np.array(self.spectral_indices, dtype=np.float64)

        has_response = "ra_response" in hvis.datasets
        if not has_response and wmax > 1:
            raise ValueError(
                "Durations longer than one sample require the filter response "
                "(the output of LSQMFilterHybridVis without CorrectFilterGain)."
            )
        if has_response:
            lags = list(hvis.index_map["lag"])
            lag_index = {d: ii for ii, d in enumerate(lags)}
            if max(lags) < wmax - 1:
                raise ValueError(
                    f"The filter response extends to lag {max(lags)}, but durations "
                    f"up to {wmax} samples require lag {wmax - 1}."
                )

        # Select polarisations and baselines
        pol = list(hvis.index_map["pol"])
        ipol = [pol.index(p) for p in self.pol]
        groups = (
            [list(range(len(ipol)))]
            if self.combine_pol
            else [[pp] for pp in range(len(ipol))]
        )
        opol = np.array(["co"]) if self.combine_pol else np.array(self.pol)

        ew = hvis.index_map["ew"][:]
        iew = list(range(ew.size)) if self.ew_index is None else list(self.ew_index)

        freq = hvis.freq[:]
        band_index, band_map = self._assign_bands(freq)
        nband = len(band_map)
        scale = (
            band_map["centre"][:, np.newaxis] / self.reference_freq
        ) ** alphas  # band, alpha

        el = hvis.index_map["el"][:]
        ra = hvis.ra[:]
        nra = ra.size
        dra = np.radians(np.median(np.diff(ra)))
        periodic = _is_full_day(ra)

        # Telescope-x grid
        x = self._x_grid(freq, ew[iew])
        nx = x.size
        dx = x[1] - x[0] if nx > 1 else 1.0
        self.log.info(
            f"Using {nx} telescope-x samples spanning +/-{x[-1]:0.4f} (spacing {dx:0.5f})."
        )

        # Primary beam for every frequency, polarisation, elevation, and x,
        # distributed over elevation to match the data
        pbeam = self._evaluate_beam(freq, [pol[ii] for ii in ipol], el, x)

        # Hour angle and declination of every (el, x)
        xx, yy = np.meshgrid(x, el, indexing="xy")
        with np.errstate(invalid="ignore"):
            ha, dec = spherical.ground_to_sph(xx, yy, self.latitude)

        # Noise weights have no elevation axis, so they remain distributed over
        # frequency when the stream is redistributed.  Gather them.
        hvis.redistribute("freq")
        wname = "noise_weight" if has_response else hvis._weight_dset_name
        if "elevation_vis_weight" in hvis.datasets:
            raise ValueError("Elevation-dependent weights are not supported.")
        weight_all = hvis[wname][:].allgather()[ipol][:, :, iew].astype(np.float32)

        # Datasets without an el axis stay distributed over freq
        hvis.redistribute(["el", "freq"])
        vis = hvis.vis[:].local_array
        el_slc = hvis.vis[:].local_bounds
        el_local = el[el_slc]
        nel_local = el_local.size

        # Output containers
        common = {
            "pol": opol,
            "x": x,
            "axes_from": hvis,
            "attrs_from": hvis,
            "distributed": hvis.distributed,
            "comm": hvis.comm,
        }
        trials = containers.TransientMatchedFilterTrials(
            duration=np.array(durations), spectral_index=alphas, **common
        )
        bands = containers.TransientMatchedFilter(
            freq=band_map, duration=np.array(band_durations), **common
        )
        for cont in (trials, bands):
            if self.save_imag:
                cont.add_dataset("amplitude_imag")
            cont.redistribute("el")
            cont.datasets["ha"][:] = np.degrees(ha)
            cont.datasets["dec"][:] = np.degrees(dec)
            cont.attrs["ew_used"] = ew[iew]
            cont.attrs["pol_used"] = [pol[ii] for ii in ipol]
            cont.attrs["duration_unit_seconds"] = 86164.0905 / 360.0 * np.degrees(dra)
            cont.attrs["time_axis"] = "start of the transient"
        trials.attrs["reference_freq"] = self.reference_freq
        bands.attrs["band_start"] = self.band_start
        bands.attrs["band_width"] = self.band_width

        # Accumulate sum s Re N and sum s^2 D in the trial datasets
        tnum = trials.amplitude[:].local_array
        tden = trials.weight[:].local_array
        tnum[:] = 0.0
        tden[:] = 0.0
        tnim = trials.amplitude_imag[:].local_array if self.save_imag else None
        if tnim is not None:
            tnim[:] = 0.0

        bamp = bands.amplitude[:].local_array
        bwgt = bands.weight[:].local_array
        bamp[:] = 0.0
        bwgt[:] = 0.0
        bimag = bands.amplitude_imag[:].local_array if self.save_imag else None

        # Track of a source starting at each (el, x): grid index of x after j samples
        track = self._tracks(ha[el_slc], dec[el_slc], x, dra, wmax)  # [j, el, x]

        lmbda = scipy.constants.c / (freq * 1e6)
        u = ew[iew][np.newaxis, :] / lmbda[:, np.newaxis]  # freq, ew

        for bb in range(nband):
            fsel = np.flatnonzero(band_index == bb)
            wb = weight_all[:, fsel]  # pol, freq, ew, ra

            # Coefficients of the denominator along each lag diagonal:
            # A_0(t) = w_t (1 - P[t, t]) and A_d(t) = -w_t P[t, t + d], as
            # [pol, freq, ew, lag, ra] for lags -(wmax - 1) ... wmax - 1
            lag_coeff = self._lag_coefficients(
                hvis.ra_response if has_response else None,
                lag_index if has_response else None,
                wb,
                fsel,
                ipol,
                iew,
                wmax,
            )

            for e0 in range(0, nel_local, self.nel_chunk):
                esl = slice(e0, min(e0 + self.nel_chunk, nel_local))
                yel = el_local[esl]

                # Template [pol, freq, ew, el, x]
                phase = (
                    2.0
                    * np.pi
                    * u[fsel][:, :, np.newaxis, np.newaxis]
                    * (
                        np.cos(self.rot) * x[np.newaxis, :]
                        + np.sin(self.rot) * yel[:, np.newaxis]
                    )
                )
                template = (
                    pbeam[:, fsel][:, :, np.newaxis, esl] * np.exp(1.0j * phase)
                ).astype(np.complex64)

                data = vis[:, :, :, esl][
                    np.ix_(ipol, fsel, iew)
                ]  # pol, freq, ew, el, ra

                for gg, grp in enumerate(groups):
                    tc = template[grp].reshape(-1, *template.shape[3:])  # k, el, x
                    wk = wb[grp].reshape(-1, nra)  # k, ra
                    dk = data[grp].reshape(-1, *data.shape[3:])  # k, el, ra
                    ak = lag_coeff[grp].reshape(-1, *lag_coeff.shape[3:])  # k, lag, ra

                    # Per-sample numerator [el, x, ra], summed along the tracks
                    n1 = np.matmul(
                        tc.transpose(1, 2, 0).conj(),
                        (wk[:, np.newaxis] * dk).transpose(1, 0, 2),
                    )
                    ntrack = self._track_numerator(n1, track[:, esl], durations)

                    # Denominator along the tracks
                    dtrack = self._track_denominator(tc, ak, track[:, esl], durations)

                    for iw, W in enumerate(durations):
                        numer = ntrack[iw]
                        denom = dtrack[iw]
                        valid = denom > 0.0
                        if not periodic:
                            valid &= (np.arange(nra) <= nra - W)[
                                np.newaxis, np.newaxis, :
                            ]
                        denom = np.where(valid, denom, 0.0)
                        numer = np.where(valid, numer, 0.0)

                        # Output order is [el, ra, x]
                        nre = numer.real.transpose(0, 2, 1)
                        nim = numer.imag.transpose(0, 2, 1)
                        den = denom.transpose(0, 2, 1)

                        for ia in range(alphas.size):
                            sb = scale[bb, ia]
                            tnum[gg, iw, ia, esl] += sb * nre
                            tden[gg, iw, ia, esl] += sb**2 * den
                            if tnim is not None:
                                tnim[gg, iw, ia, esl] += sb * nim

                        if W in band_durations:
                            ib = band_durations.index(W)
                            inv = tools.invert_no_zero(den)
                            bamp[gg, ib, bb, esl] = nre * inv
                            bwgt[gg, ib, bb, esl] = 2.0 * den
                            if bimag is not None:
                                bimag[gg, ib, bb, esl] = nim * inv

        # Convert the accumulated sums into amplitudes and weights
        inv = tools.invert_no_zero(tden)
        if tnim is not None:
            tnim *= inv
        tnum *= inv
        tden *= 2.0

        return trials, bands

    # ------------------------------------------------------------------
    # Matched filter helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _tracks(ha, dec, x, dra, wmax):
        """Index of the x grid along the track of a source starting at each (el, x).

        Returns
        -------
        track : np.ndarray[wmax, el, x] of int
            Index ``j`` samples after the start, or -1 if the source has left the grid.
        """
        dx = x[1] - x[0] if x.size > 1 else 1.0
        j = np.arange(wmax)[:, np.newaxis, np.newaxis]
        with np.errstate(invalid="ignore"):
            xj = -np.cos(dec)[np.newaxis] * np.sin(ha[np.newaxis] + j * dra)
            idx = np.rint((xj - x[0]) / dx)
        ok = np.isfinite(idx) & (idx >= 0) & (idx < x.size)
        return np.where(ok, idx, -1).astype(np.int64)

    @staticmethod
    def _track_numerator(n1, track, durations):
        """Sum the numerator along the source tracks.

        Parameters
        ----------
        n1 : np.ndarray[el, x, ra]
            Numerator for each sample, evaluated with the template at each grid x.
        track : np.ndarray[wmax, el, x]
            From :meth:`_tracks`.
        durations : list of int
            Durations in samples.

        Returns
        -------
        ntrack : list of np.ndarray[el, x, ra]
            Numerator for each duration, indexed by the start of the transient.
        """
        nacc = np.zeros(n1.shape, dtype=np.complex128)
        ntrack = []
        wset = set(durations)

        for jj in range(max(durations)):
            idx = track[jj]
            ok = (idx >= 0)[..., np.newaxis]
            nj = np.take_along_axis(n1, np.maximum(idx, 0)[..., np.newaxis], axis=1)
            # The statistic for a transient starting at tau uses sample tau + j
            nacc += np.where(ok, np.roll(nj, -jj, axis=-1), 0.0)
            if jj + 1 in wset:
                ntrack.append(nacc.copy())

        return ntrack

    @staticmethod
    def _track_denominator(tc, ak, track, durations):
        """Sum the denominator along the source tracks.

        ``D(tau) = sum_k sum_{j, j' < W} conj(g_kj) g_kj' A_k,(j' - j)(tau + j)``,
        where ``g_kj`` is the template of visibility ``k`` at the grid x reached after
        ``j`` samples, and ``A`` is from :meth:`_lag_coefficients`.

        Parameters
        ----------
        tc : np.ndarray[k, el, x]
            Template at each grid x.
        ak : np.ndarray[k, lag, ra]
            From :meth:`_lag_coefficients`.
        track : np.ndarray[wmax, el, x]
            From :meth:`_tracks`.
        durations : list of int
            Durations in samples.

        Returns
        -------
        dtrack : list of np.ndarray[el, x, ra]
            Denominator for each duration, indexed by the start of the transient.
        """
        wmax = max(durations)
        nk, nel, nx = tc.shape
        nra = ak.shape[-1]
        lag0 = (ak.shape[1] - 1) // 2

        # Template along the track at each step [j][k, el * x]
        g = []
        for jj in range(wmax):
            idx = track[jj]
            gj = np.take_along_axis(tc, np.maximum(idx, 0)[np.newaxis], axis=2)
            g.append(np.where(idx >= 0, gj, 0.0).reshape(nk, -1).astype(np.complex128))

        acc = np.zeros((nel * nx, nra), dtype=np.complex128)
        dtrack = []
        wset = set(durations)

        for W in range(1, wmax + 1):
            jn = W - 1
            pairs = [(jn, jp) for jp in range(W)] + [(jj, jn) for jj in range(W - 1)]
            for jj, jp in pairs:
                coeff = g[jj].conj() * g[jp]  # k, el * x
                lag_ra = np.roll(ak[:, lag0 + jp - jj], -jj, axis=-1)  # A_d(tau + j)
                acc += coeff.T @ lag_ra
            if W in wset:
                dtrack.append(acc.real.reshape(nel, nx, nra).copy())

        return dtrack

    def _gather_freq(self, dset, fsel):
        """Gather a band of frequencies of a frequency-distributed dataset."""
        arr = dset[:]
        lo = arr.local_offset[1]
        nloc = arr.local_shape[1]
        local = [f for f in fsel if lo <= f < lo + nloc]
        part = arr.local_array[:, [f - lo for f in local]]
        pieces = self.comm.allgather((local, part))
        found = {
            f: piece[:, ii] for flist, piece in pieces for ii, f in enumerate(flist)
        }
        return np.stack([found[f] for f in fsel], axis=1)

    def _lag_coefficients(self, response, lag_index, wb, fsel, ipol, iew, wmax):
        """Return ``w_t (delta_d0 - P[t, t + d])`` for ``|d| < wmax``.

        Without a filter response, ``P = 0``.

        Returns
        -------
        coeff : np.ndarray[pol, freq, ew, lag, ra]
        """
        npol, nf, ne, nra = wb.shape
        coeff = np.zeros((npol, nf, ne, 2 * wmax - 1, nra), dtype=np.complex64)
        coeff[:, :, :, wmax - 1] = wb

        if response is not None:
            resp = self._gather_freq(response, fsel)[ipol][
                :, :, iew
            ]  # pol, freq, ew, ra, lag
            for d in range(-(wmax - 1), wmax):
                coeff[:, :, :, wmax - 1 + d] -= wb * resp[..., lag_index[d]]

        return coeff

    def _assign_bands(self, freq):
        """Assign each frequency channel to a band.

        Returns
        -------
        band_index : np.ndarray[nfreq]
            Band of each channel, or -1 if it is in none.
        band_map : np.ndarray[nband]
            Structured array with the centre and width of each populated band.
        """
        if self.single_band:
            df = np.median(np.abs(np.diff(freq))) if freq.size > 1 else 0.0
            edges = np.array([freq.min() - 0.5 * df, freq.max() + 0.5 * df])
        elif self.band_edges is not None:
            edges = np.sort(np.array(self.band_edges, dtype=np.float64))
        else:
            nedge = int(np.ceil((freq.max() - self.band_start) / self.band_width)) + 1
            edges = self.band_start + self.band_width * np.arange(max(nedge, 2))

        raw = np.searchsorted(edges, freq, side="right") - 1
        raw[(raw < 0) | (raw >= edges.size - 1) | (freq == edges[-1])] = -1

        # Drop bands without channels, preserving order of decreasing frequency
        # to match the CHIME frequency axis.
        used = np.unique(raw[raw >= 0])[::-1]
        band_index = np.full(freq.size, -1, dtype=int)
        for bb, rb in enumerate(used):
            band_index[raw == rb] = bb

        band_map = np.zeros(
            used.size, dtype=[("centre", np.float64), ("width", np.float64)]
        )
        band_map["centre"] = 0.5 * (edges[used] + edges[used + 1])
        band_map["width"] = edges[used + 1] - edges[used]

        return band_index, band_map

    def _beam_profile_hwhm(self, freq, pol):
        """Half width at half maximum of the beam in x at zenith, per frequency."""
        beam = self.beam
        beam.redistribute("freq")

        bfreq_all = beam.freq[:]
        lo = beam.beam.local_offset[0]
        nloc = beam.beam.local_shape[0]
        bfreq = bfreq_all[lo : lo + nloc]

        bpol = list(beam.pol)
        ipol = [bpol.index(p) for p in pol]

        phi = np.asarray(beam.phi)
        isort = np.argsort(phi)
        phi = phi[isort]
        itheta = np.argmin(np.abs(np.asarray(beam.theta)))

        hwhm = []
        for ff, nu in enumerate(bfreq):
            if np.min(np.abs(freq - nu)) > 1e-3:
                continue
            b = beam.beam[:].local_array[ff, ipol, 0, itheta].real
            w = beam.weight[:].local_array[ff, ipol, 0, itheta]
            if not np.all(np.any(w > 0, axis=-1)):
                continue
            prof = np.mean(b, axis=0)[isort]
            pk = prof.max()
            if pk <= 0:
                continue
            above = phi[prof >= 0.5 * pk]
            hwhm.append(0.5 * (above.max() - above.min()))

        allh = [h for rank in self.comm.allgather(hwhm) for h in rank]
        if not allh:
            raise RuntimeError("Could not determine the primary beam width.")

        return np.min(allh), np.max(allh)

    def _x_grid(self, freq, ew_used):
        """Construct the grid of telescope-x."""
        if self.x_max is not None and self.x_spacing is not None:
            hw_min = hw_max = None
        else:
            hw_min, hw_max = self._beam_profile_hwhm(freq, self.pol)

        x_max = self.x_max if self.x_max is not None else self.x_nfwhm * hw_max

        if self.x_spacing is not None:
            dx = self.x_spacing
        else:
            dx = 0.5 * hw_min
            dmax = np.max(np.abs(ew_used)) if len(ew_used) else 0.0
            if dmax > 0.0:
                lmbda_min = scipy.constants.c / (freq.max() * 1e6)
                dx = min(dx, lmbda_min / (self.x_oversample * dmax))

        n = int(np.ceil(x_max / dx - 1e-6))
        return dx * np.arange(-n, n + 1)

    def _evaluate_beam(self, freq, pol, y, x, pad=8):
        """Evaluate the co-polar primary beam, distributed over y.

        Returns
        -------
        pbeam : np.ndarray[pol, freq, y_local, x]
            Primary beam for the local elevations of the default distribution.
        """
        beam = self.beam
        beam.redistribute("freq")

        bfreq_all = beam.freq[:]
        lo = beam.beam.local_offset[0]
        nloc = beam.beam.local_shape[0]

        # Match each data frequency to a beam frequency
        bindex = np.array([np.argmin(np.abs(bfreq_all - nu)) for nu in freq])
        if np.any(np.abs(bfreq_all[bindex] - freq) > 1e-3):
            raise ValueError("The beam model does not contain all data frequencies.")
        if np.any(np.diff(bindex) <= 0) and freq.size > 1:
            raise ValueError("Data and beam frequencies must be ordered consistently.")

        local = np.flatnonzero((bindex >= lo) & (bindex < lo + nloc))

        bpol = list(beam.pol)
        ipol = [bpol.index(p) for p in pol]

        theta = np.asarray(beam.theta)
        phi = np.asarray(beam.phi)
        isort = np.argsort(phi)
        phi = phi[isort]

        if np.any(np.diff(theta) <= 0):
            raise ValueError("Beam theta axis must be increasing.")

        # Restrict the splines to the x range required
        i0 = max(int(np.searchsorted(phi, x.min(), side="right")) - 1 - pad, 0)
        i1 = min(int(np.searchsorted(phi, x.max(), side="left")) + pad + 1, phi.size)
        xsel = isort[i0:i1]
        phi = phi[i0:i1]

        bdset = beam.beam[:].local_array
        wdset = beam.weight[:].local_array

        pb = np.zeros((local.size, len(pol), y.size, x.size), dtype=np.float32)
        for ii, ff in enumerate(local):
            lf = bindex[ff] - lo
            for pp, ip in enumerate(ipol):
                b = bdset[lf, ip, 0][:, xsel]
                w = wdset[lf, ip, 0][:, xsel]

                # A zero-weight point is invalid only if its beam is non-zero
                flag = (w > 0.0) | (np.abs(b) == 0.0)
                if not np.any(w > 0.0):
                    continue

                bval = np.where(flag, b.real, 0.0)
                val = scipy.interpolate.RectBivariateSpline(theta, phi, bval)(y, x)

                if not np.all(flag):
                    fval = scipy.interpolate.RectBivariateSpline(
                        theta, phi, flag.astype(np.float32)
                    )(y, x)
                    val = np.where(np.abs(fval - 1.0) < 0.01, val, 0.0)

                pb[ii, pp] = val

        # Redistribute from frequency to elevation
        pb = mpiarray.MPIArray.wrap(pb, axis=0, comm=self.comm).redistribute(axis=2)

        return np.ascontiguousarray(pb.local_array.transpose(1, 0, 2, 3))


def collapse_bands(numer, denom, scale):
    """Combine the numerator and denominator of each band for each spectral index.

    Parameters
    ----------
    numer, denom : np.ndarray[pol, duration, band, ...]
        Real part of the numerator and the denominator of each band.
    scale : np.ndarray[band, alpha]
        ``(nu_b / nu_ref)^alpha``.

    Returns
    -------
    amp, weight : np.ndarray[pol, duration, alpha, ...]
        Amplitude at the reference frequency and its inverse variance.
    """
    s = np.moveaxis(scale, 0, -1)  # alpha, band
    num = np.einsum("ab,pwb...->pwa...", s, numer)
    den = np.einsum("ab,pwb...->pwa...", s**2, denom)
    return num * tools.invert_no_zero(den), 2.0 * den


class CollapseMatchedFilterBands(tasklib.base.ContainerTask):
    """Combine the bands of a TransientMatchedFilter for a set of spectral indices.

    Attributes
    ----------
    spectral_indices : list of float
        Spectral indices.  Default is [-3, -1.5, 0, 1.5, 3].
    reference_freq : float
        Reference frequency in MHz.  Default is 600.
    exclude_freq_ranges : list of [float, float]
        Exclude bands with centres in these ranges, in MHz.
    """

    spectral_indices = config.list_type(
        (int, float), default=[-3.0, -1.5, 0.0, 1.5, 3.0]
    )
    reference_freq = config.Property(proptype=float, default=600.0)
    exclude_freq_ranges = config.Property(proptype=list, default=[])

    def process(self, mf):
        """Collapse over bands.

        Parameters
        ----------
        mf : ch_pipeline.core.containers.TransientMatchedFilter
            Matched filter output for each band.

        Returns
        -------
        out : ch_pipeline.core.containers.TransientMatchedFilterTrials
            Matched filter output for each spectral index.
        """
        mf.redistribute("el")

        centre = mf.index_map["freq"]["centre"][:]

        include = np.ones(centre.size, dtype=bool)
        for rng in self.exclude_freq_ranges:
            include &= ~((centre >= min(rng)) & (centre <= max(rng)))

        if not np.any(include):
            raise ValueError("All bands have been excluded.")

        alphas = np.array(self.spectral_indices, dtype=np.float64)
        scale = (centre[:, np.newaxis] / self.reference_freq) ** alphas
        scale = np.where(include[:, np.newaxis], scale, 0.0)

        out = containers.TransientMatchedFilterTrials(
            spectral_index=alphas,
            axes_from=mf,
            attrs_from=mf,
            distributed=mf.distributed,
            comm=mf.comm,
        )
        save_imag = "amplitude_imag" in mf.datasets
        if save_imag:
            out.add_dataset("amplitude_imag")
        out.redistribute("el")

        out.datasets["ha"][:] = mf.datasets["ha"][:]
        out.datasets["dec"][:] = mf.datasets["dec"][:]

        amp = mf.amplitude[:].local_array
        wgt = mf.weight[:].local_array

        # Re N_b = A_b w_b / 2 and D_b = w_b / 2
        oamp, owgt = collapse_bands(0.5 * wgt * amp, 0.5 * wgt, scale)
        out.amplitude[:].local_array[:] = oamp
        out.weight[:].local_array[:] = owgt
        if save_imag:
            oimag, _ = collapse_bands(
                0.5 * wgt * mf.amplitude_imag[:].local_array, 0.5 * wgt, scale
            )
            out.amplitude_imag[:].local_array[:] = oimag

        out.attrs["reference_freq"] = self.reference_freq
        out.attrs["bands_included"] = centre[include]

        return out


class MaxMatchedFilterTrials(tasklib.base.ContainerTask):
    """Find the trial with the highest signal-to-noise ratio.

    The maximum is over duration, spectral index, and telescope-x, for each
    elevation and start time.
    """

    def process(self, trials):
        """Reduce over trials.

        Parameters
        ----------
        trials : ch_pipeline.core.containers.TransientMatchedFilterTrials
            Matched filter output.

        Returns
        -------
        out : ch_pipeline.core.containers.TransientCandidateMap
            Best signal-to-noise ratio and the corresponding trial parameters.
        """
        trials.redistribute("el")

        amp = trials.amplitude[:].local_array  # pol, dur, alpha, el, ra, x
        wgt = trials.weight[:].local_array
        snr = amp * np.sqrt(wgt)

        npol, ndur, nalpha, nel, nra, nx = snr.shape
        # [pol, el, ra, dur, alpha, x], flattened over the trial parameters
        flat = np.moveaxis(snr, (1, 2), (3, 4)).reshape(npol, nel, nra, -1)
        best = np.argmax(flat, axis=-1)
        idur, ialpha, ix = np.unravel_index(best, (ndur, nalpha, nx))

        out = containers.TransientCandidateMap(
            axes_from=trials,
            attrs_from=trials,
            distributed=trials.distributed,
            comm=trials.comm,
        )
        out.redistribute("el")

        pp, ee, rr = np.meshgrid(
            np.arange(npol), np.arange(nel), np.arange(nra), indexing="ij"
        )
        out.snr[:].local_array[:] = np.take_along_axis(
            flat, best[..., np.newaxis], axis=-1
        )[..., 0]
        out.amplitude[:].local_array[:] = amp[pp, idur, ialpha, ee, rr, ix]
        out.weight[:].local_array[:] = wgt[pp, idur, ialpha, ee, rr, ix]
        out.datasets["duration"][:].local_array[:] = trials.index_map["duration"][idur]
        out.datasets["spectral_index"][:].local_array[:] = trials.index_map[
            "spectral_index"
        ][ialpha]
        out.datasets["x_index"][:].local_array[:] = ix
        out.datasets["ha"][:] = trials.datasets["ha"][:]
        out.datasets["dec"][:] = trials.datasets["dec"][:]

        return out
