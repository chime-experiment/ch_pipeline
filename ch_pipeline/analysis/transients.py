"""Tasks for the search for long period transients in hybrid beamformed data."""

import inspect
import json
import time
from typing import ClassVar

import numpy as np
import scipy.constants
import scipy.interpolate
import scipy.ndimage
from caput import config, mpiarray
from caput.algorithms import median
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

        # The DAYENU covariance uses unwrapped RA differences
        stream.attrs["ra_periodic"] = False

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
    wrap_ra : bool
        Treat a full sidereal day as periodic, with the last sample next to the
        first.  The two ends of a single day were observed 24 hours apart, so the
        data generally jump there, and wrapping spreads that jump into the filtered
        data around the boundary.  Required by the full-day fit, which is periodic
        by construction.  Default is False.
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
    wrap_ra = config.Property(proptype=bool, default=False)
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
        full_day = _is_full_day(ra)
        if self.fir_length == 0 and not (full_day and self.wrap_ra):
            raise ValueError(
                "The global filter requires a full sidereal day and wrap_ra = True. "
                "Set fir_length to filter a partial day or without wrapping."
            )
        periodic = full_day and self.wrap_ra
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
        # Record whether the filter treated the RA axis as periodic, so that the
        # matched filter does not combine samples across the boundary otherwise
        out.attrs["ra_periodic"] = periodic

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
                    # Built for each polarisation below; None if no data
                    filt = None

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

                if filt is not None:
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


class SubtractScintillation(tasklib.base.ContainerTask):
    """Fit and subtract the scintillation of the foregrounds from filtered data.

    Ionospheric scintillation modulates the brightness of compact sources from one
    sample to the next by a small fraction ``g(t, freq)``, which is the same on every
    baseline and smooth in frequency.  The m-mode filter cannot remove it, so the
    filtered data contain ``g f``, where ``f`` is the foreground.  This task fits

        d(freq) = f_hat(freq) * sum_j a_j phi_j(freq)

    at each polarisation, east-west baseline, elevation and time, and subtracts it.
    The spectral basis is ``phi_0 = (freq / nu0)**beta`` and, if ``nparam`` is 2,
    ``phi_1 = phi_0 ln(freq / nu0)``, which follows changes of the spectral index.
    The second term removes a little more of the scintillation (99% of its variance
    instead of 97% next to bright sources), but a complex ``a_0 + a_1 ln(freq)``
    also follows the slow phase ramp across the band of a transient anywhere in the
    beam on a short baseline, so it removes much more of the transients.
    The ``a_j`` are complex, since the filter turns part of the modulation into
    quadrature, and have a Gaussian prior, so that nothing is subtracted where the
    scintillation of the foreground is below the noise.  A transient is removed in
    proportion to ``rho^2 q / (1 + q)``, where ``rho`` is its overlap with the
    template and ``q`` the prior variance times the weighted power of the template,
    so the prior should only be large where the foreground really scintillates.
    Diffuse emission does not scintillate, but is bright, so a fixed prior would
    remove transients near the meridian of most rows.  With ``prior_mode`` "block"
    the prior variance is instead measured from the data, separately for each
    polarisation group, baseline, elevation and block of RA: it is the variance of
    the unregularised amplitudes in excess of their noise (weighted for the
    samples where the template is bright, and clipped at ``prior_clip`` times the
    model), kept between ``prior_floor`` and ``prior_cap``.

    The foreground estimate is the low-pass part of the data at sample ``t`` from a
    fit that leaves out sample ``t`` itself,

        f_hat = v - d / (1 - P[t, t]),

    where ``v`` is the unfiltered and ``d`` the filtered visibility.  This is exact
    for the least-squares m-mode filters, so a transient lasting one sample is not
    in its own template, and is only removed to the extent that it resembles the
    template.  Longer transients leak into the template of their other samples.

    The fit is weighted by the inverse variance of the filtered data, plus the
    self-noise of the foreground, ``self_noise |f_hat|^2 / (dnu tau)`` for channel
    width ``dnu`` and sample spacing ``tau``.

    The unfiltered data contain everything in the beam, including diffuse emission
    and sources that do not scintillate, so ``f_hat`` is the wrong template unless one
    compact source dominates.  With ``template`` "neighbours", the template of each
    sample is instead learned from the filtered data of its neighbours, which only
    contain what scintillates: since the modulation is white from one sample to the
    next and the foreground changes slowly, the filtered data near ``t`` are copies
    of one shape with random amplitudes.  Starting from ``phi_0 f_hat``, the
    amplitudes ``a(t')`` are fit at every sample and the template is rebuilt as

        e(t) = sum_t' conj(a(t')) R(t, t') d(t') / sum_t' |a(t')|^2,

    channel by channel, over ``gap < |t' - t| <= window``, where ``R`` rotates the
    fringe of the row's meridian from ``t'`` to ``t``; this is repeated ``niter``
    times (power iteration towards the leading singular vector).  Leaving out the
    samples within ``gap`` keeps a transient at ``t``, and its ringing in the
    filtered data of its neighbours, out of its own template.  The amplitudes used
    as weights are clipped at ``clip`` times their typical value in the row.  A
    bright transient would still reach the templates of its neighbours through its
    ringing in their filtered data (which extends beyond ``gap``) and through its
    neighbours' starting templates.  This only matters for transients much
    brighter than the local scintillation.  Scintillation can only be a small
    fraction of the foreground, while a transient can be any size, so samples
    whose filtered data, in excess of the noise, have more than ``flag_fraction``
    of the weighted amplitude of the starting template are flagged, and neither
    they nor the samples within ``ring`` of them are used to build templates.  (A test of how well the starting
    template explains the data would also flag the scintillation spikes in rows
    where that template is wrong, which are what the template has to learn.)

    Attributes
    ----------
    beta : float
        Spectral index of the modulation.  Default is -1.8.
    nu0 : float
        Reference frequency in MHz.  Default is 600.
    nparam : int
        Number of spectral basis functions, 1 or 2.  Default is 1.
    prior_mode : str
        "block" (default) to measure the prior in blocks of RA, or "fixed" to use
        ``prior_sigma``.
    prior_sigma : list of float
        Width of the prior on each amplitude (one value for all, or one per basis
        function), and the starting point of the block estimate.  Default is 0.01.
    nblock : int
        Number of blocks of RA for the block prior.  Default is 48.
    prior_floor, prior_cap : float
        Range of the block prior width.  Defaults are 1e-5 and 0.03.
    prior_clip : float
        Clip each squared amplitude at this many times its modelled variance in
        the block estimate.  Default is 20.
    self_noise : float
        Coefficient of the self-noise term in the weights.  Zero ignores it.
        Default is 1.
    min_gain : float
        Leave samples whose filter gain ``1 - P[t, t]`` is lower unchanged.
        Default is 0.1.
    combine_pol : bool
        Fit the polarisations together with one set of amplitudes, since the
        scintillation is the same in both.  Default is True.
    subtract : bool
        Subtract the fit.  If False, only fit, for diagnostics.  Default is True.
    in_place : bool
        Modify the filtered stream in place.  Default is True.
    nrow_chunk : int
        Number of elevations to process at once.  Default is 64.
    template : str
        "foreground" (default) for ``phi f_hat``, or "neighbours" to learn the
        template from the filtered data of the neighbouring samples (one term only).
    window, gap : int
        Neighbours used for the "neighbours" template.  Defaults are 20 and 5.
    niter : int
        Iterations of the "neighbours" template.  Default is 2.
    clip : float
        Clip the amplitudes used to build the "neighbours" template at this many
        times the square root of their median power in the row.  Default is 5.
    flag_fraction : float
        Flag samples whose filtered data, in excess of the noise (whose level is
        the median of the row), are larger than this fraction of the starting
        template, ``sum w |d|^2 - noise > flag_fraction^2 sum w |phi_0 f_hat|^2``, and
        more than 5 sigma above the noise.  Default is 0.1, well above the largest
        scintillation seen (1-2% rms).
    ring : int
        Samples within this of a flagged sample are not used to build templates.
        Default is the half-length of the stored filter response.
    """

    beta = config.Property(proptype=float, default=-1.8)
    nu0 = config.Property(proptype=float, default=600.0)
    nparam = config.Property(proptype=int, default=1)
    prior_mode = config.enum(["block", "fixed"], default="block")
    prior_sigma = config.list_type(float, default=[0.01])
    nblock = config.Property(proptype=int, default=48)
    prior_floor = config.Property(proptype=float, default=1e-5)
    prior_cap = config.Property(proptype=float, default=0.03)
    prior_clip = config.Property(proptype=float, default=20.0)
    self_noise = config.Property(proptype=float, default=1.0)
    min_gain = config.Property(proptype=float, default=0.1)
    combine_pol = config.Property(proptype=bool, default=True)
    subtract = config.Property(proptype=bool, default=True)
    in_place = config.Property(proptype=bool, default=True)
    nrow_chunk = config.Property(proptype=int, default=64)
    template = config.enum(["foreground", "neighbours"], default="foreground")
    window = config.Property(proptype=int, default=20)
    gap = config.Property(proptype=int, default=5)
    niter = config.Property(proptype=int, default=2)
    clip = config.Property(proptype=float, default=5.0)
    flag_fraction = config.Property(proptype=float, default=0.1)
    ring = config.Property(proptype=int, default=None)

    def process(self, stream, unfiltered):
        """Fit and subtract the scintillation.

        Parameters
        ----------
        stream : ch_pipeline.core.containers.MFilteredHybridVisStream
            Output of :class:`LSQMFilterHybridVis`, with the filter response and
            without the gain correction.
        unfiltered : draco.core.containers.HybridVisStream
            The input to the filter.

        Returns
        -------
        stream : ch_pipeline.core.containers.MFilteredHybridVisStream
            The filtered visibilities with the fit subtracted.
        fit : ch_pipeline.core.containers.HybridVisScintillationFit
            The fitted amplitudes.
        """
        if "ra_response" not in stream.datasets:
            raise ValueError("SubtractScintillation needs the filter response.")
        if stream.attrs.get("filter_gain_corrected", False):
            raise ValueError(
                "SubtractScintillation needs the uncorrected filter output."
            )
        if self.nparam not in (1, 2):
            raise ValueError("nparam must be 1 or 2.")
        if self.template == "neighbours" and self.nparam != 1:
            raise ValueError("The neighbours template has a single term (nparam = 1).")
        if self.template == "neighbours" and not 0 <= self.gap < self.window:
            raise ValueError("Need 0 <= gap < window.")
        _check_same_grid(stream, unfiltered)

        if not self.in_place:
            stream = stream.copy()
        stream.redistribute("freq")
        unfiltered.redistribute("freq")

        nparam = self.nparam
        sigma = np.array(self.prior_sigma, dtype=np.float64)
        if sigma.size == 1:
            sigma = np.repeat(sigma, nparam)
        if sigma.size != nparam or np.any(sigma <= 0.0):
            raise ValueError(
                "prior_sigma needs one positive value or one per parameter."
            )

        # Match the polarisations and baselines of the unfiltered stream
        pol = [str(p) for p in stream.index_map["pol"]]
        upol = [str(p) for p in unfiltered.index_map["pol"]]
        ipol_u = [upol.index(p) for p in pol]
        ew = stream.index_map["ew"][:]
        uew = unfiltered.index_map["ew"][:]
        iew_u = [int(np.argmin(np.abs(uew - b))) for b in ew]
        if not np.allclose(uew[iew_u], ew):
            raise ValueError("The east-west baselines of the streams differ.")

        el = stream.index_map["el"][:]
        nel = el.size
        npol, new = len(pol), ew.size
        nfit = 1 if self.combine_pol else npol

        # Spectral basis and self-noise coefficient for the local frequencies
        fslc = stream.vis[:].local_bounds
        freq = stream.freq[fslc]
        width = stream.index_map["freq"]["width"][fslc]
        ra = stream.ra[:]
        tau = np.median(np.diff(ra)) / 360.0 * 86164.0905
        lnu = np.log(freq / self.nu0)
        phi = np.array([np.exp(self.beta * lnu), np.exp(self.beta * lnu) * lnu])[
            :nparam
        ]
        cself = self.self_noise / (width * 1e6 * tau)

        lag = list(stream.index_map["lag"])
        self._ring = max(lag) if self.ring is None else self.ring
        gain = 1.0 - stream.ra_response[:].local_array[..., lag.index(0)].real
        weight = stream.weight[:].local_array  # pol, freq, ew, ra
        uweight = unfiltered.weight[:].local_array
        vis = stream.vis[:].local_array  # pol, freq, ew, el, ra
        uvis = unfiltered.vis[:].local_array

        valid = (
            (weight > 0)
            & (uweight[ipol_u][:, :, iew_u] > 0)
            & (gain > max(self.min_gain, 0.0))
        )  # pol, freq, ew, ra
        inv_gain = np.where(valid, tools.invert_no_zero(gain), 0.0)

        fit = containers.HybridVisScintillationFit(
            axes_from=stream,
            attrs_from=stream,
            param=np.array(["amplitude", "slope"][:nparam]),
            distributed=stream.distributed,
            comm=stream.comm,
        )
        fit.redistribute("el")
        fit.attrs["beta"] = self.beta
        fit.attrs["nu0"] = self.nu0
        fit.attrs["prior_sigma"] = sigma
        fit.attrs["prior_mode"] = self.prior_mode
        fit.add_dataset("prior_sigma")
        psig_out = fit.datasets["prior_sigma"][:].local_array
        # Block of RA of each sample, for the block prior
        nblock = max(min(self.nblock, ra.size), 1)
        blk = (np.arange(ra.size) * nblock) // ra.size
        blk_start = np.flatnonzero(np.r_[True, np.diff(blk) > 0])
        fit.attrs["combine_pol"] = self.combine_pol
        fit.attrs["template"] = self.template
        fit.attrs["subtracted"] = self.subtract
        amp_out = fit.amplitude[:].local_array
        pow_out = fit.template_power[:].local_array
        e_lo = fit.amplitude[:].local_offset[2]
        e_hi = e_lo + amp_out.shape[2]

        def template(pp, ff, esl):
            """Leave-one-out foreground and fit weights for one pol and frequency."""
            ok = valid[pp, ff][:, np.newaxis, :]  # ew, 1, ra
            d = vis[pp, ff, :, esl]
            v = uvis[ipol_u[pp], ff, iew_u, esl]
            fh = np.where(ok, v - d * inv_gain[pp, ff][:, np.newaxis, :], 0.0)
            w = weight[pp, ff][:, np.newaxis, :].astype(np.float64)
            wt = np.where(ok, w / (1.0 + w * cself[ff] * np.abs(fh) ** 2), 0.0)
            return fh, d, wt

        def allreduce(*arrays):
            if stream.distributed and stream.comm.size > 1:
                from mpi4py import MPI

                for arr in arrays:
                    stream.comm.Allreduce(MPI.IN_PLACE, arr, op=MPI.SUM)

        nfreq_local = vis.shape[1]
        nra = ra.size
        if self.template == "neighbours":
            from ch_ephem.observers import chime

            # Fringe rate of the meridian of each row: a source near the meridian at
            # declination dec has x = -cos(dec) ha, so the visibility at t + lag is
            # rotated to t by exp(2 pi i u cos(dec) lag dra)
            cos_dec = np.cos(np.arcsin(np.clip(el, -1, 1)) + np.radians(chime.latitude))
            dra = np.radians(np.median(np.diff(ra)))
            u_bf = ew[:, np.newaxis] * freq[np.newaxis, :] * 1e6 / scipy.constants.c
            lags = [
                lag
                for lag in range(-self.window, self.window + 1)
                if abs(lag) > self.gap
            ]

        def neighbour_amplitudes(e_chunk, esl):
            """Unregularised amplitudes of the current templates, clipped."""
            ne = esl.stop - esl.start
            G1 = np.zeros((nfit, new, ne, nra), dtype=np.float64)
            B1 = np.zeros((nfit, new, ne, nra), dtype=np.complex128)
            for ff in range(nfreq_local):
                for pp in range(npol):
                    _, d, wt = template(pp, ff, esl)
                    ef = e_chunk[pp, ff]
                    pf = 0 if self.combine_pol else pp
                    G1[pf] += wt * np.abs(ef) ** 2
                    B1[pf] += wt * ef.conj() * d
            allreduce(G1, B1)
            return G1, B1

        def data_power(esl):
            """Weighted power of the filtered data, and its expectation for noise."""
            ne = esl.stop - esl.start
            pw = np.zeros((nfit, new, ne, nra), dtype=np.float64)
            npw = np.zeros((nfit, new, ne, nra), dtype=np.float64)
            for ff in range(nfreq_local):
                for pp in range(npol):
                    _, d, wt = template(pp, ff, esl)
                    pf = 0 if self.combine_pol else pp
                    pw[pf] += wt * np.abs(d) ** 2
                    # E|n|^2 = 1 / w, so the weighted noise power is wt / w
                    w = weight[pp, ff][:, np.newaxis, :].astype(np.float64)
                    npw[pf] += np.where(wt > 0, wt / np.where(w > 0, w, 1.0), 0.0)
            allreduce(pw, npw)
            return pw, npw

        def rebuild(a, esl, use):
            """Template of each sample from its neighbours, channel by channel.

            Only the samples where ``use`` is True contribute.
            """
            ne = esl.stop - esl.start
            e_chunk = np.zeros((npol, nfreq_local, new, ne, nra), dtype=np.complex64)
            for ff in range(nfreq_local):
                for pp in range(npol):
                    d = vis[pp, ff, :, esl]
                    ok = valid[pp, ff][:, np.newaxis, :]  # ew, 1, ra
                    pf = 0 if self.combine_pol else pp
                    ap = np.where(use[pf], a[pf], 0.0)  # ew, el, ra
                    num = np.zeros(d.shape, dtype=np.complex128)
                    den = np.zeros(d.shape, dtype=np.float64)
                    for lag in lags:
                        lo, hi = max(0, -lag), min(nra, nra - lag)
                        src = slice(lo + lag, hi + lag)
                        rot = np.exp(
                            2j
                            * np.pi
                            * u_bf[:, ff, np.newaxis]
                            * cos_dec[np.newaxis, esl]
                            * lag
                            * dra
                        )[..., np.newaxis]
                        num[..., lo:hi] += ap[..., src].conj() * rot * d[..., src]
                        den[..., lo:hi] += np.abs(ap[..., src]) ** 2 * ok[..., src]
                    e_chunk[pp, ff] = np.where(
                        ok & (den > 0), num / np.where(den > 0, den, 1.0), 0.0
                    )
            return e_chunk

        for e0 in range(0, nel, self.nrow_chunk):
            esl = slice(e0, min(e0 + self.nrow_chunk, nel))
            ne = esl.stop - esl.start

            if self.template == "neighbours":
                e_chunk = np.zeros(
                    (npol, nfreq_local, new, ne, nra), dtype=np.complex64
                )
                for ff in range(nfreq_local):
                    for pp in range(npol):
                        fh, _, _ = template(pp, ff, esl)
                        e_chunk[pp, ff] = phi[0, ff] * fh
                g_init = None
                for it in range(self.niter + 1):
                    G1, B1 = neighbour_amplitudes(e_chunk, esl)
                    if g_init is None:
                        g_init = G1.copy()
                        use = self._unexplained(G1, *data_power(esl))
                    else:
                        # Keep the weighted norm of the starting template at each
                        # sample, so that the amplitudes stay fractions of the
                        # foreground and the prior means the same as without
                        good = (G1 > 0) & (g_init > 0)
                        scale = np.where(
                            good, np.sqrt(g_init / np.where(good, G1, 1.0)), 0.0
                        )
                        sp = scale[[0] * npol] if self.combine_pol else scale
                        e_chunk *= sp[:, np.newaxis].astype(np.float32)
                        B1 = B1 * scale
                        G1 = np.where(good, g_init, 0.0)
                    if it == self.niter:
                        break
                    a = np.where(G1 > 0, B1 / np.where(G1 > 0, G1, 1.0), 0.0)
                    # Clip the weights at a few times their typical size in the row
                    power = np.where(G1 > 0, np.abs(a) ** 2, np.nan)
                    typical = np.nanmedian(power, axis=-1, keepdims=True)
                    typical = np.where(np.isfinite(typical), typical, 0.0)
                    inv_g = np.divide(1.0, G1, out=np.zeros_like(G1), where=G1 > 0)
                    lim = np.sqrt(self.clip * (typical + inv_g))
                    big = np.abs(a) > lim
                    a = np.where(big, a / np.where(big, np.abs(a), 1.0) * lim, a)
                    e_chunk = rebuild(a, esl, use)
                G = G1[..., np.newaxis, np.newaxis]
                B = B1[..., np.newaxis]
            else:
                # Normal equations, summed over the local frequencies and then over
                # ranks
                G = np.zeros((nfit, new, ne, nra, nparam, nparam), dtype=np.float64)
                B = np.zeros((nfit, new, ne, nra, nparam), dtype=np.complex128)
                for ff in range(nfreq_local):
                    for pp in range(npol):
                        fh, d, wt = template(pp, ff, esl)
                        pw = wt * np.abs(fh) ** 2
                        cross = wt * fh.conj() * d
                        pf = 0 if self.combine_pol else pp
                        for j in range(nparam):
                            B[pf, ..., j] += phi[j, ff] * cross
                            for k in range(nparam):
                                G[pf, ..., j, k] += phi[j, ff] * phi[k, ff] * pw
                allreduce(G, B)

            has = G[..., 0, 0] > 0
            if self.prior_mode == "block":
                var = self._block_prior(G, B, has, blk_start, blk, sigma**2)
            else:
                var = np.broadcast_to(sigma**2, (*G.shape[:-2], nparam))
            prior = np.zeros_like(G)
            for j in range(nparam):
                prior[..., j, j] = 1.0 / var[..., j]
            A = np.linalg.solve(G + prior, B[..., np.newaxis])[..., 0]
            A[~has] = 0.0

            # Store the rows of this chunk that are local to the fit container
            lo, hi = max(e0, e_lo), min(esl.stop, e_hi)
            if hi > lo:
                for pp in range(npol):
                    pf = 0 if self.combine_pol else pp
                    amp_out[pp, :, lo - e_lo : hi - e_lo] = A[pf, :, lo - e0 : hi - e0]
                    pow_out[pp, :, lo - e_lo : hi - e_lo] = G[
                        pf, :, lo - e0 : hi - e0, :, 0, 0
                    ]
                    psig_out[pp, :, lo - e_lo : hi - e_lo] = np.sqrt(
                        var[pf, :, lo - e0 : hi - e0, :, 0]
                    )

            if not self.subtract:
                continue

            for ff in range(nfreq_local):
                for pp in range(npol):
                    fh, _, wt = template(pp, ff, esl)
                    pf = 0 if self.combine_pol else pp
                    if self.template == "neighbours":
                        model = A[pf, ..., 0] * e_chunk[pp, ff]
                    else:
                        g = np.tensordot(A[pf], phi[:, ff], axes=([-1], [0]))
                        model = g * fh
                    model = np.where(wt > 0, model, 0.0)
                    vis[pp, ff, :, esl] -= model.astype(vis.dtype)

        stream.attrs["scintillation_subtracted"] = self.subtract

        return stream, fit

    def _unexplained(self, g_init, pw, npw):
        """Samples to use for building templates: not too bright, nor near one.

        Samples whose filtered data, in excess of the noise, have more than
        ``flag_fraction`` of the weighted amplitude of the starting template cannot
        be scintillation and are flagged, and the samples within ``ring`` of them
        are left out as well.
        """
        ok = g_init > 0
        # The noise level of each row, relative to the weights, is the median over
        # RA of the measured to the expected power (the weights do not include the
        # noise scale)
        ratio = np.where(ok & (npw > 0), pw / np.where(npw > 0, npw, 1.0), np.nan)
        scale = np.nanmedian(ratio, axis=-1, keepdims=True)
        scale = np.where(np.isfinite(scale), scale, 1.0)
        excess = pw - scale * npw
        bad = (
            ok
            & (excess > self.flag_fraction**2 * g_init)
            & (excess > 5.0 * scale * np.sqrt(2.0 * npw))
        )
        if self._ring > 0 and np.any(bad):
            bad = scipy.ndimage.maximum_filter1d(
                bad.astype(np.uint8), size=2 * self._ring + 1, axis=-1, mode="constant"
            ).astype(bool)
        return ok & ~bad

    def _block_prior(self, G, B, has, blk_start, blk, var0, niter=5):
        """Prior variance of each amplitude, measured in blocks of RA.

        Uses the diagonal of the normal equations: the unregularised amplitude
        ``a = B_j / G_jj`` has noise variance ``1 / G_jj``, so ``|a|^2`` estimates the
        prior variance plus that.

        Returns
        -------
        var : np.ndarray[..., nra, nparam]
            Prior variance of each amplitude at each sample.
        """
        nparam = G.shape[-1]
        lo, hi = self.prior_floor**2, self.prior_cap**2
        var = np.zeros((*G.shape[:-2], nparam), dtype=np.float64)
        for j in range(nparam):
            gjj = np.where(has, G[..., j, j], 1.0)
            noise = np.where(has, 1.0 / gjj, 0.0)
            x = np.where(has, np.abs(B[..., j] / gjj) ** 2, 0.0)
            s = np.full((*G.shape[:-3], blk_start.size), np.clip(var0[j], lo, hi))
            for _ in range(niter):
                tot = s[..., blk] + noise
                u = np.where(has, 1.0 / tot**2, 0.0)
                xc = np.minimum(x, self.prior_clip * tot)
                num = np.add.reduceat(u * (xc - noise), blk_start, axis=-1)
                den = np.add.reduceat(u, blk_start, axis=-1)
                s = np.clip(
                    np.where(den > 0, num / np.where(den > 0, den, 1.0), lo), lo, hi
                )
            var[..., j] = s[..., blk]
        return var


class SubtractElevationMedian(tasklib.base.ContainerTask):
    """Subtract the median over elevation from hybrid beamformed visibilities.

    This removes structure that is common to all elevations at a given frequency and
    time, such as the fluctuations of the autocorrelations on the 0 m baseline.  The
    real and imaginary parts are treated separately.

    Attributes
    ----------
    pol : list of str
        Polarisations to process.  Default is all.
    ew_index : list of int
        Indices of the east-west baselines to process.  Default is all.
    in_place : bool
        Modify the input.  This is only safe if no other task uses the input.
        Default is False.
    """

    pol = config.list_type(str, default=None)
    ew_index = config.list_type(int, default=None)
    in_place = config.Property(proptype=bool, default=False)

    def process(self, stream):
        """Subtract the median.

        Parameters
        ----------
        stream : draco.core.containers.HybridVisStream
            Hybrid beamformed visibilities, for example the output of
            :class:`LSQMFilterHybridVis`.

        Returns
        -------
        stream : draco.core.containers.HybridVisStream
            Visibilities with the median over elevation subtracted.
        """
        if not self.in_place:
            stream = stream.copy()

        stream.redistribute("freq")

        pol = list(stream.index_map["pol"])
        ipol = (
            list(range(len(pol)))
            if self.pol is None
            else [pol.index(p) for p in self.pol]
        )
        new = stream.index_map["ew"].size
        iew = list(range(new)) if self.ew_index is None else list(self.ew_index)

        vis = stream.vis[:].local_array  # pol, freq, ew, el, ra
        for ff in range(vis.shape[1]):
            for pp in ipol:
                for ee in iew:
                    v = vis[pp, ff, ee]
                    med = np.median(v.real, axis=0)
                    if np.iscomplexobj(v):
                        med = med + 1.0j * np.median(v.imag, axis=0)
                    v -= med[np.newaxis, :]

        stream.attrs["elevation_median_subtracted"] = True

        return stream


class EstimateElevationNoiseScale(tasklib.base.ContainerTask):
    """Measure how the noise of hybrid visibilities depends on elevation.

    For each polarisation, frequency, east-west baseline and elevation, the
    visibilities are put in units of the noise expected from their weights, and the
    scale is the square of their median absolute deviation over right ascension
    (times 1.4826), i.e. the ratio of the measured to the expected noise variance.
    The real part of the noise has variance ``1 / (2 w)``, except on the 0 m baseline,
    where the co-polar hybrid visibilities are real and it is ``1 / w``.  Elsewhere
    the imaginary part is used as well.

    Attributes
    ----------
    min_samples : int
        Minimum number of samples for an estimate.  Elevations with fewer are given a
        scale of zero.  Default is 100.
    """

    min_samples = config.Property(proptype=int, default=100)

    def process(self, stream):
        """Estimate the scale.

        Parameters
        ----------
        stream : draco.core.containers.HybridVisStream
            Hybrid beamformed visibilities with weights that do not depend on
            elevation, for example the output of :class:`LSQMFilterHybridVis`.

        Returns
        -------
        scale : ch_pipeline.core.containers.HybridVisNoiseScale
            Ratio of the measured to the expected noise variance.
        """
        if "elevation_vis_weight" in stream.datasets:
            raise ValueError("Elevation-dependent weights are not supported.")

        stream.redistribute("freq")

        out = containers.HybridVisNoiseScale(
            axes_from=stream,
            attrs_from=stream,
            distributed=stream.distributed,
            comm=stream.comm,
        )
        out.redistribute("freq")
        scale = out.scale[:].local_array
        nsample = out.nsample[:].local_array
        scale[:] = 0.0
        nsample[:] = 0

        self_conj = _self_conjugate(stream.index_map["pol"], stream.index_map["ew"][:])

        vis = stream.vis[:].local_array  # pol, freq, ew, el, ra
        weight = stream.weight[:].local_array  # pol, freq, ew, ra

        for pp in range(vis.shape[0]):
            for ff in range(vis.shape[1]):
                for ee in range(vis.shape[2]):
                    valid = weight[pp, ff, ee] > 0.0
                    if valid.sum() < self.min_samples:
                        continue

                    # Visibilities in units of the expected noise of the real part
                    real_only = self_conj[pp, ee]
                    sw = np.sqrt(
                        (1.0 if real_only else 2.0) * weight[pp, ff, ee, valid]
                    )
                    v = vis[pp, ff, ee][:, valid]
                    s = v.real * sw
                    if not real_only:
                        s = np.concatenate([s, v.imag * sw], axis=1)

                    med = np.median(s, axis=1, keepdims=True)
                    mad = np.median(np.abs(s - med), axis=1)
                    scale[pp, ff, ee] = (1.4826 * mad) ** 2
                    nsample[pp, ff, ee] = s.shape[1]

        return out


class ApplyElevationNoiseScale(tasklib.base.ContainerTask):
    """Attach an elevation-dependent noise scale to m-mode filtered visibilities.

    The scale is stored in the ``noise_scale`` dataset, which
    :class:`TransientMatchedFilter` uses to divide the noise weights at each
    elevation.  Entries without a valid scale are given a scale of one.  The input is
    modified.

    Entries whose scale exceeds ``max_scale`` are left out (set to zero, which every
    later step treats as no data): such a channel and elevation carries little
    information and its excess noise, from RFI, is unlikely to be Gaussian.  The
    flags of each polarisation and baseline are then closed over ``close_chan``
    channels and ``close_el`` elevations on either side, which fills holes in
    flagged regions (where the ripple of the scale with frequency and elevation dips
    below the limit) without growing them, and channels with at least
    ``chan_frac`` of their elevations flagged are left out entirely.  The ripple
    reaches a scale of ~3-4 at 400-450 MHz on 2418, so the limit should be above
    that: with 5 (and the defaults below), 20% of the entries with data are left
    out but 1.6% of the information (the channels of the local TV transmitters, and
    rows near the horizon and its alias in others).

    Attributes
    ----------
    max_scale : float
        Largest usable scale.  Zero (default) keeps every entry.
    close_chan : int
        Half-width of the closing in channels.  Default is 2.
    close_el : int
        Half-width of the closing in elevations.  Default is 40.
    chan_frac : float
        Leave out a channel with at least this fraction of its elevations flagged.
        Default is 0.8.
    """

    max_scale = config.Property(proptype=float, default=0.0)
    close_chan = config.Property(proptype=int, default=2)
    close_el = config.Property(proptype=int, default=40)
    chan_frac = config.Property(proptype=float, default=0.8)

    def process(self, stream, scale):
        """Attach the scale.

        Parameters
        ----------
        stream : ch_pipeline.core.containers.MFilteredHybridVisStream
            Output of :class:`LSQMFilterHybridVis`.
        scale : ch_pipeline.core.containers.HybridVisNoiseScale
            Output of :class:`EstimateElevationNoiseScale`, for example from another
            day or an average over days.

        Returns
        -------
        stream : ch_pipeline.core.containers.MFilteredHybridVisStream
            The input with the ``noise_scale`` dataset.
        """
        for ax in ("pol", "ew", "el"):
            a = np.asarray(stream.index_map[ax][:])
            b = np.asarray(scale.index_map[ax][:])
            same = (
                np.array_equal(a.astype(str), b.astype(str))
                if ax == "pol"
                else a.shape == b.shape and np.allclose(a, b)
            )
            if not same:
                raise ValueError(f"The {ax} axes of the stream and scale differ.")
        if not np.allclose(stream.freq, scale.freq):
            raise ValueError("The frequency axes of the stream and scale differ.")

        stream.redistribute("freq")
        scale.redistribute("freq")

        if "noise_scale" not in stream.datasets:
            stream.add_dataset("noise_scale")

        s = scale.scale[:].local_array
        good = np.isfinite(s) & (s > 0.0)
        out = np.where(good, s, 1.0)
        if self.max_scale > 0:
            # The flags need every channel and elevation of a polarisation
            sall = scale.scale[:].allgather()  # pol, freq, ew, el
            gall = np.isfinite(sall) & (sall > 0.0)
            flag = np.zeros(sall.shape, dtype=bool)
            for pp in range(sall.shape[0]):
                for ee in range(sall.shape[2]):
                    flag[pp, :, ee] = _scale_flags(
                        sall[pp, :, ee],
                        gall[pp, :, ee],
                        self.max_scale,
                        self.close_chan,
                        self.close_el,
                        self.chan_frac,
                    )
            lo = scale.scale[:].local_offset[1]
            flag = flag[:, lo : lo + s.shape[1]]
            out = np.where(flag, 0.0, out)
            self.log.info(
                f"{int(flag.sum()) if self.comm.size == 1 else self.comm.allreduce(int(flag.sum()))}"
                f" entries left out ({np.sum(flag & good) / max(good.sum(), 1):.3f} of the "
                f"local entries with a scale): scale above {self.max_scale}, closed over "
                f"{self.close_chan} channels and {self.close_el} elevations, and channels "
                f"with {self.chan_frac} of their elevations flagged."
            )
        stream.noise_scale[:].local_array[:] = out

        nbad = self.comm.allreduce(int((~good).sum()))
        if nbad:
            self.log.info(f"{nbad} entries without a valid noise scale set to one.")

        return stream


def _scale_flags(scale, have, max_scale, close_chan, close_el, chan_frac):
    """Flags [freq, el] of a noise scale: threshold, closing, whole channels.

    The closing pads the array by repeating its edges, so that regions reaching the
    horizon are not eroded there.
    """
    flag = have & (scale > max_scale)
    if (close_chan > 0 or close_el > 0) and flag.any():
        pad = ((close_chan, close_chan), (close_el, close_el))
        f = np.pad(flag, pad, mode="edge")
        st = np.ones((2 * close_chan + 1, 2 * close_el + 1), dtype=bool)
        f = scipy.ndimage.binary_erosion(
            scipy.ndimage.binary_dilation(f, structure=st), structure=st
        )
        flag |= f[
            close_chan : close_chan + scale.shape[0], close_el : close_el + scale.shape[1]
        ]
    nh = have.sum(axis=1)
    frac = (flag & have).sum(axis=1) / np.maximum(nh, 1)
    flag |= ((nh > 0) & (frac >= chan_frac))[:, np.newaxis]
    return flag & have


class ExtractSourceCutouts(tasklib.base.ContainerTask):
    """Extract hybrid visibilities in windows around the transits of sources.

    Attributes
    ----------
    sources : list of str
        Names of the sources in `ch_ephem.sources.source_dictionary`, or "sun".
    positions : dict
        Additional windows, as {name: [ra, dec]} in degrees (CIRS).
    ra_width : float
        Half-width of the window in right ascension in degrees.  Default is 15.
    el_width : int
        Half-width of the window in number of elevations.  Default is 32.
    """

    sources = config.list_type(
        type_=str, default=["sun", "CYG_A", "CAS_A", "TAU_A", "VIR_A"]
    )
    positions = config.Property(proptype=dict, default={})
    ra_width = config.Property(proptype=float, default=15.0)
    el_width = config.Property(proptype=int, default=32)

    def process(self, stream):
        """Extract the windows.

        Parameters
        ----------
        stream : draco.core.containers.HybridVisStream
            Hybrid beamformed visibilities on a uniform RA grid.  The ``lsd``
            attribute sets the epoch of the source positions.

        Returns
        -------
        out : ch_pipeline.core.containers.HybridVisSourceCutouts
            The windows around each source.
        """
        from ch_ephem.observers import chime

        stream.redistribute("freq")

        ra = stream.ra[:]
        el = stream.index_map["el"][:]
        nra, nel = ra.size, el.size
        dra = np.median(np.diff(ra))
        wrap = _is_full_day(ra)

        names, src_ra, src_dec = _source_positions(
            self.sources, self.positions, _stream_lsd(stream)
        )
        src_el = np.sin(np.radians(src_dec - chime.latitude))

        nw_ra = int(np.round(self.ra_width / dra))
        ra_off = np.arange(-nw_ra, nw_ra + 1)
        el_off = np.arange(-self.el_width, self.el_width + 1)

        out = containers.HybridVisSourceCutouts(
            source=np.array(names),
            ra_offset=ra_off,
            el_offset=el_off,
            axes_from=stream,
            attrs_from=stream,
            distributed=stream.distributed,
            comm=stream.comm,
        )
        out.redistribute("freq")
        out.attrs["ra_width"] = self.ra_width
        out.attrs["ra_periodic"] = wrap

        vis = stream.vis[:].local_array  # pol, freq, ew, el, ra
        weight = stream.weight[:].local_array  # pol, freq, ew, ra
        ovis = out.vis[:].local_array
        oweight = out.weight[:].local_array
        ovis[:] = 0.0
        oweight[:] = 0.0

        for ss in range(len(names)):
            ira0 = _nearest_ra_index(ra, src_ra[ss])
            iel0 = int(np.argmin(np.abs(el - src_el[ss])))

            ira = ira0 + ra_off
            if wrap:
                ira %= nra
            iel = iel0 + el_off
            ira = np.where((ira >= 0) & (ira < nra), ira, -1)
            iel = np.where((iel >= 0) & (iel < nel), iel, -1)

            out.datasets["ra_index"][ss] = ira
            out.datasets["el_index"][ss] = iel
            out.datasets["ra"][ss] = np.where(ira >= 0, ra[ira], 0.0)
            out.datasets["el"][ss] = np.where(iel >= 0, el[iel], 0.0)
            out.datasets["source_ra"][ss] = src_ra[ss]
            out.datasets["source_dec"][ss] = src_dec[ss]

            vr = ira >= 0
            ve = iel >= 0
            if not vr.any() or not ve.any():
                continue

            block = vis[..., iel[ve], :][..., ira[vr]]
            ovis[ss][..., np.flatnonzero(ve)[:, None], np.flatnonzero(vr)] = block
            oweight[ss][..., vr] = weight[..., ira[vr]]

        self.log.info(
            f"Extracted {len(names)} windows of {el_off.size} x {ra_off.size} samples."
        )

        return out


def _inv_var(err):
    """Inverse variance from a standard error, or zero if it is not finite."""
    return 1.0 / err**2 if np.isfinite(err) and err > 0 else 0.0


def _stream_lsd(stream):
    """The sidereal day of a stream, from its ``lsd`` or ``csd`` attribute."""
    lsd = stream.attrs.get("lsd", stream.attrs.get("csd"))
    if lsd is None:
        raise ValueError("The stream has no lsd attribute.")
    return float(np.atleast_1d(lsd)[0])


def _source_positions(sources, positions, lsd):
    """Apparent (CIRS) positions of sources at their transit on a sidereal day.

    Parameters
    ----------
    sources : list of str
        Names in `ch_ephem.sources.source_dictionary`, or "sun".
    positions : dict
        Additional positions, as {name: [ra, dec]} in degrees.
    lsd : float
        Sidereal day.

    Returns
    -------
    names : list of str
    ra, dec : np.ndarray
        Right ascension and declination in degrees.
    """
    from caput.astro.skyfield import skyfield_wrapper
    from ch_ephem import sources as ch_sources
    from ch_ephem.observers import chime

    names, ra, dec = [], [], []
    for name in sources:
        if name.lower() == "sun":
            body = skyfield_wrapper.ephemeris["sun"]
        else:
            body = ch_sources.source_dictionary[name]
        r, d = chime.object_coords(body, chime.lsd_to_unix(lsd + 0.5), deg=True)
        # Iterate once, so moving bodies are evaluated at their own transit
        r, d = chime.object_coords(body, chime.lsd_to_unix(lsd + r / 360.0), deg=True)
        names.append(name)
        ra.append(r)
        dec.append(d)
    for name, (r, d) in positions.items():
        names.append(name)
        ra.append(float(r))
        dec.append(float(d))

    return names, np.array(ra) % 360.0, np.array(dec, dtype=np.float64)


def _nearest_ra_index(ra, ra0):
    """Index of a right ascension on a uniform RA axis, modulo 360 degrees.

    An RA axis that extends below 0 or above 360 degrees has two samples near some
    right ascensions; the one within [0, 360) is preferred.  If the RA is not
    covered, the index is outside the axis, on the side nearest to it.
    """
    dra = np.median(np.diff(ra))
    best = None
    for shift in (0.0, -360.0, 360.0):
        k = int(np.round((ra0 % 360.0 + shift - ra[0]) / dra))
        outside = max(0, -k, k - (ra.size - 1))
        key = (outside, shift != 0.0)
        if best is None or key < best[0]:
            best = (key, k)
    return best[1]


def _channel_rfi_mask(mask, stream, el_slc):
    """Expand a mask of bands to the channels of a stream, at its local elevations.

    Parameters
    ----------
    mask : draco.core.containers.LocalizedSiderealRFIMask or None
        Mask with a frequency axis of bands (centre and width), for example the
        output of :class:`FlagMovingRFI`.
    stream : draco.core.containers.HybridVisStream
        Stream distributed over elevation.
    el_slc : slice
        Local elevations of the stream.

    Returns
    -------
    rfi : np.ndarray[freq, el_local, ra] or None
        True where a channel lies in a flagged band.
    """
    if mask is None:
        return None
    for ax in ("ra", "el"):
        a, b = stream.index_map[ax][:], mask.index_map[ax][:]
        if a.shape != b.shape or not np.allclose(a, b):
            raise ValueError(f"The {ax} axes of the stream and the RFI mask differ.")

    mask.redistribute("el")
    m = mask.mask[:]
    if m.local_bounds != el_slc:
        raise RuntimeError("The RFI mask is not distributed like the data.")
    m = m.local_array  # band, ra, el

    fmap = mask.index_map["freq"][:]
    lo = fmap["centre"] - 0.5 * fmap["width"]
    hi = fmap["centre"] + 0.5 * fmap["width"]
    freq = stream.freq[:]
    rfi = np.zeros((freq.size, m.shape[2], m.shape[1]), dtype=bool)
    for bb in range(fmap.size):
        sel = (freq >= lo[bb]) & (freq < hi[bb])
        rfi[sel] |= m[bb].T[np.newaxis]
    return rfi


def _combine_beams(beams):
    """Geometric mean of power beams [nbeam, ...], zero where any is not positive."""
    beams = np.asarray(beams)
    if beams.shape[0] == 1:
        return beams[0]
    pos = np.all(beams > 0.0, axis=0)
    logb = np.log(np.where(pos, beams, 1.0))
    return np.where(pos, np.exp(logb.mean(axis=0)), 0.0)


def _self_conjugate(pol, ew):
    """Which [pol, ew] hybrid visibilities are formed with their own conjugate.

    On the 0 m baseline the co-polar hybrid visibilities are the sum of each baseline
    and its conjugate (see draco's MakeVisGrid), so they are real.
    """
    copol = np.array([p[0] == p[1] for p in pol])
    return copol[:, np.newaxis] & (np.asarray(ew) == 0.0)[np.newaxis, :]


def _is_full_day(ra):
    """Whether a uniform RA axis in degrees spans one sidereal day."""
    dra = np.median(np.diff(ra))
    return bool(np.abs(ra.size * dra - 360.0) < 1e-3 * dra)


def _ground_to_sph_clipped(x, y, lat):
    """Hour angle and declination in radians of telescope coordinates (x, y).

    As :func:`caput.astro.coordinates.spherical.ground_to_sph`, but directions
    with ``x^2 + y^2 > 1`` are moved to the nearest point on the horizon.
    """
    r2 = x**2 + y**2
    scale = np.where(r2 > 1.0, 1.0 / np.sqrt(np.where(r2 > 0, r2, 1.0)), 1.0)
    x, y = x * scale, y * scale
    z = np.sqrt(np.maximum(1.0 - x**2 - y**2, 0.0))

    xe = z * np.cos(lat) - y * np.sin(lat)
    ze = y * np.cos(lat) + z * np.sin(lat)

    ha = -np.arctan2(x, xe)
    dec = np.arctan2(ze, np.sqrt(xe**2 + x**2))

    return ha, dec


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


class _BandMixin:
    """Group frequency channels into bands.

    Attributes
    ----------
    band_start : float
        Lower edge of the first band in MHz.  Default is 398.
    band_width : float
        Width of each band in MHz.  Default is 6.
    band_edges : list of float
        Explicit band edges in MHz.  Overrides band_start and band_width.
    single_band : bool
        Combine all frequencies into a single band.  Default is False.
    """

    band_start = config.Property(proptype=float, default=398.0)
    band_width = config.Property(proptype=float, default=6.0)
    band_edges = config.list_type((int, float), default=None)
    single_band = config.Property(proptype=bool, default=False)

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


class FlagMovingRFI(_BandMixin, tasklib.base.ContainerTask):
    """Flag RFI that moves in telescope-y, band by band.

    Signals reflected by aircraft (and other objects moving across the sky) cross
    the narrow east-west beam in a few samples while moving through a large range
    of telescope-y, so in the maps they are thin tracks that are near vertical or
    tilted, lit in the bands of their transmitters.  A celestial source stays at a
    fixed telescope-y.  This flags the tracks in each band separately, so that a
    source near or under a track keeps the other bands.

    At each polarisation, east-west baseline, elevation and sample, the filtered
    visibilities are averaged over the channels of each band, weighted by their
    inverse noise variance (from the weights and the noise scale), and the power of
    the average in units of its thermal variance is summed over the polarisations
    and baselines: ``Z``, which for noise has a gamma distribution with shape the
    number of terms ``n``.  The variance from the sky (the residual of bright
    sources, scintillation) is allowed for by dividing ``Z`` by a local scale ``s``,
    the running median (centred, over the samples with data) within
    ``local_window`` samples on either side in the same band and row of ``Z`` over
    its median for noise, and at least one.  A track crosses a row in a few
    samples, so it barely changes the median.  Where the sky dominates (``s`` above
    ``sky_scale_max``, near the transits of the Sun and the brightest sources), its
    fluctuations are common to all rows at once and look like vertical tracks, so
    no pixel is lit there; those samples have little weight in the matched filter
    anyway.

    In each band, a pixel is lit if ``Z / s`` is above the value that noise exceeds
    with the probability of a Gaussian S/N above ``snr_lit`` in magnitude, and its
    excess ``Z / s - n`` is above ``rel**2`` times the largest excess within
    ``line_dy`` in telescope-y and two samples on either side (so that the
    sidelobes of the synthesized beam around a bright source are not lit).  A line
    segment
    ``line_dy`` long in telescope-y, vertical or with each of the ``slopes`` (rows
    per sample, either sign), is detected where at least ``lit_frac`` of its pixels
    with data are lit, and at least half of its pixels have data.  The lit pixels of
    the detected segments are flagged, and the flags grown by ``grow_dy`` in
    telescope-y and ``grow_dt`` samples.  A source, even a bright one, is too short
    in telescope-y to fill a segment, and the track of a source across the day
    (in the far sidelobes) moves far slower than the smallest slope.

    The line test leaves the "heads" of the tracks: compact bright spots at the top
    of an arc (where an aircraft passes closest), at its turnarounds, and on the arc
    in other bands than its arms.  A head is a local maximum (within ``line_dy`` and
    two samples) of the largest excess ``Z / s - n`` over the bands not yet flagged,
    above ``head_snr**2``, that is

    - spiky in frequency: the ``head_nband`` largest excesses over all bands hold at
      least ``head_spiky`` of their sum (a broadband pulse spreads its power over
      many bands), and
    - attached to a piece of the flags of all bands combined at least half of
      ``line_dy`` tall: within one sample and ``head_tol_dy`` plus
      ``head_tol_slope`` times its slope (rows per sample) of its flags, or in the
      cone beyond one of its ends, up to ``head_max_dt`` samples along the slope
      of the end (centroid rows of its last five samples), within ``head_tol_dy``
      plus ``head_tol_slope`` times the slope times the samples.

    In each band where a pixel within ``grow_dy`` and ``grow_dt`` of the head is
    above the value for an |S/N| of ``head_snr_band``, the lit pixels connected to
    the lit pixels there, within ``line_dy`` and ``head_dt`` samples of the head,
    are flagged too, and grown like the tracks.

    The output is a mask with one entry per band (its frequency axis holds the
    centre and width of each band), shared by the polarisations and baselines.
    :class:`EstimateSkyNoise` and :class:`TransientMatchedFilterSkyNoise` take it
    as an optional requirement and leave out the channels of flagged bands.

    Attributes
    ----------
    pol : list of str
        Polarisations to use.  Default is ["XX", "YY"].
    ew_index : list of int
        Indices of the east-west baselines to use.  Default is all except 0 m,
        whose co-polar visibilities are real.
    snr_lit : float
        Equivalent Gaussian |S/N| of a lit pixel.  Default is 2.5.
    rel : float
        Smallest amplitude of a lit pixel relative to the largest nearby.  Default is
        0.15.
    line_dy : float
        Length of the line segments in telescope-y.  Default is 0.067 (41 rows of
        the usual grid), longer than a bright source in a single band (~0.03).
    slopes : list of float
        Slopes of the tilted segments in rows per sample.  Default is
        [1.5, 2, 3, 5, 10].
    lit_frac : float
        Fraction of a segment's pixels that must be lit.  Default is 0.5.
    grow_dy, grow_dt : float, int
        Growth of the flags in telescope-y and in samples.  Defaults are 0.0033 (2
        rows) and 1.
    head_snr : float
        Smallest equivalent |S/N| (square root of the excess) of a head.  Zero turns
        the heads off.  Default is 8.
    head_spiky : float
        Smallest fraction of the excess over all bands in the ``head_nband``
        largest.  Default is 0.75 (aircraft heads 0.83-1.00, J1634+44 0.23).
    head_nband : int
        Number of bands for ``head_spiky``.  Default is 3.
    head_snr_band : float
        Equivalent |S/N| above which a band is flagged at a head.  Default is 4.
    head_tol_dy : float
        Distance in telescope-y from a piece of the flags.  Default is 0.0164 (10
        rows).
    head_max_dt : int
        Largest number of samples beyond the end of a piece.  Default is 6.
    head_tol_slope : float
        Allowance for the change of the slope beyond the end of a piece.  Default is
        1.5.
    head_dt : int
        Half-width in samples of the region flagged around a head.  Default is 3.
    local_window : int
        Half-width in samples of the running median.  Default is 30 (5 minutes).
    sky_scale_max : float
        Largest local scale (total over thermal variance) at which pixels can be
        lit.  Zero turns the limit off.  Default is 3.
    nsample_chunk : int
        Approximate number of samples averaged at once.  Default is 65536.
    """

    pol = config.list_type(str, default=["XX", "YY"])
    ew_index = config.list_type(int, default=None)
    snr_lit = config.Property(proptype=float, default=2.5)
    rel = config.Property(proptype=float, default=0.15)
    line_dy = config.Property(proptype=float, default=0.067)
    slopes = config.list_type(float, default=[1.5, 2.0, 3.0, 5.0, 10.0])
    lit_frac = config.Property(proptype=float, default=0.5)
    grow_dy = config.Property(proptype=float, default=0.0033)
    grow_dt = config.Property(proptype=int, default=1)
    head_snr = config.Property(proptype=float, default=8.0)
    head_spiky = config.Property(proptype=float, default=0.75)
    head_nband = config.Property(proptype=int, default=3)
    head_snr_band = config.Property(proptype=float, default=4.0)
    head_tol_dy = config.Property(proptype=float, default=0.0164)
    head_max_dt = config.Property(proptype=int, default=6)
    head_tol_slope = config.Property(proptype=float, default=1.5)
    head_dt = config.Property(proptype=int, default=3)
    local_window = config.Property(proptype=int, default=30)
    sky_scale_max = config.Property(proptype=float, default=3.0)
    nsample_chunk = config.Property(proptype=int, default=65536)

    def process(self, stream):
        """Flag the moving RFI.

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
        from scipy.special import gammainccinv
        from scipy.stats import gamma, norm

        if "elevation_vis_weight" in stream.datasets:
            raise ValueError("Elevation-dependent weights are not supported.")

        terms = _power_terms(stream, self.pol, self.ew_index, self.log)
        nterm = len(terms)

        freq = stream.freq[:]
        band_index, band_map = self._assign_bands(freq)
        nband = band_map.size
        chans = [np.flatnonzero(band_index == bb) for bb in range(nband)]

        el = stream.index_map["el"][:]
        nel = el.size
        dy = np.median(np.abs(np.diff(el)))
        ra = stream.ra[:]
        nra = ra.size
        periodic = _is_full_day(ra) and bool(stream.attrs.get("ra_periodic", True))

        # For noise with n terms: the median of Z, and the threshold of a lit pixel
        nn = np.arange(1, nterm + 1)
        z_med = np.concatenate([[1.0], gamma(nn).median()])
        z_lit = np.concatenate(
            [[np.inf], gammainccinv(nn, 2.0 * norm.sf(self.snr_lit))]
        )

        t0 = time.time()
        Z, nz, _ = _band_power(stream, terms, chans, self.nsample_chunk, self.log)
        nel_local = Z.shape[1]
        self.log.debug(f"Formed the band powers in {time.time() - t0:0.1f} seconds.")

        # Each band over all elevations
        Z = mpiarray.MPIArray.wrap(Z, axis=1, comm=self.comm).redistribute(0)
        nz = mpiarray.MPIArray.wrap(nz, axis=1, comm=self.comm).redistribute(0)
        Z, nz = Z.local_array, nz.local_array

        L = 2 * max(1, round(0.5 * self.line_dy / dy)) + 1
        lines = _line_offsets(L, self.slopes)
        grow = np.ones(
            (2 * round(self.grow_dy / dy) + 1, 2 * self.grow_dt + 1), dtype=bool
        )
        flag = np.zeros(Z.shape, dtype=bool)
        t0 = time.time()
        for bb in range(Z.shape[0]):
            valid = nz[bb] > 0
            if not valid.any():
                continue
            s = _power_scale(Z[bb], nz[bb], z_med, self.local_window, periodic)
            zs = np.where(valid, Z[bb] / s, 0.0)
            excess = np.where(valid, zs - nz[bb], 0.0)
            near = scipy.ndimage.maximum_filter(
                excess, size=(2 * L + 1, 5), mode="constant"
            )
            lit = valid & (zs > z_lit[nz[bb]]) & (excess > self.rel**2 * near)
            if self.sky_scale_max > 0:
                lit &= s <= self.sky_scale_max
                # nothing stands out where the sky dominates, for the heads either
                zs = np.where(s <= self.sky_scale_max, zs, nz[bb])
            flag[bb] = _line_mask(lit, valid, lines, self.lit_frac)
            if flag[bb].any():
                flag[bb] = scipy.ndimage.binary_dilation(flag[bb], structure=grow)
            Z[bb] = zs
        self.log.debug(f"Searched for tracks in {time.time() - t0:0.1f} seconds.")

        if self.head_snr > 0:
            t0 = time.time()
            z_band = np.concatenate(
                [[np.inf], gammainccinv(nn, 2.0 * norm.sf(self.head_snr_band))]
            )
            flag |= self._flag_heads(Z, nz, flag, L, grow, z_lit, z_band, dy)
            self.log.debug(f"Flagged the heads in {time.time() - t0:0.1f} seconds.")

        flag = mpiarray.MPIArray.wrap(flag, axis=0, comm=self.comm).redistribute(1)
        if flag.local_shape[1] != nel_local:
            raise RuntimeError("The flags are not distributed like the data.")
        flag = flag.local_array

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

        nflag = self.comm.allreduce(flag.sum(axis=(1, 2)))
        frac = nflag / float(nel * nra)
        top = np.argsort(frac)[::-1][:5]
        self.log.info(
            f"Flagged {frac.mean():0.3%} of the band samples; most in "
            + ", ".join(
                f"{band_map['centre'][bb]:0.1f} MHz ({frac[bb]:0.3%})" for bb in top
            )
            + "."
        )

        return out

    def _flag_heads(self, zs, nz, flag, L, grow, z_lit, z_band, dy):
        """Flags for the heads attached to the tracks.

        Parameters
        ----------
        zs : np.ndarray[band_local, el, ra]
            Band power over its local scale.
        nz : np.ndarray[band_local, el, ra]
            Number of terms.
        flag : np.ndarray[band_local, el, ra]
            Flags of the tracks.
        L : int
            Length of the line segments in rows.
        grow : np.ndarray[2 * gy + 1, 2 * gt + 1]
            Growth of the flags.
        z_lit, z_band : np.ndarray
            Lit and significant values of ``zs`` for each number of terms.
        dy : float
            Spacing of the rows in telescope-y.

        Returns
        -------
        heads : np.ndarray[band_local, el, ra]
            The flags of the heads.
        """
        from mpi4py import MPI

        nb_local, nel, nra = zs.shape
        comm = self.comm
        b_off = comm.exscan(nb_local) or 0
        nband = comm.allreduce(nb_local)

        # Largest excess over the bands not flagged, and the number of bands flagged
        peak = np.zeros((nel, nra), dtype=np.float32)
        nflag = np.zeros((nel, nra), dtype=np.int32)
        for bb in range(nb_local):
            ok = (nz[bb] > 0) & ~flag[bb]
            np.maximum(peak, np.where(ok, zs[bb] - nz[bb], 0.0), out=peak)
            nflag += flag[bb]
        comm.Allreduce(MPI.IN_PLACE, peak, op=MPI.MAX)
        comm.Allreduce(MPI.IN_PLACE, nflag, op=MPI.SUM)

        cand = (peak >= self.head_snr**2) & (
            peak == scipy.ndimage.maximum_filter(peak, size=(L, 5), mode="constant")
        )
        ce, ct = np.nonzero(cand)
        ncand = ce.size
        out = np.zeros(zs.shape, dtype=bool)
        if ncand == 0:
            return out

        # The excess in every band at the candidates, and whether each band is
        # significant next to them
        gy, gt = grow.shape[0] // 2, grow.shape[1] // 2
        spec = np.zeros((nband, ncand), dtype=np.float64)
        sig = np.zeros((nband, ncand), dtype=np.float64)
        for bb in range(nb_local):
            ok = nz[bb] > 0
            spec[b_off + bb] = np.where(ok, zs[bb] - nz[bb], 0.0)[ce, ct].clip(0.0)
            big = ok & (zs[bb] > z_band[nz[bb]])
            big = scipy.ndimage.maximum_filter(big, size=(2 * gy + 1, 2 * gt + 1))
            sig[b_off + bb] = big[ce, ct]
        comm.Allreduce(MPI.IN_PLACE, spec, op=MPI.SUM)
        comm.Allreduce(MPI.IN_PLACE, sig, op=MPI.SUM)

        top = -np.sort(-spec, axis=0)[: self.head_nband].sum(axis=0)
        spiky = top >= self.head_spiky * np.maximum(spec.sum(axis=0), 1e-30)

        cols, ends = _flag_pieces(nflag > 0, max(1, L // 2))
        tol = self.head_tol_dy / dy
        head = np.array(
            [
                spiky[ii]
                and _attached(
                    ct[ii],
                    ce[ii],
                    cols,
                    ends,
                    tol,
                    self.head_max_dt,
                    self.head_tol_slope,
                )
                for ii in range(ncand)
            ],
            dtype=bool,
        )
        self.log.info(
            f"{ncand} bright spots outside the tracks, {spiky.sum()} spiky in "
            f"frequency, {head.sum()} attached to tracks."
        )

        # In the significant bands, the lit pixels connected to each head
        h = L // 2
        dt = self.head_dt
        for ii in np.flatnonzero(head):
            e0, e1 = max(ce[ii] - h, 0), min(ce[ii] + h + 1, nel)
            t0, t1 = max(ct[ii] - dt, 0), min(ct[ii] + dt + 1, nra)
            for bb in range(nb_local):
                if not sig[b_off + bb, ii]:
                    continue
                n_ = nz[bb, e0:e1, t0:t1]
                lit = (n_ > 0) & (zs[bb, e0:e1, t0:t1] > z_lit[n_])
                blob, _ = scipy.ndimage.label(lit)
                ee, tt = ce[ii] - e0, ct[ii] - t0
                seed = blob[max(ee - gy, 0) : ee + gy + 1, max(tt - gt, 0) : tt + gt + 1]
                seed = np.unique(seed[seed > 0])
                if seed.size:
                    out[bb, e0:e1, t0:t1] |= np.isin(blob, seed)
        for bb in range(nb_local):
            if out[bb].any():
                out[bb] = scipy.ndimage.binary_dilation(out[bb], structure=grow)
        return out


def _flag_pieces(mask, min_rows, nend=5):
    """Pieces of a mask [el, ra] at least ``min_rows`` tall: their columns and ends.

    Pixels within one row or sample of each other are joined.

    Returns
    -------
    cols : np.ndarray[ncol, 4]
        For each sample of each piece: the sample, the lowest and highest row, and
        the slope in rows per sample of the centroid rows there.
    ends : np.ndarray[nend, 4]
        For each end of each piece: the side (-1 for the first sample, +1 for the
        last), the sample, the centroid row, and the slope of the centroid rows of
        the ``nend`` samples at that end.
    """
    lab, _ = scipy.ndimage.label(
        scipy.ndimage.binary_dilation(mask, structure=np.ones((3, 3), dtype=bool))
    )
    lab = np.where(mask, lab, 0)
    cols, ends = [], []
    for ii, sl in enumerate(scipy.ndimage.find_objects(lab)):
        if sl is None or sl[0].stop - sl[0].start < min_rows:
            continue
        sub = lab[sl] == ii + 1
        cc = np.flatnonzero(sub.any(axis=0))
        rows = np.arange(sl[0].start, sl[0].stop, dtype=np.float64)
        t = sl[1].start + cc
        sub = sub[:, cc]
        e = (sub * rows[:, np.newaxis]).sum(axis=0) / sub.sum(axis=0)
        lo = np.where(sub, rows[:, np.newaxis], np.inf).min(axis=0)
        hi = np.where(sub, rows[:, np.newaxis], -np.inf).max(axis=0)
        slope = np.gradient(e, t) if t.size > 1 else np.zeros(1)
        cols.append(np.stack([t, lo, hi, slope], axis=1))
        for side, kk in ((-1, slice(0, nend)), (1, slice(-nend, None))):
            sl_end = np.polyfit(t[kk], e[kk], 1)[0] if t[kk].size >= 2 else 0.0
            jj = 0 if side < 0 else -1
            ends.append((side, t[jj], e[jj], sl_end))
    cols = np.concatenate(cols) if cols else np.zeros((0, 4))
    return cols, np.array(ends, dtype=np.float64).reshape(-1, 4)


def _attached(t, e, cols, ends, tol, max_dt, tol_slope):
    """Whether the pixel (row e, sample t) is attached to a piece of the flags.

    It is if, within one sample, a piece comes within ``tol + tol_slope |slope|``
    rows (the slope of the piece there), or if the pixel is up to ``max_dt``
    samples beyond an end of a piece, within ``tol + tol_slope |slope| dt`` rows of
    the line along the slope of the end.
    """
    near = np.abs(cols[:, 0] - t) <= 1
    if near.any():
        c = cols[near]
        off = np.maximum(c[:, 1] - e, e - c[:, 2]).clip(0.0)
        if np.any(off <= tol + tol_slope * np.abs(c[:, 3])):
            return True
    if ends.size == 0:
        return False
    side, te, ee, slope = ends.T
    dt = (t - te) * side
    off = np.abs(e - (ee + slope * (t - te)))
    ok = (dt >= 1) & (dt <= max_dt) & (off <= tol + tol_slope * np.abs(slope) * dt)
    return bool(ok.any())


def _power_terms(stream, pols, ew_index, log):
    """(pol, ew) index pairs of the terms of the band power, without real 0 m."""
    pol = list(stream.index_map["pol"])
    ipol = [pol.index(p) for p in pols]
    ew = stream.index_map["ew"][:]
    iew = (
        [ii for ii in range(ew.size) if ew[ii] != 0.0]
        if ew_index is None
        else list(ew_index)
    )
    self_conj = _self_conjugate(pols, ew[iew])
    terms = [
        (pp, ee)
        for ii, pp in enumerate(ipol)
        for jj, ee in enumerate(iew)
        if not self_conj[ii, jj]
    ]
    if len(terms) < self_conj.size:
        log.warning("Leaving out the real co-polar 0 m visibilities.")
    if not terms:
        raise ValueError("No polarisations and baselines to use.")
    return terms


def _band_power(stream, terms, chans, nsample_chunk, log):
    """Power of the band averages of filtered visibilities, in units of the noise.

    The visibilities of each term (polarisation and baseline) are averaged over the
    channels of each band, weighted by their inverse noise variance (the weights
    and, if present, the noise scale), and ``|average|^2`` over its variance is
    summed over the terms.  The stream is left distributed over elevation.

    Returns
    -------
    Z : np.ndarray[band, el_local, ra]
        The summed power; for noise, gamma distributed with shape ``n``.
    n : np.ndarray[band, el_local, ra]
        The number of terms with data.
    el_slc : slice
        The local elevations.
    """
    # Weights of the filtered data have no el axis: gather them
    stream.redistribute("freq")
    weight = stream.weight[:].allgather()  # pol, freq, ew, ra

    stream.redistribute(["el", "freq"])
    vis = stream.vis[:].local_array
    el_slc = stream.vis[:].local_bounds
    nel_local, nra = vis.shape[3:]
    if "noise_scale" in stream.datasets:
        ns = stream.noise_scale[:]
        if ns.local_bounds != el_slc:
            raise RuntimeError("noise_scale is not distributed like the data.")
        nscale = ns.local_array  # pol, freq, ew, el
    else:
        log.info("No noise_scale dataset: using the weights alone.")
        nscale = None

    Z = np.zeros((len(chans), nel_local, nra), dtype=np.float32)
    nz = np.zeros((len(chans), nel_local, nra), dtype=np.int16)
    nrow = max(1, nsample_chunk // nra)
    for e0 in range(0, nel_local, nrow):
        esl = slice(e0, min(e0 + nrow, nel_local))
        for pp, ee in terms:
            for bb, cc in enumerate(chans):
                d = vis[pp, cc, ee, esl]  # chan, el, ra
                a = np.broadcast_to(weight[pp, cc, ee][:, np.newaxis], d.shape)
                if nscale is not None:
                    isc = tools.invert_no_zero(nscale[pp, cc, ee, esl])
                    a = a * isc[..., np.newaxis]
                ok = np.isfinite(d) & (a > 0)
                a = np.where(ok, a, 0.0)
                A = a.sum(axis=0)
                y = np.sum(a * np.where(ok, d, 0.0), axis=0)
                Z[bb, esl] += np.abs(y) ** 2 * tools.invert_no_zero(A)
                nz[bb, esl] += A > 0
    return Z, nz, el_slc


def _power_scale(Z, nz, z_med, window, periodic):
    """Local scale of the band power ``Z`` [el, ra] with ``nz`` terms.

    The running median along RA of ``Z`` over its median for noise (``z_med[n]``),
    over the samples with data within ``window`` on either side, and at least one.
    """
    if window <= 0:
        return np.ones(Z.shape)
    valid = nz > 0
    nra = Z.shape[-1]
    r = np.where(valid, Z / z_med[nz], 0.0).astype(np.float64)
    w = valid.astype(np.float64)
    pad = window if periodic else 0
    if pad:
        r = np.concatenate([r[:, -pad:], r, r[:, :pad]], axis=1)
        w = np.concatenate([w[:, -pad:], w, w[:, :pad]], axis=1)
    s = median.moving_weighted_median(
        np.ascontiguousarray(r), np.ascontiguousarray(w), size=(1, 2 * window + 1)
    )
    return np.maximum(s[:, pad : pad + nra], 1.0)


def _line_offsets(L, slopes):
    """Pixel offsets (rows, samples) of line segments L rows long.

    One vertical segment, and two (either sign) for each slope in rows per sample.
    """
    h = L // 2
    rows = np.arange(-h, h + 1)
    out = [np.stack([rows, np.zeros_like(rows)], axis=1)]
    for sl in slopes:
        for sign in (1, -1):
            out.append(np.stack([rows, np.rint(sign * rows / sl).astype(int)], axis=1))
    return out


def _shift_sum(a, offsets, pad):
    """Sum over offsets (dy, dt) of a[y + dy, t + dt], zero outside the array."""
    p = np.pad(a, pad)
    ny, nt = a.shape
    out = np.zeros(a.shape, dtype=np.float32)
    for dy_, dt_ in offsets:
        out += p[pad + dy_ : pad + dy_ + ny, pad + dt_ : pad + dt_ + nt]
    return out


def _line_mask(lit, valid, lines, frac):
    """Lit pixels on line segments that are at least ``frac`` lit.

    Parameters
    ----------
    lit, valid : np.ndarray[y, t] of bool
        Lit pixels, and pixels with data.
    lines : list of np.ndarray[npix, 2]
        Offsets of the segments, from :func:`_line_offsets`.
    frac : float
        Fraction of the pixels of a segment with data that must be lit.  Segments
        with data in fewer than half of their pixels are not used.

    Returns
    -------
    hit : np.ndarray[y, t] of bool
    """
    pad = max(int(np.abs(off).max()) for off in lines)
    litf, validf = lit.astype(np.float32), valid.astype(np.float32)
    hit = np.zeros(lit.shape, dtype=bool)
    for off in lines:
        nl = _shift_sum(litf, off, pad)
        nv = _shift_sum(validf, off, pad)
        det = (nv >= 0.5 * len(off)) & (nl >= frac * np.maximum(nv, 1.0))
        if det.any():
            hit |= _shift_sum(det.astype(np.float32), -off, pad) > 0
    return hit & lit


class EstimateInbandNoise(_BandMixin, tasklib.base.ContainerTask):
    """Measure the local noise of filtered visibilities from their scatter in frequency.

    The noise expected from the weights and the static noise scale (a whole-day
    estimate for each polarisation, channel, baseline and elevation) misses noise
    that changes across the sky: the sky in the formed beams (the Galaxy, the Sun)
    raises the system temperature of the rows that point at it, and bright sources
    add self-noise.  Both are independent between channels, unlike the signals we
    look for (sources, scintillation, transients), which are smooth in frequency.

    At each polarisation, baseline, elevation and sample, the filtered visibilities
    are fitted across the channels (with data) by discrete prolate spheroidal
    sequences up to the delay ``delay_pad + |b| x_max / c`` (the fringe of a source
    within ``x_max`` of the meridian on baseline ``b``, plus the structure of the
    spectra and the beam), weighted by the inverse of their expected noise variance
    ``w / s`` (weights and static scale; the static scale is averaged over blocks of
    ``el_block`` elevations for the weights of the fit).  In each band, the residual
    power in units of the expected noise, and its expectation for unit-variance
    noise (the sum of one minus the leverage of each channel), are summed over
    ``el_window`` elevations and ``ra_window`` samples on either side, and their
    ratio is the scale.

    The scale is the product of two factors.  One is common to all elevations
    and is measured at every sample without averaging in time: the residual
    power of each elevation in units of its own smooth level, summed over all
    elevations.  It follows jumps of the system temperature that last a sample
    or two and raise the noise of every formed beam at once (bursts from the Sun
    near the beam on 2418), which the smoothed weights and a window in time
    would dilute.  The other is the smooth factor of each elevation, summed over
    the windows after dividing out the common factor (starting from the power
    summed over all elevations at each sample over its running median in time,
    then two passes).  Where the
    common factor exceeds ``flag_common`` in a band, those samples are flagged
    (``flag``), and consumers leave them out.
    Because the fit removes anything smooth in frequency, a pulse does not raise its
    own noise, and the estimate can include the sample itself.  Samples flagged in
    an RFI mask (given to setup) are left out of the sums.

    The noise of neighbouring channels is slightly correlated (+0.03 for adjacent
    channels on 2418), which raises the noise of a band sum above that implied by
    independent channels; the scale of each band is multiplied by
    ``1 + 2 channel_correlation n_pairs / n``, with ``n`` its channels with data and
    ``n_pairs`` the adjacent pairs among them, so that it describes band sums.

    The part of the sky noise that is common to the channels of a band
    (scintillation and jitter of bright sources) is not measured here; that is the
    coherent term of :class:`EstimateSkyNoise`.  :class:`EstimateSkyNoise` and
    :class:`TransientMatchedFilterSkyNoise` take the output as an optional
    requirement and multiply the thermal variance by it.

    Attributes
    ----------
    pol : list of str
        Polarisations.  Default is all of the stream's.
    ew_index : list of int
        Indices of the east-west baselines.  Default is all.
    delay_pad : float
        Delay in ns added to the fringe delay of the fit.  Default is 50.
    x_max : float
        Largest telescope-x for the fringe delay.  Default is 0.3.
    ridge : float
        Regularisation of the fit, relative to the mean of the diagonal of its
        normal equations.  Default is 1e-6.
    el_window : int
        Half-width of the sums in elevations.  Default is 3.
    ra_window : int
        Half-width of the sums in samples.  Default is 5.
    channel_correlation : float
        Correlation of the noise of adjacent channels.  Default is 0.03.
    min_dof : float
        Smallest number of degrees of freedom of the sums; below it the scale is
        one.  Default is 20.
    el_block : int
        Number of elevations that share the weights of the fit (the mean of their
        static scale).  Default is 8.
    extra_modes : int
        Sequences beyond ``2 N W`` (the time-bandwidth product).  The first ``2 N W``
        leave ~3e-3 of the power of a smooth spectrum; ten more leave ~1e-10.
        Default is 10.
    flag_common : float
        Flag the samples of a band where the factor common to all elevations
        exceeds this.  Zero turns the flags off.  Default is 4.
    max_leverage : float
        Channels whose leverage in the fit exceeds this (isolated between masked
        channels, so that the fit nearly reproduces them) are left out of the
        estimate.  Default is 0.5.
    min_common_dof : float
        Smallest number of degrees of freedom, summed over the elevations, of the
        factor common to all elevations at a sample; below it the factor is one.
        Default is 100.
    """

    pol = config.list_type(str, default=None)
    ew_index = config.list_type(int, default=None)
    delay_pad = config.Property(proptype=float, default=50.0)
    x_max = config.Property(proptype=float, default=0.3)
    ridge = config.Property(proptype=float, default=1e-6)
    el_window = config.Property(proptype=int, default=3)
    ra_window = config.Property(proptype=int, default=5)
    channel_correlation = config.Property(proptype=float, default=0.03)
    min_dof = config.Property(proptype=float, default=20.0)
    el_block = config.Property(proptype=int, default=8)
    extra_modes = config.Property(proptype=int, default=10)
    flag_common = config.Property(proptype=float, default=4.0)
    max_leverage = config.Property(proptype=float, default=0.5)
    min_common_dof = config.Property(proptype=float, default=100.0)

    rfi_mask = None

    def setup(self, mask=None):
        """Set the RFI mask.

        Parameters
        ----------
        mask : draco.core.containers.LocalizedSiderealRFIMask, optional
            Bands, elevations and samples to leave out.
        """
        self.rfi_mask = mask

    def process(self, stream):
        """Estimate the scale.

        Parameters
        ----------
        stream : ch_pipeline.core.containers.MFilteredHybridVisStream
            Filtered visibilities, with the ``noise_scale`` dataset if the static
            scale is to be included.

        Returns
        -------
        out : ch_pipeline.core.containers.HybridVisInbandNoise
            The scale of each band at every elevation and sample.
        """
        from scipy.signal.windows import dpss

        if "elevation_vis_weight" in stream.datasets:
            raise ValueError("Elevation-dependent weights are not supported.")

        freq = stream.freq[:]
        band_index, band_map = self._assign_bands(freq)
        nband = band_map.size
        pol = [str(p) for p in stream.index_map["pol"]]
        ipol = list(range(len(pol))) if self.pol is None else [pol.index(p) for p in self.pol]
        ew = stream.index_map["ew"][:]
        iew = list(range(ew.size)) if self.ew_index is None else list(self.ew_index)
        ra = stream.ra[:]
        nra = ra.size
        periodic = _is_full_day(ra) and bool(stream.attrs.get("ra_periodic", True))

        # Weights of the filtered data have no el axis: gather them
        stream.redistribute("freq")
        weight = stream.weight[:].allgather()  # pol, freq, ew, ra

        stream.redistribute(["el", "freq"])
        vis = stream.vis[:].local_array
        el_slc = stream.vis[:].local_bounds
        nel_local = vis.shape[3]
        nel = stream.index_map["el"].size
        if "noise_scale" in stream.datasets:
            ns = stream.noise_scale[:]
            if ns.local_bounds != el_slc:
                raise RuntimeError("noise_scale is not distributed like the data.")
            nscale = ns.local_array  # pol, freq, ew, el
            # All elevations, for the weights of the fit of each block of elevations
            nscale_all = stream.noise_scale[:].allgather()
        else:
            self.log.info("No noise_scale dataset: using the weights alone.")
            nscale = None
            nscale_all = None
        # Blocks of elevations on the global grid, so that the result does not depend
        # on the distribution
        gel = np.arange(nel)[el_slc]
        blocks = [
            (np.flatnonzero(gel // self.el_block == bk), bk)
            for bk in np.unique(gel // self.el_block)
        ]
        rfi = _channel_rfi_mask(self.rfi_mask, stream, el_slc)  # freq, el, ra
        flagged = None
        if rfi is not None:
            flagged = np.zeros((nband, nel_local, nra), dtype=bool)
            for bb in range(nband):
                flagged[bb] = rfi[band_index == bb].any(axis=0)

        # Channels on a uniform grid, for the sequences
        dnu = np.min(np.abs(np.diff(freq))) if freq.size > 1 else 1.0
        grid = np.rint((freq - freq.min()) / dnu).astype(int)
        ngrid = grid.max() + 1
        # Sums over the channels of each band, and pairs of adjacent channels in a band
        bsum = (band_index[np.newaxis, :] == np.arange(nband)[:, np.newaxis]).astype(
            np.float64
        )
        same = np.zeros(freq.size, dtype=bool)
        same[:-1] = (band_index[:-1] == band_index[1:]) & (band_index[:-1] >= 0)

        shp = (len(ipol), nband, len(iew), nel_local, nra)
        num = np.zeros(shp, dtype=np.float32)
        den = np.zeros(shp, dtype=np.float32)
        dof = np.zeros((len(ipol), nband, len(iew), nra), dtype=np.float32)
        corr = np.ones((len(ipol), nband, len(iew), nra), dtype=np.float32)

        t0 = time.time()
        for oe, ie in enumerate(iew):
            tau = 1e-9 * self.delay_pad + abs(ew[ie]) * self.x_max / scipy.constants.c
            nw = tau * dnu * 1e6 * ngrid
            nmode = int(np.clip(np.floor(2 * nw) + self.extra_modes, 1, ngrid // 2))
            A = dpss(ngrid, max(nw, 0.5), Kmax=nmode).T[grid]  # freq, mode
            for op, ip in enumerate(ipol):
                w = weight[ip, :, ie]  # freq, ra
                sc = (
                    nscale[ip, :, ie]
                    if nscale is not None
                    else np.ones((freq.size, nel_local), dtype=np.float32)
                )
                isc = tools.invert_no_zero(sc)
                for bi_, (rows, bk) in enumerate(blocks):
                    esl = slice(rows[0], rows[-1] + 1)
                    # The fit is weighted by the mean inverse static scale of the
                    # block, which changes slowly with elevation
                    gsl = slice(bk * self.el_block, min((bk + 1) * self.el_block, nel))
                    isc_b = (
                        np.mean(tools.invert_no_zero(nscale_all[ip, :, ie, gsl]), axis=1)
                        if nscale_all is not None
                        else np.ones(freq.size)
                    )
                    # Channels left out at some elevation of the block (noise scale
                    # zero) are left out of the block's fit
                    blk_ok = np.all(sc[:, esl] > 0, axis=1)
                    for tt in range(nra):
                        W = w[:, tt] * isc_b * blk_ok  # inverse variance of each channel
                        m = W > 0
                        if m.sum() <= nmode + 2:
                            continue
                        Am = A[m]
                        G = Am.T @ (W[m, np.newaxis] * Am)
                        G += self.ridge * np.trace(G) / nmode * np.eye(nmode)
                        Gi = np.linalg.inv(G)
                        lev = np.zeros(freq.size)
                        lev[m] = W[m] * np.einsum("fk,kl,fl->f", Am, Gi, Am)
                        # Channels the fit nearly reproduces on their own (isolated
                        # between masked stretches) say little about the noise and
                        # make the ratio unstable: leave them out
                        use = m & (lev < self.max_leverage)
                        h1 = np.where(use, 1.0 - lev, 0.0)
                        d = vis[ip, :, ie, esl, tt]  # freq, el
                        ok = m[:, np.newaxis] & (sc[:, esl] > 0) & np.isfinite(d)
                        d = np.where(ok, d, 0.0)
                        res = d - A @ (Gi @ (Am.T @ (W[m, np.newaxis] * d[m])))
                        p2 = np.where(
                            ok & use[:, np.newaxis],
                            w[:, tt, np.newaxis] * isc[:, esl] * np.abs(res) ** 2,
                            0.0,
                        )
                        num[op, :, oe, esl, tt] = bsum @ p2
                        den[op, :, oe, esl, tt] = (bsum @ h1)[:, np.newaxis]
                        if bi_ == 0:
                            nb = bsum @ m
                            dof[op, :, oe, tt] = bsum @ h1
                            pairs = bsum @ (same & m & np.roll(m, -1))
                            corr[op, :, oe, tt] = 1.0 + (
                                2.0 * self.channel_correlation * pairs
                            ) * tools.invert_no_zero(nb)
        if flagged is not None:
            num *= ~flagged[np.newaxis, :, np.newaxis]
            den *= ~flagged[np.newaxis, :, np.newaxis]
        self.log.debug(f"Fitted across frequency in {time.time() - t0:0.1f} seconds.")

        # Sums over elevations and samples, with every elevation of a band local
        def smooth(a):
            a = mpiarray.MPIArray.wrap(a, axis=3, comm=self.comm).redistribute(1)
            loc = a.local_array
            loc = scipy.ndimage.uniform_filter1d(
                loc, 2 * self.el_window + 1, axis=3, mode="constant"
            )
            loc = scipy.ndimage.uniform_filter1d(
                loc, 2 * self.ra_window + 1, axis=4, mode="wrap" if periodic else "constant"
            )
            return mpiarray.MPIArray.wrap(loc, axis=1, comm=self.comm).redistribute(3)

        # The scale is the product of a factor for each sample, common to all
        # elevations (no smoothing in time: it follows jumps of the system
        # temperature such as solar bursts), and a smooth factor for each elevation
        # (summed over the windows after dividing out the common factor, so that
        # jumps do not leak into the neighbouring samples).  Two passes.
        nwin = (2 * self.el_window + 1) * (2 * self.ra_window + 1)
        sden = smooth(den).local_array
        enough = sden * nwin >= self.min_dof
        # Start from the residual power summed over all elevations at each sample,
        # over its running median in time: the jumps, without the slow changes
        # (which the factor of each elevation follows)
        cnum = self.comm.allreduce(
            np.sum(np.where(enough, num, 0.0), axis=3, dtype=np.float64)
        )
        cden = self.comm.allreduce(
            np.sum(np.where(enough, den, 0.0), axis=3, dtype=np.float64)
        )
        valid = cden >= self.min_common_dof
        c0 = np.where(valid, cnum * tools.invert_no_zero(cden), 1.0)
        slow = median.moving_weighted_median(
            np.ascontiguousarray(c0.reshape(-1, nra)),
            np.ascontiguousarray(valid.reshape(-1, nra).astype(np.float64)),
            size=(1, 2 * self.ra_window + 1),
        ).reshape(c0.shape)
        common = np.where(valid & (slow > 0), c0 * tools.invert_no_zero(slow), 1.0)
        for _ in range(2):
            snum = smooth(
                (num * tools.invert_no_zero(common)[:, :, :, np.newaxis]).astype(
                    np.float32
                )
            ).local_array
            h = np.where(enough, snum * tools.invert_no_zero(sden), 1.0)
            # Residual power of each elevation in units of its own smooth level,
            # summed over all elevations (each weighted by its degrees of freedom)
            cnum = np.sum(
                np.where(enough, num * tools.invert_no_zero(h), 0.0),
                axis=3,
                dtype=np.float64,
            )
            cnum = self.comm.allreduce(cnum)
            common = np.where(valid, cnum * tools.invert_no_zero(cden), 1.0)

        scale = h * common[:, :, :, np.newaxis] * corr[:, :, :, np.newaxis, :]

        # Samples of a band where the noise of every elevation jumped by more than
        # flag_common (in any polarisation or baseline)
        flag = (
            np.any(common > self.flag_common, axis=(0, 2))
            if self.flag_common > 0
            else np.zeros((nband, nra), dtype=bool)
        )
        if flag.any():
            self.log.info(
                f"Flagged {flag.sum()} samples of bands ({flag.any(axis=0).sum()} "
                f"samples in at least one band, {flag.any(axis=1).sum()} bands) where "
                f"the noise of all elevations is more than {self.flag_common} times "
                "its expectation."
            )

        out = containers.HybridVisInbandNoise(
            freq=band_map,
            pol=np.array([pol[ip] for ip in ipol]),
            ew=ew[iew],
            axes_from=stream,
            attrs_from=stream,
            distributed=stream.distributed,
            comm=stream.comm,
        )
        out.redistribute("el")
        if out.scale[:].local_shape != scale.shape:
            raise RuntimeError("The scale is not distributed like the data.")
        out.scale[:].local_array[:] = scale
        out.dof[:] = dof
        out.datasets["common"][:] = common
        out.datasets["flag"][:] = flag
        out.attrs["delay_pad"] = self.delay_pad
        out.attrs["x_max"] = self.x_max
        out.attrs["el_window"] = self.el_window
        out.attrs["ra_window"] = self.ra_window
        out.attrs["channel_correlation"] = self.channel_correlation
        out.attrs["flag_common"] = self.flag_common

        good = scale[enough]
        if good.size:
            q = np.percentile(good, [16, 50, 84])
            self.log.info(
                "Scale on this rank (16/50/84th percentiles): "
                + ", ".join(f"{v:0.3f}" for v in q)
                + f"; {np.mean(enough):0.3f} of its samples have enough degrees of "
                "freedom."
            )
        return out


def _inband_factor(noise, stream, el_slc, pols, ews):
    """The scale of :class:`EstimateInbandNoise` for the channels of a stream.

    Parameters
    ----------
    noise : ch_pipeline.core.containers.HybridVisInbandNoise or None
        The scale for each band.
    stream : draco.core.containers.HybridVisStream
        Stream distributed over elevation.
    el_slc : slice
        Local elevations of the stream.
    pols : list of str
        Polarisations wanted.
    ews : np.ndarray
        East-west baselines wanted.

    Returns
    -------
    factor : np.ndarray[pol, nband + 1, ew, el_local, ra] or None
        The scale of each band, with a last entry of ones.
    chan_band : np.ndarray[freq]
        Index into the second axis for each channel of the stream (the last entry
        for channels in no band).
    flag : np.ndarray[freq, ra] or None
        True for the samples of each channel flagged by the noise estimate.
    """
    if noise is None:
        return None, None, None
    for ax in ("ra", "el"):
        a, b = stream.index_map[ax][:], noise.index_map[ax][:]
        if a.shape != b.shape or not np.allclose(a, b):
            raise ValueError(f"The {ax} axes of the stream and the noise scale differ.")
    noise.redistribute("el")
    sc = noise.scale[:]
    if sc.local_bounds != el_slc:
        raise RuntimeError("The noise scale is not distributed like the data.")
    npol = [str(p) for p in noise.index_map["pol"]]
    new = noise.index_map["ew"][:]
    ip = [npol.index(str(p)) for p in pols]
    ie = [int(np.argmin(np.abs(new - e))) for e in ews]
    if np.any(np.abs(new[ie] - np.asarray(ews)) > 1e-6):
        raise ValueError("The noise scale does not have all of the baselines.")
    s = sc.local_array[ip][:, :, ie]  # pol, band, ew, el, ra
    ones = np.ones(s[:, :1].shape, dtype=s.dtype)
    factor = np.concatenate([s, ones], axis=1)

    fmap = noise.index_map["freq"][:]
    lo = fmap["centre"] - 0.5 * fmap["width"]
    hi = fmap["centre"] + 0.5 * fmap["width"]
    freq = stream.freq[:]
    chan_band = np.full(freq.size, fmap.size, dtype=int)
    for bb in range(fmap.size):
        chan_band[(freq >= lo[bb]) & (freq < hi[bb])] = bb
    flag = None
    if "flag" in noise.datasets:
        fb = noise.datasets["flag"][:]
        if np.any(fb):
            fb = np.concatenate([fb, np.zeros((1, fb.shape[1]), dtype=bool)], axis=0)
            flag = fb[chan_band]
    return factor, chan_band, flag


def _merge_flags(rfi, flag, nel_local):
    """Add flags of channels and samples [freq, ra] to a local RFI mask [freq, el, ra]."""
    if flag is None:
        return rfi
    full = np.broadcast_to(flag[:, np.newaxis, :], (flag.shape[0], nel_local, flag.shape[1]))
    return full.copy() if rfi is None else rfi | full


def _optional_inputs(*args):
    """Sort optional requirements into an RFI mask and an in-band noise scale."""
    mask, noise = None, None
    for a in args:
        if a is None:
            continue
        if isinstance(a, containers.HybridVisInbandNoise):
            noise = a
        elif isinstance(a, draco_containers.LocalizedSiderealRFIMask):
            mask = a
        else:
            raise TypeError(f"Unexpected requirement {type(a).__name__}.")
    return mask, noise


class EstimateSkyNoise(_BandMixin, tasklib.base.ContainerTask):
    """Fit the noise of m-mode filtered visibilities that scales with the sky.

    Bright sources leave a residual after the m-mode filter that is a small,
    rapidly fluctuating fraction of their unfiltered signal.  Part of it is common to
    all channels and both polarisations, and part is independent between channels
    (the self-noise of the source).  This task fits the model described in
    :class:`ch_pipeline.core.containers.HybridVisSkyNoise`, separately for each band
    and east-west baseline, using the unfiltered visibilities as the sky term.

    Three statistics are fitted in turn, by maximum likelihood for exponentially
    distributed powers (iteratively reweighted least squares), leaving out samples
    whose power is more than ``clip`` times the model:

    1. the power of single channels, for ``alpha_single`` and ``kappa2_tot``;
    2. the power of the sum over the channels in a band, weighted by their inverse
       noise variance, for each polarisation, for ``alpha_band`` and ``kappa2_coh``
       with ``kappa2_tot`` fixed;
    3. the power of that sum over both polarisations, for ``alpha_pol`` and
       ``rho_pol`` with the others fixed;
    4. the product of the band sums of two bands, for ``rho_band``, the
       correlation of the common term between the bands.  Its expectation is
       ``rho_band kappa_coh,b kappa_coh,b' Rb_b Rb_b'^*``, with ``kappa_coh`` from
       the block fits, and it is fitted by weighted least squares over the samples
       where the sky term is significant in both bands, with a jackknife error
       over the blocks of right ascension.

    Samples whose sky term is negligible (below 1e-3 of the noise for a coefficient
    of ``kappa_prior``) only constrain ``alpha``, so a random fraction
    ``noise_fraction`` of them is used.  Band sums use the samples with at least
    two valid channels.  The weights (inverse variances) for the whole day are from
    a jackknife over the blocks of right ascension.  Those of the fits within blocks
    and around sources are from the Fisher matrix, which ignores the correlation of
    neighbouring elevations and channels, so they overestimate the precision.
    Coefficients that could not be fitted have zero weight.

    The 0 m baseline is excluded by default: its co-polar visibilities are real and
    its unfiltered visibilities are dominated by the receiver noise rather than the
    sky.  If an RFI mask is given (see :class:`FlagMovingRFI`), the flagged
    samples are left out.

    Attributes
    ----------
    pol : list of str
        Polarisations to use.  Default is ["XX", "YY"].
    ew_index : list of int
        Indices of the east-west baselines to fit.  Default is all except 0 m.
    nblock : int
        Number of blocks of right ascension.  Default is 12.
    el_edges : list of float
        Edges of bands of telescope-y.  If given, the coefficients are also fitted in
        cells of RA block and telescope-y band (the ``_cell`` datasets).  Rows
        outside the edges belong to the nearest band.  Default is none.
    clip : float
        Leave out samples with power above this multiple of the model.  Default 20.
    niter : int
        Maximum number of reweighting iterations.  Default is 20.
    kappa_prior : float
        Typical coefficient, used to select the samples with a significant sky term.
        Default is 1e-3.
    noise_fraction : float
        Fraction of the other samples to use.  Default is 0.02.
    sources : list of str
        Sources (names in `ch_ephem.sources.source_dictionary`, or "sun") around
        whose transit the coefficients are also fitted.  Default is none.
    positions : dict
        Additional positions, as {name: [ra, dec]} in degrees.
    source_el_width : int
        Half-width of the window around each source in elevations.  Default is 10.
    source_ra_width : float
        Half-width of the window around each source in degrees.  Default is 3.
    qmin, qmax : float
        Range of the sky term (in units of the noise) for the binned diagnostics.
        Defaults are 1e-2 and 1e8.
    nqbin : int
        Number of logarithmic bins.  The first bin also contains everything below
        ``qmin``.  Default is 40.
    seed : int
        Seed for the random selection of samples.  Default is 0.
    """

    pol = config.list_type(str, default=["XX", "YY"])
    ew_index = config.list_type(int, default=None)
    nblock = config.Property(proptype=int, default=12)
    el_edges = config.list_type(float, default=None)
    clip = config.Property(proptype=float, default=20.0)
    niter = config.Property(proptype=int, default=20)
    kappa_prior = config.Property(proptype=float, default=1e-3)
    noise_fraction = config.Property(proptype=float, default=0.02)
    sources = config.list_type(str, default=[])
    positions = config.Property(proptype=dict, default={})
    source_el_width = config.Property(proptype=int, default=10)
    source_ra_width = config.Property(proptype=float, default=3.0)
    qmin = config.Property(proptype=float, default=1e-2)
    qmax = config.Property(proptype=float, default=1e8)
    nqbin = config.Property(proptype=int, default=40)
    seed = config.Property(proptype=int, default=0)

    rfi_mask = None
    inband_noise = None

    def setup(self, mask=None, noise=None):
        """Set the RFI mask and the in-band noise scale, in either order.

        Parameters
        ----------
        mask : draco.core.containers.LocalizedSiderealRFIMask, optional
            Mask of bands, elevations and samples to leave out, for example the
            output of :class:`FlagMovingRFI`.
        noise : ch_pipeline.core.containers.HybridVisInbandNoise, optional
            Local scale of the noise (:class:`EstimateInbandNoise`), which
            multiplies the noise expected from the weights and the noise scale.
        """
        self.rfi_mask, self.inband_noise = _optional_inputs(mask, noise)

    def process(self, stream, unfiltered):
        """Fit the coefficients.

        Parameters
        ----------
        stream : ch_pipeline.core.containers.MFilteredHybridVisStream
            m-mode filtered visibilities, with the ``noise_scale`` dataset if the
            noise is to be measured relative to it.
        unfiltered : draco.core.containers.HybridVisStream
            The visibilities before filtering, on the same grid.

        Returns
        -------
        out : ch_pipeline.core.containers.HybridVisSkyNoise
            The fitted coefficients and diagnostics.
        """
        from ch_ephem.observers import chime

        # Check that the two streams are on the same grid
        ra = stream.ra[:]
        el = stream.index_map["el"][:]
        if ra.shape != unfiltered.ra.shape or not np.allclose(ra, unfiltered.ra[:]):
            raise ValueError("The RA axes of the streams differ.")
        if el.shape != unfiltered.index_map["el"].shape or not np.allclose(
            el, unfiltered.index_map["el"][:]
        ):
            raise ValueError("The el axes of the streams differ.")
        freq = stream.freq[:]
        if freq.shape != unfiltered.freq.shape or not np.allclose(
            freq, unfiltered.freq[:]
        ):
            raise ValueError("The frequency axes of the streams differ.")

        fpol = list(stream.index_map["pol"])
        upol = list(unfiltered.index_map["pol"])
        ipf = [fpol.index(p) for p in self.pol]
        ipu = [upol.index(p) for p in self.pol]

        ew = stream.index_map["ew"][:]
        ewu = unfiltered.index_map["ew"][:]
        iew = (
            [ii for ii in range(ew.size) if ew[ii] != 0.0]
            if self.ew_index is None
            else list(self.ew_index)
        )
        iewu = []
        for ii in iew:
            match = np.flatnonzero(np.isclose(ewu, ew[ii]))
            if match.size != 1:
                raise ValueError(f"No unfiltered baseline matches ew = {ew[ii]} m.")
            iewu.append(int(match[0]))
        if any(ew[ii] == 0.0 for ii in iew):
            self.log.warning(
                "Fitting the 0 m baseline: its noise is not circular and its sky term "
                "includes the receiver noise, so the fit is not meaningful."
            )

        band_index, band_map = self._assign_bands(freq)
        nband = band_map.size

        nra = ra.size
        dra = np.median(np.diff(ra))
        block = (np.arange(nra) * self.nblock) // nra
        block_start = ra[np.searchsorted(block, np.arange(self.nblock))]

        # Source windows
        lsd = _stream_lsd(stream) if (self.sources or self.positions) else None
        if lsd is not None:
            names, src_ra, src_dec = _source_positions(
                self.sources, self.positions, lsd
            )
        else:
            names, src_ra, src_dec = [], np.zeros(0), np.zeros(0)
        nsrc = len(names)

        # Weights of the filtered data have no el axis: gather them
        stream.redistribute("freq")
        weight = stream.weight[:].allgather()  # pol, freq, ew, ra

        stream.redistribute(["el", "freq"])
        unfiltered.redistribute(["el", "freq"])
        vf = stream.vis[:].local_array
        vr = unfiltered.vis[:].local_array
        el_slc = stream.vis[:].local_bounds
        if unfiltered.vis[:].local_bounds != el_slc:
            raise RuntimeError("The streams are not distributed in the same way.")
        iel = np.arange(el.size)[el_slc]
        nel_local = iel.size

        if "noise_scale" in stream.datasets:
            ns = stream.noise_scale[:]
            if ns.local_bounds != el_slc:
                raise RuntimeError("noise_scale is not distributed like the data.")
            scale = ns.local_array  # pol, freq, ew, el
        else:
            self.log.info("No noise_scale dataset: using the weights alone.")
            scale = None
        rfi = _channel_rfi_mask(self.rfi_mask, stream, el_slc)
        gfac, gband, gflag = _inband_factor(
            self.inband_noise, stream, el_slc, self.pol, ew[iew]
        )
        rfi = _merge_flags(rfi, gflag, nel_local)

        src_mask = np.zeros((nsrc, nel_local, nra), dtype=bool)
        ra_off = ((ra[np.newaxis, :] - src_ra[:, np.newaxis] + 180.0) % 360.0) - 180.0
        src_y = np.sin(np.radians(src_dec - chime.latitude))
        for ss in range(nsrc):
            e0 = int(np.argmin(np.abs(el - src_y[ss])))
            in_el = np.abs(iel - e0) <= self.source_el_width
            in_ra = np.abs(ra_off[ss]) <= self.source_ra_width
            src_mask[ss] = in_el[:, np.newaxis] & in_ra[np.newaxis, :]

        qedges = np.concatenate(
            [[0.0], np.logspace(np.log10(self.qmin), np.log10(self.qmax), self.nqbin)]
        )

        # Bands of telescope-y for the cell fits
        if self.el_edges:
            yedges = np.asarray(self.el_edges, dtype=np.float64)
            nyb = yedges.size - 1
            ycell = np.clip(
                np.searchsorted(yedges, el[iel], side="right") - 1, 0, nyb - 1
            )
        else:
            yedges = np.array([-1.0, 1.0])
            nyb = 1
            ycell = None

        out = containers.HybridVisSkyNoise(
            freq=band_map,
            band=band_map["centre"],
            ew=ew[iew],
            block=block_start,
            source=np.array(names, dtype=str),
            qbin=qedges[1:],
            yband=yedges[:-1],
            attrs_from=stream,
        )
        out.add_dataset("rho_band")
        out.add_dataset("rho_band_weight")
        if ycell is not None:
            for name in ("kappa2_tot_cell", "kappa2_coh_cell"):
                out.add_dataset(name)
                out.add_dataset(f"{name}_weight")
            out.attrs["yband_edges"] = yedges
        self._ycell, self._nyb = ycell, nyb
        for name in out.datasets:
            out.datasets[name][:] = 0.0
        out.attrs["pol_used"] = list(self.pol)
        out.attrs["kappa_prior"] = self.kappa_prior

        # Expected self-noise: one over the number of samples in a channel
        tsamp = 86164.0905 * dra / 360.0
        fmap = stream.index_map["freq"][:]
        width = fmap["width"] if fmap.dtype.names else np.full(freq.size, np.nan)
        for bb in range(nband):
            chans = band_index == bb
            out.datasets["kappa2_self"][bb] = 1.0 / (
                np.median(width[chans]) * 1e6 * tsamp
            )

        kp2 = self.kappa_prior**2
        rng = np.random.default_rng([self.seed, self.comm.rank])

        def subsample(valid, informative):
            keep = rng.random(valid.shape, dtype=np.float32) < self.noise_fraction
            return valid & (informative | keep)

        for oe, (xf, xu) in enumerate(zip(iew, iewu)):
            # Band sums of each band, kept for the correlation between bands
            kept = {}
            for bb in range(nband):
                chans = np.flatnonzero(band_index == bb)
                t0 = time.time()

                # Gather the samples for the single-channel fit, and form the band
                # sums for each polarisation
                s1 = {k: [] for k in ("X", "N", "R2", "t", "e")}
                sums = []
                for jp, (pf, pu) in enumerate(zip(ipf, ipu)):
                    F = vf[pf, chans, xf]  # chan, el, ra
                    R = vr[pu, chans, xu]
                    w = weight[pf, chans, xf]  # chan, ra
                    sc = (
                        scale[pf, chans, xf]
                        if scale is not None
                        else np.ones((chans.size, nel_local), dtype=np.float32)
                    )
                    N = sc[:, :, np.newaxis] * tools.invert_no_zero(w)[:, np.newaxis, :]
                    if gfac is not None:
                        N = N * gfac[jp, gband[chans], oe]
                    valid = (
                        (w > 0)[:, np.newaxis, :]
                        & (sc > 0)[:, :, np.newaxis]
                        & np.isfinite(F)
                        & np.isfinite(R)
                    )
                    if rfi is not None:
                        valid &= ~rfi[chans]
                    X = np.abs(F) ** 2
                    R2 = np.abs(R) ** 2

                    keep = subsample(valid, R2 * kp2 > 1e-3 * N)
                    _, ke, kt = np.nonzero(keep)
                    s1["X"].append(X[keep].astype(np.float64))
                    s1["N"].append(N[keep].astype(np.float64))
                    s1["R2"].append(R2[keep].astype(np.float64))
                    s1["e"].append(ke)
                    s1["t"].append(kt)

                    a = np.where(valid, tools.invert_no_zero(N), 0.0)
                    nvalid = valid.sum(axis=0)
                    sums.append(
                        {
                            "m": np.sum(a * np.where(valid, F, 0.0), axis=0),
                            "Rb": np.sum(a * np.where(valid, R, 0.0), axis=0),
                            "A": np.sum(a, axis=0),
                            "B": np.sum(a**2 * np.where(valid, R2, 0.0), axis=0),
                            "ok": nvalid >= 2,
                        }
                    )
                    del F, R, N, X, R2, a, valid, keep

                s1 = {k: np.concatenate(v) for k, v in s1.items()}
                if self.comm.allreduce(s1["X"].size) == 0:
                    continue

                # 1. Single channels: X = alpha N + kappa2_tot |R|^2
                cols = np.stack([s1["N"], s1["R2"]], axis=1)
                fit1 = self._fit(s1["X"], cols, np.zeros_like(s1["X"]), [1.0, kp2])
                alpha1, kt2 = fit1["theta"]
                if not np.all(np.isfinite(fit1["theta"])):
                    continue
                self._store(
                    out,
                    "kappa2_tot",
                    bb,
                    oe,
                    fit1,
                    1,
                    s1["t"],
                    s1["e"],
                    block,
                    src_mask,
                )
                out.datasets["alpha_single"][bb, oe] = alpha1

                if chans.size < 2:
                    continue

                # 2. Band sums for each pol:
                #    X = alpha A + kappa2_tot B + kappa2_coh (|Rb|^2 - B)
                s2 = {k: [] for k in ("X", "A", "Q", "off", "t", "e")}
                ninf = 0
                for sm in sums:
                    rb2 = np.abs(sm["Rb"]) ** 2
                    inf = rb2 * kp2 > 1e-3 * sm["A"]
                    keep = subsample(sm["ok"], inf)
                    ninf += int((sm["ok"] & inf).sum())
                    ke, kt = np.nonzero(keep)
                    s2["X"].append(np.abs(sm["m"][keep]) ** 2)
                    s2["A"].append(sm["A"][keep])
                    s2["Q"].append(rb2[keep] - sm["B"][keep])
                    s2["off"].append(kt2 * sm["B"][keep])
                    s2["e"].append(ke)
                    s2["t"].append(kt)
                s2 = {
                    k: (
                        np.concatenate(v).astype(np.float64)
                        if k not in ("t", "e")
                        else np.concatenate(v)
                    )
                    for k, v in s2.items()
                }
                out.datasets["nsample"][bb, oe] = self.comm.allreduce(ninf)

                cols = np.stack([s2["A"], s2["Q"]], axis=1)
                fit2 = self._fit(s2["X"], cols, s2["off"], [alpha1, kp2])
                alpha2, kc2 = fit2["theta"]
                if not np.all(np.isfinite(fit2["theta"])):
                    continue
                # The block and cell fits use the kappa_tot^2 of their own block or
                # cell in the fixed part, rather than the whole-day value
                fit2["shift"] = (
                    "kappa2_tot",
                    kt2,
                    s2["off"] / kt2 if kt2 != 0 else 0.0,
                )
                self._store(
                    out,
                    "kappa2_coh",
                    bb,
                    oe,
                    fit2,
                    1,
                    s2["t"],
                    s2["e"],
                    block,
                    src_mask,
                )
                out.datasets["alpha_band"][bb, oe] = alpha2

                if kc2 > 0:
                    kept[bb] = [
                        {
                            "m": sm["m"].astype(np.complex64),
                            "Rb": sm["Rb"].astype(np.complex64),
                            "mu": (
                                alpha2 * sm["A"]
                                + kt2 * sm["B"]
                                + kc2 * (np.abs(sm["Rb"]) ** 2 - sm["B"])
                            ).astype(np.float32),
                            "A": sm["A"].astype(np.float32),
                            "ok": sm["ok"],
                        }
                        for sm in sums
                    ]

                # Binned diagnostics of the band sums, for all valid samples, in bins of
                # the modelled sky term so that each bin has a similar expected power
                cnt = np.zeros(self.nqbin)
                exc = np.zeros(self.nqbin)
                mod = np.zeros(self.nqbin)
                for sm in sums:
                    ok = sm["ok"] & (sm["A"] > 0)
                    A = sm["A"][ok]
                    rb2 = np.abs(sm["Rb"][ok]) ** 2
                    B = sm["B"][ok]
                    X = np.abs(sm["m"][ok]) ** 2
                    mu = alpha2 * A + kt2 * B + kc2 * (rb2 - B)
                    good = X < self.clip * np.maximum(mu, 0.5 * alpha2 * A)
                    q = (mu - alpha2 * A) / A
                    ib = np.clip(
                        np.searchsorted(qedges, q, side="right") - 1, 0, self.nqbin - 1
                    )[good]
                    cnt += np.bincount(ib, minlength=self.nqbin)
                    exc += np.bincount(ib, weights=(X / A)[good], minlength=self.nqbin)
                    mod += np.bincount(ib, weights=(mu / A)[good], minlength=self.nqbin)
                cnt = self.comm.allreduce(cnt)
                exc = self.comm.allreduce(exc)
                mod = self.comm.allreduce(mod)
                out.datasets["count"][bb, oe] = cnt
                out.datasets["excess"][bb, oe] = exc * tools.invert_no_zero(cnt)
                out.datasets["model"][bb, oe] = mod * tools.invert_no_zero(cnt)

                # 3. Sum over both polarisations:
                #    X = alpha (A_X + A_Y) + [fixed] + rho kappa2_coh 2 Re(Rb_X Rb_Y*)
                if len(sums) == 2 and kc2 > 0:
                    sx, sy = sums
                    ok = sx["ok"] & sy["ok"]
                    M = sx["m"] + sy["m"]
                    A = sx["A"] + sy["A"]
                    fixed = sum(
                        kt2 * s["B"] + kc2 * (np.abs(s["Rb"]) ** 2 - s["B"])
                        for s in (sx, sy)
                    )
                    cross = 2.0 * (sx["Rb"] * np.conj(sy["Rb"])).real
                    rb2 = np.abs(sx["Rb"] + sy["Rb"]) ** 2
                    keep = subsample(ok, rb2 * kp2 > 1e-3 * A)
                    ke, kt = np.nonzero(keep)
                    cols = np.stack([A[keep], cross[keep]], axis=1).astype(np.float64)
                    fit3 = self._fit(
                        (np.abs(M[keep]) ** 2).astype(np.float64),
                        cols,
                        fixed[keep].astype(np.float64),
                        [alpha2, kc2],
                    )
                    if np.all(np.isfinite(fit3["theta"])):
                        err = self._jackknife(fit3, 1, block[kt]) / kc2
                        out.datasets["alpha_pol"][bb, oe] = fit3["theta"][0]
                        out.datasets["rho_pol"][bb, oe] = fit3["theta"][1] / kc2
                        out.datasets["rho_pol_weight"][bb, oe] = _inv_var(err)

                self.log.debug(
                    f"ew {ew[xf]:.0f} m, band {band_map['centre'][bb]:.1f} MHz: "
                    f"kappa_tot^2 {kt2:.3g}, kappa_coh^2 {kc2:.3g} "
                    f"({time.time() - t0:.1f} s)"
                )

            # 4. Correlation of the common term between bands
            t0 = time.time()
            self._fit_rho_band(out, oe, kept, block)
            self.log.info(
                f"ew {ew[xf]:.0f} m: correlation between {len(kept)} bands "
                f"({time.time() - t0:.1f} s)"
            )
            del kept

        return out

    def _fit_rho_band(self, out, oe, kept, block):
        """Fit the correlation of the common term between every pair of bands.

        For bands ``b`` and ``b'`` the cross product of the band sums has expectation
        ``rho k_b k_b' Rb_b Rb_b'^*``, where ``k`` is ``kappa_coh`` in the block of each
        sample.  ``rho`` is the weighted least-squares fit over samples where the sky
        term of both bands is significant, with weights ``1 / (mu_b mu_b')`` from the
        modelled power of each band sum.  The error is a jackknife over blocks.
        """
        d = out.datasets
        nblk = self.nblock
        bands = sorted(kept)
        nb = len(bands)
        for bb in range(d["rho_band"].shape[0]):
            d["rho_band"][bb, bb, oe] = 1.0
        if nb < 2:
            return

        # kappa_coh of each band in the block of each sample
        kblk = {}
        for bb in bands:
            k2 = np.where(
                d["kappa2_coh_block_weight"][bb, oe] > 0,
                d["kappa2_coh_block"][bb, oe],
                d["kappa2_coh"][bb, oe],
            )
            kblk[bb] = np.sqrt(np.clip(k2, 0.0, None))[block]  # ra

        kp2 = self.kappa_prior**2
        pairs = [(b1, b2) for i, b1 in enumerate(bands) for b2 in bands[i + 1 :]]
        num = np.zeros((len(pairs), nblk))
        den = np.zeros((len(pairs), nblk))
        for ip, (b1, b2) in enumerate(pairs):
            k12 = (kblk[b1] * kblk[b2])[np.newaxis, :]
            for s1, s2 in zip(kept[b1], kept[b2]):
                ok = s1["ok"] & s2["ok"]
                ok &= np.abs(s1["Rb"]) ** 2 * kp2 > 1e-3 * s1["A"]
                ok &= np.abs(s2["Rb"]) ** 2 * kp2 > 1e-3 * s2["A"]
                if not ok.any():
                    continue
                wt = np.where(ok, tools.invert_no_zero(s1["mu"] * s2["mu"]), 0.0)
                q = s1["Rb"] * np.conj(s2["Rb"])  # el, ra
                y = s1["m"] * np.conj(s2["m"])
                n = (wt * (y * np.conj(q)).real * k12).sum(axis=0)
                dd = (wt * np.abs(q) ** 2 * k12**2).sum(axis=0)
                num[ip] += np.bincount(block, weights=n, minlength=nblk)
                den[ip] += np.bincount(block, weights=dd, minlength=nblk)
        num = self.comm.allreduce(num)
        den = self.comm.allreduce(den)

        tot_n, tot_d = num.sum(axis=1), den.sum(axis=1)
        for ip, (b1, b2) in enumerate(pairs):
            if tot_d[ip] <= 0:
                continue
            rho = tot_n[ip] / tot_d[ip]
            # Jackknife over the blocks that have samples
            have = den[ip] > 0
            jack = np.array(
                [
                    (tot_n[ip] - num[ip, k]) / (tot_d[ip] - den[ip, k])
                    for k in np.flatnonzero(have)
                    if tot_d[ip] - den[ip, k] > 0
                ]
            )
            if jack.size >= 2:
                n = jack.size
                err = np.sqrt((n - 1) / n * np.sum((jack - jack.mean()) ** 2))
                wgt = _inv_var(err)
            else:
                wgt = 0.0
            for i, j in ((b1, b2), (b2, b1)):
                d["rho_band"][i, j, oe] = rho
                d["rho_band_weight"][i, j, oe] = wgt

    def _fit(self, X, cols, offset, theta0):
        """Maximum-likelihood fit of mean = offset + cols @ theta for exponential X.

        Returns the parameters (NaN if the fit is singular, for example without
        samples), and what is needed to form the normal equations for
        subsets of the samples: the regressors, the final weights and the data minus
        the offset.
        """
        theta = np.array(theta0, dtype=np.float64)
        floor = 0.5 * np.abs(theta[0]) * cols[:, 0]
        for _ in range(self.niter):
            mu = np.maximum(offset + cols @ theta, floor)
            good = (mu > 0) & (X < self.clip * mu)
            wt = np.where(good, tools.invert_no_zero(mu) ** 2, 0.0)
            G = self.comm.allreduce((cols * wt[:, np.newaxis]).T @ cols)
            b = self.comm.allreduce((cols * wt[:, np.newaxis]).T @ (X - offset))
            try:
                new = np.linalg.solve(G, b)
            except np.linalg.LinAlgError:
                theta = np.full_like(theta, np.nan)
                break
            done = np.allclose(new, theta, rtol=1e-4, atol=0.0)
            theta = new
            if done:
                break
        return {
            "theta": theta,
            "wt": wt,
            "cols": cols,
            "resid": X - offset,
            "offset": offset,
        }

    def _group_fits(self, fit, ip, group, ngroup, extra=0.0):
        """Fit parameter ``ip`` separately in groups of samples.

        The other parameters are held at their whole-day values, and ``extra`` is
        added to the fixed part of the mean of each sample.  The fit is iterated
        with the weights and clipping of the mean in each group, so that a group
        whose coefficient is far from the whole-day value is not biased by the
        clipping of the whole-day fit.

        Returns
        -------
        val, info : np.ndarray[ngroup]
            The value and its Fisher information (zero without samples).
        """
        theta = fit["theta"]
        cols, offset = fit["cols"], fit["offset"]
        X = fit["resid"] + offset
        q = cols[:, ip]
        rest = offset + extra + cols @ theta - q * theta[ip]
        floor = 0.5 * np.abs(theta[0]) * cols[:, 0]
        val = np.full(ngroup, theta[ip])
        info = np.zeros(ngroup)
        for _ in range(self.niter):
            mu = np.maximum(rest + val[group] * q, floor)
            good = (mu > 0) & (X < self.clip * mu)
            wt = np.where(good, tools.invert_no_zero(mu) ** 2, 0.0)
            num = self.comm.allreduce(
                np.bincount(group, weights=wt * (X - rest) * q, minlength=ngroup)
            )
            info = self.comm.allreduce(
                np.bincount(group, weights=wt * q * q, minlength=ngroup)
            )
            new = np.where(info > 0, num * tools.invert_no_zero(info), theta[ip])
            done = np.allclose(new, val, rtol=1e-4, atol=1e-3 * abs(theta[ip]))
            val = new
            if done:
                break
        return val, info

    def _normal(self, fit, sel):
        """Normal equations restricted to a subset of the samples (summed over ranks)."""
        cw = fit["cols"][sel] * fit["wt"][sel, np.newaxis]
        G = self.comm.allreduce(cw.T @ fit["cols"][sel])
        b = self.comm.allreduce(cw.T @ fit["resid"][sel])
        return G, b

    def _jackknife(self, fit, ip, blk):
        """Jackknife error of parameter ``ip`` over the RA blocks."""
        if not np.all(np.isfinite(fit["theta"])):
            return np.nan
        G, b = self._normal(fit, np.ones(blk.size, dtype=bool))
        jack = []
        for kk in range(self.nblock):
            Gk, bk = self._normal(fit, blk == kk)
            if Gk[ip, ip] > 0:
                try:
                    jack.append(np.linalg.solve(G - Gk, b - bk)[ip])
                except np.linalg.LinAlgError:
                    pass
        if len(jack) < 2:
            return np.nan
        jack = np.array(jack)
        n = jack.size
        return np.sqrt((n - 1) / n * np.sum((jack - jack.mean()) ** 2))

    def _store(self, out, name, bb, oe, fit, ip, t, e, block, src_mask):
        """Store a coefficient with its weight, and the block, cell and source fits.

        The weight is the inverse of the jackknife variance, or zero if that could
        not be formed.  The block and cell fits are fitted again within each block or
        cell (see :meth:`_group_fits`), keeping the other parameters at the
        whole-day values; their weights are from the Fisher matrix.  The source fits
        are a single step from the whole-day fit.
        """
        theta = fit["theta"]
        out.datasets[name][bb, oe] = theta[ip]

        blk = block[t]

        def shift(sub):
            """Change of the fixed part when a held coefficient takes the values sub.

            ``fit["shift"]`` is (name, whole-day value, regressor per sample) of a
            coefficient held fixed in this fit; ``sub`` is its value in the block or
            cell of each sample.
            """
            if "shift" not in fit:
                return 0.0
            _, day, reg = fit["shift"]
            return (sub - day) * reg

        def held(kind, group):
            """The held coefficient in each block or cell, or the whole-day value."""
            if "shift" not in fit:
                return 0.0
            tname, tday, _ = fit["shift"]
            val = np.where(
                out.datasets[f"{tname}_{kind}_weight"][bb, oe] > 0,
                out.datasets[f"{tname}_{kind}"][bb, oe],
                tday,
            ).ravel()
            return shift(val[group])

        val, info = self._group_fits(fit, ip, blk, self.nblock, held("block", blk))
        out.datasets[f"{name}_block"][bb, oe] = np.where(info > 0, val, 0.0)
        out.datasets[f"{name}_block_weight"][bb, oe] = info
        out.datasets[f"{name}_weight"][bb, oe] = _inv_var(self._jackknife(fit, ip, blk))

        # Cells of RA block and telescope-y band
        if getattr(self, "_ycell", None) is not None and f"{name}_cell" in out.datasets:
            nyb = self._nyb
            cell = blk * nyb + self._ycell[e]
            val, info = self._group_fits(
                fit, ip, cell, self.nblock * nyb, held("cell", cell)
            )
            out.datasets[f"{name}_cell"][bb, oe] = np.where(info > 0, val, 0.0).reshape(
                self.nblock, nyb
            )
            out.datasets[f"{name}_cell_weight"][bb, oe] = info.reshape(self.nblock, nyb)

        for ss in range(src_mask.shape[0]):
            Gs, bs = self._normal(fit, src_mask[ss][e, t])
            if Gs[ip, ip] > 0:
                other = bs[ip] - np.dot(np.delete(Gs[ip], ip), np.delete(theta, ip))
                out.datasets[f"{name}_source"][bb, oe, ss] = other / Gs[ip, ip]
                out.datasets[f"{name}_source_weight"][bb, oe, ss] = Gs[ip, ip]


class TransientMatchedFilter(_BandMixin, tasklib.base.ContainerTask):
    """Matched filter for transients in m-mode filtered hybrid beamformed visibilities.

    Searches over the start time, duration, telescope-x, and spectral index of a
    transient in each elevation.  The template follows the source as it drifts
    through the primary beam, using the nearest x on the grid at each sample.  For
    visibilities filtered as ``(I - P) y``, the amplitude is estimated as
    ``Re(b^H W (I - P) y) / b^H W (I - P) b``, where ``b`` is the template and ``W``
    the inverse noise variance, using the elements of ``P`` near the diagonal from
    the input.  Frequencies are combined within bands, by default the 6 MHz
    television channels starting at 398 MHz, and the bands are combined for each
    spectral index.  If the input has a ``noise_scale`` dataset (see
    :class:`ApplyElevationNoiseScale`), the noise weights at each elevation are
    divided by it.

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
        Also save the imaginary part, as a null test or for signals of unknown
        phase (the cross-polar products).  Default is False.
    cross_beam : str
        Primary beam of the cross-polar products XY and YX: "copol" for the
        geometric mean of the XX and YY beams, the response to a polarised source
        on axis, or "file" for the cross-polar beams of the beam model.  Default is
        "copol".
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
    pol = config.list_type(str, default=["XX", "YY"])
    combine_pol = config.Property(proptype=bool, default=True)
    ew_index = config.list_type(int, default=None)
    x_max = config.Property(proptype=float, default=None)
    x_nfwhm = config.Property(proptype=float, default=1.0)
    x_spacing = config.Property(proptype=float, default=None)
    x_oversample = config.Property(proptype=float, default=8.0)
    save_imag = config.Property(proptype=bool, default=False)
    cross_beam = config.enum(["copol", "file"], default="copol")
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
        # Combine samples across the RA boundary only if the data are periodic and
        # the filter treated them as such
        periodic = _is_full_day(ra) and bool(hvis.attrs.get("ra_periodic", True))

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

        # Hour angle and declination of every (el, x).  Directions below the
        # horizon are given the coordinates of the horizon and zero weight.
        xx, yy = np.meshgrid(x, el, indexing="xy")
        visible = xx**2 + yy**2 < 1.0
        ha, dec = _ground_to_sph_clipped(xx, yy, self.latitude)

        # Noise weights have no elevation axis, so they remain distributed over
        # frequency when the stream is redistributed.  Gather them.
        hvis.redistribute("freq")
        wname = "noise_weight" if has_response else hvis._weight_dset_name
        if "elevation_vis_weight" in hvis.datasets:
            raise ValueError("Elevation-dependent weights are not supported.")
        weight_all = hvis[wname][:].allgather()[ipol][:, :, iew].astype(np.float32)

        # On the 0 m baseline, the co-polar hybrid visibilities are formed from each
        # baseline and its conjugate (see draco's MakeVisGrid), so they are real and
        # all of the noise is in the real part: Var(Re) = 1 / w rather than the
        # 1 / (2 w) of circular noise assumed below.  Halving their weight makes
        # the two equivalent.
        self_conj = _self_conjugate([pol[ii] for ii in ipol], ew[iew])
        weight_all *= np.where(self_conj, 0.5, 1.0).astype(np.float32)[
            :, np.newaxis, :, np.newaxis
        ]

        # Datasets without an el axis stay distributed over freq
        hvis.redistribute(["el", "freq"])
        vis = hvis.vis[:].local_array
        el_slc = hvis.vis[:].local_bounds
        el_local = el[el_slc]
        nel_local = el_local.size

        # Elevation-dependent scale of the noise variance, if present.  Dividing the
        # weights by it is equivalent to dividing the template and data by its square
        # root.
        if "noise_scale" in hvis.datasets:
            ns = hvis.noise_scale[:]
            if ns.local_bounds != el_slc:
                raise RuntimeError("noise_scale is not distributed like the data.")
            inv_sqrt_scale = tools.invert_no_zero(
                np.sqrt(ns.local_array[np.ix_(ipol, np.arange(freq.size), iew)])
            ).astype(
                np.float32
            )  # pol, freq, ew, el
        else:
            inv_sqrt_scale = None

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
        track = self._tracks(ha[el_slc], dec[el_slc], visible[el_slc], x, dra, wmax)

        # Zero template below the horizon
        pbeam *= visible[el_slc][np.newaxis, np.newaxis]

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

                if inv_sqrt_scale is not None:
                    isq = inv_sqrt_scale[:, fsel][..., esl, np.newaxis]
                    template = template * isq
                    data = data * isq

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
    def _tracks(ha, dec, visible, x, dra, wmax):
        """Index of the x grid along the track of a source starting at each (el, x).

        Tracks that start below the horizon (``visible`` is False) are invalid.

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
        ok = np.isfinite(idx) & (idx >= 0) & (idx < x.size) & visible[np.newaxis]
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

    def _beam_pol_index(self, beam, pol):
        """Indices into the beam's polarisations for each polarisation.

        A list per polarisation; with ``cross_beam`` "copol", the cross-polar
        products use the XX and YY beams, combined by :func:`_combine_beams`.
        """
        bpol = [str(p) for p in beam.pol]
        out = []
        for p in pol:
            p = str(p)
            if p[0] != p[1] and self.cross_beam == "copol":
                out.append([bpol.index("XX"), bpol.index("YY")])
            else:
                out.append([bpol.index(p)])
        return out

    def _beam_profile_hwhm(self, freq, pol):
        """Half width at half maximum of the beam in x at zenith, per frequency."""
        beam = self.beam
        beam.redistribute("freq")

        bfreq_all = beam.freq[:]
        lo = beam.beam.local_offset[0]
        nloc = beam.beam.local_shape[0]
        bfreq = bfreq_all[lo : lo + nloc]

        ipol = self._beam_pol_index(beam, pol)

        phi = np.asarray(beam.phi)
        isort = np.argsort(phi)
        phi = phi[isort]
        itheta = np.argmin(np.abs(np.asarray(beam.theta)))

        hwhm = []
        for ff, nu in enumerate(bfreq):
            if np.min(np.abs(freq - nu)) > 1e-3:
                continue
            b = beam.beam[:].local_array[ff, :, 0, itheta].real
            w = beam.weight[:].local_array[ff, :, 0, itheta]
            if not all(np.any(w[ip] > 0, axis=-1).all() for ip in ipol):
                continue
            prof = np.mean([_combine_beams(b[ip]) for ip in ipol], axis=0)[isort]
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

        ipol = self._beam_pol_index(beam, pol)

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
            for pp, ips in enumerate(ipol):
                vals = []
                for ip in ips:
                    b = bdset[lf, ip, 0][:, xsel]
                    w = wdset[lf, ip, 0][:, xsel]

                    # A zero-weight point is invalid only if its beam is non-zero
                    flag = (w > 0.0) | (np.abs(b) == 0.0)
                    if not np.any(w > 0.0):
                        break

                    bval = np.where(flag, b.real, 0.0)
                    val = scipy.interpolate.RectBivariateSpline(theta, phi, bval)(y, x)

                    if not np.all(flag):
                        fval = scipy.interpolate.RectBivariateSpline(
                            theta, phi, flag.astype(np.float32)
                        )(y, x)
                        val = np.where(np.abs(fval - 1.0) < 0.01, val, 0.0)
                    vals.append(val)
                else:
                    pb[ii, pp] = _combine_beams(np.array(vals))

        # Redistribute from frequency to elevation
        pb = mpiarray.MPIArray.wrap(pb, axis=0, comm=self.comm).redistribute(axis=2)

        return np.ascontiguousarray(pb.local_array.transpose(1, 0, 2, 3))


def _shrink_blocks(day, val, wgt, shrink=True, floor=0.0):
    """Block values of a coefficient, shrunk towards the whole-day value.

    Each block value is combined with the day value weighted by the inverse of the
    scatter of the blocks about the day value beyond their own errors.  If the
    blocks agree with the day value within their errors, every block gets the day
    value; blocks that could not be fitted always do.  The result is at least
    ``floor`` times the day value, since a fit near zero in one block is more
    likely a poor fit than an absence of the effect.
    """
    ok = (wgt > 0) & np.isfinite(val)
    out = np.full(val.shape, day, dtype=np.float64)
    if ok.any() and not shrink:
        out[ok] = val[ok]
    elif ok.any():
        var = 1.0 / wgt[ok]
        tau2 = np.mean((val[ok] - day) ** 2 - var)
        if tau2 > 0:
            out[ok] = (val[ok] / var + day / tau2) / (1.0 / var + 1.0 / tau2)
    return np.maximum(out, floor * day)


def _shrink_cells(parent, val, wgt, shrink=True):
    """Cell values [block, yband] shrunk towards the value of their block.

    As :func:`_shrink_blocks`, with the scatter of all the cells about their block
    values.  Cells that could not be fitted get the value of their block.
    """
    ok = (wgt > 0) & np.isfinite(val)
    out = np.broadcast_to(parent[:, np.newaxis], val.shape).astype(np.float64).copy()
    if ok.any() and not shrink:
        out[ok] = val[ok]
    elif ok.any():
        var = 1.0 / wgt[ok]
        par = out[ok]
        tau2 = np.mean((val[ok] - par) ** 2 - var)
        if tau2 > 0:
            out[ok] = (val[ok] / var + par / tau2) / (1.0 / var + 1.0 / tau2)
    return out


def _nearest_psd_correlation(rho):
    """Clip the negative eigenvalues of a correlation matrix and renormalise it."""
    lam, vec = np.linalg.eigh(0.5 * (rho + rho.T))
    c = (vec * np.clip(lam, 0.0, None)) @ vec.T
    d = np.sqrt(np.clip(np.diag(c), 1e-12, None))
    return c / d[:, np.newaxis] / d[np.newaxis, :]


class TransientMatchedFilterSkyNoise(TransientMatchedFilter):
    """Matched filter including the noise that scales with the sky.

    The filtered data are modelled as thermal noise plus the residual fitted by
    :class:`EstimateSkyNoise`: for visibility ``k`` (polarisation, channel and
    baseline) with unfiltered visibility ``S_k``, ``kappa_coh e S_k + kappa_inc e_k
    S_k``.  ``e`` is common to the channels and baselines of a band, with
    correlation ``rho_pol`` between polarisations and ``rho_band`` between bands,
    and ``e_k`` is independent.  Both are assumed independent between samples.

    The weights are the inverse of the total variance of each visibility,
    ``W_k = 1 / (sigma_k^2 + kappa_tot^2 |S_k|^2)``, where ``sigma_k^2`` is the
    thermal variance from the weights and the noise scale.  For a template ``g`` at
    one sample the numerator is ``N = sum_k g_k^* W_k d_k`` and the response to a
    transient of unit amplitude is ``R = sum_k |g_k|^2 W_k (1 - P_kk)``, so the
    amplitude is ``Re N / R`` and

        Var(Re N) = 1/2 [sum_k |g_k|^2 W_k^2 (sigma_k^2 (1 - P_kk) + kappa_inc^2 |S_k|^2)
                         + sum_bb' rho_bb' Re(V_b V_b'^*)]

    where ``V_b`` is ``kappa_coh`` times the numerator of band ``b`` applied to the
    unfiltered data, combined over the polarisations with ``rho_pol``.  For each
    spectral index the bands are weighted by ``s_b = (nu_b / nu_ref)^alpha`` as in
    :class:`TransientMatchedFilter`.  With ``kappa = 0`` the result is that of
    :class:`TransientMatchedFilter`.

    For a duration of W samples the template follows the source along its track as
    in :class:`TransientMatchedFilter`.  The numerator, the diagonal of the response
    and the variance above are summed along the track, and the filter couples the
    samples of the track: the response gets
    ``sum_k sum_{j != j'} g_kj^* g_kj' W_k(t_j) (-P_k[t_j, t_j'])`` and the variance
    the thermal ``sum_k sum_{j != j'} g_kj^* g_kj' W_k(t_j) (-P_k[t_j, t_j'])
    sigma_k^2(t_j') W_k(t_j')``, since the thermal noise after the filter has
    covariance ``(delta - P[t, t']) sigma^2(t')``.  The sky term is taken as
    independent between samples (it is white at 10 s after the filter).  These need
    the filter response out to lag W - 1.  The per-band output and the group maps
    are single-sample only (``band_durations`` must be [1]), and with the local
    rescaling ``local_gap`` must be at least W - 1, so that a pulse does not raise
    its own noise.

    The sub-band maps of :class:`CollapseBandsToGroups` are made here too, since
    the correlation between bands is needed for their noise.  Each entry of
    ``group_x_reduce`` gives one ``TransientBandGroupMap``, returned after the
    trials and the bands.  The single-band maps can still be made from the band
    output with :class:`CollapseBandsToGroups`.

    The coefficients are the fits within blocks of right ascension, shrunk towards
    the whole-day value (see :func:`_shrink_blocks`), or, if the fit has them, the
    fits within cells of RA block and telescope-y band, shrunk towards the value of
    their block (see :func:`_shrink_cells`).  They are kept above ``kappa_floor``
    times the whole-day value, interpolated linearly between the block centres, and
    constant within each band of telescope-y.  Bands or baselines without a fit have
    no sky term.

    If an RFI mask is given (see :class:`FlagMovingRFI`), the flagged samples
    are given zero weight, here and in the local rescaling below.  If a local
    noise scale is given (see :class:`EstimateInbandNoise`), it multiplies the
    thermal variance ``sigma_k^2``, here and in the local rescaling.

    Attributes
    ----------
    groups : dict
        Groups of bands, as {name: [[lo, hi], ...]} in MHz (see
        :class:`CollapseBandsToGroups`).  Default is none.
    group_x_reduce : list
        For each group map, "max" for the best telescope-x or a value of telescope-x.
        Default is ["max"].
    group_spectral_index : float
        Spectral index of the group maps.  Default is 0.
    shrink_blocks : bool
        Shrink the block fits towards the whole-day value.  Default is True.
    kappa_floor : float
        Lower limit of the block values of ``kappa^2``, as a fraction of the
        whole-day value.  Default is 0.1.
    rho_band_default : float
        Correlation between bands where it was not measured.  Default is 1, which
        overestimates the noise of band sums if the bands decorrelate.
    rho_pol_default : float
        Correlation between polarisations where it was not measured.  Default is 1.
    local_window : int
        Half-width in samples of the window of the local rescaling of the sky term
        (see below).  Zero turns it off.  Default is 30.
    local_gap : int
        Samples within this many of the sample being rescaled are left out of its
        window, so that a transient shorter than this does not raise its own noise.
        Default is 5.
    local_prior : float
        Weight of the prior r = 1, in units of a sample dominated by the sky.
        Default is 2.
    local_cap : float
        Each sample contributes at most this multiple of its modelled power.
        Default is 20.
    local_min : float
        Lower limit of the rescaling.  Default is 0.1.
    inband_drop_inc : bool
        With a local noise scale from :class:`EstimateInbandNoise`, which measures
        the noise that is independent between channels (including the self-noise
        of bright sources), leave out the incoherent sky term ``kappa_inc`` (use
        ``kappa_tot = kappa_coh``) so that it is not counted twice.  Default is
        True.

    The sky term changes faster than the block or cell fits can follow (on some
    nights over a few minutes).  So it is rescaled at each sample by the ratio of
    the measured to the modelled power of the sky term in the samples around it in
    the same row, for each band:

        r = (sum_j u_j (min(X_j / mu_j, cap) - th_j / mu_j) + prior) / (sum_j u_j^2 + prior)

    over ``local_gap < |j| <= local_window``, summed over polarisations and
    baselines, where X is the power of the sum of the filtered data over the
    channels of the band (weighted by the thermal noise), mu its model, th its
    thermal part, and u = (mu - th) / mu the fraction of the model from the sky.
    Samples without sky signal have u = 0, so where there is none r = 1.
    """

    groups = config.Property(proptype=dict, default={})
    group_x_reduce = config.Property(proptype=list, default=["max"])
    group_spectral_index = config.Property(proptype=float, default=0.0)
    shrink_blocks = config.Property(proptype=bool, default=True)
    kappa_floor = config.Property(proptype=float, default=0.1)
    inband_drop_inc = config.Property(proptype=bool, default=True)
    rho_band_default = config.Property(proptype=float, default=1.0)
    rho_pol_default = config.Property(proptype=float, default=1.0)
    local_window = config.Property(proptype=int, default=30)
    local_gap = config.Property(proptype=int, default=5)
    local_prior = config.Property(proptype=float, default=2.0)
    local_cap = config.Property(proptype=float, default=20.0)
    local_min = config.Property(proptype=float, default=0.1)

    rfi_mask = None

    def setup(self, manager, beam, mask=None, noise=None):
        """Set the telescope, the primary beam model, the RFI mask and noise scale.

        Parameters
        ----------
        manager : ProductManager, BeamTransfer, or TransitTelescope
            Object from which a telescope instance is extracted.
        beam : draco.core.containers.GridBeam
            Primary beam model in telescope coordinates.
        mask : draco.core.containers.LocalizedSiderealRFIMask, optional
            Mask of bands, elevations and samples to leave out, for example the
            output of :class:`FlagMovingRFI`.
        noise : ch_pipeline.core.containers.HybridVisInbandNoise, optional
            Local scale of the thermal noise (:class:`EstimateInbandNoise`).  The
            mask and the noise scale may be given in either order.
        """
        super().setup(manager, beam)
        self.rfi_mask, self.inband_noise = _optional_inputs(mask, noise)

    def process(self, hvis, unfiltered, sky):
        """Apply the matched filter.

        Parameters
        ----------
        hvis : ch_pipeline.core.containers.MFilteredHybridVisStream
            Output of :class:`LSQMFilterHybridVis`.
        unfiltered : draco.core.containers.HybridVisStream
            The visibilities before filtering, on the same grid.
        sky : ch_pipeline.core.containers.HybridVisSkyNoise
            Output of :class:`EstimateSkyNoise` for the same data.

        Returns
        -------
        trials : ch_pipeline.core.containers.TransientMatchedFilterTrials
            Result for each spectral index.
        bands : ch_pipeline.core.containers.TransientMatchedFilter
            Result for each band.
        groups : ch_pipeline.core.containers.TransientBandGroupMap
            One for each entry of ``group_x_reduce``, if ``groups`` is set.
        """
        durations = sorted(set(self.durations))
        if min(durations) < 1:
            raise ValueError("Durations must be at least one sample.")
        if sorted(set(self.band_durations)) != [1] or durations[0] != 1:
            raise ValueError(
                "With the sky noise, durations must include 1 and band_durations must "
                "be [1]."
            )
        wmax = max(durations)
        if self.local_window > 0 and (
            self.local_gap < wmax - 1 or self.local_window <= self.local_gap
        ):
            raise ValueError(
                f"Durations up to {wmax} samples need local_gap >= {wmax - 1} (and "
                "local_window > local_gap), or local_window = 0."
            )
        alphas = np.array(self.spectral_indices, dtype=np.float64)

        _check_same_grid(hvis, unfiltered)

        has_response = "ra_response" in hvis.datasets
        if has_response:
            lag_index = {d: ii for ii, d in enumerate(hvis.index_map["lag"])}
        if wmax > 1 and (not has_response or max(hvis.index_map["lag"]) < wmax - 1):
            raise ValueError(
                f"Durations up to {wmax} samples require the filter response out to "
                f"lag {wmax - 1}."
            )

        pol = list(hvis.index_map["pol"])
        ipol = [pol.index(p) for p in self.pol]
        upol = list(unfiltered.index_map["pol"])
        ipu = [upol.index(pol[ii]) for ii in ipol]
        pgroups = (
            [list(range(len(ipol)))]
            if self.combine_pol
            else [[pp] for pp in range(len(ipol))]
        )
        opol = np.array(["co"]) if self.combine_pol else np.array(self.pol)
        ngp = len(pgroups)

        ew = hvis.index_map["ew"][:]
        iew = list(range(ew.size)) if self.ew_index is None else list(self.ew_index)
        ewu = unfiltered.index_map["ew"][:]
        iewu = []
        for ii in iew:
            match = np.flatnonzero(np.isclose(ewu, ew[ii]))
            if match.size != 1:
                raise ValueError(f"No unfiltered baseline matches ew = {ew[ii]} m.")
            iewu.append(int(match[0]))

        freq = hvis.freq[:]
        band_index, band_map = self._assign_bands(freq)
        nband = len(band_map)
        scale = (band_map["centre"][:, np.newaxis] / self.reference_freq) ** alphas

        el = hvis.index_map["el"][:]
        ra = hvis.ra[:]
        nra = ra.size
        dra = np.radians(np.median(np.diff(ra)))

        x = self._x_grid(freq, ew[iew])
        nx = x.size
        self.log.info(f"Using {nx} telescope-x samples spanning +/-{x[-1]:0.4f}.")
        pbeam = self._evaluate_beam(freq, [pol[ii] for ii in ipol], el, x)

        xx, yy = np.meshgrid(x, el, indexing="xy")
        visible = xx**2 + yy**2 < 1.0
        ha, dec = _ground_to_sph_clipped(xx, yy, self.latitude)

        # Coefficients of the sky term, interpolated onto the RA axis
        k2coh, k2tot, yband, rho_pol, rho_band = self._sky_coefficients(
            sky, band_map, ew[iew], ra, el
        )
        if getattr(self, "inband_noise", None) is not None and self.inband_drop_inc:
            self.log.info(
                "Leaving out the incoherent sky term: the local noise scale measures it."
            )
            k2tot = k2coh.copy()
        k2inc = np.clip(k2tot - k2coh, 0.0, None)
        kcoh = np.sqrt(k2coh)  # band, ew, yband, ra
        lam, qvec = self._modes(rho_band)

        # Thermal weights, gathered over frequency
        hvis.redistribute("freq")
        wname = "noise_weight" if has_response else hvis._weight_dset_name
        if "elevation_vis_weight" in hvis.datasets:
            raise ValueError("Elevation-dependent weights are not supported.")
        weight_all = hvis[wname][:].allgather()[ipol][:, :, iew].astype(np.float32)
        self_conj = _self_conjugate([pol[ii] for ii in ipol], ew[iew])
        weight_all *= np.where(self_conj, 0.5, 1.0).astype(np.float32)[
            :, np.newaxis, :, np.newaxis
        ]

        hvis.redistribute(["el", "freq"])
        unfiltered.redistribute(["el", "freq"])
        vis = hvis.vis[:].local_array
        uvis = unfiltered.vis[:].local_array
        el_slc = hvis.vis[:].local_bounds
        if unfiltered.vis[:].local_bounds != el_slc:
            raise RuntimeError("The streams are not distributed in the same way.")
        el_local = el[el_slc]
        nel_local = el_local.size
        yband_local = yband[el_slc]

        if "noise_scale" in hvis.datasets:
            ns = hvis.noise_scale[:]
            if ns.local_bounds != el_slc:
                raise RuntimeError("noise_scale is not distributed like the data.")
            nscale = ns.local_array[np.ix_(ipol, np.arange(freq.size), iew)]
        else:
            nscale = np.ones((len(ipol), freq.size, len(iew), nel_local), np.float32)
        rfi = _channel_rfi_mask(self.rfi_mask, hvis, el_slc)
        gfac, gband, gflag = _inband_factor(
            getattr(self, "inband_noise", None),
            hvis,
            el_slc,
            [pol[ii] for ii in ipol],
            ew[iew],
        )
        rfi = _merge_flags(rfi, gflag, nel_local)

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
            freq=band_map, duration=np.array([1]), **common
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
            cont.attrs["sky_noise"] = True
        trials.attrs["reference_freq"] = self.reference_freq
        bands.attrs["band_start"] = self.band_start
        bands.attrs["band_width"] = self.band_width

        # Per-band sums kept for combining the bands: [pol group, band, el, x, ra]
        shp = (ngp, nband, nel_local, nx, nra)
        numer = np.zeros(shp, dtype=np.float32)
        numer_im = np.zeros(shp, dtype=np.float32) if self.save_imag else None
        resp = np.zeros(shp, dtype=np.float32)
        vinc = np.zeros(shp, dtype=np.float32)
        vplus = np.zeros(shp, dtype=np.complex64)
        two_pol = any(len(g) == 2 for g in pgroups)
        vminus = np.zeros(shp, dtype=np.complex64) if two_pol else None

        pbeam *= visible[el_slc][np.newaxis, np.newaxis]
        lmbda = scipy.constants.c / (freq * 1e6)
        u = ew[iew][np.newaxis, :] / lmbda[:, np.newaxis]  # freq, ew

        # For durations longer than one sample: the tracks, and the coupling of the
        # samples of a track by the filter, summed over the bands for each spectral
        # index [spectral index, duration, pol group, el, x, ra]
        long_w = [W for W in durations if W > 1]
        periodic = _is_full_day(ra) and bool(hvis.attrs.get("ra_periodic", True))
        if long_w:
            track = self._tracks(ha[el_slc], dec[el_slc], visible[el_slc], x, dra, wmax)
            cshp = (alphas.size, len(long_w), ngp, nel_local, nx, nra)
            cross_r = np.zeros(cshp, dtype=np.float32)
            cross_v = np.zeros(cshp, dtype=np.float32)

        for bb in range(nband):
            fsel = np.flatnonzero(band_index == bb)
            wb = weight_all[:, fsel]  # pol, freq, ew, ra
            if has_response:
                resp_t = self._gather_freq(hvis.ra_response, fsel)[ipol][:, :, iew]
                one_m_p = (1.0 - resp_t[..., lag_index[0]].real).astype(np.float32)
            else:
                one_m_p = np.ones(wb.shape, dtype=np.float32)
            inv_w = tools.invert_no_zero(wb)[:, :, :, np.newaxis, :]
            omp = one_m_p[:, :, :, np.newaxis, :]
            # In-band scale of the thermal noise [pol, freq, ew, el, ra]
            gb = gfac[:, gband[fsel]] if gfac is not None else None

            # Local rescaling of the sky term [el, ra]
            rloc = self._local_scale(
                vis,
                uvis,
                nscale,
                rfi,
                inv_w,
                fsel,
                ipol,
                ipu,
                iew,
                iewu,
                k2coh[bb],
                k2tot[bb],
                yband_local,
                gb,
            )

            for e0 in range(0, nel_local, self.nel_chunk):
                esl = slice(e0, min(e0 + self.nel_chunk, nel_local))
                yel = el_local[esl]
                # Coefficients for these elevations: [(pol), (freq), ew, el, ra]
                yb = yband_local[esl]
                rl = rloc[esl][np.newaxis]  # (ew), el, ra
                kt = (k2tot[bb][:, yb] * rl)[np.newaxis, np.newaxis]
                ki = (k2inc[bb][:, yb] * rl)[np.newaxis, np.newaxis]
                kc = (kcoh[bb][:, yb] * np.sqrt(rl))[np.newaxis]
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
                ).astype(
                    np.complex64
                )  # pol, freq, ew, el, x

                data = vis[:, :, :, esl][
                    np.ix_(ipol, fsel, iew)
                ]  # pol, freq, ew, el, ra
                sky_v = uvis[:, :, :, esl][np.ix_(ipu, fsel, iewu)]
                var_th = nscale[:, fsel][..., esl, np.newaxis] * inv_w
                if gb is not None:
                    var_th = var_th * gb[..., esl, :]
                s2 = np.abs(sky_v) ** 2
                ok = (
                    (inv_w > 0)
                    & (nscale[:, fsel][..., esl, np.newaxis] > 0)
                    & np.isfinite(data)
                    & np.isfinite(sky_v)
                )
                if rfi is not None:
                    ok &= ~rfi[fsel][:, esl][np.newaxis, :, np.newaxis]
                wt = np.where(ok, tools.invert_no_zero(var_th + kt * s2), 0.0)
                data = np.where(ok, data, 0.0)
                sky_v = np.where(ok, sky_v, 0.0)

                wd = (wt * data).astype(np.complex64)
                wr = (wt * omp).astype(np.float32)
                wv = (wt**2 * (var_th * omp + ki * s2)).astype(np.float32)
                ws = (wt * kc * sky_v).astype(np.complex64)

                for gg, grp in enumerate(pgroups):
                    tc = template[grp].reshape(-1, *template.shape[3:])  # k, el, x
                    tch = tc.transpose(1, 2, 0).conj()  # el, x, k
                    g2 = np.abs(tch) ** 2

                    def kmat(a, g=grp):
                        return (
                            a[g].reshape(-1, *a.shape[3:]).transpose(1, 0, 2)
                        )  # el, k, ra

                    n1 = np.matmul(tch, kmat(wd))  # el, x, ra
                    numer[gg, bb, esl] = n1.real
                    if numer_im is not None:
                        numer_im[gg, bb, esl] = n1.imag
                    resp[gg, bb, esl] = np.matmul(g2, kmat(wr))
                    vinc[gg, bb, esl] = np.matmul(g2, kmat(wv))

                    if long_w:
                        cr, cv = self._track_cross(
                            tc,
                            kmat(wt),
                            kmat(var_th),
                            resp_t[grp].reshape(-1, *resp_t.shape[3:]),
                            lag_index,
                            track[:, esl],
                            long_w,
                        )
                        for iw in range(len(long_w)):
                            for ia in range(alphas.size):
                                sb2 = scale[bb, ia] ** 2
                                cross_r[ia, iw, gg, esl] += sb2 * cr[iw]
                                cross_v[ia, iw, gg, esl] += sb2 * cv[iw]

                    vp = []
                    for pp in grp:
                        tp = template[pp].reshape(-1, *template.shape[3:])
                        vp.append(
                            np.matmul(
                                tp.transpose(1, 2, 0).conj(),
                                ws[pp].reshape(-1, *ws.shape[3:]).transpose(1, 0, 2),
                            )
                        )
                    if len(vp) == 2:
                        r = rho_pol[bb]
                        vplus[gg, bb, esl] = (vp[0] + vp[1]) * np.sqrt(0.5 * (1 + r))
                        vminus[gg, bb, esl] = (vp[0] - vp[1]) * np.sqrt(0.5 * (1 - r))
                    else:
                        vplus[gg, bb, esl] = vp[0]

        # Each band on its own
        bamp = bands.amplitude[:].local_array
        bwgt = bands.weight[:].local_array
        bimag = bands.amplitude_imag[:].local_array if self.save_imag else None
        for gg in range(ngp):
            for bb in range(nband):
                var = 0.5 * (vinc[gg, bb] + np.abs(vplus[gg, bb]) ** 2)
                if vminus is not None:
                    var += 0.5 * np.abs(vminus[gg, bb]) ** 2
                amp, wgt = self._amp_weight(numer[gg, bb], resp[gg, bb], var)
                bamp[gg, 0, bb] = amp.transpose(0, 2, 1)
                bwgt[gg, 0, bb] = wgt.transpose(0, 2, 1)
                if bimag is not None:
                    im, _ = self._amp_weight(numer_im[gg, bb], resp[gg, bb], var)
                    bimag[gg, 0, bb] = im.transpose(0, 2, 1)

        # All bands, for each spectral index
        tamp = trials.amplitude[:].local_array
        twgt = trials.weight[:].local_array
        timag = trials.amplitude_imag[:].local_array if self.save_imag else None
        allb = np.arange(nband)
        for ia in range(alphas.size):
            sb = scale[:, ia]
            for gg in range(ngp):
                for ee in range(nel_local):
                    nm, rs, vr, ni = self._combine(
                        gg,
                        ee,
                        allb,
                        sb,
                        lam,
                        qvec,
                        numer,
                        numer_im,
                        resp,
                        vinc,
                        vplus,
                        vminus,
                    )
                    amp, wgt = self._amp_weight(nm, rs, vr)
                    tamp[gg, 0, ia, ee] = amp.T
                    twgt[gg, 0, ia, ee] = wgt.T
                    if timag is not None:
                        im, _ = self._amp_weight(ni, rs, vr)
                        timag[gg, 0, ia, ee] = im.T
                    if not long_w:
                        continue

                    # Longer durations: sums along the tracks, plus the coupling
                    tr = track[:, ee : ee + 1]
                    parts = [nm, rs, vr] + ([ni] if ni is not None else [])
                    sums = [
                        self._track_numerator(a[np.newaxis], tr, durations)
                        for a in parts
                    ]
                    for iw, W in enumerate(long_w):
                        iwd = durations.index(W)
                        valid = (
                            np.ones(nra, dtype=bool)
                            if periodic
                            else np.arange(nra) <= nra - W
                        )
                        rs_w = sums[1][iwd][0].real + cross_r[ia, iw, gg, ee]
                        vr_w = sums[2][iwd][0].real + 0.5 * cross_v[ia, iw, gg, ee]
                        rs_w = np.where(valid, rs_w, 0.0)
                        amp, wgt = self._amp_weight(sums[0][iwd][0].real, rs_w, vr_w)
                        tamp[gg, iwd, ia, ee] = amp.T
                        twgt[gg, iwd, ia, ee] = wgt.T
                        if timag is not None:
                            im, _ = self._amp_weight(sums[3][iwd][0].real, rs_w, vr_w)
                            timag[gg, iwd, ia, ee] = im.T

        group_maps = self._group_maps(
            hvis,
            band_map,
            x,
            opol,
            ha,
            dec,
            numer,
            resp,
            vinc,
            vplus,
            vminus,
            rho_band,
            nel_local,
        )
        return (trials, bands, *group_maps)

    @staticmethod
    def _track_cross(tc, wt, var_th, presp, lag_index, track, long_w):
        """Coupling by the filter of the different samples along the tracks.

        For each duration W in ``long_w``, the terms ``j != j'`` (both < W) of the
        response and of the thermal variance of the numerator (see the class
        docstring), for a transient starting at each sample.

        Parameters
        ----------
        tc : np.ndarray[k, el, x]
            Template at each grid x.
        wt, var_th : np.ndarray[el, k, ra]
            Total weight and thermal variance of each visibility.
        presp : np.ndarray[k, ra, lag]
            Filter response ``P[t, t + d]``.
        lag_index : dict
            Index of each lag ``d`` in ``presp``.
        track : np.ndarray[wmax, el, x]
            From :meth:`TransientMatchedFilter._tracks`.
        long_w : list of int
            Durations, all longer than one sample, in increasing order.

        Returns
        -------
        resp_cross, var_cross : list of np.ndarray[el, x, ra]
            For each duration in ``long_w``.
        """
        wmax = max(long_w)
        _, nel, nx = tc.shape
        nra = wt.shape[-1]

        # Template along the track at each step [j][el, x, k]
        g = []
        for jj in range(wmax):
            idx = track[jj]
            gj = np.take_along_axis(tc, np.maximum(idx, 0)[np.newaxis], axis=2)
            g.append(np.where(idx[np.newaxis] >= 0, gj, 0.0).transpose(1, 2, 0))
        wv = wt * var_th

        def terms(jj, jp):
            # -W(t) P[t, t + d] at t = tau + j, and the thermal variance and weight
            # at t' = tau + j'
            a_r = np.roll(
                -wt * presp[np.newaxis, :, :, lag_index[jp - jj]], -jj, axis=-1
            )
            return a_r, a_r * np.roll(wv, -jp, axis=-1)

        # Only the real part is kept, and the coefficient of (j', j) is the
        # conjugate of that of (j, j'), so each pair takes one product:
        # Re[c a + c^* b] = Re[c (a + b^*)]
        acc_r = np.zeros((nel, nx, nra), dtype=np.complex128)
        acc_v = np.zeros((nel, nx, nra), dtype=np.complex128)
        resp_cross, var_cross = [], []
        for W in range(2, wmax + 1):
            jn = W - 1
            for jj in range(jn):
                coeff = g[jj].conj() * g[jn]  # el, x, k
                a_r, a_v = terms(jj, jn)
                b_r, b_v = terms(jn, jj)
                acc_r += np.matmul(coeff, (a_r + b_r.conj()).astype(np.complex64))
                acc_v += np.matmul(coeff, (a_v + b_v.conj()).astype(np.complex64))
            if W in long_w:
                resp_cross.append(acc_r.real.astype(np.float32))
                var_cross.append(acc_v.real.astype(np.float32))

        return resp_cross, var_cross

    def _local_scale(
        self,
        vis,
        uvis,
        nscale,
        rfi,
        inv_w,
        fsel,
        ipol,
        ipu,
        iew,
        iewu,
        k2coh,
        k2tot,
        yband,
        gb=None,
    ):
        """Rescaling of the sky term of one band at each local elevation and sample.

        See the class docstring.  ``k2coh`` and ``k2tot`` are [ew, yband, ra],
        ``rfi`` is the local RFI mask [freq, el, ra] or None, and ``gb`` the
        in-band scale of the thermal noise [pol, freq, ew, el, ra] or None.

        Returns
        -------
        r : np.ndarray[el_local, ra]
        """
        nel, nra = vis.shape[3], vis.shape[4]
        if self.local_window <= 0:
            return np.ones((nel, nra), dtype=np.float32)

        num = np.zeros((nel, nra))
        den = np.zeros((nel, nra))
        for e0 in range(0, nel, self.nel_chunk):
            esl = slice(e0, min(e0 + self.nel_chunk, nel))
            yb = yband[esl]
            d = vis[:, :, :, esl][np.ix_(ipol, fsel, iew)]  # pol, freq, ew, el, ra
            sv = uvis[:, :, :, esl][np.ix_(ipu, fsel, iewu)]
            var = nscale[:, fsel][..., esl, np.newaxis] * inv_w
            if gb is not None:
                var = var * gb[..., esl, :]
            ok = (var > 0) & np.isfinite(d) & np.isfinite(sv)
            if rfi is not None:
                ok &= ~rfi[fsel][:, esl][np.newaxis, :, np.newaxis]
            a = np.where(ok, tools.invert_no_zero(var), 0.0)
            d = np.where(ok, d, 0.0)
            sv = np.where(ok, sv, 0.0)
            m = np.sum(a * d, axis=1)  # pol, ew, el, ra
            rb = np.sum(a * sv, axis=1)
            A = np.sum(a, axis=1)
            B = np.sum(a**2 * np.abs(sv) ** 2, axis=1)
            nch = ok.sum(axis=1)
            kt = k2tot[:, yb][np.newaxis]  # (pol), ew, el, ra
            kc = k2coh[:, yb][np.newaxis]
            mu = A + kt * B + kc * (np.abs(rb) ** 2 - B)
            use = (nch >= 2) & (A > 0) & (mu > 0)
            inv_mu = tools.invert_no_zero(np.where(use, mu, 0.0))
            u = np.where(use, (mu - A) * inv_mu, 0.0)
            y = np.where(
                use,
                np.minimum(np.abs(m) ** 2 * inv_mu, self.local_cap) - A * inv_mu,
                0.0,
            )
            num[esl] = np.sum(u * y, axis=(0, 1))
            den[esl] = np.sum(u * u, axis=(0, 1))

        kern = np.ones(2 * self.local_window + 1)
        g = min(self.local_gap, self.local_window)
        kern[self.local_window - g : self.local_window + g + 1] = 0.0
        num = scipy.ndimage.convolve1d(num, kern, axis=-1, mode="constant")
        den = scipy.ndimage.convolve1d(den, kern, axis=-1, mode="constant")
        r = (num + self.local_prior) / (den + self.local_prior)
        return np.maximum(r, self.local_min).astype(np.float32)

    @staticmethod
    def _amp_weight(numer, resp, var):
        """Amplitude ``Re N / R`` and its inverse variance ``R^2 / Var(Re N)``."""
        valid = (resp > 0) & (var > 0)
        amp = np.where(valid, numer * tools.invert_no_zero(resp), 0.0)
        wgt = np.where(valid, resp**2 * tools.invert_no_zero(var), 0.0)
        return amp.astype(np.float32), wgt.astype(np.float32)

    @staticmethod
    def _modes(rho):
        """Eigen-decomposition of the correlation between bands, without null modes."""
        lam, vec = np.linalg.eigh(rho)
        keep = lam > 1e-6 * max(lam.max(), 1e-30)
        return lam[keep], vec[:, keep]

    @staticmethod
    def _combine(
        gg, ee, idx, sb, lam, qvec, numer, numer_im, resp, vinc, vplus, vminus
    ):
        """Sum the bands ``idx`` with weights ``sb`` at one elevation.

        Returns the real and imaginary numerator, the response and the variance of
        the real part, each [x, ra].
        """
        s = sb[idx].astype(np.float64)
        nm = np.tensordot(s, numer[gg, idx, ee], axes=1)
        ni = (
            np.tensordot(s, numer_im[gg, idx, ee], axes=1)
            if numer_im is not None
            else None
        )
        rs = np.tensordot(s**2, resp[gg, idx, ee], axes=1)
        var = np.tensordot(s**2, vinc[gg, idx, ee], axes=1)
        coeff = (s[:, np.newaxis] * qvec) * np.sqrt(lam)[np.newaxis, :]  # band, mode
        for vv in (vplus, vminus):
            if vv is None:
                continue
            y = np.tensordot(coeff.T.astype(np.complex64), vv[gg, idx, ee], axes=1)
            var = var + np.sum(np.abs(y) ** 2, axis=0)
        return nm, rs, 0.5 * var, ni

    def _group_maps(
        self,
        hvis,
        band_map,
        x,
        opol,
        ha,
        dec,
        numer,
        resp,
        vinc,
        vplus,
        vminus,
        rho_band,
        nel_local,
    ):
        """S/N maps for the groups of bands, one for each entry of group_x_reduce."""
        if not self.groups:
            return []
        centre = band_map["centre"]
        names, used = [], []
        for name, ranges in self.groups.items():
            inc = np.zeros(centre.size, dtype=bool)
            for lo, hi in ranges:
                inc |= (centre >= lo) & (centre < hi)
            if not inc.any():
                self.log.info(f"Skipping group {name}, which contains no bands.")
                continue
            names.append(str(name))
            used.append(inc)
        used = np.array(used)

        fixed = []
        for xr in self.group_x_reduce:
            if isinstance(xr, str) and xr.strip().lower() == "max":
                fixed.append(None)
            else:
                fixed.append(int(np.argmin(np.abs(x - float(xr)))))

        outs = []
        for ix0 in fixed:
            out = containers.TransientBandGroupMap(
                group=np.array(names),
                freq=band_map,
                pol=opol,
                x=x,
                axes_from=hvis,
                attrs_from=hvis,
                distributed=hvis.distributed,
                comm=hvis.comm,
            )
            if ix0 is None:
                out.add_dataset("x_index")
            out.redistribute("el")
            out.datasets["band_used"][:] = used
            out.datasets["ha"][:] = np.degrees(ha)
            out.datasets["dec"][:] = np.degrees(dec)
            out.attrs["duration"] = 1
            out.attrs["reference_freq"] = self.reference_freq
            out.attrs["spectral_index"] = self.group_spectral_index
            out.attrs["x_reduce"] = "max" if ix0 is None else float(x[ix0])
            out.attrs["sky_noise"] = True
            out.snr[:].local_array[:] = 0.0
            if ix0 is None:
                out.datasets["x_index"][:].local_array[:] = 0
            outs.append(out)

        sb = (centre / self.reference_freq) ** self.group_spectral_index
        ngp = numer.shape[0]
        for ig, inc in enumerate(used):
            idx = np.flatnonzero(inc)
            lam, qvec = self._modes(rho_band[np.ix_(idx, idx)])
            sg = np.zeros(centre.size)
            sg[idx] = sb[idx]
            for gg in range(ngp):
                for ee in range(nel_local):
                    nm, rs, vr, _ = self._combine(
                        gg,
                        ee,
                        idx,
                        sg,
                        lam,
                        qvec,
                        numer,
                        None,
                        resp,
                        vinc,
                        vplus,
                        vminus,
                    )
                    snr = np.where(
                        (rs > 0) & (vr > 0), nm * tools.invert_no_zero(np.sqrt(vr)), 0.0
                    )  # x, ra
                    for out, ix0 in zip(outs, fixed):
                        if ix0 is None:
                            ix = np.argmax(snr, axis=0)
                            out.snr[:].local_array[gg, ig, ee] = np.take_along_axis(
                                snr, ix[np.newaxis], 0
                            )[0]
                            out.datasets["x_index"][:].local_array[gg, ig, ee] = ix
                        else:
                            out.snr[:].local_array[gg, ig, ee] = snr[ix0]
        self.log.info(f"Made maps for {len(names)} groups of bands.")
        return outs

    def _sky_coefficients(self, sky, band_map, ew_used, ra, el):
        """Coefficients of the sky term on the data's axes.

        Returns kappa_coh^2 and kappa_tot^2 [band, ew, yband, ra], the band of
        telescope-y of each elevation [el], rho_pol [band] and rho_band [band, band].
        """
        nband = len(band_map)
        new = len(ew_used)
        d = sky.datasets
        cells = "kappa2_coh_cell" in d and "kappa2_tot_cell" in d
        if cells:
            yedges = np.asarray(sky.attrs["yband_edges"], dtype=np.float64)
            nyb = yedges.size - 1
            yband = np.clip(np.searchsorted(yedges, el, side="right") - 1, 0, nyb - 1)
            self.log.info(f"Using the sky-noise fits in {nyb} bands of telescope-y.")
        else:
            nyb = 1
            yband = np.zeros(el.size, dtype=int)
        k2coh = np.zeros((nband, new, nyb, ra.size))
        k2tot = np.zeros((nband, new, nyb, ra.size))
        rho_pol = np.full(nband, self.rho_pol_default)

        sc = sky.index_map["freq"]["centre"][:]
        sew = sky.index_map["ew"][:]
        sband = np.array(
            [
                int(np.argmin(np.abs(sc - c))) if np.min(np.abs(sc - c)) < 1e-3 else -1
                for c in band_map["centre"]
            ]
        )
        sbase = np.array(
            [
                (
                    int(np.argmin(np.abs(sew - b)))
                    if np.min(np.abs(sew - b)) < 1e-3
                    else -1
                )
                for b in ew_used
            ]
        )
        if np.any(sbase < 0):
            self.log.warning(
                f"No sky-noise fit for baselines {ew_used[sbase < 0]} m: no sky term."
            )
        if np.any(sband < 0):
            self.log.warning(
                f"No sky-noise fit for {np.sum(sband < 0)} bands: no sky term."
            )

        # Block centres, using the blocks of EstimateSkyNoise on this RA axis
        nblk = sky.index_map["block"].shape[0]
        blk = (np.arange(ra.size) * nblk) // ra.size
        centres = np.array([ra[blk == k].mean() for k in range(nblk)])
        if not np.allclose(
            ra[np.searchsorted(blk, np.arange(nblk))], sky.index_map["block"][:]
        ):
            self.log.warning("The RA blocks of the sky-noise fit differ from the data.")

        for bb in range(nband):
            jb = sband[bb]
            if jb < 0:
                continue
            for ee in range(new):
                je = sbase[ee]
                if je < 0:
                    continue
                for name, arr in (("kappa2_coh", k2coh), ("kappa2_tot", k2tot)):
                    day = d[name][jb, je] if d[f"{name}_weight"][jb, je] > 0 else 0.0
                    val = _shrink_blocks(
                        day,
                        d[f"{name}_block"][jb, je],
                        d[f"{name}_block_weight"][jb, je],
                        self.shrink_blocks,
                        self.kappa_floor,
                    )
                    if cells:
                        val = _shrink_cells(
                            val,
                            d[f"{name}_cell"][jb, je],
                            d[f"{name}_cell_weight"][jb, je],
                            self.shrink_blocks,
                        )  # block, yband
                    else:
                        val = val[:, np.newaxis]
                    val = np.maximum(val, self.kappa_floor * day)
                    for yy in range(nyb):
                        arr[bb, ee, yy] = np.clip(
                            np.interp(ra, centres, val[:, yy]), 0.0, None
                        )
            w = d["rho_pol_weight"][jb][sbase[sbase >= 0]]
            if np.sum(w) > 0:
                rho_pol[bb] = np.sum(w * d["rho_pol"][jb][sbase[sbase >= 0]]) / np.sum(
                    w
                )
        k2tot = np.maximum(k2tot, k2coh)
        rho_pol = np.clip(rho_pol, -1.0, 1.0)

        rho = np.full((nband, nband), np.nan)
        if "rho_band" in d and np.all(sbase >= 0) and np.all(sband >= 0):
            r = d["rho_band"][:][np.ix_(sband, sband, sbase)]
            w = d["rho_band_weight"][:][np.ix_(sband, sband, sbase)]
            ws = w.sum(axis=-1)
            rho = np.where(
                ws > 0, np.sum(w * r, axis=-1) * tools.invert_no_zero(ws), np.nan
            )
        else:
            self.log.warning(
                f"No correlation between bands in the sky-noise fit: using {self.rho_band_default}."
            )
        # Fill the pairs without a measurement with the median at the same separation
        sep = np.abs(np.arange(nband)[:, np.newaxis] - np.arange(nband)[np.newaxis, :])
        for dd in range(1, nband):
            on = sep == dd
            have = on & np.isfinite(rho)
            fill = np.median(rho[have]) if have.any() else self.rho_band_default
            rho[on & ~np.isfinite(rho)] = fill
        np.fill_diagonal(rho, 1.0)
        rho = _nearest_psd_correlation(np.clip(rho, -1.0, 1.0))

        self.log.info(
            f"Sky term: median kappa_coh {np.median(np.sqrt(k2coh[k2coh > 0])) if (k2coh > 0).any() else 0:.2e}, "
            f"median rho_band between neighbours "
            f"{np.median(np.diag(rho, 1)) if nband > 1 else 1:.2f}."
        )
        return k2coh, k2tot, yband, rho_pol, rho


def _check_same_grid(a, b):
    """Check that two streams have the same RA, el and frequency axes."""
    for name, ax_a, ax_b in (
        ("RA", a.ra[:], b.ra[:]),
        ("el", a.index_map["el"][:], b.index_map["el"][:]),
        ("frequency", a.freq[:], b.freq[:]),
    ):
        if ax_a.shape != ax_b.shape or not np.allclose(ax_a, ax_b):
            raise ValueError(f"The {name} axes of the streams differ.")


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


class CollapseBandsToGroups(tasklib.base.ContainerTask):
    """Sum the single-sample matched filter over groups of bands.

    For each group, the bands are combined as in :class:`CollapseMatchedFilterBands`
    at every telescope-x.  Either the best S/N over x and spectral index is kept, or
    the (signed) S/N at a single telescope-x.  This gives maps that are much smaller
    than the per-band output, for example for sub-bands, or for every band on its
    own.

    Attributes
    ----------
    groups : dict
        Groups of bands, as {name: [[lo, hi], ...]} in MHz.  A band is in a group if
        its centre is in any of the ranges.  Groups without bands are left out.
    each_band : bool
        Also make a group of every band on its own, named by its centre frequency.
        Default is False.
    spectral_indices : list of float
        Spectral indices to search.  Default is [0].
    reference_freq : float
        Reference frequency for the spectral index in MHz.  Default is 600.
    duration : int
        Duration (in samples) of the trials to use.  Default is 1.
    x_reduce : str or float
        "max" to keep the best telescope-x (stored in ``x_index``), or a value of
        telescope-x to keep the S/N at the nearest x on the grid, which requires a
        single spectral index.  Default is "max".
    nel_chunk : int
        Number of elevations to process at once.  Default is 16.
    """

    groups = config.Property(proptype=dict, default={})
    each_band = config.Property(proptype=bool, default=False)
    spectral_indices = config.list_type((int, float), default=[0.0])
    reference_freq = config.Property(proptype=float, default=600.0)
    duration = config.Property(proptype=int, default=1)
    x_reduce = config.Property(default="max")
    nel_chunk = config.Property(proptype=int, default=16)

    def process(self, mf):
        """Sum over the groups of bands.

        Parameters
        ----------
        mf : ch_pipeline.core.containers.TransientMatchedFilter
            Matched filter output for each band.

        Returns
        -------
        out : ch_pipeline.core.containers.TransientBandGroupMap
            S/N for each group of bands.
        """
        mf.redistribute("el")

        x = mf.index_map["x"][:]
        if isinstance(self.x_reduce, str) and self.x_reduce.strip().lower() == "max":
            xsel = slice(None)
            fixed_x = None
        else:
            ix0 = int(np.argmin(np.abs(x - float(self.x_reduce))))
            xsel = slice(ix0, ix0 + 1)
            fixed_x = x[ix0]
            if len(self.spectral_indices) > 1:
                raise ValueError("A fixed telescope-x needs a single spectral index.")

        centre = mf.index_map["freq"]["centre"][:]
        durations = list(mf.index_map["duration"][:])
        if self.duration not in durations:
            raise ValueError(f"Duration {self.duration} is not in {durations}.")
        idur = durations.index(self.duration)

        names, used = [], []
        for name, ranges in self.groups.items():
            inc = np.zeros(centre.size, dtype=bool)
            for lo, hi in ranges:
                inc |= (centre >= lo) & (centre < hi)
            if not inc.any():
                self.log.info(f"Skipping group {name}, which contains no bands.")
                continue
            names.append(str(name))
            used.append(inc)
        if self.each_band:
            for bb, nu in enumerate(centre):
                names.append(f"{nu:.1f}")
                used.append(np.arange(centre.size) == bb)
        if not names:
            raise ValueError("No groups of bands were given.")
        used = np.array(used)
        if mf.attrs.get("sky_noise", False) and np.any(used.sum(axis=1) > 1):
            self.log.warning(
                "The bands include the noise that scales with the sky, which is "
                "correlated between bands; summing bands here ignores that.  Use the "
                "groups of TransientMatchedFilterSkyNoise instead."
            )

        alphas = np.array(self.spectral_indices, dtype=np.float64)
        scale = (centre[np.newaxis, :] / self.reference_freq) ** alphas[:, np.newaxis]

        out = containers.TransientBandGroupMap(
            group=np.array(names),
            axes_from=mf,
            attrs_from=mf,
            distributed=mf.distributed,
            comm=mf.comm,
        )
        if fixed_x is None:
            out.add_dataset("x_index")
        if len(alphas) > 1:
            out.add_dataset("spectral_index")
        out.redistribute("el")
        out.datasets["band_used"][:] = used
        out.datasets["ha"][:] = mf.datasets["ha"][:]
        out.datasets["dec"][:] = mf.datasets["dec"][:]
        out.attrs["duration"] = self.duration
        out.attrs["reference_freq"] = self.reference_freq
        out.attrs["x_reduce"] = "max" if fixed_x is None else float(fixed_x)

        amp = mf.amplitude[:].local_array[:, idur, ..., xsel]  # pol, band, el, ra, x
        wgt = mf.weight[:].local_array[:, idur, ..., xsel]

        osnr = out.snr[:].local_array
        oix = out.datasets["x_index"][:].local_array if fixed_x is None else None
        oalpha = (
            out.datasets["spectral_index"][:].local_array if len(alphas) > 1 else None
        )
        osnr[:] = 0.0
        if oix is not None:
            oix[:] = 0
        if oalpha is not None:
            oalpha[:] = 0.0

        npol, _, nel = amp.shape[:3]
        for pp in range(npol):
            for e0 in range(0, nel, self.nel_chunk):
                esl = slice(e0, min(e0 + self.nel_chunk, nel))
                for gg, inc in enumerate(used):
                    best = None
                    for aa, s in enumerate(scale):
                        # Accumulate one band at a time to avoid large temporaries
                        num = 0.0
                        den = 0.0
                        for bb in np.flatnonzero(inc):
                            w = wgt[pp, bb, esl].astype(np.float64)  # el, ra, x
                            num = num + s[bb] * w * amp[pp, bb, esl]
                            den = den + s[bb] ** 2 * w
                        snr = num * tools.invert_no_zero(np.sqrt(den))  # el, ra, x
                        ix = np.argmax(snr, axis=-1)
                        smax = np.take_along_axis(snr, ix[..., np.newaxis], -1)[..., 0]
                        if best is None:
                            best = (smax, ix, np.zeros_like(smax))
                        else:
                            better = smax > best[0]
                            best = (
                                np.where(better, smax, best[0]),
                                np.where(better, ix, best[1]),
                                np.where(better, alphas[aa], best[2]),
                            )
                    osnr[pp, gg, esl] = best[0]
                    if oix is not None:
                        oix[pp, gg, esl] = best[1]
                    if oalpha is not None:
                        oalpha[pp, gg, esl] = best[2]

        where = "best telescope-x" if fixed_x is None else f"telescope-x {fixed_x:+.4f}"
        self.log.info(f"Made maps for {len(names)} groups of bands, at the {where}.")

        return out


class CollapseBandsIncoherent(tasklib.base.ContainerTask):
    """Sum the power of the single-sample matched filter over groups of bands.

    For signals whose phase is unknown and changes from band to band, such as the
    cross-polar products: XY holds ``(U + iV)`` times the instrumental phase
    between the X and Y arrays, a delay of a few ns that turns through several
    cycles over the band but little within one band.  At each telescope-x, each band
    (and polarisation) contributes ``w (a_re^2 + a_im^2)``, with ``a_re`` and
    ``a_im`` the real and imaginary amplitudes and ``w`` their inverse variance, and
    the sum ``q`` over the bands of a group (and over the polarisations if
    ``combine_pol``) is chi-squared with ``K`` = 2 x (number of terms with data)
    degrees of freedom for noise.  It is converted to the equivalent Gaussian S/N
    by the Wilson-Hilferty approximation,

        z = ((q / K)^(1/3) - (1 - 2 / (9 K))) / sqrt(2 / (9 K)),

    and either the best z over telescope-x is kept, or z at a single telescope-x.

    The matched filter must save the imaginary part (``save_imag``).  The real and
    imaginary parts are taken to have the same variance, and the terms to be
    independent: the correlation of the noise that scales with the sky between
    bands and between polarisations is ignored.

    Attributes
    ----------
    groups : dict
        Groups of bands, as {name: [[lo, hi], ...]} in MHz.  A band is in a group if
        its centre is in any of the ranges.  Groups without bands are left out.
    each_band : bool
        Also make a group of every band on its own.  Default is False.
    duration : int
        Duration (in samples) of the trials to use.  Default is 1.
    x_reduce : str or float
        "max" to keep the best telescope-x (stored in ``x_index``), or a value of
        telescope-x.  Default is "max".
    combine_pol : bool
        Sum over the polarisations too, into one named by joining their names with
        "+".  Default is True.
    nel_chunk : int
        Number of elevations to process at once.  Default is 16.
    """

    groups = config.Property(proptype=dict, default={})
    each_band = config.Property(proptype=bool, default=False)
    duration = config.Property(proptype=int, default=1)
    x_reduce = config.Property(default="max")
    combine_pol = config.Property(proptype=bool, default=True)
    nel_chunk = config.Property(proptype=int, default=16)

    def process(self, mf):
        """Sum the power over the groups of bands.

        Parameters
        ----------
        mf : ch_pipeline.core.containers.TransientMatchedFilter
            Matched filter output for each band, with ``amplitude_imag``.

        Returns
        -------
        out : ch_pipeline.core.containers.TransientBandGroupMap
            Equivalent Gaussian S/N for each group of bands.
        """
        if "amplitude_imag" not in mf.datasets:
            raise ValueError("The matched filter output has no imaginary part.")
        mf.redistribute("el")

        x = mf.index_map["x"][:]
        if isinstance(self.x_reduce, str) and self.x_reduce.strip().lower() == "max":
            xsel = slice(None)
            fixed_x = None
        else:
            ix0 = int(np.argmin(np.abs(x - float(self.x_reduce))))
            xsel = slice(ix0, ix0 + 1)
            fixed_x = x[ix0]

        centre = mf.index_map["freq"]["centre"][:]
        durations = list(mf.index_map["duration"][:])
        if self.duration not in durations:
            raise ValueError(f"Duration {self.duration} is not in {durations}.")
        idur = durations.index(self.duration)

        names, used = [], []
        for name, ranges in self.groups.items():
            inc = np.zeros(centre.size, dtype=bool)
            for lo, hi in ranges:
                inc |= (centre >= lo) & (centre < hi)
            if not inc.any():
                self.log.info(f"Skipping group {name}, which contains no bands.")
                continue
            names.append(str(name))
            used.append(inc)
        if self.each_band:
            for bb, nu in enumerate(centre):
                names.append(f"{nu:.1f}")
                used.append(np.arange(centre.size) == bb)
        if not names:
            raise ValueError("No groups of bands were given.")
        used = np.array(used)

        pol = [str(p) for p in mf.index_map["pol"]]
        pol_groups = [list(range(len(pol)))] if self.combine_pol else [[pp] for pp in range(len(pol))]
        opol = np.array(["+".join(pol[pp] for pp in grp) for grp in pol_groups])

        out = containers.TransientBandGroupMap(
            group=np.array(names),
            pol=opol,
            axes_from=mf,
            attrs_from=mf,
            distributed=mf.distributed,
            comm=mf.comm,
        )
        if fixed_x is None:
            out.add_dataset("x_index")
        out.redistribute("el")
        out.datasets["band_used"][:] = used
        out.datasets["ha"][:] = mf.datasets["ha"][:]
        out.datasets["dec"][:] = mf.datasets["dec"][:]
        out.attrs["duration"] = self.duration
        out.attrs["x_reduce"] = "max" if fixed_x is None else float(fixed_x)
        out.attrs["statistic"] = "incoherent"

        are = mf.amplitude[:].local_array[:, idur, ..., xsel]  # pol, band, el, ra, x
        aim = mf.datasets["amplitude_imag"][:].local_array[:, idur, ..., xsel]
        wgt = mf.weight[:].local_array[:, idur, ..., xsel]

        osnr = out.snr[:].local_array
        oix = out.datasets["x_index"][:].local_array if fixed_x is None else None
        osnr[:] = 0.0
        if oix is not None:
            oix[:] = 0

        nel = are.shape[2]
        for op, grp in enumerate(pol_groups):
            for e0 in range(0, nel, self.nel_chunk):
                esl = slice(e0, min(e0 + self.nel_chunk, nel))
                for gg, inc in enumerate(used):
                    q = 0.0
                    k = 0
                    for pp in grp:
                        for bb in np.flatnonzero(inc):
                            w = wgt[pp, bb, esl].astype(np.float64)  # el, ra, x
                            q = q + w * (
                                are[pp, bb, esl].astype(np.float64) ** 2
                                + aim[pp, bb, esl].astype(np.float64) ** 2
                            )
                            k = k + 2 * (w > 0)
                    z = _chi2_to_gaussian(q, k)
                    ix = np.argmax(z, axis=-1)
                    osnr[op, gg, esl] = np.take_along_axis(z, ix[..., np.newaxis], -1)[
                        ..., 0
                    ]
                    if oix is not None:
                        oix[op, gg, esl] = ix

        where = "best telescope-x" if fixed_x is None else f"telescope-x {fixed_x:+.4f}"
        self.log.info(
            f"Made incoherent maps for {len(names)} groups of bands and polarisations "
            f"{list(opol)}, at the {where}."
        )

        return out


def _chi2_to_gaussian(q, k):
    """Equivalent Gaussian S/N of a chi-squared value ``q`` with ``k`` dof.

    The Wilson-Hilferty approximation; zero where ``k`` is zero.
    """
    k = np.asarray(k, dtype=np.float64)
    ok = k > 0
    kk = np.where(ok, k, 1.0)
    c = 2.0 / (9.0 * kk)
    z = (np.cbrt(np.where(ok, q, 0.0) / kk) - (1.0 - c)) / np.sqrt(c)
    return np.where(ok, z, 0.0)


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


class FillRAGaps(tasklib.base.ContainerTask):
    """Insert the samples missing from gaps in the RA axis, with zero weight.

    The data of some days have gaps (dropped integrations, or hours without data)
    in an otherwise uniform time axis, which the m-mode filters do not accept.
    Every step that is close to a whole number of samples is filled with the
    missing samples, which are zero in every dataset with an RA axis (so they have
    zero weight), except ``effective_ra``, which is set to the RA of the sample.  A
    step that is not close to a whole number of samples is an error.  Without gaps
    the input is returned unchanged.

    Attributes
    ----------
    tolerance : float
        Largest difference from a whole number of samples, as a fraction of the
        sample spacing.  Default is 0.01.
    """

    tolerance = config.Property(proptype=float, default=0.01)

    def process(self, cont):
        """Fill the gaps.

        Parameters
        ----------
        cont : container with an ``ra`` axis, distributed over another axis
            For example a draco.core.containers.HybridVisStream.

        Returns
        -------
        out : container of the same type
            With a uniform RA axis.
        """
        ra = cont.ra[:]
        step = np.diff(ra)
        dra = np.median(step)
        nstep = np.rint(step / dra).astype(int)
        if np.any(np.abs(step - nstep * dra) > self.tolerance * dra) or np.any(
            nstep < 1
        ):
            bad = np.flatnonzero(np.abs(step - nstep * dra) > self.tolerance * dra)
            raise ValueError(
                f"{bad.size} steps of the RA axis are not a whole number of samples, "
                f"for example {step[bad[:3]] / dra} samples."
            )
        if np.all(nstep == 1):
            self.log.info("The RA axis has no gaps.")
            return cont

        # Index of each input sample on the filled axis
        index = np.concatenate([[0], np.cumsum(nstep)])
        nnew = index[-1] + 1
        new_ra = ra[0] + dra * np.arange(nnew)
        new_ra[index] = ra  # keep the original values exactly
        gaps = np.flatnonzero(nstep > 1)
        self.log.info(
            f"Filling {gaps.size} gaps in the RA axis with {nnew - ra.size} samples: "
            + ", ".join(f"{nstep[g] - 1} at RA {ra[g]:.2f}" for g in gaps[:10])
        )

        dist_axis = None
        for dset in cont.datasets.values():
            if getattr(dset, "distributed", False):
                dist_axis = dset.attrs["axis"][dset.distributed_axis]
                break
        if dist_axis == "ra":
            raise ValueError("The input must not be distributed over RA.")

        out = cont.__class__(
            ra=new_ra,
            axes_from=cont,
            attrs_from=cont,
            distributed=cont.distributed,
            comm=cont.comm,
        )
        for name in cont.datasets:
            if name not in out.datasets:
                out.add_dataset(name)

        for name, dset in cont.datasets.items():
            odset = out.datasets[name]
            axes = list(dset.attrs["axis"])
            src = dset[:]
            src = src.local_array if hasattr(src, "local_array") else src
            dst = odset[:]
            dst_local = dst.local_array if hasattr(dst, "local_array") else dst
            if "ra" not in axes:
                dst_local[:] = src
                continue
            iax = axes.index("ra")
            dst_local[:] = 0
            sel = [slice(None)] * src.ndim
            sel[iax] = index
            dst_local[tuple(sel)] = src
            if name == "effective_ra":
                shape = [1] * src.ndim
                shape[iax] = nnew
                fill = np.ones(nnew, dtype=bool)
                fill[index] = False
                sel[iax] = fill
                dst_local[tuple(sel)] = np.broadcast_to(
                    new_ra.reshape(shape), dst_local.shape
                )[tuple(sel)]

        return out


class SelectRARange(tasklib.base.ContainerTask):
    """Keep the samples of a container whose right ascension is within a range.

    For example, to keep only the candidates within the nominal sidereal day when
    the whole daily file (which extends a little beyond it) was processed, so that
    neighbouring days do not report the same samples.  Datasets without an RA
    axis are copied unchanged.

    Attributes
    ----------
    ra_min, ra_max : float
        Keep samples with ``ra_min <= ra < ra_max``, in degrees.  Defaults are 0
        and 360.
    """

    ra_min = config.Property(proptype=float, default=0.0)
    ra_max = config.Property(proptype=float, default=360.0)

    def process(self, cont):
        """Select the samples.

        Parameters
        ----------
        cont : container with an ``ra`` axis
            For example a TransientCandidateMap.

        Returns
        -------
        out : same type as ``cont``
            The samples within the range.
        """
        from caput.containers import copy_datasets_filter

        ra = cont.index_map["ra"][:]
        keep = np.flatnonzero((ra >= self.ra_min) & (ra < self.ra_max))
        if keep.size == 0:
            raise ValueError(
                f"No samples with {self.ra_min} <= RA < {self.ra_max} "
                f"(the axis spans {ra.min():.2f} to {ra.max():.2f})."
            )
        if np.any(np.diff(keep) != 1):
            raise ValueError("The RA axis is not monotonic within the range.")
        sel = slice(keep[0], keep[-1] + 1)

        out = cont.__class__(
            axes_from=cont,
            attrs_from=cont,
            ra=ra[sel],
            skip_datasets=True,
            distributed=cont.distributed,
            comm=cont.comm,
        )
        copy_datasets_filter(
            cont, out, selection={"ra": sel}, copy_without_selection=True
        )

        self.log.info(
            f"Kept {keep.size} of {ra.size} samples "
            f"(RA {ra[sel][0]:.2f} to {ra[sel][-1]:.2f} deg)."
        )
        return out


# ----------------------------------------------------------------------------
# Grouping candidates and sifting them
# ----------------------------------------------------------------------------

# Speed of light in m MHz, for the north-south grating lobes
_C_M_MHZ = scipy.constants.c / 1e6


def _local_peaks(img, nrow, ncol, threshold):
    """Local maxima of a map at or above a threshold.

    A pixel is a peak if it is the largest within +-``nrow`` rows and +-``ncol``
    columns.  Of a plateau of equal maxima, only the first pixel is kept.

    Returns
    -------
    rows, cols : np.ndarray[int]
        Indices of the peaks, sorted by decreasing value.
    """
    mx = scipy.ndimage.maximum_filter(
        img, size=(2 * nrow + 1, 2 * ncol + 1), mode="constant", cval=-np.inf
    )
    peak = (img == mx) & (img >= threshold)
    lab, npeak = scipy.ndimage.label(peak, structure=np.ones((3, 3)))
    if npeak == 0:
        return np.zeros(0, dtype=int), np.zeros(0, dtype=int)
    flat = lab.ravel()
    idx = np.flatnonzero(flat)
    _, first = np.unique(flat[idx], return_index=True)
    rows, cols = np.unravel_index(idx[first], img.shape)
    order = np.argsort(img[rows, cols], kind="stable")[::-1]
    return rows[order], cols[order]


def _merge_overlapping(rows, cols, dur, nrow, slack):
    """Drop the peaks whose trial overlaps that of a brighter peak.

    The maps are indexed by the start of the trial, so a pulse lasting W samples
    lights trials starting anywhere from ~W samples before it to its end.  A peak
    is dropped if, within +-``nrow`` rows, a brighter peak that is kept has a trial
    [t, t + W) that overlaps its own, widened by ``slack`` samples each side.

    Parameters
    ----------
    rows, cols : np.ndarray[int]
        Peaks, sorted by decreasing value (as from ``_local_peaks``).
    dur : np.ndarray[int]
        Duration in samples of the best trial at each pixel of the map.
    nrow, slack : int
        Half-width in rows, and the extra samples either side.

    Returns
    -------
    rows, cols : np.ndarray[int]
        The peaks kept, in the same order.
    """
    if rows.size == 0:
        return rows, cols
    nr, nc = dur.shape
    w = np.maximum(dur[rows, cols].astype(int), 1)
    claimed = np.zeros((nr, nc + 1), dtype=bool)
    keep = np.zeros(rows.size, dtype=bool)
    for k in range(rows.size):
        e, t = rows[k], cols[k]
        if claimed[e, t : t + w[k]].any():
            continue
        keep[k] = True
        claimed[
            max(e - nrow, 0) : e + nrow + 1,
            max(t - slack, 0) : min(t + w[k] + slack, nc),
        ] = True
    return rows[keep], cols[keep]


def _run_length(mask, i):
    """Length of the run of True values in a 1D mask that contains index i."""
    if not mask[i]:
        return 0
    lo = i
    while lo > 0 and mask[lo - 1]:
        lo -= 1
    hi = i
    while hi < mask.size - 1 and mask[hi + 1]:
        hi += 1
    return hi - lo + 1


def _wrap_deg(a):
    """Wrap angles in degrees to [-180, 180)."""
    return (np.asarray(a) + 180.0) % 360.0 - 180.0


def _telescope_y_alt(ha, dec, lat):
    """Telescope-y and altitude (deg) of a source at hour angle ``ha`` (deg).

    ``dec`` is in degrees and ``lat`` in radians.  The small rotation of the
    telescope is ignored.
    """
    h = np.radians(ha)
    d = np.radians(dec)
    y = np.cos(lat) * np.sin(d) - np.sin(lat) * np.cos(d) * np.cos(h)
    salt = np.sin(lat) * np.sin(d) + np.cos(lat) * np.cos(d) * np.cos(h)
    return y, np.degrees(np.arcsin(np.clip(salt, -1.0, 1.0)))


def _spectrum_stats(freq, amp, weight, snr_min, ref_freq, alphas):
    """Summary of a spectrum of band amplitudes.

    Returns the number of bands with S/N above ``snr_min``, the fraction of the
    positive S/N^2 in the brightest band, the centre of the brightest band, the
    spectral index and reduced chi-squared of the best-fitting power law (from the
    grid ``alphas``), and the fraction of the spectrum's power that the power law
    explains, 1 - chi^2 / chi^2_0, with chi^2_0 that of a zero spectrum.
    """
    ok = (weight > 0) & np.isfinite(amp)
    if not np.any(ok):
        return -1, np.nan, np.nan, np.nan, np.nan, np.nan
    s = np.where(ok, amp * np.sqrt(np.where(ok, weight, 0.0)), 0.0)
    pos = np.clip(s, 0.0, None) ** 2
    frac = pos.max() / pos.sum() if pos.sum() > 0 else np.nan
    f, a, w = freq[ok], amp[ok], weight[ok]
    m = (f[:, np.newaxis] / ref_freq) ** alphas[np.newaxis, :]
    norm = np.sum(
        w[:, np.newaxis] * m * a[:, np.newaxis], axis=0
    ) * tools.invert_no_zero(np.sum(w[:, np.newaxis] * m**2, axis=0))
    chi2 = np.sum(w[:, np.newaxis] * (a[:, np.newaxis] - norm * m) ** 2, axis=0)
    k = int(np.argmin(chi2))
    chi2_zero = np.sum(w * a**2)
    return (
        int(np.sum(s > snr_min)),
        frac,
        freq[int(np.argmax(s))],
        alphas[k],
        chi2[k] / max(f.size - 2, 1),
        1.0 - chi2[k] / chi2_zero if chi2_zero > 0 else np.nan,
    )


def _decode_names(names):
    return [n.decode() if isinstance(n, bytes) else str(n) for n in names]


class GroupTransientCandidates(tasklib.base.ContainerTask):
    """Find the candidates in the maps of the transient search and describe them.

    A candidate is a peak of the best-x S/N map (all bands) at or above ``seed_snr``
    that is the largest within +-``peak_dy`` in telescope-y and +-``peak_dt``
    samples in start time, and whose best trial [start, start + duration), widened
    by ``peak_dt`` either side, does not overlap that of a brighter candidate within
    +-``peak_dy`` (a long pulse lights trials over a range of start times).  So a
    pulse and the sidelobes of the synthesized beam at the same time are one
    candidate, while pulses at different times stay separate.
    For each candidate the output table has (columns of the ``candidate`` table):

    - Position and best trial: ``el_index``, ``ra_index``, ``x_index``, ``y``
      (telescope-y), ``x`` (telescope-x), ``ra`` (start of the pulse, the RA of the
      meridian in degrees), ``unix`` (time of the start), ``snr``, ``amplitude``
      (Jy at the reference frequency), ``duration`` (s) and ``spectral_index``.
    - Sky position: ``ra_icrs`` and ``dec_icrs`` (degrees) and the hour angle ``ha``
      (degrees) of the best telescope-x.
    - Shape: ``npix``, ``fwhm_y`` and ``fwhm_t`` (s) of the region above half the
      peak, and ``x_spread``, the standard deviation of the best telescope-x index in
      that region.  The "island" is the connected region above ``grow_snr`` around the
      peak, within +-``box_dy`` and +-``box_dra``: its number of pixels, extent in
      telescope-y and in time (s), and the slope (telescope-y per degree of RA) and
      correlation of telescope-y with RA of its pixels, weighted by S/N.  Arcs from
      moving objects have large extents in both and a high correlation.  The
      "extent" is the same but with pixels above ``grow_snr`` joined across gaps of
      up to ``join_dy`` in telescope-y and ``join_dt`` samples, so that a chain of
      blobs (an arc broken up by the fringes) counts as one: ``ext_npix`` (its
      pixels above ``grow_snr``), ``ext_dy`` and ``ext_dt`` (s).  A pulse from a
      point source extends ~0.01 in telescope-y.
    - Along the row: ``row_npeak``, the number of other peaks (local maxima within 1
      row and +-``row_peak_dra``, merged as for the candidates, whose trials do not
      overlap the candidate's) within +-``row_dy`` in telescope-y and
      +-``row_dra`` degrees of RA with S/N at least ``row_snr`` and ``row_ratio``
      times that of the candidate; ``row_max_ratio``, the S/N of the brightest
      other peak there over the candidate's; and ``row_spacing`` (s) and
      ``row_spacing_cv``, the median and the coefficient of variation of the
      spacing between those peaks and the candidate, to tell periodic pulses from
      scintillation.
    - At the same time: ``col_nrows``, the number of rows with |S/N| above
      ``col_snr`` at telescope-x = 0 (group "all" of ``groups_x0``) within one sample
      of the candidate's start; ``col_nrows_bg``, the running median of that number
      over the samples with data within +-``col_bg_dt``, which is high during
      scintillation episodes and low for a burst of broadband RFI; and
      ``col_ncand``, the number of candidates within one sample.
    - Spectrum: ``group_snr`` (a separate dataset) is the S/N of each group of bands
      at the best telescope-x of that group; ``group_nsig`` is the number of groups
      (other than "all") above ``group_snr_min``, ``group_frac_max`` the fraction of
      the positive S/N^2 in the brightest group and ``group_best`` its index.  The
      ``band_*`` columns are left empty here (see :class:`AddCandidateSpectra`).
    - ``weight_ratio``: the weight at the candidate over the median of its row.
    - The Sun and bright sources (``bright_sources``): ``sun_ha`` and ``sun_alt``
      (degrees); the source above the horizon whose track is closest in
      telescope-y (``track_name``, ``track_dy``, ``track_ha``); and the source
      above the horizon, with the smallest hour angle, whose first north-south
      grating lobe falls on the candidate's row at a frequency within
      ``freq_range`` (``lobe_name``, ``lobe_ha``, ``lobe_freq``).  The lobe of a
      source at telescope-y y_s is at y_s +- c / (nu d), with d the feed spacing
      ``ns_spacing``.
    - Catalogue (if ``catalogue`` is set): of the sources within +-``cat_ddec`` in
      dec and +-``cat_dra`` in RA cos(dec) of the candidate's sky position, the
      brightest (``cat_name``, ``cat_flux`` in Jy at 600 MHz, and its offsets
      ``cat_dra`` and ``cat_ddec`` in degrees), and their number ``cat_nmatch``.

    Attributes
    ----------
    pol : int
        Index of the polarisation of the maps to use.  Default is 0.
    seed_snr : float
        Smallest peak S/N for a candidate.  Default is 8.
    peak_dy : float
        Half-width in telescope-y within which a candidate is the largest peak.
        Default is 0.02.
    peak_dt : int
        Half-width in samples within which a candidate is the largest peak.
        Default is 3.
    grow_snr : float
        Threshold of the island around a candidate.  Default is 5.
    box_dy, box_dra : float
        Half-widths in telescope-y and in degrees of RA of the box in which the shape
        is measured.  Defaults are 0.1 and 2.
    join_dy, join_dt : float, int
        Largest gaps in telescope-y and in samples bridged in the extent.  Defaults
        are 0.005 (3 rows) and 1.
    row_snr, row_ratio : float
        Smallest S/N of another peak along the row, absolute and relative to the
        candidate.  Defaults are 8 and 1/3.
    row_dy, row_dra : float
        Half-widths in telescope-y and in degrees of RA of the search along the row.
        Defaults are 0.01 and 5.
    row_peak_dra : float
        Half-width in degrees of RA of the local maxima along the row.  Default is
        0.3.
    col_snr : float
        |S/N| threshold for counting the rows at the same time.  Default is 5.
    col_bg_dt : int
        Half-width in samples of the running median of the number of rows lit.
        Default is 30 (5 minutes).
    group_snr_min : float
        S/N above which a group of bands counts as detected.  Default is 4 (the
        maximum over telescope-x of noise alone is ~2).
    bright_sources : list of str
        The Sun and bright sources, as names in ``ch_ephem.sources.source_dictionary``
        or "sun".  Default is ["sun", "CYG_A", "CAS_A", "TAU_A", "VIR_A"].
    ns_spacing : float
        North-south feed spacing in metres.  Default is 0.3048.
    freq_range : list of float
        Frequency range in MHz for the grating lobes.  Default is [400, 800].
    catalogue : str
        HDF5 file with datasets ``name``, ``ra``, ``dec`` (ICRS, degrees) and
        ``flux_600`` (Jy), for example from scripts/catalog/reduce_specfind.py.  If
        not set, the catalogue columns are left empty.
    cat_ddec, cat_dra : float
        Half-widths of the catalogue match in degrees.  Defaults are 0.5 and 2.
    max_candidates : int
        Keep at most this many candidates (the brightest).  Default is 50000.
    """

    pol = config.Property(proptype=int, default=0)
    seed_snr = config.Property(proptype=float, default=8.0)
    peak_dy = config.Property(proptype=float, default=0.02)
    peak_dt = config.Property(proptype=int, default=3)
    grow_snr = config.Property(proptype=float, default=5.0)
    box_dy = config.Property(proptype=float, default=0.1)
    box_dra = config.Property(proptype=float, default=2.0)
    join_dy = config.Property(proptype=float, default=0.005)
    join_dt = config.Property(proptype=int, default=1)
    row_snr = config.Property(proptype=float, default=8.0)
    row_ratio = config.Property(proptype=float, default=1.0 / 3.0)
    row_dy = config.Property(proptype=float, default=0.01)
    row_dra = config.Property(proptype=float, default=5.0)
    row_peak_dra = config.Property(proptype=float, default=0.3)
    col_snr = config.Property(proptype=float, default=5.0)
    col_bg_dt = config.Property(proptype=int, default=30)
    group_snr_min = config.Property(proptype=float, default=4.0)
    bright_sources = config.list_type(
        type_=str, default=["sun", "CYG_A", "CAS_A", "TAU_A", "VIR_A"]
    )
    ns_spacing = config.Property(proptype=float, default=0.3048)
    freq_range = config.list_type(type_=float, length=2, default=[400.0, 800.0])
    catalogue = config.Property(proptype=str, default=None)
    cat_ddec = config.Property(proptype=float, default=0.5)
    cat_dra = config.Property(proptype=float, default=2.0)
    max_candidates = config.Property(proptype=int, default=50000)

    def setup(self):
        """Load the catalogue, if any."""
        self._cat = None
        if self.catalogue:
            import h5py

            with h5py.File(self.catalogue, "r") as f:
                dec = f["dec"][:]
                order = np.argsort(dec)
                self._cat = {
                    "name": f["name"][:][order],
                    "ra": f["ra"][:][order],
                    "dec": dec[order],
                    "flux": f["flux_600"][:][order],
                }
            self.log.info(
                f"Loaded {self._cat['dec'].size} catalogue sources from {self.catalogue}."
            )

    def process(self, cand, groups_maxx, groups_x0):
        """Find and describe the candidates.

        Parameters
        ----------
        cand : containers.TransientCandidateMap
            Best trial at each elevation and start time (all bands).
        groups_maxx : containers.TransientBandGroupMap
            S/N of each group of bands at the best telescope-x.
        groups_x0 : containers.TransientBandGroupMap
            Signed S/N of each group of bands at telescope-x = 0, with a group "all".

        Returns
        -------
        out : containers.TransientCandidateTable
            One row per candidate.
        """
        import warnings

        from caput.astro import skyfield as csf
        from ch_ephem.observers import chime
        from mpi4py import MPI
        from skyfield.units import Angle

        for cont in (cand, groups_maxx, groups_x0):
            cont.redistribute("el")

        el = cand.index_map["el"][:]
        ra = cand.index_map["ra"][:]
        xgrid = cand.index_map["x"][:]
        for g in (groups_maxx, groups_x0):
            if g.index_map["el"].size != el.size or not np.allclose(
                g.index_map["ra"][:], ra
            ):
                raise ValueError(
                    "The group maps must have the same elevations and start times "
                    "as the candidate map."
                )
        lsd = _stream_lsd(cand)
        pp = self.pol
        dy = np.median(np.diff(el))
        dra = np.median(np.diff(ra))
        tsamp = float(cand.attrs.get("duration_unit_seconds", dra * 240.0 * 0.99727))
        lat = np.radians(chime.latitude)

        # The maps of the best trial and of telescope-x = 0, on every rank
        snr = cand.snr[:].allgather()[pp].astype(np.float32)
        wgt = cand.weight[:].allgather()[pp]
        xidx = cand.datasets["x_index"][:].allgather()[pp]
        valid = (wgt > 0) & np.isfinite(snr)
        img = np.where(valid, snr, 0.0).astype(np.float32)

        names0 = _decode_names(groups_x0.index_map["group"][:])
        if "all" not in names0:
            raise ValueError("groups_x0 must have a group named 'all'.")
        x0 = mpiarray.MPIArray.wrap(
            np.ascontiguousarray(groups_x0.snr[:].local_array[pp, names0.index("all")]),
            axis=0,
            comm=groups_x0.comm,
        ).allgather()

        # Candidates, and the finer peaks used along the rows
        dur = cand.datasets["duration"][:].allgather()[pp]
        dur = np.where(valid, np.nan_to_num(dur), 1).astype(int)
        de = max(1, round(self.peak_dy / dy))
        pe, pt = _local_peaks(img, de, self.peak_dt, self.seed_snr)
        pe, pt = _merge_overlapping(pe, pt, dur, de, self.peak_dt)
        if pe.size > self.max_candidates:
            self.log.warning(
                f"Keeping the brightest {self.max_candidates} of {pe.size} candidates."
            )
            pe, pt = pe[: self.max_candidates], pt[: self.max_candidates]
        ncand = pe.size
        nfine = max(1, round(self.row_peak_dra / dra))
        fe, ft = _local_peaks(img, 1, nfine, self.row_snr)
        fe, ft = _merge_overlapping(fe, ft, dur, 1, nfine)
        order = np.argsort(fe, kind="stable")
        fe, ft = fe[order], ft[order]
        fs = img[fe, ft]
        fra = ra[ft]
        fend = ft + dur[fe, ft]

        # Rows lit at the same time, and candidates at the same time
        hot = (np.abs(np.nan_to_num(x0)) > self.col_snr) & valid
        hot3 = hot.copy()
        hot3[:, 1:] |= hot[:, :-1]
        hot3[:, :-1] |= hot[:, 1:]
        col_count = hot3.sum(axis=0)
        col_bg = median.moving_weighted_median(
            col_count.astype(np.float64),
            valid.any(axis=0).astype(np.float64),
            2 * self.col_bg_dt + 1,
        )
        cand_count = np.bincount(pt, minlength=ra.size)
        cand_count3 = cand_count.copy()
        cand_count3[1:] += cand_count[:-1]
        cand_count3[:-1] += cand_count[1:]

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            wrow = np.nanmedian(np.where(wgt > 0, wgt, np.nan), axis=1)

        names_g = _decode_names(groups_maxx.index_map["group"][:])
        out = containers.TransientCandidateTable(
            candidate=ncand,
            group=np.array(names_g),
            freq=groups_maxx.index_map["freq"][:],
            attrs_from=cand,
            comm=cand.comm,
        )
        rec = np.zeros(ncand, dtype=out.candidate[:].dtype)
        for name in rec.dtype.names:
            if rec.dtype[name].kind == "f":
                rec[name] = np.nan
        rec["band_nsig"] = -1
        out.attrs["lsd"] = lsd
        for key in (
            "seed_snr", "peak_dy", "peak_dt", "grow_snr", "join_dy", "join_dt",
            "row_snr", "row_ratio", "row_dy", "row_dra", "col_snr", "group_snr_min",
            "cat_ddec", "cat_dra",
        ):  # fmt: skip
            out.attrs[key] = getattr(self, key)
        if self.catalogue:
            out.attrs["catalogue"] = self.catalogue

        if ncand == 0:
            self.log.info("No candidates.")
            return out

        # Position and best trial
        ix = xidx[pe, pt].astype(int)
        rec["el_index"] = pe
        rec["ra_index"] = pt
        rec["x_index"] = ix
        rec["y"] = el[pe]
        rec["x"] = xgrid[ix]
        rec["ra"] = ra[pt]
        rec["unix"] = chime.lsd_to_unix(lsd + ra[pt] / 360.0)
        rec["snr"] = img[pe, pt]
        for name in ("amplitude", "duration", "spectral_index"):
            vals = np.zeros(ncand, dtype=np.float64)
            dset = cand.datasets[name][:]
            off, nloc = dset.local_offset[1], dset.local_shape[1]
            mine = (pe >= off) & (pe < off + nloc)
            vals[mine] = dset.local_array[pp, pe[mine] - off, pt[mine]]
            cand.comm.Allreduce(MPI.IN_PLACE, vals, op=MPI.SUM)
            rec[name] = vals
        rec["duration"] *= tsamp
        rec["weight_ratio"] = wgt[pe, pt] * tools.invert_no_zero(
            np.nan_to_num(wrow[pe])
        )

        # Sky position
        ha_c = cand.datasets["ha"][:][pe, ix]
        dec_c = cand.datasets["dec"][:][pe, ix]
        ra_cirs = (ra[pt] - ha_c) % 360.0
        epoch = csf.unix_to_skyfield_time(chime.lsd_to_unix(lsd + 0.5))
        star = chime.star_cirs(Angle(degrees=ra_cirs), Angle(degrees=dec_c), epoch)
        rec["ra_icrs"] = np.atleast_1d(star.ra._degrees) % 360.0
        rec["dec_icrs"] = np.atleast_1d(star.dec.degrees)
        rec["ha"] = ha_c

        # Shape, and the other peaks along the row
        nbox_e = max(1, round(self.box_dy / dy))
        nbox_t = max(1, round(self.box_dra / dra))
        nrow = max(0, round(self.row_dy / dy))
        eight = np.ones((3, 3))
        join = np.ones(
            (2 * max(0, round(self.join_dy / dy)) + 1, 2 * max(0, self.join_dt) + 1)
        )
        for k in range(ncand):
            e, t = pe[k], pt[k]
            peak = img[e, t]
            e0, e1 = max(e - nbox_e, 0), min(e + nbox_e + 1, el.size)
            t0, t1 = max(t - nbox_t, 0), min(t + nbox_t + 1, ra.size)
            sub = img[e0:e1, t0:t1]
            ce, ct = e - e0, t - t0

            lab, _ = scipy.ndimage.label(sub >= max(0.5 * peak, self.grow_snr), eight)
            comp = lab == lab[ce, ct]
            rec["npix"][k] = comp.sum()
            rec["fwhm_y"][k] = _run_length(comp[:, ct], ce) * dy
            rec["fwhm_t"][k] = _run_length(comp[ce], ct) * tsamp
            rec["x_spread"][k] = np.std(xidx[e0:e1, t0:t1][comp])

            lab, _ = scipy.ndimage.label(sub >= self.grow_snr, eight)
            isl = lab == lab[ce, ct]
            ie, it = np.nonzero(isl)
            rec["island_npix"][k] = ie.size
            rec["island_dy"][k] = (np.ptp(ie) + 1) * dy
            rec["island_dt"][k] = (np.ptp(it) + 1) * tsamp
            if ie.size >= 3 and np.ptp(ie) > 0 and np.ptp(it) > 0:
                w = sub[ie, it]
                yy, rr = el[e0 + ie], ra[t0 + it]
                ym, rm = np.average(yy, weights=w), np.average(rr, weights=w)
                cyr = np.average((yy - ym) * (rr - rm), weights=w)
                vr = np.average((rr - rm) ** 2, weights=w)
                vy = np.average((yy - ym) ** 2, weights=w)
                rec["island_slope"][k] = cyr / vr
                rec["island_corr"][k] = cyr / np.sqrt(vr * vy)

            lit = sub >= self.grow_snr
            lab, _ = scipy.ndimage.label(
                scipy.ndimage.binary_dilation(lit, structure=join), eight
            )
            ie, it = np.nonzero(lit & (lab == lab[ce, ct]))
            rec["ext_npix"][k] = ie.size
            rec["ext_dy"][k] = (np.ptp(ie) + 1) * dy
            rec["ext_dt"][k] = (np.ptp(it) + 1) * tsamp

            lo, hi = np.searchsorted(fe, [e - nrow, e + nrow + 1])
            near = np.abs(fra[lo:hi] - ra[t])
            # other pulses: trials that do not overlap the candidate's
            other = (near <= self.row_dra) & (
                (fend[lo:hi] + self.peak_dt <= t)
                | (ft[lo:hi] >= t + dur[e, t] + self.peak_dt)
            )
            if np.any(other):
                rec["row_max_ratio"][k] = fs[lo:hi][other].max() / peak
            else:
                rec["row_max_ratio"][k] = 0.0
            strong = other & (fs[lo:hi] >= max(self.row_snr, self.row_ratio * peak))
            rec["row_npeak"][k] = strong.sum()
            if strong.any():
                times = np.sort(np.concatenate([fra[lo:hi][strong], [ra[t]]]))
                gaps = np.diff(times) * 240.0 * 0.99727
                rec["row_spacing"][k] = np.median(gaps)
                if gaps.size >= 2:
                    rec["row_spacing_cv"][k] = np.std(gaps) / np.mean(gaps)

        rec["col_nrows"] = col_count[pt]
        rec["col_nrows_bg"] = col_bg[pt]
        rec["col_ncand"] = cand_count3[pt]

        # S/N of each group of bands at the candidate
        gsnr = np.zeros((ncand, len(names_g)), dtype=np.float64)
        loc = groups_maxx.snr[:]
        off, nloc = loc.local_offset[2], loc.local_shape[2]
        mine = (pe >= off) & (pe < off + nloc)
        if np.any(mine):
            gsnr[mine] = loc.local_array[pp][:, pe[mine] - off, pt[mine]].T
        groups_maxx.comm.Allreduce(MPI.IN_PLACE, gsnr, op=MPI.SUM)
        out.datasets["group_snr"][:] = gsnr
        sub = np.array([n != "all" for n in names_g])
        if np.any(sub):
            gs = gsnr[:, sub]
            pos = np.clip(gs, 0.0, None) ** 2
            rec["group_nsig"] = np.sum(gs > self.group_snr_min, axis=1)
            rec["group_frac_max"] = pos.max(axis=1) * tools.invert_no_zero(
                pos.sum(axis=1)
            )
            rec["group_best"] = np.flatnonzero(sub)[np.argmax(gs, axis=1)]

        # The Sun and bright sources
        lst = ra[pt]
        _, sra, sdec = _source_positions(["sun"], {}, lsd)
        sun_ha = _wrap_deg(lst - sra[0])
        _, sun_alt = _telescope_y_alt(sun_ha, sdec[0], lat)
        rec["sun_ha"] = sun_ha
        rec["sun_alt"] = sun_alt
        if self.bright_sources:
            bnames, bra, bdec = _source_positions(self.bright_sources, {}, lsd)
            fmin, fmax = self.freq_range
            best_track = np.full(ncand, np.inf)
            best_lobe = np.full(ncand, np.inf)
            for name, r_b, d_b in zip(bnames, bra, bdec):
                ha_b = _wrap_deg(lst - r_b)
                y_b, alt_b = _telescope_y_alt(ha_b, d_b, lat)
                up = alt_b > 0
                dyk = el[pe] - y_b
                better = up & (np.abs(dyk) < best_track)
                best_track[better] = np.abs(dyk[better])
                rec["track_name"][better] = name
                rec["track_dy"][better] = dyk[better]
                rec["track_ha"][better] = ha_b[better]
                flobe = _C_M_MHZ / (self.ns_spacing * np.maximum(np.abs(dyk), 1e-6))
                hit = up & (flobe >= fmin) & (flobe <= fmax)
                better = hit & (np.abs(ha_b) < best_lobe)
                best_lobe[better] = np.abs(ha_b[better])
                rec["lobe_name"][better] = name
                rec["lobe_ha"][better] = ha_b[better]
                rec["lobe_freq"][better] = flobe[better]

        # Catalogue
        if self._cat is not None:
            cdec = self._cat["dec"]
            lo = np.searchsorted(cdec, rec["dec_icrs"] - self.cat_ddec)
            hi = np.searchsorted(cdec, rec["dec_icrs"] + self.cat_ddec)
            for k in range(ncand):
                if hi[k] == lo[k]:
                    continue
                d0 = rec["dec_icrs"][k]
                dr = _wrap_deg(self._cat["ra"][lo[k] : hi[k]] - rec["ra_icrs"][k])
                dr = dr * np.cos(np.radians(d0))
                within = np.flatnonzero(np.abs(dr) <= self.cat_dra)
                if within.size == 0:
                    continue
                j = within[np.argmax(self._cat["flux"][lo[k] : hi[k]][within])]
                rec["cat_name"][k] = self._cat["name"][lo[k] + j]
                rec["cat_flux"][k] = self._cat["flux"][lo[k] + j]
                rec["cat_dra"][k] = dr[j]
                rec["cat_ddec"][k] = self._cat["dec"][lo[k] + j] - d0
                rec["cat_nmatch"][k] = within.size
            nomatch = rec["cat_nmatch"] == 0
            rec["cat_flux"][nomatch] = 0.0

        out.candidate[:] = rec
        self.log.info(
            f"Found {ncand} candidates with S/N >= {self.seed_snr:g} "
            f"(brightest {rec['snr'][0]:.1f})."
        )
        return out


def _candidate_positions(table, bands, match_duration=True, duration=1):
    """Map indices of the candidates in the band output, and their durations.

    Returns el index, start-time index, x index and duration (samples) of each.
    """
    rec = table.candidate[:]
    ncand = rec.size
    ra_b = bands.index_map["ra"][:]
    dra = np.median(np.diff(ra_b))
    tb = np.clip(np.searchsorted(ra_b, rec["ra"] - 0.5 * dra), 0, ra_b.size - 1)
    if ncand and np.any(np.abs(ra_b[tb] - rec["ra"]) > 0.01 * dra):
        raise ValueError("The start times of the candidates are not in the bands.")
    if bands.index_map["el"].size <= (rec["el_index"].max() if ncand else 0):
        raise ValueError("The elevations of the candidates are not in the bands.")
    if match_duration:
        tsamp = float(
            table.attrs.get(
                "duration_unit_seconds",
                bands.attrs.get("duration_unit_seconds", dra * 240.0 * 0.99727),
            )
        )
        wc = np.maximum(np.rint(np.nan_to_num(rec["duration"]) / tsamp), 1).astype(int)
    else:
        wc = np.full(ncand, duration, dtype=int)
    return rec["el_index"].astype(int), tb, rec["x_index"].astype(int), wc


def _band_cutouts(bands, pol, e, tb, xi, wc, row_off, time_off):
    """Amplitude and weight of each band around candidates, from this rank's rows.

    For candidate k, row ``e[k] + row_off[i]`` and start ``tb[k] + time_off[j]``, at
    telescope-x index ``xi[k]`` and duration ``wc[k]``: from the band output for that
    duration if it has it, or else the inverse-variance weighted mean of the
    single-sample output along the track of the sky position of that row's template
    (see :class:`AddCandidateSpectra`).  Entries outside the maps, or on rows held by
    other ranks, are zero (sum over ranks to complete them).

    Returns
    -------
    amp, wgt : np.ndarray[ncand, freq, len(row_off), len(time_off)]
    """
    durations = [int(d) for d in bands.index_map["duration"][:]]
    la, lw = bands.amplitude[:], bands.weight[:]
    off, nloc = la.local_offset[3], la.local_shape[3]
    ra_b = bands.index_map["ra"][:]
    nra = ra_b.size
    dra = np.radians(np.median(np.diff(ra_b)))
    periodic = _is_full_day(ra_b)
    xg = bands.index_map["x"][:]
    dx = xg[1] - xg[0] if xg.size > 1 else 1.0
    ha = np.radians(bands.datasets["ha"][:])
    dec = np.radians(bands.datasets["dec"][:])

    row_off = np.asarray(row_off, dtype=int)
    toff = np.asarray(time_off, dtype=int)
    ncand, nf = len(e), la.local_shape[2]
    amp = np.zeros((ncand, nf, row_off.size, toff.size), dtype=np.float64)
    wgt = np.zeros_like(amp)
    for k in range(ncand):
        W = int(wc[k])
        direct = W in durations
        if not direct and 1 not in durations:
            raise ValueError(
                "The bands have neither the candidates' durations nor duration 1."
            )
        idur = durations.index(W) if direct else durations.index(1)
        ts = tb[k] + toff
        if periodic:
            okt = np.ones(ts.size, dtype=bool)
            ts = ts % nra
        else:
            okt = (ts >= 0) & (ts < nra)
            ts = np.clip(ts, 0, nra - 1)
        for ir, dr in enumerate(row_off):
            er = e[k] + dr
            if er < off or er >= off + nloc:
                continue
            A = la.local_array[pol, idur, :, er - off]  # freq, ra, x
            Wt = lw.local_array[pol, idur, :, er - off]
            if direct:
                amp[k, :, ir] = np.where(okt, A[:, ts, xi[k]], 0.0)
                wgt[k, :, ir] = np.where(okt, Wt[:, ts, xi[k]], 0.0)
                continue
            j = np.arange(W)
            xj = -np.cos(dec[er, xi[k]]) * np.sin(ha[er, xi[k]] + j * dra)
            ij = np.rint((xj - xg[0]) / dx).astype(int)
            okj = np.cumprod((ij >= 0) & (ij < xg.size)).astype(bool)
            ij = np.clip(ij, 0, xg.size - 1)
            tt = ts[:, np.newaxis] + j[np.newaxis]  # time, j
            if periodic:
                tt = tt % nra
                ok = okj[np.newaxis] & okt[:, np.newaxis]
            else:
                ok = np.cumprod(tt < nra, axis=1).astype(bool) & okj & okt[:, np.newaxis]
                tt = np.clip(tt, 0, nra - 1)
            a = A[:, tt, ij[np.newaxis]]  # freq, time, j
            w = Wt[:, tt, ij[np.newaxis]] * ok[np.newaxis]
            ws = w.sum(axis=-1)
            amp[k, :, ir] = (w * a).sum(axis=-1) * tools.invert_no_zero(ws)
            wgt[k, :, ir] = ws
    return amp, wgt


# Number of feeds on a cylinder (for the redundancy of the north-south baselines)
_NS_NFEED = 256


class _NSBeam:
    """The north-south synthesized beam of BeamformNS, and its noise correlation.

    BeamformNS (``weight: triangular``, ``scaled: False``) sums the north-south
    baselines n d up to the longest present, ``n_max d``, with the triangular
    window ``g_n = 1 - |n| / n_max``.  A point source at telescope-y ``y0`` gives
    ``B(y - y0)`` at row y, ``B(dy) = sum_n g_n cos(2 pi n d nu dy / c) / sum_n g_n``,
    a function of ``u = nu dy / c`` only.  The noise of baseline n has variance
    ``~ 1 / (N_feed - |n|)`` (its redundancy, for a pair of cylinders), so the noise
    of rows y and y' has correlation ``G(u) = sum_n g_n^2 / R_n cos(2 pi n d u) /
    sum_n g_n^2 / R_n``.  Both are tabulated in u.
    """

    def __init__(self, ns_spacing=0.3048, ns_max=73.0, umax=0.5, du=2e-6):
        nmax = int(np.floor(ns_max / ns_spacing + 1e-6))
        n = np.arange(-nmax, nmax + 1)
        g = 1.0 - np.abs(n) / nmax
        r = _NS_NFEED - np.abs(n)
        self.u = np.arange(0.0, umax + du, du)
        self.F = np.zeros(self.u.size)
        self.G = np.zeros(self.u.size)
        for i0 in range(0, self.u.size, 4096):
            ph = 2 * np.pi * ns_spacing * np.outer(self.u[i0 : i0 + 4096], n)
            c = np.cos(ph)
            self.F[i0 : i0 + 4096] = c @ g / g.sum()
            self.G[i0 : i0 + 4096] = c @ (g**2 / r) / np.sum(g**2 / r)

    def beam(self, u):
        return np.interp(np.abs(u), self.u, self.F, right=0.0)

    def corr(self, u):
        return np.interp(np.abs(u), self.u, self.G, right=0.0)


def _ns_profile_fit(nsb, yr, amp, wgt, freq, shifts, scales, min_rows, eig_tol=1e-9):
    """Fit the north-south beam to a candidate's profile in each band.

    The model in band b is ``a_b B_b((y - y0) / s)`` with a free amplitude ``a_b``
    per band and a common shift ``y0`` and width scale ``s`` (1 for a point source).
    The noise of the rows has the covariance of :class:`_NSBeam`, scaled by the
    weights; the rows oversample the beam, so the covariance is near-singular and
    the fit uses the eigenvectors above ``eig_tol`` of the largest eigenvalue.

    Parameters
    ----------
    yr : np.ndarray[nrow]
        Telescope-y of the rows relative to the candidate's.
    amp, wgt : np.ndarray[nband, nrow]
    freq : np.ndarray[nband]
        Band centres (MHz).
    shifts, scales : np.ndarray
        Grids of y0 and s.

    Returns
    -------
    chi2 : np.ndarray[nshift, nscale]
        Summed over the bands, with the amplitudes at their best values.
    snr2 : np.ndarray[nshift, nscale]
        The S/N^2 of the fitted model, summed over the bands.
    dof : int
        Number of independent rows minus the amplitudes and the two parameters.
    nband : int
        Number of bands used.
    """
    chi2 = np.zeros((shifts.size, scales.size))
    snr2 = np.zeros_like(chi2)
    dof, nb = 0, 0
    for b in range(amp.shape[0]):
        v = (wgt[b] > 0) & np.isfinite(amp[b])
        if v.sum() < min_rows:
            continue
        y = yr[v]
        f = freq[b] * 1e6 / scipy.constants.c
        sig = 1.0 / np.sqrt(wgt[b, v])
        cov = sig[:, np.newaxis] * sig[np.newaxis] * nsb.corr(f * (y[:, None] - y[None]))
        lam, vec = np.linalg.eigh(cov)
        keep = lam > eig_tol * lam.max()
        wm = vec[:, keep] / np.sqrt(lam[keep])
        dw = wm.T @ amp[b, v]
        arg = f * (y[None, None] - shifts[:, None, None]) / scales[None, :, None]
        mw = nsb.beam(arg) @ wm  # shift, scale, mode
        num = mw @ dw
        den = np.sum(mw**2, axis=-1)
        fit = num**2 * tools.invert_no_zero(den)
        chi2 += dw @ dw - fit
        snr2 += fit
        dof += int(keep.sum()) - 1
        nb += 1
    return chi2, snr2, (dof - 2 if nb else 0), nb


class AddCandidateSpectra(tasklib.base.ContainerTask):
    """Add the amplitude in each band to a table of candidates.

    For each candidate, the amplitude and weight of each band of the per-band
    matched filter at its telescope-y, start time and telescope-x are stored in the
    ``spectrum`` and ``spectrum_weight`` datasets, and summarised in the ``band_*``
    columns: ``band_nsig``, the number of bands with S/N above ``snr_min``;
    ``band_frac_max``, the fraction of the positive S/N^2 in the brightest band;
    ``band_best_freq``, its centre (MHz); ``band_alpha`` and ``band_chi2``, the
    spectral index and reduced chi-squared of the best-fitting power law; and
    ``band_fit_frac``, the fraction of the spectrum's power that it explains
    (1 - chi^2 / chi^2 of a zero spectrum).  RFI puts its power in a few bands and
    is poorly fitted; an astrophysical pulse is smooth.

    With ``match_duration`` (the default), each candidate's spectrum is for its own
    best duration W: from the band output for W if it has it, or else the
    inverse-variance weighted mean of the single-sample band amplitudes along the
    track of the source (the telescope-x of the candidate's sky position at each of
    the W samples).  The latter leaves out the coupling of the samples by the
    filter, so its weights are approximate, but the shape of the spectrum, which the
    tests use, is barely affected.  Without it, every spectrum is for ``duration``.

    Attributes
    ----------
    pol : int
        Index of the polarisation.  Default is 0.
    duration : int
        Duration of the trial (in samples) without ``match_duration``.  Default is 1.
    match_duration : bool
        Use each candidate's best duration.  Default is True.
    snr_min : float
        S/N above which a band counts as detected.  Default is 3.
    reference_freq : float
        Reference frequency (MHz) of the power-law fit.  Default is 600.
    alphas : list of float
        Spectral indices of the fit.  Default is -5 to 5 in steps of 0.1.
    """

    pol = config.Property(proptype=int, default=0)
    duration = config.Property(proptype=int, default=1)
    match_duration = config.Property(proptype=bool, default=True)
    snr_min = config.Property(proptype=float, default=3.0)
    reference_freq = config.Property(proptype=float, default=600.0)
    alphas = config.list_type(
        type_=float, default=list(np.round(np.arange(-5, 5.01, 0.1), 1))
    )

    def process(self, table, bands):
        """Add the spectra.

        Parameters
        ----------
        table : containers.TransientCandidateTable
            Candidates, from :class:`GroupTransientCandidates`.
        bands : containers.TransientMatchedFilter
            Matched filter output for each band, on the same elevations and with the
            start times of the candidates (it may extend beyond them).

        Returns
        -------
        out : containers.TransientCandidateTable
            A copy of the input, with the spectra added.
        """
        from mpi4py import MPI

        bands.redistribute("el")
        centre = bands.index_map["freq"]["centre"][:]
        tcentre = table.index_map["freq"]["centre"][:]
        if centre.size != tcentre.size or not np.allclose(centre, tcentre):
            raise ValueError("The bands differ from those of the candidate table.")
        durations = [int(d) for d in bands.index_map["duration"][:]]
        if not self.match_duration and self.duration not in durations:
            raise ValueError(f"Duration {self.duration} is not in {durations}.")

        rec = table.candidate[:].copy()
        ncand = rec.size
        e, tb, xi, wc = _candidate_positions(
            table, bands, self.match_duration, self.duration
        )
        amp, wgt = _band_cutouts(bands, self.pol, e, tb, xi, wc, [0], [0])
        amp, wgt = amp[:, :, 0, 0], wgt[:, :, 0, 0]
        bands.comm.Allreduce(MPI.IN_PLACE, amp, op=MPI.SUM)
        bands.comm.Allreduce(MPI.IN_PLACE, wgt, op=MPI.SUM)

        out = table.copy()
        for name in ("spectrum", "spectrum_weight"):
            if name not in out.datasets:
                out.add_dataset(name)
        out.datasets["spectrum"][:] = amp
        out.datasets["spectrum_weight"][:] = wgt

        alphas = np.array(self.alphas, dtype=np.float64)
        for k in range(ncand):
            stats = _spectrum_stats(
                centre, amp[k], wgt[k], self.snr_min, self.reference_freq, alphas
            )
            for name, val in zip(
                (
                    "band_nsig",
                    "band_frac_max",
                    "band_best_freq",
                    "band_alpha",
                    "band_chi2",
                    "band_fit_frac",
                ),
                stats,
            ):
                rec[name][k] = val
        out.candidate[:] = rec
        out.attrs["band_snr_min"] = self.snr_min
        out.attrs["band_match_duration"] = self.match_duration

        self.log.info(f"Added the spectra of {ncand} candidates.")
        return out


class AddNSProfile(tasklib.base.ContainerTask):
    """Fit the north-south synthesized beam to each candidate's profile in telescope-y.

    The north-south shape of a point source is known exactly: BeamformNS forms the
    rows from the north-south baselines up to ~73 m with a triangular window, so a
    source at telescope-y ``y0`` appears at row y as ``B_nu(y - y0)``, a beam whose
    width scales as 1 / nu (FWHM 0.0087 at 420 MHz, 0.0061 at 600 MHz; see
    :class:`_NSBeam`).  RFI in the near field (aircraft) or reaching the telescope
    over a range of directions is wider.

    For each candidate, the per-band matched filter is taken over +-``nrow`` rows at
    the candidate's start time, telescope-x and duration (as for the spectra, see
    :class:`AddCandidateSpectra`).  In each band with at least ``min_rows`` rows of
    data, the model is ``a_b B_b((y - y_c - dy) / s)``: the amplitude ``a_b`` of each
    band is free (scintillation and calibration errors change it), while the shift
    ``dy`` and the width scale ``s`` are common to all bands.  The rows oversample
    the beam and their noise is correlated (0.84 between neighbours at 600 MHz); the
    fit uses the covariance from the beamforming, scaled by the weights.

    Columns added: ``ns_nband``, the bands used; ``ns_dof``; ``ns_snr``, the S/N of
    the fitted point-source model; ``ns_dy``, its shift; ``ns_chi2``, its reduced
    chi-squared (s = 1); ``ns_scale``, the best width scale, with ``ns_scale_lo`` and
    ``ns_scale_hi`` where the chi-squared (minimised over the shift) is 1 above its
    minimum (the edges of the grid if it does not get there); and ``ns_dchi2``, the
    chi-squared of the point source minus that at the best scale.  A point source
    has ``ns_scale`` ~ 1 and small ``ns_dchi2``.  The primary beam is taken as
    constant over the rows of the fit.

    Attributes
    ----------
    pol : int
        Index of the polarisation.  Default is 0.
    match_duration : bool
        Use each candidate's best duration (else single samples).  Default is True.
    nrow : int
        Half-width of the profile in rows.  Default is 12 (+-0.02 in telescope-y).
    min_rows : int
        Smallest number of rows with data for a band to be used.  Default is 5.
    max_shift : float
        Largest shift, in rows.  Default is 1.5.
    scale_range : list of float
        Range of the width scale.  Default is [0.5, 8].
    ns_spacing : float
        North-south feed spacing in metres.  Default is 0.3048.
    ns_max : float
        Longest north-south baseline (m), as for MaskBaselines (mask_long_ns).
        Default is 73.
    """

    pol = config.Property(proptype=int, default=0)
    match_duration = config.Property(proptype=bool, default=True)
    nrow = config.Property(proptype=int, default=12)
    min_rows = config.Property(proptype=int, default=5)
    max_shift = config.Property(proptype=float, default=1.5)
    scale_range = config.list_type(type_=float, length=2, default=[0.5, 8.0])
    ns_spacing = config.Property(proptype=float, default=0.3048)
    ns_max = config.Property(proptype=float, default=73.0)

    def process(self, table, bands):
        """Fit the profiles.

        Parameters
        ----------
        table : containers.TransientCandidateTable
            Candidates.
        bands : containers.TransientMatchedFilter
            Matched filter output for each band (as for AddCandidateSpectra).

        Returns
        -------
        out : containers.TransientCandidateTable
            A copy of the input with the ``ns_*`` columns filled.
        """
        from caput.util import mpitools
        from mpi4py import MPI

        bands.redistribute("el")
        comm = bands.comm
        centre = bands.index_map["freq"]["centre"][:]
        el = bands.index_map["el"][:]
        e, tb, xi, wc = _candidate_positions(table, bands, self.match_duration, 1)
        rec = table.candidate[:].copy()
        ncand = rec.size

        offs = np.arange(-self.nrow, self.nrow + 1)
        amp, wgt = _band_cutouts(bands, self.pol, e, tb, xi, wc, offs, [0])
        amp, wgt = amp[..., 0], wgt[..., 0]  # cand, band, row
        comm.Allreduce(MPI.IN_PLACE, amp, op=MPI.SUM)
        comm.Allreduce(MPI.IN_PLACE, wgt, op=MPI.SUM)

        dy = np.median(np.diff(el))
        shifts = dy * np.linspace(
            -self.max_shift, self.max_shift, 2 * int(round(10 * self.max_shift)) + 1
        )
        lo, hi = np.log2(self.scale_range)
        scales = 2.0 ** (np.arange(np.floor(8 * lo), np.ceil(8 * hi) + 1) / 8.0)
        i1 = int(np.argmin(np.abs(scales - 1.0)))
        umax = 1.1 * centre.max() * 1e6 / scipy.constants.c * (
            (self.nrow + self.max_shift + 1) * dy / scales.min()
        )
        nsb = _NSBeam(self.ns_spacing, self.ns_max, umax=umax)

        names = ("ns_nband", "ns_dof", "ns_snr", "ns_dy", "ns_chi2", "ns_scale",
                 "ns_scale_lo", "ns_scale_hi", "ns_dchi2")  # fmt: skip
        res = np.full((ncand, len(names)), np.nan)
        _, s0, s1 = mpitools.split_local(ncand, comm=comm)
        for k in range(s0, s1):
            rows = e[k] + offs
            inside = (rows >= 0) & (rows < el.size)
            yr = np.where(inside, el[np.clip(rows, 0, el.size - 1)] - el[e[k]], 0.0)
            w = np.where(inside[np.newaxis], wgt[k], 0.0)
            chi2, snr2, dof, nb = _ns_profile_fit(
                nsb, yr, amp[k], w, centre, shifts, scales, self.min_rows
            )
            res[k, 0], res[k, 1] = nb, dof
            if nb == 0 or dof <= 0:
                continue
            c1 = chi2[:, i1]
            j = int(np.argmin(c1))
            prof = chi2.min(axis=0)
            ok = np.flatnonzero(prof <= prof.min() + 1.0)
            res[k, 2] = np.sqrt(max(snr2[j, i1], 0.0))
            res[k, 3] = shifts[j]
            res[k, 4] = c1[j] / dof
            res[k, 5] = scales[int(np.argmin(prof))]
            res[k, 6], res[k, 7] = scales[ok.min()], scales[ok.max()]
            res[k, 8] = c1[j] - prof.min()
        # Each candidate was fitted on one rank: take the finite entries
        have = np.isfinite(res)
        tot = np.where(have, res, 0.0)
        comm.Allreduce(MPI.IN_PLACE, tot, op=MPI.SUM)
        cnt = have.astype(np.int64)
        comm.Allreduce(MPI.IN_PLACE, cnt, op=MPI.SUM)
        res = np.where(cnt > 0, tot, np.nan)
        for i, name in enumerate(names):
            vals = res[:, i]
            if rec.dtype[name].kind in "iu":
                rec[name] = np.nan_to_num(vals).astype(rec.dtype[name])
            else:
                rec[name] = vals

        out = table.copy()
        out.candidate[:] = rec
        out.attrs["ns_nrow"] = self.nrow
        out.attrs["ns_max"] = self.ns_max
        nfit = int(np.sum(rec["ns_dof"] > 0))
        self.log.info(f"Fitted the north-south profiles of {nfit} of {ncand} candidates.")
        return out


class ExtractCandidateCutouts(tasklib.base.ContainerTask):
    """Windows of the per-band matched filter around candidates.

    For the candidates with S/N of at least ``snr_min`` (and only those kept by the
    sifter if ``keep_only`` and the table has been sifted), the brightest
    ``max_cutouts`` of them: the amplitude and weight of every band over
    +-``nrow`` rows and +-``nra`` start times, at the candidate's telescope-x, for
    single samples.  The result is distributed over candidates.

    Attributes
    ----------
    pol : int
        Index of the polarisation.  Default is 0.
    snr_min : float
        Smallest S/N.  Default is 10.
    keep_only : bool
        Only the candidates kept by the sifter.  Default is False.
    max_cutouts : int
        Largest number of cutouts (the brightest).  Default is 1000.
    nrow, nra : int
        Half-widths of the window in rows and samples.  Defaults are 16 and 30.
    """

    pol = config.Property(proptype=int, default=0)
    snr_min = config.Property(proptype=float, default=10.0)
    keep_only = config.Property(proptype=bool, default=False)
    max_cutouts = config.Property(proptype=int, default=1000)
    nrow = config.Property(proptype=int, default=16)
    nra = config.Property(proptype=int, default=30)

    def process(self, table, bands):
        """Extract the cutouts.

        Parameters
        ----------
        table : containers.TransientCandidateTable
            Candidates (sifted or not).
        bands : containers.TransientMatchedFilter
            Matched filter output for each band, with duration 1.

        Returns
        -------
        out : containers.TransientCandidateCutouts
        """
        from mpi4py import MPI

        bands.redistribute("el")
        comm = bands.comm
        if 1 not in [int(d) for d in bands.index_map["duration"][:]]:
            raise ValueError("The bands need duration 1 for the cutouts.")
        rec = table.candidate[:]
        sel = rec["snr"] >= self.snr_min
        if self.keep_only and "sift" in table.datasets:
            sel &= table.datasets["sift"][:]["keep"]
        idx = np.flatnonzero(sel)
        idx = idx[np.argsort(rec["snr"][idx], kind="stable")[::-1]][: self.max_cutouts]
        ncut = idx.size

        e, tb, xi, _ = _candidate_positions(table, bands, False, 1)
        e, tb, xi = e[idx], tb[idx], xi[idx]
        roff = np.arange(-self.nrow, self.nrow + 1)
        toff = np.arange(-self.nra, self.nra + 1)
        el = bands.index_map["el"][:]
        ra_b = bands.index_map["ra"][:]
        periodic = _is_full_day(ra_b)

        out = containers.TransientCandidateCutouts(
            candidate=ncut,
            freq=bands.index_map["freq"],
            el_offset=roff,
            ra_offset=toff,
            attrs_from=table,
            distributed=True,
            comm=comm,
        )
        out.datasets["candidate_index"][:] = idx
        out.datasets["x_index"][:] = xi
        rows = e[:, np.newaxis] + roff[np.newaxis]
        rin = (rows >= 0) & (rows < el.size)
        out.datasets["el_index"][:] = np.where(rin, rows, -1)
        out.datasets["el"][:] = np.where(rin, el[np.clip(rows, 0, el.size - 1)], np.nan)
        ts = tb[:, np.newaxis] + toff[np.newaxis]
        if periodic:
            tin = np.ones(ts.shape, dtype=bool)
            ts = ts % ra_b.size
        else:
            tin = (ts >= 0) & (ts < ra_b.size)
        out.datasets["ra_index"][:] = np.where(tin, ts, -1)
        out.datasets["ra"][:] = np.where(tin, ra_b[np.clip(ts, 0, ra_b.size - 1)], np.nan)

        # Each rank's block of candidates, summed from the rows on every rank
        la = out.amplitude[:]
        offs = comm.allgather((la.local_offset[0], la.local_shape[0]))
        for r, (o, nl) in enumerate(offs):
            k = slice(o, o + nl)
            a, w = _band_cutouts(
                bands, self.pol, e[k], tb[k], xi[k], np.ones(nl, int), roff, toff
            )
            a, w = a.astype(np.float32), w.astype(np.float32)
            ra_ = np.zeros_like(a) if comm.rank == r else None
            rw_ = np.zeros_like(w) if comm.rank == r else None
            comm.Reduce(a, ra_, op=MPI.SUM, root=r)
            comm.Reduce(w, rw_, op=MPI.SUM, root=r)
            if comm.rank == r:
                out.amplitude[:].local_array[:] = ra_
                out.weight[:].local_array[:] = rw_
        self.log.info(f"Extracted the cutouts of {ncut} candidates.")
        return out


class SiftTransientCandidates(tasklib.base.ContainerTask):
    """Flag the candidates that are explained by something other than a transient.

    Each test sets a bit in ``flags`` of the ``sift`` table, and a candidate is kept
    if no test flags it.  The bits are (with the properties that set them):

    - 0 ``snr``: S/N below ``snr_min``.
    - 1 ``sun``: within ``sun_ha_max`` degrees of hour angle of the Sun.
    - 2 ``track``: within ``track_dy_max`` in telescope-y of the track of the Sun or
      a bright source.
    - 3 ``lobe``: on a grating lobe of the Sun or a bright source within
      ``lobe_ha_max`` degrees of hour angle.
    - 4 ``broadband``: ``col_nrows_max`` or more rows lit at the same time, more
      than the running median over the neighbouring samples.
    - 5 ``row_repeat``: more than ``row_npeak_max`` other peaks along the row, unless
      they are periodic.
    - 6 ``catalogue``: a catalogue source brighter than ``cat_flux_ratio`` times the
      candidate's amplitude within the match region, which could leave a residual
      of that size.
    - 7 ``narrowband``: more than ``band_frac_max`` of the S/N^2 in one band (if the
      spectra were added), or else more than ``group_frac_max`` in one group of
      bands.
    - 8 ``arc``: an island extending at least ``arc_dy_min`` in telescope-y and
      ``arc_dt_min`` seconds, with |correlation| of at least ``arc_corr_min``.
    - 9 ``wide``: wider than ``fwhm_y_max`` in telescope-y at half the peak.
    - 10 ``low_weight``: weight below ``weight_ratio_min`` times the median of the
      row (mostly flagged data).
    - 11 ``horizon``: |telescope-y| above ``horizon_y_max``, where the sensitivity is
      low and RFI from the ground is strong.
    - 12 ``unsmooth`` (only with the spectra): the spectrum is not a power law,
      with reduced chi-squared above ``smooth_chi2_max`` and less than
      ``smooth_frac_min`` of its power explained by the fit.  Both are needed:
      bright pulses have a large chi-squared from small deviations but are well
      explained, and faint ones are poorly explained only because of the noise.
    - 13 ``extended``: the extent (pixels joined across small gaps) reaches at least
      ``ext_dy_min`` in telescope-y, far more than a point source.  This catches
      arcs broken into blobs and short vertical streaks, which the ``arc`` test
      misses.
    - 14 ``ns_shape`` (only with the north-south fit, see :class:`AddNSProfile`):
      the profile across telescope-y is not the synthesized beam: the best width
      scale is above ``ns_scale_max`` or below ``ns_scale_min`` and improves the
      chi-squared by at least ``ns_dchi2_min`` over the point source.
    - 16 ``periodic`` (information only): at least ``periodic_npeak_min`` other
      peaks along the row, spaced by at least ``periodic_spacing_min`` seconds with
      a coefficient of variation below ``periodic_cv_max``.  The minimum spacing
      keeps out the flicker of scintillation, whose local maxima are spaced by about
      the width of the window that finds them.

    Attributes
    ----------
    select : bool
        Return only the candidates that are kept.  Default is False (all, with
        their flags).
    """

    FLAGS: ClassVar = {
        "snr": 0,
        "sun": 1,
        "track": 2,
        "lobe": 3,
        "broadband": 4,
        "row_repeat": 5,
        "catalogue": 6,
        "narrowband": 7,
        "arc": 8,
        "wide": 9,
        "low_weight": 10,
        "horizon": 11,
        "unsmooth": 12,
        "extended": 13,
        "ns_shape": 14,
        "periodic": 16,
    }
    INFO: ClassVar = ("periodic",)

    snr_min = config.Property(proptype=float, default=10.0)
    sun_ha_max = config.Property(proptype=float, default=20.0)
    track_dy_max = config.Property(proptype=float, default=0.02)
    lobe_ha_max = config.Property(proptype=float, default=10.0)
    col_nrows_max = config.Property(proptype=int, default=50)
    row_npeak_max = config.Property(proptype=int, default=0)
    periodic_npeak_min = config.Property(proptype=int, default=3)
    periodic_cv_max = config.Property(proptype=float, default=0.15)
    periodic_spacing_min = config.Property(proptype=float, default=240.0)
    cat_flux_ratio = config.Property(proptype=float, default=10.0)
    band_frac_max = config.Property(proptype=float, default=0.5)
    group_frac_max = config.Property(proptype=float, default=0.7)
    arc_dy_min = config.Property(proptype=float, default=0.02)
    arc_dt_min = config.Property(proptype=float, default=20.0)
    arc_corr_min = config.Property(proptype=float, default=0.7)
    fwhm_y_max = config.Property(proptype=float, default=0.03)
    ext_dy_min = config.Property(proptype=float, default=0.05)
    weight_ratio_min = config.Property(proptype=float, default=0.05)
    horizon_y_max = config.Property(proptype=float, default=0.95)
    smooth_chi2_max = config.Property(proptype=float, default=5.0)
    smooth_frac_min = config.Property(proptype=float, default=0.7)
    ns_dchi2_min = config.Property(proptype=float, default=25.0)
    ns_scale_max = config.Property(proptype=float, default=1.5)
    ns_scale_min = config.Property(proptype=float, default=0.67)
    select = config.Property(proptype=bool, default=False)

    def process(self, table):
        """Sift the candidates.

        Parameters
        ----------
        table : containers.TransientCandidateTable
            Candidates, from :class:`GroupTransientCandidates`.

        Returns
        -------
        out : containers.TransientCandidateTable
            A copy with the ``sift`` table, or only the candidates kept if ``select``.
        """
        rec = table.candidate[:]
        flags = np.zeros(rec.size, dtype=np.uint32)

        def flag(name, cond):
            flags[np.asarray(cond, dtype=bool)] |= np.uint32(1 << self.FLAGS[name])

        with np.errstate(invalid="ignore"):
            flag("snr", rec["snr"] < self.snr_min)
            flag("sun", np.abs(rec["sun_ha"]) < self.sun_ha_max)
            flag("track", np.abs(rec["track_dy"]) < self.track_dy_max)
            flag(
                "lobe",
                np.isfinite(rec["lobe_freq"])
                & (np.abs(rec["lobe_ha"]) < self.lobe_ha_max),
            )
            flag(
                "broadband",
                rec["col_nrows"].astype(int) - rec["col_nrows_bg"]
                >= self.col_nrows_max,
            )
            periodic = (
                (rec["row_npeak"] >= self.periodic_npeak_min)
                & (rec["row_spacing_cv"] < self.periodic_cv_max)
                & (rec["row_spacing"] >= self.periodic_spacing_min)
            )
            flag("periodic", periodic)
            flag("row_repeat", (rec["row_npeak"] > self.row_npeak_max) & ~periodic)
            flag(
                "catalogue",
                rec["cat_flux"] >= self.cat_flux_ratio * np.abs(rec["amplitude"]),
            )
            has_bands = rec["band_nsig"] >= 0
            flag(
                "narrowband",
                np.where(
                    has_bands,
                    rec["band_frac_max"] > self.band_frac_max,
                    rec["group_frac_max"] > self.group_frac_max,
                ),
            )
            flag(
                "arc",
                (rec["island_dy"] >= self.arc_dy_min)
                & (rec["island_dt"] >= self.arc_dt_min)
                & (np.abs(rec["island_corr"]) >= self.arc_corr_min),
            )
            flag("wide", rec["fwhm_y"] > self.fwhm_y_max)
            flag("extended", rec["ext_dy"] >= self.ext_dy_min)
            flag("low_weight", rec["weight_ratio"] < self.weight_ratio_min)
            flag("horizon", np.abs(rec["y"]) > self.horizon_y_max)
            flag(
                "unsmooth",
                has_bands
                & (rec["band_chi2"] > self.smooth_chi2_max)
                & (rec["band_fit_frac"] < self.smooth_frac_min),
            )
            flag(
                "ns_shape",
                (rec["ns_dof"] > 0)
                & (rec["ns_dchi2"] >= self.ns_dchi2_min)
                & (
                    (rec["ns_scale"] > self.ns_scale_max)
                    | (rec["ns_scale"] < self.ns_scale_min)
                ),
            )

        reject = np.uint32(
            sum(1 << b for n, b in self.FLAGS.items() if n not in self.INFO)
        )
        keep = (flags & reject) == 0

        counts = ", ".join(
            f"{n} {np.sum((flags >> b) & 1)}" for n, b in self.FLAGS.items()
        )
        self.log.info(
            f"Kept {keep.sum()} of {rec.size} candidates.  Flagged: {counts}."
        )

        rows = np.flatnonzero(keep) if self.select else np.arange(rec.size)
        out = containers.TransientCandidateTable(
            candidate=rows.size,
            axes_from=table,
            attrs_from=table,
            comm=table.comm,
        )
        for name in table.datasets:
            if name not in out.datasets:
                out.add_dataset(name)
            out.datasets[name][:] = table.datasets[name][:][rows]
        if "sift" not in out.datasets:
            out.add_dataset("sift")
        sift = np.zeros(rows.size, dtype=out.datasets["sift"][:].dtype)
        sift["keep"] = keep[rows]
        sift["flags"] = flags[rows]
        out.datasets["sift"][:] = sift
        out.attrs["sift_flags"] = json.dumps(self.FLAGS)
        for key in (
            "snr_min", "sun_ha_max", "track_dy_max", "lobe_ha_max", "col_nrows_max",
            "row_npeak_max", "periodic_npeak_min", "periodic_cv_max",
            "periodic_spacing_min", "cat_flux_ratio",
            "band_frac_max", "group_frac_max", "arc_dy_min", "arc_dt_min",
            "arc_corr_min", "fwhm_y_max", "ext_dy_min", "weight_ratio_min",
            "horizon_y_max",
            "smooth_chi2_max", "smooth_frac_min", "ns_dchi2_min", "ns_scale_max",
            "ns_scale_min",
        ):  # fmt: skip
            out.attrs[f"sift_{key}"] = getattr(self, key)
        return out
