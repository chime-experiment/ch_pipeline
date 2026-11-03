"""Tasks for the search for long period transients in hybrid beamformed data."""

import inspect
import json
import time
from typing import ClassVar

import numpy as np
import scipy.constants
import scipy.interpolate
import scipy.ndimage
import scipy.special
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

    inv_denominator = tools.invert_no_zero(np.where(valid, denominator, 0.0))
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
                    npw[pf] += wt * tools.invert_no_zero(w)
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
                    inv_g = tools.invert_no_zero(G1)
                    a = B1 * inv_g
                    # Clip the weights at a few times their typical size in the row
                    power = np.where(G1 > 0, np.abs(a) ** 2, np.nan)
                    typical = np.nanmedian(power, axis=-1, keepdims=True)
                    typical = np.where(np.isfinite(typical), typical, 0.0)
                    lim = np.sqrt(self.clip * (typical + inv_g))
                    big = np.abs(a) > lim
                    a = np.where(big, a * tools.invert_no_zero(np.abs(a)) * lim, a)
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
        ratio = np.where(ok & (npw > 0), pw * tools.invert_no_zero(npw), np.nan)
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
            noise = tools.invert_no_zero(np.where(has, G[..., j, j], 0.0))
            x = np.abs(B[..., j] * noise) ** 2
            s = np.full((*G.shape[:-3], blk_start.size), np.clip(var0[j], lo, hi))
            for _ in range(niter):
                tot = s[..., blk] + noise
                u = np.where(has, tools.invert_no_zero(tot**2), 0.0)
                xc = np.minimum(x, self.prior_clip * tot)
                num = np.add.reduceat(u * (xc - noise), blk_start, axis=-1)
                den = np.add.reduceat(u, blk_start, axis=-1)
                s = np.clip(
                    np.where(den > 0, num * tools.invert_no_zero(den), lo), lo, hi
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
    scale = np.where(r2 > 1.0, tools.invert_no_zero(np.sqrt(r2)), 1.0)
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
            zs = np.where(valid, Z[bb] * tools.invert_no_zero(s), 0.0)
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
    r = np.where(valid, Z * tools.invert_no_zero(z_med[nz]), 0.0).astype(np.float64)
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
        # The correction for correlated channels of each elevation, from its block
        corr = np.ones(shp, dtype=np.float32)

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
                for rows, bk in blocks:
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
                    # zero) are left out of the block's fit.  All of the block's
                    # elevations, not only the local ones: a block spans ranks, and
                    # the fit must not depend on how it is split
                    blk_ok = (
                        np.all(nscale_all[ip, :, ie, gsl] > 0, axis=1)
                        if nscale_all is not None
                        else np.ones(freq.size, dtype=bool)
                    )
                    for tt in range(nra):
                        W = w[:, tt] * isc_b * blk_ok  # inverse variance of each channel
                        m = W > 0
                        if m.sum() <= nmode + 2:
                            continue
                        Am = A[m]
                        G = Am.T @ (W[m, np.newaxis] * Am)
                        G += self.ridge * np.trace(G) / nmode * np.eye(nmode)
                        Gi = np.linalg.inv(G)
                        # Diagonal of the hat matrix, as a matrix product (a
                        # three-operand einsum is ~50x slower and dominated the task)
                        lev = np.zeros(freq.size)
                        lev[m] = W[m] * np.sum((Am @ Gi) * Am, axis=1)
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
                        nb = bsum @ m
                        pairs = bsum @ (same & m & np.roll(m, -1))
                        corr[op, :, oe, esl, tt] = (
                            1.0
                            + (2.0 * self.channel_correlation * pairs)
                            * tools.invert_no_zero(nb)
                        )[:, np.newaxis]
                        # The degrees of freedom saved are those of the first block
                        # (on rank 0, which writes them)
                        if bk == 0:
                            dof[op, :, oe, tt] = bsum @ h1
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

        scale = h * common[:, :, :, np.newaxis] * corr

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


def _log_memory(task, label):
    """Log the largest resident memory of a rank and of a node (collective)."""
    try:
        import psutil
    except ImportError:
        return
    from mpi4py import MPI

    # Private memory of the rank: the arrays shared by the ranks of a node (see
    # _NodeSharedFreq) count once, in the node total
    mi = psutil.Process().memory_info()
    rss = mi.rss - getattr(mi, "shared", 0)
    vm = psutil.virtual_memory()
    node = vm.total - vm.available
    comm = task.comm
    if comm is not None and comm.size > 1:
        rss = comm.allreduce(rss, op=MPI.MAX)
        node = comm.allreduce(node, op=MPI.MAX)
    if comm is None or comm.rank == 0:
        task.log.info(
            f"Memory ({label}): largest rank {rss / 2**30:.1f} GiB (private), largest "
            f"node {node / 2**30:.0f} GiB."
        )


def _node_comms(comm):
    """Communicators of the ranks of each node, and of one rank (the leader) per node.

    The leaders' communicator is ``MPI.COMM_NULL`` on the other ranks.
    """
    from mpi4py import MPI

    node = comm.Split_type(MPI.COMM_TYPE_SHARED, key=comm.rank)
    leaders = comm.Split(0 if node.rank == 0 else MPI.UNDEFINED, comm.rank)
    return node, leaders


class _NodeSharedFreq:
    """The whole frequency axis of a frequency-distributed dataset, one copy per node.

    The copy is in memory shared by the ranks of a node (an MPI shared window), with
    frequency as its first axis; ``array`` is a read-only view.  Each rank writes its
    own frequencies, and the leaders of the nodes broadcast them to each other, so
    every node receives the dataset once, whatever the number of ranks on it.  Call
    :meth:`free` (collective) when done.

    Parameters
    ----------
    dset : MPIArray
        Distributed over its second axis (frequency).
    comm : MPI.Comm
        The communicator of ``dset``.
    take : callable, optional
        Applied to the local array [axis 0, freq, ...] before it is stored, for a
        selection or a change of type (the frequency axis must stay).
    """

    def __init__(self, dset, comm, take=None):
        from mpi4py import MPI

        local = dset.local_array if take is None else take(dset.local_array)
        local = np.ascontiguousarray(np.moveaxis(local, 1, 0))
        lo, nloc = dset.local_offset[1], dset.local_shape[1]
        shape = (dset.global_shape[1], *local.shape[1:])
        dtype = local.dtype

        node, leaders = _node_comms(comm)
        self._comms = (node, leaders)
        nbytes = int(np.prod(shape)) * dtype.itemsize if node.rank == 0 else 0
        self._win = MPI.Win.Allocate_shared(nbytes, dtype.itemsize, comm=node)
        buf, _ = self._win.Shared_query(0)
        full = np.ndarray(buffer=buf, dtype=dtype, shape=shape)
        full[lo : lo + nloc] = local
        del local

        # The leader of each rank's node (its rank among the leaders) and the
        # frequencies of each rank
        lid = node.bcast(leaders.rank if node.rank == 0 else None, root=0)
        owner = comm.allgather((lid, lo, nloc))
        node.Barrier()
        if node.rank == 0 and leaders.size > 1:
            for root, o, n in owner:
                if n > 0:
                    leaders.Bcast(full[o : o + n], root=root)
        node.Barrier()
        full.flags.writeable = False
        self.array = full

    def free(self):
        """Release the shared memory (collective over the communicator)."""
        from mpi4py import MPI

        self.array = None
        self._win.Free()
        node, leaders = self._comms
        node.Free()
        if leaders != MPI.COMM_NULL:
            leaders.Free()


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
    of ``kappa_prior``) only constrain ``alpha``, the nuisance scale of the
    thermal noise, so every ``noise_stride``-th of them is used, on a regular grid
    in global (channel, elevation, RA) indices (the same for any distribution over
    the ranks).  Band sums use the samples with at least
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
    noise_stride : int
        Use one in this many of the other samples (0 for none).  Default is 50.
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
    """

    pol = config.list_type(str, default=["XX", "YY"])
    ew_index = config.list_type(int, default=None)
    nblock = config.Property(proptype=int, default=12)
    el_edges = config.list_type(float, default=None)
    clip = config.Property(proptype=float, default=20.0)
    niter = config.Property(proptype=int, default=20)
    kappa_prior = config.Property(proptype=float, default=1e-3)
    noise_stride = config.Property(proptype=int, default=50)
    sources = config.list_type(str, default=[])
    positions = config.Property(proptype=dict, default={})
    source_el_width = config.Property(proptype=int, default=10)
    source_ra_width = config.Property(proptype=float, default=3.0)
    qmin = config.Property(proptype=float, default=1e-2)
    qmax = config.Property(proptype=float, default=1e8)
    nqbin = config.Property(proptype=int, default=40)

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
        # Flattened global index of each local (elevation, RA) sample
        gidx = iel.astype(np.int64)[:, np.newaxis] * nra + np.arange(nra)

        def subsample(valid, informative, chans=None):
            # Every noise_stride-th sample by global index ([chan,] el, ra)
            if self.noise_stride <= 0:
                return valid & informative
            idx = gidx
            if chans is not None:
                idx = idx + chans.astype(np.int64)[:, np.newaxis, np.newaxis] * (
                    el.size * nra
                )
            return valid & (informative | (idx % self.noise_stride == 0))

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

                    keep = subsample(valid, R2 * kp2 > 1e-3 * N, chans)
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


def _interp_weights(frac, m, bandwidth):
    """Weights of a band-limited interpolation between grid points.

    The least-squares weights ``c_m`` of the grid points at offsets ``m`` that
    give a signal at offset ``frac`` (between 0 and 1) from the grid point at
    ``m = 0``, for signals with frequencies up to ``bandwidth`` cycles per grid
    point: ``sum_m c_m e^{2 pi i f m}`` is closest to ``e^{2 pi i f frac}`` over
    ``|f| <= bandwidth``.

    Parameters
    ----------
    frac : np.ndarray[n]
        Offsets.
    m : np.ndarray[ntap] of int
        Offsets of the grid points used.
    bandwidth : float
        Largest frequency, in cycles per grid point.

    Returns
    -------
    weights : np.ndarray[n, ntap]
    """
    b2 = 2.0 * bandwidth
    gram = b2 * np.sinc(b2 * (m[:, np.newaxis] - m[np.newaxis, :]))
    gram += 1e-10 * np.trace(gram) * np.eye(m.size)
    rhs = b2 * np.sinc(b2 * (m[:, np.newaxis] - np.asarray(frac)[np.newaxis, :]))
    return np.linalg.solve(gram, rhs).T


def _q_factors(rx, ry):
    """Factors of XX and YY in Q that cancel I, split into telescope-x and time.

    Q is ``f_XX N_XX - f_YY N_YY``, which has no I where ``f_XX R_XX = f_YY R_YY``.
    The responses are close to ``R_p(x, t) = A_p(x) w_p(t)``: the beam squared,
    summed over the band, times the weights, which change in time (the sky term,
    the in-band scale, flags) and differ between the products (XX and YY are
    calibrated on Cyg A, where their beams differ).  The factors are taken in that
    form, ``f_p = a_p(x) b_p(t)``, with ``a = 1 -/+ lambda(x)`` from the mean of the
    responses over time and ``b = 1 -/+ mu(t)`` from their sum over x relative to
    its mean.  They then cancel I exactly where ``R_p`` separates, and otherwise up
    to the change of the beam within the band times that of the weights.  Being
    separable, the factor in x scales the template and the factor in time the
    weights, so the response, variance and coupling of the samples of a track of Q
    follow from them.  Both factors lie in [0, 2].

    Parameters
    ----------
    rx, ry : np.ndarray[el, x, ra]
        Single-sample responses of XX and YY in the band.

    Returns
    -------
    qa : np.ndarray[2, el, x]
        ``a`` of XX and YY.
    qb : np.ndarray[2, el, ra]
        ``b`` of XX and YY.
    """
    ax = rx.mean(axis=-1, dtype=np.float64)
    ay = ry.mean(axis=-1, dtype=np.float64)
    lam = (ax - ay) * tools.invert_no_zero(ax + ay)

    wx = rx.sum(axis=1, dtype=np.float64)
    wy = ry.sum(axis=1, dtype=np.float64)
    wx = wx * tools.invert_no_zero(wx.mean(axis=-1, keepdims=True))
    wy = wy * tools.invert_no_zero(wy.mean(axis=-1, keepdims=True))
    mu = (wx - wy) * tools.invert_no_zero(wx + wy)

    qa = np.stack([1.0 - lam, 1.0 + lam]).astype(np.float32)
    qb = np.stack([1.0 - mu, 1.0 + mu]).astype(np.float32)
    return qa, qb


class _Track:
    """Tracks of sources through the telescope-x grid, as interpolation taps.

    ``index`` and ``weight`` are ``[step, ..., x, tap]``: a quantity at the
    position of a source ``step`` samples after it was at grid point ``x`` is the
    sum over taps of ``weight`` times its value at grid point ``index``.  Samples
    where the source is off the grid have zero weight.  Indexing a track indexes
    its leading axes (the step and the elevation).
    """

    def __init__(self, index, weight):
        self.index = index
        self.weight = weight

    def __getitem__(self, key):
        return _Track(self.index[key], self.weight[key])

    def taps(self, j):
        """(index, weight) ``[..., x]`` of each tap of step ``j`` that is used."""
        idx, wt = self.index[j], self.weight[j]
        for t in range(idx.shape[-1]):
            if np.any(wt[..., t]):
                yield idx[..., t], wt[..., t]

    def template(self, tc, j):
        """The template ``tc`` [k, el, x] at the position of step ``j``."""
        out = np.zeros(tc.shape, dtype=np.result_type(tc, np.complex64))
        for idx, wt in self.taps(j):
            out += wt[np.newaxis] * np.take_along_axis(tc, idx[np.newaxis], axis=2)
        return out

    def power(self, tc, j, tc2=None):
        """``|tc|^2`` [k, el, x] interpolated to the position of step ``j``.

        This is what a quantity of the form ``sum_k |g_k|^2 a_k`` on the grid
        becomes when it is interpolated along the track; it differs from
        ``|template(tc, j)|^2``, the exact value for the interpolated template.
        With ``tc2``, ``Re(conj(tc) tc2)`` instead.
        """
        out = np.zeros(tc.shape, dtype=np.float64)
        for idx, wt in self.taps(j):
            g = np.take_along_axis(tc, idx[np.newaxis], axis=2)
            if tc2 is None:
                out += wt[np.newaxis] * (g.real**2 + g.imag**2)
            else:
                g2 = np.take_along_axis(tc2, idx[np.newaxis], axis=2)
                out += wt[np.newaxis] * (g.real * g2.real + g.imag * g2.imag)
        return out


class TransientMatchedFilter(_BandMixin, tasklib.base.ContainerTask):
    """Matched filter for transients in m-mode filtered hybrid beamformed visibilities.

    Searches over the start time, duration, telescope-x, and spectral index of a
    transient in each elevation.  The template follows the source as it drifts
    through the primary beam: the numerator at each sample is interpolated from
    the grid of telescope-x to the source's position (band-limited, over
    ``track_taps`` grid points), and the denominator uses the same interpolated
    template.  The nearest grid point at each sample (``track_taps`` 1) loses much
    of a long pulse: the filter leaves mostly the differences between its samples,
    which the changing phase error of the template then corrupts.  For
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
    track_taps : int
        Number of grid points of telescope-x over which the track is interpolated
        for durations longer than one sample (1 for the nearest).  Default is 6.
    track_margin : float
        Band of the interpolation beyond the highest fringe rate on the grid
        (``max(ew) nu_max / c`` times the spacing), in cycles per grid point, for
        the primary beam.  Default is 0.02.
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
    track_taps = config.Property(proptype=int, default=6)
    track_margin = config.Property(proptype=float, default=0.02)
    stokes = config.Property(proptype=bool, default=False)

    def setup(self, manager, beam):
        """Set the telescope and primary beam model.

        Parameters
        ----------
        manager : ProductManager, BeamTransfer, or TransitTelescope
            Object from which a telescope instance is extracted.
        beam : draco.core.containers.GridBeam
            Primary beam model in telescope coordinates.
        """
        if getattr(self, "stokes", False) and type(self) is TransientMatchedFilter:
            raise NotImplementedError(
                "The Stokes search needs TransientMatchedFilterSkyNoise (noise=inband, "
                "sky or skycell)."
            )
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

        # Track of a source starting at each (el, x): interpolation taps after j samples
        track = self._make_tracks(
            ha[el_slc], dec[el_slc], visible[el_slc], x, dra, wmax, freq, ew[iew]
        )

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

    def _make_tracks(self, ha, dec, visible, x, dra, wmax, freq, ew_used):
        """The tracks of :meth:`_tracks`, with the taps and band of this grid."""
        ntap = max(int(self.track_taps), 1)
        dx = x[1] - x[0] if x.size > 1 else 1.0
        umax = np.max(np.abs(ew_used)) * np.max(freq) * 1e6 / scipy.constants.c
        bandwidth = umax * dx + self.track_margin
        if ntap > 1 and bandwidth >= 0.5:
            self.log.warning(
                f"The telescope-x grid samples the fringes too coarsely to interpolate "
                f"({bandwidth:.2f} cycles per point); using the nearest x instead."
            )
            ntap = 1
        return self._tracks(ha, dec, visible, x, dra, wmax, ntap, bandwidth)

    @staticmethod
    def _tracks(ha, dec, visible, x, dra, wmax, ntap=1, bandwidth=0.25):
        """The track of a source starting at each (el, x), as interpolation taps.

        ``j`` samples after the start the source is at telescope-x
        ``-cos(dec) sin(ha + j dra)``.  A quantity there is the sum of its values at
        ``ntap`` neighbouring grid points with the weights of
        :func:`_interp_weights` (band-limited to ``bandwidth`` cycles per grid
        point), or the value at the nearest grid point if ``ntap`` is 1, if the
        neighbours run off the grid, or if the source is on a grid point (always the
        case at the start).  Tracks that start below the horizon (``visible`` is
        False), and the samples where the source has left the grid, have no taps.

        Returns
        -------
        track : _Track
            Taps ``[wmax, el, x, ntap]``.
        """
        dx = x[1] - x[0] if x.size > 1 else 1.0
        nx = x.size
        j = np.arange(wmax)[:, np.newaxis, np.newaxis]
        with np.errstate(invalid="ignore"):
            xj = -np.cos(dec)[np.newaxis] * np.sin(ha[np.newaxis] + j * dra)
            pos = (xj - x[0]) / dx
            near = np.rint(pos)
        ok = np.isfinite(near) & (near >= 0) & (near < nx) & visible[np.newaxis]
        near = np.where(ok, near, 0).astype(np.int64)
        pos = np.where(ok, pos, near)

        index = np.zeros((*near.shape, ntap), dtype=np.int64)
        weight = np.zeros((*near.shape, ntap), dtype=np.float64)
        centre = (ntap - 1) // 2
        index[...] = near[..., np.newaxis]
        weight[..., centre] = ok

        if ntap > 1:
            m = np.arange(ntap) - centre
            base = np.floor(pos).astype(np.int64)
            taps = base[..., np.newaxis] + m
            interp = (
                ok
                & (taps[..., 0] >= 0)
                & (taps[..., -1] < nx)
                & (np.abs(pos - near) > 1e-6)
            )
            if np.any(interp):
                index[interp] = taps[interp]
                weight[interp] = _interp_weights(
                    pos[interp] - base[interp], m, bandwidth
                )

        return _Track(index, weight.astype(np.float32))

    @staticmethod
    def _track_numerator(n1, track, durations):
        """Sum the numerator along the source tracks.

        Parameters
        ----------
        n1 : np.ndarray[el, x, ra]
            Numerator for each sample, evaluated with the template at each grid x.
        track : _Track
            From :meth:`_tracks`, ``[wmax, el, x, tap]``.
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
            # The statistic for a transient starting at tau uses sample tau + j
            nj = np.zeros(n1.shape, dtype=np.complex128)
            for idx, wt in track.taps(jj):
                nj += wt[..., np.newaxis] * np.take_along_axis(
                    n1, idx[..., np.newaxis], axis=1
                )
            nacc += np.roll(nj, -jj, axis=-1)
            if jj + 1 in wset:
                ntrack.append(nacc.copy())

        return ntrack

    @staticmethod
    def _track_denominator(tc, ak, track, durations):
        """Sum the denominator along the source tracks.

        ``D(tau) = sum_k sum_{j, j' < W} conj(g_kj) g_kj' A_k,(j' - j)(tau + j)``,
        where ``g_kj`` is the template of visibility ``k`` interpolated to the
        source's position after ``j`` samples (as the numerator is), and ``A`` is
        from :meth:`_lag_coefficients`.

        Parameters
        ----------
        tc : np.ndarray[k, el, x]
            Template at each grid x.
        ak : np.ndarray[k, lag, ra]
            From :meth:`_lag_coefficients`.
        track : _Track
            From :meth:`_tracks`, ``[wmax, el, x, tap]``.
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
        g = [
            track.template(tc, jj).reshape(nk, -1).astype(np.complex128)
            for jj in range(wmax)
        ]

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
    the filter response out to lag W - 1.  The group maps are single-sample only.
    The per-band output is for the durations in ``band_durations`` (which must
    include 1): for a longer one, the band's own numerator, response and variance
    summed along the track with its own coupling terms, so that the amplitude of
    each band at that duration is unbiased (the sum of the single-sample output of
    a band along a track leaves out the coupling, which removes much of a long
    pulse, more at the higher frequencies).  Each longer band duration holds
    ``[band, pol group, el, x, ra]`` coupling terms for an elevation pass, so it
    is meant for small inputs (see :class:`CandidateBandSpectra`).

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
    are given zero weight.  If an in-band noise scale is given (see
    :class:`EstimateInbandNoise`), it multiplies the thermal variance
    ``sigma_k^2``.

    The sky term changes faster than the block or cell fits can follow (on some
    nights over a few minutes).  That is left to :class:`CalibrateTrialNoise`,
    which rescales the output by its measured spread around each sample.

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
    inband_drop_inc : bool
        With an in-band noise scale from :class:`EstimateInbandNoise`, which measures
        the noise that is independent between channels (including the self-noise
        of bright sources), leave out the incoherent sky term ``kappa_inc`` (use
        ``kappa_tot = kappa_coh``) so that it is not counted twice.  Default is
        True.
    stokes : bool
        Search in Stokes parameters (see below).  ``pol`` must be the four products
        XX, XY, YX and YY; ``combine_pol`` is ignored.  Default is False.
    xy_delay : float
        Delay of the X-Y phase in ns (stokes).  Default is -0.95 (B0329+54 and
        J1634+44, 2024-2026).
    xy_offset : float
        Offset of the X-Y phase in degrees (stokes).  Default is 20.
    rm_max : float
        Largest |RM| of the search in linear polarisation in rad/m^2 (stokes).
        Default is 300.
    rm_oversample : float
        RM trials per half width of the RM spread function (stokes).  Default is 3.
    rm_spectral_indices : list of float
        Spectral indices of the trials for which the RM search is made; for the
        others the linear-polarisation trials have zero weight.  Default is [0].
    rm_ntrials_eff : float
        Number of independent RM trials in the conversion of the RM search to a
        Gaussian S/N.  Default depends on the level ``T``: ``n(T) = 1 + 2 rm_max
        sqrt(4 Var(l^2) T / (2 pi))``, from the expected number of upcrossings of
        a chi-squared field with 2 dof (Worsley 1994), with the variance of
        ``l^2`` over the bands weighted by ``s_b^2`` and their total response
        (in simulations of noise the S/N then has mean ~0, standard deviation
        ~1 and the tail of a unit Gaussian to ~10%; the RM range over the width
        of the RM spread function gives too few, a mean of +0.6).
    rm_chunk : int
        RM trials processed at once.  Default is 32.
    el_pass : int
        Number of local elevations processed in each pass.  The per-band sums
        (numerators, responses, variances and sky terms of every band, telescope-x
        and sample) are held for one pass at a time, which bounds the memory; the
        filter response of each band is gathered once per pass.  Default is None
        (all at once).
    rm_sky_modes : int
        Modes of the correlation between bands for which the sky term of Q in the
        RM search is computed at each RM; the others are taken at their mean over
        RM.  Default is 3 (the leading mode holds ~88% of the trace on 2418 and
        3035).

    With ``stokes``, the products are combined in each band before the bands are
    combined.  I is formed as for the co-polar search (``combine_pol``, so its
    trials are the same).  Q is ``f_XX N_XX - f_YY N_YY`` with factors that cancel
    I in the responses of the matched filter itself, for each band, elevation,
    telescope-x and sample (see :func:`_q_factors`).  These must follow the weights
    in time: XX and YY are calibrated on Cyg A, where their beams differ, so their
    weights differ (by ~2 on 2418), and the sky term changes them along a transit.
    The factors are the product of one in telescope-x, which scales the template of
    each product, and one in time, which scales its weights, so Q is a matched
    filter with those templates and weights and its response, variance and coupling
    of the samples of a track follow from them.  (A factor fixed in time, from the
    beam and a weight common to both products, left ~1/3 of I in Q on 2418.
    Normalising each product by its own response at each sample cancelled I, but
    its coupling of the samples was taken to be that of I, and it amplified a
    product without signal where its beam is far below the other's: the variance
    of the long durations of L was then wrong by large factors.)  A sample is used
    only where all four products have weight.  With the X-Y phase
    ``phi_b = xy_offset + 2 pi nu_b xy_delay``, ``XY ~ (U + iV) e^{i phi}`` and
    ``YX ~ (U - iV) e^{-i phi}``, so ``e^{-i phi_b} (N_XY + conj N_YX)`` has U in
    its real and V in its imaginary part (YX is conjugated, which leaves the
    response and the coupling unchanged).
    Q has the sky term of the polarisation difference; U and V have no sky term
    (the unpolarised sky does not reach them except by leakage, which the in-band
    scale measures).

    The trials then have the polarisations I, V and L:

    - I: as above (the imaginary part, if any, is the null test).
    - V: circular polarisation, coherent over the bands with the spectral index of
      the trial.  The imaginary part holds U at RM = 0.
    - L: linear polarisation, ``P = Q + iU`` searched over RM:
      ``S = sum_b s_b e^{-2 i RM l_b^2} (N_Q,b + i N_U,b)`` along the track, and
      ``T = |S|^2 / Var``, chi-squared with 2 dof for noise, maximised over the RM
      trials and converted to a Gaussian S/N ``z`` as the largest of
      ``rm_ntrials_eff`` independent trials.  The amplitude is ``sign(z) |P|`` (Jy
      at the reference frequency) and the weight ``(z / |P|)^2``, so that the S/N
      is ``z``; the imaginary part holds the best RM.  The response and the
      thermal variance are the means of those of Q and U, which differ by the beam
      and the weights of the products; the depolarisation within a band is left out
      of the template (``sinc(RM d l^2)``, ~0.5 at 400 MHz for |RM| = 120).  The
      sky term of Q is coherent between the bands only near RM = 0: its variance in
      ``S`` is ``sum_bb' c_b c_b'^* rho_bb' Re(V_b V_b'^*) / 2`` with
      ``c_b = s_b e^{-2 i RM l_b^2}``, computed at every RM for the leading
      ``rm_sky_modes`` modes of ``rho`` and at its mean over RM (the diagonal) for
      the rest.  Using the RM = 0 value at every RM suppressed L wherever the sky
      term is large (the night clouds, the horizon).  The coefficients of
      ``n(t) = a + c sqrt(t)`` are kept in the ``rm_ntrials_coef`` attribute, and the
      S/N of Q at RM 0 (with its sky term) in ``snr_q``, for the noise calibration
      of L (see :class:`CalibrateTrialNoise`).  The statistic takes Q and U to have
      the same variance; it is ``(V_Q - V_U) / (V_Q + V_U)`` <~ 0.1 for
      |telescope-y| < 0.85, and up to ~0.6 beyond, where the XX and YY responses
      differ by up to ~80 and the tail of L is too heavy at RM ~ 0 (the sifter's
      ``horizon_y_max``).

    The per-band output has I, Q, U and V.  The group maps are made for I.
    """

    groups = config.Property(proptype=dict, default={})
    group_x_reduce = config.Property(proptype=list, default=["max"])
    group_spectral_index = config.Property(proptype=float, default=0.0)
    shrink_blocks = config.Property(proptype=bool, default=True)
    kappa_floor = config.Property(proptype=float, default=0.1)
    inband_drop_inc = config.Property(proptype=bool, default=True)
    rho_band_default = config.Property(proptype=float, default=1.0)
    rho_pol_default = config.Property(proptype=float, default=1.0)
    xy_delay = config.Property(proptype=float, default=-0.95)
    xy_offset = config.Property(proptype=float, default=20.0)
    rm_max = config.Property(proptype=float, default=300.0)
    rm_oversample = config.Property(proptype=float, default=3.0)
    rm_spectral_indices = config.list_type((int, float), default=[0.0])
    rm_ntrials_eff = config.Property(proptype=float, default=None)
    rm_chunk = config.Property(proptype=int, default=32)
    rm_sky_modes = config.Property(proptype=int, default=3)
    el_pass = config.Property(
        proptype=lambda v: None if v is None else int(v), default=None
    )

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
        _log_memory(self, "matched filter, start")
        durations = sorted(set(self.durations))
        if min(durations) < 1:
            raise ValueError("Durations must be at least one sample.")
        band_durations = sorted(set(self.band_durations))
        if (
            durations[0] != 1
            or band_durations[0] != 1
            or not set(band_durations) <= set(durations)
        ):
            raise ValueError(
                "With the sky noise, durations must include 1 and band_durations must "
                "include 1 and be a subset of durations."
            )
        blong = [W for W in band_durations if W > 1]
        wmax = max(durations)
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
        if self.stokes:
            # Groups I, Q and C (see the class docstring), and the products whose
            # coupling of the samples of a track each needs (cmap: its entry); that
            # of Q is formed with its own template
            sidx = self._stokes_index()
            co = [sidx["XX"], sidx["YY"]]
            pgroups = [co, co, [sidx["XY"], sidx["YX"]]]
            opol = np.array(["I", "V", "L"])
            bpol = np.array(["I", "Q", "U", "V"])
            ngp = 3
            cmap = [0, 1, 2]
        else:
            pgroups = (
                [list(range(len(ipol)))]
                if self.combine_pol
                else [[pp] for pp in range(len(ipol))]
            )
            opol = np.array(["co"]) if self.combine_pol else np.array(self.pol)
            bpol = opol
            ngp = len(pgroups)
            cmap = list(range(ngp))

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
                "Leaving out the incoherent sky term: the in-band noise scale measures it."
            )
            k2tot = k2coh.copy()
        k2inc = np.clip(k2tot - k2coh, 0.0, None)
        kcoh = np.sqrt(k2coh)  # band, ew, yband, ra
        lam, qvec = self._modes(rho_band)

        # Thermal weights and the filter response, gathered over frequency once per
        # node into memory shared by its ranks (every rank needs all frequencies)
        hvis.redistribute("freq")
        wname = "noise_weight" if has_response else hvis._weight_dset_name
        if "elevation_vis_weight" in hvis.datasets:
            raise ValueError("Elevation-dependent weights are not supported.")
        self_conj = _self_conjugate([pol[ii] for ii in ipol], ew[iew])
        half = np.where(self_conj, 0.5, 1.0).astype(np.float32)[
            :, np.newaxis, :, np.newaxis
        ]

        def take_weight(a):
            w = a[ipol][:, :, iew].astype(np.float32) * half
            if self.stokes:
                # Only the samples where all four products are present
                w *= np.all(w > 0, axis=0, keepdims=True)
            return w

        shared = [_NodeSharedFreq(hvis[wname][:], hvis.comm, take=take_weight)]
        weight_all = np.moveaxis(shared[0].array, 0, 1)  # pol, freq, ew, ra
        if has_response:
            shared.append(
                _NodeSharedFreq(
                    hvis.ra_response[:], hvis.comm, take=lambda a: a[ipol][:, :, iew]
                )
            )
            resp_all = shared[1].array  # freq, pol, ew, ra, lag
        _log_memory(self, "matched filter, weights and filter response gathered")

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
            freq=band_map, duration=np.array(band_durations), **{**common, "pol": bpol}
        )
        for cont in (trials, bands):
            if (self.save_imag and not (self.stokes and cont is bands)) or (
                self.stokes and cont is trials
            ):
                cont.add_dataset("amplitude_imag")
            cont.redistribute("el")
            cont.datasets["ha"][:] = np.degrees(ha)
            cont.datasets["dec"][:] = np.degrees(dec)
            cont.attrs["ew_used"] = ew[iew]
            cont.attrs["pol_used"] = [pol[ii] for ii in ipol]
            cont.attrs["duration_unit_seconds"] = 86164.0905 / 360.0 * np.degrees(dra)
            cont.attrs["time_axis"] = "start of the transient"
            cont.attrs["sky_noise"] = True
        if self.stokes:
            # Q at RM 0, for the noise calibration of L (CalibrateTrialNoise); 0
            # where there is no trial (NaN would trip the pipeline's NaN check)
            trials.add_dataset("snr_q")
            trials.datasets["snr_q"][:].local_array[:] = 0.0
        trials.attrs["reference_freq"] = self.reference_freq
        bands.attrs["band_start"] = self.band_start
        bands.attrs["band_width"] = self.band_width

        if self.stokes:
            # X-Y phase of each band, and the RM trials
            xy_phase = np.radians(self.xy_offset) + 2.0 * np.pi * (
                band_map["centre"] * 1e6
            ) * (self.xy_delay * 1e-9)
            lam2_b = (scipy.constants.c / (band_map["centre"] * 1e6)) ** 2
            rm = self._rm_grid(lam2_b)
            rm_coef = np.exp(-2.0j * rm[np.newaxis, :] * lam2_b[:, np.newaxis])
            rm_alpha = [
                ia
                for ia, a in enumerate(alphas)
                if np.any(np.isclose(a, self.rm_spectral_indices))
            ]
            self.log.info(
                f"Stokes search: X-Y phase {self.xy_offset:+.1f} deg + delay "
                f"{self.xy_delay:+.2f} ns; {rm.size} RM trials over +-{self.rm_max:g} "
                f"rad/m^2 at spectral indices {list(alphas[rm_alpha])}."
            )
            for cont in (trials, bands):
                cont.attrs["xy_delay_ns"] = self.xy_delay
                cont.attrs["xy_offset_deg"] = self.xy_offset
            trials.attrs["rm_trials"] = rm
            trials.attrs["rm_spectral_indices"] = alphas[rm_alpha]
            trials.attrs["amplitude_imag"] = "I: imaginary part; V: U at RM 0; L: best RM"

        from mpi4py import MPI

        # The local elevations are processed in passes of el_pass, so that the
        # per-band sums [pol group, band, el, x, ra] are held for one pass at a
        # time.  The gathers of the filter response are collective, so every rank
        # makes the same number of passes (some of them empty).
        el_pass = nel_local if self.el_pass is None else max(int(self.el_pass), 1)
        npass = -(-nel_local // el_pass) if nel_local > 0 else 0
        npass = self.comm.allreduce(npass, op=MPI.MAX)
        if self.el_pass is not None and self.comm.rank == 0:
            self.log.info(
                f"Processing the elevations in {npass} passes of up to {el_pass}."
            )
        # With stokes, only I and Q have a sky term
        nsky = 2 if self.stokes else ngp
        two_pol = self.stokes or any(len(g) == 2 for g in pgroups)

        pbeam *= visible[el_slc][np.newaxis, np.newaxis]
        lmbda = scipy.constants.c / (freq * 1e6)
        u = ew[iew][np.newaxis, :] / lmbda[:, np.newaxis]  # freq, ew

        # For durations longer than one sample: the tracks, and the coupling of the
        # samples of a track by the filter, summed over the bands for each spectral
        # index [spectral index, duration, pol group, el, x, ra]
        long_w = [W for W in durations if W > 1]
        periodic = _is_full_day(ra) and bool(hvis.attrs.get("ra_periodic", True))
        if long_w:
            track = self._make_tracks(
                ha[el_slc], dec[el_slc], visible[el_slc], x, dra, wmax, freq, ew[iew]
            )

        # Outputs
        bamp = bands.amplitude[:].local_array
        bwgt = bands.weight[:].local_array
        bimag = (
            bands.amplitude_imag[:].local_array
            if "amplitude_imag" in bands.datasets
            else None
        )
        tamp = trials.amplitude[:].local_array
        twgt = trials.weight[:].local_array
        timag = (
            trials.amplitude_imag[:].local_array
            if "amplitude_imag" in trials.datasets
            else None
        )
        tsnrq = trials.datasets["snr_q"][:].local_array if self.stokes else None
        gsl = slice(0, 1) if self.stokes else slice(None)
        groups = self._group_maps_setup(hvis, band_map, x, opol[gsl], ha, dec, rho_band)
        # Each band on its own: (group, part) of each output polarisation, part 0
        # the real and 1 the imaginary numerator
        if self.stokes:
            bsrc = [(0, 0), (1, 0), (2, 0), (2, 1)]  # I, Q, U, V
            # All bands: (output, group, swap the real and imaginary parts)
            srcs = [(0, 0, False), (1, 2, True)]  # I, and V from C
            rtot = np.zeros(nband, dtype=np.float64)
        else:
            bsrc = [(gg, 0) for gg in range(ngp)]
            srcs = [(gg, gg, False) for gg in range(ngp)]
        allb = np.arange(nband)

        def valid_w(W):
            return np.ones(nra, dtype=bool) if periodic else np.arange(nra) <= nra - W

        def per_duration(ia, ee, le, gg, nm, rs, vr, ni):
            """[numerator, response, variance, imaginary numerator] per duration.

            ``ee`` is the local elevation and ``le`` its index in the pass.
            """
            out = [(nm, rs, vr, ni)]
            if not long_w:
                return out
            tr = track[:, ee : ee + 1]
            parts = [rs, vr] + [a for a in (nm, ni) if a is not None]
            sums = [
                self._track_numerator(a[np.newaxis], tr, durations) for a in parts
            ]
            for iw, W in enumerate(long_w):
                iwd = durations.index(W)
                rs_w = sums[0][iwd][0].real + cross_r[ia, iw, cmap[gg], le]
                vr_w = sums[1][iwd][0].real + 0.5 * cross_v[ia, iw, cmap[gg], le]
                rs_w = np.where(valid_w(W), rs_w, 0.0)
                rest = [s[iwd][0].real for s in sums[2:]]
                nm_w = rest.pop(0) if nm is not None else None
                ni_w = rest.pop(0) if ni is not None else None
                out.append((nm_w, rs_w, vr_w, ni_w))
            return out

        def write(op, ia, ee, res):
            for iwd, (nm, rs, vr, ni) in enumerate(res):
                amp, wgt = self._amp_weight(nm, rs, vr)
                tamp[op, iwd, ia, ee] = amp.T
                twgt[op, iwd, ia, ee] = wgt.T
                if timag is not None and ni is not None:
                    im, _ = self._amp_weight(ni, rs, vr)
                    timag[op, iwd, ia, ee] = im.T

        def combine(gg, le, sb):
            return self._combine(
                gg, le, allb, sb, lam, qvec, numer, numer_im, resp, vinc, vplus, vminus
            )

        _log_memory(self, "matched filter, before the passes")
        for ipass in range(npass):
            t_pass = time.time()
            p0 = min(ipass * el_pass, nel_local)
            p1 = min(p0 + el_pass, nel_local)
            npe = p1 - p0
            shp = (ngp, nband, npe, nx, nra)
            numer = np.zeros(shp, dtype=np.float32)
            numer_im = (
                np.zeros(shp, dtype=np.float32)
                if self.save_imag or self.stokes
                else None
            )
            resp = np.zeros(shp, dtype=np.float32)
            vinc = np.zeros(shp, dtype=np.float32)
            vplus = np.zeros((nsky, *shp[1:]), dtype=np.complex64)
            vminus = np.zeros((nsky, *shp[1:]), dtype=np.complex64) if two_pol else None
            if long_w:
                cshp = (alphas.size, len(long_w), len(pgroups), npe, nx, nra)
                cross_r = np.zeros(cshp, dtype=np.float32)
                cross_v = np.zeros(cshp, dtype=np.float32)
            if blong:
                # The coupling of each band on its own, for the band durations
                bshp = (nband, len(blong), len(pgroups), npe, nx, nra)
                bcross_r = np.zeros(bshp, dtype=np.float32)
                bcross_v = np.zeros(bshp, dtype=np.float32)

            for bb in range(nband):
                fsel = np.flatnonzero(band_index == bb)
                wb = weight_all[:, fsel]  # pol, freq, ew, ra
                if has_response:
                    # A view of the shared copy where the band's channels are contiguous
                    fs = (
                        slice(fsel[0], fsel[-1] + 1)
                        if fsel.size and np.all(np.diff(fsel) == 1)
                        else fsel
                    )
                    resp_t = np.moveaxis(resp_all[fs], 0, 1)  # pol, freq, ew, ra, lag
                    one_m_p = (1.0 - resp_t[..., lag_index[0]].real).astype(np.float32)
                else:
                    one_m_p = np.ones(wb.shape, dtype=np.float32)
                inv_w = tools.invert_no_zero(wb)[:, :, :, np.newaxis, :]
                omp = one_m_p[:, :, :, np.newaxis, :]
                # In-band scale of the thermal noise [pol, freq, ew, el, ra]
                gb = gfac[:, gband[fsel]] if gfac is not None else None

                for e0 in range(p0, p1, self.nel_chunk):
                    # esl: the local elevations, psl: their place in the pass
                    esl = slice(e0, min(e0 + self.nel_chunk, p1))
                    psl = slice(esl.start - p0, esl.stop - p0)
                    yel = el_local[esl]
                    # Coefficients for these elevations: [(pol), (freq), ew, el, ra]
                    yb = yband_local[esl]
                    kt = k2tot[bb][:, yb][np.newaxis, np.newaxis]
                    ki = k2inc[bb][:, yb][np.newaxis, np.newaxis]
                    kc = kcoh[bb][:, yb][np.newaxis]
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
                    if self.stokes:
                        ok &= np.all(ok, axis=0, keepdims=True)
                    wt = np.where(ok, tools.invert_no_zero(var_th + kt * s2), 0.0)
                    data = np.where(ok, data, 0.0)
                    sky_v = np.where(ok, sky_v, 0.0)

                    wd = (wt * data).astype(np.complex64)
                    wr = (wt * omp).astype(np.float32)
                    wv = (wt**2 * (var_th * omp + ki * s2)).astype(np.float32)
                    ws = (wt * kc * sky_v).astype(np.complex64)

                    if self.stokes:
                        qa, qb = self._stokes_band(
                            bb,
                            psl,
                            sidx,
                            template,
                            wd,
                            wr,
                            wv,
                            ws,
                            rho_pol[bb],
                            xy_phase[bb],
                            numer,
                            numer_im,
                            resp,
                            vinc,
                            vplus,
                            vminus,
                        )
                        for ci, grp in enumerate(pgroups):
                            if not long_w:
                                break
                            tc = template[grp].reshape(-1, *template.shape[3:])
                            tf = None
                            wt_g, wr_g, wv_g = wt[grp], wr[grp], wv[grp]
                            if ci == 1:
                                # Q: the template and the weights of each product
                                # scaled as in its numerator, against the signal
                                # template tc
                                tf = (template[grp] * qa[:, None, None]).reshape(
                                    tc.shape
                                )
                                fb = qb[:, None, None]
                                wt_g, wr_g, wv_g = wt_g * fb, wr_g * fb, wv_g * fb**2

                            def kg(a):
                                return a.reshape(-1, *a.shape[3:]).transpose(1, 0, 2)

                            cr, cv = self._track_cross(
                                tc if tf is None else tf,
                                kg(wt_g),
                                kg(var_th[grp]),
                                kg(wr_g),
                                kg(wv_g),
                                resp_t[grp].reshape(-1, *resp_t.shape[3:]),
                                lag_index,
                                track[:, esl],
                                long_w,
                                tc_sig=None if tf is None else tc,
                            )
                            for iw in range(len(long_w)):
                                for ia in range(alphas.size):
                                    sb2 = scale[bb, ia] ** 2
                                    cross_r[ia, iw, ci, psl] += sb2 * cr[iw]
                                    cross_v[ia, iw, ci, psl] += sb2 * cv[iw]
                            for ib, W in enumerate(blong):
                                bcross_r[bb, ib, ci, psl] = cr[long_w.index(W)]
                                bcross_v[bb, ib, ci, psl] = cv[long_w.index(W)]
                        continue

                    for gg, grp in enumerate(pgroups):
                        tc = template[grp].reshape(-1, *template.shape[3:])  # k, el, x
                        tch = tc.transpose(1, 2, 0).conj()  # el, x, k
                        g2 = np.abs(tch) ** 2

                        def kmat(a, g=grp):
                            return (
                                a[g].reshape(-1, *a.shape[3:]).transpose(1, 0, 2)
                            )  # el, k, ra

                        n1 = np.matmul(tch, kmat(wd))  # el, x, ra
                        numer[gg, bb, psl] = n1.real
                        if numer_im is not None:
                            numer_im[gg, bb, psl] = n1.imag
                        resp[gg, bb, psl] = np.matmul(g2, kmat(wr))
                        vinc[gg, bb, psl] = np.matmul(g2, kmat(wv))

                        if long_w:
                            cr, cv = self._track_cross(
                                tc,
                                kmat(wt),
                                kmat(var_th),
                                kmat(wr),
                                kmat(wv),
                                resp_t[grp].reshape(-1, *resp_t.shape[3:]),
                                lag_index,
                                track[:, esl],
                                long_w,
                            )
                            for iw in range(len(long_w)):
                                for ia in range(alphas.size):
                                    sb2 = scale[bb, ia] ** 2
                                    cross_r[ia, iw, gg, psl] += sb2 * cr[iw]
                                    cross_v[ia, iw, gg, psl] += sb2 * cv[iw]
                            for ib, W in enumerate(blong):
                                bcross_r[bb, ib, gg, psl] = cr[long_w.index(W)]
                                bcross_v[bb, ib, gg, psl] = cv[long_w.index(W)]

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
                            vplus[gg, bb, psl] = (vp[0] + vp[1]) * np.sqrt(0.5 * (1 + r))
                            vminus[gg, bb, psl] = (vp[0] - vp[1]) * np.sqrt(
                                0.5 * (1 - r)
                            )
                        else:
                            vplus[gg, bb, psl] = vp[0]
            t_bands = time.time() - t_pass

            # Each band on its own
            for op, (gg, part) in enumerate(bsrc):
                for bb in range(nband):
                    var = 0.5 * vinc[gg, bb]
                    for vv in (vplus, vminus):
                        if vv is not None and gg < vv.shape[0]:
                            var = var + 0.5 * np.abs(vv[gg, bb]) ** 2
                    nm = numer[gg, bb] if part == 0 else numer_im[gg, bb]
                    amp, wgt = self._amp_weight(nm, resp[gg, bb], var)
                    bamp[op, 0, bb, p0:p1] = amp.transpose(0, 2, 1)
                    bwgt[op, 0, bb, p0:p1] = wgt.transpose(0, 2, 1)
                    if bimag is not None:
                        im, _ = self._amp_weight(numer_im[gg, bb], resp[gg, bb], var)
                        bimag[op, 0, bb, p0:p1] = im.transpose(0, 2, 1)
                    if not blong:
                        continue
                    # Longer durations: the sums along the track, with the band's own
                    # coupling of the samples (that of its pair of products, cmap)
                    tr = track[:, p0:p1]
                    parts = [resp[gg, bb], var, nm]
                    if bimag is not None:
                        parts.append(numer_im[gg, bb])
                    sums = [self._track_numerator(a, tr, blong) for a in parts]
                    for ib, W in enumerate(blong):
                        rs_w = sums[0][ib].real + bcross_r[bb, ib, cmap[gg]]
                        vr_w = sums[1][ib].real + 0.5 * bcross_v[bb, ib, cmap[gg]]
                        rs_w = np.where(valid_w(W), rs_w, 0.0)
                        iwb = band_durations.index(W)
                        amp, wgt = self._amp_weight(sums[2][ib].real, rs_w, vr_w)
                        bamp[op, iwb, bb, p0:p1] = amp.transpose(0, 2, 1)
                        bwgt[op, iwb, bb, p0:p1] = wgt.transpose(0, 2, 1)
                        if bimag is not None:
                            im, _ = self._amp_weight(sums[3][ib].real, rs_w, vr_w)
                            bimag[op, iwb, bb, p0:p1] = im.transpose(0, 2, 1)

            # All bands, for each spectral index
            for ia in range(alphas.size):
                sb = scale[:, ia]
                for le in range(npe):
                    ee = p0 + le
                    for op, gg, swap in srcs:
                        nm, rs, vr, ni = combine(gg, le, sb)
                        if swap:
                            nm, ni = ni, nm
                        write(op, ia, ee, per_duration(ia, ee, le, gg, nm, rs, vr, ni))
                    if not self.stokes:
                        continue
                    if ia not in rm_alpha:
                        tamp[2, :, ia, ee] = 0.0
                        twgt[2, :, ia, ee] = 0.0
                        timag[2, :, ia, ee] = 0.0
                        continue
                    # Linear polarisation: response and thermal variance of Q and U;
                    # the sky term of Q depends on the RM and is added in _rm_search
                    _, rsq, vrq, _ = self._combine(
                        1, le, allb, sb, lam, qvec, numer, None, resp, vinc, None, None
                    )
                    _, rsc, vrc, _ = combine(2, le, sb)
                    rv = [
                        (0.5 * (q[1] + c[1]), 0.5 * (q[2] + c[2]))
                        for q, c in zip(
                            per_duration(ia, ee, le, 1, None, rsq, vrq, None),
                            per_duration(ia, ee, le, 2, None, rsc, vrc, None),
                        )
                    ]
                    # The largest chi-squared over RM and |P|; converted to a
                    # Gaussian S/N after the last pass, when the weights of the
                    # bands (summed over all elevations) are known
                    tmax, pol_amp, brm = self._rm_search(
                        numer[1, :, le],
                        numer[2, :, le],
                        sb[:, np.newaxis] * rm_coef,
                        rm,
                        None,
                        rv,
                        track[:, ee] if long_w else None,
                        durations,
                        [valid_w(W) for W in durations],
                        self._rm_sky(le, sb, lam, qvec, vplus, vminus),
                    )
                    tamp[2, :, ia, ee] = tmax.transpose(0, 2, 1)
                    twgt[2, :, ia, ee] = pol_amp.transpose(0, 2, 1)
                    timag[2, :, ia, ee] = brm.transpose(0, 2, 1)
                    # Q at RM 0 with its sky term (U at RM 0 is the imaginary part
                    # of V), whose measured spread calibrates the noise of L
                    nmq, rsq0, vrq0, _ = combine(1, le, sb)
                    for iwd, (nm, rs, vr, _) in enumerate(
                        per_duration(ia, ee, le, 1, nmq, rsq0, vrq0, None)
                    ):
                        ok = (rs > 0) & (vr > 0)
                        snr = nm * tools.invert_no_zero(np.sqrt(np.where(ok, vr, 0.0)))
                        tsnrq[iwd, ia, ee] = np.where(ok, snr, 0.0).T
            if self.stokes:
                rtot += np.sum(resp[1], axis=(1, 2, 3), dtype=np.float64)

            self._group_maps_fill(
                groups,
                numer[gsl],
                resp[gsl],
                vinc[gsl],
                vplus[gsl],
                vminus[gsl] if vminus is not None else None,
                p0,
            )
            _log_memory(
                self,
                f"matched filter, end of pass {ipass + 1}/{npass} ({t_bands:.0f} s in "
                f"the band loop, {time.time() - t_pass:.0f} s in all)",
            )
            # Free the arrays of the pass (the closures above refer to them by name)
            numer = numer_im = resp = vinc = vplus = vminus = None
            cross_r = cross_v = bcross_r = bcross_v = None

        # No view of the shared memory may outlive it (the per-band arrays are copies)
        del weight_all
        if has_response:
            del resp_all
        for sh in shared:
            sh.free()

        if self.stokes:
            # Linear polarisation as a Gaussian S/N, with the number of independent
            # RM trials from the weights of the bands over all elevations (kept
            # for the noise calibration, which goes back to chi-squared)
            coef = []
            for ia in rm_alpha:
                coef.append(
                    self._rm_ntrials_coef(lam2_b, scale[:, ia] ** 2 * rtot, hvis.comm)
                )
                ntrials = _rm_ntrials_fn(*coef[-1])
                self.log.info(
                    f"Spectral index {alphas[ia]:+.1f}: {ntrials(25.0):.0f} "
                    "independent RM trials at chi2 = 25."
                )
                amp, wgt = self._rm_gaussian(tamp[2, :, ia], twgt[2, :, ia], ntrials)
                tamp[2, :, ia] = amp
                twgt[2, :, ia] = wgt
            # [a, c] of n(t) = a + c sqrt(t) at each of rm_spectral_indices
            trials.attrs["rm_ntrials_coef"] = np.array(coef, dtype=np.float64).reshape(
                -1, 2
            )

        group_maps = groups[0] if groups is not None else []
        if group_maps:
            self.log.info(f"Made maps for {len(groups[2])} groups of bands.")
        return (trials, bands, *group_maps)

    def _stokes_index(self):
        """Index in ``pol`` of each of the four products, for ``stokes``."""
        need = ["XX", "XY", "YX", "YY"]
        if sorted(self.pol) != need:
            raise ValueError(f"stokes needs the polarisations {need}, not {self.pol}.")
        return {p: self.pol.index(p) for p in need}

    def _rm_grid(self, lam2):
        """RM trials for bands at ``lam2``."""
        span = float(np.ptp(lam2))
        if span <= 0:
            raise ValueError("The RM search needs more than one band.")
        # Half width at half maximum of the RM spread function, sqrt(3) / span
        step = np.sqrt(3.0) / span / self.rm_oversample
        nhalf = int(np.ceil(self.rm_max / step))
        return step * np.arange(-nhalf, nhalf + 1)

    def _rm_ntrials(self, lam2, wb, comm):
        """Number of independent RM trials as a function of the level (see
        ``rm_ntrials_eff``), for band weights ``wb`` summed over the ranks."""
        return _rm_ntrials_fn(*self._rm_ntrials_coef(lam2, wb, comm))

    def _rm_ntrials_coef(self, lam2, wb, comm):
        """``(a, c)`` with ``n(t) = a + c sqrt(t)`` (see :meth:`_rm_ntrials`)."""
        if self.rm_ntrials_eff is not None:
            return float(self.rm_ntrials_eff), 0.0
        if comm is not None and comm.size > 1:
            wb = comm.allreduce(wb)
        wb = np.where(wb > 0, wb, 0.0)
        if wb.sum() <= 0:
            return 1.0, 0.0
        mean = np.sum(wb * lam2) / wb.sum()
        var = np.sum(wb * (lam2 - mean) ** 2) / wb.sum()
        c = 2.0 * self.rm_max * np.sqrt(4.0 * var / (2.0 * np.pi))
        return 1.0, float(c)

    @staticmethod
    def _stokes_band(
        bb,
        esl,
        sidx,
        template,
        wd,
        wr,
        wv,
        ws,
        rho,
        phi,
        numer,
        numer_im,
        resp,
        vinc,
        vplus,
        vminus,
    ):
        """Sums of one band for the groups I, Q and C (see the class docstring).

        Returns
        -------
        qa : np.ndarray[2, el, x]
            The factors of the templates of XX and YY in Q.
        qb : np.ndarray[2, el, ra]
            The factors of the weights of XX and YY in Q.
        """
        single = {}
        for name, pp in sidx.items():
            tch = template[pp].reshape(-1, *template.shape[3:]).transpose(1, 2, 0).conj()
            g2 = np.abs(tch) ** 2

            def kmat(a, p=pp):
                return a[p].reshape(-1, *a.shape[3:]).transpose(1, 0, 2)  # el, k, ra

            vp = np.matmul(tch, kmat(ws)) if name in ("XX", "YY") else None
            single[name] = (
                np.matmul(tch, kmat(wd)),
                np.matmul(g2, kmat(wr)),
                np.matmul(g2, kmat(wv)),
                vp,
            )

        nx_, rx, vx, px = single["XX"]
        ny_, ry, vy, py = single["YY"]
        a = np.sqrt(0.5 * (1.0 + rho))
        b = np.sqrt(0.5 * (1.0 - rho))
        r = rx + ry

        n = nx_ + ny_
        numer[0, bb, esl] = n.real
        numer_im[0, bb, esl] = n.imag
        resp[0, bb, esl] = r
        vinc[0, bb, esl] = vx + vy
        vplus[0, bb, esl] = (px + py) * a
        vminus[0, bb, esl] = (px - py) * b

        qa, qb = _q_factors(rx, ry)
        fx = qa[0][..., np.newaxis] * qb[0][:, np.newaxis]
        fy = qa[1][..., np.newaxis] * qb[1][:, np.newaxis]
        n = fx * nx_ - fy * ny_
        numer[1, bb, esl] = n.real
        numer_im[1, bb, esl] = n.imag
        resp[1, bb, esl] = fx * rx + fy * ry
        vinc[1, bb, esl] = fx**2 * vx + fy**2 * vy
        vplus[1, bb, esl] = (fx * px - fy * py) * a
        vminus[1, bb, esl] = (fx * px + fy * py) * b

        nxy, rxy, vxy, _ = single["XY"]
        nyx, ryx, vyx, _ = single["YX"]
        n = np.exp(-1.0j * phi) * (nxy + nyx.conj())
        numer[2, bb, esl] = n.real
        numer_im[2, bb, esl] = n.imag
        resp[2, bb, esl] = rxy + ryx
        vinc[2, bb, esl] = vxy + vyx
        return qa, qb

    def _rm_search(
        self, nq, nu, coef, rm, ntrials, rv, track, durations, valid, sky=None
    ):
        """Search in linear polarisation over RM at one elevation.

        Parameters
        ----------
        nq, nu : np.ndarray[band, x, ra]
            Numerators of Q and U for each band (single samples).
        coef : np.ndarray[band, rm]
            ``s_b e^{-2 i RM l_b^2}``.
        rm : np.ndarray[rm]
            The RM trials.
        ntrials : callable or None
            Number of independent RM trials at a level of chi-squared.  If None,
            the largest chi-squared and |P| are returned instead of the amplitude and
            weight (see :meth:`_rm_gaussian`).
        rv : list of (np.ndarray[x, ra], np.ndarray[x, ra])
            Response and thermal variance of the real part for each duration.
        track : _Track or None
            From :meth:`_tracks` at this elevation, ``[wmax, x, tap]`` (None for
            single samples only).
        durations : list of int
            The durations, starting with 1.
        valid : list of np.ndarray[ra] of bool
            Start samples with a complete trial, for each duration.
        sky : tuple, optional
            Sky term of Q, from :meth:`_rm_sky`; it is independent between samples,
            so its variance is summed along the track.

        Returns
        -------
        amp, wgt, best_rm : np.ndarray[duration, x, ra]
            Or the largest chi-squared (-1 where there is no trial) and |P| in place
            of amp and wgt, if ``ntrials`` is None.
        """
        nb, nx, nra = nq.shape
        ndur = len(durations)
        wmax = max(durations)
        z = (nq + 1.0j * nu).astype(np.complex64).reshape(nb, -1)
        best_t = np.full((ndur, nx, nra), -1.0, dtype=np.float32)
        best_s = np.zeros((ndur, nx, nra), dtype=np.float32)
        best_k = np.zeros((ndur, nx, nra), dtype=np.int32)
        var_th = [v.astype(np.float32) for _, v in rv]

        def update(iwd, s, vs, k0):
            v = var_th[iwd] if vs is None else var_th[iwd] + vs
            t = (s.real**2 + s.imag**2) * tools.invert_no_zero(v)  # rm, x, ra
            k = np.argmax(t, axis=0)
            tk = np.take_along_axis(t, k[np.newaxis], 0)[0]
            better = tk > best_t[iwd]
            best_t[iwd] = np.where(better, tk, best_t[iwd])
            sk = np.abs(np.take_along_axis(s, k[np.newaxis], 0)[0])
            best_s[iwd] = np.where(better, sk, best_s[iwd])
            best_k[iwd] = np.where(better, k + k0, best_k[iwd])

        for k0 in range(0, rm.size, self.rm_chunk):
            ks = slice(k0, min(k0 + self.rm_chunk, rm.size))
            s1 = (coef[:, ks].T.astype(np.complex64) @ z).reshape(-1, nx, nra)
            v1 = None if sky is None else self._rm_sky_var(coef[:, ks], sky, nx, nra)
            update(0, s1, v1, k0)
            if track is None:
                continue
            acc = np.zeros_like(s1)
            accv = None if v1 is None else np.zeros_like(v1)
            for jj in range(wmax):
                sj = np.zeros_like(s1)
                vj = None if accv is None else np.zeros_like(v1)
                for ii, wt in track.taps(jj):
                    w3 = wt[np.newaxis, :, np.newaxis]
                    sj += w3 * s1[:, ii]
                    if vj is not None:
                        vj += w3 * v1[:, ii]
                acc += np.roll(sj, -jj, axis=-1)
                if accv is not None:
                    accv += np.roll(vj, -jj, axis=-1)
                if jj > 0 and jj + 1 in durations:
                    update(durations.index(jj + 1), acc, accv, k0)

        tmax = np.full((ndur, nx, nra), -1.0, dtype=np.float32)
        pol_amp = np.zeros((ndur, nx, nra), dtype=np.float32)
        for iwd in range(ndur):
            rs, vr = rv[iwd]
            ok = (rs > 0) & (vr > 0) & valid[iwd][np.newaxis] & (best_t[iwd] >= 0)
            tmax[iwd] = np.where(ok, best_t[iwd], -1.0)
            pol_amp[iwd] = np.where(ok, best_s[iwd] * tools.invert_no_zero(rs), 0.0)
        if ntrials is None:
            return tmax, pol_amp, rm[best_k].astype(np.float32)
        amp, wgt = self._rm_gaussian(tmax, pol_amp, ntrials)
        return amp, wgt, rm[best_k].astype(np.float32)

    @staticmethod
    def _rm_gaussian(tmax, pol_amp, ntrials):
        """Amplitude ``sign(z) |P|`` and weight ``(z / |P|)^2`` of the RM search.

        ``z`` is the Gaussian S/N of the largest chi-squared ``tmax`` (-1 where
        there is no trial) over ``ntrials`` independent RM trials.  Where ``|P|`` is
        zero the weight is zero (no data), not infinite.
        """
        ok = (tmax >= 0) & (pol_amp > 0)
        tt = np.where(ok, tmax, 0.0)
        zz = _chi2max_to_gaussian(tt, ntrials(tt))
        amp = np.where(ok, np.sign(zz) * pol_amp, 0.0).astype(np.float32)
        wgt = (zz * tools.invert_no_zero(np.where(ok, pol_amp, 0.0))) ** 2
        wgt = wgt.astype(np.float32)
        return amp, wgt

    def _rm_sky(self, ee, sb, lam, qvec, vplus, vminus):
        """Sky term of Q for the RM search at one elevation.

        Returns None if Q has no sky term here, else (data, wmode, rest): the real
        and imaginary parts of the sky term of Q of each band (for each of vplus and
        vminus) [band, x * ra], the weights ``sqrt(lam_m) q_bm`` of the leading
        ``rm_sky_modes`` modes of the correlation between bands [band, mode], and
        the variance of the other modes at its mean over RM [x, ra].
        """
        parts = [
            vv[1, :, ee]
            for vv in (vplus, vminus)
            if vv is not None and np.any(vv[1, :, ee])
        ]
        if not parts or lam.size == 0:
            return None
        nb, nx, nra = parts[0].shape
        order = np.argsort(lam)[::-1]
        keep, drop = order[: self.rm_sky_modes], order[self.rm_sky_modes :]
        wmode = (qvec[:, keep] * np.sqrt(lam[keep])).astype(np.float32)
        dband = np.sum(qvec[:, drop] ** 2 * lam[drop], axis=1)  # band
        data = []
        rest = np.zeros((nx, nra), dtype=np.float64)
        for v in parts:
            for a in (v.real, v.imag):
                data.append(np.ascontiguousarray(a.reshape(nb, -1), dtype=np.float32))
            rest += np.tensordot(sb**2 * dband, np.abs(v) ** 2, axes=1)
        return data, wmode, (0.25 * rest).astype(np.float32)

    @staticmethod
    def _rm_sky_var(coef, sky, nx, nra):
        """Variance of the sky term of L at the RM trials ``coef`` [band, rm].

        For single samples [rm, x, ra], in the units of the variance of L (half the
        expected ``|S|^2``): ``(1/4) sum_m (|sum_b c_b w_bm A_b|^2 + |sum_b c_b w_bm
        B_b|^2)`` with ``V_b = A_b + i B_b``, plus the other modes.
        """
        data, wmode, rest = sky
        nb, nr = coef.shape
        cm = (coef[:, np.newaxis, :] * wmode[:, :, np.newaxis]).reshape(nb, -1)
        cr = np.ascontiguousarray(
            np.concatenate([cm.real, cm.imag], axis=1).T, dtype=np.float32
        )  # (re/im, mode, rm), band
        out = np.zeros((nr, nx * nra), dtype=np.float32)
        for d in data:
            y = cr @ d
            np.square(y, out=y)
            out += y.reshape(-1, nr, nx * nra).sum(axis=0)
        return 0.25 * out.reshape(nr, nx, nra) + rest[np.newaxis]

    @staticmethod
    def _track_cross(
        tc, wt, var_th, rdiag, vdiag, presp, lag_index, track, long_w, tc_sig=None
    ):
        """Coupling by the filter of the different samples along the tracks.

        For each duration W in ``long_w``, the terms ``j != j'`` (both < W) of the
        response and of the thermal variance of the numerator (see the class
        docstring), for a transient starting at each sample.

        Each also holds a correction to the terms ``j = j'``.  Those are the
        single-sample response and variance interpolated along the track, which for
        a sum ``sum_k |g_k|^2 a_k`` gives ``|g|^2`` interpolated rather than the
        ``|g~|^2`` of the interpolated template ``g~`` that the numerator and the
        terms ``j != j'`` use.  For a long pulse the two parts nearly cancel, so the
        difference is added here.

        The pair ``(j, j + d)`` and its mirror contribute ``Re[c X_d(t + j)]`` with
        ``c = g_j^* g_{j+d}`` and ``X_d`` depending only on the lag ``d``, so ``X_d``
        is formed once per lag, the shift by ``j`` is applied to the product (it
        commutes with the sum over visibilities), and only the real part of the
        product is computed (one real matrix product over ``(Re, Im)`` stacked
        along the visibilities, half the work of a complex one).

        Parameters
        ----------
        tc : np.ndarray[k, el, x]
            Template at each grid x.
        wt, var_th : np.ndarray[el, k, ra]
            Total weight and thermal variance of each visibility.
        rdiag, vdiag : np.ndarray[el, k, ra]
            The single-sample response and variance of each visibility, without the
            template (``|g_k|^2`` times them gives the single-sample terms).
        presp : np.ndarray[k, ra, lag]
            Filter response ``P[t, t + d]``.
        lag_index : dict
            Index of each lag ``d`` in ``presp``.
        track : _Track
            From :meth:`TransientMatchedFilter._tracks`, ``[wmax, el, x, tap]``.
        long_w : list of int
            Durations, all longer than one sample, in increasing order.
        tc_sig : np.ndarray[k, el, x], optional
            Template of the signal, if it is not ``tc`` (the template of the
            numerator).  The response then has ``conj(g_kj) s_kj'`` with ``s`` the
            signal template along the track, and the pair and its mirror need
            separate products.

        Returns
        -------
        resp_cross, var_cross : list of np.ndarray[el, x, ra]
            For each duration in ``long_w``.
        """
        wmax = max(long_w)
        _, nel, nx = tc.shape
        nra = wt.shape[-1]

        # Template along the track at each step [j][el, x, k]
        g = [track.template(tc, jj).transpose(1, 2, 0) for jj in range(wmax)]
        gs = (
            g
            if tc_sig is None
            else [track.template(tc_sig, jj).transpose(1, 2, 0) for jj in range(wmax)]
        )
        wv = wt * var_th

        def stack(x):
            # [el, k, ra] complex -> [el, 2k, ra] real (Re over Im)
            return np.concatenate([x.real, x.imag], axis=1).astype(np.float32)

        def shift_add(acc, m, j):
            # acc += roll(m, -j) along RA
            if j == 0:
                acc += m
            else:
                acc[..., : nra - j] += m[..., j:]
                acc[..., nra - j :] += m[..., :j]

        # New pairs at each level jn = j + d (the pairs of duration jn + 1 that
        # duration jn does not have)
        new_r = np.zeros((wmax, nel, nx, nra), dtype=np.float32)
        new_v = np.zeros((wmax, nel, nx, nra), dtype=np.float32)
        for d in range(1, wmax):
            # -W(t) P[t, t + d] and its mirror -W(t + d) P[t + d, t], with the
            # thermal variance at the other end; Re[c a + c^* b] = Re[c (a + b^*)]
            a_d = -wt * presp[np.newaxis, :, :, lag_index[d]]
            b_d = np.roll(-wt * presp[np.newaxis, :, :, lag_index[-d]], -d, axis=-1)
            x_v = stack(a_d * np.roll(wv, -d, axis=-1) + b_d.conj() * wv)
            if tc_sig is None:
                x_r = stack(a_d + b_d.conj())
            else:
                # Re[conj(g_j) s_j+d a] + Re[conj(g_j+d) s_j b]
                x_ra, x_rb = stack(a_d), stack(b_d.conj())
            for j in range(wmax - d):
                c = g[j].conj() * g[j + d]  # el, x, k
                cs = np.concatenate([c.real, -c.imag], axis=-1).astype(np.float32)
                shift_add(new_v[j + d], np.matmul(cs, x_v), j)
                if tc_sig is None:
                    shift_add(new_r[j + d], np.matmul(cs, x_r), j)
                    continue
                for c, xr in (
                    (g[j].conj() * gs[j + d], x_ra),
                    (gs[j].conj() * g[j + d], x_rb),
                ):
                    cs = np.concatenate([c.real, -c.imag], axis=-1).astype(np.float32)
                    shift_add(new_r[j + d], np.matmul(cs, xr), j)

        # The terms j = j' with the interpolated template, less the interpolated
        # single-sample terms (none at the start, which is on the grid)
        rd = rdiag.astype(np.float32)
        vd = vdiag.astype(np.float32)
        for j in range(1, wmax):
            e = np.abs(g[j]) ** 2 - track.power(tc, j).transpose(1, 2, 0)
            if tc_sig is None:
                er = e
            else:
                er = (g[j].conj() * gs[j]).real - track.power(tc, j, tc_sig).transpose(
                    1, 2, 0
                )
            if np.any(er):
                shift_add(new_r[j], np.matmul(er.astype(np.float32), rd), j)
            if np.any(e):
                shift_add(new_v[j], np.matmul(e.astype(np.float32), vd), j)

        resp_cross, var_cross = [], []
        acc_r = np.zeros((nel, nx, nra), dtype=np.float64)
        acc_v = np.zeros((nel, nx, nra), dtype=np.float64)
        for W in range(2, wmax + 1):
            acc_r += new_r[W - 1]
            acc_v += new_v[W - 1]
            if W in long_w:
                resp_cross.append(acc_r.astype(np.float32))
                var_cross.append(acc_v.astype(np.float32))

        return resp_cross, var_cross

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
            if vv is None or gg >= vv.shape[0]:
                continue
            y = np.tensordot(coeff.T.astype(np.complex64), vv[gg, idx, ee], axes=1)
            var = var + np.sum(np.abs(y) ** 2, axis=0)
        return nm, rs, 0.5 * var, ni

    def _group_maps_setup(self, hvis, band_map, x, opol, ha, dec, rho_band):
        """Containers of the group maps, one for each entry of group_x_reduce.

        Returns None without groups, else (outputs, telescope-x index of each
        output (None for the best), group names, and for each group the bands, the
        modes of their correlation and the band weights).
        """
        if not self.groups:
            return None
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
        modes = []
        for inc in used:
            idx = np.flatnonzero(inc)
            lam, qvec = self._modes(rho_band[np.ix_(idx, idx)])
            sg = np.zeros(centre.size)
            sg[idx] = sb[idx]
            modes.append((idx, lam, qvec, sg))
        return outs, fixed, names, modes

    def _group_maps_fill(self, groups, numer, resp, vinc, vplus, vminus, e0):
        """S/N of the group maps for the elevations of one pass (from local ``e0``)."""
        if groups is None:
            return
        outs, fixed, _, modes = groups
        ngp, _, npe = numer.shape[:3]
        for ig, (idx, lam, qvec, sg) in enumerate(modes):
            for gg in range(ngp):
                for le in range(npe):
                    ee = e0 + le
                    nm, rs, vr, _ = self._combine(
                        gg, le, idx, sg, lam, qvec, numer, None, resp, vinc, vplus, vminus
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



class CandidateBandSpectra(TransientMatchedFilterSkyNoise):
    """The per-band matched filter at each candidate's own duration.

    The single-sample band output summed along a candidate's track is biased for a
    pulse longer than a sample: the m-mode filter removes much of a long pulse (it
    drifts at the fringe rate, in the middle of the stop band), the matched filter
    of the trials accounts for that through the coupling of the samples of the
    track, and the plain sum does not.  The stop band scales with frequency, so the
    loss is larger at the high frequencies and the spectral index of a long pulse
    comes out too steep (by ~-1.5 to -2 for 4 to 16 samples on the injections of
    2418).

    This task runs the matched filter of :class:`TransientMatchedFilterSkyNoise`
    (same configuration) again on a window of one row and W samples at each
    candidate with a duration W > 1 and S/N of at least ``snr_min``, with the
    candidate's duration among the band durations, so that each band has its own
    numerator, response, variance and coupling along the track starting at the
    candidate's telescope-x.  The window has the data, weights, filter response,
    noise scale, RFI mask and in-band noise of the full day at those samples; the
    telescope-x grid, the primary beam and the sky-noise coefficients are those of
    the full day.  Each rank does the candidates on its own rows, so the cost is
    that of the matched filter on ~(number of candidates x W) samples of one row.
    Candidates whose window runs off the end of a day that is not periodic are left
    out (their spectra stay the sum of single samples).

    The result has the amplitude and weight of each band and polarisation of the
    bands (I, Q, U and V for ``stokes``) at the candidate's start, duration and
    telescope-x, for :class:`AddCandidateSpectra`.

    Attributes
    ----------
    snr_min : float
        Smallest S/N of the candidates.  Default is 10.
    """

    snr_min = config.Property(proptype=float, default=10.0)

    # The window of the current candidate (in the helper that runs the matched filter)
    _win = None

    def _x_grid(self, freq, ew_used):
        if self._win is not None:
            return self._win["x"]
        return super()._x_grid(freq, ew_used)

    def _evaluate_beam(self, freq, pol, y, x, pad=8):
        if self._win is not None:
            return self._win["pbeam"]
        return super()._evaluate_beam(freq, pol, y, x, pad)

    def _sky_coefficients(self, sky, band_map, ew_used, ra, el):
        if self._win is not None:
            k2coh, k2tot, yband, rho_pol, rho = self._win["sky"]
            ras, ee = self._win["ras"], self._win["el"]
            return k2coh[..., ras], k2tot[..., ras], yband[ee], rho_pol, rho
        return super()._sky_coefficients(sky, band_map, ew_used, ra, el)

    def process(self, hvis, unfiltered, sky, table):
        """Run the matched filter at the candidates.

        Parameters
        ----------
        hvis : ch_pipeline.core.containers.MFilteredHybridVisStream
            The input of the matched filter.
        unfiltered : draco.core.containers.HybridVisStream
            The visibilities before filtering, on the same grid.
        sky : ch_pipeline.core.containers.HybridVisSkyNoise
            Output of :class:`EstimateSkyNoise`.
        table : containers.TransientCandidateTable
            Candidates, from :class:`GroupTransientCandidates`.

        Returns
        -------
        out : containers.TransientCandidateCutouts
            Each candidate's bands at its duration (one row and start time).
        """
        import copy
        import logging

        from mpi4py import MPI

        comm = self.comm
        rec = table.candidate[:]
        el = hvis.index_map["el"][:]
        ra = hvis.ra[:]
        nra = ra.size
        dra = np.median(np.diff(ra))
        periodic = _is_full_day(ra) and bool(hvis.attrs.get("ra_periodic", True))
        tsamp = float(table.attrs.get("duration_unit_seconds", dra * 240.0 * 0.99727))
        wc = np.maximum(np.rint(np.nan_to_num(rec["duration"]) / tsamp), 1).astype(int)
        tb = np.clip(np.searchsorted(ra, rec["ra"] - 0.5 * dra), 0, nra - 1)
        sel = (rec["snr"] >= self.snr_min) & (wc > 1) & np.isin(wc, self.durations)
        if not periodic:
            sel &= tb + wc <= nra
        idx = np.flatnonzero(sel)
        ncut = idx.size

        # The full-day quantities the windows share: the telescope-x grid, the beam
        # (on the rows of each rank) and the sky-noise coefficients
        pol = list(hvis.index_map["pol"])
        ipol = [pol.index(p) for p in self.pol]
        ew = hvis.index_map["ew"][:]
        iew = list(range(ew.size)) if self.ew_index is None else list(self.ew_index)
        freq = hvis.freq[:]
        _, band_map = self._assign_bands(freq)
        x = self._x_grid(freq, ew[iew])
        pbeam = self._evaluate_beam(freq, [pol[ii] for ii in ipol], el, x)
        sky_coef = self._sky_coefficients(sky, band_map, ew[iew], ra, el)

        hvis.redistribute(["el", "freq"])
        unfiltered.redistribute(["el", "freq"])
        el_slc = hvis.vis[:].local_bounds
        if unfiltered.vis[:].local_bounds != el_slc:
            raise RuntimeError("The streams are not distributed in the same way.")
        has_response = "ra_response" in hvis.datasets
        wname = "noise_weight" if has_response else hvis._weight_dset_name
        shared = [_NodeSharedFreq(hvis[wname][:], comm)]  # freq, pol, ew, ra
        if has_response:
            shared.append(_NodeSharedFreq(hvis.ra_response[:], comm))
        vis = hvis.vis[:].local_array
        uvis = unfiltered.vis[:].local_array
        nscale = (
            hvis.noise_scale[:].local_array if "noise_scale" in hvis.datasets else None
        )
        mask = self.rfi_mask
        if mask is not None:
            mask.redistribute("el")
        noise = self.inband_noise
        if noise is not None:
            noise.redistribute("el")

        bpol = ["I", "Q", "U", "V"] if self.stokes else (
            ["co"] if self.combine_pol else list(self.pol)
        )
        nband = len(band_map)
        amp = np.zeros((ncut, len(bpol), nband), dtype=np.float64)
        wgt = np.zeros_like(amp)

        helper = copy.copy(self)
        helper.comm = MPI.COMM_SELF
        helper.groups = {}
        helper.el_pass = None
        helper.spectral_indices = [0.0]
        # Its log is quiet (it would repeat the matched filter's messages per window)
        quiet = logging.getLogger(f"{__name__}.CandidateBandSpectra.window")
        quiet.setLevel(logging.WARNING)
        helper._log = quiet
        self_comm = MPI.COMM_SELF

        def window(cont, name, sl):
            return cont.datasets[name][:].local_array[sl]

        t0 = time.time()
        ndone = 0
        for ii, k in enumerate(idx):
            e = int(rec["el_index"][k])
            if not (el_slc.start <= e < el_slc.stop):
                continue
            le = e - el_slc.start
            W = int(wc[k])
            ras = (tb[k] + np.arange(W)) % nra

            sub = type(hvis)(
                axes_from=hvis, attrs_from=hvis, el=el[e : e + 1], ra=ra[ras],
                comm=self_comm, distributed=True,
            )
            sub.attrs["ra_periodic"] = False
            sub.vis[:].local_array[:] = vis[:, :, :, le : le + 1][..., ras]
            sub.add_dataset(wname)
            sub.datasets[wname][:].local_array[:] = np.moveaxis(
                shared[0].array[:, :, :, ras], 0, 1
            )
            if has_response:
                sub.add_dataset("ra_response")
                sub.ra_response[:].local_array[:] = np.moveaxis(
                    shared[1].array[:, :, :, ras], 0, 1
                )
            if nscale is not None:
                sub.add_dataset("noise_scale")
                sub.noise_scale[:].local_array[:] = nscale[..., le : le + 1]
            usub = type(unfiltered)(
                axes_from=unfiltered, attrs_from=unfiltered, el=el[e : e + 1],
                ra=ra[ras], comm=self_comm, distributed=True,
            )
            usub.vis[:].local_array[:] = uvis[:, :, :, le : le + 1][..., ras]

            helper.rfi_mask = helper.inband_noise = None
            if mask is not None:
                msub = type(mask)(
                    axes_from=mask, attrs_from=mask, el=el[e : e + 1], ra=ra[ras],
                    comm=self_comm, distributed=True,
                )
                msub.mask[:].local_array[:] = window(mask, "mask", (slice(None), ras))[
                    ..., le : le + 1
                ]
                helper.rfi_mask = msub
            if noise is not None:
                nsub = type(noise)(
                    axes_from=noise, attrs_from=noise, el=el[e : e + 1], ra=ra[ras],
                    comm=self_comm, distributed=True,
                )
                nsub.scale[:].local_array[:] = noise.scale[:].local_array[
                    :, :, :, le : le + 1
                ][..., ras]
                for name in ("dof", "common", "flag"):
                    if name in noise.datasets and name in nsub.datasets:
                        nsub.datasets[name][:] = noise.datasets[name][:][..., ras]
                helper.inband_noise = nsub

            helper.durations = [1, W]
            helper.band_durations = [1, W]
            helper._win = {
                "x": x,
                "pbeam": pbeam[:, :, le : le + 1],
                "sky": sky_coef,
                "ras": ras,
                "el": [e],
            }
            res = TransientMatchedFilterSkyNoise.process(helper, sub, usub, sky)
            bands = res[1]
            xi = int(rec["x_index"][k])
            amp[ii] = bands.amplitude[:].local_array[:, 1, :, 0, 0, xi]
            wgt[ii] = bands.weight[:].local_array[:, 1, :, 0, 0, xi]
            ndone += 1

        for sh in shared:
            sh.free()
        comm.Allreduce(MPI.IN_PLACE, amp, op=MPI.SUM)
        comm.Allreduce(MPI.IN_PLACE, wgt, op=MPI.SUM)
        nmax = comm.allreduce(ndone, op=MPI.MAX)
        self.log.info(
            f"Ran the matched filter at {ncut} candidates (W > 1, S/N >= "
            f"{self.snr_min:g}), up to {nmax} on a rank, in {time.time() - t0:.0f} s."
        )

        out = containers.TransientCandidateCutouts(
            candidate=ncut,
            pol=np.array(bpol),
            freq=band_map,
            el_offset=np.array([0]),
            ra_offset=np.array([0]),
            attrs_from=table,
            distributed=True,
            comm=comm,
        )
        out.datasets["candidate_index"][:] = idx
        out.datasets["x_index"][:] = rec["x_index"][idx]
        out.datasets["el_index"][:] = rec["el_index"][idx, np.newaxis]
        out.datasets["ra_index"][:] = tb[idx, np.newaxis]
        out.datasets["el"][:] = el[rec["el_index"][idx], np.newaxis]
        out.datasets["ra"][:] = ra[tb[idx], np.newaxis]
        out.datasets["duration"][:] = wc[idx]
        la = out.amplitude[:]
        sl = slice(la.local_offset[0], la.local_offset[0] + la.local_shape[0])
        la.local_array[:] = amp[sl, :, :, np.newaxis, np.newaxis]
        out.weight[:].local_array[:] = wgt[sl, :, :, np.newaxis, np.newaxis]
        return out

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
    pol : list of str
        Polarisations to use.  Default is all.
    nel_chunk : int
        Number of elevations to process at once.  Default is 16.
    """

    groups = config.Property(proptype=dict, default={})
    each_band = config.Property(proptype=bool, default=False)
    pol = config.list_type(str, default=None)
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

        pol = _decode_names(mf.index_map["pol"][:])
        if self.pol is None:
            use = list(range(len(pol)))
        else:
            missing = [p for p in self.pol if p not in pol]
            if missing:
                raise ValueError(f"Polarisations {missing} are not in {pol}.")
            use = [pol.index(p) for p in self.pol]

        out = containers.TransientBandGroupMap(
            group=np.array(names),
            pol=np.array([pol[pp] for pp in use]),
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

        amp = mf.amplitude[:].local_array[use, idur, ..., xsel]  # pol, band, el, ra, x
        wgt = mf.weight[:].local_array[use, idur, ..., xsel]

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

    The matched filter must save the imaginary part (``save_imag``), except for
    the Stokes output of :class:`TransientMatchedFilterSkyNoise` (``stokes``),
    whose Q, U and V are real: each then contributes ``w a^2`` with one degree of
    freedom, so ``pol: [Q, U, V]`` sums all of the polarised power, Q^2 + U^2 + V^2
    (3 dof per band), which does not depend on the X-Y phase or the RM.  The real
    and imaginary parts are taken to have the same variance, and the terms to be
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
    pol : list of str
        Polarisations to use.  Default is all.
    nel_chunk : int
        Number of elevations to process at once.  Default is 16.
    """

    groups = config.Property(proptype=dict, default={})
    each_band = config.Property(proptype=bool, default=False)
    duration = config.Property(proptype=int, default=1)
    x_reduce = config.Property(default="max")
    combine_pol = config.Property(proptype=bool, default=True)
    pol = config.list_type(str, default=None)
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
        has_imag = "amplitude_imag" in mf.datasets
        if not has_imag and "xy_delay_ns" not in mf.attrs:
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
        if self.pol is None:
            use = list(range(len(pol)))
        else:
            missing = [p for p in self.pol if p not in pol]
            if missing:
                raise ValueError(f"Polarisations {missing} are not in {pol}.")
            use = [pol.index(p) for p in self.pol]
        pol_groups = [use] if self.combine_pol else [[pp] for pp in use]
        opol = np.array(["+".join(pol[pp] for pp in grp) for grp in pol_groups])
        ndof = 2 if has_imag else 1

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
        aim = (
            mf.datasets["amplitude_imag"][:].local_array[:, idur, ..., xsel]
            if has_imag
            else None
        )
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
                            a2 = are[pp, bb, esl].astype(np.float64) ** 2
                            if aim is not None:
                                a2 = a2 + aim[pp, bb, esl].astype(np.float64) ** 2
                            q = q + w * a2
                            k = k + ndof * (w > 0)
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



_LAT_CHIME = np.radians(49.3207092194)

# Bright pulsars whose scintillation leaves a residual in the filtered maps:
# J2000 position (degrees) and catalogued RM (rad/m^2), whose sign fixes that of U
DEFAULT_POL_CALIBRATORS = [
    {"name": "B0329+54", "ra": 53.24754, "dec": 54.57870, "rm": -64.33},
]


class FitPolarisationCalibrators(tasklib.base.ContainerTask):
    """The X-Y phase of the day from the scintillation of bright pulsars.

    The m-mode filter removes a pulsar's mean flux, but its scintillation leaves a
    residual in the per-band matched filter, and scintillation scales I, Q, U and V
    together.  So along each transit (upper and lower) the samples nearest the
    pulsar's apparent position (the row and telescope-x nearest its track, for
    ``|x| < x_max``) give per band, by regression on I (corrected for the noise in
    I), its fractional ``q_b`` and ``c_b`` (C = U + iV after the X-Y phase applied
    by the matched filter, or ``(XY + conj YX) / 2`` with none applied).  Then:

    1. ``|RM|``, the linear fraction ``p`` and the angle ``chi`` from
       ``q_b = p D_b cos 2(chi + RM l_b^2)``, with ``D_b = sinc(RM dl^2_b)`` the
       depolarisation within the band, taking the sign of the catalogued RM;
    2. ``c_b e^{-2 pi i nu_b tau} = A u_b + B`` on a grid of the residual delay
       ``tau``, with ``u_b`` the fractional U from step 1 (so the sign of U is
       known): ``arg A`` is the residual offset of the X-Y phase, ``|A| ~ 1`` checks
       the amplitude, and ``B = A (i v + leakage)``.

    The totals add the residuals to the phase applied (``xy_delay_ns`` and
    ``xy_offset_deg`` of the input).  The offset needs U to change across the band:
    ``rot_deg`` is the turn of the polarisation angle across the bands used,
    ``2 |RM| (l^2_max - l^2_min)``; below ~90 deg ``A u_b`` cannot be told from
    ``B`` and only the delay is measured (B0950+08, B1133+16, B1237+25 and B1508+55,
    |RM| < 3).  A transit with fewer than ``min_bands`` bands
    whose fractional noise is below ``frac_noise_max`` is reported with ``ok``
    False.

    Attributes
    ----------
    calibrators : list of dict
        Pulsars as {name, ra, dec, rm}: J2000 position in degrees and catalogued
        RM.  Default is B0329+54.
    transits : list of str
        "upper" and/or "lower".  Default is both.
    x_max : float
        Largest |telescope-x| of the samples used.  Default is 0.02.
    frac_noise_max : float
        Largest noise of a band's fractional Q for it to be used.  Default is 0.2.
    min_bands : int
        Least number of bands for a fit.  Default is 8.
    rm_fit_max : float
        Largest |RM| of the fit in rad/m^2.  Default is 200.
    delay_max : float
        Largest |residual delay| in ns.  Default is 5.
    """

    calibrators = config.Property(proptype=list, default=DEFAULT_POL_CALIBRATORS)
    transits = config.list_type(str, default=["upper", "lower"])
    x_max = config.Property(proptype=float, default=0.02)
    frac_noise_max = config.Property(proptype=float, default=0.2)
    min_bands = config.Property(proptype=int, default=8)
    rm_fit_max = config.Property(proptype=float, default=200.0)
    delay_max = config.Property(proptype=float, default=5.0)

    def process(self, bands):
        """Fit the calibrators.

        Parameters
        ----------
        bands : containers.TransientMatchedFilter
            Per-band matched filter (single samples) with I, Q, U and V (``stokes``)
            or XX, XY, YX and YY with the imaginary part.

        Returns
        -------
        out : containers.PolarisationCalibration
        """
        from ch_ephem.observers import chime
        from mpi4py import MPI
        from skyfield.api import Star

        bands.redistribute("el")
        pol = _decode_names(bands.index_map["pol"][:])
        if {"I", "Q", "U", "V"} <= set(pol):
            mode = "stokes"
            need = ["I", "Q", "U", "V"]
            tau0 = float(bands.attrs.get("xy_delay_ns", 0.0))
            phi0 = float(bands.attrs.get("xy_offset_deg", 0.0))
        elif {"XX", "XY", "YX", "YY"} <= set(pol) and "amplitude_imag" in bands.datasets:
            mode = "products"
            need = ["XX", "XY", "YX", "YY"]
            tau0 = phi0 = 0.0
        else:
            raise ValueError(f"Need I, Q, U, V or the four products, not {pol}.")
        ip = [pol.index(p) for p in need]
        durations = list(bands.index_map["duration"][:])
        if 1 not in durations:
            raise ValueError("The per-band output must have single samples.")
        idur = durations.index(1)

        lsd = _stream_lsd(bands)
        band_map = bands.index_map["freq"][:]
        freq, width = band_map["centre"], band_map["width"]
        el = bands.index_map["el"][:]
        ra = bands.ra[:]
        xg = bands.index_map["x"][:]
        nb = freq.size

        amp = bands.amplitude[:]
        off, nloc = amp.local_offset[3], amp.local_shape[3]
        are = amp.local_array[:, idur]  # pol, band, el, ra, x
        aim = (
            bands.datasets["amplitude_imag"][:].local_array[:, idur]
            if mode == "products"
            else None
        )
        wgt = bands.weight[:].local_array[:, idur]

        rows, per_band = [], []
        for cal in self.calibrators:
            star = Star(ra_hours=float(cal["ra"]) / 15.0, dec_degrees=float(cal["dec"]))
            ra_c, dec_c = chime.object_coords(star, chime.lsd_to_unix(lsd + 0.5), deg=True)
            ra_c = float(ra_c) % 360.0
            ha = np.radians(((ra - ra_c + 180.0) % 360.0) - 180.0)
            d = np.radians(float(dec_c))
            xt = -np.cos(d) * np.sin(ha)
            yt = np.cos(_LAT_CHIME) * np.sin(d) - np.sin(_LAT_CHIME) * np.cos(d) * np.cos(ha)
            for transit in self.transits:
                upper = np.abs(ha) < 0.5 * np.pi
                sel = upper if transit == "upper" else ~upper
                sel &= (np.abs(xt) < self.x_max) & (yt >= el.min()) & (yt <= el.max())
                tsel = np.flatnonzero(sel)
                ie = np.argmin(np.abs(el[np.newaxis, :] - yt[tsel, np.newaxis]), axis=1)
                ix = np.argmin(np.abs(xg[np.newaxis, :] - xt[tsel, np.newaxis]), axis=1)

                # Values along the track [pol, band, sample], summed over the ranks
                vals = np.zeros((len(need), nb, tsel.size), dtype=np.complex128)
                ww = np.zeros((len(need), nb, tsel.size), dtype=np.float64)
                mine = (ie >= off) & (ie < off + nloc)
                if np.any(mine):
                    le, lt, lx = ie[mine] - off, tsel[mine], ix[mine]
                    for k, pp in enumerate(ip):
                        v = are[pp][:, le, lt, lx].astype(np.complex128)
                        if aim is not None:
                            v = v + 1j * aim[pp][:, le, lt, lx]
                        vals[k][:, mine] = v
                        ww[k][:, mine] = wgt[pp][:, le, lt, lx]
                if bands.comm is not None and bands.comm.size > 1:
                    bands.comm.Allreduce(MPI.IN_PLACE, vals, op=MPI.SUM)
                    bands.comm.Allreduce(MPI.IN_PLACE, ww, op=MPI.SUM)

                good = np.all(ww > 0, axis=0)

                def inv(w, g=good):
                    return tools.invert_no_zero(np.where(g, w, 0.0))

                if mode == "stokes":
                    I, Q = vals[0].real, vals[1].real
                    C = vals[2].real + 1j * vals[3].real
                    sI, sQ, sC = np.sqrt(inv(ww[0])), np.sqrt(inv(ww[1])), np.sqrt(inv(ww[2]))
                else:
                    I = 0.5 * (vals[0].real + vals[3].real)
                    Q = 0.5 * (vals[0].real - vals[3].real)
                    C = 0.5 * (vals[1] + np.conj(vals[2]))
                    sI = 0.5 * np.sqrt(inv(ww[0]) + inv(ww[3]))
                    sQ = sI
                    sC = 0.5 * np.sqrt(inv(ww[1]) + inv(ww[2]))
                res, pb = _fit_pol_calibrator(
                    I, sI, Q, sQ, C, sC, good, freq, width, float(cal["rm"]),
                    self.frac_noise_max, self.min_bands, self.rm_fit_max,
                    self.delay_max,
                )  # fmt: skip
                res.update(
                    name=str(cal["name"]),
                    transit=transit,
                    ra=ra_c,
                    dec=float(dec_c),
                    y=float(np.median(yt[tsel])) if tsel.size else np.nan,
                    x_mean=float(np.mean(xt[tsel])) if tsel.size else np.nan,
                    nsamp=int(tsel.size),
                    rm_cat=float(cal["rm"]),
                )
                if res["ok"]:
                    res["delay_ns"] = tau0 + res["delay_resid_ns"]
                    res["offset_deg"] = (
                        (phi0 + res["offset_resid_deg"] + 180.0) % 360.0
                    ) - 180.0
                    self.log.info(
                        f"{res['name']} ({transit}, {tsel.size} samples, "
                        f"{res['nband']} bands): RM {res['rm']:+.1f}, p {res['pfrac']:.3f}, "
                        f"X-Y delay {res['delay_ns']:+.2f} ns, offset "
                        f"{res['offset_deg']:+.1f} deg (residual "
                        f"{res['delay_resid_ns']:+.2f} ns, {res['offset_resid_deg']:+.1f} "
                        f"deg), |A| {res['abs_a']:.2f}, B/A "
                        f"{res['b_re']:+.3f}{res['b_im']:+.3f}i, phase rms "
                        f"{res['phase_rms_deg']:.1f} deg"
                    )
                else:
                    self.log.info(
                        f"{res['name']} ({transit}): {tsel.size} samples, "
                        f"{res['nband']} usable bands: no fit."
                    )
                rows.append(res)
                per_band.append(pb)

        out = containers.PolarisationCalibration(
            calibrator=len(rows), freq=band_map, attrs_from=bands, comm=bands.comm
        )
        tab = out.calibrator[:]
        rec = np.zeros(len(rows), dtype=tab.dtype)
        for name in rec.dtype.names:
            if rec.dtype[name].kind == "f":
                rec[name] = np.nan
        for k, res in enumerate(rows):
            for name, val in res.items():
                if name in rec.dtype.names:
                    rec[name][k] = val
        out.calibrator[:] = rec
        for name in ("q", "q_err", "c", "c_err", "band_used"):
            out.datasets[name][:] = np.array([pb[name] for pb in per_band]).reshape(
                len(rows), nb
            )
        out.attrs["lsd"] = lsd
        out.attrs["xy_delay_applied_ns"] = tau0
        out.attrs["xy_offset_applied_deg"] = phi0
        out.attrs["x_max"] = self.x_max
        return out


def _fit_pol_calibrator(
    I, sI, Q, sQ, C, sC, good, freq, width, rm_cat, frac_max, min_bands, rm_max,
    delay_max,
):  # fmt: skip
    """Polarisation and X-Y phase of a scintillating pulsar (see
    :class:`FitPolarisationCalibrators`).  I, Q, C and their noise are [band,
    sample]; returns the fit (a dict) and the per-band values."""
    nb = freq.size
    wt = tools.invert_no_zero(np.where(good, sI, 0.0) ** 2)
    den = np.sum(wt * (I**2 - sI**2), axis=1)
    use = den > 0
    idd = tools.invert_no_zero(np.where(use, den, 0.0))
    q = np.where(use, np.sum(wt * Q * I, axis=1) * idd, np.nan)
    c = np.where(use, np.sum(wt * C * I, axis=1) * idd, np.nan)
    sq = np.where(use, np.sqrt(np.sum(wt**2 * sQ**2 * I**2, axis=1)) * idd, np.inf)
    sc = np.where(use, np.sqrt(np.sum(wt**2 * sC**2 * I**2, axis=1)) * idd, np.inf)
    keep = use & np.isfinite(q) & np.isfinite(c) & (sq < frac_max)
    pb = {
        "q": np.where(use, q, 0.0),
        "q_err": np.where(np.isfinite(sq), sq, 0.0),
        "c": np.where(use, c, 0.0),
        "c_err": np.where(np.isfinite(sc), sc, 0.0),
        "band_used": keep,
    }
    res = {"ok": False, "nband": int(keep.sum())}
    if keep.sum() < min_bands:
        return res, pb

    nu = freq[keep] * 1e6
    lam2 = (scipy.constants.c / nu) ** 2
    dl2 = 2.0 * lam2 * width[keep] / freq[keep]
    qk, sqk, ck, sck = q[keep], sq[keep], c[keep], sc[keep]

    # 1. |RM|, p and chi from the fractional Q
    rms = np.arange(0.5, rm_max, 0.1)
    w = tools.invert_no_zero(sqk**2)
    ph = 2.0 * rms[:, np.newaxis] * lam2[np.newaxis, :]
    dep = np.abs(np.sinc(rms[:, np.newaxis] * dl2[np.newaxis, :] / np.pi))
    tc, ts = dep * np.cos(ph), dep * np.sin(ph)
    A_, B_, D_ = (w * tc * tc).sum(-1), (w * tc * ts).sum(-1), (w * ts * ts).sum(-1)
    u_, v_ = (w * qk * tc).sum(-1), (w * qk * ts).sum(-1)
    det = A_ * D_ - B_ * B_
    det = np.where(det > 0, det, np.nan)
    a, b = (D_ * u_ - B_ * v_) / det, (A_ * v_ - B_ * u_) / det
    d2 = a * u_ + b * v_
    k = int(np.nanargmax(d2))
    rm_abs, pfrac = float(rms[k]), float(np.hypot(a[k], b[k]))
    chi = 0.5 * np.arctan2(-b[k], a[k])
    sign = 1.0 if rm_cat >= 0 else -1.0
    rm, chi_s = sign * rm_abs, sign * chi

    # 2. X-Y phase with the sign of U known
    depk = np.abs(np.sinc(rm * dl2 / np.pi))
    ub = pfrac * depk * np.sin(2.0 * (chi_s + rm * lam2))
    taus = np.arange(-delay_max, delay_max + 1e-9, 0.02) * 1e-9
    M = np.stack([ub, np.ones_like(ub)], axis=1).astype(complex)  # band, 2
    wc = tools.invert_no_zero(sck**2)
    Wm = M * wc[:, np.newaxis]
    G = M.conj().T @ Wm
    chi2 = np.empty(taus.size)
    coefs = np.empty((taus.size, 2), dtype=complex)
    for j, tau in enumerate(taus):
        y = ck * np.exp(-2j * np.pi * nu * tau)
        cf = np.linalg.lstsq(G, Wm.conj().T @ y, rcond=1e-10)[0]
        chi2[j] = np.sum(wc * np.abs(y - M @ cf) ** 2)
        coefs[j] = cf
    j = int(np.argmin(chi2))
    A, B = coefs[j]
    model = (A * ub + B) * np.exp(2j * np.pi * nu * taus[j])
    dphi = np.angle(ck / np.where(np.abs(model) > 0, model, 1.0))
    wph = np.abs(model) ** 2 / sck**2
    res.update(
        ok=True,
        frac_noise=float(np.median(sqk)),
        rm=rm,
        rm_abs=rm_abs,
        rm_dchi2=float(d2[k]),
        rot_deg=float(np.degrees(2.0 * rm_abs * np.ptp(lam2))),
        pfrac=pfrac,
        chi_deg=float(np.degrees(chi_s)),
        delay_resid_ns=float(taus[j] * 1e9),
        offset_resid_deg=float(np.degrees(np.angle(A))),
        abs_a=float(abs(A)),
        b_re=float((B / A).real),
        b_im=float((B / A).imag),
        phase_rms_deg=float(np.degrees(np.sqrt(np.sum(wph * dphi**2) / np.sum(wph)))),
        chi2=float(chi2[j]),
        dof=int(2 * keep.sum() - 5),
    )
    return res, pb

def _chi2_to_gaussian(q, k):
    """Equivalent Gaussian S/N of a chi-squared value ``q`` with ``k`` dof.

    The Wilson-Hilferty approximation; zero where ``k`` is zero.
    """
    k = np.asarray(k, dtype=np.float64)
    ok = k > 0
    ik = tools.invert_no_zero(np.where(ok, k, 0.0))
    c = 2.0 * ik / 9.0
    z = (np.cbrt(np.where(ok, q, 0.0) * ik) - (1.0 - c)) * tools.invert_no_zero(
        np.sqrt(c)
    )
    return np.where(ok, z, 0.0)


def _chi2max_to_gaussian(t, n_eff, single=True):
    """Equivalent Gaussian S/N of the largest of ``n_eff`` chi-squared values (2 dof).

    For independent trials ``P(max < t) = (1 - e^{-t/2})^n_eff``; ``z`` has the
    same probability under a unit Gaussian.  ``n_eff`` may be an array (the
    number at each ``t``).  The result is float32 if ``single``, else float64.
    """
    t = np.clip(np.asarray(t, dtype=np.float64), 0.0, None)
    n_eff = np.asarray(n_eff, dtype=np.float64)
    e = np.exp(-0.5 * t)
    log_cdf = n_eff * np.log1p(-np.minimum(e, 1.0 - 1e-12))
    # In the far tail log(1 - cdf) = log(n_eff) - t / 2 (both branches are
    # evaluated, and the unused one can be log 0)
    with np.errstate(divide="ignore"):
        log_sf = np.where(
            n_eff * e < 1e-8, np.log(n_eff) - 0.5 * t, np.log(-np.expm1(log_cdf))
        )
    z = np.where(
        log_cdf < np.log(0.5),
        scipy.special.ndtri_exp(log_cdf),
        -scipy.special.ndtri_exp(log_sf),
    )
    return z.astype(np.float32) if single else z


def _rm_ntrials_fn(a, c):
    """``n(t) = a + c sqrt(t)``: the number of independent RM trials at level t."""
    return lambda t: a + c * np.sqrt(np.clip(t, 0.0, None))


def _gaussian_to_chi2max(z, a, c, npoint=40001):
    """Inverse of :func:`_chi2max_to_gaussian` for ``n_eff = a + c sqrt(t)``.

    ``z`` is tabulated on a grid of ``npoint`` values of ``s = sqrt(t)`` out to the
    largest ``z`` and interpolated.  With ``c > 0`` it first falls, as ``n(t)``
    grows from ``a`` (to a minimum at ``t ~ 0.3``, below any maximum over many RM
    trials), and then rises; the rising branch is used, and lower ``z`` are given
    its minimum.
    """
    n = _rm_ntrials_fn(a, c)
    z = np.asarray(z, dtype=np.float64)
    zmax = max(float(np.max(z, initial=0.0)), 1.0)
    # In the tail t = 2 (log n(t) - log P(Z > z)), solved by a few iterations
    lsf = scipy.special.log_ndtr(-zmax)
    t = -2.0 * lsf
    for _ in range(5):
        t = 2.0 * (np.log(n(t)) - lsf)
    s = np.linspace(1e-4, 1.05 * np.sqrt(t) + 1.0, npoint)
    zg = _chi2max_to_gaussian(s**2, n(s**2), single=False)
    k = int(np.argmin(zg))
    return np.interp(z, zg[k:], s[k:]) ** 2


_MAD_TO_STD = 1.482602218505602


def _el_halo(local, off, nhalo, nel, comm, axis):
    """Extend a block of rows distributed over ranks by ``nhalo`` rows either side.

    ``local`` holds the global rows ``off`` to ``off + n`` along ``axis``.  The rows
    either side are fetched from the ranks that hold them; beyond the first and last
    global row they are NaN.

    Returns
    -------
    out : np.ndarray
        ``local`` with ``n + 2 nhalo`` rows along ``axis``.
    """
    local = np.moveaxis(local, axis, 0)
    n = local.shape[0]
    out = np.full((n + 2 * nhalo,) + local.shape[1:], np.nan, dtype=local.dtype)
    out[nhalo : nhalo + n] = local
    if nhalo > 0:
        single = comm is None or comm.size == 1
        blocks = [(off, n)] if single else comm.allgather((off, n))
        sends = []
        for o, m in blocks:
            want = np.r_[np.arange(o - nhalo, o), np.arange(o + m, o + m + nhalo)]
            mine = want[(want >= off) & (want < off + n) & (want >= 0) & (want < nel)]
            sends.append((mine, local[mine - off]))
        recv = sends if single else comm.alltoall(sends)
        for rows, vals in recv:
            if rows.size:
                out[rows - off + nhalo] = vals
    return np.moveaxis(out, 0, axis)


def _rolling_robust_sigma(a, centres, nrow, window, gap, step_t, min_valid):
    """Robust spread of ``a`` around each sample, from its neighbours.

    ``a`` [..., row, ra] holds absolute values (NaN where masked).  For each row
    index in ``centres`` (at least ``nrow`` from the ends of ``a``), at every
    ``step_t``-th sample (and the last), the spread is 1.4826 times the median of
    ``a`` over rows within ``nrow`` and samples with ``gap < |dt| <= window``, NaN
    if fewer than ``min_valid`` are present; it is interpolated linearly between
    those samples.

    Returns
    -------
    sigma : np.ndarray[..., len(centres), ra]
    """
    *lead, _, nra = a.shape
    tt = np.arange(0, nra, step_t)
    if tt[-1] != nra - 1:
        tt = np.append(tt, nra - 1)
    dt = np.arange(-window, window + 1)
    use = np.abs(dt) > gap
    pad = [(0, 0)] * (len(lead) + 1) + [(window, window)]
    ap = np.pad(a.astype(np.float32), pad, constant_values=np.nan)
    coarse = np.empty((*lead, len(centres), tt.size), dtype=np.float32)
    for ic, e in enumerate(centres):
        blk = ap[..., e - nrow : e + nrow + 1, :]  # ..., rows, ra + 2 window
        win = np.lib.stride_tricks.sliding_window_view(blk, 2 * window + 1, axis=-1)
        w = win[..., tt, :][..., use]  # ..., rows, tt, used offsets
        w = np.moveaxis(w, -3, -2).reshape(*lead, tt.size, -1)
        w = np.sort(w, axis=-1)  # NaN last
        cnt = np.sum(np.isfinite(w), axis=-1)
        lo = np.take_along_axis(w, np.maximum((cnt - 1) // 2, 0)[..., None], -1)[..., 0]
        hi = np.take_along_axis(w, np.maximum(cnt // 2, 0)[..., None], -1)[..., 0]
        med = 0.5 * (lo + hi)
        med[cnt < min_valid] = np.nan
        coarse[..., ic, :] = _MAD_TO_STD * med
    return _interp_last(coarse, tt, nra)


def _interp_last(c, at, n):
    """Interpolate ``c`` [..., len(at)] given at the indices ``at`` to 0..n-1."""
    if at.size == 1:
        return np.repeat(c, n, axis=-1)
    t = np.arange(n)
    i = np.clip(np.searchsorted(at, t, side="right") - 1, 0, at.size - 2)
    f = ((t - at[i]) / (at[i + 1] - at[i])).astype(np.float32)
    return c[..., i] * (1 - f) + c[..., i + 1] * f


class _RollingNoiseParams:
    """Settings shared by the rolling noise calibration tasks."""

    window = config.Property(proptype=int, default=45)
    gap = config.Property(proptype=int, default=15)
    nrow = config.Property(proptype=int, default=3)
    step_t = config.Property(proptype=int, default=10)
    step_e = config.Property(proptype=int, default=2)
    min_valid = config.Property(proptype=int, default=50)

    def _check_params(self, wmax=1):
        if self.gap < wmax - 1:
            raise ValueError(
                f"Durations up to {wmax} samples need gap >= {wmax - 1} (gap = {self.gap})."
            )
        if self.window <= self.gap or self.nrow < 0 or min(self.step_t, self.step_e) < 1:
            raise ValueError("Need window > gap, nrow >= 0, step_t >= 1 and step_e >= 1.")

    def _sigma(self, absval, off, nel, comm):
        """Rolling spread of ``absval`` [..., el_local, ra], NaN where masked.

        sigma is evaluated on the global rows that are multiples of ``step_e`` (and
        the last) and interpolated linearly between them, so the rows either side
        of the local block are fetched out to ``nrow + step_e - 1``.
        """
        n = absval.shape[-2]
        nh = self.nrow + self.step_e - 1
        a = _el_halo(absval, off, nh, nel, comm, axis=-2)
        ce = np.arange(0, nel, self.step_e)
        if ce[-1] != nel - 1:
            ce = np.append(ce, nel - 1)
        # the coarse rows that bracket the local ones
        i0 = max(np.searchsorted(ce, off, side="right") - 1, 0)
        i1 = min(np.searchsorted(ce, off + n - 1, side="left"), ce.size - 1)
        cl = ce[i0 : i1 + 1]
        sig = _rolling_robust_sigma(
            a, cl - off + nh, self.nrow, self.window, self.gap, self.step_t,
            self.min_valid,
        )
        if cl.size == 1:
            return np.repeat(sig, n, axis=-2)
        sig = np.moveaxis(_interp_last(np.moveaxis(sig, -2, -1), cl - off, n), -1, -2)
        return sig

    def _attrs(self, cont):
        cont.attrs["noise_calibrated"] = True
        for name in ("window", "gap", "nrow", "step_t", "step_e", "min_valid"):
            cont.attrs[f"noise_cal_{name}"] = getattr(self, name)


def _any_global(mask, comm):
    """Whether ``mask`` is true anywhere on any rank."""
    loc = bool(np.any(mask))
    if comm is None or comm.size == 1:
        return loc
    from mpi4py import MPI

    return bool(comm.allreduce(loc, op=MPI.LOR))


def _global_median(vals, comm):
    """Median over ranks of the local medians of the finite values (for logging)."""
    v = vals[np.isfinite(vals)]
    loc = float(np.median(v)) if v.size else np.nan
    allv = np.array(comm.allgather(loc) if comm is not None else [loc])
    allv = allv[np.isfinite(allv)]
    return float(np.median(allv)) if allv.size else np.nan


class CalibrateTrialNoise(_RollingNoiseParams, tasklib.base.ContainerTask):
    """Rescale the noise of the matched filter by its measured spread.

    The noise model of the matched filter (the thermal noise, the in-band scale and
    the sky term) does not follow every change of the noise: the S/N of the all-band
    map has a robust spread of 1.2-1.4 over a day, up to ~2 in a scintillating night,
    and more at low frequency.  So at every elevation, start time, telescope-x and
    trial, the S/N is divided by

        sigma = 1.4826 x median |S/N|

    over the rows within ``nrow`` and the start times with ``gap < |dt| <= window``
    of the same telescope-x and trial (masked samples, of zero weight, left out).
    The gap leaves the pulse out of its own estimate, so it must be at least the
    longest duration minus one.  sigma is evaluated every ``step_t`` samples and
    ``step_e`` rows and interpolated between them; where fewer than ``min_valid``
    samples are present the weight is set to zero.  The weights are divided by sigma^2, in place; the
    amplitudes are unchanged.  The result does not depend on the number of ranks.

    L (the RM search of a search in Stokes parameters) is not Gaussian in its bulk:
    the conversion of the largest chi-squared over RM to a Gaussian S/N
    (:func:`_chi2max_to_gaussian`) is right in the tail but conservative below
    z ~ 2-3, so that for noise 1.4826 median |z| is 1.1-1.45, depending on the
    weights of the bands.  Its own spread would rescale real noise by that much.
    So L is calibrated through Q and U at RM 0, which are Gaussian: sigma_Q from
    ``snr_q`` and sigma_U from the imaginary part of V (U at RM 0, with the
    weight of V), as above.  The chi-squared of L, recovered from z with the
    ``rm_ntrials_coef`` of the search, is divided by ``(sigma_Q^2 + sigma_U^2) / 2``
    and converted back to z; the amplitude keeps |P| and takes the sign of the new
    z.  ``snr_q`` is then removed.  Without ``snr_q`` (older trials) L is calibrated
    like the others.

    Attributes
    ----------
    window : int
        Half-width of the window in samples.  Default is 45.
    gap : int
        Samples within this many of the one being calibrated are left out.  Default
        is 15.
    nrow : int
        Rows either side included.  Default is 3.
    step_t, step_e : int
        Samples and rows between the evaluations of sigma, which is interpolated
        linearly between them.  Defaults are 10 and 2 (on 2418 and 3035 the
        statistics of the calibrated maps are those of every sample and row to
        ~1%, at a third of the cost).
    min_valid : int
        Least number of samples for an estimate.  Default is 50.
    """

    def process(self, trials):
        """Calibrate the trials.

        Parameters
        ----------
        trials : containers.TransientMatchedFilterTrials
            Matched filter output.

        Returns
        -------
        trials : containers.TransientMatchedFilterTrials
            The input, with its weights rescaled.
        """
        durations = [int(d) for d in trials.index_map["duration"][:]]
        self._check_params(max(durations))
        trials.redistribute("el")
        amp = trials.amplitude[:].local_array  # pol, dur, alpha, el, ra, x
        wgt = trials.weight[:].local_array
        off = trials.weight[:].local_offset[3]
        nel = trials.weight.shape[3]
        comm = trials.comm
        npol, ndur, nalpha = amp.shape[:3]
        med = np.full((npol, ndur, nalpha), np.nan)
        pol = _decode_names(trials.index_map["pol"][:])
        il = None
        if "L" in pol and "snr_q" in trials.datasets:
            # First, while the weights of V (for U) are those of the model
            il = pol.index("L")
            med[il], qu = self._calibrate_l(trials, pol, off, nel, comm)
            del trials._data["snr_q"]
        elif "L" in pol and "V" in pol and "rm_trials" in trials.attrs:
            self.log.warning(
                "No snr_q in the trials: L is calibrated by its own spread, which "
                "is not that of a Gaussian."
            )
        for p, d, a in np.ndindex(npol, ndur, nalpha):
            if p == il:
                continue
            w = wgt[p, d, a]
            # Trials without data anywhere (the RM search at other spectral indices)
            if not _any_global(w > 0, comm):
                continue
            av = np.where(w > 0, np.abs(amp[p, d, a]) * np.sqrt(np.abs(w)), np.nan)
            sig = self._sigma(np.moveaxis(av, -1, 0), off, nel, comm)  # x, el, ra
            sig = np.moveaxis(sig, 0, -1)
            ok = np.isfinite(sig) & (sig > 0)
            wgt[p, d, a] = w * tools.invert_no_zero(np.where(ok, sig, 0.0) ** 2)
            med[p, d, a] = _global_median(sig[ok & (w > 0)], comm)
        self._attrs(trials)
        for d in range(ndur):
            self.log.info(
                f"Duration {durations[d]}: median noise rescaling "
                + ", ".join(f"{v:.2f}" for v in med[:, d].ravel())
                + (
                    ""
                    if il is None
                    else " (L: sqrt of the mean of Q and U; sigma_Q / sigma_U "
                    + ", ".join(f"{v:.2f}" for v in qu[d])
                    + ")"
                )
            )
        return trials

    def _calibrate_l(self, trials, pol, off, nel, comm):
        """Calibrate L through the spread of Q and U (see the class docstring).

        Returns the median of ``sqrt((sigma_Q^2 + sigma_U^2) / 2)`` and of
        ``sigma_Q / sigma_U`` for each duration and spectral index.
        """
        amp = trials.amplitude[:].local_array
        wgt = trials.weight[:].local_array
        imag = trials.amplitude_imag[:].local_array
        snrq = trials.datasets["snr_q"][:].local_array
        il, iv = pol.index("L"), pol.index("V")
        alphas = np.asarray(trials.index_map["spectral_index"][:], dtype=np.float64)
        rm_a = np.atleast_1d(trials.attrs["rm_spectral_indices"])
        coef = np.atleast_2d(trials.attrs["rm_ntrials_coef"])
        ndur, nalpha = amp.shape[1:3]
        med = np.full((ndur, nalpha), np.nan)
        ratio = np.full((ndur, nalpha), np.nan)

        def sigma(absval):  # [el, ra, x], NaN where masked
            s = self._sigma(np.moveaxis(absval, -1, 0), off, nel, comm)  # x, el, ra
            return np.moveaxis(s, 0, -1)

        for d, a in np.ndindex(ndur, nalpha):
            w = wgt[il, d, a]
            if not _any_global(w > 0, comm):
                continue
            k = np.flatnonzero(np.isclose(rm_a, alphas[a]))
            if k.size == 0:
                raise ValueError(
                    f"L has trials at spectral index {alphas[a]}, which was not in "
                    "the RM search."
                )
            sq = sigma(np.where(snrq[d, a] != 0, np.abs(snrq[d, a]), np.nan))
            wv = wgt[iv, d, a]
            au = np.where(wv > 0, np.abs(imag[iv, d, a]) * np.sqrt(np.abs(wv)), np.nan)
            su = sigma(au)
            ok = np.isfinite(sq) & np.isfinite(su) & (sq > 0) & (su > 0)
            s2 = np.where(ok, 0.5 * (sq**2 + su**2), 1.0)
            pa = np.abs(amp[il, d, a])
            has = ok & (w > 0) & (pa > 0)
            # z = sign(z) |P| * |z| / |P|
            z = amp[il, d, a][has] * np.sqrt(w[has])
            t = _gaussian_to_chi2max(z, *coef[k[0]]) / s2[has]
            zn = _chi2max_to_gaussian(t, _rm_ntrials_fn(*coef[k[0]])(t))
            new_a = np.zeros_like(amp[il, d, a])
            new_w = np.zeros_like(w)
            new_a[has] = np.sign(zn) * pa[has]
            new_w[has] = (zn / pa[has]) ** 2
            amp[il, d, a] = np.where(has, new_a, amp[il, d, a])
            wgt[il, d, a] = np.where(ok, new_w, 0.0)
            med[d, a] = _global_median(np.sqrt(s2[has]), comm)
            ratio[d, a] = _global_median((sq / np.where(ok, su, 1.0))[has], comm)
        return med, [ratio[d][np.isfinite(ratio[d])] for d in range(ndur)]


class CalibrateGroupNoise(_RollingNoiseParams, tasklib.base.ContainerTask):
    """Rescale the band-group maps by the measured spread of their S/N.

    As :class:`CalibrateTrialNoise`, for the single-sample maps of
    :class:`CollapseBandsToGroups` (or of the sky-noise matched filter): sigma is
    measured on the map at a fixed telescope-x (normally 0) for each group, and
    both that map and the map at the best telescope-x are divided by it.  (The
    best-x map is a maximum over telescope-x, whose spread is not that of a
    Gaussian, so it is not measured on its own.)

    Attributes
    ----------
    window, gap, nrow, step_t, min_valid
        As :class:`CalibrateTrialNoise`.
    """

    def process(self, maxx, x0):
        """Calibrate the two maps.

        Parameters
        ----------
        maxx : containers.TransientBandGroupMap
            Map at the best telescope-x.
        x0 : containers.TransientBandGroupMap
            Map at a single telescope-x, on which sigma is measured.

        Returns
        -------
        maxx, x0 : containers.TransientBandGroupMap
            The inputs, rescaled in place.
        """
        self._check_params()
        if maxx.snr.shape != x0.snr.shape:
            raise ValueError("The two group maps differ in shape.")
        if str(x0.attrs.get("x_reduce", "")) == "max":
            raise ValueError("The second map must be at a fixed telescope-x.")
        maxx.redistribute("el")
        x0.redistribute("el")
        s0 = x0.snr[:].local_array  # pol, group, el, ra
        s1 = maxx.snr[:].local_array
        off = x0.snr[:].local_offset[2]
        nel = x0.snr.shape[2]
        groups = _decode_names(x0.index_map["group"][:])
        for p, g in np.ndindex(s0.shape[:2]):
            av = np.where(s0[p, g] != 0, np.abs(s0[p, g]), np.nan)
            sig = self._sigma(av, off, nel, x0.comm)
            ok = np.isfinite(sig) & (sig > 0)
            inv = tools.invert_no_zero(np.where(ok, sig, 0.0))
            s0[p, g] *= inv
            s1[p, g] *= inv
            self.log.info(
                f"Group {groups[g]}: median noise rescaling "
                f"{_global_median(sig[ok], x0.comm):.2f}"
            )
        self._attrs(maxx)
        self._attrs(x0)
        return maxx, x0


class MaxMatchedFilterTrials(tasklib.base.ContainerTask):
    """Find the trial with the highest signal-to-noise ratio.

    The maximum is over duration, spectral index, and telescope-x, for each
    elevation and start time.

    Attributes
    ----------
    abs_pol : list of str
        Polarisations whose sign carries no preference (Stokes V): the maximum is
        of |S/N|, which is kept as the S/N, while the amplitude keeps its sign.
        Default is none.
    negate : bool
        Find the most negative trial instead (of the other polarisations): the map
        of -S/N and -amplitude, a null for the search, since noise and
        scintillation are two-sided and a pulse is positive (see
        :class:`TransientNegativePeaks`).  Default is False.
    """

    abs_pol = config.list_type(str, default=[])
    negate = config.Property(proptype=bool, default=False)

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
        pol = _decode_names(trials.index_map["pol"][:])
        sign = np.ones(npol)
        for pp, name in enumerate(pol):
            if name in self.abs_pol:
                snr[pp] = np.abs(snr[pp])
            elif self.negate:
                snr[pp] *= -1.0
                sign[pp] = -1.0
        # [pol, el, ra, dur, alpha, x], flattened over the trial parameters.  Trials
        # without data are left out, so that where every trial is negative (L can
        # be) the best of them is kept rather than an empty one with S/N 0
        flat = np.moveaxis(snr, (1, 2), (3, 4)).reshape(npol, nel, nra, -1)
        has = np.moveaxis(wgt > 0, (1, 2), (3, 4)).reshape(npol, nel, nra, -1)
        best = np.argmax(np.where(has, flat, -np.inf), axis=-1)
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
        out.amplitude[:].local_array[:] = (
            sign[:, np.newaxis, np.newaxis] * amp[pp, idur, ialpha, ee, rr, ix]
        )
        out.weight[:].local_array[:] = wgt[pp, idur, ialpha, ee, rr, ix]
        if "amplitude_imag" in trials.datasets:
            out.add_dataset("amplitude_imag")
            out.redistribute("el")
            aim = trials.datasets["amplitude_imag"][:].local_array
            out.datasets["amplitude_imag"][:].local_array[:] = aim[
                pp, idur, ialpha, ee, rr, ix
            ]
        out.datasets["duration"][:].local_array[:] = trials.index_map["duration"][idur]
        out.datasets["spectral_index"][:].local_array[:] = trials.index_map[
            "spectral_index"
        ][ialpha]
        out.datasets["x_index"][:].local_array[:] = ix
        out.datasets["ha"][:] = trials.datasets["ha"][:]
        out.datasets["dec"][:] = trials.datasets["dec"][:]
        out.attrs["negated"] = self.negate

        return out


class MaskFracLost(tasklib.base.ContainerTask):
    """Mask the samples of which a large fraction was lost in the real-time pipeline.

    Uses the ``frac_lost`` dataset of the daily system-sensitivity product
    (``sensitivity_lsd_*.h5``, a :class:`draco.core.containers.SystemSensitivity`),
    since the daily timestream does not carry it.  As
    :class:`ch_pipeline.analysis.flagging.MaskHighFracLost`, whose default of 2% is
    set by the spectral requirements of the 21 cm analysis.  Samples that are lost
    entirely already have zero weight; partly lost ones remain, and are mostly in
    the channels of RFI that the real-time excision does not fully remove.  Their
    weights, smoothed in time, do not show the loss.

    Attributes
    ----------
    threshold : float
        Mask a frequency and time if more than this fraction was lost.
        Default is 0.2.
    """

    threshold = config.Property(proptype=float, default=0.2)

    def process(self, sens):
        """Make the mask.

        Parameters
        ----------
        sens : draco.core.containers.SystemSensitivity
            Daily system sensitivity, with ``frac_lost`` [freq, time].

        Returns
        -------
        mask : draco.core.containers.RFIMask
        """
        sens.redistribute("freq")
        fl = sens.frac_lost[:].allgather()
        out = draco_containers.RFIMask(axes_from=sens, attrs_from=sens)
        mask = np.isfinite(fl) & (fl > self.threshold)
        out.mask[:] = mask
        self.log.info(
            f"{100.0 * mask.mean():.2f}% of the samples lost more than "
            f"{self.threshold:g} of their data."
        )
        return out


class FlagHighNoise(tasklib.base.ContainerTask):
    """Flag hybrid visibilities whose noise is far above that of the day.

    Sets the weight to zero where the noise from the weights, ``1 / sqrt(weight)``,
    is more than ``max_ratio`` times its median over the frequencies and times with
    data, separately for each polarisation and east-west baseline.  The system
    temperature ripples with frequency by up to a factor ~3 between the clean
    channels, so the default of 4 only removes RFI and the steep edge of the band
    (and some of the transit of the Sun).

    Attributes
    ----------
    max_ratio : float
        Largest noise relative to the median.  Default is 4.
    """

    max_ratio = config.Property(proptype=float, default=4.0)

    def process(self, stream):
        """Flag the stream in place.

        Parameters
        ----------
        stream : draco.core.containers.HybridVisStream
            Hybrid beamformed visibilities.

        Returns
        -------
        stream : draco.core.containers.HybridVisStream
            The same container, with weights set to zero where flagged.
        """
        from mpi4py import MPI

        stream.redistribute("freq")
        w = stream.weight[:].local_array  # pol, freq, ew, ra
        comm = stream.comm
        npol, _, new, _ = w.shape
        nflag, ntot = 0, 0
        for pp in range(npol):
            for ee in range(new):
                wl = w[pp, :, ee]
                have = wl > 0
                # median over all ranks (the weights of one pol and baseline, ~40 MB)
                vals = wl[have]
                allv = np.concatenate(comm.allgather(vals)) if comm.size > 1 else vals
                if allv.size == 0:
                    continue
                wmed = np.median(allv)
                # noise > max_ratio x median  <=>  weight < median weight / max_ratio^2
                bad = have & (wl < wmed / self.max_ratio**2)
                wl[bad] = 0.0
                nflag += int(bad.sum())
                ntot += int(have.sum())
        nflag = comm.allreduce(nflag, op=MPI.SUM)
        ntot = comm.allreduce(ntot, op=MPI.SUM)
        self.log.info(
            f"Flagged {100.0 * nflag / max(ntot, 1):.3f}% of the samples with noise "
            f"above {self.max_ratio:g} x the median."
        )
        return stream


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


class SelectElRange(tasklib.base.ContainerTask):
    """Keep the elevations (telescope-y) of a container within a range.

    For example to cut a small window around a source from a hybrid stream before
    the matched filter.  Datasets without an ``el`` axis (the weights, the filter
    response) are copied unchanged, unlike draco's Downselect, which drops them.

    Attributes
    ----------
    el_min, el_max : float
        Keep rows with ``el_min <= el < el_max``.  Defaults are -1 and 1.
    """

    el_min = config.Property(proptype=float, default=-1.0)
    el_max = config.Property(proptype=float, default=1.0)

    def process(self, cont):
        """Select the rows.

        Parameters
        ----------
        cont : container with an ``el`` axis
            For example a (filtered) HybridVisStream.

        Returns
        -------
        out : same type as ``cont``
            The rows within the range.
        """
        from caput.containers import copy_datasets_filter

        el = cont.index_map["el"][:]
        keep = np.flatnonzero((el >= self.el_min) & (el < self.el_max))
        if keep.size == 0:
            raise ValueError(
                f"No rows with {self.el_min} <= el < {self.el_max} "
                f"(the axis spans {el.min():.3f} to {el.max():.3f})."
            )
        sel = slice(keep[0], keep[-1] + 1)
        # Distributed over el: gather the rows on every rank first
        for ax in ("freq", "ra", "pol"):
            if ax in cont.index_map:
                try:
                    cont.redistribute(ax)
                    break
                except (ValueError, KeyError):
                    continue
        out = cont.__class__(
            axes_from=cont,
            attrs_from=cont,
            el=el[sel],
            skip_datasets=True,
            distributed=cont.distributed,
            comm=cont.comm,
        )
        copy_datasets_filter(
            cont, out, selection={"el": sel}, copy_without_selection=True
        )
        self.log.info(
            f"Kept {keep.size} of {el.size} rows (el {el[sel][0]:.4f} to {el[sel][-1]:.4f})."
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


def _sky_sep_deg(ra1, dec1, ra2, dec2):
    """Angular separation in degrees of positions in degrees."""
    r1, d1, r2, d2 = (np.radians(v) for v in (ra1, dec1, ra2, dec2))
    c = np.sin(d1) * np.sin(d2) + np.cos(d1) * np.cos(d2) * np.cos(r1 - r2)
    return np.degrees(np.arccos(np.clip(c, -1.0, 1.0)))


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
      "extent" is the same but with pixels above ``max(grow_snr, ext_frac x
      peak)`` joined across gaps of up to ``join_dy`` in telescope-y and
      ``join_dt`` samples, so that a chain of blobs (an arc broken up by the
      fringes) counts as one: ``ext_npix`` (its pixels above the threshold),
      ``ext_dy`` and ``ext_dt`` (s).  A pulse from a point source extends ~0.01 in
      telescope-y.  The threshold rises with the peak so that the sidelobes of a
      bright pulse (below 5% in telescope-y) and the noise around it do not count.
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
      ``cat_dra`` and ``cat_ddec`` in degrees), and their number ``cat_nmatch``;
      and the brightest within ``cat_near_sep`` degrees of the candidate, under it
      (``cat_near_name``, ``cat_near_flux``, ``cat_near_sep``), whose scintillation
      is the commonest background (see the ``compact`` flag of the sifter).
    - Known pulsars (if ``pulsar_catalogue`` is set): the nearest pulsar with a flux
      density at 600 MHz of at least ``psr_flux_min`` within ``psr_sep`` degrees
      (``psr_name``, ``psr_sep``, ``psr_flux`` in Jy).

    Attributes
    ----------
    pol : int
        Index of the polarisation of the maps to use.  Default is 0.
    group_pol : int
        Index of the polarisation of the group maps.  Default is ``pol``.  For the
        candidates of the Stokes search in V or L (see
        :class:`TransientMatchedFilterSkyNoise`), whose group maps are made for I
        only, 0: the column and spectrum columns then describe I there.
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
    ext_frac : float
        Threshold of the extent as a fraction of the peak, where that is above
        ``grow_snr``.  Default is 0.05.
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
    cat_near_sep : float
        Radius of the match of a source under the candidate in degrees.  Default is
        0.1 (on 2418, 55% of the kept transients have a source above 0.3 Jy this
        close, against 4% by chance).
    pulsar_catalogue : str
        HDF5 file with a dataset ``pulsar`` with fields ``psrj``, ``psrb``, ``ra``,
        ``dec`` (J2000, degrees) and ``s600`` (mJy), from
        scripts/sift/make_pulsar_catalog.py.  If not set, the pulsar columns are
        left empty.
    psr_sep : float
        Radius of the pulsar match in degrees.  Default is 0.2.
    psr_flux_min : float
        Smallest flux density at 600 MHz of a pulsar to match, in Jy.  Default is
        0.015 (on 2418 every pulsar within 0.2 deg of a kept transient was above
        ~0.02 Jy).
    max_candidates : int
        Keep at most this many candidates (the brightest).  Default is 50000.
    """

    pol = config.Property(proptype=int, default=0)
    group_pol = config.Property(proptype=int, default=None)
    seed_snr = config.Property(proptype=float, default=8.0)
    peak_dy = config.Property(proptype=float, default=0.02)
    peak_dt = config.Property(proptype=int, default=3)
    grow_snr = config.Property(proptype=float, default=5.0)
    box_dy = config.Property(proptype=float, default=0.1)
    box_dra = config.Property(proptype=float, default=2.0)
    join_dy = config.Property(proptype=float, default=0.005)
    join_dt = config.Property(proptype=int, default=1)
    ext_frac = config.Property(proptype=float, default=0.05)
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
    cat_near_sep = config.Property(proptype=float, default=0.1)
    pulsar_catalogue = config.Property(proptype=str, default=None)
    psr_sep = config.Property(proptype=float, default=0.2)
    psr_flux_min = config.Property(proptype=float, default=0.015)
    max_candidates = config.Property(proptype=int, default=50000)

    def setup(self):
        """Load the catalogues, if any."""
        self._cat = None
        self._psr = None
        if self.pulsar_catalogue:
            import h5py

            with h5py.File(self.pulsar_catalogue, "r") as f:
                p = f["pulsar"][:]
            flux = p["s600"] * 1e-3
            ok = np.isfinite(flux) & (flux >= self.psr_flux_min)
            self._psr = {
                "name": np.where(p["psrb"] != b"", p["psrb"], p["psrj"])[ok],
                "ra": p["ra"][ok],
                "dec": p["dec"][ok],
                "flux": flux[ok],
            }
            self.log.info(
                f"Loaded {ok.sum()} pulsars of at least {self.psr_flux_min:g} Jy at "
                f"600 MHz from {self.pulsar_catalogue}."
            )
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
        gp = self.pol if self.group_pol is None else self.group_pol
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
            np.ascontiguousarray(groups_x0.snr[:].local_array[gp, names0.index("all")]),
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
        out.attrs["pol"] = _decode_names(cand.index_map["pol"][:])[pp]
        for key in (
            "seed_snr", "peak_dy", "peak_dt", "grow_snr", "join_dy", "join_dt",
            "ext_frac", "row_snr", "row_ratio", "row_dy", "row_dra", "col_snr",
            "group_snr_min", "cat_ddec", "cat_dra",
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
        names_c = ["amplitude", "duration", "spectral_index"]
        if "amplitude_imag" in cand.datasets:
            names_c.append("amplitude_imag")
        for name in names_c:
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

            lit = sub >= max(self.grow_snr, self.ext_frac * peak)
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
            gsnr[mine] = loc.local_array[gp][:, pe[mine] - off, pt[mine]].T
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

            # The brightest source under the candidate
            lo = np.searchsorted(cdec, rec["dec_icrs"] - self.cat_near_sep)
            hi = np.searchsorted(cdec, rec["dec_icrs"] + self.cat_near_sep)
            rec["cat_near_flux"] = 0.0
            for k in range(ncand):
                if hi[k] == lo[k]:
                    continue
                sl = slice(lo[k], hi[k])
                sep = _sky_sep_deg(
                    rec["ra_icrs"][k], rec["dec_icrs"][k], self._cat["ra"][sl],
                    self._cat["dec"][sl],
                )
                within = np.flatnonzero(sep <= self.cat_near_sep)
                if within.size == 0:
                    continue
                j = within[np.argmax(self._cat["flux"][sl][within])]
                rec["cat_near_name"][k] = self._cat["name"][sl][j]
                rec["cat_near_flux"][k] = self._cat["flux"][sl][j]
                rec["cat_near_sep"][k] = sep[j]

        # Known pulsars
        if self._psr is not None:
            for k in range(ncand):
                sep = _sky_sep_deg(
                    rec["ra_icrs"][k], rec["dec_icrs"][k], self._psr["ra"],
                    self._psr["dec"],
                )
                j = int(np.argmin(sep)) if sep.size else -1
                if j >= 0 and sep[j] <= self.psr_sep:
                    rec["psr_name"][k] = self._psr["name"][j]
                    rec["psr_sep"][k] = sep[j]
                    rec["psr_flux"][k] = self._psr["flux"][j]

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


def _ns_profile_fit(nsb, yr, amp, wgt, freq, shifts, scales, min_rows, eig_tol=1e-2):
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


def _derotate_linear(centre, aq, wq, au, wu, rm):
    """Linear polarisation of each band derotated by the RM and the mean angle.

    Parameters
    ----------
    centre : np.ndarray[band]
        Band centres in MHz.
    aq, wq, au, wu : np.ndarray[cand, band]
        Amplitudes and inverse variances of Q and U.
    rm : np.ndarray[cand]
        RM of each candidate (rad/m^2).

    Returns
    -------
    amp, wgt : np.ndarray[cand, band]
        ``Re[(Q + iU) e^{-i phi}]`` with ``phi = 2 RM l^2 + psi`` and ``psi`` the
        angle of the weighted sum over the bands, and its inverse variance
        ``1 / (cos^2 phi / w_Q + sin^2 phi / w_U)``.
    """
    lam2 = (scipy.constants.c / (np.asarray(centre) * 1e6)) ** 2
    ok = (wq > 0) & (wu > 0)
    vq = np.where(ok, tools.invert_no_zero(wq), 0.0)
    vu = np.where(ok, tools.invert_no_zero(wu), 0.0)
    p = (aq + 1j * au) * np.exp(-2.0j * np.asarray(rm, dtype=np.float64)[:, np.newaxis] * lam2)
    wmean = np.where(ok, tools.invert_no_zero(0.5 * (vq + vu)), 0.0)
    psi = np.angle(np.sum(wmean * p, axis=1))
    phi = 2.0 * rm[:, np.newaxis] * lam2 + psi[:, np.newaxis]
    amp = np.where(ok, (p * np.exp(-1.0j * psi[:, np.newaxis])).real, 0.0)
    var = np.cos(phi) ** 2 * vq + np.sin(phi) ** 2 * vu
    wgt = np.where(ok, tools.invert_no_zero(var), 0.0)
    return amp.astype(np.float32), wgt.astype(np.float32)


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
    best duration W: from the output of :class:`CandidateBandSpectra` if it is given
    and has the candidate (exact), else from the band output for W if it has it, or
    else the inverse-variance weighted mean of the single-sample band amplitudes
    along the track of the source (the telescope-x of the candidate's sky position
    at each of the W samples).  The last leaves out the coupling of the samples by
    the filter, which removes more of a long pulse at the higher frequencies: its
    spectral index is too steep (by ~-1.5 to -2 for 4 to 16 samples on the
    injections of 2418).  Without ``match_duration``, every spectrum is for
    ``duration``.

    For the candidates of the search in linear polarisation (``lin_pol``, the
    indices of Q and U in the bands), Q and U rotate with the RM across the band, so
    their spectra are not smooth even for a real source.  The spectrum is then the
    linear polarisation derotated by the candidate's RM (the ``amplitude_imag``
    column) and by the angle of its weighted sum over the bands,
    ``Re[(Q_b + i U_b) e^{-2 i RM l_b^2 - i psi}]``, with the variance of that
    projection; for a source at the right RM it is |P| in each band (less the
    depolarisation within the band).

    Attributes
    ----------
    pol : int
        Index of the polarisation.  Default is 0.
    lin_pol : list of int
        Indices of Q and U for the derotated spectrum of linear polarisation
        (``pol`` is then unused).  Default is None.
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
    lin_pol = config.list_type(int, length=2, default=None)
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
        return self._spectra(table, bands, None)

    def _spectra(self, table, bands, exact):
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
        if not isinstance(exact, containers.TransientCandidateCutouts):
            exact = None
        if exact is not None:
            ex_pol = _decode_names(exact.index_map["pol"][:])
            ex_k = exact.datasets["candidate_index"][:]
            ex_w = exact.datasets["duration"][:]
            ex_amp = exact.amplitude[:].allgather()[..., 0, 0]  # cand, pol, band
            ex_wgt = exact.weight[:].allgather()[..., 0, 0]
            if ex_amp.shape[2] != centre.size:
                raise ValueError("The exact bands differ from those of the table.")
            use = (ex_w == wc[ex_k]) & (xi[ex_k] == exact.datasets["x_index"][:])
        bpol = _decode_names(bands.index_map["pol"][:])
        cuts = []
        for pp in [self.pol] if self.lin_pol is None else self.lin_pol:
            amp, wgt = _band_cutouts(bands, pp, e, tb, xi, wc, [0], [0])
            amp, wgt = amp[:, :, 0, 0], wgt[:, :, 0, 0]
            bands.comm.Allreduce(MPI.IN_PLACE, amp, op=MPI.SUM)
            bands.comm.Allreduce(MPI.IN_PLACE, wgt, op=MPI.SUM)
            if exact is not None:
                ip = ex_pol.index(bpol[pp])
                amp[ex_k[use]] = ex_amp[use, ip]
                wgt[ex_k[use]] = ex_wgt[use, ip]
            cuts.append((amp, wgt))
        if self.lin_pol is None:
            amp, wgt = cuts[0]
        else:
            if "amplitude_imag" not in rec.dtype.names:
                raise ValueError("lin_pol needs the RM of each candidate (amplitude_imag).")
            amp, wgt = _derotate_linear(centre, *cuts[0], *cuts[1], rec["amplitude_imag"])

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

        nex = int(np.sum(use)) if exact is not None else 0
        self.log.info(
            f"Added the spectra of {ncand} candidates ({nex} from the matched filter "
            "at their own duration)."
        )
        return out



class AddCandidateSpectraAtDuration(AddCandidateSpectra):
    """:class:`AddCandidateSpectra` with the bands at each candidate's own duration.

    The third input is the output of :class:`CandidateBandSpectra`; the candidates
    it has (same duration and telescope-x) take their spectra from it, the others
    from the band output as in :class:`AddCandidateSpectra`.  Anything else in its
    place is ignored, so that a PassOn can stand in for it (thermal noise).
    """

    def process(self, table, bands, exact):
        """Add the spectra.

        Parameters
        ----------
        table : containers.TransientCandidateTable
            Candidates, from :class:`GroupTransientCandidates`.
        bands : containers.TransientMatchedFilter
            Matched filter output for each band.
        exact : containers.TransientCandidateCutouts
            The bands at the candidates' own durations.

        Returns
        -------
        out : containers.TransientCandidateTable
            A copy of the input, with the spectra added.
        """
        return self._spectra(table, bands, exact)

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
    fit uses the covariance from the beamforming, scaled by the weights, whitened
    with its eigenvectors above ``eig_tol`` of the largest eigenvalue.  The rest are
    nearly empty in the model, but real data are not exactly the model (sky noise,
    the beamforming weights, rounding) and whitening amplifies them enormously: with
    1e-9 J1634 had a reduced chi-squared of 1.5e6, with 1e-2 it is ~3.

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
    eig_tol : float
        Eigenvalues of the row covariance below this fraction of the largest are
        left out of the fit.  Default is 1e-2.
    """

    pol = config.Property(proptype=int, default=0)
    match_duration = config.Property(proptype=bool, default=True)
    nrow = config.Property(proptype=int, default=12)
    min_rows = config.Property(proptype=int, default=5)
    max_shift = config.Property(proptype=float, default=1.5)
    scale_range = config.list_type(type_=float, length=2, default=[0.5, 8.0])
    ns_spacing = config.Property(proptype=float, default=0.3048)
    ns_max = config.Property(proptype=float, default=73.0)
    eig_tol = config.Property(proptype=float, default=1e-2)

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
                nsb, yr, amp[k], w, centre, shifts, scales, self.min_rows, self.eig_tol
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
        out.attrs["ns_eig_tol"] = self.eig_tol
        nfit = int(np.sum(rec["ns_dof"] > 0))
        self.log.info(f"Fitted the north-south profiles of {nfit} of {ncand} candidates.")
        return out


def _extend_axis(axis, idx):
    """Values of ``axis`` at the integer positions ``idx``, continued linearly with the
    end spacings beyond either end."""
    n = axis.size
    inside = axis[np.clip(idx, 0, n - 1)]
    if n < 2:
        return inside
    lo = axis[0] + idx * (axis[1] - axis[0])
    hi = axis[-1] + (idx - (n - 1)) * (axis[-1] - axis[-2])
    return np.where(idx < 0, lo, np.where(idx >= n, hi, inside))


class ExtractCandidateCutouts(tasklib.base.ContainerTask):
    """Windows of the per-band matched filter around candidates.

    For the candidates with S/N of at least ``snr_min`` (and only those kept by the
    sifter if ``keep_only`` and the table has been sifted), the brightest
    ``max_cutouts`` of them: the amplitude and weight of every band over
    +-``nrow`` rows and +-``nra`` start times, at the candidate's telescope-x, for
    single samples, in the polarisation ``pol`` or each of ``pols``.  The result is
    distributed over candidates.

    Attributes
    ----------
    pol : int
        Index of the polarisation.  Default is 0.
    pols : list of str
        Names of the polarisations of the bands to extract instead of ``pol`` (e.g.
        ["I", "Q", "U", "V"] for the Stokes search, so that the candidates of V and
        L can be seen in V and in Q and U).  Default is None.
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
    pols = config.list_type(type_=str, default=None)
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
        bpol = _decode_names(bands.index_map["pol"][:])
        if self.pols is None:
            ipol = [self.pol]
        else:
            missing = [p for p in self.pols if p not in bpol]
            if missing:
                raise ValueError(f"The bands have no polarisation {missing} ({bpol}).")
            ipol = [bpol.index(p) for p in self.pols]
        pnames = [bpol[ip] for ip in ipol]
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
            pol=np.array(pnames),
            freq=bands.index_map["freq"],
            el_offset=roff,
            ra_offset=toff,
            attrs_from=table,
            distributed=True,
            comm=comm,
        )
        out.datasets["candidate_index"][:] = idx
        out.datasets["x_index"][:] = xi
        out.datasets["duration"][:] = 1
        rows = e[:, np.newaxis] + roff[np.newaxis]
        rin = (rows >= 0) & (rows < el.size)
        out.datasets["el_index"][:] = np.where(rin, rows, -1)
        # Outside the map the coordinates continue the grid; there is no data there
        # (zero weight, index -1)
        out.datasets["el"][:] = _extend_axis(el, rows)
        ts = tb[:, np.newaxis] + toff[np.newaxis]
        if periodic:
            tin = np.ones(ts.shape, dtype=bool)
            ts = ts % ra_b.size
        else:
            tin = (ts >= 0) & (ts < ra_b.size)
        out.datasets["ra_index"][:] = np.where(tin, ts, -1)
        out.datasets["ra"][:] = _extend_axis(ra_b, ts)

        # Each rank's block of candidates, summed from the rows on every rank
        la = out.amplitude[:]
        offs = comm.allgather((la.local_offset[0], la.local_shape[0]))
        for r, (o, nl) in enumerate(offs):
            k = slice(o, o + nl)
            aw = [
                _band_cutouts(
                    bands, ip, e[k], tb[k], xi[k], np.ones(nl, int), roff, toff
                )
                for ip in ipol
            ]
            # cand, pol, freq, row, time
            a = np.stack([x[0] for x in aw], axis=1).astype(np.float32)
            w = np.stack([x[1] for x in aw], axis=1).astype(np.float32)
            ra_ = np.zeros_like(a) if comm.rank == r else None
            rw_ = np.zeros_like(w) if comm.rank == r else None
            comm.Reduce(a, ra_, op=MPI.SUM, root=r)
            comm.Reduce(w, rw_, op=MPI.SUM, root=r)
            if comm.rank == r:
                out.amplitude[:].local_array[:] = ra_
                out.weight[:].local_array[:] = rw_
        self.log.info(
            f"Extracted the cutouts of {ncut} candidates in {', '.join(pnames)}."
        )
        return out



class ExtractTrialCutouts(tasklib.base.ContainerTask):
    """Windows of the calibrated trials of the matched filter around candidates.

    For the candidates with S/N of at least ``snr_min`` (and only those kept by the
    sifter if ``keep_only`` and the table has been sifted), the brightest
    ``max_cutouts`` of them: the amplitude and weight (and imaginary part) of every
    polarisation and duration of the trials over +-``nrow`` rows and +-``nra`` start
    times, at the candidate's telescope-x and spectral index.  A polarisation with
    no weight at that spectral index (L, searched only at ``rm_spectral_indices``)
    is taken at the nearest one it has.  These are what the search saw at each
    single trial, without the tail of the best-trial maps to earlier starts (a
    window that starts before a pulse and ends on it).  The result is distributed
    over candidates.

    Attributes
    ----------
    snr_min : float
        Smallest S/N.  Default is 10.
    keep_only : bool
        Only the candidates kept by the sifter.  Default is False.
    max_cutouts : int
        Largest number of cutouts (the brightest).  Default is 1000.
    nrow, nra : int
        Half-widths of the window in rows and samples.  Defaults are 24 and 40.
    """

    snr_min = config.Property(proptype=float, default=10.0)
    keep_only = config.Property(proptype=bool, default=False)
    max_cutouts = config.Property(proptype=int, default=1000)
    nrow = config.Property(proptype=int, default=24)
    nra = config.Property(proptype=int, default=40)

    def process(self, table, trials):
        """Extract the cutouts.

        Parameters
        ----------
        table : containers.TransientCandidateTable
            Candidates (sifted or not).
        trials : containers.TransientMatchedFilterTrials
            The trials of the matched filter (after CalibrateTrialNoise).

        Returns
        -------
        out : containers.TransientCandidateTrialCutouts
        """
        from mpi4py import MPI

        trials.redistribute("el")
        comm = trials.comm
        rec = table.candidate[:]
        sel = rec["snr"] >= self.snr_min
        if self.keep_only and "sift" in table.datasets:
            sel &= table.datasets["sift"][:]["keep"]
        idx = np.flatnonzero(sel)
        idx = idx[np.argsort(rec["snr"][idx], kind="stable")[::-1]][: self.max_cutouts]
        ncut = idx.size

        el = trials.index_map["el"][:]
        ra = trials.index_map["ra"][:]
        nra_map = ra.size
        dra = np.median(np.diff(ra))
        periodic = _is_full_day(ra)
        pol = _decode_names(trials.index_map["pol"][:])
        alphas = np.asarray(trials.index_map["spectral_index"][:], dtype=np.float64)
        durations = trials.index_map["duration"][:]
        e = rec["el_index"][idx].astype(int)
        tb = np.clip(np.searchsorted(ra, rec["ra"][idx] - 0.5 * dra), 0, nra_map - 1)
        xi = rec["x_index"][idx].astype(int)

        # The spectral index of each candidate and polarisation: its own, or the
        # nearest one at which the polarisation was searched (L)
        searched = np.ones((len(pol), alphas.size), dtype=bool)
        if "L" in pol and "rm_spectral_indices" in trials.attrs:
            rm_a = np.atleast_1d(trials.attrs["rm_spectral_indices"])
            searched[pol.index("L")] = np.any(
                np.isclose(alphas[:, np.newaxis], rm_a[np.newaxis]), axis=1
            )
        cand_a = np.nan_to_num(rec["spectral_index"][idx]) if ncut else np.zeros(0)
        ia = np.zeros((ncut, len(pol)), dtype=int)
        for pp in range(len(pol)):
            dist = np.abs(cand_a[:, np.newaxis] - alphas[np.newaxis])
            dist = np.where(searched[pp][np.newaxis], dist, np.inf)
            ia[:, pp] = np.argmin(dist, axis=1)

        roff = np.arange(-self.nrow, self.nrow + 1)
        toff = np.arange(-self.nra, self.nra + 1)
        has_imag = "amplitude_imag" in trials.datasets
        out = containers.TransientCandidateTrialCutouts(
            candidate=ncut,
            pol=np.array(pol),
            duration=durations,
            el_offset=roff,
            ra_offset=toff,
            attrs_from=table,
            distributed=True,
            comm=comm,
        )
        if has_imag:
            out.add_dataset("amplitude_imag")
        out.datasets["candidate_index"][:] = idx
        out.datasets["x_index"][:] = xi
        out.datasets["spectral_index"][:] = alphas[ia]
        rows = e[:, np.newaxis] + roff[np.newaxis]
        rin = (rows >= 0) & (rows < el.size)
        out.datasets["el_index"][:] = np.where(rin, rows, -1)
        out.datasets["el"][:] = _extend_axis(el, rows)
        ts = tb[:, np.newaxis] + toff[np.newaxis]
        if periodic:
            tin = np.ones(ts.shape, dtype=bool)
            ts = ts % nra_map
        else:
            tin = (ts >= 0) & (ts < nra_map)
        out.datasets["ra_index"][:] = np.where(tin, ts, -1)
        out.datasets["ra"][:] = _extend_axis(ra, ts)

        # Each rank's block of candidates, summed from the rows on every rank
        names = ["amplitude", "weight"] + (["amplitude_imag"] if has_imag else [])
        src = [trials.datasets[n][:] for n in names]
        off, nloc = src[0].local_offset[3], src[0].local_shape[3]
        la = out.amplitude[:]
        blocks = comm.allgather((la.local_offset[0], la.local_shape[0]))
        tsafe = np.where(tin, ts, 0)
        for r, (o, nl) in enumerate(blocks):
            shape = (len(names), nl, len(pol), durations.size, roff.size, toff.size)
            buf = np.zeros(shape, dtype=np.float32)
            for j in range(nl):
                k = o + j
                for ir in range(roff.size):
                    er = rows[k, ir]
                    if not rin[k, ir] or er < off or er >= off + nloc:
                        continue
                    for pp in range(len(pol)):
                        for n, s in enumerate(src):
                            v = s.local_array[pp, :, ia[k, pp], er - off]  # dur, ra, x
                            v = v[:, tsafe[k], xi[k]]
                            buf[n, j, pp, :, ir] = np.where(tin[k], v, 0.0)
            red = np.zeros_like(buf) if comm.rank == r else None
            comm.Reduce(buf, red, op=MPI.SUM, root=r)
            if comm.rank == r:
                for n, name in enumerate(names):
                    out.datasets[name][:].local_array[:] = red[n]
        self.log.info(f"Extracted the trial cutouts of {ncut} candidates.")
        return out


class ExtractForegroundCutouts(tasklib.base.ContainerTask):
    """Windows of the unfiltered data around candidates: the foreground sky.

    For the candidates with S/N of at least ``snr_min`` (and only those kept by the
    sifter if ``keep_only`` and the table has been sifted), the brightest
    ``max_cutouts`` of them: the hybrid visibilities without the m-mode filter,
    phased to the candidate's telescope-x (each baseline and channel multiplied by
    ``exp(-2 pi i b x / lambda)``), and their real part averaged with the weights
    over the co-polar products, the baselines and the channels of each band of
    ``freq_edges``, over +-``nrow`` rows and +-``nra`` samples around the start.
    That is a beam of the sky at that telescope-x as it drifts past, in the units of
    the data and without the RFI mask: the complement of what the search sees, so a
    bright source (in a catalogue or not) under a candidate shows.  The weight is
    the sum of the weights of the visibilities.  The result is distributed over
    candidates.

    The stream is used as it is distributed (over elevation, as the matched filter
    leaves it), so the windows are summed from the rows of every rank.

    Attributes
    ----------
    snr_min : float
        Smallest S/N.  Default is 10.
    keep_only : bool
        Only the candidates kept by the sifter.  Default is False.
    max_cutouts : int
        Largest number of cutouts (the brightest).  Default is 1000.
    nrow, nra : int
        Half-widths of the window in rows and samples.  Defaults are 40 and 180.
    freq_edges : list of float
        Edges of the bands in MHz.  Default is [400, 500, 600, 700, 800].
    """

    snr_min = config.Property(proptype=float, default=10.0)
    keep_only = config.Property(proptype=bool, default=False)
    max_cutouts = config.Property(proptype=int, default=1000)
    nrow = config.Property(proptype=int, default=40)
    nra = config.Property(proptype=int, default=180)
    freq_edges = config.list_type(type_=float, default=[400.0, 500.0, 600.0, 700.0, 800.0])

    def process(self, table, stream):
        """Extract the cutouts.

        Parameters
        ----------
        table : containers.TransientCandidateTable
            Candidates (sifted or not), with ``el_index`` on the rows of the stream.
        stream : draco.core.containers.HybridVisStream
            The hybrid visibilities without the m-mode filter.

        Returns
        -------
        out : containers.TransientCandidateCutouts
            Polarisation "I"; ``freq`` holds the bands.
        """
        from mpi4py import MPI

        comm = stream.comm
        rec = table.candidate[:]
        sel = rec["snr"] >= self.snr_min
        if self.keep_only and "sift" in table.datasets:
            sel &= table.datasets["sift"][:]["keep"]
        idx = np.flatnonzero(sel)
        idx = idx[np.argsort(rec["snr"][idx], kind="stable")[::-1]][: self.max_cutouts]
        ncut = idx.size

        el = stream.index_map["el"][:]
        ra = stream.ra[:]
        nra = ra.size
        dra = np.median(np.diff(ra))
        periodic = _is_full_day(ra) and bool(stream.attrs.get("ra_periodic", True))
        freq = stream.freq[:]
        ew = stream.index_map["ew"][:]
        pol = _decode_names(stream.index_map["pol"][:])
        ico = [ii for ii, p in enumerate(pol) if p[0] == p[1]]

        edges = np.asarray(self.freq_edges, dtype=float)
        gidx = np.searchsorted(edges, freq, side="right") - 1
        gidx[(freq < edges[0]) | (freq > edges[-1])] = -1
        gidx[freq == edges[-1]] = edges.size - 2
        band_map = np.zeros(edges.size - 1, dtype=[("centre", np.float64), ("width", np.float64)])
        band_map["centre"] = 0.5 * (edges[1:] + edges[:-1])
        band_map["width"] = np.diff(edges)

        e = rec["el_index"][idx].astype(int)
        tb = np.clip(np.searchsorted(ra, rec["ra"][idx] - 0.5 * dra), 0, nra - 1)
        xc = rec["x"][idx]
        roff = np.arange(-self.nrow, self.nrow + 1)
        toff = np.arange(-self.nra, self.nra + 1)

        out = containers.TransientCandidateCutouts(
            candidate=ncut,
            pol=np.array(["I"]),
            freq=band_map,
            el_offset=roff,
            ra_offset=toff,
            attrs_from=table,
            distributed=True,
            comm=comm,
        )
        out.attrs["foreground"] = True
        out.attrs["freq_edges"] = edges
        out.datasets["candidate_index"][:] = idx
        out.datasets["x_index"][:] = rec["x_index"][idx]
        out.datasets["duration"][:] = 1
        rows = e[:, np.newaxis] + roff[np.newaxis]
        rin = (rows >= 0) & (rows < el.size)
        out.datasets["el_index"][:] = np.where(rin, rows, -1)
        out.datasets["el"][:] = _extend_axis(el, rows)
        ts = tb[:, np.newaxis] + toff[np.newaxis]
        if periodic:
            tin = np.ones(ts.shape, dtype=bool)
            ts = ts % nra
        else:
            tin = (ts >= 0) & (ts < nra)
        out.datasets["ra_index"][:] = np.where(tin, ts, -1)
        out.datasets["ra"][:] = _extend_axis(ra, ts)

        # The rows of this rank, and the weights of every frequency on each node
        stream.redistribute(["el", "freq"])
        el_slc = stream.vis[:].local_bounds
        uvis = stream.vis[:].local_array  # pol, freq, ew, el, ra
        wshared = _NodeSharedFreq(stream.weight[:], comm)  # freq, pol, ew, ra
        wall = wshared.array
        lmbda = scipy.constants.c / (freq * 1e6)
        ngroup = edges.size - 1

        def block(k):
            """This rank's sums over its rows for the candidates ``k``."""
            num = np.zeros((k.size, ngroup, roff.size, toff.size), dtype=np.float64)
            den = np.zeros_like(num)
            for jj, kk in enumerate(k):
                rr = rows[kk]
                local = rin[kk] & (rr >= el_slc.start) & (rr < el_slc.stop)
                if not local.any():
                    continue
                # The rows are contiguous: cut out the window before selecting anything
                lr = rr[local] - el_slc.start
                tt = np.where(tin[kk], ts[kk], 0)
                vw = np.take(uvis[:, :, :, lr[0] : lr[-1] + 1], tt, axis=-1)[ico]  # pol, freq, ew, row, ra
                ww = np.moveaxis(np.take(wall, tt, axis=-1)[:, ico], 0, 1)  # pol, freq, ew, ra
                ww = np.where(tin[kk], ww, 0.0)
                ph = np.exp(-2.0j * np.pi * ew[np.newaxis, :] * xc[kk] / lmbda[:, np.newaxis])  # freq, ew
                for gg in range(ngroup):
                    ff = np.flatnonzero(gidx == gg)
                    if ff.size == 0:
                        continue
                    re = (vw[:, ff] * ph[ff][np.newaxis, :, :, np.newaxis, np.newaxis]).real
                    w = ww[:, ff]
                    num[jj, gg, local] = np.einsum("pfbrt,pfbt->rt", re, w)
                    den[jj, gg, local] = np.sum(w, axis=(0, 1, 2))[np.newaxis]
            return num, den

        # Each rank's block of candidates, summed from the rows on every rank
        la = out.amplitude[:]
        offs = comm.allgather((la.local_offset[0], la.local_shape[0]))
        for r, (o, nl) in enumerate(offs):
            num, den = block(np.arange(o, o + nl))
            rnum = np.zeros_like(num) if comm.rank == r else None
            rden = np.zeros_like(den) if comm.rank == r else None
            comm.Reduce(num, rnum, op=MPI.SUM, root=r)
            comm.Reduce(den, rden, op=MPI.SUM, root=r)
            if comm.rank == r:
                out.amplitude[:].local_array[:, 0] = (rnum * tools.invert_no_zero(rden)).astype(np.float32)
                out.weight[:].local_array[:, 0] = rden.astype(np.float32)
        wshared.free()
        self.log.info(
            f"Extracted the foreground cutouts of {ncut} candidates in {ngroup} bands."
        )
        return out


class AddTemplateFlux(tasklib.base.ContainerTask):
    """Add the template of compact sources at each candidate.

    The template (configs/lpt_template.yaml) is the single-sample, flat-spectrum
    amplitude of the matched filter on the catalogue sources alone, simulated with
    the injection tasks on a fixed sidereal grid
    (:class:`ch_pipeline.synthesis.inject.MakeTemplateStream`): a
    :class:`ch_pipeline.core.containers.TransientMatchedFilterTrials` with one
    polarisation, duration and spectral index.  It serves a year of days.  The
    column ``tmpl_flux`` (Jy) of the candidate table is its amplitude at the
    nearest row, telescope-x and RA sample to the candidate's start (the meridian
    RA, ``ra``).  It has the negative east-west sidelobes of the sources, where a
    source that fades gives a positive candidate, and their grating lobes.  The
    sifter's ``template`` flag compares it with the amplitude.
    """

    def setup(self, template):
        """Keep the template.

        Parameters
        ----------
        template : containers.TransientMatchedFilterTrials
        """
        template.redistribute("el")
        amp = template.amplitude[:]
        if amp.shape[:3] != (1, 1, 1):
            self.log.warning(
                f"The template has {amp.shape[:3]} polarisations, durations and spectral "
                "indices; using the first of each."
            )
        self.el = template.index_map["el"][:]
        self.x = template.index_map["x"][:]
        self.ra = template.ra[:]
        self.r0 = amp.local_offset[3]
        self.local = np.asarray(amp.local_array[0, 0, 0])  # el, ra, x
        self.attrs = {"template_lsd": template.attrs.get("lsd", np.nan),
                      "template_tag": template.attrs.get("tag", "")}

    def process(self, table):
        """Look the candidates up.

        Parameters
        ----------
        table : containers.TransientCandidateTable

        Returns
        -------
        table : containers.TransientCandidateTable
            A copy with ``tmpl_flux`` filled.
        """
        from mpi4py import MPI

        rec = table.candidate[:].copy()
        n = rec.size
        dra = 360.0 / self.ra.size
        if n:
            ie = np.argmin(np.abs(rec["y"][:, np.newaxis] - self.el[np.newaxis, :]), axis=1)
            ix = np.argmin(np.abs(rec["x"][:, np.newaxis] - self.x[np.newaxis, :]), axis=1)
            it = np.rint(((rec["ra"] - self.ra[0]) % 360.0) / dra).astype(int) % self.ra.size
        else:
            ie = ix = it = np.zeros(0, dtype=int)
        vals = np.zeros(n)
        mine = (ie >= self.r0) & (ie < self.r0 + self.local.shape[0])
        k = np.flatnonzero(mine)
        vals[k] = self.local[ie[k] - self.r0, it[k], ix[k]]
        self.comm.Allreduce(MPI.IN_PLACE, vals, op=MPI.SUM)
        rec["tmpl_flux"] = vals
        out = table.copy()
        out.candidate[:] = rec
        for key, val in self.attrs.items():
            out.attrs[key] = val
        return out


class TransientNegativePeaks(tasklib.base.ContainerTask):
    """The peaks of the negative search: a null for the significance of candidates.

    Noise, and the scintillation of the sky through the ionosphere, are two-sided,
    while a pulse is positive.  So the peaks of the map of the most negative trial
    (:class:`MaxMatchedFilterTrials` with ``negate``), found exactly as the
    candidates are (:class:`GroupTransientCandidates`: local maxima above
    ``seed_snr`` within +-``peak_dy`` and +-``peak_dt``, without the ones whose
    trial overlaps that of a brighter peak), are what the noise alone makes, with
    its real (heavy) tails, at each place and duration.

    The matched filter puts negative lobes next to every positive excursion (on its
    rows, one to two durations before and after it, at 0.3-0.5 of its peak), which
    are not noise.  A negative peak is marked ``bowl`` (``fidelity`` table) if a
    positive peak of the map of the best trial within +-``bowl_dy`` in telescope-y
    and +-(``bowl_dt_scale`` x the longer of their durations + ``bowl_dt_pad``)
    samples has at least 1 / ``bowl_frac`` times its S/N.

    The output is a candidate table with the position and best trial of each
    negative peak (``el_index``, ``ra_index``, ``x_index``, ``y``, ``x``, ``ra``,
    ``snr`` as a positive number, ``amplitude``, ``duration`` in s, and
    ``weight_ratio``); the other columns are left empty.

    Attributes
    ----------
    pol : int
        Index of the polarisation of the maps.  Default is 0.
    seed_snr, peak_dy, peak_dt : float, float, int
        As for :class:`GroupTransientCandidates` (defaults 8, 0.02 and 3), which
        must be the same for the null to match the candidates.
    bowl_dy : float
        Half-width in telescope-y of the region of a positive peak's lobes.
        Default is 0.01.
    bowl_dt_scale, bowl_dt_pad : float, int
        Half-width in samples of that region, in units of the duration, and
        added.  Defaults are 2 and 4.
    bowl_frac : float
        Largest S/N of a lobe as a fraction of its positive peak.  Default is 0.6.
    """

    pol = config.Property(proptype=int, default=0)
    seed_snr = config.Property(proptype=float, default=8.0)
    peak_dy = config.Property(proptype=float, default=0.02)
    peak_dt = config.Property(proptype=int, default=3)
    bowl_dy = config.Property(proptype=float, default=0.01)
    bowl_dt_scale = config.Property(proptype=float, default=2.0)
    bowl_dt_pad = config.Property(proptype=int, default=4)
    bowl_frac = config.Property(proptype=float, default=0.6)

    def process(self, neg, cand):
        """Find the negative peaks.

        Parameters
        ----------
        neg : containers.TransientCandidateMap
            The most negative trial at each elevation and start time (negated).
        cand : containers.TransientCandidateMap
            The best trial, on the same axes.

        Returns
        -------
        out : containers.TransientCandidateTable
            One row per negative peak, with the ``fidelity`` table (``bowl``).
        """
        import warnings

        if not neg.attrs.get("negated", False):
            raise ValueError("The first input must be a negated map (negate=True).")
        for cont in (neg, cand):
            cont.redistribute("el")
        el = neg.index_map["el"][:]
        ra = neg.index_map["ra"][:]
        if cand.index_map["el"].size != el.size or not np.allclose(cand.index_map["ra"][:], ra):
            raise ValueError("The two maps must have the same elevations and start times.")
        pp = self.pol
        dy = np.median(np.diff(el))
        dra = np.median(np.diff(ra))
        tsamp = float(neg.attrs.get("duration_unit_seconds", dra * 240.0 * 0.99727))
        de = max(1, round(self.peak_dy / dy))

        def peaks(m):
            snr = m.snr[:].allgather()[pp].astype(np.float32)
            wgt = m.weight[:].allgather()[pp]
            valid = (wgt > 0) & np.isfinite(snr)
            img = np.where(valid, snr, 0.0).astype(np.float32)
            dur = m.datasets["duration"][:].allgather()[pp]
            dur = np.where(valid, np.nan_to_num(dur), 1).astype(int)
            pe, pt = _local_peaks(img, de, self.peak_dt, self.seed_snr)
            pe, pt = _merge_overlapping(pe, pt, dur, de, self.peak_dt)
            return pe, pt, img[pe, pt], np.maximum(dur[pe, pt], 1), wgt

        ne, nt, nsnr, ndur, wgt = peaks(neg)
        qe, qt, qsnr, qdur, _ = peaks(cand)

        # The lobes of brighter positive peaks
        bowl = np.zeros(ne.size, dtype=bool)
        nb = max(0, round(self.bowl_dy / dy))
        order = np.argsort(qe, kind="stable")
        qe, qt, qsnr, qdur = qe[order], qt[order], qsnr[order], qdur[order]
        lo = np.searchsorted(qe, ne - nb, side="left")
        hi = np.searchsorted(qe, ne + nb, side="right")
        for k in range(ne.size):
            j = np.arange(lo[k], hi[k])
            if j.size == 0:
                continue
            reach = self.bowl_dt_scale * np.maximum(qdur[j], ndur[k]) + self.bowl_dt_pad
            near = np.abs(qt[j] - nt[k]) <= reach
            bowl[k] = np.any(near & (qsnr[j] * self.bowl_frac >= nsnr[k]))

        out = containers.TransientCandidateTable(
            candidate=ne.size,
            group=np.array(["all"]),
            freq=np.zeros(1, dtype=[("centre", np.float64), ("width", np.float64)]),
            attrs_from=neg,
            comm=neg.comm,
        )
        rec = np.zeros(ne.size, dtype=out.candidate[:].dtype)
        for name in rec.dtype.names:
            if rec.dtype[name].kind == "f":
                rec[name] = np.nan
        rec["band_nsig"] = -1
        if ne.size:
            xidx = neg.datasets["x_index"][:].allgather()[pp]
            amp = neg.amplitude[:].allgather()[pp]
            with warnings.catch_warnings():  # rows with no weight
                warnings.simplefilter("ignore", RuntimeWarning)
                wrow = np.nanmedian(np.where(wgt > 0, wgt, np.nan), axis=1)
            ix = xidx[ne, nt].astype(int)
            rec["el_index"] = ne
            rec["ra_index"] = nt
            rec["x_index"] = ix
            rec["y"] = el[ne]
            rec["x"] = neg.index_map["x"][:][ix]
            rec["ra"] = ra[nt]
            rec["snr"] = nsnr
            rec["amplitude"] = amp[ne, nt]
            rec["duration"] = ndur * tsamp
            rec["weight_ratio"] = wgt[ne, nt] * tools.invert_no_zero(np.nan_to_num(wrow[ne]))
        out.candidate[:] = rec
        out.add_dataset("fidelity")
        fid = np.zeros(ne.size, dtype=out.datasets["fidelity"][:].dtype)
        fid["bowl"] = bowl
        out.datasets["fidelity"][:] = fid
        out.attrs["pol"] = _decode_names(neg.index_map["pol"][:])[pp]
        out.attrs["negative"] = True
        for key in (
            "seed_snr", "peak_dy", "peak_dt", "bowl_dy", "bowl_dt_scale", "bowl_dt_pad",
            "bowl_frac",
        ):  # fmt: skip
            out.attrs[key] = getattr(self, key)
        self.log.info(
            f"Found {ne.size} negative peaks with S/N >= {self.seed_snr:g} "
            f"({int(bowl.sum())} on the lobes of positive peaks; positive peaks {qe.size})."
        )
        return out


class AddNegativeFidelity(tasklib.base.ContainerTask):
    """Compare each candidate with the negative peaks around it.

    For each candidate, ``neg_n`` is the number of negative peaks (from
    :class:`TransientNegativePeaks`, not on the lobes of positive peaks) and
    ``pos_n`` that of candidates (itself included) with at least its S/N, within
    +-``fid_dy`` in telescope-y and +-``fid_dra`` degrees of start RA and with a
    duration within a factor ``fid_dur_ratio`` of its own.  The fidelity,
    ``1 - neg_n / pos_n`` (at least 0), estimates the probability that a candidate
    of that S/N there is not noise, from the noise's real tail rather than a
    Gaussian one (as the fidelity of blind line searches with ALMA, from their
    negative peaks).  ``neg_max`` is the S/N of the brightest negative peak in the
    region at that duration (0 if none).  The results go into the ``fidelity``
    table.

    Attributes
    ----------
    fid_dy : float
        Half-width of the region in telescope-y.  Default is 0.1.
    fid_dra : float
        Half-width of the region in degrees of RA.  Default is 10.
    fid_dur_ratio : float
        Largest ratio of durations.  Default is 2.
    """

    fid_dy = config.Property(proptype=float, default=0.1)
    fid_dra = config.Property(proptype=float, default=10.0)
    fid_dur_ratio = config.Property(proptype=float, default=2.0)

    def process(self, table, negatives):
        """Add the fidelity.

        Parameters
        ----------
        table : containers.TransientCandidateTable
            Candidates.
        negatives : containers.TransientCandidateTable
            Negative peaks, from :class:`TransientNegativePeaks` on the same maps.

        Returns
        -------
        out : containers.TransientCandidateTable
            A copy of the input, with the ``fidelity`` table.
        """
        if not negatives.attrs.get("negative", False):
            raise ValueError("The second input must come from TransientNegativePeaks.")
        rec = table.candidate[:]
        nrec = negatives.candidate[:]
        use = ~negatives.datasets["fidelity"][:]["bowl"]
        neg_n, pos_n, neg_max = _fidelity_counts(
            rec, nrec[use], self.fid_dy, self.fid_dra, self.fid_dur_ratio
        )
        out = _table_copy(table, skip=("fidelity",))
        out.add_dataset("fidelity")
        fid = np.zeros(rec.size, dtype=out.datasets["fidelity"][:].dtype)
        fid["neg_n"] = neg_n
        fid["pos_n"] = pos_n
        fid["fidelity"] = np.clip(
            1.0 - neg_n * tools.invert_no_zero(pos_n.astype(float)), 0.0, 1.0
        )
        fid["neg_max"] = neg_max
        out.datasets["fidelity"][:] = fid
        for key in ("fid_dy", "fid_dra", "fid_dur_ratio"):
            out.attrs[key] = getattr(self, key)
        out.attrs["fid_nneg"] = int(use.sum())
        self.log.info(
            f"Fidelity of {rec.size} candidates from {int(use.sum())} negative peaks: "
            f"{int(np.sum(fid['fidelity'] < 0.5))} below 0.5."
        )
        return out


def _fidelity_counts(rec, nrec, dy, dra, dur_ratio):
    """Counts of negative (``nrec``) and positive (``rec``) peaks around each of ``rec``.

    Returns neg_n, pos_n (each the number with at least the candidate's S/N within
    +-``dy`` in y, +-``dra`` degrees of RA and durations within a factor
    ``dur_ratio``) and the largest S/N of the negative peaks there.
    """
    n = rec.size
    neg_n = np.zeros(n, dtype=np.int32)
    pos_n = np.zeros(n, dtype=np.int32)
    neg_max = np.zeros(n, dtype=np.float32)
    lr = np.log(dur_ratio)

    def window(src, k):
        lo = np.searchsorted(src["ys"], rec["y"][k] - dy, side="left")
        hi = np.searchsorted(src["ys"], rec["y"][k] + dy, side="right")
        idx = src["order"][lo:hi]
        r = src["rec"]
        ok = np.abs(_wrap_deg(r["ra"][idx] - rec["ra"][k])) <= dra
        ok &= np.abs(np.log(np.maximum(r["duration"][idx], 1e-3) / max(rec["duration"][k], 1e-3))) <= lr
        return idx[ok]

    srcs = []
    for r in (nrec, rec):
        order = np.argsort(r["y"], kind="stable")
        srcs.append({"rec": r, "order": order, "ys": r["y"][order]})
    for k in range(n):
        s = rec["snr"][k]
        jn = window(srcs[0], k)
        jp = window(srcs[1], k)
        neg_n[k] = np.sum(nrec["snr"][jn] >= s)
        pos_n[k] = np.sum(rec["snr"][jp] >= s)
        neg_max[k] = nrec["snr"][jn].max() if jn.size else 0.0
    return neg_n, pos_n, neg_max


def _table_copy(table, skip=()):
    """A new candidate table with the datasets of ``table`` (except ``skip``).

    A table read from a file cannot take a new table dataset, so tasks that add one
    return a copy.
    """
    rec = table.candidate[:]
    out = containers.TransientCandidateTable(
        candidate=rec.size, axes_from=table, attrs_from=table, comm=table.comm
    )
    for name in table.datasets:
        if name in skip:
            continue
        if name not in out.datasets:
            out.add_dataset(name)
        out.datasets[name][:] = table.datasets[name][:]
    return out


class CalibrateSignificance(tasklib.base.ContainerTask):
    """The S/N of each candidate, calibrated against the local tail of the noise.

    The noise model sets the variance of the matched filter, but the ionospheric
    scintillation of the sky has heavier tails than a Gaussian of that variance, and
    more so where the sky dominates the variance and at night.  The negative search
    (:class:`TransientNegativePeaks`) shows that tail at each place, since the noise
    and the scintillation are two-sided while a pulse is positive.  For each
    candidate:

    - ``sky_q``, the sky-to-thermal variance ratio at its position: from the pixels
      of the best-trial map whose best trial is a single sample, the row's median
      weight over the weight, minus 1, as a median in blocks of ``q_block_e`` rows
      by ``q_block_t`` samples.
    - The null: the negative peaks that are not on the negative lobe of a positive
      peak (a candidate of the table) at least as bright, within +-``lobe_dy`` in
      telescope-y and +-(``lobe_dt_scale`` x the longer duration + ``lobe_dt_pad``)
      samples.  These negative-dominated structures are as common as positive-
      dominated ones if the fluctuations are symmetric.  (The matched filter puts
      lobes of ~0.8 of a pulse's peak beside it in the best-trial maps.)
    - The tail: for each class of ``sky_q`` (bins ``q_edges``) and duration (bins
      ``duration_edges``, in samples), the rate per pixel of null peaks above
      ``snr_floor`` and a generalised Pareto fit to their excess S/N (pooled with
      the neighbouring classes of ``sky_q`` until there are ``nmin_fit`` peaks).
    - ``local_ratio``: the null peaks of the candidate's duration class above
      ``snr_floor`` within +-``local_dy`` and +-``local_dra`` degrees of RA, over the
      number the class rates expect from the pixels there, as (n + ``shrink``) /
      (n_expected + ``shrink``).  It follows the night and the clouds.
    - ``far``: local_ratio x rate x survival(S/N) x the number of pixels, the expected
      number of noise peaks a day at least as bright if the whole sky were like the
      candidate's neighbourhood.
    - ``snr_cal``: the S/N with the same rate in the reference class (single samples,
      the first bin of ``sky_q``, local ratio 1).  Below ``snr_floor`` it is
      extrapolated with the reference tail's exponential scale.

    The fits go into the attribute ``significance_fits`` (JSON).

    Attributes
    ----------
    pol : int
        Index of the polarisation of the map.  Default is 0.
    snr_floor : float
        Threshold of the tail fits.  Default is 8 (the seed of the candidates).
    q_edges : list of float
        Edges of the classes of ``sky_q``.  Default is [0.25, 1, 3, 10, 30].
    duration_edges : list of int
        Edges of the classes of duration (samples).  Default is [2, 5, 10]: 1, 2-4,
        5-9 and 10 or more.
    q_block_e, q_block_t : int
        Size of the blocks of the ``sky_q`` map.  Defaults are 4 and 8.
    lobe_dy, lobe_dt_scale, lobe_dt_pad : float, float, int
        The lobe window.  Defaults are 0.01, 2 and 4.
    local_dy, local_dra : float
        The local region.  Defaults are 0.1 and 10.
    shrink : float
        Pseudo-count of the local ratio.  Default is 5.
    nmin_fit : int
        Least number of peaks for a tail fit.  Default is 40.
    """

    pol = config.Property(proptype=int, default=0)
    snr_floor = config.Property(proptype=float, default=8.0)
    q_edges = config.list_type(float, default=[0.25, 1.0, 3.0, 10.0, 30.0])
    duration_edges = config.list_type(int, default=[2, 5, 10])
    q_block_e = config.Property(proptype=int, default=4)
    q_block_t = config.Property(proptype=int, default=8)
    lobe_dy = config.Property(proptype=float, default=0.01)
    lobe_dt_scale = config.Property(proptype=float, default=2.0)
    lobe_dt_pad = config.Property(proptype=int, default=4)
    local_dy = config.Property(proptype=float, default=0.1)
    local_dra = config.Property(proptype=float, default=10.0)
    shrink = config.Property(proptype=float, default=5.0)
    nmin_fit = config.Property(proptype=int, default=40)

    def process(self, table, negatives, cand):
        """Calibrate the candidates.

        Parameters
        ----------
        table : containers.TransientCandidateTable
            Candidates, from the best-trial map ``cand``.
        negatives : containers.TransientCandidateTable
            The negative peaks of the same search (:class:`TransientNegativePeaks`).
        cand : containers.TransientCandidateMap
            The best-trial map of the candidates.

        Returns
        -------
        out : containers.TransientCandidateTable
            A copy of the table, with the ``significance`` table.
        """
        if not negatives.attrs.get("negative", False):
            raise ValueError("The second input must come from TransientNegativePeaks.")
        cand.redistribute("el")
        el = cand.index_map["el"][:]
        ra = cand.index_map["ra"][:]
        dy = np.median(np.diff(el))
        dra = np.median(np.diff(ra))
        tsamp = float(cand.attrs.get("duration_unit_seconds", dra * 240.0 * 0.99727))
        rec = table.candidate[:]
        nrec = negatives.candidate[:]

        qb, nvalid = self._q_blocks(cand)
        nqb = len(self.q_edges) + 1
        be, bt = self.q_block_e, self.q_block_t

        def q_at(r):
            i = np.clip(r["el_index"] // be, 0, qb.shape[0] - 1)
            j = np.clip(r["ra_index"] // bt, 0, qb.shape[1] - 1)
            return qb[i, j]

        def qclass(q):
            return np.searchsorted(self.q_edges, np.nan_to_num(q), side="right")

        def wclass(r):
            w = np.maximum(np.rint(np.nan_to_num(r["duration"]) / tsamp), 1)
            return np.searchsorted(self.duration_edges, w, side="right")

        nwc = len(self.duration_edges) + 1

        # The null: negative-dominated peaks above the floor
        null = nrec[_negative_dominant(nrec, rec, max(0, round(self.lobe_dy / dy)),
                                       self.lobe_dt_scale, self.lobe_dt_pad, tsamp)]
        null = null[null["snr"] >= self.snr_floor]
        null_q = qclass(q_at(null))
        null_w = wclass(null)

        # Pixels per class of q (per block, for the local sums)
        bcls = np.where(nvalid > 0, qclass(qb), -1)
        npix = np.array([nvalid[bcls == b].sum() for b in range(nqb)])
        ntot = nvalid.sum()
        cum = np.zeros((nqb, qb.shape[0] + 1, qb.shape[1] + 1))
        for b in range(nqb):
            cum[b, 1:, 1:] = np.where(bcls == b, nvalid, 0).cumsum(0).cumsum(1)

        # Tail fits per class
        rate = np.zeros((nwc, nqb))
        shape = np.zeros((nwc, nqb, 2))  # xi, sigma
        nused = np.zeros((nwc, nqb), dtype=int)
        for wk in range(nwc):
            for b in range(nqb):
                lo = hi = b
                sel = (null_w == wk) & (null_q == b)
                while sel.sum() < self.nmin_fit and (lo > 0 or hi < nqb - 1):
                    lo, hi = max(lo - 1, 0), min(hi + 1, nqb - 1)
                    sel = (null_w == wk) & (null_q >= lo) & (null_q <= hi)
                shape[wk, b] = _gpd_fit(null["snr"][sel] - self.snr_floor)
                nused[wk, b] = sel.sum()
                rate[wk, b] = np.sum((null_w == wk) & (null_q == b)) / max(npix[b], 1)

        # Each candidate
        q = q_at(rec)
        cq, cw = qclass(q), wclass(rec)
        order = np.argsort(null["y"], kind="stable")
        ys = null["y"][order]
        half_e = max(0, round(self.local_dy / dy))
        half_t = max(0, round(self.local_dra / dra))
        n = rec.size
        local = np.ones(n)
        far = np.zeros(n)
        snr_cal = np.zeros(n)
        xi_r, sig_r = shape[0, 0]
        for k in range(n):
            wk = cw[k]
            # expected null peaks in the region, from the class rates
            e0 = max(rec["el_index"][k] - half_e, 0) // be
            e1 = min(rec["el_index"][k] + half_e, el.size - 1) // be + 1
            exp = 0.0
            for t0, t1 in _wrapped_ranges(
                rec["ra_index"][k] - half_t, rec["ra_index"][k] + half_t + 1, ra.size
            ):
                b0, b1 = t0 // bt, min(-(-t1 // bt), qb.shape[1])
                cnt = cum[:, e1, b1] - cum[:, e0, b1] - cum[:, e1, b0] + cum[:, e0, b0]
                exp += np.sum(cnt * rate[wk])
            lo = np.searchsorted(ys, rec["y"][k] - self.local_dy, side="left")
            hi = np.searchsorted(ys, rec["y"][k] + self.local_dy, side="right")
            idx = order[lo:hi]
            near = (np.abs(_wrap_deg(null["ra"][idx] - rec["ra"][k])) <= self.local_dra) & (
                null_w[idx] == wk
            )
            local[k] = (near.sum() + self.shrink) / (exp + self.shrink)
            xi, sig = shape[wk, cq[k]]
            lam = local[k] * rate[wk, cq[k]] * _gpd_sf(rec["snr"][k] - self.snr_floor, xi, sig)
            far[k] = lam * ntot
            p = lam / rate[0, 0] if rate[0, 0] > 0 else np.inf
            if p >= 1.0 or not np.isfinite(p):
                snr_cal[k] = self.snr_floor - sig_r * np.log(p) if np.isfinite(p) else 0.0
            else:
                snr_cal[k] = self.snr_floor + _gpd_isf(p, xi_r, sig_r)

        out = _table_copy(table, skip=("significance",))
        out.add_dataset("significance")
        sg = np.zeros(n, dtype=out.datasets["significance"][:].dtype)
        sg["sky_q"] = q
        sg["local_ratio"] = local
        sg["far"] = far
        sg["snr_cal"] = snr_cal
        out.datasets["significance"][:] = sg
        out.attrs["significance_fits"] = json.dumps(
            {
                "rate_per_pixel": rate.tolist(),
                "xi": shape[..., 0].tolist(),
                "sigma": shape[..., 1].tolist(),
                "npeak": nused.tolist(),
                "npix": npix.tolist(),
                "nnull": int(null.size),
            }
        )
        for key in (
            "snr_floor", "q_edges", "duration_edges", "q_block_e", "q_block_t", "lobe_dy",
            "lobe_dt_scale", "lobe_dt_pad", "local_dy", "local_dra", "shrink", "nmin_fit",
        ):  # fmt: skip
            out.attrs[f"sig_{key}"] = getattr(self, key)
        self.log.info(
            f"Calibrated {n} candidates against {null.size} negative-dominated peaks: "
            f"{int(np.sum(snr_cal >= 10))} at snr_cal >= 10 "
            f"({int(np.sum(rec['snr'] >= 10))} at S/N >= 10)."
        )
        return out

    def _q_blocks(self, cand):
        """sky_q per block of the map, and the number of valid pixels per block."""
        import warnings

        pp = self.pol
        w = cand.weight[:].allgather()[pp]
        snr = cand.snr[:].allgather()[pp]
        dur = cand.datasets["duration"][:].allgather()[pp]
        ok = (w > 0) & np.isfinite(snr)
        one = ok & (dur == 1)
        be, bt = self.q_block_e, self.q_block_t
        ne, nt = w.shape
        nbe, nbt = -(-ne // be), -(-nt // bt)
        pad = ((0, nbe * be - ne), (0, nbt * bt - nt))
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            med = np.nanmedian(np.where(one, w, np.nan), axis=1)
            q1 = np.where(one, med[:, np.newaxis] * tools.invert_no_zero(w) - 1.0, np.nan)
            q1 = np.pad(q1, pad, constant_values=np.nan)
            qb = np.nanmedian(q1.reshape(nbe, be, nbt, bt), axis=(1, 3))
        nvalid = np.pad(ok, pad).reshape(nbe, be, nbt, bt).sum(axis=(1, 3))
        nvalid = np.where(np.isfinite(qb), nvalid, 0)
        return qb, nvalid


def _wrapped_ranges(t0, t1, n):
    """[t0, t1) on a periodic axis of length n, as one or two ranges within [0, n)."""
    if t1 - t0 >= n:
        return [(0, n)]
    t0m, t1m = t0 % n, t1 % n
    if t1m == 0:
        t1m = n
    if t0m < t1m:
        return [(t0m, t1m)]
    return [(t0m, n), (0, t1m)]


def _negative_dominant(neg, pos, nrow, dt_scale, dt_pad, tsamp):
    """Negative peaks with no positive peak at least as bright in their lobe window."""
    keep = np.ones(neg.size, dtype=bool)
    if neg.size == 0 or pos.size == 0:
        return keep
    order = np.argsort(pos["el_index"], kind="stable")
    ep = pos["el_index"][order]
    wn = np.maximum(np.nan_to_num(neg["duration"]) / tsamp, 1)
    wp = np.maximum(np.nan_to_num(pos["duration"]) / tsamp, 1)
    lo = np.searchsorted(ep, neg["el_index"] - nrow, side="left")
    hi = np.searchsorted(ep, neg["el_index"] + nrow, side="right")
    for k in range(neg.size):
        c = order[lo[k] : hi[k]]
        if c.size == 0:
            continue
        reach = dt_scale * np.maximum(wp[c], wn[k]) + dt_pad
        near = np.abs(pos["ra_index"][c].astype(int) - int(neg["ra_index"][k])) <= reach
        keep[k] = not np.any(near & (pos["snr"][c] >= neg["snr"][k]))
    return keep


def _gpd_fit(excess):
    """Maximum-likelihood generalised Pareto (xi, sigma) of positive excesses."""
    from scipy import stats

    excess = np.asarray(excess, dtype=np.float64)
    excess = excess[excess >= 0]
    if excess.size < 10:
        return 0.0, max(float(np.mean(excess)) if excess.size else 2.0, 0.5)
    xi, _, sigma = stats.genpareto.fit(excess, floc=0.0)
    return float(xi), float(sigma)


def _gpd_sf(x, xi, sigma):
    from scipy import stats

    return stats.genpareto.sf(np.maximum(x, 0.0), xi, loc=0.0, scale=sigma)


def _gpd_isf(p, xi, sigma):
    from scipy import stats

    return stats.genpareto.isf(p, xi, loc=0.0, scale=sigma)


class AddWindowStats(tasklib.base.ContainerTask):
    """Statistics of the single samples within each candidate's duration.

    A real pulse lasting W samples is an excess in each of them, while a long
    excursion of the scintillating sky that wins a long trial is mostly a few large
    samples among others of either sign.  For each candidate, the single-sample
    output of every band (duration 1) is taken along the track of its sky position
    (its telescope-x drifts with the hour angle, as in :class:`AddCandidateSpectra`)
    for its W samples, and the bands are combined into one S/N per sample,
    sum(w a) / sqrt(sum w).  The ``window`` table gets the number of samples
    ``win_n``, the fraction above zero ``win_fpos``, the self-normalised
    ``win_t`` = mean / (std / sqrt(n)), which needs no noise model, and ``win_top2``,
    the share of the sum from its two largest samples (with ``min_samples`` or more,
    otherwise NaN).  The matched filter's response makes a pulse's samples unequal
    near its edges, so a pulse is only approximately a plateau.

    Attributes
    ----------
    pol : int
        Index of the polarisation of the band output.  Default is 0.
    min_samples : int
        Least number of samples for the statistics.  Default is 3.
    """

    pol = config.Property(proptype=int, default=0)
    min_samples = config.Property(proptype=int, default=3)

    def process(self, table, bands):
        """Add the window statistics.

        Parameters
        ----------
        table : containers.TransientCandidateTable
            Candidates.
        bands : containers.TransientMatchedFilter
            Matched filter output for each band, with duration 1.

        Returns
        -------
        out : containers.TransientCandidateTable
            A copy of the table, with the ``window`` table.
        """
        from mpi4py import MPI

        durations = [int(d) for d in bands.index_map["duration"][:]]
        if 1 not in durations:
            raise ValueError("The band output must have duration 1.")
        idur = durations.index(1)
        bands.redistribute("el")
        e, tb, xi, wc = _candidate_positions(table, bands, match_duration=True)
        n = e.size
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

        stats = np.zeros((n, 4))
        mine = np.zeros(n)
        for k in range(n):
            if not off <= e[k] < off + nloc:
                continue
            mine[k] = 1.0
            W = int(wc[k])
            j = np.arange(W)
            xj = -np.cos(dec[e[k], xi[k]]) * np.sin(ha[e[k], xi[k]] + j * dra)
            ij = np.rint((xj - xg[0]) / dx).astype(int)
            tt = tb[k] + j
            ok = (ij >= 0) & (ij < xg.size)
            if periodic:
                tt = tt % nra
            else:
                ok &= tt < nra
            tt, ij = tt[ok], ij[ok]
            A = la.local_array[self.pol, idur, :, e[k] - off]  # freq, ra, x
            Wt = lw.local_array[self.pol, idur, :, e[k] - off]
            a, w = A[:, tt, ij], Wt[:, tt, ij]  # freq, sample
            sw = w.sum(axis=0)
            good = sw > 0
            s = (w * a).sum(axis=0)[good] * tools.invert_no_zero(np.sqrt(sw[good]))
            ns = s.size
            stats[k, 0] = ns
            if ns < max(self.min_samples, 2):
                stats[k, 1:] = np.nan
                continue
            sd = np.std(s, ddof=1)
            tot = s.sum()
            stats[k, 1] = np.mean(s > 0)
            stats[k, 2] = s.mean() / (sd / np.sqrt(ns)) if sd > 0 else np.nan
            stats[k, 3] = np.sort(s)[-2:].sum() / tot if tot > 0 else np.nan
        # each candidate's row is on one rank
        stats = np.where(mine[:, np.newaxis] > 0, stats, 0.0)
        stats = np.nan_to_num(stats, nan=-1e30)
        bands.comm.Allreduce(MPI.IN_PLACE, stats, op=MPI.SUM)
        stats[stats < -1e29] = np.nan

        out = _table_copy(table, skip=("window",))
        out.add_dataset("window")
        wv = np.zeros(n, dtype=out.datasets["window"][:].dtype)
        wv["win_n"] = stats[:, 0]
        wv["win_fpos"] = stats[:, 1]
        wv["win_t"] = stats[:, 2]
        wv["win_top2"] = stats[:, 3]
        out.datasets["window"][:] = wv
        out.attrs["win_min_samples"] = self.min_samples
        self.log.info(f"Window statistics of {n} candidates.")
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
      they are periodic.  With ``row_sky_sep`` set, the peaks are the other
      candidates of the table along the row (within the ``row_dy``, ``row_dra``,
      ``row_snr`` and ``row_ratio`` of :class:`GroupTransientCandidates`) whose sky
      position is at least ``row_sky_sep`` degrees from the candidate's: the other
      pulses of a periodic source come from the same position and do not count,
      while the peaks of a region of excess noise come from anywhere.  Without it,
      ``row_npeak`` of the table (all peaks) is used.
    - 6 ``catalogue``: a catalogue source brighter than ``cat_flux_ratio`` times the
      candidate's amplitude within the match region, which could leave a residual
      of that size.  The residuals of catalogue sources are below ~0.3% of their
      flux (scripts/sift/source_residual.py), so the default of 100 leaves a margin
      of ~3 without vetoing the many faint pulses that fall near a bright source
      by chance (the match region holds a source above 0.3 Jy for ~1/3 of the sky).
    - 7 ``narrowband`` (unless ``band_frac_max`` is None, the default): more than
      ``band_frac_max`` of the S/N^2 in one band (if the spectra were added), or else
      more than ``group_frac_max`` in one group of bands.  Off by default: on the
      injected pulses of 2418 (inject=train) it removed 7 of 24 in V (steep spectra
      put most of the S/N^2 in the lowest band, and the band S/N of long candidates
      is mis-normalised), and alone it removed only ~2-8 background candidates a day.
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
    - 13 ``extended`` (unless ``ext_dy_min`` is None): the extent (pixels joined
      across small gaps) reaches at least ``ext_dy_min`` in telescope-y, far more
      than a point source.  This catches arcs broken into blobs and short vertical
      streaks, which the ``arc`` test misses.  In I, where the sky noise leaves
      wide regions of S/N > 5, the extent of a real pulse joins them (median 0.12
      for the injected pulses of 2418), so the test should be off there.
    - 14 ``ns_shape`` (only with the north-south fit, see :class:`AddNSProfile`):
      the profile across telescope-y is not the synthesized beam: the best width
      scale is above ``ns_scale_max`` or below ``ns_scale_min`` and improves the
      chi-squared by at least ``ns_dchi2_min`` over the point source.
    - 15 ``crowded`` (if ``crowd_ratio_max`` is set): at least ``crowd_nmin`` other
      candidates of the table within +-``crowd_dy`` in telescope-y and
      +-``crowd_dra`` degrees of start RA, at least ``row_sky_sep`` degrees away on
      the sky (or anywhere, without it), have at least ``crowd_ratio_max`` times the
      S/N: the candidate does not stand out of the noise around it.  Requiring
      several (``crowd_nmin`` 3 at 0.7) rather than one at 1.0 keeps the real pulses
      that happen to lie next to one bright noise candidate, and removes as much
      of the noise, which comes in clusters.
    - 16 ``periodic`` (information only): at least ``periodic_npeak_min`` other
      peaks along the row, spaced by at least ``periodic_spacing_min`` seconds with
      a coefficient of variation below ``periodic_cv_max``.  The minimum spacing
      keeps out the flicker of scintillation, whose local maxima are spaced by about
      the width of the window that finds them.
    - 17 ``fidelity`` (if ``fidelity_min`` is set and the table has the
      ``fidelity`` table of :class:`AddNegativeFidelity`): fidelity below
      ``fidelity_min``, as many negative peaks of at least the candidate's S/N and
      about its duration around it as positive ones, so that noise explains it.
    - 18 ``significance`` (if ``snr_cal_min`` is set and the table has the
      ``significance`` table of :class:`CalibrateSignificance`): ``snr_cal`` below
      ``snr_cal_min``, the S/N calibrated against the local tail of the noise.
    - 19 ``window`` (if ``win_t_min`` is set and the table has the ``window`` table
      of :class:`AddWindowStats`): a candidate of ``win_wmin`` or more samples whose
      single samples are not a sustained excess, ``win_t`` below ``win_t_min``.
    - 20 ``compact`` (if ``cat_near_ratio`` is set): a catalogue source under the
      candidate (``cat_near_flux``, within ``cat_near_sep`` of
      :class:`GroupTransientCandidates`) brighter than ``cat_near_ratio`` times its
      amplitude.  The scintillation of compact sources, by up to ~10% in the day
      and seen in V and L through leakage, is most of the background: on 2418,
      55% of the kept transients sit within 0.1 deg of a source above 0.3 Jy (4%
      by chance), and a ratio of 10 removes 14 of 79 kept I, 84 of 102 V and 15 of
      69 L candidates and none of the injected pulses.
    - 21 ``pulsar`` (if ``reject_pulsars``): a known pulsar matched by
      :class:`GroupTransientCandidates` (``psr_name``).  Their scintillation,
      nulling and single pulses pass the filter.
    - 22 ``template`` (if ``tmpl_ratio`` is set): an amplitude below ``tmpl_ratio``
      times the magnitude of the template of compact sources at the candidate
      (``tmpl_flux``, :class:`AddTemplateFlux`), so within the scintillation of
      the sources under it, in its east-west sidelobes or grating lobes.

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
        "crowded": 15,
        "periodic": 16,
        "fidelity": 17,
        "significance": 18,
        "window": 19,
        "compact": 20,
        "pulsar": 21,
        "template": 22,
    }
    INFO: ClassVar = ("periodic",)

    snr_min = config.Property(proptype=float, default=10.0)
    sun_ha_max = config.Property(proptype=float, default=20.0)
    track_dy_max = config.Property(proptype=float, default=0.02)
    lobe_ha_max = config.Property(proptype=float, default=10.0)
    col_nrows_max = config.Property(proptype=int, default=50)
    row_npeak_max = config.Property(proptype=int, default=3)
    row_sky_sep = config.Property(
        proptype=lambda v: None if v is None else float(v), default=0.3
    )
    periodic_npeak_min = config.Property(proptype=int, default=3)
    periodic_cv_max = config.Property(proptype=float, default=0.15)
    periodic_spacing_min = config.Property(proptype=float, default=240.0)
    cat_flux_ratio = config.Property(proptype=float, default=100.0)
    crowd_ratio_max = config.Property(
        proptype=lambda v: None if v is None else float(v), default=None
    )
    crowd_nmin = config.Property(proptype=int, default=1)
    crowd_dy = config.Property(proptype=float, default=0.1)
    crowd_dra = config.Property(proptype=float, default=2.0)
    band_frac_max = config.Property(
        proptype=lambda v: None if v is None else float(v), default=None
    )
    group_frac_max = config.Property(proptype=float, default=0.7)
    arc_dy_min = config.Property(proptype=float, default=0.02)
    arc_dt_min = config.Property(proptype=float, default=20.0)
    arc_corr_min = config.Property(proptype=float, default=0.7)
    fwhm_y_max = config.Property(proptype=float, default=0.03)
    ext_dy_min = config.Property(
        proptype=lambda v: None if v is None else float(v), default=0.05
    )
    weight_ratio_min = config.Property(proptype=float, default=0.05)
    horizon_y_max = config.Property(proptype=float, default=0.95)
    smooth_chi2_max = config.Property(proptype=float, default=5.0)
    smooth_frac_min = config.Property(proptype=float, default=0.7)
    ns_dchi2_min = config.Property(proptype=float, default=25.0)
    ns_scale_max = config.Property(proptype=float, default=1.5)
    ns_scale_min = config.Property(proptype=float, default=0.67)
    fidelity_min = config.Property(
        proptype=lambda v: None if v is None else float(v), default=None
    )
    snr_cal_min = config.Property(
        proptype=lambda v: None if v is None else float(v), default=None
    )
    win_t_min = config.Property(
        proptype=lambda v: None if v is None else float(v), default=None
    )
    win_wmin = config.Property(proptype=int, default=4)
    cat_near_ratio = config.Property(
        proptype=lambda v: None if v is None else float(v), default=None
    )
    reject_pulsars = config.Property(proptype=bool, default=False)
    tmpl_ratio = config.Property(
        proptype=lambda v: None if v is None else float(v), default=None
    )
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

        # The other candidates along the row, and around, at other sky positions
        sky_sep = self.row_sky_sep if self.row_sky_sep and self.row_sky_sep > 0 else None
        if sky_sep is not None:
            row_npeak, _ = _other_candidates(
                rec,
                float(table.attrs.get("row_dy", 0.01)),
                float(table.attrs.get("row_dra", 5.0)),
                sky_sep,
                float(table.attrs.get("row_snr", 8.0)),
                float(table.attrs.get("row_ratio", 1.0 / 3.0)),
            )
        else:
            row_npeak = rec["row_npeak"].astype(int)
        crowd_n = np.zeros(rec.size, dtype=int)
        crowd = np.zeros(rec.size)
        if self.crowd_ratio_max is not None:
            crowd_n, crowd = _other_candidates(
                rec, self.crowd_dy, self.crowd_dra, sky_sep, 0.0, self.crowd_ratio_max
            )

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
            flag("row_repeat", (row_npeak > self.row_npeak_max) & ~periodic)
            flag(
                "catalogue",
                rec["cat_flux"] >= self.cat_flux_ratio * np.abs(rec["amplitude"]),
            )
            has_bands = rec["band_nsig"] >= 0
            if self.band_frac_max is not None:
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
            if self.ext_dy_min is not None:
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
            if self.crowd_ratio_max is not None:
                flag("crowded", crowd_n >= self.crowd_nmin)
            if self.fidelity_min is not None and "fidelity" in table.datasets:
                fid = table.datasets["fidelity"][:]["fidelity"]
                flag("fidelity", fid < self.fidelity_min)
            if self.snr_cal_min is not None and "significance" in table.datasets:
                cal = table.datasets["significance"][:]["snr_cal"]
                flag("significance", cal < self.snr_cal_min)
            if self.win_t_min is not None and "window" in table.datasets:
                win = table.datasets["window"][:]
                long_ = win["win_n"] >= self.win_wmin
                flag("window", long_ & (win["win_t"] < self.win_t_min))
            names = rec.dtype.names
            if self.cat_near_ratio is not None and "cat_near_flux" in names:
                flag(
                    "compact",
                    rec["cat_near_flux"] >= self.cat_near_ratio * np.abs(rec["amplitude"]),
                )
            if self.reject_pulsars and "psr_name" in names:
                flag("pulsar", rec["psr_name"] != b"")
            if self.tmpl_ratio is not None and "tmpl_flux" in names:
                flag("template", np.abs(rec["amplitude"]) < self.tmpl_ratio * np.abs(rec["tmpl_flux"]))

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
            if name == "sift":  # replaced (and an older table may lack columns)
                continue
            if name not in out.datasets:
                out.add_dataset(name)
            out.datasets[name][:] = table.datasets[name][:][rows]
        out.add_dataset("sift")
        sift = np.zeros(rows.size, dtype=out.datasets["sift"][:].dtype)
        sift["keep"] = keep[rows]
        sift["flags"] = flags[rows]
        sift["row_npeak"] = row_npeak[rows]
        sift["crowd_ratio"] = crowd[rows]
        sift["crowd_n"] = crowd_n[rows]
        out.datasets["sift"][:] = sift
        out.attrs["sift_flags"] = json.dumps(self.FLAGS)
        for key in (
            "snr_min", "sun_ha_max", "track_dy_max", "lobe_ha_max", "col_nrows_max",
            "row_npeak_max", "row_sky_sep", "periodic_npeak_min", "periodic_cv_max",
            "periodic_spacing_min", "cat_flux_ratio", "crowd_ratio_max", "crowd_nmin",
            "crowd_dy", "crowd_dra",
            "band_frac_max", "group_frac_max", "arc_dy_min", "arc_dt_min",
            "arc_corr_min", "fwhm_y_max", "ext_dy_min", "weight_ratio_min",
            "horizon_y_max",
            "smooth_chi2_max", "smooth_frac_min", "ns_dchi2_min", "ns_scale_max",
            "ns_scale_min", "fidelity_min", "snr_cal_min", "win_t_min", "win_wmin",
            "cat_near_ratio", "reject_pulsars", "tmpl_ratio",
        ):  # fmt: skip
            val = getattr(self, key)
            out.attrs[f"sift_{key}"] = "none" if val is None else val
        return out


def _other_candidates(rec, dy, dra, sky_sep, snr_min, ratio):
    """The other candidates around each candidate of a table.

    For candidate k, the others within +-``dy`` in telescope-y and +-``dra`` degrees
    of start RA, and (if ``sky_sep`` is not None) at least ``sky_sep`` degrees from
    its sky position (ICRS).

    Returns
    -------
    nstrong : np.ndarray[int]
        The number of them with S/N at least max(``snr_min``, ``ratio`` x that of
        candidate k).
    max_ratio : np.ndarray[float]
        The largest S/N among them over that of candidate k (0 if there are none).
    """
    n = rec.size
    nstrong = np.zeros(n, dtype=int)
    max_ratio = np.zeros(n)
    if n == 0:
        return nstrong, max_ratio
    snr = rec["snr"].astype(np.float64)
    order = np.argsort(rec["y"], kind="stable")
    ys = rec["y"][order]
    lo = np.searchsorted(ys, rec["y"] - dy, side="left")
    hi = np.searchsorted(ys, rec["y"] + dy, side="right")
    cosd = np.cos(np.radians(rec["dec_icrs"]))
    for k in range(n):
        idx = order[lo[k] : hi[k]]
        idx = idx[idx != k]
        if idx.size == 0:
            continue
        idx = idx[np.abs(_wrap_deg(rec["ra"][idx] - rec["ra"][k])) <= dra]
        if sky_sep is not None and idx.size:
            dsky = np.hypot(
                _wrap_deg(rec["ra_icrs"][idx] - rec["ra_icrs"][k]) * cosd[k],
                rec["dec_icrs"][idx] - rec["dec_icrs"][k],
            )
            idx = idx[~(dsky < sky_sep)]
        if idx.size == 0 or not snr[k] > 0:
            continue
        s = snr[idx]
        nstrong[k] = np.sum(s >= max(snr_min, ratio * snr[k]))
        max_ratio[k] = s.max() / snr[k]
    return nstrong, max_ratio
