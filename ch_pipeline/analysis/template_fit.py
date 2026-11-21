"""Tasks that fit a template to hybrid beamformed visibilities and subtract it.

Not used by the transient search in :mod:`ch_pipeline.analysis.transients`, which
filters the sky along right ascension instead.
"""

import numpy as np
from caput import config
from caput.pipeline import tasklib
from draco.util import tools


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
