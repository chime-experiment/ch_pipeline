"""Supporting functions for the HFB tasks.

Functions
=========
- :py:func:`ra_window_pixels`
- :py:func:`interpolation_linear`
- :py:func:`ensure_list`
"""

import numpy as np
from ch_ephem.coord import bmxy_to_hadec
from draco.util import tools


def ra_window_pixels(dec, freq, dra_deg, ra_step, beam_mdl, latitude, ns_grid=None):
    """Number of RA pixels (one-sided) covering 'dra_deg' degrees at best resolution.

    The width in hour angle (i.e. RA) of a synthetic beam grows towards the
    celestial pole. This scales the requested half-width by the ratio of the
    beam width at the source's declination to the narrowest beam width, so
    every cutout contains the same number of "beam widths" of RA.

    Parameters
    ----------
    dec : float
        Declination of the source, in degrees.
    freq : float
        Frequency, in MHz, at which to evaluate the beam model.
    dra_deg : float
        Desired RA half-width, in degrees, at the best-resolution declination.
    ra_step : float
        RA-axis pixel spacing, in degrees (e.g. 360 / 4280).
    beam_mdl : beam_model.formed.FFTFormedActualBeamModel()
        Beam model instance.
    latitude : float
        Latitude of the telescope, in degrees.
    ns_grid : array_like of int, optional
        NS beam indices to scan. Default is np.arange(len(beam_mdl.reference_angles)).

    Returns
    -------
    n_ra : int
        Number of RA pixels either side of the source's own pixel.
    """
    if ns_grid is None:
        ns_grid = np.arange(len(beam_mdl.reference_angles))

    # Beam-model y position and EW FWHM of every NS beam at this frequency.
    # Get_beam_positions returns shape (nbeam, nfreq, 2)
    bmy = beam_mdl.get_beam_positions(ns_grid, [freq])[:, 0, 1]
    fwhm_x = beam_mdl.get_beam_widths(ns_grid, [freq])[:, 0, 0]

    # Convert the EW half-width of each beam into an hour-angle width.
    ha0, _ = bmxy_to_hadec(np.zeros_like(bmy), bmy)
    ha1, _ = bmxy_to_hadec(fwhm_x / 2, bmy)
    width = np.abs(ha1 - ha0)

    # Ratio of the width at the source's own NS beam to the narrowest width.
    ns_beam = int(np.argmin(np.abs(bmy - (dec - latitude))))
    ratio = width[ns_beam] / width.min()

    # Scale the requested half-width and convert degrees to pixels.
    return int(np.ceil(dra_deg * ratio / ra_step))


def interpolation_linear(x, y, w, xeval, mode="zero", xperiod=None):
    """Linear interpolation with handling of uncertainties and flagged data.

    Approximates a 1-D function `y = f(x)` using linear interpolation, while
    taking into account the uncertainties on the known points of `y`.

    Parameters
    ----------
    x : array_like
        A 1-D array of real values.
    y : array_like
        An N-D array of real values. The first axis is the interpolation axis
        and must be the same length as `x`.
    w : array_like
        An N-D array of weights, the same shape as `y`, giving the inverse
        variance of each point in `y`.
    xeval : array_like
        The points at which to evaluate the interpolation.
    mode : {"zero", "extrapolate", "wrap"}, optional
        The `mode` parameter determines the behaviour for values of `xeval` that
        fall outside the range of `x`. Default is "zero". The behaviour for each
        valid value is as follows:
        "zero"
            Returns zeros outside the range of `x`.
        "extrapolate"
            Extrapolates (linearly) using the first two or last two input values.
        "wrap"
            Treats the input as circular, wrapping around to the opposite edge.
            This requires the `xperiod` parameter to be set.
    xperiod : float, optional
        The period of `x`, in case the "wrap" mode is used (otherwise ignored).
    """
    # Invert weights to obtain variances
    var = tools.invert_no_zero(w)

    if mode == "wrap":
        # Check if xperiod, needed in the "wrap" mode, was passed
        if not xperiod:
            raise ValueError("xperiod needed if mode is 'wrap'")

        # Wrap xeval points over period
        xeval %= xperiod

    # Find indices of x points left and right of xeval points
    index = np.searchsorted(x, xeval, side="left")
    ind1 = index - 1
    ind2 = index

    if mode == "wrap":
        # Find points below the range of x and overwrite left index with the index
        # at the end of x
        below = np.flatnonzero(ind1 == -1)
        if below.size > 0:
            ind1[below] = x.size - 1

        # Find points above the range of x and overwrite right index with the index
        # at the strart of x
        above = np.flatnonzero(ind2 == x.size)
        if above.size > 0:
            ind2[above] = 0

    else:
        # Find points below the range of x and overwrite left and right indices
        # with the two indices at the start of x for extrapolation
        below = np.flatnonzero(ind1 == -1)
        if below.size > 0:
            ind1[below] = 0
            ind2[below] = 1

        # Find points above the range of x and overwrite left and right indices
        # with the two indices at the end of x for extrapolation
        above = np.flatnonzero(ind2 == x.size)
        if above.size > 0:
            ind1[above] = x.size - 2
            ind2[above] = x.size - 1

    # Compute intervals
    adx1 = xeval - x[ind1]
    adx2 = x[ind2] - xeval

    if mode == "wrap":
        # For points below and above the range of x, overwrite the interval,
        # adjusting for the period
        if below.size > 0:
            adx1[below] = xeval - x[ind1] - xperiod
        if above.size > 0:
            adx2[above] = x[ind2] + xperiod - xeval

    # Compute relative weights of left and right points
    norm = tools.invert_no_zero(adx1 + adx2)
    a1 = adx2 * norm
    a2 = adx1 * norm

    # Adjust shape of a1 and a2 to allow broadcasting with y and var
    new_axes_pos = tuple(range(1, len(y.shape)))
    a1 = np.expand_dims(a1, axis=new_axes_pos)
    a2 = np.expand_dims(a2, axis=new_axes_pos)

    # Do interpolation and extrapolation
    yeval = a1 * y[ind1, ...] + a2 * y[ind2, ...]
    weval = tools.invert_no_zero(a1**2 * var[ind1, ...] + a2**2 * var[ind2, ...])

    # Set interpolated weight to zero if either of the two weights going into
    # the interpolation is zero, indicated data flagged as bad
    flags = (w[ind1, ...] == 0.0) | (w[ind2, ...] == 0.0)
    weval[flags] = 0.0

    if mode == "zero":
        # Overwrite extrapolated points with zeros
        outside = np.concatenate((below, above))
        yeval[outside] = 0.0
        weval[outside] = 0.0

    return yeval, weval


def ensure_list(x):
    """Return `x` as a list, wrapping it in one if it is not iterable.

    Parameters
    ----------
    x : object
        A scalar or an iterable.

    Returns
    -------
    y : list
        The elements of `x`, or `[x]` if `x` is not iterable.
    """
    if hasattr(x, "__iter__"):
        y = list(x)
    else:
        y = [x]

    return y
