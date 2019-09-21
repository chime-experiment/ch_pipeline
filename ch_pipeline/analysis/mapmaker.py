"""
========================================================
Map making tasks (:mod:`~ch_pipeline.analysis.mapmaker`)
========================================================

.. currentmodule:: ch_pipeline.analysis.mapmaker

Tools for making maps from CHIME data.

Tasks
=====

.. autosummary::
    :toctree: generated/

    RingMapMaker
"""
# === Start Python 2/3 compatibility
from __future__ import absolute_import, division, print_function, unicode_literals
from future.builtins import *  # noqa  pylint: disable=W0401, W0614
from future.builtins.disabled import *  # noqa  pylint: disable=W0401, W0614

# === End Python 2/3 compatibility

import numpy as np
import scipy.constants

from caput import config

from draco.core import task
from draco.core import io
from draco.util import tools

from ch_util import ephemeris

from ..core import containers


class RingMapMaker(task.SingleTask):
    """A simple and quick map-maker that forms a series of beams on the meridian.

    This is designed to run on data after it has been collapsed down to
    non-redundant baselines only.

    Attributes
    ----------
    npix : int
        Number of map pixels in the declination dimension.  Default is 512.

    span : float
        Span of map in the declination dimension. Value of 1.0 generates a map
        that spans from horizon-to-horizon.  Default is 1.0.

    weight : string ('natural', 'uniform', or 'inverse_variance')
        How to weight the non-redundant baselines:
            'natural' - each baseline weighted by its redundancy (default)
            'uniform' - each baseline given equal weight
            'inverse_variance' - each baseline weighted by the weight attribute

    exclude_intracyl : bool
        Exclude intracylinder baselines from the calculation.  Default is False.

    include_auto: bool
        Include autocorrelations in the calculation.  Default is False.

    single_beam: bool
        Only calculate the map for the central beam. Default is False.
    """

    npix = config.Property(proptype=int, default=512)

    span = config.Property(proptype=float, default=1.0)

    weight = config.Property(proptype=str, default="natural")

    exclude_intracyl = config.Property(proptype=bool, default=False)

    include_auto = config.Property(proptype=bool, default=False)

    single_beam = config.Property(proptype=bool, default=False)

    def setup(self, tel):
        """Set the Telescope instance to use.

        Parameters
        ----------
        tel : TransitTelescope
        """

        if self.weight not in ["natural", "uniform", "inverse_variance"]:
            KeyError("Do not recognize weight = %s" % self.weight)

        self.telescope = io.get_telescope(tel)

    def process(self, sstream):
        """Computes the ringmap.

        Parameters
        ----------
        sstream : containers.SiderealStream
            The input sidereal stream.

        Returns
        -------
        rm : containers.RingMap
        """

        # Redistribute over frequency
        sstream.redistribute("freq")
        nfreq = sstream.vis.local_shape[0]

        # Extract the right ascension (or calculate from timestamp)
        ra = sstream.ra if "ra" in sstream.index_map else ephemeris.lsa(sstream.time)
        nra = ra.size

        # Construct mapping from vis array to unpacked 2D grid
        nprod = sstream.prod.shape[0]
        pind = np.zeros(nprod, dtype=np.int)
        xind = np.zeros(nprod, dtype=np.int)
        ysep = np.zeros(nprod, dtype=np.float)

        for pp, (ii, jj) in enumerate(sstream.prod):

            if self.telescope.feedconj[ii, jj]:
                ii, jj = jj, ii

            fi = self.telescope.feeds[ii]
            fj = self.telescope.feeds[jj]

            pind[pp] = 2 * int(fi.pol == "S") + int(fj.pol == "S")
            xind[pp] = np.abs(fi.cyl - fj.cyl)
            ysep[pp] = fi.pos[1] - fj.pos[1]

        abs_ysep = np.abs(ysep)
        min_ysep, max_ysep = np.percentile(abs_ysep[abs_ysep > 0.0], [0, 100])

        yind = np.round(ysep / min_ysep).astype(np.int)

        grid_index = list(zip(pind, xind, yind))

        # Define several variables describing the baseline configuration.
        nfeed = int(np.round(max_ysep / min_ysep)) + 1
        nvis_1d = 2 * nfeed - 1
        ncyl = np.max(xind) + 1
        nbeam = 1 if self.single_beam else 2 * ncyl - 1

        # Define polarisation axis
        pol = np.array(["XX", "reXY", "imXY", "YY"])
        npol = len(pol)

        # Create empty array for output
        vis = np.zeros((nfreq, npol, nra, ncyl, nvis_1d), dtype=np.complex128)
        invvar = np.zeros((nfreq, npol, nra, ncyl, nvis_1d), dtype=np.float64)
        weight = np.zeros((nfreq, npol, nra, ncyl, nvis_1d), dtype=np.float64)

        # If natural or uniform weighting was choosen, then calculate the
        # redundancy of the collated visibilities.
        if self.weight != "inverse_variance":
            redundancy = tools.calculate_redundancy(
                sstream.input_flags[:],
                sstream.index_map["prod"][:],
                sstream.reverse_map["stack"]["stack"][:],
                sstream.vis.shape[1],
            )

            if self.weight == "uniform":
                redundancy = (redundancy > 0).astype(np.float32)

        # De-reference distributed arrays outside loop to save repeated MPI calls
        ssv = sstream.vis[:]
        ssw = sstream.weight[:]

        # Unpack visibilities into new array
        for vis_ind, (p_ind, x_ind, y_ind) in enumerate(grid_index):

            # Handle different options for weighting baselines
            if self.weight == "inverse_variance":
                w = ssw[:, vis_ind]
            else:
                w = (ssw[:, vis_ind] > 0.0).astype(np.float32)
                w *= redundancy[np.newaxis, vis_ind]

            if x_ind != 0 or not self.exclude_intracyl:
                vis[:, p_ind, :, x_ind, y_ind] = ssv[:, vis_ind]
                invvar[:, p_ind, :, x_ind, y_ind] = ssw[:, vis_ind]
                weight[:, p_ind, :, x_ind, y_ind] = w

        # Remove auto-correlations
        if not self.include_auto:
            weight[..., 0, 0] = 0.0
        # Autos get double-counted at the end
        weight[..., 0, 0] *= 0.5


        # Normalize the weighting function
        # Multiply by 2 here to count negative baselines
        norm = 2 * np.sum(weight, axis=(-2, -1))
        weight *= tools.invert_no_zero(norm)[..., np.newaxis, np.newaxis]

        # Construct phase array
        el = self.span * np.linspace(-1.0, 1.0, self.npix)

        vis_pos_1d = np.fft.fftfreq(nvis_1d, d=(1.0 / (nvis_1d * min_ysep)))

        # Create empty ring map
        rm = containers.RingMap(
            beam=nbeam, el=el, pol=pol, ra=ra, axes_from=sstream, attrs_from=sstream
        )
        # Add datasets
        rm.add_dataset("rms")
        rm.add_dataset("dirty_beam")

        # Make sure ring map is distributed over frequency
        rm.redistribute("freq")

        # Estimate rms noise in the ring map by propagating estimates
        # of the variance in the visibilities
        rm.rms[:] = np.sqrt(
            2 * np.sum(tools.invert_no_zero(invvar) * weight ** 2.0, axis=(-2, -1))
        )

        # Dereference datasets
        rmm = rm.map[:]
        rmb = rm.dirty_beam[:]

        # Loop over local frequencies and fill ring map
        for lfi, fi in sstream.vis[:].enumerate(0):

            # Get the current frequency and wavelength
            fr = sstream.freq[fi]

            wv = scipy.constants.c * 1e-6 / fr

            # Create array that will be used for the inverse
            # discrete Fourier transform in y-direction
            pa = np.exp(
                -2.0j * np.pi * vis_pos_1d[:, np.newaxis] * el[np.newaxis, :] / wv
            )

            # Perform inverse discrete fourier transform in y-direction
            # and inverse fast fourier transform in x-direction
            bfm = np.dot(weight[lfi] * vis[lfi], pa)
            sb = np.dot(weight[lfi], pa)
            if self.single_beam:
                # Only need the 0th term if the irfft, equivalent to adding in EW direction
                bfm = np.sum(bfm, axis=2)[:, :, np.newaxis, ...]
                sb = np.sum(sb, axis=2)[:, :, np.newaxis, ...]
            else:
                bfm = np.fft.fftshift(
                    np.fft.ifft(bfm, nbeam, axis=2) * nbeam,
                    axes=2
                )
                sb = np.fft.fftshift(
                    np.fft.ifft(sb, nbeam, axis=2) * nbeam,
                    axes=2
                )

            # Save to container (shifting to the final axis ordering)
            # for co-pol we take twice the real part
            # to complete sum over negative baselines
            copol_ind = [0, 3]
            rmm[:, copol_ind, lfi] = 2 * bfm[copol_ind].real.transpose(2, 0, 1, 3)
            rmb[:, copol_ind, lfi] = 2 * sb[copol_ind].real.transpose(2, 0, 1, 3)
            # for cross-pol we save real and imaginary parts of complex map formed
            # by combining positive and negative baselines (which are in the other index)
            xpol_ind = [1, 2]
            rmm[:, xpol_ind, lfi] = (
                bfm[xpol_ind[0]] + bfm[xpol_ind[1]].conj()
            ).view("(2,)float").transpose(1, 3, 0, 2)
            rmb[:, xpol_ind, lfi] = (
                sb[xpol_ind[0]] + sb[xpol_ind[1]].conj()
            ).view("(2,)float").transpose(1, 3, 0, 2)

        return rm
