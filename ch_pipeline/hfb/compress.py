"""HFB tasks for compressing files."""

from typing import Tuple

import numpy as np

import scipy.linalg as la

from caput import config

from draco.core import task
from draco.util import tools

from .containers import HFBData, HFBCompressed


class CompressHFBWeights(task.SingleTask):
    """Compress weight dataset of HFB data.

    Attributes
    ----------

    method : str, optional
        Method of compression. Either "svd" or "sum". Default is "sum".
    """

    method = config.enum(["svd", "sum"], default="sum")

    def setup(self):
        """Set up compression function."""

        self._compress_fn = {
            "svd": self._compress_svd,
            "sum": self._compress_sum,
        }.get(self.method, None)

    def process(self, stream):
        """Create compressed HFB data from raw HFB data.

        Parameters
        ----------
        stream : HFBData
            Container with HFB data and weights.

        Returns
        -------
        out : HFBCompressed
            Container with HFB data and compressed weights.
        """

        # Read the sizes of the axes
        nfreq, nsubf, nbeam, ntime = stream.weight.shape

        # Initialize arrays to hold the compressed weights
        weight_subf = np.zeros((nfreq, nsubf, ntime))
        weight_beam = np.zeros((nfreq, nbeam, ntime))
        weight_norm = np.zeros((nfreq, ntime))

        # Loop over coarse frequency channels and time samples
        for ifreq in range(nfreq):
            for itime in range(ntime):
                # Find vectors in the subfrequency and beam axes whose outer
                # product gives a good approximation of the original weights,
                # modulo a normalization factor
                wsubf, wbeam, wnorm = self._compress_fn(
                    stream.weight[ifreq, :, :, itime]
                )

                weight_subf[ifreq, :, itime] = wsubf
                weight_beam[ifreq, :, itime] = wbeam
                weight_norm[ifreq, itime] = wnorm

        # Create container to hold output
        out = HFBCompressed(copy_from=stream)

        # Store compressed weights
        out.weight_subf[:] = weight_subf
        out.weight_beam[:] = weight_beam
        out.weight_norm[:] = weight_norm

        # Return output container
        return out

    def _compress_svd(self, array: np.ndarray) -> Tuple[np.ndarray, float, np.ndarray]:

        # Do the Singular Value Decomposition (SVD)
        u, s, vh = la.svd(array, full_matrices=False)

        # Select the first left singular vector, the first right singular vector,
        # and the first singular value
        rows = u[:, 0].copy()
        cols = vh[0].copy()
        norm = s[0]

        return rows, cols, norm

    def _compress_sum(self, array: np.ndarray) -> Tuple[np.ndarray, float, np.ndarray]:

        # Take the sum over the rows of the input array, over its columns, and
        # over the entire array, taking the reciprocal
        rows = array.sum(axis=1)
        cols = array.sum(axis=0)
        norm = tools.invert_no_zero(array.sum())

        return rows, cols, norm


class UnpackHFBWeights(task.SingleTask):
    """Unpack compressed weight dataset of HFB data.

    The reconstructed weights are a rank-1 approximation of the original."""

    def process(self, stream):
        """Create full HFB data by unpacking compressed weights.

        Parameters
        ----------
        stream : HFBCompressed
            Container with HFB data and compressed weights.

        Returns
        -------
        out : HFBData
            Container with HFB data and unpacked weights.
        """

        nfreq, nsubf, nbeam, ntime = stream.hfb.shape

        weight = np.zeros((nfreq, nsubf, nbeam, ntime))

        # Loop over coarse frequency channels and time samples
        for ifreq in range(nfreq):
            for itime in range(ntime):
                # Reconstruct the full weight dataset (approximately) by taking
                # the outer product of the stored weights per subfrequency and
                # per beam, multiplying by the stored normalization factor
                weight[ifreq, :, :, itime] = (
                    np.outer(
                        stream.weight_subf[ifreq, :, itime],
                        stream.weight_beam[ifreq, :, itime],
                    )
                    * stream.weight_norm[ifreq, itime]
                )

        # Create container to hold output
        out = HFBData(copy_from=stream)

        # Add unpacked weights
        out.weight[:] = weight

        # Return output container
        return out
