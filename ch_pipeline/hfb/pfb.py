"""PFB related tools."""

from typing import Callable

import numpy as np
import scipy.linalg as la
import scipy.sparse as ss
from caput import pfb
from draco.util import tools


class DeconvolvePFB:
    """Deconvolve the effects of the PFB from unchannelized HFB data.

    Default parameters represent what is done within CHIME.

    Parameters
    ----------
    N
        Length of a PFB tap.
    M
        Number of PFB taps.
    Q
        Number of subfrequencies, or equivalently the number of PFB outputs in the
        second FFT.
    window
        The window function applied in the PFB.
    nyquist
        Is the nyquist frequency included in the data? Default is False to
        match the CHIME/Caspertools PFB.
    """

    def __init__(
        self,
        N: int = 2048,
        M: int = 4,
        Q: int = 128,
        window: Callable[[int, int], np.ndarray] = pfb.sinc_hamming,
        band: int = 1,
        nyquist: bool = False,
    ):
        self.N = N
        self.M = M
        self.Q = Q
        self.band = band
        self.nyquist = nyquist

        if M > Q:
            raise ValueError(
                f"Number of subfreq ({Q=}) must be more than the number of PFB taps "
                f"({M=})."
            )

        w_pad = np.zeros(N * Q, dtype=np.float64)
        w_pad[: (M * N)] = window(M, N)

        self.Wt = np.fft.fft(w_pad).conj().reshape(N, Q).transpose(1, 0) / N
        self.Wt2 = np.abs(self.Wt) ** 2

        # self._gen_matrices(N, M, Q, band, nyquist)
        self._gen_matrices_sparse()

    def _gen_matrices_dense(self):
        # This is slow, but is a good reference implementation that the sparse matrices
        # are doing the correct thing
        Q = self.Q
        N = self.N
        band = self.band

        self.W = np.zeros((Q, N, N), dtype=np.float64)

        for ri in range(N):
            for alpha in range(-band, band):
                self.W[:, ri, (ri + alpha) % N] = self.Wt2[:, alpha]

        # Construct the matrix to project from the positive frequencies to the full
        # space
        self.Hf = np.zeros((N, N // 2 + 1), dtype=np.float64)
        for ri in range(N // 2 + 1):
            self.Hf[ri, ri] = 1.0
        for ri in range(N // 2 - 1):
            hN = N // 2
            self.Hf[hN + ri + 1, hN - ri - 1] = 1.0

        # Construct the matrix to go from the full frequencies to the positive
        # frequencies (excluding Nyquist)
        Nb = N // 2 + (1 if self.nyquist else 0)
        self.Hb = np.zeros((Nb, N), dtype=np.float64)
        for ri in range(Nb):
            self.Hb[ri, ri] = 1.0

        self.Wc = np.zeros((Q, Nb, N // 2 + 1), dtype=np.float64)

        for s in range(Q):
            self.Wc[s] = self.Hb @ self.W[s] @ self.Hf

    def _gen_matrices_sparse(self):
        Q = self.Q
        N = self.N
        band = self.band

        self.W = []

        # Pre-generate arrays for the row and col indices which are the same for each
        # subfreq
        row_ind = np.zeros((2 * band, N), dtype=np.int32)
        row_ind[:] = np.arange(N)[np.newaxis, :]
        band_ind = np.arange(-band, band, dtype=np.int32)
        col_ind = (row_ind + band_ind[:, np.newaxis]) % N

        data = np.zeros((2 * band, N), dtype=np.float64)

        for s in range(Q):
            data[:] = self.Wt2[s, band_ind][:, np.newaxis]
            self.W.append(
                ss.csr_array(
                    (data.ravel(), (row_ind.ravel(), col_ind.ravel())), shape=(N, N)
                )
            )

        # Construct the matrix to project from the positive frequencies to the full
        # space
        self.Hf = ss.lil_array((N, N // 2 + 1), dtype=np.float64)
        for ri in range(N // 2 + 1):
            self.Hf[ri, ri] = 1.0
        for ri in range(N // 2 - 1):
            hN = N // 2
            self.Hf[hN + ri + 1, hN - ri - 1] = 1.0
        self.Hf = self.Hf.tocsr()

        # Construct the matrix to go from the full frequencies to the positive
        # frequencies (excluding Nyquist)
        Nb = N // 2 + (1 if self.nyquist else 0)
        self.Hb = ss.lil_array((Nb, N), dtype=np.float64)
        for ri in range(Nb):
            self.Hb[ri, ri] = 1.0
        self.Hb = self.Hb.tocsr()

        self.Wc = [self.Hb @ self.W[s] @ self.Hf for s in range(Q)]

    def flatten(
        self, x: np.ndarray, Ni: np.ndarray, centered: bool = True
    ) -> tuple[np.ndarray, np.ndarray]:
        """Fit the data to remove the quantisation noise bias and the PFB shape.

        Parameters
        ----------
        x
            Data array packed as [freq, subfreq, time].
        Ni
            Weights (i.e. inverse noise variance) packed the same way as x.
        centered
            Is the data centered (i.e. as it is in the raw data) or shifted such that
            subfreq=0 is the DC bin.

        Returns
        -------
        fx
            Flattened data with the bias subtracted out and divided by the bandpass and
            the average flux.
        fNi
            Weights with the equivalent flattening correction applied.
        """
        # TODO: expose these parameters
        sig_a = 1e4
        sig_b = 1e4
        sig_d = 1e4

        # This is the subfreq template to fit
        w = self.Wt2.sum(axis=1)

        if centered:
            w = np.roll(w, self.Q // 2)

        # We need to construct all the products efficiently. This section tries to build
        # up all the needed combinations.
        wN = w[np.newaxis, :, np.newaxis] * Ni
        zN = Ni
        # Quadratic products
        wNw = (wN * w[np.newaxis, :, np.newaxis]).sum(axis=1)
        zNw = (zN * w[np.newaxis, :, np.newaxis]).sum(axis=1)
        zNz = zN.sum(axis=1)
        # Products against the data
        wNd = (wN * x).sum(axis=1)
        zNd = (zN * x).sum(axis=1)

        # Construct the inverse covariance term via Sherman-Morrison using the products
        # above
        Ci = np.empty((x.shape[0], 2, 2), dtype=np.float64)
        Ci[:, 0, 0] = np.sum(wNw - sig_d**2 / (1 + sig_d**2 * wNw) * wNw**2, axis=-1)
        Ci[:, 0, 1] = np.sum(zNw - sig_d**2 / (1 + sig_d**2 * wNw) * zNw * wNw, axis=-1)
        Ci[:, 1, 0] = Ci[:, 0, 1]
        Ci[:, 1, 1] = np.sum(zNz - sig_d**2 / (1 + sig_d**2 * wNw) * zNw**2, axis=-1)
        # Add in the signal term
        Ci[:, 0, 0] += sig_a**-2
        Ci[:, 1, 1] += sig_b**-2

        # Construct the "dirty" estimator
        dirty = np.empty((x.shape[0], 2), dtype=np.float64)
        dirty[:, 0] = np.sum(wNd - sig_d**2 / (1 + sig_d**2 * wNw) * wNw * wNd, axis=-1)
        dirty[:, 1] = np.sum(zNd - sig_d**2 / (1 + sig_d**2 * wNw) * zNw * wNd, axis=-1)

        # Solve for a and b, and then apply to the data
        fx = np.empty_like(x)
        fNi = np.empty_like(Ni)
        for ii in range(x.shape[0]):
            a, b = la.solve(Ci[ii], dirty[ii], assume_a="pos")
            fx[ii] = (x[ii] - b) * tools.invert_no_zero(a * w[:, np.newaxis])
            fNi[ii] = (a * w[:, np.newaxis]) ** 2 * Ni[ii]

        return fx, Ni
