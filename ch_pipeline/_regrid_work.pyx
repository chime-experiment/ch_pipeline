from cython.parallel import prange, parallel
cimport cython

import numpy as np
cimport numpy as np

cdef inline int int_max(int a, int b) nogil: return a if a >= b else b


# A routine for quickly calculating the noise part of the banded
# covariance matrix for the Wiener filter.
@cython.boundscheck(False)
cpdef _band_wiener_covariance(double [:, ::1] Rn, double [::1] Ni, int bw):

    cdef double [:, ::1] Ci = np.zeros((bw+1, Rn.shape[0]), dtype=np.float64)

    cdef unsigned int N, M
    cdef unsigned int alpha, beta, betap, j, alpha_start

    cdef double t

    N = Rn.shape[0]
    M = Rn.shape[1]

    # Loop over the band array indices to generate each one (opposite
    # order for faster parallelisation)
    for beta in prange(N, nogil=True):

        # Calculate alphas to start at
        alpha_start = int_max(0, bw - beta)
        
        for alpha in range(alpha_start, bw+1):
            betap = alpha + beta - bw
            t = 0.0
            for j in range(M):
                t = t + Rn[betap, j] * Rn[beta, j] * Ni[j]
            Ci[alpha, beta] = t

    return np.asarray(Ci)
