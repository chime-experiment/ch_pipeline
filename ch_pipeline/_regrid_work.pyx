
cimport cython

import numpy as np
cimport numpy as np

@cython.boundscheck(False)
cpdef _band_wiener_covariance(double [:, ::1] Rn, double [::1] Ni, int bw):

    cdef double [:, ::1] Ci = np.zeros((bw+1, Rn.shape[0]), dtype=np.float64)
    Ci[:] = 0.0

    cdef unsigned int N, M
    cdef unsigned int alpha, beta, betap, j

    cdef double t

    N = Rn.shape[0]
    M = Rn.shape[1]

    with nogil:
        for alpha in range(bw+1):
            for beta in range(bw - alpha, N):
                betap = alpha + beta - bw
                t = 0.0
                for j in range(M):
                    t += Rn[betap, j] * Rn[beta, j] * Ni[j]
                Ci[alpha, beta] = t

    return np.asarray(Ci)
