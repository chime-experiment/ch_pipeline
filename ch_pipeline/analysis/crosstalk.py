"""
========================================================
Crosstalk tasks (:mod:`~ch_pipeline.analysis.crosstalk`)
========================================================

.. currentmodule:: ch_pipeline.analysis.crosstalk

Crosstalk estimation and removal.

Tasks
=====

.. autosummary::
    :toctree: generated/

    ComputeCrosstalk
    RemoveCrosstalk
    MeanVisibility
    ChangeMeanVisibility
"""
import numpy as np
from caput import config, mpiutil, mpiarray

from draco.core import task

from ch_util import tools
from ch_util import andata
from ch_util import ephemeris

from ..core import containers


class ComputeCrosstalk(task.SingleTask):
    """ Fit measured visibilities to a model for the crosstalk between feeds.

    Attributes
    ----------
    norm : bool
        Normalize the orthogonal polynomials used to model
        the frequency and time dependence of the
        crosstalk coupling coefficients.
    poly_time_degree : int
        Degree of the orthogonal polynomial used to model
        the time dependence of the crosstalk coupling
        coefficients.
    """

    cheb = config.Property(proptype=bool, default=False)
    norm = config.Property(proptype=bool, default=False)
    auto = config.Property(proptype=bool, default=False)
    poly_time_degree = config.Property(proptype=int, default=4)

    max_iter = config.Property(proptype=int, default=2)
    nsig = config.Property(proptype=float, default=3.0)

    def setup(self, bt):
        """Set the beamtransfer matrices and define crosstalk model.

        Parameters
        ----------
        bt : beamtransfer.BeamTransfer
            Beam transfer manager object. This does not need to have
            pre-generated matrices as they are not needed.
        """

        self.beamtransfer = bt

        # Specify center and span to normalize variable of polynomial
        if self.cheb:
            # self.freq_norm = np.array([600.0, 205.0])
            # self.time_norm = np.array([180.0, 185.0])
            self.freq_norm = np.array([600.0, 300.0])
            self.time_norm = np.array([180.0, 270.0])
        else:
            self.freq_norm = np.array([600.0, 75.0])
            self.time_norm = np.array([180.0, 60.0])

        # Specify degree of polynomial (currently hardcoded)
        # Key has format (is_intra, is_xx)

        # inter_deg = np.array([6, 10, 10])
        # intra_deg = np.array([6, 10, 10, 10, 10])

        inter_deg = np.array([6, 10, 10])
        intra_deg = np.array([6, 16, 10])

        self.poly_freq_deg_lookup = {}
        self.poly_freq_deg_lookup[(0, 0)] = inter_deg
        self.poly_freq_deg_lookup[(0, 1)] = inter_deg
        self.poly_freq_deg_lookup[(1, 0)] = intra_deg
        self.poly_freq_deg_lookup[(1, 1)] = intra_deg

        npt = self.poly_time_degree
        self.poly_time_deg_lookup = {}
        self.poly_time_deg_lookup[(0, 0)] = np.repeat(npt, inter_deg.size)
        self.poly_time_deg_lookup[(0, 1)] = np.repeat(npt, inter_deg.size)
        self.poly_time_deg_lookup[(1, 0)] = np.repeat(npt, intra_deg.size)
        self.poly_time_deg_lookup[(1, 1)] = np.repeat(npt, intra_deg.size)

        # Ensure that deg_lookups are numpy arrays and determine maximum degree required
        max_poly_freq_deg = None
        for key, value in self.poly_freq_deg_lookup.iteritems():
            self.poly_freq_deg_lookup[key] = np.concatenate((value, value))

            max_poly_freq_deg = (np.max(value) if max_poly_freq_deg is None
                                               else max(max_poly_freq_deg, np.max(value)))

        max_poly_time_deg = None
        for key, value in self.poly_time_deg_lookup.iteritems():
            self.poly_time_deg_lookup[key] = np.concatenate((value, value))

            max_poly_time_deg = (np.max(value) if max_poly_time_deg is None
                                               else max(max_poly_time_deg, np.max(value)))

        self.max_poly_deg = np.array([max_poly_freq_deg, max_poly_time_deg])

        max_ndelay = None
        for key, value in self.poly_freq_deg_lookup.iteritems():
            max_ndelay = value.size if max_ndelay is None else max(max_ndelay, value.size)

        self.max_ndelay = max_ndelay


    def process(self, sstream):
        """Computes the best fit model for crosstalk.

        Parameters
        ----------
        sstream : containers.SiderealStream
            The input sidereal stream.

        Returns
        -------
        crosstalk : containers.Crosstalk
        """

        tel = self.beamtransfer.telescope

        # Extract dimensions of the sidereal stream
        freq = sstream.index_map['freq']['centre']
        prod = sstream.index_map['prod']
        ra   = sstream.index_map['ra']

        nfreq, nra = freq.size, ra.size

        # Check that we have enough data points to fit the polynomial
        if nfreq < (self.max_poly_deg[0] + 1):
            ValueError("Number of frequency samples (%d) is less than number of coefficients for polynomial (%d)." %
                        (nfreq, (self.max_poly_deg[0] + 1)))

        if nra < (self.max_poly_deg[1] + 1):
            ValueError("Number of time samples (%d) is less than number of coefficients for polynomial (%d)." %
                        (nra, (self.max_poly_deg[1] + 1)))

        # If requested, extract autocorrelations
        if self.auto:

            # Distribute over frequency
            sstream.redistribute('freq')

            # Get auto index
            iauto = np.array([idx for idx, (fi, fj) in enumerate(prod) if fi == fj])

            # Extract autocorrelations for local frequencies
            local_start = sstream.vis.local_offset
            local_auto_corr = sstream.vis[:, iauto, :].view(np.ndarray).real

            shp = local_auto_corr.shape
            shp = (nfreq, shp[1], shp[2])
            auto_corr = np.zeros(shp, dtype=local_auto_corr.dtype)

            nproc = 1 if sstream.comm is None else sstream.comm.size
            # Gather local distributed auto correlations to a global array for all procs
            for rank in range(nproc):
                mpiutil.gather_local(auto_corr, local_auto_corr, local_start, root=rank, comm=sstream.comm)

            # Deal with the scenario where we have already compressed over redundant baselines
            if iauto.size == 2:
                polstr = [tel.feeds[prod[idx][0]].pol for idx in iauto]

                iauto = np.array([polstr.index(feed.pol) if hasattr(feed, 'pol') else -1
                                  for feed in tel.feeds])
            else:
                iauto = range(iauto.size)

        # Distribute over products
        sstream.redistribute('prod')

        # Create two-dimensional grid in frequency and time
        gfreq, gra = np.meshgrid(freq, ra, indexing='ij')
        gfreq, gra = gfreq.flatten(), gra.flatten()

        # Normalize frequency and time
        gnu = (gfreq - self.freq_norm[0]) / self.freq_norm[1]
        gra = (gra - self.time_norm[0]) / self.time_norm[1]

        ndata = gnu.size

        # Define polynomial basis
        if self.cheb:
            poly_vander = np.polynomial.chebyshev.chebvander2d
            poly_weight = np.polynomial.chebyshev.chebweight
        else:
            poly_vander = np.polynomial.hermite.hermvander2d
            poly_weight = np.polynomial.hermite.hermweight

        # Compute polynomials
        shp = (ndata,) + tuple(self.max_poly_deg + 1)

        Hmax = poly_vander(gnu, gra, self.max_poly_deg).reshape(*shp)

        if self.norm:
            Hmax *= poly_weight(gnu)[:, np.newaxis, np.newaxis]
            Hmax *= poly_weight(gra)[:, np.newaxis, np.newaxis]

        # Create container to hold results
        crosstalk = containers.Crosstalk(fdegree=shp[1], tdegree=shp[2], path=self.max_ndelay,
                                         axes_from=sstream, attrs_from=sstream)

        # Distribute over products
        crosstalk.redistribute('prod')

        # Initialize datasets to zero
        for key in crosstalk.datasets.keys():
            crosstalk.datasets[key][:] = 0.0

        # Set attributes specifying model
        crosstalk.attrs['cheb'] = self.cheb
        crosstalk.attrs['norm'] = self.norm
        crosstalk.attrs['auto'] = self.auto
        crosstalk.attrs['fcenter'] = self.freq_norm[0]
        crosstalk.attrs['fspan'] = self.freq_norm[1]
        crosstalk.attrs['tcenter'] = self.time_norm[0]
        crosstalk.attrs['tspan'] = self.time_norm[1]
        if self.auto:
            crosstalk.attrs['iauto'] = iauto

        # Loop over products
        for lpp, pp in sstream.vis[:].enumerate(1):

            fi, fj = prod[pp]

            # Do not apply crosstalk removal to autocorrelations
            if fi == fj:
                continue

            # Do not apply crosstalk removal for non-CHIME feeds
            if not hasattr(tel.feeds[fi], 'pol') or not hasattr(tel.feeds[fj], 'pol'):
                continue

            # Do not apply crosstalk removal for cross-polarizations
            if (tel.feeds[fi].pol != tel.feeds[fj].pol):
                continue

            # Extract weights from container
            weight = sstream.weight[:, pp, :].view(np.ndarray)

            # Do not apply crosstalk removal to products containing bad feeds
            if not np.any(weight > 0.0):
                continue

            # Extract visibilities from container
            vis = sstream.vis[:, pp, :].view(np.ndarray)

            weight = weight.astype(np.float64)

            # Flatten over frequency and time
            weight = weight.flatten()
            vis = vis.flatten()

            # Extract autocorrelations
            if self.auto:
                auto_neg = auto_corr[:, iauto[fi], :].flatten()
                auto_pos = auto_corr[:, iauto[fj], :].flatten()

            # Determine the crosstalk model that will be used for this product
            is_xx = (tel.feeds[fi].pol == 'E') and (tel.feeds[fj].pol == 'E')
            is_intra = (tel.feeds[fi].cyl == tel.feeds[fj].cyl)

            poly_freq_ncoeff = self.poly_freq_deg_lookup[(is_intra, is_xx)] + 1
            poly_time_ncoeff = self.poly_time_deg_lookup[(is_intra, is_xx)] + 1

            ndelay = poly_freq_ncoeff.size / 2

            # Compute expected delays
            ddist = tel.feedpositions[fi, :] - tel.feedpositions[fj, :]
            baseline = np.sqrt(np.sum(ddist**2))

            delays = get_delays(baseline, ndelay=ndelay, is_intra=is_intra)[0]

            delays = delays[0:ndelay]
            delays = np.concatenate((delays, -delays))
            ndelay = delays.size

            # Generate polynomials
            ncoeff = poly_freq_ncoeff * poly_time_ncoeff
            nparam = np.sum(ncoeff)

            dedge = np.concatenate(([0], np.cumsum(ncoeff)))

            H = np.zeros((ndata, nparam), dtype=np.float64)
            dindex = np.zeros(nparam, dtype=np.int)
            for dd in range(ndelay):
                aa, bb = dedge[dd], dedge[dd+1]

                H[:, aa:bb] = Hmax[:, 0:poly_freq_ncoeff[dd], 0:poly_time_ncoeff[dd]].reshape(ndata, ncoeff[dd])

                dindex[aa:bb] = dd

                if self.auto:
                    if delays[dd] < 0.0:
                        H[:, aa:bb] *= auto_neg[:, np.newaxis]
                    else:
                        H[:, aa:bb] *= auto_pos[:, np.newaxis]

            # Construct delay matrix
            S = np.exp(2.0J * np.pi * gfreq[:, np.newaxis] * 1e6 * delays[np.newaxis, dindex])

            # Combine polynomial Vandermonde matrix and delay matrix
            A = S * H

            # Iterate to detect outliers
            it = 0
            while (it < self.max_iter):

                # Calculate covariance of model coefficients
                C = np.dot(A.T.conj(), weight[:, np.newaxis] * A)

                # Solve for model coefficients
                coeff = np.linalg.lstsq(C, np.dot(A.T.conj(), weight * vis))[0]

                # Calculate residuals
                resid = np.abs(vis - np.dot(A, coeff))

                # Determine outliers
                sig = 1.4826 * np.median(resid[weight > 0.0])

                not_outlier = resid < (self.nsig * sig)

                # Update weight to exclude outliers and increment counter
                weight *= not_outlier.astype(np.float64)
                it += 1

            # Save to output container
            crosstalk.delay[pp, 0:ndelay] = delays
            for dd in range(ndelay):
                aa, bb = dedge[dd], dedge[dd+1]
                this_coeff = coeff[aa:bb].reshape(poly_freq_ncoeff[dd], poly_time_ncoeff[dd])
                crosstalk.coeff[pp, dd, 0:poly_freq_ncoeff[dd], 0:poly_time_ncoeff[dd]] = this_coeff
                crosstalk.flag[pp, dd, 0:poly_freq_ncoeff[dd], 0:poly_time_ncoeff[dd]] = True
                crosstalk.weight[pp, :, :] = weight.reshape(freq.size, ra.size).astype(np.bool)

        # Return container
        return crosstalk


class RemoveCrosstalk(task.SingleTask):
    """ Remove crosstalk between feeds.
    """

    def process(self, sstream, crosstalk):
        """Remove model for crosstalk between feeds.

        Parameters
        ----------
        sstream : containers.SiderealStream
            The input sidereal stream.
        crosstalk : containers.Crosstalk

        Returns
        -------
        sstream : containers.SiderealStream
            Sidereal stream with crosstalk removed.
        """

        # Extract parameters specifying model
        cheb = crosstalk.attrs.get('cheb', False)
        norm = crosstalk.attrs.get('norm', False)
        auto = crosstalk.attrs.get('auto', False)

        # Define polynomial basis
        if cheb:
            poly_eval = np.polynomial.chebyshev.chebval2d
            poly_weight = np.polynomial.chebyshev.chebweight
        else:
            poly_eval = np.polynomial.hermite.hermval2d
            poly_weight = np.polynomial.hermite.hermweight

        # Extract dimensions of the sidereal stream
        freq = sstream.index_map['freq']['centre']
        prod = sstream.index_map['prod']
        ra = sstream.index_map['ra']

        nfreq, nra = freq.size, ra.size

        # Create two-dimensional grid in frequency and time
        gfreq, gra = np.meshgrid(freq, ra, indexing='ij')

        # Normalize frequency and time
        gnu = (gfreq - crosstalk.attrs['fcenter']) / crosstalk.attrs['fspan']
        gra = (gra - crosstalk.attrs['tcenter']) / crosstalk.attrs['tspan']

        # Extract autocorrelations
        if auto:
            # Distribute over frequency
            sstream.redistribute('freq')

            # Get auto index
            iauto = np.array([idx for idx, (fi, fj) in enumerate(prod) if fi == fj])

            # Extract autocorrelations for local frequencies
            local_start = sstream.vis.local_offset
            local_auto_corr = sstream.vis[:, iauto, :].view(np.ndarray).real

            shp = local_auto_corr.shape
            shp = (nfreq, shp[1], shp[2])
            auto_corr = np.zeros(shp, dtype=local_auto_corr.dtype)

            nproc = 1 if sstream.comm is None else sstream.comm.size
            # Gather local distributed auto correlations to a global array for all procs
            for rank in range(nproc):
                mpiutil.gather_local(auto_corr, local_auto_corr, local_start, root=rank, comm=sstream.comm)

            iauto = crosstalk.attrs['iauto']

        # Distribute over products
        sstream.redistribute('prod')
        crosstalk.redistribute('prod')

        # Loop over products
        for lpp, pp in sstream.vis[:].enumerate(1):

            coeff = crosstalk.coeff[pp].view(np.ndarray)
            flag = crosstalk.flag[pp].view(np.ndarray)
            delay = crosstalk.delay[pp].view(np.ndarray)

            if not np.any(flag):
                continue

            # Extract autocorrelations
            if auto:
                fi, fj = prod[pp]
                auto_neg = auto_corr[:, iauto[fi], :]
                auto_pos = auto_corr[:, iauto[fj], :]

            # Loop over path
            for ii, dd in enumerate(delay):

                # Check if coefficients exists for this path
                if np.any(flag[ii]):

                    # Determine degree of polynomials
                    fdeg = np.flatnonzero(np.any(flag[ii], axis=1)).size
                    tdeg = np.flatnonzero(np.any(flag[ii], axis=0)).size

                    # Evaluate model
                    cmodel = (poly_eval(gnu, gra, coeff[ii, 0:fdeg, 0:tdeg]) *
                              np.exp(2.0J * np.pi * gfreq * 1e6 * dd))

                    if norm:
                        cmodel *= poly_weight(gnu)
                        cmodel *= poly_weight(gra)

                    if auto:
                        if dd < 0.0:
                            cmodel *= auto_neg
                        else:
                            cmodel *= auto_pos

                    # Subtract model for this path from visibilities
                    sstream.vis[:, pp, :] -= cmodel


        # Return sidereal stream with crosstalk removed
        return sstream


class CrosstalkCalibration(task.SingleTask):
    """ Determine receiver gain based on response to crosstalk signal.

    Attributes
    ----------
    solve_gain: bool, default True
        If True, assume that the individual receiver gains
        are fluctuating with time.  If False, assume that the
        individual receiver temperatures are fluctuating with time.

    ymin: float, default 1.2
        Do not include baselines with N-S separation
        less than ymin in the fit.
    """

    solve_gain = config.Property(proptype=bool, default=True)
    only_intra = config.Property(proptype=bool, default=True)
    ymin = config.Property(proptype=float, default=1.2)

    def setup(self, bt):
        """Set the beamtransfer matrices.

        Parameters
        ----------
        bt : beamtransfer.BeamTransfer
            Beam transfer manager object. This does not need to have
            pre-generated matrices as they are not needed.
        """

        self.beamtransfer = bt

    def process(self, sstream, crosstalk):
        """Compute gains based on sidereal stream and crosstalk model.

        Parameters
        ----------
        sstream : containers.SiderealStream
            The input sidereal stream.
        crosstalk : containers.Crosstalk

        Returns
        -------
        sstream : containers.SiderealStream
            Sidereal stream with crosstalk removed.
        """

        # First calculate crosstalk model
        # ------------------------------------------------

        crosstalk.redistribute('prod')

        # Extract telescope information
        tel = self.beamtransfer.telescope

        # Determine polynomial
        cheb = crosstalk.attrs.get('cheb', False)

        # Extract products
        prodmap = sstream.index_map['prod']
        cprod = crosstalk.index_map['prod']

        # Extract frequency and ra
        freq = sstream.index_map['freq']['centre']
        ra = sstream.index_map['ra']

        # Define dimensions
        nprod = prodmap.shape[0]
        ncprod = cprod.shape[0]

        nfreq = freq.size
        nra = ra.size

        # Create two-dimensional grid in frequency and time
        gfreq, gra = np.meshgrid(freq, ra, indexing='ij')

        # Normalize frequency and time
        gnu = (gfreq - crosstalk.attrs['fcenter']) / crosstalk.attrs['fspan']
        gra = (gra - crosstalk.attrs['tcenter']) / crosstalk.attrs['tspan']

        # Create array to hold crosstalk model
        mdtype = sstream.vis[:].dtype

        model_pos = mpiarray.MPIArray((ncprod, nfreq, nra), axis=0, dtype=mdtype)
        model_pos[:] = 0.0

        model_neg = mpiarray.MPIArray((ncprod, nfreq, nra), axis=0, dtype=mdtype)
        model_neg[:] = 0.0

        print "before model_pos.shape:  ", model_pos.global_shape, model_pos.local_shape, np.any(model_pos[:].view(np.ndarray))
        print "before model_neg.shape:  ", model_neg.global_shape, model_neg.local_shape, np.any(model_neg[:].view(np.ndarray))
        print "crosstalk.coeff.shape:   ", crosstalk.coeff.global_shape, crosstalk.coeff.local_shape

        # Calculate crosstalk model.  Loop over products.
        for lpp, pp in crosstalk.coeff[:].enumerate(0):

            coeff = crosstalk.coeff[pp].view(np.ndarray)
            flag = crosstalk.flag[pp].view(np.ndarray)
            delay = crosstalk.delay[pp].view(np.ndarray)

            if not np.any(flag):
                continue

            # Loop over path
            for ii, dd in enumerate(delay):

                # Check if coefficients exists for this path
                if np.any(flag[ii]):

                    # Determine degree of polynomials
                    fdeg = np.flatnonzero(np.any(flag[ii], axis=1)).size
                    tdeg = np.flatnonzero(np.any(flag[ii], axis=0)).size

                    # Evaluate model
                    if cheb:
                        cmodel = (np.polynomial.chebyshev.chebval2d(gnu, gra, coeff[ii, 0:fdeg, 0:tdeg]) *
                                  np.exp(2.0J * np.pi * gfreq * 1e6 * dd))
                    else:
                        cmodel = (np.polynomial.hermite.hermval2d(gnu, gra, coeff[ii, 0:fdeg, 0:tdeg]) *
                                  np.exp(2.0J * np.pi * gfreq * 1e6 * dd))

                    if crosstalk.attrs['norm']:
                        if cheb:
                            cmodel *= np.polynomial.chebyshev.chebweight(gnu)
                            cmodel *= np.polynomial.chebyshev.chebweight(gra)
                        else:
                            cmodel *= np.polynomial.hermite.hermweight(gnu)
                            cmodel *= np.polynomial.hermite.hermweight(gra)

                    # Update model
                    if dd >= 0.0:
                        model_pos[lpp, :, :] += cmodel
                    else:
                        model_neg[lpp, :, :] += cmodel


        # Next determine antenna deviations from this model
        # -------------------------------------------------

        sstream.redistribute('freq')
        model_pos = model_pos.redistribute(axis=1)
        model_neg = model_neg.redistribute(axis=1)

        print "after model_pos.shape:  ", model_pos.global_shape, model_pos.local_shape, np.any(model_pos[:].view(np.ndarray))
        print "after model_neg.shape:  ", model_neg.global_shape, model_neg.local_shape, np.any(model_neg[:].view(np.ndarray))
        print "sstream.vis.shape:      ", sstream.vis.global_shape, sstream.vis.local_shape


        # Determine good X and Y feeds
        if self.only_intra:

            xfeeds_west = np.array([idx for idx, inp in enumerate(tel.feeds) if tools.is_chime_x(inp) and (inp.cyl == 0)])
            xfeeds_east = np.array([idx for idx, inp in enumerate(tel.feeds) if tools.is_chime_x(inp) and (inp.cyl == 1)])
            yfeeds_west = np.array([idx for idx, inp in enumerate(tel.feeds) if tools.is_chime_y(inp) and (inp.cyl == 0)])
            yfeeds_east = np.array([idx for idx, inp in enumerate(tel.feeds) if tools.is_chime_y(inp) and (inp.cyl == 1)])

            feed_groups = [xfeeds_west, xfeeds_east, yfeeds_west, yfeeds_east]

        else:

            xfeeds = np.array([idx for idx, inp in enumerate(tel.feeds) if tools.is_chime_x(inp)])
            yfeeds = np.array([idx for idx, inp in enumerate(tel.feeds) if tools.is_chime_y(inp)])

            feed_groups = [xfeeds, yfeeds]

        # Determine products for each subset of feeds
        polmap = []
        for feeds in feed_groups:
            prods = np.array([idx for idx, (fi, fj) in enumerate(prodmap) if ((fi in feeds) and (fj in feeds))])
            polmap.append((feeds, prods))

        poldim = [(2 * xx[1].size, (1 + 2*self.solve_gain) * xx[0].size, xx[0].size) for xx in polmap]

        # Determine feed positions
        feed_pos = tel.feedpositions
        vis_pos = np.array([ feed_pos[ii] - feed_pos[jj] for ii, jj in prodmap ])
        vis_pos = np.where(np.isnan(vis_pos), np.zeros_like(vis_pos), vis_pos)

        # Determine mapping between full visibility matrix and stack over redundant baselines
        vismap = tools.pack_product_array(tel.feedmap, axis=0)

        # Create container to hold results
        gcont = containers.CrosstalkGain(axes_from=sstream, attrs_from=sstream)#,
                                         #pol_x=poldim[0][1], pol_y=poldim[1][1])

        if self.solve_gain:
            gcont.add_dataset('gain')

        gcont.redistribute('freq')

        # Initialize datasets to zero
        for key in gcont.datasets.keys():
            gcont.datasets[key][:] = 0.0

        # Loop over polarizations
        for ipol, (ifeed, iprod) in enumerate(polmap):

            compmap = vismap[iprod]

            good_baseline = ~((np.abs(vis_pos[iprod, 1]) <= self.ymin) &
                              (np.abs(vis_pos[iprod, 0]) < 10.0)) & (compmap >= 0)
            good_baseline = good_baseline.astype(np.float64)


            lfeed = list(ifeed)
            polprod = [(lfeed.index(pp[0]), lfeed.index(pp[1])) for pp in prodmap[iprod]]

            # Create indices for real and imaginary components
            dims = poldim[ipol][0:2]
            nbase = poldim[ipol][0]
            nfeed = poldim[ipol][2]

            ire = np.arange(0, nbase, 2, dtype=np.int)
            iim = np.arange(1, nbase, 2, dtype=np.int)

            # Loop over frequencies
            for lff, ff in sstream.vis[:].enumerate(0):

                # Loop over right ascension
                for rr in range(nra):

                    # Extract data
                    vtemp = (sstream.vis[ff, iprod, rr].view(np.ndarray) -
                                   model_pos[compmap, lff, rr] -
                                   model_neg[compmap, lff, rr])

                    vis = np.zeros(nbase, dtype=np.float64)
                    vis[ire] = vtemp.real
                    vis[iim] = vtemp.imag

                    weight = np.zeros_like(vis)
                    weight[ire] = sstream.weight[ff, iprod, rr].view(np.ndarray) * good_baseline
                    weight[iim] = weight[ire]

                    # Create array for crosstalk model
                    A = np.zeros(dims, dtype=np.float64)

                    # Set up crosstalk model assuming receiver gain varies
                    for cnt, (aa, bb) in enumerate(polprod):
                        A[ire[cnt], aa] = model_neg[compmap[cnt], lff, rr].real
                        A[ire[cnt], bb] = model_pos[compmap[cnt], lff, rr].real

                        A[iim[cnt], aa] = model_neg[compmap[cnt], lff, rr].imag
                        A[iim[cnt], bb] = model_pos[compmap[cnt], lff, rr].imag

                    if self.solve_gain:
                        for cnt, (aa, bb) in enumerate(polprod):
                            model = model_pos[compmap[cnt], lff, rr] + model_neg[compmap[cnt], lff, rr]

                            A[ire[cnt], nfeed + aa] = model.real
                            A[ire[cnt], nfeed + bb] = model.real
                            A[ire[cnt], 2*nfeed + aa] = -model.imag
                            A[ire[cnt], 2*nfeed + bb] =  model.imag

                            A[iim[cnt], nfeed + aa] = model.imag
                            A[iim[cnt], nfeed + bb] = model.imag
                            A[iim[cnt], 2*nfeed + aa] =  model.real
                            A[iim[cnt], 2*nfeed + bb] = -model.real


                    # Calculate covariance of model coefficients
                    C = np.dot(A.T, weight[:, np.newaxis] * A)

                    # Solve for antenna gains
                    param = np.linalg.lstsq(C, np.dot(A.T, weight * vis))[0]

                    # Calculate chisq before and after subtracting
                    # receiver dependent model for crosstalk
                    gcont.chisq_before[ff, rr] += np.sum(weight * np.abs(vis)**2)
                    gcont.chisq_after[ff, rr]  += np.sum(weight * np.abs(vis - np.dot(A, param))**2)

                    # Save covariance to output container
                    # if ipol == 0:
                    #     gcont.datasets['cov_x'][ff, :, :, rr] = C
                    # else:
                    #     gcont.datasets['cov_y'][ff, :, :, rr] = C

                    # Save receiver temp to output container
                    gcont.receiver_temp[ff, ifeed, rr] = param[0:nfeed]

                    # Save gain to output container
                    if self.solve_gain:
                        gcont.gain[ff, ifeed, rr] = param[nfeed:2*nfeed] + 1.0J * param[2*nfeed:3*nfeed]

            # Return gain container
            return gcont


class MeanVisibility(task.SingleTask):
    """Calculate the weighted mean (over time) of every element of the
    visibility matrix.

    Parameters
    ----------
    keep_sources : bool
        Include data near the transit of bright point sources when
        calculating the weighted mean.  Defaule is False.
    all_data : bool
        If this is True, then use all of the input data to
        calculate the weighted mean.  If False, then use only
        night-time data.  Default is False.
    daytime : bool
        Use only day-time data to caclulate the weighted mean.
        Overriden by all_data.  Default is False.
    median : bool
        Calculate weighted median instead of weighted mean.
        Default is False.
    nblock : int
        Sidereal day is separated into blocks and the mean is
        calculated for each block.  Default is 1 block.
    """

    keep_sources = config.Property(proptype=bool, default=False)
    all_data = config.Property(proptype=bool, default=False)
    daytime = config.Property(proptype=bool, default=False)
    median = config.Property(proptype=bool, default=False)
    nblock = config.Property(proptype=int, default=1)

    def process(self, sstream):
        """
        Parameters
        ----------
        sstream : andata.CorrData or containers.SiderealStream
            Timestream or sidereal stream.

        Returns
        -------
        mustream : same as input
            Sidereal stream containing only the mean value.
        """

        from flagging import daytime_flag
        from ch_util import cal_utils, tools
        import weighted as wq
        import ephem

        # Make sure we are distributed over frequency
        sstream.redistribute('freq')

        # Extract date or lsd
        if 'date' in sstream.attrs:
            lsd = sstream.attrs['date']

            def time_func(x):
                if hasattr(x, '__iter__'):
                    y = np.array([int(xx.strftime('%Y%m%d'))
                                  for xx in ephemeris.unix_to_chime_local_datetime(x)])
                else:
                    y = ephemeris.unix_to_chime_local_datetime(x).strftime('%Y%m%d')

                return y

        else:
            lsd = sstream.attrs['lsd'] if 'lsd' in sstream.attrs else sstream.attrs['csd']

            time_func = lambda x: np.fix(ephemeris.unix_to_csd(x))

        # Pick out this particular sidereal (solar) day and determine RA.
        if isinstance(sstream, andata.CorrData):
            flag_quiet = time_func(sstream.time) == lsd

            ra = ephemeris.transit_RA(sstream.time)

            is_corr = True

        elif isinstance(sstream, containers.SiderealStream):
            flag_quiet = np.ones(len(sstream.index_map['ra'][:]), dtype=np.bool)

            ra = sstream.index_map['ra']

            is_corr = False

        else:
            raise RuntimeError('Format of `sstream` argument is unknown.')


        # Flag daytime or nightime data
        if not self.all_data:

            # Check if we are dealing with CorrData or SiderealStream
            if is_corr:

                # Find night time data
                if self.daytime:
                    flag_quiet &= daytime_flag(sstream.time)
                else:
                    flag_quiet &= ~daytime_flag(sstream.time)

            else:
                # Format lsd into list
                lsd_list = lsd if hasattr(lsd, '__iter__') else [lsd]

                # Find night time data
                for cc in lsd_list:
                    if self.daytime:
                        flag_quiet &= daytime_flag(ephemeris.csd_to_unix(cc + ra/360.0))
                    else:
                        flag_quiet &= ~daytime_flag(ephemeris.csd_to_unix(cc + ra/360.0))


        # Flag data near transit of bright point sources
        if not self.keep_sources:

            # Loop over sources in ephemeris
            for src_name, src_ephem in ephemeris.source_dictionary.iteritems():

                # Make sure they are FixedBody
                if isinstance(src_ephem, ephem.FixedBody):

                    peak_ra = ephemeris.peak_RA(src_ephem, deg=True)
                    src_window = 3.0*cal_utils.guess_fwhm(400.0, pol='X', dec=src_ephem._dec, sigma=True)

                    dra = ra - peak_ra
                    dra = dra - (dra > 180.0)*360.0 + (dra < -180.0)*360.0

                    flag_quiet &= np.abs(dra) > src_window


        # Determine the size of the blocks in time
        nra = ra.size
        npt = int(nra / float(self.nblock))
        nra = self.nblock * npt

        newra = np.mean(ra[0:nra].reshape(self.nblock, npt), axis=-1)

        # Create output container
        mustream = containers.SiderealStream(ra=newra, axes_from=sstream, attrs_from=sstream,
                                             distributed=True, comm=sstream.comm)

        mustream.redistribute('freq')

        # Extract list of baselines
        prod = sstream.index_map['prod']

        # Loop over frequencies and baselines to reduce memory usage
        for lfi, fi in sstream.vis[:].enumerate(0):
            for lbi, bi in sstream.vis[:].enumerate(1):

                # Extract visibility and weight
                data = sstream.vis[fi, bi, :].view(np.ndarray).copy()
                weight = sstream.weight[fi, bi, :].view(np.ndarray).copy()

                if prod[bi][0] != prod[bi][1]:
                    weight *= flag_quiet

                # Loop over blocks in time
                for tt in range(self.nblock):

                    start, end = tt*npt, (tt+1)*npt

                    block_data = data[start:end]
                    block_weight = weight[start:end]

                    norm = np.sum(block_weight)

                    # Calculate either weighted median or weighted mean value
                    if self.median:
                        if norm > 0.0:
                            mu = wq.median(block_data.real, block_weight) + 1.0J * wq.median(block_data.imag, block_weight)
                        else:
                            mu = 0.0 + 0.0J

                    else:
                        mu = tools.invert_no_zero(norm) * np.sum(block_weight*block_data)

                    mustream.vis[fi, bi, tt] = mu
                    mustream.weight[fi, bi, tt] = norm


        # Return sidereal stream containing the mean value
        return mustream


class ChangeMeanVisibility(task.SingleTask):
    """ Subtract or add an overall offset (over time) to every element
    of the visibility matrix.

    Parameters
    ----------
    add : bool
        Add the value instead of subtracting.
        Default is False.
    """

    add = config.Property(proptype=bool, default=False)

    def process(self, sstream, mustream):
        """
        Parameters
        ----------
        sstream : andata.CorrData or containers.SiderealStream
            Timestream or sidereal stream.

        mustream : andata.CorrData or containers.SiderealStream
            Timestream or sidereal stream with 1 element in the time axis
            that contains the value to add or subtract.

        Returns
        -------
        sstream : same as input
            Timestream or sidereal stream with value added or subtracted.
        """

        # Check that input visibilities have consistent shapes

        sshp, mshp = sstream.vis.shape, mustream.vis.shape

        if np.any(sshp[0:2] != mshp[0:2]):
            ValueError("Frequency or product axis differ between inputs.")

        if (len(mshp) != 3) or (mshp[-1] != 1):
            ValueError("Mean value has incorrect shape, must be (nfreq, nprod, 1).")

        # Ensure both inputs are distributed over frequency
        sstream.redistribute('freq')
        mustream.redistribute('freq')

        # Determine autocorrelations
        not_auto = np.array([pi != pj for pi, pj in mustream.index_map['prod'][:]])[np.newaxis, :, np.newaxis]

        # Add or subtract value to the cross-correlations
        if self.add:
            sstream.vis[:] += mustream.vis[:].view(np.ndarray) * not_auto
        else:
            sstream.vis[:] -= mustream.vis[:].view(np.ndarray) * not_auto

        # Return sidereal stream with modified offset
        return sstream


def get_delays(baseline, ndelay=10, is_intra=True):

    # Hardcoded parameters
    focus = 5.5
    bnc = 2.0
    c = 3e8

    # Parse inputs
    dist = np.array(baseline)

    nprod = dist.size
    npath = ndelay if is_intra else 3

    delay = np.zeros((nprod, npath), dtype=np.float64)

    # We have different expected delays for
    # intercylinder and intracylinder baselines
    if is_intra:

        for bb in range(ndelay):
            delay[:, bb] = bnc * np.sqrt((bb * focus)**2 + (dist / bnc)**2) / c

    else:

        delay[:, 0] = dist / c
        delay[:, 1] = (1.80*dist - 28.0) / c
        delay[:, 2] = (0.80*dist + 14.0) / c


    return delay
