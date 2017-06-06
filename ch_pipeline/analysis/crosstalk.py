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
from caput import config, mpiutil

from draco.core import task

from ch_util import tools

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

    norm = config.Property(proptype=bool, default=False)
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
        self.freq_norm = np.array([600.0, 75.0])
        self.time_norm = np.array([180.0, 60.0])

        # Specify degree of polynomial (currently hardcoded)
        # Key has format (is_intra, is_xx)

        inter_deg = np.array([6, 10, 10])
        intra_deg = np.array([6, 10, 10, 10, 10])

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

        # Distribute over products
        sstream.redistribute('prod')

        # Make list of the products relevant for crosstalk removal
        tel = self.beamtransfer.telescope

        # Extract frequency and ra
        freq = sstream.index_map['freq']['centre']
        ra = sstream.index_map['ra']

        # Check that we have enough data points to fit the polynomial
        if len(freq) < (self.max_poly_deg[0] + 1):
            ValueError("Number of frequency samples (%d) is less than number of coefficients for polynomial (%d)." %
                        (len(freq), (self.max_poly_deg[0] + 1)))

        if len(ra) < (self.max_poly_deg[1] + 1):
            ValueError("Number of time samples (%d) is less than number of coefficients for polynomial (%d)." %
                        (len(ra), (self.max_poly_deg[1] + 1)))

        # Create two-dimensional grid in frequency and time
        gfreq, gra = np.meshgrid(freq, ra, indexing='ij')
        gfreq, gra = gfreq.flatten(), gra.flatten()

        # Normalize frequency and time
        gnu = (gfreq - self.freq_norm[0]) / self.freq_norm[1]

        gra = (gra - self.time_norm[0]) / self.time_norm[1]

        ndata = gnu.size

        # Set up index for flagging outliers
        even, odd = np.arange(0, freq.size, 2), np.arange(1, freq.size, 2)

        # Compute Hermite polynomials
        shp = (ndata,) + tuple(self.max_poly_deg + 1)
        Hmax = np.polynomial.hermite.hermvander2d(gnu, gra, self.max_poly_deg).reshape(*shp)

        if self.norm:
            Hmax *= np.polynomial.hermite.hermweight(gnu)[:, np.newaxis]
            Hmax *= np.polynomial.hermite.hermweight(gra)[:, np.newaxis]

        # Create container to hold results
        crosstalk = containers.Crosstalk(fdegree=shp[1], tdegree=shp[2], path=self.max_ndelay,
                                         axes_from=sstream, attrs_from=sstream)

        # Distribute over products
        crosstalk.redistribute('prod')

        # Initialize datasets to zero
        for key in crosstalk.datasets.keys():
            crosstalk.datasets[key][:] = 0.0

        # Set attributes specifying model normalization
        crosstalk.attrs['norm'] = self.norm
        crosstalk.attrs['fcenter'] = self.freq_norm[0]
        crosstalk.attrs['fspan'] = self.freq_norm[1]
        crosstalk.attrs['tcenter'] = self.time_norm[0]
        crosstalk.attrs['tspan'] = self.time_norm[1]

        # Extract products
        prod = sstream.index_map['prod']

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
            weight = sstream.weight[:, pp, :].view(np.ndarray) > 0.0

            # Do not apply crosstalk removal to products containing bad feeds
            if not np.any(weight):
                continue

            # Extract visibilities from container
            vis = sstream.vis[:, pp, :].view(np.ndarray)

            weight = weight.astype(np.float64)

            # Flatten over frequency and time
            weight = weight.flatten()
            vis = vis.flatten()

            # Determine the crosstalk model that will be used for this product
            is_xx = (tel.feeds[fi].pol == 'E') and (tel.feeds[fj].pol == 'E')
            is_intra = (tel.feeds[fi].cyl == tel.feeds[fj].cyl)

            poly_freq_ncoeff = self.poly_freq_deg_lookup[(is_intra, is_xx)] + 1
            poly_time_ncoeff = self.poly_time_deg_lookup[(is_intra, is_xx)] + 1

            ndelay = poly_freq_ncoeff.size / 2

            # Compute expected delays
            ddist = tel.feedpositions[fi, :] - tel.feedpositions[fj, :]
            baseline = np.sqrt(np.sum(ddist**2))

            delays = get_delays(baseline, ndelay=ndelay, is_intra=is_intra)

            delays = delays[0:ndelay]
            delays = np.concatenate((delays, -delays))
            ndelay = delays.size

            # Generate Hermite polynomials
            ncoeff = poly_freq_ncoeff * poly_time_ncoeff
            nparam = np.sum(ncoeff)

            dedge = np.concatenate(([0], np.cumsum(ncoeff)))

            H = np.zeros((ndata, nparam), dtype=np.float64)
            dindex = np.zeros(nparam, dtype=np.int)
            for dd in range(ndelay):
                aa, bb = dedge[dd], dedge[dd+1]

                H[:, aa:bb] = Hmax[:, 0:poly_freq_ncoeff[dd], 0:poly_time_ncoeff[dd]].reshape(ndata, ncoeff[dd])

                dindex[aa:bb] = dd

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
                sig = 1.4826 * np.median(resid[weight > 0])

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

    Attributes
    ----------
    norm : bool
        Normalize the Hermite polynomials used to model
        the frequency and time dependence of the
        crosstalk coupling coefficients.
    """

    def process(self, sstream, crosstalk):
        """Computes the best fit model for crosstalk.

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

        # Distribute over products
        sstream.redistribute('prod')
        crosstalk.redistribute('prod')

        # Extract frequency and ra
        freq = sstream.index_map['freq']['centre']
        ra = sstream.index_map['ra']

        # Create two-dimensional grid in frequency and time
        gfreq, gra = np.meshgrid(freq, ra, indexing='ij')

        # Normalize frequency and time
        gnu = (gfreq - crosstalk.attrs['fcenter']) / crosstalk.attrs['fspan']
        gra = (gra - crosstalk.attrs['tcenter']) / crosstalk.attrs['tspan']

        ndata = gnu.size

        # Loop over products
        for lpp, pp in sstream.vis[:].enumerate(1):

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
                    cmodel = (np.polynomial.hermite.hermval2d(gnu, gra, coeff[ii, 0:fdeg, 0:tdeg]) *
                              np.exp(2.0J * np.pi * gfreq * 1e6 * dd))

                    if crosstalk.attrs['norm']:
                        cmodel *= np.polynomial.hermite.hermweight(gnu)
                        cmodel *= np.polynomial.hermite.hermweight(gra)

                    # Subtract model for this path from visibilities
                    sstream.vis[:, pp, :] -= cmodel


        # Return sidereal stream with crosstalk removed
        return sstream


class MeanVisibility(task.SingleTask):
    """Calculate the weighted mean (over time) of every element of the
    visibility matrix.

    Parameters
    ----------
    all_data : bool
        If this is True, then use all of the input data to
        calculate the weighted mean.  If False, then use only
        night-time data away from the transit of bright
        point sources.  Default is False.
    daytime : bool
        Use only day-time data away from transit of bright point
        sources to caclulate the weighted mean.  Overriden by all_data.
        Default is False.
    median : bool
        Calculate weighted median instead of weighted mean.
        Default is False.
    nblock : int
        Sidereal day is separated into blocks and the mean is
        calculated for each block.  Default is 1 block.
    """

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

        # Extract product map
        prod = sstream.index_map['prod']

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

        # Check if we are using all of the data to calculate the mean,
        # or only "quiet" periods.
        if self.all_data:

            if isinstance(sstream, andata.CorrData):
                flag_quiet = time_func(sstream.time) == lsd

                ra = ephemeris.transit_RA(sstream.time)

            elif isinstance(sstream, containers.SiderealStream):
                flag_quiet = np.ones(len(sstream.index_map['ra'][:]), dtype=np.bool)

                ra = sstream.index_map['ra']

            else:
                raise RuntimeError('Format of `sstream` argument is unknown.')

        else:

            # Check if we are dealing with CorrData or SiderealStream
            if isinstance(sstream, andata.CorrData):
                # Extract ra
                ra = ephemeris.transit_RA(sstream.time)

                # Find night time data
                if self.daytime:
                    flag_quiet = daytime_flag(sstream.time)
                else:
                    flag_quiet = ~daytime_flag(sstream.time)

                flag_quiet &= (time_func(sstream.time) == lsd)

            elif isinstance(sstream, containers.SiderealStream):
                # Extract csd and ra
                if hasattr(lsd, '__iter__'):
                    lsd_list = lsd
                else:
                    lsd_list = [lsd]

                ra = sstream.index_map['ra']

                # Find night time data
                flag_quiet = np.ones(len(ra), dtype=np.bool)
                for cc in lsd_list:
                    if self.daytime:
                        flag_quiet &= daytime_flag(ephemeris.csd_to_unix(cc + ra/360.0))
                    else:
                        flag_quiet &= ~daytime_flag(ephemeris.csd_to_unix(cc + ra/360.0))

            else:
                raise RuntimeError('Format of `sstream` argument is unknown.')

            # Find data free of bright point sources
            for src_name, src_ephem in ephemeris.source_dictionary.iteritems():

                if isinstance(src_ephem, ephem.FixedBody):

                    peak_ra = ephemeris.peak_RA(src_ephem, deg=True)
                    src_window = 3.0*cal_utils.guess_fwhm(400.0, pol='X', dec=src_ephem._dec, sigma=True)

                    dra = (ra - peak_ra) % 360.0
                    dra -= (dra > 180.0)*360.0

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

        # Loop over frequencies and baselines to reduce memory usage
        for lfi, fi in sstream.vis[:].enumerate(0):
            for lbi, bi in sstream.vis[:].enumerate(1):

                # Extract visibility and weight
                data = sstream.vis[fi, bi, :].view(np.ndarray)
                weight = flag_quiet * sstream.weight[fi, bi, :].view(np.ndarray)

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

