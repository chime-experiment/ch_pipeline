"""
====================================================
Timestream Simulation (:mod:`~ch_pipeline.simulate`)
====================================================

.. currentmodule:: ch_pipeline.simulate

Tasks for simulating timestream data.

Tasks
=====

.. autosummary::
    :toctree: generated/

    LoadBeamTransfer
    SimulateSidereal
    DayMask
    Unroll
"""

import numpy as np

from cora.util import hputil
from caput import mpiutil, pipeline, config, memh5, mpiarray

from ch_util import tools

from . import containers, task


class SimulateSidereal(task.SingleTask):
    """Create a simulated timestream.

    Attributes
    ----------
    maps : list
        List of map filenames. The sum of these form the simulated sky.
    ndays : float, optional
        Number of days of observation. Setting `ndays = None` (default) uses
        the default stored in the telescope object; `ndays = 0`, assumes the
        observation time is infinite so that the noise is zero. This allows a
        fractional number to account for higher noise.
    seed : integer, optional
        Set the random seed used for the noise simulations. Default (None) is
        to choose a random seed.
    """
    maps = config.Property(proptype=list)
    ndays = config.Property(proptype=float, default=0.0)
    seed = config.Property(proptype=int, default=None)

    done = False

    def setup(self, telescope, beamtransfer):
        """Setup the simulation.

        Parameters
        ----------
        tel : TransitTelescope
            Telescope object.
        bt : BeamTransfer
            Beam Transfer maanger.
        """
        self.beamtransfer = beamtransfer
        self.telescope = telescope

    def process(self):
        """Simulate a SiderealStream

        Returns
        -------
        ss : SiderealStream
            Stacked sidereal day.
        feeds : list of CorrInput
            Description of the feeds simulated.
        """

        if self.done:
            raise pipeline.PipelineStopIteration

        # Read in telescope system
        bt = self.beamtransfer
        tel = self.telescope

        lmax = tel.lmax
        mmax = tel.mmax
        nfreq = tel.nfreq
        npol = tel.num_pol_sky

        lfreq, sfreq, efreq = mpiutil.split_local(nfreq)
        freqmap = None

        lm, sm, em = mpiutil.split_local(mmax + 1)

        # Set the minimum resolution required for the sky.
        ntime = 2*mmax+1

        # If we want to add maps use the m-mode formalism to project a skymap
        # into visibility space

        if lfreq > 0:

            # Allocate array to store the local frequencies
            row_map = None

            # Read in and sum up the local frequencies of the supplied maps.
            for mapfile in self.maps:

                mc = containers.Map.from_file(mapfile, distributed=True)
                mc.redistribute('freq')
                freqmap = mc.index_map['freq'][:]

                if (tel.frequencies != freqmap['centre']).all():
                    raise RuntimeError('Frequencies in map file (%s) do not match those in Beam Transfers.' % mapfile)

                if row_map is None:
                    row_map = mc.map[:]
                else:
                    row_map += mc.map[:]

            # Calculate the alm's for the local sections
            row_alm = hputil.sphtrans_sky(row_map, lmax=lmax).reshape((lfreq, npol * (lmax+1), lmax+1))

        else:
            row_alm = np.zeros((lfreq, npol * (lmax+1), lmax+1), dtype=np.complex128)

        # Trim off excess m's and wrap into MPIArray
        row_alm = row_alm[..., :(mmax+1)]
        row_alm = mpiarray.MPIArray.wrap(row_alm, axis=0)

        # Perform the transposition to distribute different m's across processes. Neat
        # tip, putting a shorter value for the number of columns, trims the array at
        # the same time
        col_alm = row_alm.redistribute(axis=2)

        # Transpose and reshape to shift m index first.
        col_alm = col_alm.transpose((2, 0, 1)).reshape((None, nfreq, npol, lmax+1))

        # Create storage for visibility data
        vis_data = mpiarray.MPIArray((mmax+1, nfreq, bt.ntel), axis=0, dtype=np.complex128)
        vis_data[:] = 0.0

        # Iterate over m's local to this process and generate the corresponding
        # visibilities
        for mp, mi in vis_data.enumerate(axis=0):
            vis_data[mp] = bt.project_vector_sky_to_telescope(mi, col_alm[mp].view(np.ndarray))

        # Rearrange axes such that frequency is last (as we want to divide
        # frequencies across processors)
        row_vis = vis_data.transpose((0, 2, 1))

        # Parallel transpose to get all m's back onto the same processor
        col_vis_tmp = row_vis.redistribute(axis=2)
        col_vis_tmp = col_vis_tmp.reshape((mmax + 1, 2, tel.npairs, None))

        # Transpose the local section to make the m's the last axis and unwrap the
        # positive and negative m at the same time.
        col_vis = mpiarray.MPIArray((tel.npairs, nfreq, ntime), axis=1, dtype=np.complex128)
        col_vis[:] = 0.0
        col_vis[..., 0] = col_vis_tmp[0, 0]
        for mi in range(1, mmax+1):
            col_vis[...,  mi] = col_vis_tmp[mi, 0]
            col_vis[..., -mi] = col_vis_tmp[mi, 1].conj()  # Conjugate only (not (-1)**m - see paper)

        del col_vis_tmp

        # If we're simulating noise, create a realisation and add it to col_vis
        # if self.ndays > 0:
        #
        #     # Fetch the noise powerspectrum
        #     noise_ps = tel.noisepower(np.arange(tel.npairs)[:, np.newaxis],
        #                               np.arange(sfreq, efreq)[np.newaxis, :],
        #                               ndays=self.ndays).reshape(tel.npairs, lfreq)[:, :, np.newaxis]
        #
        #     # Seed random number generator to give consistent noise
        #     if self.seed is not None:
        #         # Must include rank such that we don't have massive power deficit from correlated noise
        #         np.random.seed(self.seed + mpiutil.rank)
        #
        #     # Create and weight complex noise coefficients
        #     noise_vis = (np.array([1.0, 1.0J]) * np.random.standard_normal(col_vis.shape + (2,))).sum(axis=-1)
        #     noise_vis *= (noise_ps / 2.0)**0.5
        #
        #     # Reset RNG
        #     if self.seed is not None:
        #         np.random.seed()
        #
        #     # Add into main noise sims
        #     col_vis += noise_vis
        #
        #     del noise_vis

        # Fourier transform m-modes back to get final timestream.
        vis_stream = np.fft.ifft(col_vis, axis=-1) * ntime
        vis_stream = vis_stream.reshape((tel.npairs, lfreq, ntime))
        vis_stream = vis_stream.transpose((1, 0, 2)).copy()

        # Try and fetch out the feed index and info from the telescope object.
        try:
            feed_index = tel.feed_index
        except AttributeError:
            feed_index = tel.nfeed

        # Construct container and set visibility data
        sstream = containers.SiderealStream(freq=freqmap, ra=ntime, input=feed_index,
                                            prod=tel.uniquepairs, distributed=True, comm=mc.comm)
        sstream.vis[:] = mpiarray.MPIArray.wrap(vis_stream, axis=0)
        sstream.weight[:] = 1.0

        self.done = True

        return sstream


class DayMask(task.SingleTask):
    """Crudely simulate a masking out of the daytime data.

    Attributes
    ----------
    start, end : float
        Start and end of masked out region.
    width : float
        Use a smooth transition of given width between the fully masked and
        unmasked data. This is interior to the region marked by start and end.
    zero_data : bool, optional
        Zero the data in addition to modifying the noise weights
        (default is True).
    remove_average : bool, optional
        Estimate and remove the mean level from each visibilty. This estimate
        does not use data from the masked region.
    """

    start = config.Property(proptype=float, default=90.0)
    end = config.Property(proptype=float, default=270.0)

    width = config.Property(proptype=float, default=60.0)

    zero_data = config.Property(proptype=bool, default=True)
    remove_average = config.Property(proptype=bool, default=True)

    def process(self, sstream):
        """Apply a day time mask.

        Parameters
        ----------
        sstream : containers.SiderealStream
            Unmasked sidereal stack.

        Returns
        -------
        mstream : containers.SiderealStream
            Masked sidereal stream.
        """

        sstream.redistribute('freq')

        ra_shift = (sstream.ra[:] - self.start) % 360.0
        end_shift = (self.end - self.start) % 360.0

        # Crudely mask the on and off regions
        mask_bool = ra_shift > end_shift

        # Put in the transition at the start of the day
        mask = np.where(ra_shift < self.width,
                        0.5 * (1 + np.cos(np.pi * (ra_shift / self.width))),
                        mask_bool)

        # Put the transition at the end of the day
        mask = np.where(np.logical_and(ra_shift > end_shift - self.width, ra_shift <= end_shift),
                        0.5 * (1 + np.cos(np.pi * ((ra_shift - end_shift) / self.width))),
                        mask)

        if self.remove_average:
            # Estimate the mean level from unmasked data
            import scipy.stats

            nanvis = sstream.vis[:] * np.where(mask_bool, 1.0, np.nan)[np.newaxis, np.newaxis, :]
            #average = np.nanmean(nanvis, axis=-1)[:, :, np.newaxis]
            average = scipy.stats.nanmedian(nanvis, axis=-1)[:, :, np.newaxis]
            sstream.vis[:] -= average

        # Apply the mask to the data
        if self.zero_data:
            sstream.vis[:] *= mask

        # Modify the noise weights
        sstream.weight[:] *= mask**2

        return sstream


def _list_of_timeranges(dlist):

    if not isinstance(list, dlist):
        pass


class Unroll(pipeline.TaskBase):
    """Unroll a sidereal stack.

    Not really tested at the moment.

    Parameters
    ----------
    start_time, end_time : float
        Start and end UNIX times of the timestream to simulate.
    int_time : float
        Integration time in seconds.

    """

    start_time = config.Property(proptype=float)
    end_time = config.Property(proptype=float)

    int_time = config.Property(proptype=float)

    def setup(self, telescope):
        self.telescope = telescope

    def next(self, sstream):

        from ch_analysis.map import sidereal

        tel = self.telescope

        # Use Kiyo's code to unroll the timestream in time.
        sunroll, times = sidereal.unroll_stream(sstream.ra, sstream.data, self.start_time, self.end_time, self.int_time)
        sunroll = mpiarray.MPIArray.wrap(sunroll, 0, comm=sstream.comm)

        allpair = tel.nfeed * (tel.nfeed + 1) / 2

        tstream = containers.TimeStream(times, tel.nfreq, allpair)

        # Iterate over all feed pairs and work out which is the correct index in the sidereal stack.
        for fi in range(tel.nfeed):
            for fj in range(fi, tel.nfeed):
                pair_ind = tools.cmap(fi, fj, tel.nfeed)
                upair_ind = tel.feedmap[fi, fj]

                # Skip if upair is not a proper index (has been masked)
                if upair_ind < 0:
                    continue

                tstream.data[:, pair_ind] = sunroll[:, upair_ind]

        return sstream
