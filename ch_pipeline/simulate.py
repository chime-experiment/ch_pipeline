"""
=====================================================
Timestreaam Simulation (:mod:`~ch_pipeline.simulate`)
=====================================================

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

import h5py
import numpy as np

from cora.util import hputil
from caput import mpiutil, pipeline, config, mpidataset

from ch_util import tools
from ch_analysis.map import sidereal

from . import containers


class LoadBeamTransfer(pipeline.TaskBase):

    product_directory = config.Property(proptype=str)

    def setup(self):
        """Load the beam transfer matrices.

        Returns
        -------
        tel : TransitTelescope
            Object describing the telescope.
        bt : BeamTransfer
            BeamTransfer manager.
        """

        from drift.core import beamtransfer

        bt = beamtransfer.BeamTransfer(self.product_directory)

        tel = bt.telescope

        return tel, bt, tel.feeds


class SimulateSidereal(pipeline.TaskBase):
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

    def next(self):
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

        ## Read in telescope system
        bt = self.beamtransfer
        tel = self.telescope

        lmax = tel.lmax
        mmax = tel.mmax
        nfreq = tel.nfreq
        npol = tel.num_pol_sky

        lfreq, sfreq, efreq = mpiutil.split_local(nfreq)
        #local_freq = range(sfreq, efreq)

        lm, sm, em = mpiutil.split_local(mmax + 1)

        # Set the minimum resolution required for the sky.
        ntime = 2*mmax+1

        col_vis = np.zeros((tel.npairs, lfreq, ntime), dtype=np.complex128)

        ## If we want to add maps use the m-mode formalism to project a skymap
        ## into visibility space.

        # Load file to find out the map shapes.
        with h5py.File(self.maps[0], 'r') as f:
            mapshape = f['map'].shape

        if lfreq > 0:

            # Allocate array to store the local frequencies
            row_map = np.zeros((lfreq,) + mapshape[1:], dtype=np.float64)

            # Read in and sum up the local frequencies of the supplied maps.
            for mapfile in self.maps:
                with h5py.File(mapfile, 'r') as f:
                    row_map += f['map'][sfreq:efreq]

            # Calculate the alm's for the local sections
            row_alm = hputil.sphtrans_sky(row_map, lmax=lmax).reshape((lfreq, npol * (lmax+1), lmax+1))

        else:
            row_alm = np.zeros((lfreq, npol * (lmax+1), lmax+1), dtype=np.complex128)

        # Perform the transposition to distribute different m's across processes. Neat
        # tip, putting a shorter value for the number of columns, trims the array at
        # the same time
        col_alm = mpiutil.transpose_blocks(row_alm, (nfreq, npol * (lmax+1), mmax+1))

        # Transpose and reshape to shift m index first.
        col_alm = np.transpose(col_alm, (2, 0, 1)).reshape(lm, nfreq, npol, lmax+1)

        # Create storage for visibility data
        vis_data = np.zeros((lm, nfreq, bt.ntel), dtype=np.complex128)

        # Iterate over m's local to this process and generate the corresponding
        # visibilities
        for mp, mi in enumerate(range(sm, em)):
            vis_data[mp] = bt.project_vector_sky_to_telescope(mi, col_alm[mp])

        # Rearrange axes such that frequency is last (as we want to divide
        # frequencies across processors)
        row_vis = vis_data.transpose((0, 2, 1))

        # Parallel transpose to get all m's back onto the same processor
        col_vis_tmp = mpiutil.transpose_blocks(row_vis, ((mmax+1), bt.ntel, nfreq))
        col_vis_tmp = col_vis_tmp.reshape(mmax + 1, 2, tel.npairs, lfreq)

        # Transpose the local section to make the m's the last axis and unwrap the
        # positive and negative m at the same time.
        col_vis[..., 0] = col_vis_tmp[0, 0]
        for mi in range(1, mmax+1):
            col_vis[...,  mi] = col_vis_tmp[mi, 0]
            col_vis[..., -mi] = col_vis_tmp[mi, 1].conj()  # Conjugate only (not (-1)**m - see paper)

        del col_vis_tmp

        ## If we're simulating noise, create a realisation and add it to col_vis
        if self.ndays > 0:

            # Fetch the noise powerspectrum
            noise_ps = tel.noisepower(np.arange(tel.npairs)[:, np.newaxis],
                                      np.arange(sfreq, efreq)[np.newaxis, :],
                                      ndays=self.ndays).reshape(tel.npairs, lfreq)[:, :, np.newaxis]

            # Seed random number generator to give consistent noise
            if self.seed is not None:
                # Must include rank such that we don't have massive power deficit from correlated noise
                np.random.seed(self.seed + mpiutil.rank)

            # Create and weight complex noise coefficients
            noise_vis = (np.array([1.0, 1.0J]) * np.random.standard_normal(col_vis.shape + (2,))).sum(axis=-1)
            noise_vis *= (noise_ps / 2.0)**0.5

            # Reset RNG
            if self.seed is not None:
                np.random.seed()

            # Add into main noise sims
            col_vis += noise_vis

            del noise_vis

        # Fourier transform m-modes back to get timestream.
        vis_stream = np.fft.ifft(col_vis, axis=-1) * ntime
        vis_stream = vis_stream.reshape(tel.npairs, lfreq, ntime)
        vis_stream = vis_stream.transpose((1, 0, 2)).copy()

        # Construct frequency array
        freq = np.zeros(nfreq, dtype=[('centre', np.float64), ('width', np.float64)])
        freq['centre'][:] = tel.frequencies
        freq['width'][:] = np.diff(tel.frequencies)[0]

        sstream = containers.SiderealStream(ntime, freq, tel.npairs)
        sstream['vis'][:] = mpidataset.MPIArray.wrap(vis_stream, axis=0)

        sstream._distributed['weight'] = mpidataset.MPIArray.wrap(np.ones(vis_stream.shape, dtype=np.float64), axis=0)

        self.done = True

        feeds = self.telescope.feeds

        return sstream, feeds


class DayMask(pipeline.TaskBase):
    """Crudely simulate a masking out of the daytime data.

    Attributes
    ----------
    start, end : float
        Start and end of masked out region.
    width : float
        Use a smooth transition of given width between the fully masked and
        unmasked data. This is interior to the region marked by start and end.
    """

    start = config.Property(proptype=float, default=90.0)
    end = config.Property(proptype=float, default=270.0)

    width = config.Property(proptype=float, default=60.0)

    def next(self, sstream):
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

        ra = sstream.ra

        # Crudely mask the on and off regions
        mask = np.logical_or(ra < self.start, ra > self.end).astype(np.float64)

        # Put in the transition at the start of the day
        mask = np.where(np.logical_and(ra > self.start, ra < self.start + self.width),
                        0.5 * (1 + np.cos(np.pi * ((ra - self.start) / self.width))),
                        mask)

        # Put the transition at the end of the day
        mask = np.where(np.logical_and(ra > self.end - self.width, ra < self.end),
                        0.5 * (1 + np.cos(np.pi * ((ra - self.end) / self.width))),
                        mask)

        # Apply the mask to the data
        sstream.vis[:] *= mask

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

        tel = self.telescope

        # Use Kiyo's code to unroll the timestream in time.
        sunroll, times = sidereal.unroll_stream(sstream.ra, sstream.data, self.start_time, self.end_time, self.int_time)
        sunroll = mpidataset.MPIArray.wrap(sunroll, 0, comm=sstream.comm)

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
