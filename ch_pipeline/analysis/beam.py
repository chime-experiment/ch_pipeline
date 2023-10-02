"""Tasks for beam measurement processing"""
import json
import yaml
from os import path, listdir
from itertools import groupby
from operator import itemgetter

import numpy as np
import scipy.signal
from scipy import constants

from caput import config, tod, mpiarray, mpiutil
from caput.pipeline import PipelineRuntimeError
from caput.time import STELLAR_S

from ch_util import ephemeris as ephem
from ch_util import tools, layout, holography
from chimedb import data_index as di
from chimedb.core import connect as connect_database


from draco.core import task, io
from draco.util import regrid
from draco.analysis.transform import Regridder
from draco.core.containers import ContainerBase, SiderealStream, TimeStream, TrackBeam
from draco.util.tools import invert_no_zero

from ..core.containers import TransitFitParams, MultiSiderealStream, MultiTimeStream
from .calibration import TransitFit, GainFromTransitFit
from .flagging import taper_mask

SIDEREAL_DAY_SEC = STELLAR_S * 24 * 3600
SPEED_LIGHT = float(constants.c) / 1e6  # 10^6 m / s
CHIME_CYL_W = 22.0  # m


class TransitGrouper(task.SingleTask):
    """Group transits from a sequence of TimeStream objects.

    Attributes
    ----------
    ha_span: float
        Span in degrees surrounding transit.
    min_span: float
        Minimum span (deg) of a transit to accept.
    source: str
        Name of the transiting source. (Must match what is used in `ch_util.ephemeris`.)
    db_source: str
        Name of the transiting source as listed in holography database.
        This is a hack until a better solution is implemented.
    """

    ha_span = config.Property(proptype=float, default=180.0)
    min_span = config.Property(proptype=float, default=0.0)
    source = config.Property(proptype=str)
    db_source = config.Property(proptype=str)

    def setup(self, observer=None):
        """Set the local observers position if not using CHIME.

        Parameters
        ----------
        observer : caput.time.Observer, optional
            Details of the observer, if not set default to CHIME.

        Raises
        ------
        `caput.config.CaputConfigError`
            If config value ``source`` doesn't match any in `ch_util.ephemeris`.
        """
        self.observer = ephem.chime if observer is None else observer
        try:
            self.src = ephem.source_dictionary[self.source]
        except KeyError:
            msg = (
                "Could not find source {} in catalogue. "
                "Must use same spelling as in `ch_util.ephemeris`.".format(self.source)
            )
            self.log.error(msg)
            raise config.CaputConfigError(msg)
        self.cur_transit = None
        self.tstreams = []
        self.last_time = 0

        # Get list of holography observations
        # Only allowed to query database from rank0
        db_runs = None
        if mpiutil.rank0:
            connect_database()
            db_runs = list(get_holography_obs(self.db_source))
            db_runs = [(int(r.id), (r.start_time, r.finish_time)) for r in db_runs]
        self.db_runs = mpiutil.bcast(db_runs, root=0)
        mpiutil.barrier()

    def process(self, tstream):
        """Take in a timestream and accumulate, group into whole transits.

        Parameters
        ----------
        tstream : TimeStream

        Returns
        -------
        ts : TimeStream
            Timestream with containing transits of specified length.
        """

        # Redistribute if needed
        tstream.redistribute("freq")

        # placeholder for finalized transit when it is ready
        final_ts = None

        # check if we jumped to another acquisition
        if (tstream.time[0] - self.last_time) > 5 * (tstream.time[1] - tstream.time[0]):
            if self.cur_transit is None:
                # will be true when we start a new transit
                pass
            else:
                # start on a new transit and return current one
                final_ts = self._finalize_transit()

        # if this is the start of a new grouping, setup transit bounds
        if self.cur_transit is None:
            tr_time = self.observer.transit_times(self.src, tstream.time[0])
            if len(tr_time) != 1:
                raise ValueError(
                    "Didn't find exactly one transit time. Found {:d}.".format(
                        len(tr_time)
                    )
                )
            self.cur_transit = tr_time[0]
            self._transit_bounds()

        # check if we've accumulated enough past the transit
        if tstream.time[-1] > self.end_t:
            self.append(tstream)
            final_ts = self._finalize_transit()
        elif tstream.time[-1] < self.start_t:
            pass
        else:
            self.append(tstream)

        self.last_time = tstream.time[-1]

        return final_ts

    def process_finish(self):
        """Return the current transit before finishing.

        Returns
        -------
        ts: TimeStream
            Last (possibly incomplete) transit.
        """
        return self._finalize_transit()

    def append(self, ts):
        """Append a timestream to the buffer list.

        This will strip eigenvector datasets if they are present.
        """
        for dname in ["evec", "eval", "erms"]:
            if dname in ts.datasets.keys():
                self.log.debug("Stripping dataset {}".format(dname))
                del ts[dname]
        self.tstreams.append(ts)

    def _finalize_transit(self):
        """Concatenate grouped time streams for the currrent transit."""

        # Find where transit starts and ends
        if len(self.tstreams) == 0 or self.cur_transit is None:
            self.log.info("Did not find any transits.")
            return None
        self.log.debug(
            "Finalising transit for {}...".format(
                ephem.unix_to_datetime(self.cur_transit)
            )
        )
        all_t = np.concatenate([ts.time for ts in self.tstreams])
        start_ind = int(np.argmin(np.abs(all_t - self.start_t)))
        stop_ind = int(np.argmin(np.abs(all_t - self.end_t)))

        # Save list of filenames
        filenames = [ts.attrs["filename"] for ts in self.tstreams]

        dt = self.tstreams[0].time[1] - self.tstreams[0].time[0]
        if dt <= 0:
            self.log.warning(
                "Time steps are not positive definite: dt={:.3f}".format(dt)
                + " Skipping."
            )
            ts = None
        if stop_ind - start_ind > int(self.min_span / 360.0 * SIDEREAL_DAY_SEC / dt):
            if len(self.tstreams) > 1:
                # Concatenate timestreams
                ts = tod.concatenate(self.tstreams, start=start_ind, stop=stop_ind)
            else:
                ts = self.tstreams[0]
            _, dec = ephem.object_coords(
                self.src, all_t[0], deg=True, obs=self.observer
            )
            ts.attrs["dec"] = dec
            ts.attrs["source_name"] = self.source
            ts.attrs["transit_time"] = self.cur_transit
            ts.attrs["observation_id"] = self.obs_id
            ts.attrs["tag"] = "{}_{:0>4d}_{}".format(
                self.source,
                self.obs_id,
                ephem.unix_to_datetime(self.cur_transit).strftime("%Y%m%dT%H%M%S"),
            )
            ts.attrs["archivefiles"] = filenames
        else:
            self.log.info("Transit too short. Skipping.")
            ts = None

        self.tstreams = []
        self.cur_transit = None

        return ts

    def _transit_bounds(self):
        """Find the start and end times of this transit.

        Compares the desired HA span to the start and end times of the observation
        recorded in the database. Also gets the observation ID."""

        # subtract half a day from start time to ensure we don't get following day
        self.start_t = self.cur_transit - self.ha_span / 360.0 / 2.0 * SIDEREAL_DAY_SEC
        self.end_t = self.cur_transit + self.ha_span / 360.0 / 2.0 * SIDEREAL_DAY_SEC

        # get bounds of observation from database
        this_run = [
            r
            for r in self.db_runs
            if r[1][0] < self.cur_transit and r[1][1] > self.cur_transit
        ]
        if len(this_run) == 0:
            self.log.warning(
                "Could not find source transit in holography database for {}.".format(
                    ephem.unix_to_datetime(self.cur_transit)
                )
            )
            # skip this file
            self.cur_transit = None
        else:
            self.start_t = max(self.start_t, this_run[0][1][0])
            self.end_t = min(self.end_t, this_run[0][1][1])
            self.obs_id = this_run[0][0]


class TaperBeam(task.SingleTask):

    ntaper = config.Property(proptype=int, default=10)

    def process(self, data):

        data.redistribute("freq")

        nfreq, npol, ninput, nha = data.beam.local_shape

        window = np.hanning(2 * self.ntaper - 1)[1:-1]
        nkeep = self.ntaper - 1

        # Dereference the datasets
        beam = data.beam[:].view(np.ndarray)
        weight = data.weight[:].view(np.ndarray)

        # Loop over frequencies to reduce memory usage
        for ff in range(nfreq):

            for pp in range(npol):

                for ii in range(ninput):

                    ind = (ff, pp, ii)

                    valid = np.flatnonzero(weight[ind] > 0.0)

                    if valid.size == 0:
                        continue

                    aa = valid[0]
                    bb = aa + nkeep

                    dd = valid[-1] + 1
                    cc = dd - nkeep

                    if bb > cc:
                        weight[ind] = 0.0
                        continue

                    beam[ind + (slice(aa, bb),)] *= window[:nkeep]
                    beam[ind + (slice(cc, dd),)] *= window[-nkeep:]

        return data


class FilterBeam(task.SingleTask):

    ncut = config.Property(proptype=float, default=2.0)
    ntransition = config.Property(proptype=float, default=1.0)

    def process(self, data):

        data.redistribute("freq")

        ifs = data.beam.local_offset[0]
        ife = ifs + data.beam.local_shape[0]

        local_freq = data.freq[ifs:ife]

        # Extract coordinates
        phi = np.radians(data.pix["phi"])
        theta = np.radians(np.median(data.pix["theta"]))

        # Determine the m-modes probed by the holography measurement
        dphi = np.median(np.abs(np.diff(phi)))
        nphi = phi.size

        m = np.fft.fftfreq(nphi, d=dphi / (2.0 * np.pi))
        dm = m[1] - m[0]

        # Dereference the datasets
        beam = data.beam[:].view(np.ndarray)
        weight = data.weight[:].view(np.ndarray)

        # Loop over frequencies to reduce memory usage
        for ff, nu in enumerate(local_freq):

            mwidth = np.pi * nu * CHIME_CYL_W * np.cos(theta) / SPEED_LIGHT
            mcut = self.ncut * mwidth
            ntransition = int(self.ntransition * mwidth // dm)

            h = 1.0 - taper_mask((np.abs(m) > mcut).astype(np.float32), ntransition)
            flag = (weight[ff] > 0.0).astype(np.float32)

            beam[ff] = np.fft.ifft(np.fft.fft(beam[ff] * flag, axis=-1) * h, axis=-1)

            c = np.sum(h**2) / nphi**2
            varf = c * np.sum(tools.invert_no_zero(weight[ff]), axis=-1, keepdims=True)
            weight[ff] = tools.invert_no_zero(varf)

        return data


class TransitRegridder(Regridder):
    """Interpolate TimeStream transits onto a regular grid in hour angle.

    Attributes
    ----------
    samples : int
        Number of samples to interpolate onto.
    ha_span: float
        Span in degrees surrounding transit.
    lanczos_width : int
        Width of the Lanczos interpolation kernel.
    snr_cov: float
        Ratio of signal covariance to noise covariance (used for Wiener filter).
    source: str
        Name of the transiting source. (Must match what is used in `ch_util.ephemeris`.)
    """

    samples = config.Property(proptype=int, default=1024)
    lanczos_width = config.Property(proptype=int, default=5)
    snr_cov = config.Property(proptype=float, default=1e-8)
    ha_span = config.Property(proptype=float, default=180.0)
    source = config.Property(proptype=str)

    def setup(self, observer=None):
        """Set the local observers position if not using CHIME.

        Parameters
        ----------
        observer : caput.time.Observer, optional
            Details of the observer, if not set default to CHIME.

        Raises
        ------
        `caput.config.CaputConfigError`
            If config value ``source`` doesn't match any in `ch_util.ephemeris`.
        """
        self.observer = ephem.chime if observer is None else observer

        # Setup bounds for interpolation grid
        self.start = -self.ha_span / 2
        self.end = self.ha_span / 2

        try:
            self.src = ephem.source_dictionary[self.source]
        except KeyError:
            msg = (
                "Could not find source {} in catalogue. "
                "Must use same spelling as in `ch_util.ephemeris`.".format(self.source)
            )
            self.log.error(msg)
            raise config.CaputConfigError(msg)

    def process(self, data):
        """Regrid visibility data onto a regular grid in hour angle.

        Parameters
        ----------
        data : TimeStream
            Time-ordered data.

        Returns
        -------
        new_data : SiderealStream
            The regridded data centered on the source RA.
        """

        # Redistribute if needed
        data.redistribute("freq")

        # View of data
        weight = data.weight[:].view(np.ndarray)
        vis_data = data.vis[:].view(np.ndarray)

        # Get apparent source RA, including precession effects
        ra, _ = ephem.object_coords(self.src, data.time[0], deg=True, obs=self.observer)
        # Get catalogue RA for reference
        ra_icrs, _ = ephem.object_coords(self.src, deg=True, obs=self.observer)

        # Convert input times to hour angle
        lha = unwrap_lha(self.observer.unix_to_lsa(data.time), ra)

        # perform regridding
        success = 1
        try:
            new_grid, new_vis, ni = self._regrid(vis_data, weight, lha)
        except np.linalg.LinAlgError as e:
            self.log.error(str(e))
            success = 0
        except ValueError as e:
            self.log.error(str(e))
            success = 0
        # Check other ranks have completed
        success = mpiutil.allreduce(success)
        if success != mpiutil.size:
            self.log.warning("Regridding failed. Skipping transit.")
            return None

        # mask out regions beyond bounds of this transit
        grid_mask = np.ones_like(new_grid)
        grid_mask[new_grid < lha.min()] = 0.0
        grid_mask[new_grid > lha.max()] = 0.0
        new_vis *= grid_mask
        ni *= grid_mask

        # Wrap to produce MPIArray
        if data.distributed:
            new_vis = mpiarray.MPIArray.wrap(new_vis, axis=data.vis.distributed_axis)
            ni = mpiarray.MPIArray.wrap(ni, axis=data.vis.distributed_axis)

        # Create new container for output
        ra_grid = (new_grid + ra) % 360.0
        new_data = SiderealStream(
            axes_from=data, attrs_from=data, ra=ra_grid, comm=data.comm
        )
        new_data.redistribute("freq")
        new_data.vis[:] = new_vis
        new_data.weight[:] = ni
        new_data.attrs["cirs_ra"] = ra
        new_data.attrs["icrs_ra"] = ra_icrs

        return new_data


class EdgeFlagger(task.SingleTask):
    """Flag the edges of the transit.

    Parameters
    ----------
    num_begin: int
        Number of samples to flag at the start of the transit.
    num_end: int
        Number of samples to flag at the end of the transit.
    """

    num_begin = config.Property(proptype=int, default=15)
    num_end = config.Property(proptype=int, default=15)

    def process(self, track):
        """Extend the region that has weight set to zero.

        Parameters
        ----------
        track: draco.core.containers.TrackBeam
            Holography track to flag.

        Returns
        -------
        track: draco.core.containers.TrackBeam
            Holography track with weight set to zero for
            `num_begin` samples at the start of the transit
            and `num_end` samples at the end.
        """
        if (self.num_begin == 0) and (self.num_end == 0):
            return track

        track.redistribute("freq")

        weight = track["weight"][:].view(np.ndarray)

        for ind in np.ndindex(*weight.shape[:-1]):

            flag = np.flatnonzero(weight[ind] > 0.0)

            if flag.size > 0:

                imin, imax = np.percentile(flag, [0, 100]).astype(np.int)
                imax = imax + 1

                if self.num_begin:
                    weight[ind][imin : imin + self.num_begin] = 0.0

                if self.num_end:
                    weight[ind][imax - self.num_end : imax] = 0.0

        return track


class FlagBeam(task.SingleTask):

    threshold = config.Property(proptype=float, default=2.0)

    def process(self, data):

        b = data.beam[:].local_array
        w = data.weight[:].local_array

        data.weight[:].local_array[:] *= ((w > self.threshold) & (np.abs(b) > 0.0)).astype(np.float32)

        return data


class TransitResampler(task.SingleTask):
    """Resample the beam at specific RAs.

    Attributes
    ----------
    lanczos_width : int
        Width of the Lanczos kernel in number of samples.
    """

    lanczos_width = config.Property(proptype=int, default=5)

    def setup(self, observer=None):
        """Set the local observers position if not using CHIME.

        Parameters
        ----------
        observer : caput.time.Observer, optional
            Details of the observer, if not set default to CHIME.
        """
        self.observer = ephem.chime if observer is None else observer

    def process(self, beam, data):
        """Resample the measured beam at arbitrary RA by convolving with a Lanczos kernel.

        Parameters
        ----------
        beam : draco.core.containers.TrackBeam
            The beam that will be resampled.
        data : draco.core.containers.VisContainer
            Must contain a `time` or `ra` axis that defines
            the RAs to sample.

        Returns
        -------
        new_beam : draco.core.containers.TrackBeam
             The input `beam` re-sampled at the RAs contained in `data`.
        """
        from mpi4py import MPI

        # Distribute over frequencies
        data.redistribute("freq")
        beam.redistribute("freq")

        # Dereference datasets
        bv = beam.beam[:].view(np.ndarray)
        bw = beam.weight[:].view(np.ndarray)

        flag_local = np.any(bw > 0.0, axis=(0, 1, 2))
        flag = np.zeros_like(flag_local)
        beam.comm.Allreduce(flag_local, flag, op=MPI.LOR)

        # Compute the hour angle of the source
        if "ra" in data.index_map:
            phi = data.index_map["ra"][:]
            ra = phi
            coords = "cirs_ra-dec"
        elif "time" in data.index_map:
            phi = data.time
            ra = self.observer.unix_to_lsa(phi)
            coords = "time-dec"
        else:
            raise RuntimeError("Unable to extract RA from input container.")

        hour_angle = beam.pix["phi"][:]

        new_hour_angle = (((ra - beam.attrs["cirs_ra"]) + 180.0) % 360.0) - 180.0

        isort = np.argsort(new_hour_angle)
        new_hour_angle_sorted = new_hour_angle[isort]

        within_range = np.flatnonzero(
            (new_hour_angle_sorted >= np.min(hour_angle[flag]))
            & (new_hour_angle_sorted <= np.max(hour_angle[flag]))
        )

        if within_range.size < (2 * self.lanczos_width):
            raise RuntimeError("Not enough overlapping samples.")

        slc = slice(
            np.min(within_range) + self.lanczos_width,
            np.max(within_range) - self.lanczos_width + 1,
        )

        new_hour_angle_sorted = new_hour_angle_sorted[slc]

        # Perform regridding
        new_bv, new_bw = self._resample(hour_angle, bv, bw, new_hour_angle_sorted)

        # Figure out mapping back to the axis (time or RA) in the data container
        new_phi = phi[isort[slc]]
        resort = np.argsort(new_phi)
        new_phi = new_phi[resort]

        # Create output container
        new_beam = TrackBeam(
            phi=new_phi,
            theta=np.repeat(np.median(beam.pix["theta"][:]), new_phi.size),
            axes_from=beam,
            attrs_from=beam,
            distributed=beam.distributed,
            comm=beam.comm,
        )

        new_beam.redistribute("freq")

        new_beam.attrs["coords"] = coords

        # Save to container
        new_beam.beam[:] = new_bv[..., resort]
        new_beam.weight[:] = new_bw[..., resort]

        return new_beam

    def _resample(self, xgrid, ygrid, wgrid, x):

        lza = regrid.lanczos_forward_matrix(xgrid, x, a=self.lanczos_width).T

        y = np.matmul(ygrid, lza)
        w = invert_no_zero(np.matmul(invert_no_zero(wgrid), lza**2))

        return y, w


class MakeHolographyBeam(task.SingleTask):
    """Repackage a holography transit into a beam container.

    The visibilities will be grouped according to their respective 26 m
    input (along the `pol` axis, labelled by the 26 m polarisation of that input).

    The form of the beam dataset is A_{i} = < E_{i} * E_{26m}^{*} >, i.e. the 26m
    input is the conjugate part of the visbility.
    """

    def process(self, data, inputmap):
        """Package a holography transit into a beam container.

        Parameters
        ----------
        data : SiderealStream
            Transit observation as generated by `TransitRegridder`.
        inputmap : list of `CorrInput`
            A list describing the inputs as they are in the file, output from
            `ch_util.tools.get_correlator_inputs()`

        Returns
        -------
        track : TrackBeam
            The transit in a beam container.
        """

        # redistribute if needed
        data.redistribute("freq")

        prod = data.index_map["prod"]
        inputs = data.index_map["input"]

        # Figure out which inputs are the 26m
        input_26m = prod["input_a"][np.where(prod["input_a"] == prod["input_b"])[0]]
        if len(input_26m) != 2:
            msg = "Did not find exactly two 26m inputs in the data."
            self.log.error(msg)
            raise PipelineRuntimeError(msg)

        # Separate products by 26 m inputs
        prod_groups = []
        for i in input_26m:
            prod_groups.append(
                np.where(np.logical_or(prod["input_a"] == i, prod["input_b"] == i))[0]
            )

        # Check we have the expected number of products
        if (
            prod_groups[0].shape[0] != inputs.shape[0]
            or prod_groups[1].shape[0] != inputs.shape[0]
        ):
            msg = (
                "Products do not separate into two groups with the length of the input map. "
                "({:d}, {:d}) != {:d}"
            ).format(prod_groups[0].shape[0], prod_groups[1].shape[0], inputs.shape[0])
            self.log.error(msg)
            raise PipelineRuntimeError(msg)

        # Sort based on the id in the layout database
        corr_id = np.array([inp.id for inp in inputmap])
        isort = np.argsort(corr_id)

        # Create new input axis using id and serial number in database
        inputs_sorted = np.array(
            [(inputmap[ii].id, inputmap[ii].input_sn) for ii in isort],
            dtype=inputs.dtype,
        )

        # Sort the products based on the input id in database and
        # determine which products should be conjugated.
        conj = []
        prod_groups_sorted = []
        for i, pg in enumerate(prod_groups):
            group_prod = prod[pg]
            group_conj = group_prod["input_a"] == input_26m[i]
            group_inputs = np.where(
                group_conj, group_prod["input_b"], group_prod["input_a"]
            )
            group_sort = np.argsort(corr_id[group_inputs])

            prod_groups_sorted.append(pg[group_sort])
            conj.append(group_conj[group_sort])

        # Regroup by co/cross-pol
        copol, xpol = [], []
        prod_groups_cox = [pg.copy() for pg in prod_groups_sorted]
        conj_cox = [pg.copy() for pg in conj]
        input_pol = np.array(
            [
                ipt.pol
                if (tools.is_array(ipt) or tools.is_holographic(ipt))
                else inputmap[input_26m[0]].pol
                for ipt in inputmap
            ]
        )
        for i, pg in enumerate(prod_groups_sorted):
            group_prod = prod[pg]
            # Determine co/cross in each prod group
            cp = (
                input_pol[
                    np.where(conj[i], group_prod["input_b"], group_prod["input_a"])
                ]
                == inputmap[input_26m[i]].pol
            )
            xp = np.logical_not(cp)
            copol.append(cp)
            xpol.append(xp)
            # Move products to co/cross-based groups
            prod_groups_cox[0][cp] = pg[cp]
            prod_groups_cox[1][xp] = pg[xp]
            conj_cox[0][cp] = conj[i][cp]
            conj_cox[1][xp] = conj[i][xp]
        # Check for compeleteness
        consistent = np.all(copol[0] + copol[1] == np.ones(copol[0].shape)) and np.all(
            xpol[0] + xpol[1] == np.ones(xpol[0].shape)
        )
        if not consistent:
            msg = (
                "Products do not separate exclusively into co- and cross-polar groups."
            )
            self.log.error(msg)
            raise PipelineRuntimeError(msg)

        # Make new index map
        ra = data.attrs["cirs_ra"]
        phi = unwrap_lha(data.ra[:], ra)
        if "dec" not in data.attrs.keys():
            msg = (
                "Input stream must have a 'dec' attribute specifying "
                "declination of holography source."
            )
            self.log.error(msg)
            raise PipelineRuntimeError(msg)
        theta = np.ones_like(phi) * data.attrs["dec"]
        pol = np.array(["co", "cross"], dtype="S5")

        # Create new container and fill
        track = TrackBeam(
            theta=theta,
            phi=phi,
            track_type="drift",
            coords="celestial",
            input=inputs_sorted,
            pol=pol,
            freq=data.freq[:],
            attrs_from=data,
            distributed=data.distributed,
        )
        for ip in range(len(pol)):
            track.beam[:, ip, :, :] = data.vis[:, prod_groups_cox[ip], :]
            track.weight[:, ip, :, :] = data.weight[:, prod_groups_cox[ip], :]
            if np.any(conj_cox[ip]):
                track.beam[:, ip, conj_cox[ip], :] = track.beam[
                    :, ip, conj_cox[ip], :
                ].conj()

        # Store 26 m inputs
        track.attrs["26m_inputs"] = [list(isort).index(ii) for ii in input_26m]

        return track


class ConstructStackedBeam(task.SingleTask):
    """Construct the effective beam for stacked baselines.

    Parameters
    ----------
    weight : string ('uniform' or 'inverse_variance')
        How to weight the baselines when stacking:
            'uniform' - each baseline given equal weight
            'inverse_variance' - each baseline weighted by the weight attribute
    """

    weight = config.enum(["uniform", "inverse_variance"], default="uniform")

    def setup(self, tel):
        """Set the Telescope instance to use.

        Parameters
        ----------
        tel : TransitTelescope
        """
        self.telescope = io.get_telescope(tel)

    def process(self, beam, data):
        """Stack

        Parameters
        ----------
        beam : TrackBeam
            The beam that will be stacked.
        data : VisContainer
            Must contain `prod` index map and `stack` reverse map
            that will be used to stack the beam.

        Returns
        -------
        stacked_beam: VisContainer
            The input `beam` stacked in the same manner as

        """
        # Distribute over frequencies
        data.redistribute("freq")
        beam.redistribute("freq")

        # Grab the stack specifications from the input sidereal stream
        prod = data.index_map["prod"]
        reverse_stack = data.reverse_map["stack"][:]

        input_flags = data.input_flags[:]
        if not np.any(input_flags):
            input_flags = np.ones_like(input_flags)

        # Create output container
        if isinstance(data, SiderealStream):
            OutputContainer = SiderealStream
            output_kwargs = {"ra": data.ra[:]}
        else:
            OutputContainer = TimeStream
            output_kwargs = {"time": data.time[:]}

        stacked_beam = OutputContainer(
            axes_from=data,
            attrs_from=beam,
            distributed=True,
            comm=data.comm,
            **output_kwargs,
        )

        stacked_beam.vis[:] = 0.0
        stacked_beam.weight[:] = 0.0

        stacked_beam.attrs["tag"] = "_".join([beam.attrs["tag"], data.attrs["tag"]])

        # Dereference datasets
        bv = beam.beam[:].view(np.ndarray)
        bw = beam.weight[:].view(np.ndarray)

        ov = stacked_beam.vis[:]
        ow = stacked_beam.weight[:]

        pol_filter = {
            "X": "X",
            "Y": "Y",
            "E": "X",
            "S": "Y",
            "co": "co",
            "cross": "cross",
        }
        pol = [pol_filter.get(pp, None) for pp in self.telescope.polarisation]
        beam_pol = [pol_filter.get(pp, None) for pp in beam.index_map["pol"][:]]

        nstack = ov.shape[1]

        # Compute the fractional variance of the beam measurement
        frac_var = invert_no_zero(bw * np.abs(bv) ** 2)

        # Create counter to increment during the stacking.
        # This will be used to normalize at the end.
        counter = np.zeros_like(ow)

        # Construct stack
        for pp, (ss, conj) in enumerate(reverse_stack):

            if (ss < 0) or (ss > (nstack - 1)):
                continue

            aa, bb = prod[pp]
            if conj:
                aa, bb = bb, aa

            try:
                aa_pol, bb_pol = self._resolve_pol(pol[aa], pol[bb], beam_pol)
            except ValueError:
                continue

            cross = bv[:, aa_pol, aa, :] * bv[:, bb_pol, bb, :].conj()

            weight = (
                input_flags[np.newaxis, aa, :]
                * input_flags[np.newaxis, bb, :]
                * invert_no_zero(
                    np.abs(cross) ** 2
                    * (frac_var[:, aa_pol, aa, :] + frac_var[:, bb_pol, bb, :])
                )
            )

            if self.weight == "inverse_variance":
                wss = weight
            else:
                wss = (weight > 0.0).astype(np.float32)

            # Accumulate variances in quadrature.  Save in the weight dataset.
            ov[:, ss, :] += wss * cross
            ow[:, ss, :] += wss**2 * invert_no_zero(weight)

            # Increment counter
            counter[:, ss, :] += wss

        # Divide through by counter to get properly weighted visibility average
        ov[:] *= invert_no_zero(counter)
        ow[:] = counter**2 * invert_no_zero(ow[:])

        return stacked_beam

    @staticmethod
    def _resolve_pol(pol1, pol2, pol_axis):

        if "co" in pol_axis:

            if pol1 == pol2:
                ipol = pol_axis.index("co")
            else:
                ipol = pol_axis.index("cross")

            return ipol, ipol

        else:

            if pol1 == pol2:
                ipol1 = pol_axis.index(pol1)
                ipol2 = pol_axis.index(pol2)
            else:
                ipol1 = pol_axis.index(pol2)
                ipol2 = pol_axis.index(pol1)

            return ipol1, ipol2


def find_contiguous_groups(index):
    ranges = []
    for key, group in groupby(enumerate(index), lambda i: i[0] - i[1]):
        group = list(map(itemgetter(1), group))
        ranges.append(slice(group[0], group[-1] + 1))
    return ranges


class ConstructMultiStackedBeam(task.SingleTask):
    """Construct the effective beam for stacked baselines.

    Parameters
    ----------
    weight : string ('uniform' or 'inverse_variance')
        How to weight the baselines when stacking:
            'uniform' - each baseline given equal weight
            'inverse_variance' - each baseline weighted by the weight attribute
    """

    nstream = config.enum([1, 2, 4], default=2)

    def setup(self, tel):
        """Set the Telescope instance to use.

        Parameters
        ----------
        tel : TransitTelescope
        """
        self.telescope = io.get_telescope(tel)

    def process(self, beam, data):
        """Stack

        Parameters
        ----------
        beam : TrackBeam
            The beam that will be stacked.
        data : VisContainer
            Must contain `prod` index map and `stack` reverse map
            that will be used to stack the beam.

        Returns
        -------
        stacked_beam: VisContainer
            The input `beam` stacked in the same manner as

        """
        # Distribute over frequencies
        data.redistribute("freq")
        beam.redistribute("freq")

        # Grab the stack specifications from the input sidereal stream
        prod = data.index_map["prod"]
        reverse_stack = data.reverse_map["stack"][:]

        # Create output container
        output_kwargs = {"stream": np.arange(self.nstream, dtype=int)}
        if isinstance(data, SiderealStream):
            OutputContainer = MultiSiderealStream
            imatch = np.searchsorted(data.ra, beam.pix["phi"])
            output_kwargs.update(ra=data.ra[imatch])
        else:
            OutputContainer = MultiTimeStream
            imatch = np.searchsorted(data.time, beam.pix["phi"])
            output_kwargs.update(time=data.time[imatch])

        stacked_beam = OutputContainer(
            axes_from=data,
            attrs_from=beam,
            distributed=True,
            comm=data.comm,
            **output_kwargs,
        )

        stacked_beam.redistribute("freq")

        stacked_beam.vis[:] = 0.0
        stacked_beam.weight[:] = 0.0

        stacked_beam.attrs["tag"] = "_".join([beam.attrs["tag"], data.attrs["tag"]])

        # Dereference datasets
        bbe = beam.beam[:].view(np.ndarray)
        bwe = beam.weight[:].view(np.ndarray)

        ovis = stacked_beam.vis[:]
        oweight = stacked_beam.weight[:]

        # Compute the fractional variance of the beam measurement
        bflag = bwe > 0.0
        var = tools.invert_no_zero(bwe * np.abs(bbe) ** 2)

        input_flags = data.input_flags[:, imatch].astype(bool)
        if not np.any(input_flags):
            input_flags = np.ones_like(input_flags)

        # Determine polarisation map
        pol_filter = {
            "X": "X",
            "Y": "Y",
            "E": "X",
            "S": "Y",
            "co": "co",
            "cross": "cross",
        }
        pol = [pol_filter.get(pp, None) for pp in self.telescope.polarisation]
        beam_pol = [pol_filter.get(pp, None) for pp in beam.index_map["pol"][:]]

        self.log.info(f"Calculating {self.nstream} stream.")
        pol_lookup = self._resolve_pol(beam_pol, self.nstream)

        nfreq, nstack, _, _ = ovis.shape

        # Find boundaries into the sorted products that separate stacks.
        stack_index = reverse_stack["stack"][:]
        stack_conj = reverse_stack["conjugate"][:].astype(bool)

        valid_stack = np.flatnonzero((stack_index >= 0) & (stack_index < nstack))
        stack_index = stack_index[valid_stack]

        isort = np.argsort(stack_index)
        sorted_stack_index = stack_index[isort]

        ivsort = valid_stack[isort]
        sorted_stack_conj = stack_conj[ivsort]
        sorted_prod = prod[ivsort]

        temp = sorted_prod.copy()
        sorted_prod["input_a"] = np.where(
            sorted_stack_conj, temp["input_b"], temp["input_a"]
        )
        sorted_prod["input_b"] = np.where(
            sorted_stack_conj, temp["input_a"], temp["input_b"]
        )

        boundary = np.concatenate(
            (
                np.atleast_1d(0),
                np.flatnonzero(np.diff(sorted_stack_index) > 0) + 1,
                np.atleast_1d(prod.size),
            )
        )

        # Create counter to increment during the stacking.
        # This will be used to normalize at the end.
        counter = np.zeros_like(oweight)

        # Loop over frequencies
        for ff in range(nfreq):

            flag = input_flags[np.newaxis, :, :] & bflag[ff]

            valid_ra = np.flatnonzero(np.any(flag, axis=(0, 1)))

            if valid_ra.size == 0:
                # Do not bother processing if all RAs are flagged
                continue

            slcs = find_contiguous_groups(valid_ra)

            for slc in slcs:

                fs = flag[:, :, slc]
                bs = bbe[ff, :, :, slc]
                vs = var[ff, :, :, slc]

                for ss, ssi in enumerate(np.unique(sorted_stack_index)):

                    prodo = sorted_prod[boundary[ss] : boundary[ss + 1]]
                    aa = prodo["input_a"]
                    bb = prodo["input_b"]

                    try:
                        pstr = pol[aa[0]] + pol[bb[0]]
                        aa_pol, bb_pol = pol_lookup[pstr]
                    except KeyError:
                        self.log.warning(
                            f"Not familiar with {pstr} polarisation.  " "Skipping."
                        )
                        continue

                    f = fs[:, aa][aa_pol] & fs[:, bb][bb_pol]

                    c = bs[:, aa][aa_pol] * bs[:, bb][bb_pol].conj()

                    v = np.abs(c) ** 2 * (vs[:, aa][aa_pol] + vs[:, bb][bb_pol])

                    ovis[ff, ssi, :, slc] = np.sum(f * c, axis=1)
                    oweight[ff, ssi, :, slc] = np.sum(f * v, axis=1)
                    counter[ff, ssi, :, slc] = np.sum(f, axis=1)

        # Divide through by counter to get properly weighted visibility average
        ovis[:] *= tools.invert_no_zero(counter)
        oweight[:] = counter**2 * tools.invert_no_zero(oweight[:])

        return stacked_beam

    @staticmethod
    def _resolve_pol(pol_axis, nstream):

        # We first define the lookup table for the nstream == 4 case.
        # We need different indices depending on whether we are using the
        # new convention (co, cross) or old convention (X, Y) for labeling
        # the polarisation axis of the holographic measurements.
        if "co" in pol_axis:
            ico = pol_axis.index("co")
            icr = pol_axis.index("cross")

            lookup = {
                "XX": [(ico, ico), (ico, icr), (icr, ico), (icr, icr)],
                "XY": [(ico, icr), (ico, ico), (icr, icr), (icr, ico)],
                "YX": [(icr, ico), (icr, icr), (ico, ico), (ico, icr)],
                "YY": [(icr, icr), (icr, ico), (ico, icr), (ico, ico)],
            }

        else:
            ix = pol_axis.index("X")
            iy = pol_axis.index("Y")

            lookup = {
                key: [(ix, ix), (ix, iy), (iy, ix), (iy, iy)]
                for key in ["XX", "XY", "YX", "YY"]
            }

        # Now restrict the lookup table if nstream == 2 or nstream == 1
        if nstream == 2:

            lookup = {key: [val[0], val[3]] for key, val in lookup.items()}

        elif nstream == 1:

            keys = sorted(lookup.keys())
            lookup = {key: [lookup[key][kk]] for kk, key in enumerate(keys)}

        # Reformat lookup table in numpy index arrays
        lookup = {
            key: tuple([np.array([v[0] for v in val]), np.array([v[1] for v in val])])
            for key, val in lookup.items()
        }

        return lookup


class HolographyTransitFit(TransitFit):
    """Fit a model to the transit.

    Attributes
    ----------
    pol : str
        The polarization product to fit.  One of either "co" or "cross".
    model : str
        Name of the model to fit.  One of "gauss_amp_poly_phase" or
        "poly_log_amp_poly_phase" or "poly_real_poly_imag"
    nsigma : float
        Number of standard deviations away from transit to fit.
    absolute_sigma : bool
        Set to True if the errors provided are absolute.  Set to False if
        the errors provided are relative, in which case the parameter covariance
        will be scaled by the chi-squared per degree-of-freedom.
    poly_type : str
        Type of polynomial.  Either 'standard', 'hermite', or 'chebychev'.
        Relevant if `model = "poly_log_amp_poly_phase"` or `model = "poly_real_poly_imag"`.
    poly_deg_amp : int
        Degree of the polynomial to fit to amplitude.
        Relevant if `model = "poly_log_amp_poly_phase"`.
    poly_deg_phi : int
        Degree of the polynomial to fit to phase.
        Relevant if `model = "poly_log_amp_poly_phase"`.
    niter : int
        Number of times to update the errors using model amplitude.
        Relevant if `model = "poly_log_amp_poly_phase"`.
    moving_window : int
        Number of standard deviations away from peak to fit.
        The peak location is updated with each iteration.
        Must be less than `nsigma`.  Relevant if `model = "poly_log_amp_poly_phase"`.
    poly_deg : int
        Degree of the polynomial to fit to real and imaginary component.
        Relevant if `model = "poly_real_poly_imag"`.
    even : bool
        Force the polynomial to be even.  Keeps all even coefficients up to `poly_deg + 1`.
        Relevant if `model = "poly_real_poly_imag"`.
    odd : bool
        Force the polynomial to be odd.  Keeps all odd coefficients up to `poly_deg + 1`.
        Relevant if `model = "poly_real_poly_imag"`.
    """

    pol = config.enum(["co", "cross"], default="co")

    # Disable NaN checks by default since they can be valid outputs of the fit
    nan_check = config.Property(default=False, proptype=bool)
    nan_skip = config.Property(default=False, proptype=bool)
    nan_dump = config.Property(default=False, proptype=bool)

    def process(self, transit):
        """Perform the fit.

        Parameters
        ----------
        transit: TrackBeam
            Transit to be fit.

        Returns
        -------
        fit: TransitFitParams
            Fit parameters.
        """
        transit.redistribute("freq")

        # Set initial estimate of beam sigma
        local_slice = slice(
            transit.beam.local_offset[0],
            transit.beam.local_offset[0] + transit.beam.local_shape[0],
        )
        ninput = transit.beam.local_shape[2]
        freq = transit.freq[local_slice]
        sigma = (0.7 * SPEED_LIGHT / (CHIME_CYL_W * freq)) * (360.0 / np.pi)
        sigma = sigma[:, np.newaxis] * np.ones((1, ninput), dtype=sigma.dtype)

        # Find index into pol axis that yields desired polarization products
        pol_axis = list(transit.index_map["pol"])
        if self.pol in pol_axis:
            pol_slice = (slice(None), pol_axis.index(self.pol))
        else:
            ns, ew = (
                (pol_axis.index("S"), pol_axis.index("E"))
                if self.pol == "co"
                else (pol_axis.index("E"), pol_axis.index("S"))
            )
            this_pol = np.array(
                [ns if not ((ii // 256) % 2) else ew for ii in range(ninput)]
            )
            pol_slice = (slice(None), this_pol, np.arange(ninput))

        # Dereference datasets
        ha = transit.pix["phi"][:]

        vis = transit.beam[:].view(np.ndarray)
        vis = vis[pol_slice]

        err = transit.weight[:].view(np.ndarray)
        err = np.sqrt(tools.invert_no_zero(err[pol_slice]))

        # Flag data that is outside the fit window set by nsigma config parameter
        if self.nsigma is not None:
            err *= (
                np.abs(ha[np.newaxis, np.newaxis, :])
                <= (self.nsigma * sigma[:, :, np.newaxis])
            ).astype(err.dtype)

        # Instantiate the model fitter
        model = self.ModelClass(**self.model_kwargs)

        # Fit the model
        model.fit(ha, vis, err, width=sigma, **self.fit_kwargs)

        # Pack into container
        fit = TransitFitParams(
            param=model.parameter_names,
            component=model.component,
            axes_from=transit,
            attrs_from=transit,
            distributed=transit.distributed,
            comm=transit.comm,
        )

        fit.add_dataset("chisq")
        fit.add_dataset("ndof")

        fit.redistribute("freq")

        # Transfer fit information to container attributes
        fit.attrs["model_kwargs"] = json.dumps(model.model_kwargs)
        fit.attrs["model_class"] = ".".join(
            [getattr(self.ModelClass, key) for key in ["__module__", "__name__"]]
        )

        # Save datasets
        fit.parameter[:] = model.param[:]
        fit.parameter_cov[:] = model.param_cov[:]
        fit.chisq[:] = model.chisq[:]
        fit.ndof[:] = model.ndof[:]

        return fit


DetermineHolographyGainsFromFits = GainFromTransitFit


class ApplyHolographyGains(task.SingleTask):
    """Apply gains to a holography transit

    Parameters
    ----------
    overwrite: bool (default: False)
        If True, overwrite the input TrackBeam.
    """

    overwrite = config.Property(proptype=bool, default=False)
    pol = config.enum(["co", "cross", "both"], default="both")
    common_mode = config.Property(proptype=bool, default=False)

    def process(self, track_in, gain):
        """Apply gain

        Parameters
        ----------
        track: draco.core.containers.TrackBeam
            Holography track to apply gains to. Will apply gains to
            track['beam'], expecting axes to be freq, pol, input, ha
        gain: np.array
            Gain to apply. Expected axes are freq, pol, and input
        """

        # Create output container
        if self.overwrite:
            track = track_in
        else:
            track = TrackBeam(
                axes_from=track_in,
                attrs_from=track_in,
                distributed=track_in.distributed,
                comm=track_in.comm,
            )
            track["beam"] = track_in["beam"][:]
            track["weight"] = track_in["weight"][:]

        # Find index into pol axis that yields desired polarization products
        pol_axis = list(track_in.index_map["pol"])
        ninput = track_in["beam"].shape[2]
        if self.pol == "both":
            in_slice = (slice(None), None, slice(None), None)
            out_slice = slice(None)

        elif self.pol in pol_axis:
            in_slice = (slice(None), slice(None), None)
            out_slice = (slice(None), pol_axis.index(self.pol))

        else:
            in_slice = (slice(None), slice(None), None)
            ns, ew = (
                (pol_axis.index("S"), pol_axis.index("E"))
                if self.pol == "co"
                else (pol_axis.index("E"), pol_axis.index("S"))
            )
            this_pol = np.array(
                [ns if not ((ii // 256) % 2) else ew for ii in range(ninput)]
            )
            out_slice = (slice(None), this_pol, np.arange(ninput))

        if self.common_mode:
            rawg = gain.gain[:]
            flag = (np.abs(rawg) > 0.0) & (gain.weight[:] > 0.0)

            g = np.zeros_like(rawg)
            for pol in [0, 1]:
                this_pol = np.array([((ii // 256) % 2) == pol for ii in range(ninput)])

                this_g = np.nanmedian(
                    np.where(flag & this_pol[np.newaxis, :], np.real(rawg), np.nan),
                    axis=1,
                ) + 1.0j * np.nanmedian(
                    np.where(flag & this_pol[np.newaxis, :], np.imag(rawg), np.nan),
                    axis=1,
                )

                g[:, this_pol] = np.where(np.isfinite(this_g), this_g, 0.0 + 0.0j)[
                    :, np.newaxis
                ]

        else:
            g = gain.gain[:]

        g = g[in_slice]

        # Apply gains
        track["beam"][:][out_slice] *= g
        track["weight"][:][out_slice] *= tools.invert_no_zero(np.abs(g) ** 2)

        return track


class MedianBeam(io.LoadFilesFromParams):

    _attrs = ["filename", "observation_id", "transit_time", "archivefiles"]

    def setup(self):

        super().setup()

        self._nobs = len(self.files)
        self._counter = 0

    def process(self):
        """Load the next file and save observation to one large array."""

        data = super().process()
        data.redistribute("freq")

        # If this is the first file, then generate the output container
        # and the arrays to hold the data products.
        if self._counter == 0:
            self._out = TrackBeam(
                axes_from=data,
                attrs_from=data,
                distributed=data.distributed,
                comm=data.comm,
            )

            self._out.add_dataset("nsample")

            self._out.redistribute("freq")

            for name in self._attrs:
                self._out.attrs[name] = []

            bshp = (self._nobs,) + data.beam.local_shape

            self._beam = np.zeros(bshp, dtype=data.beam.dtype)
            self._flag = np.zeros(bshp, dtype=bool)

        # Save the attributes
        for name in self._attrs:
            self._out.attrs[name] += list(np.atleast_1d(data.attrs[name]))

        self._beam[self._counter] = data.beam[:].view(np.ndarray).copy()
        self._flag[self._counter] = data.weight[:].view(np.ndarray) > 0.0

        self._counter += 1

    def process_finish(self):
        """Take the median over observations."""

        # Dereference datasets
        obeam = self._out.beam[:].view(np.ndarray)
        oweight = self._out.weight[:].view(np.ndarray)
        onum = self._out["nsample"][:].view(np.ndarray)

        nfreq, npol, ninput, nha = obeam.shape

        # Loop over local frequencies, pol, input to reduce memory
        for ff in range(nfreq):

            for pp in range(npol):

                for ii in range(ninput):

                    iin = (slice(None), ff, pp, ii)
                    iout = (ff, pp, ii)

                    onum[iout] = np.sum(self._flag[iin], axis=0)

                    nan_beam = np.where(
                        self._flag[iin], self._beam[iin], np.nan + 1.0j * np.nan
                    )

                    med_beam = np.nanmedian(
                        nan_beam.real, axis=0
                    ) + 1.0j * np.nanmedian(nan_beam.imag, axis=0)

                    mad_beam = 1.48625 * np.nanmedian(
                        np.abs(nan_beam - med_beam[np.newaxis, ...]), axis=0
                    )
                    mad_weight = onum[iout] * tools.invert_no_zero(mad_beam**2)

                    med_flag = np.isfinite(med_beam) & np.isfinite(mad_weight)
                    obeam[iout] = np.where(med_flag, med_beam, 0.0 + 00j)
                    oweight[iout] = np.where(med_flag, mad_weight, 0.0)

        return self._out


class TransitStacker(task.SingleTask):
    """Stack a number of transits together.

    The weights will be inverted and stacked as variances. The variance
    between transits is evaluated and recorded in the output container.

    All transits should be on a common grid in hour angle.

    Attributes
    ----------
    weight: str (default: uniform)
        The weighting to use in the stack. One of `uniform` or `inverse_variance`.
    """

    weight = config.enum(["uniform", "inverse_variance"], default="uniform")

    def setup(self):
        """Initialise internal variables."""

        self.stack = None
        self.variance = None
        self.pseudo_variance = None
        self.norm = None

    def process(self, transit):
        """Add a transit to the stack.

        Parameters
        ----------
        transit: draco.core.containers.TrackBeam
            A holography transit.
        """

        self.log.info("Weight is %s" % self.weight)

        if self.stack is None:
            self.log.info("Initializing transit stack.")
            self.stack = TrackBeam(
                axes_from=transit, distributed=transit.distributed, comm=transit.comm
            )

            self.stack.add_dataset("sample_variance")
            self.stack.add_dataset("nsample")
            self.stack.redistribute("freq")

            self.log.info("Adding %s to stack." % transit.attrs["tag"])

            # Copy over relevant attributes
            self.stack.attrs["filename"] = [transit.attrs["tag"]]
            self.stack.attrs["observation_id"] = [transit.attrs["observation_id"]]
            self.stack.attrs["transit_time"] = [transit.attrs["transit_time"]]
            self.stack.attrs["archivefiles"] = list(transit.attrs["archivefiles"])

            self.stack.attrs["dec"] = transit.attrs["dec"]
            self.stack.attrs["source_name"] = transit.attrs["source_name"]
            self.stack.attrs["icrs_ra"] = transit.attrs["icrs_ra"]
            self.stack.attrs["cirs_ra"] = transit.attrs["cirs_ra"]

            # Copy data for first transit
            flag = (transit.weight[:] > 0.0).astype(np.int)
            if self.weight == "inverse_variance":
                coeff = transit.weight[:]
            else:
                coeff = flag.astype(np.float32)

            self.stack.beam[:] = coeff * transit.beam[:]
            self.stack.weight[:] = (coeff**2) * invert_no_zero(transit.weight[:])
            self.stack.nsample[:] = flag.astype(np.int)

            self.variance = coeff * np.abs(transit.beam[:]) ** 2
            self.pseudo_variance = coeff * transit.beam[:] ** 2
            self.norm = coeff

        else:
            if list(transit.beam.shape) != list(self.stack.beam.shape):
                self.log.error(
                    "Transit has different shape than stack: {}, {}".format(
                        transit.beam.shape, self.stack.beam.shape
                    )
                    + " Skipping."
                )
                return None

            self.log.info("Adding %s to stack." % transit.attrs["tag"])

            self.stack.attrs["filename"].append(transit.attrs["tag"])
            self.stack.attrs["observation_id"].append(transit.attrs["observation_id"])
            self.stack.attrs["transit_time"].append(transit.attrs["transit_time"])
            self.stack.attrs["archivefiles"] += list(transit.attrs["archivefiles"])

            # Accumulate transit data
            flag = (transit.weight[:] > 0.0).astype(np.int)
            if self.weight == "inverse_variance":
                coeff = transit.weight[:]
            else:
                coeff = flag.astype(np.float32)

            self.stack.beam[:] += coeff * transit.beam[:]
            self.stack.weight[:] += (coeff**2) * invert_no_zero(transit.weight[:])
            self.stack.nsample[:] += flag

            self.variance += coeff * np.abs(transit.beam[:]) ** 2
            self.pseudo_variance += coeff * transit.beam[:] ** 2
            self.norm += coeff

        return None

    def process_finish(self):
        """Normalise the stack and return the result.

        Includes the sample variance over transits within the stack.

        Returns
        -------
        stack: draco.core.containers.TrackBeam
            Stacked transits.
        """
        # Divide by norm to get average transit
        inv_norm = invert_no_zero(self.norm)
        self.stack.beam[:] *= inv_norm
        self.stack.weight[:] = invert_no_zero(self.stack.weight[:]) * self.norm**2

        self.variance = self.variance * inv_norm - np.abs(self.stack.beam[:]) ** 2
        self.pseudo_variance = self.pseudo_variance * inv_norm - self.stack.beam[:] ** 2

        # Calculate the covariance between the real and imaginary component
        # from the accumulated variance and psuedo-variance
        self.stack.sample_variance[0] = 0.5 * (
            self.variance + self.pseudo_variance.real
        )
        self.stack.sample_variance[1] = 0.5 * self.pseudo_variance.imag
        self.stack.sample_variance[2] = 0.5 * (
            self.variance - self.pseudo_variance.real
        )

        # Create tag
        time_range = np.percentile(self.stack.attrs["transit_time"], [0, 100])
        self.stack.attrs["tag"] = "{}_{}_to_{}".format(
            self.stack.attrs["source_name"],
            ephem.unix_to_datetime(time_range[0]).strftime("%Y%m%dT%H%M%S"),
            ephem.unix_to_datetime(time_range[1]).strftime("%Y%m%dT%H%M%S"),
        )

        return self.stack


class FilterHolographyProcessed(task.MPILoggedTask):
    """Filter holography transit DataIntervals produced by `io.QueryDatabase`
    to exclude those already processed.

    Attributes
    ----------
    processed_dir: str or list of str
        Directory or list of directories to look in for processed files.
    source: str
        The name of the holography source (as used in the holography database).
    """

    processed_dir = config.Property(
        proptype=lambda x: x if isinstance(x, list) else [x]
    )
    source = config.Property(proptype=str)

    def setup(self):
        """Get a list of existing processed files."""

        # Find processed transit files
        self.proc_transits = []
        for processed_dir in self.processed_dir:
            self.log.debug(
                "Looking for processed transits in {}...".format(processed_dir)
            )
            # Expand path
            processed_dir = path.expanduser(processed_dir)
            processed_dir = path.expandvars(processed_dir)

            try:
                processed_files = listdir(processed_dir)
            except FileNotFoundError:
                processed_files = []
            for fname in processed_files:
                if not path.splitext(fname)[1] == ".h5":
                    continue
                with ContainerBase.from_file(
                    fname, ondisk=True, distributed=False, mode="r"
                ) as fh:
                    obs_id = fh.attrs.get("observation_id", None)
                    if obs_id is not None:
                        self.proc_transits.append(obs_id)
        self.log.debug("Found {:d} processed transits.".format(len(self.proc_transits)))

        # Query database for observations of this source
        hol_obs = None
        if mpiutil.rank0:
            hol_obs = list(get_holography_obs(self.source))
        self.hol_obs = mpiutil.bcast(hol_obs, root=0)
        mpiutil.barrier()

    def next(self, intervals):
        """Filter input files to exclude those already processed.

        Parameters
        ----------
        intervals: list of chimedb.data_index.DataInterval
            List intervals to filter.

        Returns
        -------
        files: list of str
            List of files to be processed.
        """

        self.log.info("Starting next for task %s" % self.__class__.__name__)

        self.comm.Barrier()

        files = []
        for fi in intervals:
            start, end = fi[1]
            # find holography observation that overlaps this set
            this_obs = [
                o
                for o in self.hol_obs
                if (o.start_time >= start and o.start_time <= end)
                or (o.finish_time >= start and o.finish_time <= end)
                or (o.start_time <= start and o.finish_time >= end)
            ]

            if len(this_obs) == 0:
                self.log.warning(
                    "Could not find source transit in holography database for {}.".format(
                        ephem.unix_to_datetime(start)
                    )
                )
            elif this_obs[0].id in self.proc_transits:
                self.log.warning(
                    "Already processed transit for {}. Skipping.".format(
                        ephem.unix_to_datetime(start)
                    )
                )
            else:
                files += fi[0]

        self.log.info("Leaving next for task %s" % self.__class__.__name__)
        return files


def unwrap_lha(lsa, src_ra):
    """Convert LSA into HA for a source's RA. Ensures HA is monotonically increasing.

    Parameters
    ----------
    lsa: array
        Local sidereal angle.
    src_ra: float
        The RA of the source.

    Returns
    -------
    ha: array
        Hour angle.
    """
    # ensure monotonic
    start_lsa = lsa[0]
    lsa -= start_lsa
    lsa[lsa < 0] += 360.0
    lsa += start_lsa
    # subtract source RA
    return np.where(
        np.abs(lsa - src_ra) < np.abs(lsa - src_ra + 360.0),
        lsa - src_ra,
        lsa - src_ra + 360.0,
    )


def get_holography_obs(src):
    """Query database for list of all holography observations for the given
    source.

    Parameters
    ----------
    src: str
        Source name.

    Returns
    -------
    db_obs: list of ch_util.holography.HolographyObservation
        Observations of this source.
    """
    connect_database()
    db_src = holography.HolographySource.get(holography.HolographySource.name == src)
    db_obs = holography.HolographyObservation.select().where(
        holography.HolographyObservation.source == db_src
    )
    return db_obs
