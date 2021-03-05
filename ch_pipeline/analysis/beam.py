""" Tasks for beam measurement processing.

    Tasks
    =====

    .. autosummary::
        :toctree:

        TransitGrouper
        TransitRegridder
        EdgeFlagger
        TransitResampler
        MakeHolographyBeam
        ConstructStackedBeam
        HolographyTransitFit
        DetermineHolographyGainsFromFits
        ApplyHolographyGains
        TransitStacker
        FilterHolographyProcessed
"""
import json
import yaml
from os import path, listdir

import numpy as np
from scipy import constants

from caput import config, tod, mpiarray, mpiutil
from caput.pipeline import PipelineConfigError, PipelineRuntimeError
from caput.time import STELLAR_S

from ch_util import ephemeris as ephem
from ch_util import tools, layout, holography
from chimedb import data_index as di
from chimedb.core import connect as connect_database


from draco.core import task, io
from draco.util import regrid
from draco.analysis.transform import Regridder
from draco.core.containers import ContainerBase, SiderealStream, SystemSensitivity, TimeStream, TrackBeam
from draco.util.tools import invert_no_zero, calculate_redundancy

from ..core.containers import TransitFitParams
from .calibration import TransitFit, GainFromTransitFit


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
        """
        self.observer = ephem.chime_observer() if observer is None else observer
        self.sky_obs = wrap_observer(self.observer)
        try:
            self.src = ephem.source_dictionary[self.source]
        except KeyError:
            msg = (
                "Could not find source {} in catalogue. "
                "Must use same spelling as in `ch_util.ephemeris`.".format(self.source)
            )
            self.log.error(msg)
            raise PipelineConfigError(msg)
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

        # Update observer time
        self.sky_obs.date = tstream.time[0]

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
            self.cur_transit = self.sky_obs.next_transit(self.src)
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
            _, dec = self.sky_obs.radec(self.src)
            ts.attrs["dec"] = dec._degrees
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
        """
        self.observer = ephem.chime_observer() if observer is None else observer
        self.sky_obs = wrap_observer(self.observer)

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
            raise PipelineConfigError(msg)

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

        # Update observer time
        self.sky_obs.date = data.time[0]

        # Get apparent source RA, including precession effects
        ra, _ = self.sky_obs.cirs_radec(self.src)
        ra = ra._degrees
        # Get catalogue RA for reference
        ra_icrs, _ = self.sky_obs.radec(self.src)
        ra_icrs = ra_icrs._degrees

        # Convert input times to hour angle
        lha = unwrap_lha(self.sky_obs.unix_to_lsa(data.time), ra)

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
        self.observer = ephem.chime_observer() if observer is None else observer
        self.sky_obs = wrap_observer(self.observer)

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
        # Distribute over frequencies
        data.redistribute("freq")
        beam.redistribute("freq")

        # Dereference datasets
        bv = beam.beam[:].view(np.ndarray)
        bw = beam.weight[:].view(np.ndarray)

        flag = np.any(bw > 0.0, axis=(0, 1, 2))

        # Compute the hour angle of the source
        if "ra" in data.index_map:
            ra = data.index_map["ra"][:]
        elif "time" in data.index_map:
            ra = self.sky_obs.unix_to_lsa(data.time)
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

        # Perform regridding
        new_bv, new_bw = self._resample(hour_angle, bv, bw, new_hour_angle_sorted[slc])

        # Create output container
        new_beam = TrackBeam(
            phi=new_hour_angle,
            theta=np.repeat(np.median(beam.pix["theta"][:]), new_hour_angle.size),
            axes_from=beam,
            attrs_from=beam,
            distributed=beam.distributed,
            comm=beam.comm,
        )

        new_beam.redistribute("freq")

        # Save to container
        new_beam.beam[:] = 0.0
        new_beam.weight[:] = 0.0

        new_beam.beam[:, :, :, isort[slc]] = new_bv
        new_beam.weight[:, :, :, isort[slc]] = new_bw

        return new_beam

    def _resample(self, xgrid, ygrid, wgrid, x):

        lza = regrid.lanczos_forward_matrix(xgrid, x, a=self.lanczos_width).T

        y = np.matmul(ygrid, lza)
        w = invert_no_zero(np.matmul(invert_no_zero(wgrid), lza ** 2))

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
            **output_kwargs
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

        # Compute the fractional variance of the beam measurement
        frac_var = invert_no_zero(bw * np.abs(bv) ** 2)

        # Create counter to increment during the stacking.
        # This will be used to normalize at the end.
        counter = np.zeros_like(ow)

        # Construct stack
        for pp, (ss, conj) in enumerate(reverse_stack):

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
            ow[:, ss, :] += wss ** 2 * invert_no_zero(weight)

            # Increment counter
            counter[:, ss, :] += wss

        # Divide through by counter to get properly weighted visibility average
        ov[:] *= invert_no_zero(counter)
        ow[:] = counter ** 2 * invert_no_zero(ow[:])

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


class HolographyTransitFit(TransitFit):
    """Fit a model to the transit.

    Attributes
    ----------
    model : str
        Name of the model to fit.  One of 'gauss_amp_poly_phase' or
        'poly_log_amp_poly_phase'.
    nsigma : float
        Number of standard deviations away from transit to fit.
    absolute_sigma : bool
        Set to True if the errors provided are absolute.  Set to False if
        the errors provided are relative, in which case the parameter covariance
        will be scaled by the chi-squared per degree-of-freedom.
    poly_type : str
        Type of polynomial.  Either 'standard', 'hermite', or 'chebychev'.
    poly_deg_amp : int
        Degree of the polynomial to fit to amplitude.
        Relevant if `poly = True`.
    poly_deg_phi : int
        Degree of the polynomial to fit to phase.
        Relevant if `poly = True`.
    niter : int
        Number of times to update the errors using model amplitude.
        Relevant if `poly = True`.
    moving_window : int
        Number of standard deviations away from peak to fit.
        The peak location is updated with each iteration.
        Must be less than `nsigma`.  Relevant if `poly = True`.
    """

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

        # Find index into pol axis that yields copolar products
        pol_axis = list(transit.index_map["pol"])
        if "co" in pol_axis:
            copolar_slice = (slice(None), pol_axis.index("co"))
        else:
            this_pol = np.array(
                [
                    pol_axis.index("S")
                    if not ((ii // 256) % 2)
                    else pol_axis.index("E")
                    for ii in range(ninput)
                ]
            )
            copolar_slice = (slice(None), this_pol, np.arange(ninput))

        # Dereference datasets
        ha = transit.pix["phi"][:]

        vis = transit.beam[:].view(np.ndarray)
        vis = vis[copolar_slice]

        err = transit.weight[:].view(np.ndarray)
        err = np.sqrt(invert_no_zero(err[copolar_slice]))

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


class ApplyHolographyGains(task.SingleTask):
    """Apply gains to a holography transit

    Attributes
    ----------
    overwrite: bool (default: False)
        If True, overwrite the input TrackBeam.
    """

    overwrite = config.Property(proptype=bool, default=False)

    def process(self, track_in, gain):
        """Apply gain

        Parameters
        ----------
        track: draco.core.containers.TrackBeam
            Holography track to apply gains to. Will apply gains to
            track['beam'], expecting axes to be freq, pol, input, ha
        gain: np.array
            Gain to apply. Expected axes are freq, pol, and input

        Returns
        -------
        track: draco.core.containers.TrackBeam
            Holography track with gains applied.
        """

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

        track["beam"][:] *= gain.gain[:][:, np.newaxis, :, np.newaxis]
        track["weight"][:] *= invert_no_zero(np.abs(gain.gain[:]) ** 2)[
            :, np.newaxis, :, np.newaxis
        ]

        track["beam"][:] = np.where(np.isfinite(track["beam"][:]), track["beam"][:], 0)
        track["weight"][:] = np.where(
            np.isfinite(track["weight"][:]), track["weight"][:], 0
        )

        return track


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

            self.stack.add_dataset("observed_variance")
            self.stack.add_dataset("number_of_observations")
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
            self.stack.weight[:] = (coeff ** 2) * invert_no_zero(transit.weight[:])
            self.stack.number_of_observations[:] = flag.astype(np.int)

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
            self.stack.weight[:] += (coeff ** 2) * invert_no_zero(transit.weight[:])
            self.stack.number_of_observations[:] += flag

            self.variance += coeff * np.abs(transit.beam[:]) ** 2
            self.pseudo_variance += coeff * transit.beam[:] ** 2
            self.norm += coeff

        return None

    def process_finish(self):
        """Normalise the stack and return the result.

        Includes the observed variance between transits within the stack,

        Returns
        -------
        stack: draco.core.containers.TrackBeam
            Stacked transits.
        """
        # Divide by norm to get average transit
        inv_norm = invert_no_zero(self.norm)
        self.stack.beam[:] *= inv_norm
        self.stack.weight[:] = invert_no_zero(self.stack.weight[:]) * self.norm ** 2

        self.variance = self.variance * inv_norm - np.abs(self.stack.beam[:]) ** 2
        self.pseudo_variance = self.pseudo_variance * inv_norm - self.stack.beam[:] ** 2

        # Calculate the covariance between the real and imaginary component
        # from the accumulated variance and psuedo-variance
        self.stack.observed_variance[0] = 0.5 * (
            self.variance + self.pseudo_variance.real
        )
        self.stack.observed_variance[1] = 0.5 * self.pseudo_variance.imag
        self.stack.observed_variance[2] = 0.5 * (
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
        """Get a list of existing processed files.
        """

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

class ComputeHolographicSensitivity(task.SingleTask):

    nan_check = config.Property(default=False, proptype=bool)
    nan_skip = config.Property(default=False, proptype=bool)
    nan_dump = config.Property(default=False, proptype=bool)

    def setup(self, pm):
        """ Load the telescope instance.
 
        Parameters
        ----------
        pm : drift.manager.ProductManager
            The on-disk telescope model containing layout information.

        """
        self.telescope = io.get_telescope(pm)
        self._POL_DICT = {"XX" : 1, "XY" : 2, "YX" : 2, "YY" : 0} # fixed on 8/18/2020 10:25 PM to match measured convention
        self._26m_POL = [1225, 1521]

    def process(self, hol_data, ch_data): 
	# Distribute both datasets over frequency. 
        hol_data.redistribute("freq")
        ch_data.redistribute("freq")
	
	# Obtain the shape of the holographic visibilities and load the product map. 
        nfreq, nprod, ntime = hol_data.vis.local_shape
        npol = 2 # 2 co-pol products

        prodmap_h = hol_data.prod[:]

	# Get the start and end times for the holography dataset and get the overlapping portion of the chimestack. 
        hol_start, hol_stop = hol_data.time[[0, -1]]
        self.log.warning("The start and stop times for the holography are {}->{}".format(hol_start, hol_stop)) # debug
        timerange = ((ch_data.time[:] > hol_start) & (ch_data.time[:] < hol_stop))
        self.log.warning("Overlap is {}".format(np.sum(timerange))) # debug

        # Dereference the two visibility datasets 
        vis_ho = hol_data.vis[:].view(np.ndarray)
        vis_st = ch_data.vis[:].view(np.ndarray)
        self.log.warning("We were able to get the data shape and some numbers {}, {}, {}, max {} ".format(nfreq, nprod, ntime, np.max(vis_ho[0, 0, :].real)))
        self.log.warning("We were able to get the stack shape and some numbers {}, {}, {}, max {} ".format(*(ch_data.vis.local_shape), np.max(vis_st[0, 0, :].real)))

        # Get the gain dataset from the chimestack to calibrate the holographic weights
        # Also get the fraction of packets lost 
        gain = ch_data.gain[:].view(np.ndarray)
        frac_lost = ch_data.flags["frac_lost"][:]

	# Get the flags for the two co-pol products (we ignore x-pol entirely for flagging purposes).
        copols = ComputeHolographicSensitivity._get_copols(prodmap_h)

        # Calculate the redundancy of the chimestack products
        inpflg = ch_data.input_flags[:].view(np.ndarray)

        stack_cnt = calculate_redundancy(
                inpflg.astype(np.float32),
                ch_data.prod,
                ch_data.reverse_map["stack"]["stack"],
                ch_data.stack.size,
	)

	# Derefence the weight dataset 
        weight = hol_data.weight[:].view(np.ndarray)	

	# Average the weight and visibility datasets down to match chimestack cadence 
        rwe = np.array([(weight[..., i] + weight[..., i+1]) / 2. for i in range(0, ntime-1, 2)])
        rwe = np.swapaxes(np.swapaxes(rwe, 0, 2), 1, 0)

        rvi = np.array([(vis_ho[..., i].real + vis_ho[..., i+1].real) / 2. for i in range(0, ntime-1, 2)])
        rvi = np.swapaxes(np.swapaxes(rvi, 0, 2), 1, 0)

        # Select only the stack data within the holography time range
        vis_st = vis_st[..., timerange]
        gain = gain[..., timerange]
        stack_cnt = stack_cnt[..., timerange]
        frac_lost = frac_lost[..., timerange]

        ntreduced = vis_st.shape[-1]

        # Feels hacky, but currently necessary to resolve the shapes exactly
        if ntreduced > rwe.shape[-1]:
            vis_st = vis_st[..., :rwe.shape[-1]]
            gain = gain[..., :rwe.shape[-1]]
            stack_cnt = stack_cnt[..., :rwe.shape[-1]]
            frac_lost = frac_lost[..., :rwe.shape[-1]]
            nt = rwe.shape[-1]
        elif ntreduced < rwe.shape[-1]:
            rwe = rwe[..., :ntreduced]
            rvi = rvi[..., :ntreduced]
            nt = ntreduced
        else:
            nt = np.sum(timerange)

        # Invert the weights, flag for only the positive weights
        inv_weight = tools.invert_no_zero(rwe) 
        wflag = (rwe > 0.0)

        # Initialize the variance and counter containers
        var =  np.zeros((nfreq, npol, nt), dtype=np.float32) 
        counter = np.zeros((nfreq, npol, nt), dtype=np.float32) 

        for ff in range(nfreq):

            for ipol in range(npol):

                inputs = np.array([ina if ina != self._26m_POL[ipol] else inb for (ina, inb) in prodmap_h[copols[ipol]]]) 

                self.log.warning("Pol index {} and over {} baselines ".format(ipol, np.sum(copols[ipol])))
                self.log.warning("Weight flag -> {} good elements".format(np.sum(wflag)))
                self.log.warning("Gain dataset on these inputs {}".format(gain[ff, inputs, :]))

                var[ff, ipol, :] = np.sum(2.0 * wflag[ff, copols[ipol]] * (np.abs(gain[ff, inputs, :]) ** 2) * inv_weight[ff, copols[ipol]], axis=0)
                counter[ff, ipol, :] = np.sum(wflag[ff, copols[ipol]], axis=0)

	# Normalize
        var *= tools.invert_no_zero(counter ** 2)

	# To obtain the radiometric estimate, get the stacked CHIME autocorrelations
        prodstack = ch_data.prod[ch_data.stack["prod"]]
        input_a, input_b = prodstack["input_a"], prodstack["input_b"]

        nfreq_st, nstack, _ = ch_data.vis.local_shape

        auto_stack = vis_st[:].real
        auto_flag_ch = (input_a == input_b)
        auto_stack_id = np.flatnonzero(auto_flag_ch)
        auto_inputs = prodstack[auto_stack_id]["input_a"]

        auto_pol = np.array([self.telescope.polarisation[ai] for ai in auto_inputs])

        hol_auto_id = [2450, 3568] # Y, X 
        hol_auto_pol = ["Y", "X"]

	# Limit the chimestack data to the holography timerange (debug)
        self.log.warning("Holography start and stop times: {}->{}".format(hol_start, hol_stop))
        self.log.warning("Chime stack start and stop times: {}->{}".format(ch_data.time[0], ch_data.time[-1]))
        if np.sum(timerange)  == 0:
            msg = "The holography and chimestack datasets you selected do not overlap in time. Aborting..."
            self.log.warning(msg)
            #raise ValueError(msg)

        # Initialize the radiometer and counter
        radiometer = np.zeros((nfreq_st, npol, nt), dtype=np.float32)	
        radiometer_counter = np.zeros((nfreq_st, npol, nt), dtype=np.float32)

        for ii, (cha, chap) in enumerate(zip(auto_stack_id, auto_pol)):

            for jj, (hoa, hoap) in enumerate(zip(hol_auto_id, hol_auto_pol)):

                try:
                    pp = self._POL_DICT[chap + hoap]
                except KeyError:
                    msg = "Unknown polarization product: {}{}".format(chap, hoap)		
                    self.log.warning(msg)

                if pp == 2: # Ignore x-pol for now
                    continue

                self.log.warning("Shapes: {}, {}, {}".format(stack_cnt[cha].shape, auto_stack[:, cha, :].shape, rvi[:, hoa, :].shape))
                radiometer[:, pp, :] += stack_cnt[cha] * auto_stack[:, cha, :] * rvi[:, hoa, :]
                radiometer_counter[:, pp, :] += stack_cnt[cha] 
                self.log.warning("Added {} to rad_counter for stack_id {}".format(stack_cnt[cha], cha))

        tint = np.abs(np.median(np.diff(hol_data.time[:])))
        dnu = np.abs(np.median(np.diff(hol_data.freq[:]))) * 1e6

        nint = (tint * dnu) * (1.0 - frac_lost.astype(np.float32))

        radiometer *= tools.invert_no_zero(nint[:, np.newaxis, :] * (radiometer_counter ** 2))
        self.log.warning("Radiometer {}".format(np.max(radiometer)))

        times = np.linspace(hol_start, hol_stop, nt) # Hacky for now

        metrics = SystemSensitivity(
            pol=np.array([b"YY", b"XX"]),
            time=times,
            axes_from=hol_data,
            attrs_from=hol_data,
            comm=hol_data.comm,
            distributed=hol_data.distributed,
	)

        metrics.redistribute("freq")
        metrics.radiometer[:] = np.sqrt(2 * radiometer)
        metrics.measured[:] = np.sqrt(var)
	
        metrics.weight[:] = counter
        #metrics.frac_lost[:] = frac_lost

        self.log.info("Wrapping up...")

        return metrics

    @staticmethod
    def _get_copols(prod_map):
        """Return two lists of indices into the `prod` axis that represent 
        respectively the two sets of co-polarization products for a holography
        dataset. 

        Parameters
        ----------
        prod_map : np.ndarray [nprod]
            The CHIME-26m products represented by the corresponding
            index of the holographic CorrData `prod` axis.  

        Returns
        -------
        copols : np.ndarray [nprod]
            Masks representing, respectively, the co-polarization products
            for each polarization. 

        """
        # This implementation feels pretty hacky. May need to explore if there are better ways. 
        input_a, input_b = prod_map["input_a"], prod_map["input_b"]

        Y_26m_flag1, Y_26m_flag2 = ((input_a == 1225),  (input_b == 1225))
        X_26m_flag1, X_26m_flag2 = ((input_a == 1521),  (input_b == 1521))

        ina_pol = np.array(["Y" if ((ina // 256) % 2 == 0) else "X" for ina in input_a])
        inb_pol = np.array(["Y" if ((inb // 256) % 2 == 0) else "X" for inb in input_b])

        ina_pol[Y_26m_flag1] = "Y"
        ina_pol[X_26m_flag1] = "X"
        inb_pol[Y_26m_flag2] = "Y"
        inb_pol[X_26m_flag2] = "X"

        polprods = np.array([pola + polb for (pola, polb) in zip(ina_pol, inb_pol)])
        YY = np.flatnonzero((polprods == "YY"))
        XX = np.flatnonzero((polprods == "XX"))

        copols = np.zeros((2, len(YY) - 1), dtype=np.int)
        copols[0] = YY[YY != 2450]
        copols[1] = XX[XX != 3568]

        return copols 

class ApplyRFIMask(task.SingleTask):

    # Mask type
    # Mask out entire frequencies, if most time samples are bad
    majority_mask = config.Property(default=True, proptype=bool)

    # Apply the mask with full time resolution, may not interact with fit well
    full_mask = config.Property(default=False, proptype=bool)

    def process(self, transit, mask):

        transit.redistribute("freq")

        beam = transit.beam[:].view(np.ndarray)
        weight = transit.weight[:].view(np.ndarray)
        ma = (~(mask.mask[:].view(np.ndarray))).astype(np.float32)

        nfreq = transit.beam.local_shape[0]
        npol = transit.beam.local_shape[1]
        ninput = transit.beam.local_shape[2]
        npix = transit.beam.local_shape[3]

        local_slice = slice(
            transit.beam.local_offset[0],
            transit.beam.local_offset[0] + nfreq
        )

        if self.majority_mask:

             rfimask1d = np.median(ma, axis=1)

             beam *= rfimask1d[local_slice, None, None, None] 

        if self.full_mask:

             bb = np.mean(abs(beam.reshape((nfreq, npol*ninput, npix))), axis=1) 

             non0 = np.where(bb != 0)[-1]
             st, en = non0.min(), non0.max()

             drange = 2 * (en - st)

             reducedmask = np.array([(ma[:, ii] + ma[:, ii + 1]) / 2 for ii in range(0, drange, 2)])
             reducedmask[reducedmask < 1.0] = 0.0

             beam[..., st:en] *= reducedmask.T[local_slice, None, None, :]
             weight[..., st:en] *= reducedmask.T[local_slice, None, None, :]

        transit.beam[:] = beam
        transit.weight[:] = weight

        return transit

def wrap_observer(obs):
    """Wrap a `ch_util.ephemeris.chime_observer()` with the
    `ch_util.ephemeris.SkyfieldObserverWrapper` class.

    Parameters
    ----------
    obs: caput.time.Observer
        CHIME observer.

    Returns
    -------
    obs: ch_util.ephemeris.SkyfieldObserverWrapper
        Wrapped observer.
    """
    return ephem.SkyfieldObserverWrapper(
        lon=obs.longitude,
        lat=obs.latitude,
        alt=obs.altitude,
        lsd_start=obs.lsd_start_day,
    )


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
