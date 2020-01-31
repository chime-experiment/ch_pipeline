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
"""
import json

import numpy as np

from caput import config, tod, mpiarray, mpiutil
from caput.pipeline import PipelineConfigError, PipelineRuntimeError

from draco.core import task, io
from draco.util import regrid
from draco.analysis.transform import Regridder
from draco.core.containers import SiderealStream, TimeStream, TrackBeam
from draco.util import tools

from ..core.processed_db import (
    RegisterProcessedFiles,
    append_product,
    get_proc_transits,
)
from ..core.containers import TransitFitParams
from .calibration import TransitFit, GainFromTransitFit

from ch_util import ephemeris as ephem
from ch_util import tools, layout
from ch_util import data_index as di

from os import path

from skyfield.constants import ANGVEL

SIDEREAL_DAY_SEC = 2 * np.pi / ANGVEL
SPEED_LIGHT = 299.7  # 10^6 m / s
CHIME_CYL_W = 20.0  # m


class TransitGrouper(task.SingleTask):
    """ Group transits from a sequence of TimeStream objects.

        Attributes
        ----------
        ha_span: float
            Span in degrees surrounding transit.
        source: str
            Name of the transiting source. (Must match what is used in `ch_util.ephemeris`.)
        db_source: str
            Name of the transiting source as listed in holography database.
            This is a hack until a better solution is implemented.
    """

    ha_span = config.Property(proptype=float, default=180.0)
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
            di.connect_database()
            db_runs = list(get_holography_obs(self.db_source))
            db_runs = [(int(r.id), (r.start_time, r.finish_time)) for r in db_runs]
        self.db_runs = mpiutil.bcast(db_runs, root=0)
        mpiutil.barrier()

    def process(self, tstream):
        """ Take in a timestream and accumulate, group into whole transits.

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
        """ Return the current transit before finishing.

            Returns
            -------
            ts: TimeStream
                Last (possibly incomplete) transit.
        """
        return self._finalize_transit()

    def append(self, ts):
        """ Append a timestream to the buffer list.
            This will strip eigenvector datasets if they are present.
        """
        for dname in ["evec", "eval", "erms"]:
            if dname in ts.datasets.keys():
                self.log.debug("Stripping dataset {}".format(dname))
                del ts[dname]
        self.tstreams.append(ts)

    def _finalize_transit(self):

        # Find where transit starts and ends
        if len(self.tstreams) == 0:
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

        # Concatenate timestreams
        ts = tod.concatenate(self.tstreams, start=start_ind, stop=stop_ind)
        _, dec = self.sky_obs.radec(self.src)
        ts.attrs["dec"] = dec._degrees
        ts.attrs["source_name"] = self.source
        ts.attrs["transit_time"] = self.cur_transit
        ts.attrs["observation_id"] = self.obs_id
        ts.attrs["tag"] = "{}_{}".format(
            self.source,
            ephem.unix_to_datetime(self.cur_transit).strftime("%Y%m%dT%H%M%S"),
        )
        ts.attrs["archivefiles"] = filenames

        self.tstreams = []
        self.cur_transit = None

        return ts

    def _transit_bounds(self):

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
    """ Interpolate TimeStream transits onto a regular grid in hour angle.

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

        ra, _ = self.sky_obs.radec(self.src)
        ra = ra._degrees

        # Convert input times to hour angle
        lha = unwrap_lha(self.sky_obs.unix_to_lsa(data.time), ra)

        # perform regridding
        success = 1
        try:
            new_grid, new_vis, ni = self._regrid(vis_data, weight, lha)
        except np.linalg.LinAlgError as e:
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
        if mpiutil.size > 1:
            new_vis = mpiarray.MPIArray.wrap(new_vis, axis=data.vis.distributed_axis)
            ni = mpiarray.MPIArray.wrap(ni, axis=data.vis.distributed_axis)

        # Create new container for output
        ra_grid = (new_grid + ra) % 360.0
        new_data = SiderealStream(axes_from=data, attrs_from=data, ra=ra_grid)
        new_data.redistribute("freq")
        new_data.vis[:] = new_vis
        new_data.weight[:] = ni
        new_data.attrs["source_ra"] = ra

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

        for ind in np.ndindex(*weight.shape):

            flag = np.flatnonzero(weight[ind] > 0.0)

            if flag.size > 0:

                imin, imax = np.percentile(flag, [0, 100]).astype(np.int)
                imax = imax + 1

                if self.num_begin:
                    weight[ind][imin:imin+self.num_begin] = 0.0

                if self.num_end:
                    weight[ind][imax-self.num_end:imax] = 0.0

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
        data.redistribute('freq')
        beam.redistribute('freq')

        # Dereference datasets
        bv = beam.beam[:].view(np.ndarray)
        bw = beam.weight[:].view(np.ndarray)

        flag = np.any(bw > 0.0, axis=(0, 1, 2))

        # Compute the hour angle of the source
        if 'ra' in data.index_map:
            ra = data.index_map['ra'][:]
        elif 'time' in data.index_map:
            ra = self.sky_obs.unix_to_lsa(data.time)
        else:
            raise RuntimeError("Unable to extract RA from input container.")

        hour_angle = beam.pix['phi'][:]

        new_hour_angle = (((ra - beam.attrs['cirs_ra']) + 180.0) % 360.0) - 180.0

        isort = np.argsort(new_hour_angle)
        new_hour_angle_sorted = new_hour_angle[isort]

        within_range = np.flatnonzero((new_hour_angle_sorted >= np.min(hour_angle[flag])) &
                                      (new_hour_angle_sorted <= np.max(hour_angle[flag])))

        if within_range.size < (2 * self.lanczos_width):
            raise RuntimeError("Not enough overlapping samples.")

        slc = slice(np.min(within_range) + self.lanczos_width,
                    np.max(within_range) - self.lanczos_width + 1)

        # Perform regridding
        new_bv, new_bw = self._resample(hour_angle, bv, bw, new_hour_angle_sorted[slc])

        # Create output container
        new_beam = TrackBeam(phi=new_hour_angle,
                             theta=np.repeat(np.median(beam.pix['theta'][:]),
                                          new_hour_angle.size),
                             axes_from=beam,
                             attrs_from=beam,
                             distributed=beam.distributed,
                             comm=beam.comm)

        new_beam.redistribute('freq')

        # Save to container
        new_beam.beam[:] = 0.0
        new_beam.weight[:] = 0.0

        new_beam.beam[:, :, :, isort[slc]] = new_bv
        new_beam.weight[:, :, :, isort[slc]] = new_bw

        return new_beam

    def _resample(self, xgrid, ygrid, wgrid, x):

        lza = regrid.lanczos_forward_matrix(xgrid, x, a=self.lanczos_width).T

        y = np.matmul(ygrid, lza)
        w = tools.invert_no_zero(np.matmul(tools.invert_no_zero(wgrid), lza**2))

        return y, w


class MakeHolographyBeam(task.SingleTask):
    """ Repackage a holography transit into a beam container.
        The visibilities will be grouped according to their respective 26 m
        input (along the `pol` axis, labelled by the 26 m polarisation of that input).
    """

    def process(self, data, inputmap):
        """ Package a holography transit into a beam container.

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

        # Sort by input
        for i, pg in enumerate(prod_groups):
            ipt_to_sort = (
                "input_a"
                if np.sum(np.where(prod[pg]["input_a"] == input_26m[i])[0]) == 1
                else "input_b"
            )
            pg = pg[np.argsort(prod[pg][ipt_to_sort])]
        inputs_sorted = inputs[np.argsort(inputs["chan_id"])]

        # Make new index map
        ra = data.attrs["source_ra"]
        phi = unwrap_lha(data.ra[:], ra)
        if "dec" not in data.attrs.keys():
            msg = (
                "Input stream must have a 'dec' attribute specifying "
                "declination of holography source."
            )
            self.log.error(msg)
            raise PipelineRuntimeError(msg)
        theta = np.ones_like(phi) * data.attrs["dec"]
        pol = np.array([inputmap[i].pol for i in input_26m], dtype="S1")

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
            track.beam[:, ip, :, :] = data.vis[:, prod_groups[ip], :]
            track.weight[:, ip, :, :] = data.weight[:, prod_groups[ip], :]

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
        data.redistribute('freq')
        beam.redistribute('freq')

        # Grab the stack specifications from the input sidereal stream
        prod = data.index_map['prod']
        reverse_stack = data.reverse_map['stack'][:]

        input_flags = data.input_flags[:]
        if not np.any(input_flags):
            input_flags = np.ones_like(input_flags)

        # Create output container
        if isinstance(data, SiderealStream):
            OutputContainer = SiderealStream
            output_kwargs = {'ra': data.ra[:]}
        else:
            OutputContainer = TimeStream
            output_kwargs = {'time': data.time[:]}

        stacked_beam = OutputContainer(axes_from=data, attrs_from=beam,
                                       distributed=True, comm=data.comm,
                                       **output_kwargs)

        stacked_beam.vis[:] = 0.0
        stacked_beam.weight[:] = 0.0

        stacked_beam.attrs['tag'] = '_'.join([beam.attrs['tag'],
                                              data.attrs['tag']])

        # Dereference datasets
        bv = beam.beam[:].view(np.ndarray)
        bw = beam.weight[:].view(np.ndarray)

        ov = stacked_beam.vis[:]
        ow = stacked_beam.weight[:]

        pol_filter = {'X': 'X', 'Y': 'Y',
                      'E': 'X', 'S': 'Y',
                      'co': 'co', 'cross': 'cross'}
        pol = [pol_filter.get(pp, None) for pp in self.telescope.polarisation]
        beam_pol = [pol_filter.get(pp, None) for pp in beam.index_map['pol'][:]]

        # Compute the fractional variance of the beam measurement
        frac_var = tools.invert_no_zero(bw * np.abs(bv)**2)

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

            weight = (input_flags[np.newaxis, aa, :] * input_flags[np.newaxis, bb, :] *
                      tools.invert_no_zero(np.abs(cross)**2 * (frac_var[:, aa_pol, aa, :] +
                                                               frac_var[:, bb_pol, bb, :])))

            if self.weight == 'inverse_variance':
                wss = weight
            else:
                wss = (weight > 0.0).astype(np.float32)

            # Accumulate variances in quadrature.  Save in the weight dataset.
            ov[:, ss, :] += wss * cross
            ow[:, ss, :] += wss**2 * tools.invert_no_zero(weight)

            # Increment counter
            counter[:, ss, :] += wss

        # Divide through by counter to get properly weighted visibility average
        ov[:] *= tools.invert_no_zero(counter)
        ow[:] = counter**2 * tools.invert_no_zero(ow[:])

        return stacked_beam

    @staticmethod
    def _resolve_pol(pol1, pol2, pol_axis):

        if 'co' in pol_axis:

            if pol1 == pol2:
                ipol = pol_axis.index('co')
            else:
                ipol = pol_axis.index('cross')

            return ipol, ipol

        else:

            if pol1 == pol2:
                ipol1 = pol_axis.index(pol1)
                ipol2 = pol_axis.index(pol2)
            else:
                ipol1 = pol_axis.index(pol2)
                ipol2 = pol_axis.index(pol1)

            return ipol1, ipol2


class RegisterHolographyProcessed(RegisterProcessedFiles):
    """ Register processed holography transit in temporary processed data
        database.
    """

    def process(self, output):
        """ Register and save an output file.

            Parameters
            ----------
            output: TrackBeam
                Transit to be saved.
        """

        # Create a tag for the output file name
        tag = output.attrs["tag"] if "tag" in output.attrs else self._count

        # Construct the filename
        outfile = self.output_root + str(tag) + ".h5"

        # Expand any variables in the path
        outfile = path.expanduser(outfile)
        outfile = path.expandvars(outfile)

        self.write_output(outfile, output)

        obs_id = output.attrs.get("observation_id", None)
        files = output.attrs.get("archivefiles", None)

        if mpiutil.rank0:
            # Add entry in database
            # TODO: check for duplicates ?
            append_product(
                self.db_fname,
                outfile,
                self.product_type,
                config=None,
                tag=self.tag,
                git_tags=self.git_tags,
                holobs_id=obs_id,
                archivefiles=files,
            )

        return None


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

    def process(self, transit):
        """ Perform the fit.

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
        freq = transit.index_map['freq']['centre'][local_slice]
        sigma = (0.7 * SPEED_LIGHT / (CHIME_CYL_W * freq)) * (
            360.0 / np.pi
        )
        sigma = sigma[:, np.newaxis] * np.ones(
            (1, ninput), dtype=sigma.dtype
        )

        # Find index into pol axis that yields copolar products
        pol_axis = list(transit.index_map["pol"])
        if "co" in pol_axis:
            copolar_slice = (slice(None), pol_axis.index("co"))
        else:
            this_pol =  np.array([pol_axis.index('S') if not ((ii // 256) % 2) else pol_axis.index('E')
                                  for ii in range(ninput)])
            copolar_slice = (slice(None), this_pol, np.arange(ninput))

        # Dereference datasets
        ha = transit.pix["phi"][:]

        vis = transit.beam[:].view(np.ndarray)
        vis = vis[copolar_slice]

        err = transit.weight[:].view(np.ndarray)
        err = np.sqrt(tools.invert_no_zero(err[copolar_slice]))

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

        # Save datasets while removing anomalous data
        fit.parameter[:] = np.where(np.isfinite(model.param[:]), model.param[:], 0.0)
        fit.parameter_cov[:] = np.where(np.isfinite(model.param_cov[:]), model.param_cov[:], 0.0)
        fit.chisq[:] = np.where(np.isfinite(model.chisq[:]), model.chisq, 0.0)
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

        if self.overwrite:
            track = track_in
        else:
            track = TrackBeam(axes_from=track_in, attrs_from=track_in,
                              distributed=track_in.distributed,
                              comm=track_in.comm)
            track["beam"] = track_in["beam"][:]
            track["weight"] = track_in["weight"][:]

        track["beam"][:] *= gain.gain[:][:, np.newaxis, :, np.newaxis]
        track["weight"][:] *= tools.invert_no_zero(np.abs(gain.gain[:]) ** 2)[
            :, np.newaxis, :, np.newaxis
        ]

        track["beam"][:] = np.where(np.isfinite(track["beam"][:]), track["beam"][:], 0)
        track["weight"][:] = np.where(np.isfinite(track["weight"][:]), track["weight"][:], 0)

        return track


class TransitStacker(task.SingleTask):
    """Apply gains to a holography transit

    Parameters
    ----------
    overwrite: bool (default: False)
        If True, overwrite the input TrackBeam.
    """

    weight = config.enum(["uniform", "inverse_variance"], default="uniform")

    def setup(self):
        self.stack = None
        self.variance = None
        self.pseudo_variance = None
        self.norm = None

    def process(self, transit):

        self.log.info("Weight is %s" % self.weight)

        if self.stack is None:
            self.log.info("Initializing transit stack.")
            self.stack = TrackBeam(axes_from=transit,
                                   distributed=transit.distributed,
                                   comm=transit.comm)

            #self.stack.add_dataset("observed_variance")
            #self.stack.add_dataset("number_of_observations")
            self.stack.redistribute("freq")

            self.log.info("Adding %s to stack." % transit.attrs['tag'])

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
            self.stack.weight[:] = (coeff ** 2) * tools.invert_no_zero(transit.weight[:])
            #self.stack.number_of_observations[:] = flag.astype(np.int)

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

            self.log.info("Adding %s to stack." % transit.attrs['tag'])

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
            self.stack.weight[:] += (coeff ** 2) * tools.invert_no_zero(transit.weight[:])
            #self.stack.number_of_observations[:] += flag

            self.variance += coeff * np.abs(transit.beam[:]) ** 2
            self.pseudo_variance += coeff * transit.beam[:] ** 2
            self.norm += coeff

        return None

    def process_finish(self):
        # Divide by norm to get average transit
        inv_norm = tools.invert_no_zero(self.norm)
        self.stack.beam[:] *= inv_norm
        self.stack.weight[:] = tools.invert_no_zero(self.stack.weight[:]) * self.norm ** 2

        self.variance = self.variance * inv_norm - np.abs(self.stack.beam[:]) ** 2
        self.pseudo_variance = self.pseudo_variance * inv_norm - self.stack.beam[:] ** 2

        # Calculate the covariance between the real and imaginary component
        # from the accumulated variance and psuedo-variance
        #self.stack.observed_variance[0] = 0.5 * (self.variance + self.pseudo_variance.real)
        #self.stack.observed_variance[1] = 0.5 * self.pseudo_variance.imag
        #self.stack.observed_variance[2] = 0.5 * (self.variance - self.pseudo_variance.real)

        # Create tag
        time_range = np.percentile(self.stack.attrs["transit_time"], [0, 100])
        self.stack.attrs["tag"] = "{}_{}_to_{}".format(
            self.stack.attrs["source_name"],
            ephem.unix_to_datetime(time_range[0]).strftime("%Y%m%dT%H%M%S"),
            ephem.unix_to_datetime(time_range[1]).strftime("%Y%m%dT%H%M%S"),
        )

        return self.stack


class FilterHolographyProcessed(task.MPILoggedTask):
    """ Filter list of archive files to exlclude holography transits
        that have already been processed for the given source, based
        on the records in the processed data database (file).
    """

    db_fname = config.Property(proptype=str)
    source = config.Property(proptype=str)

    def setup(self):
        # Read database of processed transits
        self.proc_transits = get_proc_transits(self.db_fname)

        # Query database for observations of this source
        hol_obs = None
        if mpiutil.rank0:
            hol_obs = list(get_holography_obs(self.source))
        self.hol_obs = mpiutil.bcast(hol_obs, root=0)
        mpiutil.barrier()

    def next(self, intervals):
        """ Filter list of files and time intervals.

            Parameters
            ----------
            intervals: ch_util.data_index.DataIntervalList
                Files and time intervals for transits.
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
            elif this_obs[0].id in [int(t["holobs_id"]) for t in self.proc_transits]:
                self.log.warning(
                    "Already processed transit for {}. Skipping.".format(
                        ephem.unix_to_datetime(start)
                    )
                )
            else:
                files += fi[0]

        self.log.info("Leaving next for task %s" % self.__class__.__name__)

        return files


def wrap_observer(obs):
    return ephem.SkyfieldObserverWrapper(
        lon=obs.longitude,
        lat=obs.latitude,
        alt=obs.altitude,
        lsd_start=obs.lsd_start_day,
    )


def unwrap_lha(lsa, src_ra):
    # ensure monotonic
    start_lsa = lsa[0]
    lsa = lsa - start_lsa
    lsa[lsa < 0] += 360.0
    lsa += start_lsa
    # subtract source RA
    return np.where(
        np.abs(lsa - src_ra) < np.abs(lsa - src_ra + 360.0),
        lsa - src_ra,
        lsa - src_ra + 360.0,
    )


def get_holography_obs(src):
    di.connect_database()
    db_src = di.HolographySource.get(di.HolographySource.name == src)
    db_obs = di.HolographyObservation.select().where(
        di.HolographyObservation.source == db_src
    )
    return db_obs
