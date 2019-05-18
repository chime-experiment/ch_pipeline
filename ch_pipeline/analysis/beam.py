""" Tasks for beam measurement processing.

    Tasks
    =====

    .. autosummary::
        :toctree:

        TransitGrouper
        TransitRegridder
        MakeHolographyBeam
"""

import numpy as np

from caput import config, tod, mpiarray, mpiutil
from caput.pipeline import PipelineConfigError, PipelineRuntimeError

from draco.core import task
from draco.analysis.transform import Regridder
from draco.core.containers import SiderealStream, TrackBeam
from draco.util import tools

from ..core.processed_db import RegisterProcessedFiles, append_product, get_proc_transits
from ..core.containers import HolographyTransitFitParams, HolographyTransitGain

from ch_util import ephemeris as ephem
from ch_util import tools, layout
from ch_util import data_index as di
from ch_util.cal_utils import fit_point_source_transit

from os import path

from skyfield.constants import ANGVEL
SIDEREAL_DAY_SEC = 2 * np.pi / ANGVEL
SPEED_LIGHT = 299.7  # 10^6 m / s
CHIME_CYL_W = 20.  # m


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

    ha_span = config.Property(proptype=float, default=180.)
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
            msg = ("Could not find source {} in catalogue. "
                   "Must use same spelling as in `ch_util.ephemeris`.".format(self.source))
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
        if (tstream.time[0] - self.last_time) > 5*(tstream.time[1] - tstream.time[0]):
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
        for dname in ['evec', 'eval', 'erms']:
            if dname in ts.datasets.keys():
                self.log.debug("Stripping dataset {}".format(dname))
                del ts[dname]
        self.tstreams.append(ts)

    def _finalize_transit(self):

        # Find where transit starts and ends
        if len(self.tstreams) == 0:
            self.log.info("Did not find any transits.")
            return None
        self.log.debug("Finalising transit for {}...".format(
            ephem.unix_to_datetime(self.cur_transit))
        )
        all_t = np.concatenate([ts.time for ts in self.tstreams])
        start_ind = int(np.argmin(np.abs(all_t - self.start_t)))
        stop_ind = int(np.argmin(np.abs(all_t - self.end_t)))

        # Save list of filenames
        filenames = [ts.attrs['filename'] for ts in self.tstreams]

        # Concatenate timestreams
        ts = tod.concatenate(self.tstreams, start=start_ind, stop=stop_ind)
        _, dec = self.sky_obs.radec(self.src)
        ts.attrs['dec'] = dec._degrees
        ts.attrs['source_name'] = self.source
        ts.attrs['transit_time'] = self.cur_transit
        ts.attrs['observation_id'] = self.obs_id
        ts.attrs['tag'] = "{}_{}".format(
            self.source, ephem.unix_to_datetime(
                self.cur_transit).strftime("%Y%m%dT%H%M%S")
        )
        ts.attrs['archivefiles'] = filenames

        self.tstreams = []
        self.cur_transit = None

        return ts

    def _transit_bounds(self):

        # subtract half a day from start time to ensure we don't get following day
        self.start_t = self.cur_transit - self.ha_span / 360. / 2. * SIDEREAL_DAY_SEC
        self.end_t = self.cur_transit + self.ha_span / 360. / 2. * SIDEREAL_DAY_SEC

        # get bounds of observation from database
        this_run = [
            r for r in self.db_runs if r[1][0] < self.cur_transit and r[1][1] > self.cur_transit
        ]
        if len(this_run) == 0:
            self.log.warning("Could not find source transit in holography database for {}."
                             .format(ephem.unix_to_datetime(self.cur_transit)))
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
    ha_span = config.Property(proptype=float, default=180.)
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
        self.start = - self.ha_span / 2
        self.end = self.ha_span / 2

        try:
            self.src = ephem.source_dictionary[self.source]
        except KeyError:
            msg = ("Could not find source {} in catalogue. "
                   "Must use same spelling as in `ch_util.ephemeris`.".format(self.source))
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
        data.redistribute('freq')

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
        grid_mask[new_grid < lha.min()] = 0.
        grid_mask[new_grid > lha.max()] = 0.
        new_vis *= grid_mask
        ni *= grid_mask

        # Wrap to produce MPIArray
        if mpiutil.size > 1:
            new_vis = mpiarray.MPIArray.wrap(
                new_vis, axis=data.vis.distributed_axis)
            ni = mpiarray.MPIArray.wrap(ni, axis=data.vis.distributed_axis)

        # Create new container for output
        ra_grid = (new_grid + ra) % 360.
        new_data = SiderealStream(axes_from=data, attrs_from=data,
                                  ra=ra_grid)
        new_data.redistribute('freq')
        new_data.vis[:] = new_vis
        new_data.weight[:] = ni
        new_data.attrs['source_ra'] = ra

        return new_data


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
        data.redistribute('freq')

        prod = data.index_map['prod']
        inputs = data.index_map['input']

        # Figure out which inputs are the 26m
        input_26m = prod['input_a'][np.where(
            prod['input_a'] == prod['input_b'])[0]]
        if len(input_26m) != 2:
            msg = ("Did not find exactly two 26m inputs in the data.")
            self.log.error(msg)
            raise PipelineRuntimeError(msg)

        # Separate products by 26 m inputs
        prod_groups = []
        for i in input_26m:
            prod_groups.append(np.where(
                np.logical_or(prod['input_a'] == i, prod['input_b'] == i)
            )[0])

        # Check we have the expected number of products
        if (prod_groups[0].shape[0] != inputs.shape[0] or
            prod_groups[1].shape[0] != inputs.shape[0]):
            msg = ("Products do not separate into two groups with the length of the input map. "
                   "({:d}, {:d}) != {:d}").format(prod_groups[0].shape[0],
                                                  prod_groups[1].shape[0], inputs.shape[0])
            self.log.error(msg)
            raise PipelineRuntimeError(msg)

        # Sort by input
        for i, pg in enumerate(prod_groups):
            ipt_to_sort = (
                'input_a' if np.sum(np.where(prod[pg]['input_a'] == input_26m[i])[0]) == 1
                else 'input_b'
            )
            pg = pg[np.argsort(prod[pg][ipt_to_sort])]
        inputs_sorted = inputs[np.argsort(inputs['chan_id'])]

        # Make new index map
        ra = data.attrs['source_ra']
        phi = unwrap_lha(data.ra[:], ra)
        if 'dec' not in data.attrs.keys():
            msg = ("Input stream must have a 'dec' attribute specifying "
                   "declination of holography source.")
            self.log.error(msg)
            raise PipelineRuntimeError(msg)
        theta = np.ones_like(phi) * data.attrs['dec']
        pol = np.array([inputmap[i].pol for i in input_26m], dtype='S1')

        # Create new container and fill
        track = TrackBeam(theta=theta, phi=phi, track_type='drift', coords='celestial',
                          input=inputs_sorted, pol=pol, freq=data.freq[:], attrs_from=data,
                          distributed=data.distributed)
        for ip in range(len(pol)):
            track.beam[:, ip, :, :] = data.vis[:, prod_groups[ip], :]
            track.weight[:, ip, :, :] = data.weight[:, prod_groups[ip], :]

        return track


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
        tag = output.attrs['tag'] if 'tag' in output.attrs else self._count

        # Construct the filename
        outfile = self.output_root + str(tag) + '.h5'

        # Expand any variables in the path
        outfile = path.expanduser(outfile)
        outfile = path.expandvars(outfile)

        self.write_output(outfile, output)

        obs_id = output.attrs.get('observation_id', None)
        files = output.attrs.get('archivefiles', None)

        if mpiutil.rank0:
            # Add entry in database
            # TODO: check for duplicates ?
            append_product(self.db_fname, outfile, self.product_type, config=None,
                           tag=self.tag, git_tags=self.git_tags, holobs_id=obs_id,
                           archivefiles=files)

        return None


class HolographyTransitFit(task.SingleTask):
    """ Fit a gaussian to a transit.
    """

    def process(self, transit):
        """ Perform the gaussian fit.

            Parameters
            ----------
            transit: TrackBeam
                Transit to be fit to.

            Returns
            -------
            fit: HolographyTransitFitParams
                Fit parameters.
        """

        transit.beam.redistribute('freq')

        # Set bounds of fit to twice FWHM
        local_slice = slice(transit.beam.local_offset,
                            transit.beam.local_offset + transit.beam.local_shape[0])
        fit_bnd = (2 * SPEED_LIGHT / transit.freq[local_slice] /
                   CHIME_CYL_W / 1.95 / 2 / np.pi * 360.)

        # Collapse polarization axis to use function from ch_util
        tmp_shape = (transit.beam.shape[0], transit.beam.shape[1]*transit.beam.shape[2],
                     transit.beam.shape[3])

        # Flag missing data and outside bounds
        flagged = (transit.weight != 0.).reshape(tmp_shape)
        flagged = np.logical_and(
            flagged, (np.abs(transit.pix['phi']) < fit_bnd[:, np.newaxis])[:, np.newaxis, :]
        )

        # Perform fit
        res = fit_point_source_transit(
            transit.pix['phi'],
            transit.beam[:].reshape(tmp_shape),
            np.sqrt(tools.invert_no_zero(transit.weight[:].reshape(tmp_shape))),
            flagged
        )

        # Pack into container
        param_labels = ['peak_amplitude', 'centroid', 'fwhm', 'phase_intercept', 'phase_slope',
                        'phase_quad', 'phase_cube', 'phase_quart', 'phase_quint']
        fit = HolographyTransitFitParams(
            parameter=res[0].reshape(transit.beam.shape[:-1]),
            parameter_cov=res[1].reshape(transit.beam.shape[:-1]),
            param=param_labels, axes_from=transit
        )

        return fit


class DetermineHolographyGainsFromFits(task.SingleTask):
    """Determine holography gains of a transit from Gaussian fits to
    the transit.
    """

    def process(self, fits):
        """Use transit phase and 1/peak amplitude as the gain, peak normalizing
        the transit.

        Parameters
        ----------
        fits: TransitFitParams
        """

        gain = HolographyTransitGain(axes_from=fits)
        gain.gain[:] = (np.exp(-1j * np.radians(fits.parameter['phase_intercept'])) /
                        fits.parameter['peak_amplitude'])

        return gain


class ApplyHolographyGains(task.SingleTask):

    """Apply gains to a holography transit

    Parameters
    ----------
    overwrite: bool (default: False)
        if True, overwrite the input TrackBeam
    """

    overwrite = config.Property(proptype=bool, default=False)

    def process(self, track_in, gain):
        """Apply gain

        Parameters
        ----------
        track: draco.core.containers.TrackBeam
            holography track to apply gains to. Will apply gains to
            track['beam'], expecting axes to be freq, pol, input, ha
        gain: np.array
            gain to apply. Expected axes are freq, pol, and input
        """

        if self.overwrite:
            track = track_in
        else:
            track = TrackBeam(axes_from=track_in, attrs_from=track_in)
            track['beam'] = track_in['beam'][:]
            track['weight'] = track_in['weight'][:]

        track['beam'] *= gain.gain[:, :, :, np.newaxis]
        track['weight'] *= tools.invert_no_zero((gain.gain**2)[:, :, :, np.newaxis])

        return track


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
                o for o in self.hol_obs
                if (o.start_time >= start and o.start_time <= end) or
                (o.finish_time >= start and o.finish_time <= end) or
                (o.start_time <= start and o.finish_time >= end)
            ]

            if len(this_obs) == 0:
                self.log.warning("Could not find source transit in holography database for {}."
                                 .format(ephem.unix_to_datetime(start)))
            elif this_obs[0].id in [int(t['holobs_id']) for t in self.proc_transits]:
                self.log.warning("Already processed transit for {}. Skipping."
                                 .format(ephem.unix_to_datetime(start)))
            else:
                files += fi[0]

        self.log.info("Leaving next for task %s" % self.__class__.__name__)

        return files


def wrap_observer(obs):
    return ephem.SkyfieldObserverWrapper(
        lon=obs.longitude,
        lat=obs.latitude,
        alt=obs.altitude,
        lsd_start=obs.lsd_start_day
    )


def unwrap_lha(lsa, src_ra):
    # ensure monotonic
    start_lsa = lsa[0]
    lsa -= start_lsa
    lsa[lsa < 0] += 360.
    lsa += start_lsa
    # subtract source RA
    return np.where(np.abs(lsa - src_ra) < np.abs(lsa - src_ra + 360.),
                    lsa - src_ra, lsa - src_ra + 360.)


def get_holography_obs(src):
    di.connect_database()
    db_src = di.HolographySource.get(di.HolographySource.name == src)
    db_obs = di.HolographyObservation.select().where(
        di.HolographyObservation.source == db_src
    )
    return db_obs
