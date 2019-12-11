""" Tasks for beam measurement processing.

    Tasks
    =====

    .. autosummary::
        :toctree:

        TransitGrouper
        TransitRegridder
        MakeHolographyBeam
        RegisterHolographyProcessed
        FilterHolographyProcessed
"""

import numpy as np

from caput import config, tod, mpiarray, mpiutil
from caput.pipeline import PipelineConfigError, PipelineRuntimeError

from draco.core import task
from draco.analysis.transform import Regridder
from draco.core.containers import SiderealStream, TrackBeam

from ..core.processed_db import RegisterProcessedFiles, append_product

from ch_util import ephemeris as ephem
from ch_util import tools, layout
from ch_util import data_index as di

from os import path
import yaml

from caput.time import STELLAR_S
SIDEREAL_DAY_SEC = STELLAR_S * 24 * 3600


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

    ha_span = config.Property(proptype=float, default=180.)
    min_span = config.Property(proptype=float, default=0.)
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
        for dname in ['evec', 'eval', 'erms']:
            if dname in ts.datasets.keys():
                self.log.debug("Stripping dataset {}".format(dname))
                del ts[dname]
        self.tstreams.append(ts)

    def _finalize_transit(self):
        """Concatenate grouped time streams for the currrent transit."""

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

        dt = self.tstreams[0].time[1] - self.tstreams[0].time[0]
        if dt <= 0:
            self.log.warning("Time steps are not positive definite: dt={:.3f}".format(dt)
                             + " Skipping.")
            ts = None
        if stop_ind - start_ind > int(self.min_span / 360. * SIDEREAL_DAY_SEC / dt):
            if len(self.tstreams) > 1:
                # Concatenate timestreams
                ts = tod.concatenate(self.tstreams, start=start_ind, stop=stop_ind)
            else:
                ts = ts[0]
            _, dec = self.sky_obs.radec(self.src)
            ts.attrs['dec'] = dec._degrees
            ts.attrs['source_name'] = self.source
            ts.attrs['transit_time'] = self.cur_transit
            ts.attrs['observation_id'] = self.obs_id
            ts.attrs['tag'] = "{}_{}".format(
                self.source, ephem.unix_to_datetime(self.cur_transit).strftime("%Y%m%dT%H%M%S")
            )
            ts.attrs['archivefiles'] = filenames
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
        grid_mask[new_grid < lha.min()] = 0.
        grid_mask[new_grid > lha.max()] = 0.
        new_vis *= grid_mask
        ni *= grid_mask

        # Wrap to produce MPIArray
        if data.distributed:
            new_vis = mpiarray.MPIArray.wrap(new_vis, axis=data.vis.distributed_axis)
            ni = mpiarray.MPIArray.wrap(ni, axis=data.vis.distributed_axis)

        # Create new container for output
        ra_grid = (new_grid + ra) % 360.
        new_data = SiderealStream(axes_from=data, attrs_from=data,
                                  ra=ra_grid, comm=data.comm)
        new_data.redistribute('freq')
        new_data.vis[:] = new_vis
        new_data.weight[:] = ni
        new_data.attrs['cirs_ra'] = ra
        new_data.attrs['icrs_ra'] = ra_icrs

        return new_data


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
        data.redistribute('freq')

        prod = data.index_map['prod']
        inputs = data.index_map['input']

        # Figure out which inputs are the 26m
        input_26m = prod['input_a'][np.where(prod['input_a'] == prod['input_b'])[0]]
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
        if prod_groups[0].shape[0] != inputs.shape[0] or prod_groups[1].shape[0] != inputs.shape[0]:
            msg = ("Products do not separate into two groups with the length of the input map. "
                   "({:d}, {:d}) != {:d}").format(prod_groups[0].shape[0],
                                                  prod_groups[1].shape[0], inputs.shape[0])
            self.log.error(msg)
            raise PipelineRuntimeError(msg)

        # Sort based on the id in the layout database
        corr_id = np.array([inp.id for inp in inputmap])
        isort = np.argsort(corr_id)

        # Create new input axis using id and serial number in database
        inputs_sorted = np.array([(inputmap[ii].id, inputmap[ii].input_sn) for ii in isort],
                                 dtype=inputs.dtype)

        # Sort the products based on the input id in database and
        # determine which products should be conjugated.
        conj = []
        prod_groups_sorted = []
        for i, pg in enumerate(prod_groups):
            group_prod = prod[pg]
            group_conj = group_prod['input_a'] == input_26m[i]
            group_inputs = np.where(group_conj, group_prod['input_b'], group_prod['input_a'])
            group_sort = np.argsort(corr_id[group_inputs])

            prod_groups_sorted.append(pg[group_sort])
            conj.append(group_conj[group_sort])

        # Regroup by co/cross-pol
        copol, xpol = [], []
        prod_groups_cox = [pg.copy() for pg in prod_groups_sorted]
        conj_cox = [pg.copy() for pg in conj]
        input_pol = np.array(
            [ipt.pol if (tools.is_array(ipt) or tools.is_holographic(ipt))
             else inputmap[input_26m[0]].pol for ipt in inputmap]
        )
        for i, pg in enumerate(prod_groups_sorted):
            group_prod = prod[pg]
            # Determine co/cross in each prod group
            cp = (input_pol[np.where(conj[i], group_prod['input_b'], group_prod['input_a'])] ==
                  inputmap[input_26m[i]].pol)
            xp = np.logical_not(cp)
            copol.append(cp)
            xpol.append(xp)
            # Move products to co/cross-based groups
            prod_groups_cox[0][cp] = pg[cp]
            prod_groups_cox[1][xp] = pg[xp]
            conj_cox[0][cp] = conj[i][cp]
            conj_cox[1][xp] = conj[i][xp]
        # Check for compeleteness
        consistent = (
            np.all(copol[0] + copol[1] == np.ones(copol[0].shape)) and
            np.all(xpol[0] + xpol[1] == np.ones(xpol[0].shape))
        )
        if not consistent:
            msg = ("Products do not separate exclusively into co- and cross-polar groups.")
            self.log.error(msg)
            raise PipelineRuntimeError(msg)

        # Make new index map
        ra = data.attrs['cirs_ra']
        phi = unwrap_lha(data.ra[:], ra)
        if 'dec' not in data.attrs.keys():
            msg = ("Input stream must have a 'dec' attribute specifying "
                   "declination of holography source.")
            self.log.error(msg)
            raise PipelineRuntimeError(msg)
        theta = np.ones_like(phi) * data.attrs['dec']
        pol = np.array(['co', 'cross'], dtype='S5')

        # Create new container and fill
        track = TrackBeam(theta=theta, phi=phi, track_type='drift', coords='celestial',
                          input=inputs_sorted, pol=pol, freq=data.freq[:], attrs_from=data,
                          distributed=data.distributed)
        for ip in range(len(pol)):
            track.beam[:, ip, :, :] = data.vis[:, prod_groups_cox[ip], :]
            track.weight[:, ip, :, :] = data.weight[:, prod_groups_cox[ip], :]
            if np.any(conj_cox[ip]):
                track.beam[:, ip, conj_cox[ip], :] = track.beam[:, ip, conj_cox[ip], :].conj()

        # Store 26 m inputs
        track.attrs['26m_inputs'] = [inputs[ii] for ii in input_26m]

        return track


class RegisterHolographyProcessed(RegisterProcessedFiles):
    """Register a processed (fringestopped, regridded) holography transit in
    the processed data database (at the moment this is a YAML file,
    specified by the 'db_fname' config parameter).

    Saves the transit to with the prefix specified in the  'output_root'
    config parameter.
    """

    def process(self, output):
        """Register and save a processed transit.

        Parameters
        ----------
        output: TrackBeam
            The transit to be saved. Should include attributes
            'observation_id' and 'archivefiles.
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

        if output.distributed and output.comm.rank != 0:
            pass
        else:
            # Add entry in database
            # TODO: check for duplicates ?
            append_product(self.db_fname, outfile, self.product_type, config=None,
                           tag=self.tag, git_tags=self.git_tags, holobs_id=obs_id,
                           archivefiles=files)

        return None


class FilterHolographyProcessed(task.MPILoggedTask):
    """Filter holography transit DataIntervals produced by `io.QueryDatabase`
    to exclude those that have already been registered in the database
    (in the file specified by config parameter 'db_fname').
    """

    db_fname = config.Property(proptype=str)
    source = config.Property(proptype=str)

    def setup(self):
        """Read processed transits from database ('db_fname' parameter) and
        and load list of observations from holography database.
        """

        # Read database of processed transits
        self.proc_transits = get_proc_transits(self.db_fname)

        # Query database for observations of this source
        hol_obs = None
        if mpiutil.rank0:
            hol_obs = list(get_holography_obs(self.source))
        self.hol_obs = mpiutil.bcast(hol_obs, root=0)
        mpiutil.barrier()

    def next(self, intervals):
        """Filter files and time intervals to exclude those already processed.

        Parameters
        ----------
        intervals: list of ch_util.data_index.DataInterval
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
            lsd_start=obs.lsd_start_day
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
    lsa[lsa < 0] += 360.
    lsa += start_lsa
    # subtract source RA
    return np.where(np.abs(lsa - src_ra) < np.abs(lsa - src_ra + 360.),
                    lsa - src_ra, lsa - src_ra + 360.)


def get_holography_obs(src):
    """Query database for list of all holography observations for the given
    source.

    Parameters
    ----------
    src: str
        Source name.

    Returns
    -------
    db_obs: list of ch_util.data_index.HolographyObservation
        Observations of this source.
    """
    di.connect_database()
    db_src = di.HolographySource.get(di.HolographySource.name == src)
    db_obs = di.HolographyObservation.select().where(
        di.HolographyObservation.source == db_src
    )
    return db_obs


def get_proc_transits(db_fname):
    """Read processed holography transits from the processed database
    YAML file.

    Parameters
    ----------
    db_fname: str
        Path to YAML database file.
    """

    with open(db_fname, 'r') as fh:
        entries = yaml.load(fh)
    entries_filt = []
    for e in entries:
        if isinstance(e, dict) and 'holobs_id' in e.keys():
            entries_filt.append(e)
    return entries_filt
