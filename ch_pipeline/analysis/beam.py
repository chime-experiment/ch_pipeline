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

from ch_util import ephemeris as ephem
from ch_util import tools, layout
from ch_util import data_index as di


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
            db_src = di.HolographySource.get(di.HolographySource.name == self.db_source)
            db_runs = list(di.HolographyObservation.select().where(
                di.HolographyObservation.source == db_src
            ))
            db_runs = [(r.start_time, r.finish_time) for r in db_runs]
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
            self._transit_bounds(tstream.time[0])

        # check if we've accumulated enough past the transit
        if tstream.time[-1] > self.end_t:
            self.tstreams.append(tstream)
            final_ts = self._finalize_transit()
        elif tstream.time[-1] < self.start_t:
            pass
        else:
            self.tstreams.append(tstream)

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

        # Concatenate timestreams
        ts = tod.concatenate(self.tstreams, start=start_ind, stop=stop_ind)
        _, dec = self.sky_obs.radec(self.src)
        ts.attrs['dec'] = dec._degrees
        ts.attrs['source_name'] = self.source
        ts.attrs['transit_time'] = self.cur_transit
        ts.attrs['tag'] = "{}_{}".format(
            self.source, ephem.unix_to_datetime(self.cur_transit).strftime("%Y%m%dT%H%M%S")
        )

        self.tstreams = []
        self.cur_transit = None

        return ts

    def _transit_bounds(self, t0):
        # shortcut
        obs = self.sky_obs

        # subtract half a day from start time to ensure we don't get following day
        self.start_t = obs.lsa_to_unix(obs.unix_to_lsa(self.cur_transit) -
                                       self.ha_span / 2, t0 - 43200)
        self.end_t = obs.lsa_to_unix(obs.unix_to_lsa(self.cur_transit) +
                                     self.ha_span / 2, t0)

        # get bounds of observation from database
        this_run = [
            r for r in self.db_runs if r[0] < self.cur_transit and r[1] > self.cur_transit
        ]
        if len(this_run) == 0:
            msg = "Could not find source transit in holography database."
            self.log.error(msg)
            raise PipelineRuntimeError(msg)
        else:
            self.start_t = max(self.start_t, this_run[0][0])
            self.end_t = min(self.end_t, this_run[0][1])


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
        new_grid, new_vis, ni = self._regrid(vis_data, weight, lha)

        # mask out regions beyond bounds of this transit
        grid_mask = np.ones_like(new_grid)
        grid_mask[new_grid < lha.min()] = 0.
        grid_mask[new_grid > lha.max()] = 0.
        new_vis *= grid_mask
        ni *= grid_mask

        # Wrap to produce MPIArray
        if mpiutil.size > 1:
            new_vis = mpiarray.MPIArray.wrap(new_vis, axis=data.vis.distributed_axis)
            ni = mpiarray.MPIArray.wrap(ni, axis=data.vis.distributed_axis)

        # Create new container for output
        ra_grid = (new_grid + ra) % 360.
        new_data = SiderealStream(axes_from=data, attrs_from=data,
                                  ra=ra_grid)
        new_data.redistribute('freq')
        new_data.vis[:] = new_vis
        new_data.weight[:] = ni

        return new_data


class MakeHolographyBeam(task.SingleTask):
    """ Repackage a holography transit into a beam container.
    """

    def process(self, data, inputmap):

        # redistribute if needed
        data.redistribute('freq')

        prod = data.index_map['prod']

        # Figure out which inputs are the 26m
        input_26m = prod['input_a'][np.where(prod['input_a'] == prod['input_b'])[0]]
        if len(input_26m) != 2:
            msg = ("Did not find exactly two 26m inputs in the data.")
            self.log.error(msg)
            raise PipelineRuntimeError(msg)

        # Separate products into 26m pol
        


        # Get polarizations from inputs
        ant = []
        pol = np.empty(len(inputmap), dtype='<U1')
        pol[:] = '0'
        input_26m = []
        #input_by_pol = {'E': [], 'S': []}
        feed = np.empty(len(inputmap) // 2,
                        dtype=[('id_E', np.int32), ('id_S', np.int32), ('antenna', '<U8')])
        feed[:] = (-1, -1, '0')
        for i, ipt in enumerate(inputmap):
            if (isinstance(ipt, tools.CHIMEAntenna) or
                    isinstance(ipt, tools.HolographyAntenna)):
                if ipt.antenna not in ant:
                    ant.append(ipt.antenna)
                    feed_ind = len(ant) - 1
                    feed[feed_ind]['antenna'] = ipt.antenna
                else:
                    feed_ind = ant.index(ipt.antenna)
                pol[i] = ipt.pol
                #input_by_pol[ipt.pol].append(i)
                feed[feed_ind]['id_'+ipt.pol] = ipt.id
            else:
                continue
            if ipt.reflector == '26m_dish':
                input_26m.append(i)
        input_26m = np.array(input_26m)
        if len(input_26m) != 2:
            msg = ("Did not find exactly two 26m inputs in the data.")
            self.log.error(msg)
            raise PipelineRuntimeError(msg)
        #if len(input_by_pol['S']) != len(input_by_pol['E']):
        #    msg = ("Did not find same number of E and S inputs.")
        #    self.log.error(msg)
        #    raise PipelineRuntimeError(msg)

        # Map products onto feeds and polarizations
        prod_map = {k: [] for k in ['EE', 'SS', 'ES', 'SE']}
        feed_map = {k: [] for k in ['EE', 'SS', 'ES', 'SE']}
        for pi, p in enumerate(prod):
            this_pol = (pol[data.input[p['input_a']]['chan_id']] +
                        pol[data.input[p['input_b']]['chan_id']])
            if '0' in this_pol:
                continue
            feed_ind = np.where(feed['id_'+this_pol[0]] == data.input[p['input_a']]['chan_id'])[0]
            if len(feed_ind) == 0:
                feed_ind = np.where(feed['id_'+this_pol[1]] == data.input[p['input_b']]['chan_id'])[0]
            if len(feed_ind) == 0:
                continue
            prod_map[this_pol].append(pi)
            feed_map[this_pol].append(feed_ind[0])

        # 26m cross-pol is a special case
        cross_26m = np.where(np.logical_or(
            np.logical_and(prod['input_a'] == input_26m[0],
                           prod['input_b'] == input_26m[1]),
            np.logical_and(prod['input_a'] == input_26m[0],
                           prod['input_b'] == input_26m[1])
        ))[0]
        if len(cross_26m) != 1:
            msg = ("Did not find exactly one 26m cross-polar product in the data.")
            self.log.error(msg)
            raise PipelineRuntimeError(msg)
        cross_26m = cross_26m[0]
        missing_pol = (pol[data.input[prod[cross_26m]['input_a']]['chan_id']] +
                       pol[data.input[prod[cross_26m]['input_b']]['chan_id']])[::-1]
        feed_ind = np.where(feed['id_'+missing_pol[0]] == data.input[prod[cross_26m]['input_a']]['chan_id'])[0]
        if len(feed_ind) == 0:
            feed_ind = np.where(feed['id_'+missing_pol[1]] == data.input[prod[cross_26m]['input_b']]['chan_id'])[0]
        if len(feed_ind > 0):
            prod_map[missing_pol].append(cross_26m)
            feed_map[missing_pol].append(feed_ind[0])

        prod_map = {k: np.array(prod_map[k]) for k in prod_map.keys()}
        feed_map = {k: np.array(feed_map[k]) for k in feed_map.keys()}

        # Sort products by polarization
        #prod = data.index_map['prod']
        #prod_Xa = pol[data.input[prod['input_a']]['chan_id']] == 'E'
        #prod_Xb = pol[data.input[prod['input_b']]['chan_id']] == 'E'
        #prod_Ya = pol[data.input[prod['input_a']]['chan_id']] == 'S'
        #prod_Yb = pol[data.input[prod['input_b']]['chan_id']] == 'S'
        #prod_by_pol = {}
        #prod_by_pol['EE'] = np.logical_and(prod_Xa, prod_Xb)
        #prod_by_pol['SS'] = np.logical_and(prod_Ya, prod_Yb)
        #prod_by_pol['ES'] = np.logical_and(prod_Xa, prod_Yb)
        #prod_by_pol['SE'] = np.logical_and(prod_Ya, prod_Yb)

        # 26m cross-pol is a special case
        #cross_26m = np.where(np.logical_or(
        #    np.logical_and(prod['input_a'] == input_26m[0],
        #                   prod['input_b'] == input_26m[1]),
        #    np.logical_and(prod['input_a'] == input_26m[0],
        #                   prod['input_b'] == input_26m[1])
        #))[0]
        #if len(cross_26m) != 1:
        #    msg = ("Did not find exactly one 26m cross-polar product in the data.")
        #    self.log.error(msg)
        #    raise PipelineRuntimeError(msg)
        #cross_26m = cross_26m[0]
        #missing_pol = 'SE' if prod_by_pol['ES'][cross_26m] else 'SE'
        #prod_by_pol[missing_pol][cross_26m] = True

        ## Check dimensions
        #nfeed = np.sum(prod_by_pol['EE'])
        #same_dim = (
        #    nfeed != np.sum(prod_by_pol['SS']) and
        #    nfeed != np.sum(prod_by_pol['SE']) and
        #    nfeed != np.sum(prod_by_pol['ES']) and
        #    nfeed != len(input_by_pol['E']) and
        #    nfeed != len(input_by_pol['S'])
        #)
        #if not same_dim:
        #    msg = ("Polarization pairs found in data do not all have same dimensions: "
        #           "{}, {}, {}, {}".format(*[(p, len(prod_by_pol[p])) for p in prod_by_pol.keys()]))
        #    self.log.error(msg)
        #    raise PipelineRuntimeError(msg)

        # Make new index map
        phi = data.ra[:]
        if 'dec' not in data.attrs.keys():
            msg = ("Input stream must have a 'dec' attribute specifying "
                   "declination of holography source.")
            self.log.error(msg)
            raise PipelineRuntimeError(msg)
        theta = np.ones_like(phi) * data.attrs['dec']
        pol = np.array(['EE', 'SS', 'ES', 'SE'])
        #feed = np.array(
        #    [(input_by_pol['E'][i], input_by_pol['S'][i], ant[input_by_pol['S'][i]])
        #        for i in range(nfeed)],
        #    dtype=[('id_E', 'id_S', 'ant')]
        #)


        # Create new container and fill
        track = TrackBeam(theta=theta, phi=phi, track_type='drift', coords='celestial',
                          feed=feed, pol=pol, freq=data.freq[:], attrs_from=data,
                          distributed=data.distributed)
        for ip, p in enumerate(pol):
            print feed_map[p].shape
            print prod_map[p].shape
            #prod_ind = np.where(prod_by_pol[p])[0]
            track.beam[:, ip, feed_map[p], :] = data.vis[:, prod_map[p], :]
            track.weight[:, ip, feed_map[p], :] = data.weight[:, prod_map[p], :]
        #missing_pol_i = pol.index(missing_pol)
        #track.beam[:, missing_pol_i, :, :] = np.conjugate(track.beam[:, missing_pol_i, :, :])

        # TODO: Rotate polarization into correct basis?

        return track


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

