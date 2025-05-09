"""Find and characterize spectral lines."""

import numpy as np
import scipy.constants
from caput import config, time
from ch_util import cal_utils
from cora.util import units
from draco.analysis.ringmapmaker import find_grid_indices
from draco.core import io, task
from draco.util import tools
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import pdist
from skyfield.units import Angle

from ..core import containers


class FindSpectralLines(task.SingleTask):
    """Identify candidate spectral line features in continuum-filtered RingMap data.

    This task detects statistically significant spectral excursions, either positive
    (emission) or negative (absorption), that are consistent across polarizations.
    Outliers are clustered in (frequency, ra, el) space using a distance metric
    scaled to the instrument's spectral and angular resolution. Each cluster is
    interpreted as a potential spectral line and recorded in a `SpectralLineCatalog`.

    Attributes
    ----------
    delay_cutoff : float
        Delay cutoff in microseconds used during the continuum filtering step.
        Defines the resolution in frequency and sets the scale for associating
        outliers into candidate features.
    nsigma : float
        Maximum allowed clustering distance (in units of normalized coordinate space)
        between outliers to be grouped as a single spectral line.
    """

    delay_cutoff = config.Property(proptype=float, default=0.20)
    nsigma = config.Property(proptype=float, default=2.0)

    def setup(self, manager):
        """Set up the telescope instance.

        Parameters
        ----------
        manager : ProductManager
            Product manager from which a telescope instance is extracted.
        """
        self.tel = io.get_telescope(manager)

        # Save the minimum north-south separation
        xind, yind, min_xsep, min_ysep = find_grid_indices(self.tel.baselines)
        self.min_xsep = min_xsep
        self.max_ysep = min_ysep * np.max(np.abs(yind))

    def process(self, ringmap, mask):
        """Search for spectral line candidates in a continuum-filtered RingMap.

        This method identifies pixels marked as outliers in the input mask,
        clusters them based on proximity in frequency and sky position, and
        selects the most significant pixel from each cluster as a candidate
        spectral line. Both emission and absorption features are retained.

        Coordinates are scaled by the maps effective spectral and spatial
        resolution to define a normalized clustering metric. Clustering is performed
        using single-linkage hierarchical clustering, and results are returned
        as entries in a `SpectralLineCatalog`.

        Parameters
        ----------
        ringmap : containers.RingMap
            Continuum-filtered ringmap.

        mask : containers.RingMapMask
            Binary mask marking outlier pixels, with `True` values indicating
            statistically significant excursions.  This can be generated using
            the `draco.analysis.flagging.FindBeamformedOutliers` task.

        Returns
        -------
        out : containers.SpectralLineCatalog
            Catalog of detected spectral lines. Each entry includes the
            sky position (RA, Dec) in ICRS coordinates, redshift
            (estimated from line frequency *assuming* a 21 cm signal),
            line depth, signal-to-noise ratio, and number of outlier pixels.
        """
        # Redistribute both ringmap and mask over frequency
        ringmap.redistribute("freq")
        mask.redistribute("freq")

        # Get the date
        unix = float(self.tel.lsd_to_unix(np.mean(ringmap.attrs["lsd"])))
        skytime = time.unix_to_skyfield_time(unix)

        # Extract axes
        freq = mask.freq[:]
        ra = mask.ra[:]
        el = mask.index_map["el"][:]

        dfreq = np.median(np.abs(np.diff(freq)))

        # Calculate the average of the map over polarisation.
        rmp = ringmap.map[0].local_array
        wmp = ringmap.weight[:].local_array

        if rmp.shape[0] > 1:
            wm = np.sum(wmp, axis=0)
            rm = np.sum(wmp * rmp, axis=0) * tools.invert_no_zero(wm)
        else:
            wm = wmp[0]
            rm = rmp[0]

        # Extract parameters describing how the data is distributed across frequencies
        offset_local = mask.mask[:].local_offset[1:]
        shape_local = mask.mask[:].local_shape[1:]

        local_freq_index = np.arange(offset_local[0], offset_local[0] + shape_local[0])

        # Identify outliers from mask.  Require it be an outlier in all polarisations.
        flag = np.all(mask.mask[:].local_array, axis=0)

        outlier_local = np.nonzero(flag)

        # outlier_local indexes into the local array, so we need to add the offsets
        # to obtain the global index
        outlier = [list(ind + off) for ind, off in zip(outlier_local, offset_local)]

        # Combine the global index of outliers across all ranks
        def concatenate_ranks(x, comm):
            return np.array([r for rank in comm.allgather(x) for r in rank])

        index = [concatenate_ranks(olist, self.comm) for olist in outlier]

        # Extract the 3D coordinates corresponding to outliers
        noutlier = index[0].size
        raw_coord = np.zeros((noutlier, 3), dtype=float)
        raw_coord[:, 0] = freq[index[0]]
        raw_coord[:, 1] = ra[index[1]]
        raw_coord[:, 2] = el[index[2]]

        src_dec = np.arcsin(raw_coord[:, 2]) + np.radians(self.tel.latitude)

        # Define the expected separation for a single source
        max_ysep = self.max_ysep
        freq_eval = raw_coord[:, 0]
        if "beamform_ns_nsmax" in ringmap.attrs:
            max_ysep = ringmap.attrs["beamform_ns_nsmax"]
            if ringmap.attrs["beamform_ns_scaled"]:
                freq_eval = ringmap.attrs["beamform_ns_freqmin"]

        wavelength = scipy.constants.c / (freq_eval * 1e6)

        delta = np.zeros((noutlier, 3), dtype=float)
        delta[:, 0] = 1.0 / (2.35482 * self.delay_cutoff)
        delta[:, 1] = cal_utils.guess_fwhm(freq_eval, pol="X", dec=src_dec, sigma=True)
        delta[:, 2] = 0.85 * wavelength / (2.35482 * max_ysep)

        coord = raw_coord / delta

        # Compute pairwise distances and cluster
        dists = pdist(coord)  # pairwise distances in rescaled space
        linkage_matrix = linkage(dists, method="single")  # or 'complete', 'average'

        cluster_ids = fcluster(linkage_matrix, t=self.nsigma, criterion="distance")

        uniq_clusters = np.unique(cluster_ids)
        nuniq = uniq_clusters.size

        # Create output container
        out = containers.SpectralLineCatalog(object_id=nuniq)

        # Loop over clusters
        for cc, cluster_id in enumerate(uniq_clusters):

            # Identify which outliers are in this cluster,
            # then identify which are held on this rank
            this_group = np.flatnonzero(cluster_ids == cluster_id)
            this_rank = np.array(
                [gg for gg in this_group if index[0][gg] in local_freq_index]
            )
            local_npix = this_rank.size

            # Check if we have cluster members on this rank
            if local_npix > 0:

                # Need to convert back to local indices to extract map values
                local_ind = tuple(
                    [ind[this_rank] - off for ind, off in zip(index, offset_local)]
                )

                map_values = rm[local_ind]
                s2n_values = np.abs(map_values) * np.sqrt(wm[local_ind])

                # Identify the maximum pixel within the local part of this cluster
                imax = np.argmax(np.abs(map_values))
                local_max = map_values[imax]
                local_s2n = s2n_values[imax]
                local_imax = this_rank[imax]

            else:
                local_max = 0.0
                local_s2n = 0.0
                local_imax = -1

            # Take maximum over all ranks
            local_info = [local_max, local_s2n, local_imax, local_npix]
            all_info = self.comm.allgather(local_info)

            global_max, global_s2n, global_imax, _ = max(
                all_info, key=lambda x: np.abs(x[0])
            )

            global_npix = np.sum([info[-1] for info in all_info])

            # Get the coordinates of the global maximum
            src_freq, src_ra, src_el = raw_coord[global_imax, :]

            # Convert from CIRS to ICRS
            src_body = self.tel.star_cirs(
                ra=Angle(radians=np.radians(src_ra)),
                dec=Angle(radians=np.arcsin(src_el) + np.radians(self.tel.latitude)),
                epoch=skytime,
            )

            # Save all results for this source to the output container
            out["position"]["ra"][cc] = src_body.ra._degrees
            out["position"]["dec"][cc] = src_body.dec._degrees
            out["frequency"]["freq"][cc] = src_freq
            out["frequency"]["freq_error"][cc] = dfreq
            out["redshift"]["z"][cc] = units.nu21 / src_freq - 1.0
            out["redshift"]["z_error"][cc] = units.nu21 * dfreq / src_freq**2
            out["flux"]["continuum"][cc] = 0.0
            out["flux"]["line_depth"][cc] = global_max
            out["significance"]["signal_to_noise"][cc] = global_s2n
            out["significance"]["npixels"][cc] = global_npix

            # Print out message
            self.log.info(
                f"Adding cluster {cc} of {nuniq} to catalog:  "
                f"RA = {src_body.ra._degrees:0.2f} deg, "
                f"Dec = {src_body.dec._degrees:0.2f} deg, "
                f"freq = {src_freq:0.2f} MHz, "
                f"flux = {1000 * global_max:0.1f} mJy, "
                f"S/N = {global_s2n:0.1f}"
            )

        # Re-sort the catalog based on line depth
        isort = np.argsort(np.abs(out["flux"]["line_depth"][:]))[::-1]

        for dset in out.datasets.keys():
            for field in out[dset].dtype.names:
                out[dset][field][:] = out[dset][field][isort]

        # Return output container
        return out
