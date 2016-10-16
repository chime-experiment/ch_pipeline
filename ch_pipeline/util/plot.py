import os
import time
import numpy as np
import inspect

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from matplotlib.backends.backend_pdf import PdfPages

from scipy.interpolate import interp1d

from caput import pipeline, config, mpiutil

from ch_util import ephemeris
from ch_util import cal_utils
from ch_util import fluxcat

from ..core import task, containers

class Destripe(task.SingleTask):

    def process(self, cmap):

        # Make sure we are distributed over frequency
        cmap.redistribute('freq')

        # Extract csd, ra for this map
        csd = cmap.attrs['csd']
        if hasattr(csd, '__iter__'):
            csd_list = csd
        else:
            csd_list = [csd]

        ra = cmap.index_map['ra']

        # Determine quiet regions based on time
        flag_quiet = np.ones(len(ra), dtype=np.bool)
        for cc in csd_list:
            flag_quiet &= flag_quiet_time(ephemeris.csd_to_unix(cc + ra/360.0))

        # Only subtract from RAs that have nonzero weight
        if 'rms' not in cmap.datasets:
            flag_sub = np.ones(len(ra), dtype=np.bool)
            partial_sub = 0
        else:
            partial_sub = 1

        # Loop over frequencies and polarizations
        for lfi, fi in cmap.map[:].enumerate(0):
            for lpi, pi in cmap.map[:].enumerate(1):

                this_map = cmap.map[fi, pi]

                if partial_sub:
                    flag_sub = cmap.datasets['rms'][fi, pi] > 0.0
                    flag_comp =  flag_quiet & flag_sub
                else:
                    flag_comp = flag_quiet

                # Subtract median value taken over quiet region of the sky
                if np.any(flag_sub) and np.any(flag_comp):
                    cmap.map[fi, pi][flag_sub] -= np.median(this_map[flag_comp], axis=0, keepdims=True)

        # Return destriped map
        return cmap


class ExtractPhotometry(task.SingleTask):

    pol = config.Property(proptype=list, default=['EE','SS'])
    beam = config.Property(proptype=int, default=0)

    nsources = config.Property(proptype=int, default=100)
    min_dec = config.Property(proptype=float, default=-20.0)

    window_nsig = config.Property(proptype=float, default=3.0)
    real_map = config.Property(proptype=bool, default=True)

    plot_index = config.Property(proptype=list, default=[])
    plot_file = config.Property(proptype=str, default="photometry")

    def setup(self):

        # Sort the sources based on flux mid-band
        source_list = fluxcat.FluxCatalog.sort()[0:self.nsources]
        self.source_list = [ss for ss in source_list if fluxcat.FluxCatalog[ss].dec > self.min_dec]

        self.nsources = len(self.source_list)

        # Determine the fit model
        self.func = cal_utils.func_real_dirty_gauss if self.real_map else cal_utils.func_dirty_gauss
        self.param_name = inspect.getargspec(self.func(None)).args[1:]

        self.ioffset = self.param_name.index('offset')

    def process(self, cmap):

        # Make sure we are distributed over frequency
        cmap.redistribute('freq')

        freq = cmap.freq['centre'][:]
        bwidth = cmap.freq['width'][:]

        # Determine polarization selection
        ipol = np.sort([ ii for ii, pp in enumerate(cmap.index_map['pol']) if pp in self.pol ])
        pol = [ cmap.index_map['pol'][ii] for ii in ipol ]

        # Extract ra, dec
        ra = cmap.index_map['ra'][:]
        el = cmap.index_map['el'][:]
        dec = cal_utils._el_to_dec(el)

        # Create container to hold results
        photo = containers.Photometry(source=np.array(self.source_list, dtype='S'),
                                      param=np.array(self.param_name, dtype='S'),
                                      pol=np.array(pol), axes_from=cmap, distributed=True)

        # Create pdf files for plotting
        if self.plot_index:
            # Define filename
            tag = cmap.attrs['tag'] if 'tag' in cmap.attrs else self._count
            plot_file = self.plot_file + str(tag) + '_'

            # Open pdf files
            self.plot = {}
            for lfi, fi in cmap.map[:].enumerate(0):
                for lpi, pi in enumerate(ipol):
                    pind = [fi, pi]
                    if pind in self.plot_index:
                        this_plot_file = plot_file + pol[lpi] + "_%0.2fMHz.pdf" % freq[fi]
                        self.plot[tuple(pind)] = OutputPdf(this_plot_file, dpi=50)

        # Loop over sources
        for ss, source_name in enumerate(self.source_list):

            print "(%d of %d) %s" % (ss+1, self.nsources, source_name)

            # Create PyEphem body for the source
            fsrc = fluxcat.FluxCatalog[source_name]
            src = ephemeris._ephem_body_from_ra_dec(fsrc.ra, fsrc.dec, source_name)
            src.compute()

            # Loop over frequencies
            for lfi, fi in cmap.map[:].enumerate(0):

                # Loop over polarizations
                for lpi, pi in enumerate(ipol):

                    source_window_ra = (self.window_nsig *
                                        cal_utils.guess_fwhm(freq[fi], pol=pol[lpi][0], dec=src.dec, sigma=True))
                    source_window_dec = 1.5 * self.window_nsig * guess_fwhm_synth(freq[fi], sigma=True)

                    # Extract map near source
                    map_slc_ra, map_slc_dec = _point_source_slice(cmap, src,
                                                                  source_window=[source_window_ra, source_window_dec])

                    sra = ra[map_slc_ra]
                    sdec = dec[map_slc_dec]
                    smap = cmap.map[fi, pi, map_slc_ra, self.beam, map_slc_dec]
                    srms = cmap.rms[fi, pi, map_slc_ra]

                    # Extract dirty beam near source
                    beam_slc_ra, beam_slc_dec = _point_source_slice(cmap, src,
                                                                    source_window=[source_window_ra, None])

                    sbeam = (ra[beam_slc_ra],
                             dec[beam_slc_dec],
                             cmap.dirty_beam[fi, pi, beam_slc_ra, self.beam, beam_slc_dec])

                    # Fit subregion to model
                    param_name, param, param_cov, resid_rms = cal_utils.fit_point_source_map(
                                                                sra, sdec, smap, srms, dirty_beam=sbeam,
                                                                real_map=self.real_map, freq=freq[fi],
                                                                ra0=ephemeris.peak_RA(src, deg=True),
                                                                dec0=np.degrees(src.dec))

                    # Save results to container
                    photo.parameter[fi, lpi, :, ss] = param
                    photo.parameter_cov[fi, lpi, :, :, ss] = param_cov
                    photo.rms[fi, lpi, ss] = resid_rms

                    # Plot best-fit
                    pind = [fi, pi]
                    if pind in self.plot_index:

                        unit = cmap.map.attrs.get('unit', 'Jansky')
                        axis = cmap.map.attrs.get('axis')

                        title = ("Frequency %0.2f MHz | Bandwidth %0.2f MHz | Polarization %s" %
                                (freq[fi], bwidth[fi], pol[lpi]))

                        save_kwargs = _plot_photo_fit(sra, sdec, smap, param, dirty_beam=sbeam,
                                                      unit=unit, source_name=src, freq=freq[fi],
                                                      real_map=self.real_map, title=title)

                        save_kwargs['bbox_inches'] = 'tight'

                        self.plot[tuple(pind)].save(**save_kwargs)

                        plt.close()


                    # Subtract best-fit model
                    db_ra, db_el = sbeam[0], cal_utils._dec_to_el(sbeam[1])
                    model = self.func(interp1d(db_el, sbeam[2],
                                               copy=False, kind='cubic', axis=-1,
                                               bounds_error=False, fill_value=0.0))

                    model_est = np.reshape(model([db_ra, db_el], *param), sbeam[2].shape) - param[self.ioffset]
                    cmap.map[fi, pi, beam_slc_ra, self.beam, beam_slc_dec] -= model_est


        # Close pdf files
        for pdf in self.plot.values():
            pdf.close()

        # Return
        return photo

def guess_fwhm_synth(freq, sigma=False):

    fwhm = np.degrees((3e2 / freq) / 20.0)

    # If requested return standard deviation, otherwise return fwhm
    if sigma:
        return fwhm / 2.35482
    else:
        return fwhm


def _point_source_slice(cmap, source_name, source_window=[10.0, 10.0]):

    import ephem

    # Find the source in ephemeris
    if isinstance(source_name, str):
        if source_name not in ephemeris.source_dictionary:
            ValueError("%s not in ephemeris." % source_name)

        src = ephemeris.source_dictionary[source_name]
        src.compute()

    elif isinstance(source_name, ephem.FixedBody):
        src = source_name

    else:
        ValueError("source_name must be string or PyEphem body.")

    # Extract peak_RA, Dec
    ra = ephemeris.peak_RA(src, deg=True)
    dec = np.degrees(src.dec)

    # Determine source window
    if hasattr(source_window, '__iter__'):
        if len(source_window) == 1:
            source_window = [source_window[0], source_window[0]]
        elif len(source_window) > 2:
            source_window = source_window[0:2]
    else:
        source_window = [source_window, source_window]

    # Define small region around source
    if source_window[0] is not None:
        ra_range = [ra - source_window[0], ra + source_window[0]]
    else:
        ra_range = None

    if source_window[1] is not None:
        dec_range = [dec - source_window[1], dec + source_window[1]]
    else:
        dec_range = None

    # Extract ra, el axis
    ra = cmap.index_map['ra']
    el = cmap.index_map['el']

    # Convert el to dec
    dec = np.degrees(np.arcsin(el)) + ephemeris.CHIMELATITUDE

    # Create slice that defines subregion
    if ra_range is None:
        slc_ra = slice(None)
    else:
        slc_ra = slice(*[np.argmin(np.abs(bb - ra)) for bb in ra_range])

    if dec_range is None:
        slc_el = slice(None)
    else:
        slc_el = slice(*[np.argmin(np.abs(bb - dec)) for bb in dec_range])

    # Extract subregion around source
    return slc_ra, slc_el


def _get_map(cmap, dataset='map', index=None, avg_beam=False):
    """ This function extracts a dataset and flips the axis
        to make sure the last two axis are (dec, ra) for
        display purposes.
    """

    # Deal with requests for multiple datasets
    if hasattr(dataset, '__iter__'):
        output = []
        for dset in dataset:
            ra, dec, arr = _get_map(cmap, dataset=dset, index=index, avg_beam=avg_beam)
            output.append(arr)
        return ra, dec, output

    # Extract axis
    axis = list(cmap.datasets[dataset].attrs['axis'])

    # Extract array
    arr = np.array(cmap.datasets[dataset][:])

    # If requested, sum over beam axis
    if avg_beam:
        try:
            bind = axis.index('beam')
            arr = np.mean(arr, axis=bind)
            axis.pop(bind)
        except ValueError:
            pass

    # Get ra, dec
    ra = cmap.index_map['ra']
    el = cmap.index_map['el']

    if np.abs(cmap.index_map['el']).max() > 1.0:
        sin_za = np.linspace(-1.0, 1.0, cmap.index_map['el'].size)
    else:
        sin_za = cmap.index_map['el']
    dec = np.degrees(np.arcsin(sin_za)) + ephemeris.CHIMELATITUDE

    # If requested apply index
    if index is not None:
        arr = arr[index]

    # Return
    return ra, dec, arr


def _plot_photo_fit(ra, dec, submap, param, unit='Jy', freq=600.0,
                             source_name=None, dirty_beam=None, real_map=False,
                             title=None, color_map='inferno', fontsize=16):

    import ephem

    font = {'family' : 'sans-serif',
            'weight' : 'normal',
            'size'   : 16}

    plt.rc('font', **font)

    output_kwargs = {}

    el = cal_utils._dec_to_el(dec)

    do_dirty = (dirty_beam is not None) and (len(dirty_beam) == 3)
    if do_dirty:

        db_ra, db_dec, db = dirty_beam

        db_el = cal_utils._dec_to_el(db_dec)

        # Create 1d vectors that span the (ra, dec) grid
        #coord = [xx for xx in np.meshgrid(ra, el)]

        coord = [ra, el]

        if real_map:
            model = cal_utils.func_real_dirty_gauss(interp1d(db_el, db, copy=False, kind='cubic', axis=-1,
                                                                        bounds_error=False, fill_value=0.0))
        else:
            model = cal_utils.func_dirty_gauss(interp1d(db_el, db, copy=False, kind='cubic', axis=-1,
                                                                   bounds_error=False, fill_value=0.0))

        param_name = inspect.getargspec(model).args[1:]

    else:
        model = cal_utils.func_2d_gauss
        param_name = inspect.getargspec(model).args[1:]

        # Create 1d vectors that span the (ra, dec) grid
        coord = [ra, dec]

    # Determine model and residuals
    submap_model = np.reshape(model(coord, *param), submap.shape)
    submap_resid = submap - submap_model

    # Extract coordinates of the source
    ra0 = param[param_name.index('centroid_x')]
    dec0 = param[param_name.index('centroid_y')]

    # Find the source in ephemeris
    if isinstance(source_name, str):
        if source_name not in ephemeris.source_dictionary:
            ValueError("%s not in ephemeris." % source_name)

        src = ephemeris.source_dictionary[source_name]
        src.compute()

    elif isinstance(source_name, ephem.FixedBody):
        src = source_name
        source_name = src.name

    else:
        source_name = None

    # Determine expected ra, dec
    if source_name is not None:
        ra_src = ephemeris.peak_RA(src, deg=True)
        dec_src = np.degrees(src.dec)
    else:
        ra_src = None
        dec_src = None

    # Determine range
    vmin = np.min([submap.min(), submap_model.min()])
    vmax = 1.1*np.max([submap.max(), submap_model.max()])

    xrng = [ra.min(), ra.max()]
    yrng = [dec.min(), dec.max()]

    extent = xrng + yrng

    # Set plot parameters
    cm = matplotlib.cm.__dict__[color_map]

    mrk = ['*', 'o']
    mrk_clr = ['blue', 'fuchsia']
    mrk_sz = [14, 12]

    ls = ['--', '--']
    ls_sz = [2.0, 2.0]

    # Define sub-routines for handling common tasks
    def plot_centroid():

        if (ra_src is not None) and (dec_src is not None):
            plt.plot(ra_src, dec_src, marker=mrk[0], color=mrk_clr[0], markersize=mrk_sz[0], linestyle='None')

        if (ra0 is not None) and (dec0 is not None):
            plt.plot(ra0, dec0, marker=mrk[1], color=mrk_clr[1], markersize=mrk_sz[1], linestyle='None')

    def plot_centroid_ra():

        if ra_src is not None:
            plt.vlines(ra_src, *plt.ylim(), color=mrk_clr[0], linestyle=ls[0], linewidth=ls_sz[0])

        if ra0 is not None:
            plt.vlines(ra0, *plt.ylim(), color=mrk_clr[1], linestyle=ls[1], linewidth=ls_sz[1])

    def plot_centroid_dec():

        if ra_src is not None:
            plt.vlines(dec_src, *plt.ylim(), color=mrk_clr[0], linestyle=ls[0], linewidth=ls_sz[0])

        if ra0 is not None:
            plt.vlines(dec0, *plt.ylim(), color=mrk_clr[1], linestyle=ls[1], linewidth=ls_sz[1])

    def show_colorbar():

        cbar = plt.colorbar(img, cmap=cm, pad=0.05)
        cbar.ax.get_yaxis().labelpad = 18
        cbar.ax.set_ylabel(unit)

    # Create text boxt
    txt_box = []
    if source_name is not None:
        txt_box.append(source_name.replace('_', ' '))
        txt_box.append('')

        if source_name in fluxcat.FluxCatalog:
            exp_flux = fluxcat.FluxCatalog[source_name].predict_flux(freq)
            txt_box.append('Expected Peak = %0.1f Jy' % exp_flux)

    # Measured flux
    meas_flux = param[param_name.index('peak_amplitude')]
    txt_box.append('Measured Peak = %0.1f Jy' % meas_flux)
    txt_box.append('')

    # RMS
    rms = 1.4826*np.median(np.abs(submap_resid - np.median(submap_resid)))

    txt_box.append('RMS = %0.1f Jy' % rms)

    # S/N ratio
    txt_box.append('S/N = %0.1f' % (meas_flux / rms))

    # Open figure
    fig = plt.figure(num=1, figsize=(20, 15), dpi=400)

    # Data
    plt.subplot(3,3,1)

    img = plt.imshow(submap.T, origin='lower', aspect='auto', cmap=cm,
                              extent=extent, vmin=submap.min(), vmax=submap.max())

    plot_centroid()

    plt.title('Data')
    plt.xlabel('RA [deg]')
    plt.ylabel('Dec [deg]')

    show_colorbar()

    plt.xlim(xrng)
    plt.ylim(yrng)

    # Model
    plt.subplot(3,3,2)

    img = plt.imshow(submap_model.T, origin='lower', aspect='auto', cmap=cm,
                              extent=extent, vmin=submap.min(), vmax=submap.max())

    plot_centroid()

    plt.title('Model')
    plt.xlabel('RA [deg]')
    plt.ylabel('Dec [deg]')

    plt.xlim(xrng)
    plt.ylim(yrng)

    cbar = plt.colorbar(img, cmap=cm, pad=0.05)
    cbar.ax.get_yaxis().labelpad = 18
    cbar.ax.set_ylabel(unit)

    # Residuals
    plt.subplot(3,3,3)

    img = plt.imshow(submap_resid.T, origin='lower', aspect='auto', cmap=cm,
                              extent=extent, vmin=submap_resid.min(), vmax=submap_resid.max())

    plot_centroid()

    plt.title('Residuals')
    plt.xlabel('RA [deg]')
    plt.ylabel('Dec [deg]')

    plt.xlim(xrng)
    plt.ylim(yrng)

    cbar = plt.colorbar(img, cmap=cm, pad=0.05)
    cbar.ax.get_yaxis().labelpad = 18
    cbar.ax.set_ylabel(unit)

    # RA Slice
    plt.subplot(3,3,4)
    islc = np.argmin(np.abs(ra - ra0))
    plt.plot(dec, submap_model[islc,:], color='r', linewidth=2.0)
    plt.plot(dec, submap[islc,:], color='b', marker='+')

    plt.title('RA = %0.2f deg' % ra0)
    plt.xlabel('Dec [deg]')
    plt.ylabel(unit)

    plt.xlim(yrng)
    plt.ylim([vmin, vmax])

    plot_centroid_dec()

    # Dec Slice
    plt.subplot(3,3,5)
    islc = np.argmin(np.abs(dec - dec0))
    plt.plot(ra, submap_model[:,islc], color='r', linewidth=2.0)
    plt.plot(ra, submap[:,islc], color='b', marker='+')

    plt.title('Dec = %0.2f deg' % dec0)
    plt.xlabel('RA [deg]')
    plt.ylabel(unit)

    plt.xlim(xrng)
    plt.ylim([vmin, vmax])

    plot_centroid_ra()

    # Text box
    ax = plt.gca()
    txt_str = '\n'.join(txt_box)
    plt.text(1.05, 1.0, txt_str, fontsize=2*font['size'],
                        verticalalignment='top',
                        transform=ax.transAxes)


    # RA Slice Residuals
    plt.subplot(3,3,7)
    islc = np.argmin(np.abs(ra - ra0))
    plt.plot(dec, submap_resid[islc,:], color='b', marker='+')
    plt.hlines(0.0, *yrng, color='r', linewidth=2.0)

    plt.xlabel('Dec [deg]')
    plt.ylabel(unit)

    plt.xlim(yrng)
    plt.ylim([submap_resid.min(), submap_resid.max()])

    plot_centroid_dec()

    plt.subplot(3,3,8)
    islc = np.argmin(np.abs(dec - dec0))
    plt.plot(ra, submap_resid[:,islc], color='b', marker='+')
    plt.hlines(0.0, *xrng, color='r', linewidth=2.0)

    plt.xlabel('RA [deg]')
    plt.ylabel(unit)

    plt.xlim(xrng)
    plt.ylim([submap_resid.min(), submap_resid.max()])

    plot_centroid_ra()

    # Create title
    if title is not None:
        stitle = plt.suptitle(title, y=1.05, fontsize=fontsize+4)
        output_kwargs['bbox_extra_artists'] = [stitle]

    plt.tight_layout()

    return output_kwargs


class PlotRingMapVersusFreq(task.SingleTask):

    pol = config.Property(proptype=str, default='SS')
    beam = config.Property(proptype=int, default=0)

    alias = config.Property(proptype=bool, default=True)
    pt_src = config.Property(proptype=bool, default=True)

    vmin = config.Property(proptype=float, default=None)
    vmax = config.Property(proptype=float, default=None)

    ffmpeg_path = config.Property(proptype=str, default=None)

    def process(self, cmap):

        # Make sure we are distributed over frequency
        cmap.redistribute('freq')

        # Determine movie or pdf output
        do_movie = (cmap.comm is not None) and (cmap.comm.size > 1)

        # Extract map parameters
        ra = cmap.index_map['ra']
        freq = cmap.index_map['freq']['centre']
        bwidth = cmap.index_map['freq']['width']

        units = cmap.map.attrs.get('units', 'Jansky')
        tag = cmap.attrs.get('tag', '')

        nfreq = len(freq)

        # Set plot parameters
        ipol = list(cmap.index_map['pol']).index(self.pol)
        ibeam = list(cmap.index_map['beam']).index(self.beam)

        ra, dec, amap = _get_map(cmap, dataset='map')

        # Create directory to hold results
        output_dir = 'plot'
        if mpiutil.rank0:
            make_directory(output_dir)

        # Define filenames
        suffix = filter(None, [tag, '%s' % self.pol, 'beam%d' % self.beam])
        output_file_base = self.output_root + '_'.join(suffix)
        if output_file_base[-1] == '_':
            output_file_base = output_file_base[:-1]

        if do_movie:
            png_dir = None
            if mpiutil.rank0:
                png_dir = os.path.join(output_dir, 'temp_png_%d' % time.time())
                make_directory(png_dir)
            png_dir = mpiutil.world.bcast(png_dir, root=0)

            file_list = np.array([ os.path.join(png_dir, output_file_base + "_%04d.png") % ff for ff in range(nfreq)[::-1] ])
            file_glob = os.path.join(png_dir, output_file_base + "_*.png")
            movie_file = os.path.join(output_dir, output_file_base + "_vs_freq.mp4")

        else:
            output_file = os.path.join(output_dir, output_file_base + '_vs_freq.pdf')
            out = OutputPdf(output_file)

        # Loop over frequency
        for lfi, fi in cmap.map[:].enumerate(0):

            this_map = amap[lfi, ipol, :, ibeam, :]

            title = ("Frequency %0.2f MHz | Bandwidth %0.2f MHz | Polarization %s | %s " %
                    (freq[fi], bwidth[fi], self.pol, tag))

            alias_line = freq[fi] if self.alias else None

            if do_movie:
                out = file_list[fi]

            plot_ring_map(this_map, ra, plot_dec=False, destripe=False, csd=None,
                                             units=units, plot_title=title, fontsize=18,
                                             alias_line=alias_line, pt_src=self.pt_src, ref_lines=False,
                                             linear=True, vmin=self.vmin, vmax=self.vmax,
                                             cb_shrink=0.5, color_map='inferno',
                                             filename=out)

        # Create FFMPEG movie
        if do_movie:

            cmap.comm.Barrier()

            if mpiutil.rank0:

                try:
                    os.remove(movie_file)
                except OSError:
                    pass

                framerate = 4
                fps = 4

                if self.ffmpeg_path is None:
                    ffmpeg_path = os.path.join(os.path.expanduser('~'), 'ffmpeg', 'ffmpeg')

                command = ("{} -framerate {:f} -pattern_type glob -i '{}' -c:v libx264 -r {:d} -pix_fmt yuv420p " +
                           "-vf 'scale=trunc(iw/2)*2:trunc(ih/2)*2' {}").format(ffmpeg_path, framerate, file_glob, fps, movie_file)

                result = os.system(command)
                if result:
                    print "FFMPEG did not run successfully."

                else:
                    # Delete the png files that were created
                    for filename in file_list:
                        os.remove(filename)

                    os.rmdir(png_dir)

            cmap.comm.Barrier()

        else:

            out.close()

        # Return (unmodified) cmap
        return cmap


def plot_ring_map(input_map, ra, plot_dec=False, vmin=None, vmax=None, destripe=True, csd=None,
                                 units='correlator units', plot_title=None, fontsize=10,
                                 alias_line=None, pt_src=False, ref_lines=True,
                                 linear=False, log=False, vrestricted=False,
                                 source=None, source_window=10.0,
                                 cb_shrink=0.6, color_map='inferno',
                                 fignum=1, filename=None):

    # Define dimensions
    nra = input_map.shape[0]
    ndec = input_map.shape[1]

    if len(ra) != nra:
        InputError("Size of ra must be the same as the size of the second dimension of the input map.")

    sin_el = np.linspace(-1.0, 1.0, ndec)
    lat = ephemeris.CHIMELATITUDE

    if plot_dec:
        dec = np.arcsin(sin_el)*180.0/np.pi + lat
        aspect = 1.0
        ylabel = 'Declination [deg]'
        yfontsize = fontsize
        yticks = np.arange(np.around(dec[0], -1), np.around(dec[-1], -1), 10)
    else:
        dec = sin_el
        aspect = 90.0
        ylabel = r'$\sin{\theta}$'
        yfontsize = fontsize + 4
        yticks = np.arange(-1.0, 1.2, 0.2)

    xlabel = "Right Ascension [deg]"
    xticks = np.arange(0.0, 390.0, 30.0)

    # Set font
    font = {'family' : 'sans-serif',
            'weight' : 'normal',
            'size'   : fontsize}

    plt.rc('font', **font)

    # Loop over declinations and subtract the mean value
    if destripe or log:
        map_plot = input_map.copy()
    else:
        map_plot = input_map

    if destripe:
        if csd is None:
            InputError("Must pass csd keyword.")

        _destripe(map_plot, ra, csd, axis=0)

    if log:
        map_plot = np.abs(map_plot)

    # Set the color map
    cm = matplotlib.cm.__dict__[color_map]
    cm.set_under("k")

    # Set the units
    if linear or log:
        cb_label = "%s" % units
    else:
        map_plot = 10*np.log10(np.abs(map_plot))
        cb_label = "dB %s" % units

    # Set the scale
    if vrestricted:
        vrange = np.percentile(map_plot, [2, 98])
    else:
        vrange = np.percentile(map_plot, [0, 100])

    if vmin is not None:
        vrange[0] = vmin

    if vmax is not None:
        vrange[1] = vmax

    if log:
        pkw = {'norm':LogNorm(vmin=vrange[0], vmax=vrange[1])}
    else:
        pkw = {'vmin':vrange[0], 'vmax':vrange[1]}

    # Plot
    fig = plt.figure(num=fignum, figsize=(20, 15), dpi=400)
    if plot_dec:
        im = plt.pcolormesh(ra, dec, map_plot.T, cmap=cm, **pkw)

    else:
        im = plt.imshow(map_plot.T, origin='lower', aspect='auto', cmap=cm,
                                  extent=(ra[0], ra[-1], dec[0], dec[-1]),
                                  **pkw)

    # Axis
    if source is None:
        # Full sky
        plt.xlim([ra[0], ra[-1]])
        plt.ylim([dec[0], dec[-1]])

    else:
        # Restrict range to point source
        if source not in ephemeris.source_dictionary:
            KeyError("Do not recognize source %s." % source)

        ephm = ephemeris.source_dictionary[source]
        ephm.compute()
        source_ra, source_dec = np.degrees(ephm.a_ra), np.degrees(ephm.a_dec)

        source_ra_rng  = [source_ra - source_window, source_ra + source_window]
        source_dec_rng = [source_dec - source_window, source_dec + source_window]

        if not plot_dec:
            source_dec_rng = [np.sin((dd - lat)*np.pi/180.0) for dd in source_dec_rng]

        plt.xlim(source_ra_rng)
        plt.ylim(source_dec_rng)

    plt.tick_params(axis='both', labelsize=fontsize-2, color='w')

    # Set aspect ratio
    gca = plt.gca()
    gca.set_xticks(xticks)
    gca.set_yticks(yticks)
    gca.set_aspect(aspect)

    plt.xlabel(xlabel, fontsize=fontsize)
    plt.ylabel(ylabel, fontsize=yfontsize)
    if plot_title is not None:
        plt.title(plot_title, fontsize=fontsize)

    # Colorbar
    cbar = plt.colorbar(im, cmap=cm, shrink=cb_shrink, fraction=0.046, pad=0.08)
    im.colorbar.set_label(cb_label, size=fontsize)
    im.colorbar.ax.tick_params(labelsize=fontsize-2)

    # Annotations
    if ref_lines and plot_dec:
        # Overhead
        plt.hlines(49, ra[0], ra[-1], color='w', linewidth=1.2, linestyles='dashed')
        plt.annotate(' overhead', (ra[-1], 49), style='italic', fontsize=fontsize-2, verticalalignment='center')

        # Horizon
        plt.hlines(49-90, ra[0], ra[-1], color='w', linewidth=1.2, linestyles='dashed')
        plt.annotate(' horizon', (ra[-1], 49-90), style='italic', fontsize=fontsize-2)

    # Aliasing critical point
    if alias_line is not None:
        # Where alias_line is the freq in MHz, and 60cm is twice the feed seperation
        alias_delta = 3e4 / alias_line / 60.0
        if plot_dec:
            alias_delta = np.degrees(np.arcsin(alias_delta))
            alias_lb = lat - alias_delta
            alias_ub = lat + alias_delta
        else:
            alias_lb = -alias_delta
            alias_ub =  alias_delta

        if (alias_lb > dec[0]) and (alias_ub < dec[-1]):
            plt.hlines([alias_lb, alias_ub], ra[0], ra[-1], color='w', linewidth=1.2, linestyles='dashed')
            plt.annotate(' alias\n limit', (ra[-1], alias_lb),
                           style='italic', fontsize=fontsize-2, verticalalignment='center')
            plt.annotate(' alias\n limit', (ra[-1], alias_ub),
                            style='italic', fontsize=fontsize-2, verticalalignment='center')

    # Point source annotations
    if pt_src:
        # Create annotation
        for name, ephm in ephemeris.source_dictionary.iteritems():

            ephm.compute()
            src_ra = np.degrees(ephm.a_ra)

            if (src_ra - ra[0]) > (ra[-1] - src_ra):
                offset_x = -20
            else:
                offset_x = 20

            src_dec = np.degrees(ephm.a_dec)

            if src_dec < 50:
                offset_y = -20 if plot_dec else -0.20
            else:
                offset_y = +20 if plot_dec else 0.20

            if not plot_dec:
                src_dec = np.sin((src_dec - lat)*np.pi/180.0)

            ann_pt_src(src_ra, src_dec, name, fontsize=fontsize-4, offset=(offset_x, offset_y))

    # If requested, output figure to file
    if filename is not None:
        if isinstance(filename, OutputPdf):
            filename.save(bbox_inches='tight')
        elif isinstance(filename, basestring):
            plt.savefig(filename, dpi=200, bbox_inches='tight')
        else:
            InputError("Do not recognize filename type.")
        plt.close(fig)


def destripe_ringmap(ringmap):

    csd = ringmap.attrs['csd']
    ra = ringmap.index_map['ra']
    ra_axis = list(ringmap.map.attrs['axis']).index('ra')

    ringmap.map[:] = _destripe(ringmap.map[:], ra, csd, axis=ra_axis)


def _destripe(input_map, ra, csd, axis=-1):

    slc = [slice(None)] * len(input_map.shape)

    flag_quiet = np.ones(len(ra), dtype=np.bool)

    if hasattr(csd, '__iter__'):
        csd_list = csd
    else:
        csd_list = [csd]

    for ii, cc in enumerate(csd_list):
        flag_quiet &= flag_quiet_time(ephemeris.csd_to_unix(cc + ra/360.0))

    slc[axis] = flag_quiet

    input_map -= np.median(input_map[slc], axis=axis, keepdims=True) * (input_map != 0.0)

    return input_map


def flag_quiet_time(time, src_window=1800.0, sun_extension=0.0):

    ntime = len(time)
    delta_time = np.median(np.diff(time))

    flag = np.zeros(ntime, dtype=np.bool)

    # Sun
    rise_times = ephemeris.solar_rising(time[0] - 24.0*3600.0, end_time=time[-1])

    for rise_time in rise_times:

        set_time = ephemeris.solar_setting(rise_time, end_time=time[-1])

        if len(set_time) == 0:
            set_time = time[-1]
        else:
            set_time = set_time[0]

        flag = flag | ((time >= (rise_time - sun_extension)) & (time <= (set_time + sun_extension)))


    # Bright point sources
    for src_name, src_ephem in ephemeris.source_dictionary.iteritems():

        peak_ra = ephemeris.peak_RA(src_ephem, deg=True)

        src_peak_times = get_peak_times(time, ephemeris.transit_RA(time), peak_ra)

        for tt in src_peak_times:

            flag = flag | ((time >= (tt - src_window)) & (time <= (tt + src_window)))


    # Check if there is any remaining data
    if np.all(flag):
        print "Warning:  No data available for calculating the mean."
        return None

    # Switch from flagging bad times to flagging good times
    flag = np.logical_not(flag)

    return flag


def get_sun_peak_ra(src, times):

    obs = ephemeris._get_chime()

    ntimes = np.size(times)
    peak_ra = np.zeros(ntimes, dtype=np.float)

    for tt in range(ntimes):

        utime = ephemeris.ensure_unix(times[tt])
        obs.date = ephemeris.unix_to_ephem_time(utime)
        src.compute(obs)

        peak_ra[tt] = get_peak_ra(src)

    return peak_ra


def get_peak_times(time, ra, peak_ra):

    """ Interpolates time(ra) to determine the
        times coresponding to peak_ra.
    """

    ntime = np.size(time)

    if np.size(ra) != ntime:
        ValueError("ra and time must be the same size.")

    index0 = np.arange(ntime)
    flag = (peak_ra >= ra) & (peak_ra < np.roll(ra, -1))

    if not np.any(flag):
        ValueError("could not find any times corresponding to peak_ra.")

    index0 = index0[flag]
    index1 = index0 + 1

    if np.any(index1 == ntime):
        index1[index1 == ntime] = ntime - 1

    slope = (time[index1] - time[index0]) / (ra[index1] - ra[index0])

    peak_time = time[index0] + slope*(peak_ra - ra[index0])

    return peak_time

def ann_pt_src(ra, dec, label, fontsize=8, color='w', offset=(10,-10)):

    plt.annotate(label, (ra, dec), xytext=(ra + offset[0], dec + offset[1]), textcoords='data',
                 arrowprops=dict(arrowstyle="->", color=color), color=color, fontsize=fontsize)


def _list_or_glob(files):
    # Take in a list of lists/glob patterns of filenames
    import glob

    if isinstance(files, str):
        files = sorted(glob.glob(files))
    elif isinstance(files, list):
        pass
    else:
        raise RuntimeError('Must be list or glob pattern.')

    return files


class PlotCorrInputs(task.SingleTask):

    monitor = config.Property(proptype=bool, default=True)

    test = config.Property(proptype=str, default=None)

    threshold = config.Property(proptype=float, default=0.7)

    files = config.Property(proptype=_list_or_glob, default='')

    output_file = config.Property(proptype=str, default=None)


    def setup(self):

        if self.output_file is None:
            ValueError("Must specify output filename.")

        self.output_file = os.path.expandvars(os.path.expanduser(self.output_file))

        self.output = OutputPdf(self.output_file)


    def process(self):

        from ch_util import rfi
        import glob

        files = self.files

        nfiles = len(files)

        # Only plot with head node
        if mpiutil.rank0:

            # Load results
            for ff, filename in enumerate(files):

                if self.monitor:
                    corr_input_mon = containers.CorrInputMonitor.from_file(filename, distributed=False)

                else:
                    corr_input_mon = containers.CorrInputTest.from_file(filename, distributed=False)

                    i_power = list(corr_input_mon.test).index('is_chime')

                    if self.test is not None:
                        i_test = list(corr_input_mon.test).index(test)
                    else:
                        i_test = None

                # If this is the first file, then creat arrays to hold results for all files
                if ff == 0:

                    freq = corr_input_mon.freq['centre'][:]
                    input_map = corr_input_mon.input[:]

                    nfreq = len(freq)
                    ninput = len(input_map)

                    input_mask = np.ones((nfiles, ninput), dtype=np.bool)
                    input_powered = np.ones((nfiles, ninput), dtype=np.bool)

                    if self.monitor:
                        freq_mask = np.ones((nfiles, nfreq), dtype=np.bool)
                        freq_powered = np.ones((nfiles, nfreq), dtype=np.bool)

                    csd = np.zeros(nfiles, dtype=np.int)

                # Save results to array
                if self.monitor:
                    input_mask[ff, :] = corr_input_mon.input_mask[:]
                    input_powered[ff, :] = corr_input_mon.input_powered[:]
                    freq_mask[ff, :] = corr_input_mon.freq_mask[:]
                    freq_powered[ff, :] = corr_input_mon.freq_powered[:]

                else:
                    if i_test is None:
                        input_mask[ff, :] = corr_input_mon.input_mask[:]
                    else:
                        input_mask[ff, :] = (np.sum(corr_input_mon.passed_test[:, :, i_test], axis=0) >
                                                    self.threshold*len(self.freq))

                    input_powered[ff, :] = (np.sum(corr_input_mon.passed_test[:, :, i_power], axis=0) >
                                                    self.threshold*len(self.freq))

                csd[ff] = corr_input_mon.attrs['csd']

            # Sort based on CSD
            isort = np.argsort(csd)
            input_mask = input_mask[isort, :]
            input_powered = input_powered[isort, :]

            if self.monitor:
                freq_mask = freq_mask[isort, :]
                freq_powered = freq_powered[isort, :]
            csd = csd[isort]

            # Take the AND of the mask over all days and stack on the end of the array
            input_mask = np.vstack((input_mask, np.all(input_mask, axis=0)[np.newaxis, :]))
            input_powered = np.vstack((input_powered, np.all(input_powered, axis=0)[np.newaxis, :]))

            # Calculate percentages good and powered on
            percentage_good_input = (np.sum(input_mask, axis=-1) / float(ninput))*100.0
            percentage_powered_input = (np.sum(input_powered, axis=-1) / float(ninput))*100.0

            # Create labels
            csd_label_input = ['%d' % cc for cc in csd]
            csd_label_input.append('ALL')

            percentage_text_input = [r'  %d%% | %d%%' % (percentage_good_input[ii],
                                                         percentage_powered_input[ii])
                                     for ii in range(nfiles+1)]

            # Create pdf file


            # Define some variables for plotting
            lbls = ['OFF - BAD', 'OFF - GOOD', 'ON - BAD', 'ON - GOOD']
            cmap = matplotlib.colors.ListedColormap(['black', 'sage', 'crimson', 'forestgreen'])
            nstatus = len(lbls)

            bnds = range(nstatus+1)
            norm = matplotlib.colors.BoundaryNorm(bnds, cmap.N)

            hlines = np.arange(nfiles) - 0.5
            dhlines_input = nfiles - 0.5

            input_step = 16

            fnt = 16.0

            # Plot csd vs input mask
            # -----------------------
            fig = plt.figure(num=1, figsize=(20, 10), dpi=400)

            # Convert input_powered and input_mask into single status indictator
            # that ranges from 0 - 3
            status = input_powered.astype(np.int)*2 + input_mask.astype(np.int)

            # Plot the status
            img = plt.imshow(status, aspect='auto', interpolation='nearest', origin='lower',
                                     cmap=cmap, norm=norm)

            # Create colorbar
            cbar = plt.colorbar(img, cmap=cmap, norm=norm, boundaries=bnds, pad=0.125)
            cbar.ax.get_yaxis().set_ticks([])
            for jj, lbl in enumerate(lbls):
                cbar.ax.text(0.5, (2*jj + 1) / (2.0*nstatus), lbl,
                             ha='center', va='center', color='white', rotation=270)
            cbar.ax.get_yaxis().labelpad = 18
            cbar.ax.set_ylabel('Channel Monitor Status', rotation=270)

            # Seperate CSD with horizontal lines
            plt.hlines(hlines, *plt.xlim(), color='white', linestyle=':', linewidth=1.0)
            plt.hlines(dhlines_input, *plt.xlim(), color='white', linestyle='-', linewidth=1.5)

            # Put the major ticks at the middle of each cell
            input_ticks = np.arange(0, status.shape[1], input_step)

            ax = plt.gca()
            ax.set_xticks(input_ticks, minor=False)
            ax.set_yticks(np.arange(status.shape[0]), minor=False)

            # Set tick labels
            ax.set_xticklabels(input_ticks, minor=False)
            ax.set_yticklabels(csd_label_input, minor=False)

            # Set percentage text
            for ii, tt in enumerate(percentage_text_input):
                plt.text(ninput+1, ii, tt, fontsize=fnt)

            plt.text(ninput, ii+1, 'GOOD | PWD')

            # Set labels
            plt.xlabel('Correlator Input Channel')
            plt.ylabel('CSD')
            plt.rcParams.update({'font.size': fnt})

            self.output.save()

            plt.close()

            # Plot csd vs freq mask
            # -----------------------

            # Frequency mask plot
            if self.monitor:
                # Put in order of ascending frequency
                ifsort = np.argsort(freq)
                freq_mask = freq_mask[:, ifsort]
                freq_powered = freq_powered[:, ifsort]
                freq = freq[ifsort]

                # Take the AND of the mask over all days and stack on the end of the array
                freq_mask = np.vstack((freq_mask, np.all(freq_mask, axis=0)[np.newaxis, :]))
                freq_powered = np.vstack((freq_powered, np.all(freq_powered, axis=0)[np.newaxis, :]))

                rfi_mask = ~rfi.frequency_mask(freq)
                freq_mask = np.vstack((freq_mask, rfi_mask[np.newaxis, :]))
                freq_powered = np.vstack((freq_powered, np.ones((1, nfreq), dtype=np.bool)))

                # Calculate percentages good and powered on
                percentage_good_freq = (np.sum(freq_mask, axis=-1) / float(nfreq))*100.0
                percentage_powered_freq = (np.sum(freq_powered, axis=-1) / float(nfreq))*100.0

                # Calculate percentages good and powered on
                percentage_good_freq = (np.sum(freq_mask, axis=-1) / float(nfreq))*100.0
                percentage_powered_freq = (np.sum(freq_powered, axis=-1) / float(nfreq))*100.0

                # Create labels
                csd_label_freq = csd_label_input
                csd_label_freq.append('RFI')

                percentage_text_freq = [r'  %d%% | %d%%' % (percentage_good_freq[ii],
                                                            percentage_powered_freq[ii])
                                         for ii in range(nfiles+2)]

                # Define some variables for plotting
                dhlines_freq  = nfiles + np.array([-0.5, 0.5])
                freq_step = 50.0

                # Open figure
                fig = plt.figure(num=1, figsize=(20, 10), dpi=400)

                # Convert input_powered and input_mask into single status indictator
                # that ranges from 0 - 3
                status = freq_powered.astype(np.int)*2 + freq_mask.astype(np.int)

                # Set axis bounds
                freq_a = np.min(freq) - np.median(np.diff(freq))
                freq_b = np.max(freq)

                csd_a = -0.5
                csd_b = nfiles + 1.5

                # Plot the status
                img = plt.imshow(status, aspect='auto', interpolation='nearest', origin='lower',
                                         cmap=cmap, norm=norm, extent=(freq_a, freq_b, csd_a, csd_b))

                # Create colorbar
                cbar = plt.colorbar(img, cmap=cmap, norm=norm, boundaries=bnds, pad=0.125)
                cbar.ax.get_yaxis().set_ticks([])
                for jj, lbl in enumerate(lbls):
                    cbar.ax.text(0.5, (2*jj + 1) / (2.0*nstatus), lbl,
                                 ha='center', va='center', color='white', rotation=270)
                cbar.ax.get_yaxis().labelpad = 18
                cbar.ax.set_ylabel('Channel Monitor Status', rotation=270)

                # Seperate CSD with horizontal lines
                plt.hlines(hlines, *plt.xlim(), color='white', linestyle=':', linewidth=1.0)
                plt.hlines(dhlines_freq, *plt.xlim(), color='white', linestyle='-', linewidth=1.5)

                # Put the major ticks at the middle of each cell
                freq_ticks = np.arange(freq_a, freq_b, freq_step)

                ax = plt.gca()
                ax.set_xticks(freq_ticks, minor=False)
                ax.set_yticks(np.arange(status.shape[0]), minor=False)

                # Set tick labels
                ax.set_xticklabels(freq_ticks, minor=False)
                ax.set_yticklabels(csd_label_freq, minor=False)

                # Set percentage text
                for ii, tt in enumerate(percentage_text_freq):
                    plt.text(freq_b, ii, tt, fontsize=fnt)

                plt.text(freq_b, ii+1, 'GOOD | PWD')

                # Set labels
                plt.xlabel('Frequency [MHz]')
                plt.ylabel('CSD')
                plt.rcParams.update({'font.size': fnt})

                self.output.save()

                plt.close()

        # Stop iteration
        raise pipeline.PipelineStopIteration

    def finish(self):

        self.output.close()


def make_directory(path):
    try:
        os.makedirs(path)
    except OSError:
        if not os.path.isdir(path):
            raise

class OutputPdf():

    def __init__(self, pdf_file, dpi=100):

        self.pdf_file = pdf_file

        self.dpi = dpi

        if isinstance(pdf_file, basestring):
            self.pdf_pages = PdfPages(pdf_file)
        else:
            self.pdf_pages = None

    def __nonzero__(self):
        return self.pdf_pages is not None

    def save(self, **kwargs):

        if self:
            self.pdf_pages.savefig(dpi=self.dpi, **kwargs)

    def close(self):

        if self:
            self.pdf_pages.close()