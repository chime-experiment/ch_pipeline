import os
import sys
import numpy as np

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from matplotlib.backends.backend_pdf import PdfPages

import healpy
import h5py

from ch_util import ephemeris


def output_ring_map_versus_freq(input_file, suffix=None, pol='I', beam=0, 
                                            freq_start=0, freq_stop=None, freq_skip=1, 
                                            alias=False, do_movie=False, ffmpeg_path=None, **kwargs):
        
    # Extract relevant dimensions
    with h5py.File(input_file, 'r') as h5file:
        
        # Extract frequencies
        freq = h5file['index_map']['freq'][:]
        
        # Extract right ascension
        ra = h5file['index_map']['ra'][:]
        
        # Extract polarization and beam indices
        ipol = np.where(pol == h5file['index_map']['pol'][:])
        if not ipol:
            InputError("File does not contain requested polarization:  %s" % pol)
        else:
            ipol = ipol[0][0]
            
        ibeam = np.where(beam == h5file['index_map']['beam'][:])
        if not ibeam:
            InputError("File does not contain requested beam:  %d" % beam)
        else:
            ibeam = ibeam[0][0]
        
        # Determine the csd
        tag = h5file.attrs['tag']
        csd = int(tag.split('_')[1])

    # Determine what frequencies to plot
    if freq_stop is None:
        freq_stop = len(freq)
        
    ifreq = range(freq_start,freq_stop,freq_skip)
    nfreq = len(ifreq)
    
    # Create directory
    output_dir = os.path.join(os.path.dirname(input_file), 'plot')
    make_directory(output_dir)
    
    output_file_base = os.path.splitext(os.path.basename(input_file))[0]
    
    # Create the output file
    if do_movie:
        png_dir = os.path.join(output_dir, 'temp_png')
        make_directory(png_dir)
        
        file_list = np.array([os.path.join(png_dir, output_file_base + "_%03d.png") % ff for ff in range(nfreq)])
        file_glob = os.path.join(png_dir, output_file_base + "_*.png")
        
        movie_file = os.path.join(output_dir, output_file_base + ".mp4")
        
    else:
        output_file = os.path.join(output_dir, output_file_base + '.pdf')
        out = OutputPdf(output_file)
    
    # Loop over frequencies
    for ii, ff in enumerate(ifreq):
                
        # Extract the map for this frequency from the h5 file
        with h5py.File(input_file, 'r') as h5file:
            input_map = h5file['map'][ff,ipol,:,ibeam,:]
            
        # Determine arguments
        if do_movie:
            out = file_list[ii]
            
        title = "Frequency %0.2f MHz, Bandwidth %0.2f MHz" % (freq[ff][0], freq[ff][1])
        
        if alias:
            alias_line = freq[ff][0]
        else:
            alias_line = None
            
        # Plot ring map
        plot_ring_map(input_map, ra, csd=csd, filename=out, plot_title=title, alias_line=alias_line, **kwargs)
        
        
    # If requested, use ffmpeg to create the movie
    if do_movie:
        
        try:
            os.remove(movie_file)
        except OSError:
            pass
            
        framerate = 1
        fps = 30
        
        if ffmpeg_path is None:
            ffmpeg_path = os.path.join(os.path.expanduser('~'), 'ffmpeg', 'ffmpeg')
        
        command = ("{} -framerate {:f} -pattern_type glob -i '{}' -c:v libx264 -r {:d} -pix_fmt yuv420p " +
                   "-vf 'scale=trunc(iw/2)*2:trunc(ih/2)*2' {}").format(ffmpeg_path, framerate, file_glob, fps, movie_file)
                   
        result = os.system(command)
        if result:
            print "FFMPEG did not run successfully."
            
        # Delete the png files that were created
        for filename in file_list:
            os.remove(filename)
            
        os.rmdir(png_dir)
        
    else:
        
        out.close()
                   


def plot_ring_map(input_map, ra, plot_dec=False, destripe=True, csd=None, vrange=None,
                                      units='correlator units', plot_title=None, fontsize=10,  
                                      alias_line=None, pt_src=False, ref_lines=True, 
                                      linear=False, log=False, vrestricted=False,
                                      cb_shrink=0.6, color_map='inferno', 
                                      fignum=1, filename=None):
    
    # Define dimensions
    nra = input_map.shape[0]
    ndec = input_map.shape[1]
    
    if len(ra) != nra:
        InputError("Size of ra must be the same as the size of the first dimension of the input map.")
        
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
    
    # Loop over declinations and subtract the mean value
    map_plot = input_map.copy()
    if destripe:
        if csd is None:
            InputError("Must pass csd keyword.")
            
        time = recover_time(ra, csd)
        
        flag_quiet = flag_quiet_time(time) 
        
        map_plot -= np.median(input_map[flag_quiet,...], axis=0)[np.newaxis,...]
        
    # Get axis in the proper order for imshow
    map_plot = np.transpose(map_plot)
        
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
    if vrange is None:
        if vrestricted:
            vrange = np.percentile(map_plot, [2, 98])
        else:
            vrange = np.percentile(map_plot, [0, 100])
    
    # Plot
    fig = plt.figure(num=fignum, figsize=(20, 20), dpi=400)
    if plot_dec:
        if log:
            im = plt.pcolormesh(ra, dec, np.abs(map_plot), cmap=cm,
                                        norm=LogNorm(vmin=vrange[0], vmax=vrange[1]))
        else:
            im = plt.pcolormesh(ra, dec, map_plot, cmap=cm,
                                        vmin=vrange[0], vmax=vrange[1])
                                        
    else:
        map_plot = np.flipud(map_plot)
        if log:
            im = plt.imshow(np.abs(map_plot), aspect='equal', interpolation='nearest', cmap=cm,
                                        extent=(ra[0], ra[-1], dec[0], dec[-1]),
                                        norm=LogNorm(vmin=vrange[0], vmax=vrange[1]))
        else:
            im = plt.imshow(map_plot, aspect='equal', interpolation='nearest', cmap=cm,
                                        extent=(ra[0], ra[-1], dec[0], dec[-1]),
                                        vmin=vrange[0], vmax=vrange[1])
                    
    # Colorbar
    plt.colorbar(im, shrink=cb_shrink, pad=0.10)
    im.colorbar.set_label(cb_label, size=fontsize)
    im.colorbar.ax.tick_params(labelsize=fontsize-2)
    
    # Axis
    plt.xlim([ra[0], ra[-1]])
    plt.ylim([dec[0], dec[-1]])
    
    plt.tick_params(axis='both', labelsize=fontsize-2, color='w')
    
    # Set aspect ratio
    plt.gca().set_xticks(xticks)
    plt.gca().set_yticks(yticks)
    plt.gca().set_aspect(aspect)
    
    plt.xlabel(xlabel, fontsize=fontsize)
    plt.ylabel(ylabel, fontsize=yfontsize)
    if plot_title is not None:
        plt.title(plot_title, fontsize=fontsize)

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
        plt.hlines([alias_lb, alias_ub], ra[0], ra[-1], color='w', linewidth=1.2, linestyles='dashed')
        plt.annotate(' alias\n limit', (ra[-1], alias_lb), 
                       style='italic', fontsize=fontsize-2, verticalalignment='center')
        plt.annotate(' alias\n limit', (ra[-1], alias_ub), 
                        style='italic', fontsize=fontsize-2, verticalalignment='center')
        
    if pt_src:
        # Get ephemeris objects for the four bright point sources
        source_dict = {'CasA': ephemeris.CasA,
                       'CygA': ephemeris.CygA,
                       'TauA': ephemeris.TauA,
                       'VirA': ephemeris.VirA}
                        
        # Create annotation
        for name, ephm in source_dict.iteritems():

            ephm.compute()
            src_ra = np.degrees(ephm.a_ra)
            
            if (src_ra - ra[0]) > (ra[-1] - src_ra):
                offset_x = -20
            else:
                offset_x = 20
                
            src_dec = np.degrees(ephm.a_dec)
            
            if src_dec < 50:
                offset_y = -20
            else:
                offset_y = +20
                
            if not plot_dec:
                src_dec = np.sin((src_dec - lat)*np.pi/180.0)
            
            ann_pt_src(src_ra, src_dec, name, fontsize=fontsize-4, offset=(offset_x, offset_y))
            
    # If requested, output figure to file
    if filename is not None:
        if isinstance(filename, OutputPdf):
            filename.save(bbox_inches='tight')
        elif isinstance(filename, basestring):
            plt.savefig(filename, dpi=400, bbox_inches='tight')
        else:
            InputError("Do not recognize filename type.")
        plt.close(fig)


def flag_quiet_time(time, src_window=1800.0, sun_extension=0.0):
    
    _source_dict = {'CasA': ephemeris.CasA,
                    'CygA': ephemeris.CygA,
                    'TauA': ephemeris.TauA,
                    'VirA': ephemeris.VirA}
    
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
    for src_name, src_ephem in _source_dict.iteritems():
        
        peak_ra = get_peak_ra(src_ephem)

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
    
def get_peak_ra(src):
    
    """ Calculates the RA where a source is expected to peak in the beam.
        Note that this is not the same as the RA where the source is at
        transit, since the pathfinder is rotated with respect to north.
    
        src is an ephem.FixedBody
    """
        
    _PF_ROT = np.radians(1.986)  # Rotation angle of pathfinder
    _PF_LAT = np.radians(49.0)   # Latitude of pathfinder

    # Estimate the RA at which the transiting source peaks
    peak_ra = src._ra + np.tan(_PF_ROT) * (src._dec - _PF_LAT) / np.cos(_PF_LAT)
    
    return np.degrees(peak_ra)
    
    
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
    
    
def recover_time(ra, csd):
    
    time = ephemeris.CSD_ZERO + (24.0 * 3600.0 * ephemeris.SIDEREAL_S)*(csd + ra / 360.0)
    
    return time
    
    
def ann_pt_src(ra, dec, label, fontsize=8, color='w', offset=(10,-10)):
    
    plt.annotate(label, (ra, dec), xytext=offset, textcoords='offset points',\
             arrowprops=dict(arrowstyle="->", color=color), color=color, fontsize=fontsize)
            
def make_directory(path):
    try: 
        os.makedirs(path)
    except OSError:
        if not os.path.isdir(path):
            raise 
             
class OutputPdf():
        
    def __init__(self, pdf_file, dpi=200):
        
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