"""to do the fringestop of the holography data
"""

from ch_util import ephemeris
from ch_util import tools
from draco.core import containers,task
from caput import config,mpiutil
from datetime import datetime



class FringeStop(task.SingleTask):
    """apply the fringestop of the holography data

    Parameters
    ----------
    source : string
        the source of the holography measurement
    source_file : string
        the file under ch_util/catalogs which contain the list of sources
    overwrite : bool
        whether overwrite the original timestream data with the fringestopped timestream data
    """

    source = config.Property(proptype=str, default='CAS_A')
    source_file = config.Property(proptype=str, default='primary_calibrators_perley2016.json')
    overwrite = config.Property(proptype=bool, default=False)

    def process(self, tstream):
        """Apply the fringestop to holography data

        Parameters
        ----------
        tstream : andata.CorrData
            timestream data to be fringestoped
        Returns
        ----------
        tstream/tstream_fs : andata.CorrData
            returns the same timestream object but fringestopped
        """

        tstream.redistribute('freq')

        start_freq = tstream.vis.local_offset[0]
        nfreq = tstream.vis.local_shape[0]
        end_freq = start_freq + nfreq
        print end_freq
        print start_freq
        print nfreq
        freq = tstream.freq[start_freq:end_freq]
        prod_map = tstream.index_map['prod']
        src = ephemeris.get_source_dictionary(self.source_file)[self.source] 
        dtime = ephemeris.unix_to_datetime(tstream.time[0])
        if mpiutil.rank==0:
            corr_inputs = tools.get_correlator_inputs(dtime, correlator='chime')
        else:
            corr_inputs = None 
        self.corr_inputs = mpiutil.world.bcast(corr_inputs,root=0)
        feeds = [self.corr_inputs[tstream.input[i][0]] for i in range(len(tstream.input))]
        # todo: change the feeds calling from the tools.reorder_correlator_inputs after Python3 fix 

        fs_vis=tools.fringestop_time(tstream.vis, times=tstream.time, freq=freq, feeds=feeds, src=src, prod_map=prod_map)

        if self.overwrite:
            tstream.vis[:] = fs_vis
            return tstream
        else:
            tstream_fs = containers.empty_like(tstream)
            tstream_fs.vis[:] = fs_vis
            return tstream_fs



