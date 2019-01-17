"""to do the fringestop of the holography data
"""

from ch_util import andata
from ch_util import data_index
from ch_util import ephemeris
from ch_util import tools

from datetime import datetime



class FringeStop(task.SingleTask):
    """apply the fringestop of the holography data
    """

    source = config.Property(proptype=string, default='CasA')

    def process(self, tstream):
        """Apply the fringestop to holography data

        Parameters
        ----------
        tstream: andata.CorrData
            timestream data to be fringestoped
        Returns
        ----------
        tstream: andata.CorrData
            returns the same timestream object but fringestopped
        """

        tstream.redistribute('freq')
        
        freq = tstream.freq['centre']
        prod_map = tstream.index_map['prod']
        src = ephemeris.get_source_dictionary()[source] 
        dtime = ephemeris.unix_to_datetime(utime[0])
        corr_inputs = tools.get_correlator_inputs(dtime), correlator='chime')
        feeds = [corr_inputs[tstream.input[i][0]] for i in range(len(tstream.input))]

        tstream.vis=tools.fringestop_time(tstream.vis,times=tstream.time, freq=freq,feeds=feeds,src=source,prod_map=prod_map)

        return tstream



