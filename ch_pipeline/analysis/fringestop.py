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

    #source = config.Property(proptype=string, default='CasA')
    #update_weight = config.Property(proptype=bool, default=False)
    #smoothing_length = config.Property(proptype=float, default=None)

    def process(self, tstream, source):

        tstream.redistribute('freq')

        utime=tstream.index_map['time']['ctime']
        freq=tstream.index_map['freq']['centre']
        dtime=ephemeris.unix_to_datetime(utime[0])
        corr_inputs = tools.get_correlator_inputs(dtime), correlator='chime')
        feeds=[corr_inputs[data.index_map['input'][i][0]] for i in range(len(data.index_map['input']))]
        prod_map=tstream.index_map['prod']

        tstream.vis=tools.fringestop_time(tstream.vis,times=utime, freq=freq,feeds=feeds,src=source,prod_map=prod_map)

        return tstream



