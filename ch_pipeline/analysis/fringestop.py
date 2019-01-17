"""Miscellaneous pipeline tasks with no where better to go.

Tasks should be proactively moved out of """

import numpy as np

from caput import config, mpiutil
from ch_util import andata
from ch_util import data_index
from ch_util import ephemeris
from ch_util import ni_utils
from ch_util import layout
from ch_util import tools

from datetime import datetime
from datetime import timedelta



class FringeStop(task.SingleTask):
    """Apply a set of gains to a timestream or sidereal stack.

    Attributes
    ----------
    inverse : bool, optional
        Apply the gains directly, or their inverse.
    update_weight : bool, optional
        Scale the weight array with the updated gains.
    smoothing_length : float, optional
        Smooth the gain timestream across the given number of seconds.
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



