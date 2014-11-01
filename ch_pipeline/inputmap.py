"""
========================================
Input Map (:mod:`~ch_pipeline.inputmap`)
========================================

.. currentmodule:: ch_pipeline.inputmap

Query the layout database to generate a mapping from the input channels in a
file to what is actually connected to them. It returns a collection of objects
representing the inputs.


Classes
=======

.. autosummary::
    :toctree: generated/

    CorrInput
    Antenna
    Blank
    RFIAntenna
    HolographyAntenna
    CHIMEAntenna
    NoiseSource

Routines
========

.. autosummary::
    :toctree: generated/

    get_channels

"""

import datetime

from ch_util import layout, connectdb

# Fetch a handle for the layout database

if connectdb._connect_this_rank():
    db = layout.db.from_connection(connectdb.current_connector.get_connection())


class CorrInput(object):
    """Describes a correlator input.

    Meant to be subclassed by actual types of inputs.

    Attributes
    ----------
    input : str
        Unique serial number of input.
    corr : str
        Unique serial numbe of correlator. Set to `None` if no correlator is
        connected.
    """
    def __init__(self, **input_dict):
        import inspect

        for basecls in inspect.getmro(type(self))[::-1]:
            for k in basecls.__dict__.keys():
                if k[0] != '_':
                    self.__dict__[k] = input_dict[k] if k in input_dict else None

    def __repr__(self):

        kv = ['%s=%s' % (k, v) for k, v in self.__dict__.items() if k[0] != '_']

        return "%s(%s)" % (self.__class__.__name__, ', '.join(kv))

    input_sn = None
    corr = None


class Blank(CorrInput):
    """Unconnected input.
    """
    pass


class Antenna(CorrInput):
    """An antenna input.

    Attributes
    ----------
    reflector : str
        The name of the reflector the antenna is on.
    antenna : str
        Serial number of the antenna.
    """
    reflector = None
    antenna = None


class RFIAntenna(Antenna):
    """RFI monitoring antenna
    """
    pass


class NoiseSource(CorrInput):
    """Broad band noise calibration source.
    """
    pass


class CHIMEAntenna(Antenna):
    """CHIME antenna mounted on a cylinder.

    Attributes
    ----------
    cyl : int
        Index of cylinder we are on.
    pos : float
        Position from the north end in cm.
    pol : str
        Orientation of the polarisation.
    """
    cyl = None
    pos = None
    pol = None


class HolographyAntenna(Antenna):
    """Antenna used for holography.

    Attributes
    ----------
    pol : str
        Orientation of the polarisation.
    """

    pol = None


def _get_channel_props(corr_input, lay):
    """Fetch all the required properties of an ADC channel.

    Parameters
    ----------
    corr_input : chdb.comp
        ADC channel or correlator input.
    lay : layout.layout
        Layout instance to search from.

    Returns
    -------
    channel : CorrInput
        An instance of `CorrInput` containing the channel properties.
    """

    block = ["correlator card slot", "ADC board", "reflector"]

    if corr_input is None:
        raise Exception("ADC channel not valid.")

    # Get the correlator
    corr = lay.closest_of_type(corr_input, "correlator", ctype_exclude='SMA coax')
    corr_sn = corr.sn if corr is not None else None

    # Get the antenna
    ant = lay.closest_of_type(corr_input, "antenna", ctype_exclude=block)

    # If the antenna does not exist, it might be the RFI antenna, the noise source, or empty
    if ant is None:

        # Check to see if this is an RFI antenna
        ant = lay.closest_of_type(corr_input, "RFI antenna", ctype_exclude=block)
        if ant is not None:
            rfl = lay.closest_of_type(ant, "RFI antenna", ctype_exclude=["correlator card slot", "ADC board"])
            return RFIAntenna(input_sn=corr_input.sn, corr=corr_sn, reflector=rfl.sn, antenna=ant.sn)

        # Check to see if it is a noise source
        noise = lay.closest_of_type(corr_input, "noise source", ctype_exclude=block)
        if noise is not None:
            return NoiseSource(input_sn=corr_input.sn, corr=corr_sn)

        # If we get to here, it's probably a blank input
        return Blank(input_sn=corr_input.sn, corr=corr_sn)

    # Get the reflector and pol
    rfl = lay.closest_of_type(ant, "reflector", ctype_exclude=["correlator card slot", "ADC board"])
    pol = lay.closest_of_type(corr_input, "polarisation", ctype_exclude=block)

    # If we are here, we must be a CHIME antenna, or a 26m one
    slt = lay.closest_of_type(ant, "cassette slot", ctype_exclude=block)
    cas = lay.closest_of_type(slt, "cassette", ctype_exclude=block) if slt is not None else None

    try:
        keydict = {'H': "hpol_orient",
                   'V': "vpol_orient",
                   '1': "pol1_orient",
                   '2': "pol2_orient"}

        pkey = keydict[pol.sn[-1]]
        pdir = lay.get_prop(ant, pkey)

    except:
        pdir = None

    if slt is None:
        # Must be holography antenna

        return HolographyAntenna(input_sn=corr_input.sn, corr=corr_sn,
                                 reflector=rfl.sn, pol=pdir, antenna=ant.sn)

    ## If we are still here, we are a CHIME feed

    # Try and figure out the NS position
    try:
        d1 = float(lay.get_prop(cas, "dist_to_n_end"))
        d2 = float(lay.get_prop(slt, "dist_to_edge"))
        orient = lay.get_prop(cas, "slot_zero_pos")

        pos_y = d1 + d2 if orient == 'N' else d1 - d2

    except:
        pos_y = None

    # # Try and get any roll for the feed
    # try:
    #     rollval = lay.get_prop(cas, 'roll')

    #     if isinstance(rollval, basestring):
    #         rollval = rollval.split('~')[-1]  # Attempt to parse rollval
    #     roll = float(rollval)

    # except:
    #     roll = 0.0
    pos_dict = { 'W_cylinder': 0, 'E_cylinder': 1}
    cyl = pos_dict[rfl.sn]

    return CHIMEAntenna(input_sn=corr_input.sn, corr=corr_sn, reflector=rfl.sn,
                        cyl=cyl, pos=pos_y, pol=pdir, antenna=ant.sn)


def get_channels(lay, correlator=None):
    """Get the information for all channels in a layout.

    Parameters
    ----------
    lay : layout.layout or int or datetime
        Layout object or number.
    correlator : str, optional
        Fetch only for specified correlator (use serial number). If `None`
        return for all correlators.

    Returns
    -------
    channels : list
        List of :class:`CorrInput` instances. Returns `None` for MPI ranks
        other than zero.
    """

    chanlist = None

    if connectdb._connect_this_rank():
        # Fetch layout if we received a layout num
        if isinstance(lay, int) or isinstance(lay, datetime.datetime):
            lay = db.get_layout(lay)

        if not isinstance(lay, layout.layout):
            raise Exception("Could not find layout.")

        channels = lay.comp(ctype="ADC channel") + lay.comp(ctype="correlator input")
        channels = sorted(channels, key=lambda adc: adc.sn)

        # Fetch channel properties and filter for only channels connected to the
        # given correlator
        chanlist = [ _get_channel_props(channel, lay) for channel in channels ]
        chanlist = [ ch for ch in chanlist if (correlator is None) or (ch.corr == correlator)]

    return chanlist
