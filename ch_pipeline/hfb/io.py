"""HFB tasks for reading and writing files
"""

import gc
import numpy as np

from caput import pipeline
from caput import config

from ch_util import ephemeris

from draco.core import io

from beam_model.formed import FFTFormedActualBeamModel

from .containers import HFBReader


class LoadHFBDataFiles(io.BaseLoadFiles):
    """Load CHIME HFB data from a file list passed into the setup routine.

    Attributes
    ----------
    source_dec : float
        Declination of source in degrees.
    freq_physical : list
        List of physical frequencies (in MHz) to read. The first frequency
        in this list is also used in evaluating beam positions (for selecting
        the beams closest to a transiting source).

    Selections
    ----------
    Selections in frequency and beams can be done in two ways:
    1. By passing a `source_dec` attribute (for the beam selection) and/or
       a `freq_physical` attribute (for the frequency selection).
    2. By manually passing indices in the `selections` attribute
       (see documentation in :class:`draco.core.io.BaseLoadFiles`).
    Method 1 takes precedence over method 2. If no relevant attributes are
    passed, all frequencies/beams are read.
    """

    files = None

    _file_ptr = 0

    source_dec = config.Property(proptype=float, default=None)
    freq_physical = config.Property(proptype=list, default=[])

    def setup(self, files):
        """Set the list of files to load; set up frequency and beam selection.

        Parameters
        ----------
        files : list
        """
        if not isinstance(files, (list, tuple)):
            raise RuntimeError("Argument must be list of files.")

        self.files = files

        # Resolve any selections provided through the `selections` attribute
        self._sel = self._resolve_sel()

        # Set up frequency selection.
        if self.freq_physical:
            basefreq = np.linspace(800.0, 400.0, 1024, endpoint=False)
            self.freq_sel = sorted(
                set([np.argmin(np.abs(basefreq - freq)) for freq in self.freq_physical])
            )
        elif "freq_sel" in self._sel:
            self.freq_sel = self._sel["freq_sel"]
        else:
            self.freq_sel = slice(None)

        # Set up beam selection
        if self.source_dec:
            beam_index_ns = self._find_beam()
            self.beam_sel = slice(beam_index_ns, 1024, 256)
        elif "beam_sel" in self._sel:
            self.beam_sel = self._sel["beam_sel"]
        else:
            self.beam_sel = slice(None)

    def process(self):
        """Load in each sidereal day.

        Returns
        -------
        ts : HFBData
            The timestream of each sidereal day.
        """

        if len(self.files) == self._file_ptr:
            raise pipeline.PipelineStopIteration

        # Collect garbage to remove any prior data objects
        gc.collect()

        # Fetch and remove the first item in the list
        file_ = self.files[self._file_ptr]
        self._file_ptr += 1

        # Handle file lists including time ranges
        if isinstance(file_, tuple):
            time_range = file_[1]
            file_ = file_[0]
        else:
            time_range = (None, None)

        # Set up the reader
        rd = HFBReader(file_)

        # Select time range
        rd.select_time_range(time_range[0], time_range[1])

        # Select frequency range
        rd.freq_sel = self.freq_sel

        # Select beams
        rd.beam_sel = self.beam_sel

        # Read file(s)
        self.log.info(f"Reading file {self._file_ptr} of {len(self.files)}. ({file_})")
        ts = rd.read()

        # Return timestream
        return ts

    def _find_beam(self):
        """Find NS beam number of beam closest to source at transit

        Returns
        -------
        beam_index_ns : int
            North-south index of beam closest to source at transit.
        """

        # Find source's telescope-y coordinate
        src_y = self.source_dec - ephemeris.CHIMELATITUDE

        # Choose beam model
        mdl = FFTFormedActualBeamModel()

        # Grid of beam numbers with EW beam number 1
        beams_ind = np.arange(1000, 1256)

        # Decide frequency (in MHz) at which to evaluate beam positions
        freq = self.freq_physical[0] if self.freq_physical else 600.0

        # Find beam positions
        beams_xy = mdl.get_beam_positions(beams_ind, freq).squeeze()

        # Find NS beam number of beam closest to calibration source
        beam_index_ns = np.abs(beams_xy[:, 1] - src_y).argmin()

        return beam_index_ns
