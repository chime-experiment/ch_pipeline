"""Tasks for simulating timing distribution errors.

Tasks
=====

.. autosummary::
    :toctree:

    TimingErrors
"""


import numpy as np
from mpi4py import MPI

from caput import config, mpiarray, mpiutil
from draco.synthesis import gain


class TimingErrors(gain.BaseGains):
    r"""Simulate timing distribution errors and propagate them to gain errors.

    This task simulates errors in the delay and calculates the errors in the
    phase of the gain according to

    .. math::
        \delta \phi_i = 2 \pi \nu \delta \tau_i

    The phase errors are generated for times matching the input timestream file.

    There are severeal options to run the timing distribution simulations.
    Choosing a `sim_type` defines if the delay errors are non common-mode or
    common-mode within a cylinder or iceboard.

    Furthermore there are two different options to simulate delay errors that
    are common-mode within a cylinder (`common_mode_cyl`). `Random` generates
    random fluctuations in phase whereas 'sinusoidal' as the name suggests
    simulates sinusoids for each cylinder with frequencies specified in
    the attribute `sinusoidal_period`.

    Attributes
    ----------
    corr_length_delay : float
        Correlation length for delay fluctuations in seconds.
    sigma_delay : float
        Size of fluctuations for delay fluctuations (s).
    sim_type: string, optional
        Timing error simulation type. List of allowed options are `relative`
        (non common-mode delay errors), `commonmode` (common-mode within number of
        channels nchannels)
    nchannels : int
        Number of channels that share a delay error. Default: 128.
    commonmode_type : string, optional
        Options are 'random' and 'sinusoidal' if sim_type `commonmode` is chosen.
        'Sinusoidal' variations are patterned after CHIME's chiller cycling (doclib
        #704) and exclusively commonmode to huts.
    sinusoidal_period : list, optional
        Specify the periods of the sinusoids for each hut. Needs to be specified
        when simulating `sinusoidal` timing errors.
    ncyl : int
        Number of cylinders in this simulation. Default : 2.
    """
    corr_length_delay = config.Property(proptype=float, default=3600)
    sigma_delay = config.Property(proptype=float, default=1e-12)

    sim_type = config.enum(
        ["relative", "commonmode"], default="relative"
    )

    nchannels = config.Property(proptype=int, default=128)

    commonmode_type = config.enum(["random", "sinusoidal"], default="random")

    # Default periods of chime specific timing jitter with the clock
    # distribution system informed by data see doclib 704
    sinusoidal_period = config.Property(proptype=list, default=[333, 500])
    ncyl = config.Property(proptype=int, default=2)

    ndays = config.Property(proptype=float, default=1)

    _prev_delay = None
    _prev_time = None

    amp = False

    def _generate_phase(self, time):
        ntime = len(time)
        freq = self.freq
        nfreq = len(freq)

        # Generate the correlation function
        cf_delay = self._corr_func(self.corr_length_delay, self.sigma_delay)

        # Check if we are simulating relative delays or common mode delays
        if self.sim_type == "relative":
            n_realisations = self.ninput_local

            # Generate delay fluctuations
            delay_error = gain.generate_fluctuations(
                time, cf_delay, n_realisations, self._prev_time, self._prev_delay
            )

            gain_phase = (
                2.0
                * np.pi
                * freq[:, np.newaxis, np.newaxis]
                * 1e6
                * delay_error[np.newaxis, :, :]
                / np.sqrt(self.ndays)
            )

        if self.sim_type == "commonmode":
            ninput = self.ninput_global

            if ninput % self.nchannels != 0:
                raise ValueError("Total number of inputs must be devisable by number of channels that have commonmode timing errors")

            n_realisations = ninput // self.nchannels

            comm = MPI.COMM_WORLD

            # Check if random or sinusoidal
            if self.commonmode_type == "random":


                if comm.rank == 0:
                    delay_error = gain.generate_fluctuations(
                        time,
                        cf_delay,
                        n_realisations,
                        self._prev_time,
                        self._prev_delay,
                        )

                else:
                    delay_error = np.zeros((n_realisations, ntime), dtype=float)

                comm.Bcast(delay_error, root=0)

                phase = (
                    2.0
                    * np.pi
                    * freq[:, np.newaxis, np.newaxis]
                    * 1e6
                    * delay_error[np.newaxis, :, :]
                    / np.sqrt(self.ndays)
                    )

                gain_phase = mpiarray.MPIArray((nfreq, ninput, ntime), axis=1, dtype=complex)

                gain_phase[:] = 0.0

                for il, ig in gain_phase.enumerate(axis=1):
                    bi = int(ig / self.nchannels)
                    gain_phase[:, il] = phase[:, bi]

                gain_phase = gain_phase.view(np.ndarray)

                if np.sum(np.isnan(gain_phase)) != 0:
                    raise RuntimeError("Found nans in gain phase")

            elif self.commonmode_type == "sinusoidal":
                P1 = self.sinusoidal_period[0]
                P2 = self.sinusoidal_period[1]
                omega1 = 2 * np.pi / P1
                omega2 = 2 * np.pi / P2

                if comm.rank == 0:
                    delay_error = (
                        self.sigma_delay * np.sqrt(2)
                        * (np.sin(omega1 * time) - np.sin(omega2 * time))[np.newaxis, :]
                        )
                else:
                    delay_error = np.zeros((1, ntime), dtype=float)

                comm.Bcast(delay_error, root=0)

                # If even number of cylinders, put half of the channels in one receiving hut
                if self.ncyl % 2 == 0:
                    nchannels = ninput // 2
                # Otherwise put 1/3 channels in one receiving hut.
                else:
                    nchannels = ninput // 3

                # Split frequencies to processes.
                lfreq, sfreq, efreq = mpiutil.split_local(nfreq)

                # Create and array to hold all inputs, which are common-mode within
                # a cylinder
                gain_phase = np.zeros((lfreq, ninput, ntime), dtype=complex)

                gain_phase[:, nchannels:, :] = (
                    2.0
                    * np.pi
                    * freq[sfreq:efreq, np.newaxis, np.newaxis]
                    * 1e6
                    * delay_error[np.newaxis, :, :]
                    / np.sqrt(self.ndays)
                    )

                gain_phase = mpiarray.MPIArray.wrap(gain_phase, axis=0)
                # Redistribute over input to match rest of the code
                gain_phase = gain_phase.redistribute(axis=1)
                gain_phase = gain_phase.view(np.ndarray)

        self._prev_delay = delay_error
        self._prev_time = time

        return gain_phase



class SiderealTimingErrors(TimingErrors, gain.SiderealGains):
    """Generate sidereal timing errors gains on a sidereal grid.

    See the documentation for `TimingErrors` and `SiderealGains` for more detail.
    """

    pass
