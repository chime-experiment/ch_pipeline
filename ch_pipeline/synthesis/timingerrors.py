"""Tasks for simulating timing distribution errors."""


import numpy as np

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
        (non common-mode delay errors), `common_mode_cyl` (common-mode within
        a cylinder) and `common_mode_iceboard` (common-mode within an iceboard)
    common_mode_type : string, optional
        Options are 'random' and 'sinusoidal' if sim_type `common_mode_cyl` is chosen.
    sinusoidal_period : list, optional
        Specify the periods of the sinusoids for each cylinder. Needs to be specified
        when simulating `sinusoidal` `common_mode_cyl` timing errors.
    """
    ndays = config.Property(proptype=float, default=733)
    corr_length_delay = config.Property(proptype=float, default=3600)
    sigma_delay = config.Property(proptype=float, default=1e-12)

    sim_type = config.enum(
        ["relative", "common_mode_cyl", "common_mode_iceboard"], default="relative"
    )

    common_mode_type = config.enum(["random", "sinusoidal"], default="random")

    # Default periods of chime specific timing jitter with the clock
    # distribution system informed by data see doclib 704
    sinusoidal_period = config.Property(proptype=list, default=[333, 500])

    _prev_delay = None
    _prev_time = None

    amp = False

    nchannel = 16
    ncyl = 2

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
            self.delay_error = gain.generate_fluctuations(
                time, cf_delay, n_realisations, self._prev_time, self._prev_delay
            )

            gain_phase = (
                2.0
                * np.pi
                * freq[:, np.newaxis, np.newaxis]
                * 1e6
                * self.delay_error[np.newaxis, :, :]
                / np.sqrt(self.ndays)
            )

        if self.sim_type == "common_mode_cyl":
            n_realisations = 1
            ninput = self.ninput_global

            # Generates as many random delay errors as there are cylinders
            if self.comm.rank == 0:
                if self.common_mode_type == "sinusoidal":
                    P1 = self.sinusoidal_period[0]
                    P2 = self.sinusoidal_period[1]
                    omega1 = 2 * np.pi / P1
                    omega2 = 2 * np.pi / P2

                    delay_error = (
                        self.sigma_delay
                        * (np.sin(omega1 * time) - np.sin(omega2 * time))[np.newaxis, :]
                    )

                if self.common_mode_type == "random":
                    delay_error = gain.generate_fluctuations(
                        time,
                        cf_delay,
                        n_realisations,
                        self._prev_time,
                        self._prev_delay,
                    )
            else:
                delay_error = None

            # Broadcast to other ranks
            self.delay_error = self.comm.bcast(delay_error, root=0)

            # Split frequencies to processes.
            lfreq, sfreq, efreq = mpiutil.split_local(nfreq)

            # Create an array to hold all inputs, which are common-mode within
            # a cylinder
            gain_phase = np.zeros((lfreq, ninput, ntime), dtype=complex)
            # Since we have 2 cylinders populate half of them with a delay)
            # TODO: generalize this for 3 or even 4 cylinders in the future.
            gain_phase[:, ninput // self.ncyl :, :] = (
                2.0
                * np.pi
                * freq[sfreq:efreq, np.newaxis, np.newaxis]
                * 1e6
                * self.delay_error[np.newaxis, :, :]
                / np.sqrt(self.ndays)
            )

            gain_phase = mpiarray.MPIArray.wrap(gain_phase, axis=0, comm=self.comm)
            # Redistribute over input to match rest of the code
            gain_phase = gain_phase.redistribute(axis=1)
            gain_phase = gain_phase.view(np.ndarray)

        if self.sim_type == "common_mode_iceboard":
            nchannel = self.nchannel
            ninput = self.ninput_global
            # Number of channels on a board
            nboards = ninput // nchannel

            # Generates as many random delay errors as there are iceboards
            if self.comm.rank == 0:
                delay_error = gain.generate_fluctuations(
                    time, cf_delay, nboards, self._prev_time, self._prev_delay
                )
            else:
                delay_error = None

            # Broadcast to other ranks
            self.delay_error = self.comm.bcast(delay_error, root=0)

            # Calculate the corresponding phase by multiplying with frequencies
            phase = (
                2.0
                * np.pi
                * freq[:, np.newaxis, np.newaxis]
                * 1e6
                * self.delay_error[np.newaxis, :]
                / np.sqrt(self.ndays)
            )

            # Create an array to hold all inputs, which are common-mode within
            # one iceboard
            gain_phase = mpiarray.MPIArray(
                (nfreq, ninput, ntime), axis=1, dtype=np.complex128, comm=self.comm
            )
            gain_phase[:] = 0.0

            # Loop over inputs and and group common-mode phases on every board
            for il, ig in gain_phase.enumerate(axis=1):
                # Get the board number bi
                bi = int(ig / nchannel)
                gain_phase[:, il] = phase[:, bi]

            gain_phase = gain_phase.view(np.ndarray)

        self._prev_delay = self.delay_error
        self._prev_time = time

        return gain_phase


class SiderealTimingErrors(TimingErrors, gain.SiderealGains):
    """Generate sidereal timing errors gains on a sidereal grid.

    See the documentation for `TimingErrors` and `SiderealGains` for more detail.
    """

    pass
