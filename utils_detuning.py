import numpy as np

from scipy.integrate import solve_ivp
from scipy.linalg import block_diag

from matplotlib import pyplot as plt

import utils_data

def make_a_m(t, f, q, fe: float = 0.):
    """Make matrix A of a mechanical mode.

    The matrix is parameterized by frequency ``f`` in hertz and quality ``q``.
    The frequency can be made time-varying by setting frequency
    factor ``fe`` to something else than 0, e.g., if fe > 0,
    then the frequency will increase with time, and if
    fe < 0, the frequency will decrease. Time ``t``
    represents current simulation time.
    """
    f = f + fe * t
    w = 2 * np.pi * f
    return np.array([
        [ 0,             1  ],
        [-np.square(w), -w/q]])

def make_b_m(t, f, k, fe: float = 0.):
    f = f + fe * t
    w = 2 * np.pi * f
    return np.array([
        [0              ],
        [-k*np.square(w)]])

def cavfun(t, x, sim_self):
    """
    Executes the logic of cavity differential equations. Input parameters ``t``
    and ``x`` are time and state, respectively.
    """

    pctr_on_rf = sim_self.pctr_on_rf
    K_rf       = sim_self.K_rf
    v_rf       = sim_self.v_rf
    modes_m_n  = sim_self.modes_m_n
    t_m        = sim_self.t_m
    a_rf       = sim_self.a_rf
    b_rf       = sim_self.b_rf

    # --! current state of a cavity field: real and imaginary components
    x_rf = np.array(x[:2]).reshape((-1, 1))

    # --! current states of all mechanical modes: displacements and velocities
    x_m = np.array(x[2:]).reshape((-1, 1))

    # --! input to cavity field: real and imaginary parts of a generator voltage
    u_rf = np.zeros((2, 1))
    if pctr_on_rf:
        # --! proportional control is on, so calculate an actuation signal u
        r_rf = np.array([
            [v_rf[0]],
            [v_rf[1]]])

        e_rf = r_rf - x_rf

        u_rf = K_rf * e_rf
    else:
        # --! proportional control is off, so our setpoint becomes our actuation signal
        u_rf = np.array([
            [v_rf[0]],
            [v_rf[1]]])

    # --! input to mechanical mode: accelerating field gradient squared
    #
    # --! field gradient has units MV/m, but since we simulate only one cell,
    # --! and one cell is approximately 0.1615 meters, then
    # --! we need to adjust the total gradient
    grad = np.sqrt(np.square(x_rf[0]) + np.square(x_rf[1]))
    grad = grad * 0.1615
    u_m = np.square(grad)

    # --! update detuning in cavity system dynamics
    disp_m = np.sum([x_m[2*i] for i in range(modes_m_n)])
    a_rf[0, 1] = -disp_m
    a_rf[1, 0] =  disp_m

    f_m  = sim_self.f_m
    fe_m = sim_self.fe_m
    q_m  = sim_self.q_m

    k_m = np.ones_like(f_m) * 2 * np.pi * 1. # coupling with units (2 * pi * Hz) / (MV/m)^2

    # --! assemble mechanical system and input matrices, A and B
    a_m = block_diag(*[make_a_m(t, f, q, fe) for f, fe, q in zip(f_m, fe_m, q_m)])
    b_m = np.concatenate([make_b_m(t, f, k, fe) for f, fe, k in zip(f_m, fe_m, k_m)], axis=0)

    # --! create an additional instance of mechanical matrix B and split it into
    # --! per-mode submatrices
    b_m_var  = b_m
    bs_m_var = np.split(b_m_var, b_m_var.shape[0] // 2, axis=0)

    # --! split mechanical time boundary array into per-mode parts
    ts_m = np.split(t_m, t_m.shape[0], axis=0) # split into rows

    for mat, timespan in zip(bs_m_var, ts_m):
        if not (timespan[0, 0] <= t and t < timespan[0, 1]):
            mat[:] = 0.

    # --! calculate derivatives
    dx_rf = a_rf @ x_rf + b_rf    @ u_rf
    dx_m  = a_m  @ x_m  + b_m_var * u_m

    return np.array([
        *dx_rf.flatten(),
        *dx_m.flatten()])


class detuning_sim:
    def __init__(self, config):

        q_rf       = config['q_rf']
        f_rf       = config['f_rf']
        w_rf       = 2 * np.pi * f_rf
        w_hbw_rf   = w_rf/2/q_rf  # half-bandwidth of RF cavity in rad/s
        f_hbw_rf   = w_hbw_rf/2/np.pi
        self.t_rf  = round(1/f_hbw_rf, 2)

        print(f'inf >> half-bandwidth of this radio frequency cavity is {f_hbw_rf:.2f} Herz')
        print(f'inf >> cavity filling time is {self.t_rf:.2f} seconds')

        self.a_rf = np.array([
            [-w_hbw_rf,  0.      ],
            [ 0.,       -w_hbw_rf]])

        self.b_rf = np.array([
            [w_hbw_rf, 0       ],
            [0,        w_hbw_rf]])

        self.pctr_on_rf = config['pctr_on_rf']
        self.K_rf       = config['K_rf']
        self.v_rf       = config['v_rf']

        # --! additional placeholders for the properties of cavity mechanical modes
        self.modes_m_n  = None
        self.t_m        = None

    def __call__(self, param, noise=None):
        """Simulates detuning according to given parameters ``param``."""

        # --! simulate with different parameters
        sim_o = [self.__sim(p) for p in param]

        # --! sum detuning
        #
        # --! the first two data in y are rf i and q, after that come mechanical modes
        # --! as displacement and velocity
        detuning = [self.__sum(o.y[2:]) for o in sim_o]

        # --! we want to skip the transient process of an RF cavity when scaling data,
        # --! so we specify the start
        start = 10

        # --! skip the first transient samples and reshape detuning row arrays into column arrays
        detuning = [d[:, start:].T for d in detuning]

        # --! add noise if specified
        if noise is not None:
            detuning = [d + np.random.normal(0, noise, size=d.shape) for d in detuning]

        return detuning

    def __sim(self, param):
        """
        Simulate a cavity resonance detuning by solving the cavity differential equation.
        The equation is parameterized using ``param``.
        """

        self.f_m  = param['f_m']
        self.fe_m = param['fe_m']
        self.q_m  = param['q_m']

        self.modes_m_n = len(self.f_m)
        print(f'inf >> number of mechanical modes specified: {self.modes_m_n}')

        # --! define timing parameters
        t_span = [0, param['t_rf_n'] * self.t_rf]
        dt     = param['dt']
        t      = np.arange(t_span[0], t_span[1], dt)

        # --! create a mechanical time boundary matrix, where
        # --! 'dont care' parameter -1 is replaced
        # --! by the actual time boundaries
        self.t_m = param['t_m']
        for timespan in np.split(self.t_m, self.t_m.shape[0], axis=0): # split into rows
            if timespan[0, 0] == -1:
                timespan[0, 0] = t[0]
            if timespan[0, 1] == -1:
                timespan[0, 1] = t[-1]

        # --! every mechanical mode has two states:
        # --! 1. displacement
        # --! 2. velocity
        modes_m = np.zeros(self.modes_m_n * 2)

        # --! define zero initial conditions
        x0 = [
            0,         # cavity field real
            0,         # cavity field imaginary
            *modes_m ] # all mechanical modes

        # --! simulate
        return solve_ivp(cavfun, t_span, x0, method='RK45', t_eval=t, args=(self,))

    def __sum(self, modes):
        """
        Detuning is a sum of mechanical mode displacements. Input ``modes``
        is shaped as [N, T], where N and T are the number of modes
        and the length of time series, respectively.
        """
        summed = np.zeros_like(modes[:2])

        modes_n = len(modes) // 2

        # --! sum displacements
        for i in range(modes_n):
            summed[0] = summed[0] + modes[2*i]
            summed[1] = summed[1] + modes[2*i + 1]
        return summed

    def disp(self, detuning, timestep=0.001):

        l = len(detuning[:, 0])
        t = np.arange(0., l*timestep, timestep)

        plt.figure(figsize=(6, 3))
        plt.subplot(1, 2, 1)
        plt.plot(t, detuning[:, 0])
        plt.xlabel('Time [s]')
        plt.ylabel('Detuning [rad/s]')
        plt.tight_layout()

        plt.subplot(1, 2, 2)
        plt.plot(t, detuning[:, 1])
        plt.xlabel('Time [s]')
        plt.ylabel('dDetuning/dt [rad/s^2]')
        plt.tight_layout()

        plt.show()

