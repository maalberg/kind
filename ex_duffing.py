# --! example: Duffing oscillator --!

import numpy as np
import scipy as sp

import os

import torch

from scipy import signal
from scipy.integrate import solve_ivp

import util_data
import util_dyna


def duffing_update(t, state, sim, u):
    """Generates a Duffing ``state`` update."""

    x1, x2 = state
    dx1 = x2
    dx2 = -sim.delta * x2 - sim.alpha * x1 - sim.beta * x1**3 + u + sim.gamma * np.cos(sim.omega * t)
    return [dx1, dx2]


class duffing(util_dyna.environment):
    """Simulates a Duffing oscillator. In addition, implements Dyna environment interface."""

    def __init__(self, beta, gamma, alpha=-1.0, delta=0.2, omega=1.2, dt_sim=1e-4, dt_control=1e-2, t_final=100.0):

        self.alpha = alpha
        self.beta  = beta
        self.gamma = gamma
        self.delta = delta
        self.omega = omega

        self.ic = [0.0, 0.0]
        self.x = self.ic[0]
        self.dx = self.ic[1]
        self.policy = None

        # --! define simulation timing
        self.dt_sim = dt_sim
        self.dt_control = dt_control
        self.jstep = 0
        self.t_final = t_final
        self.nstep = int(self.t_final / self.dt_control)

    def reset(self):

        # --! reset the state to the initial condition
        self.x = self.ic[0]
        self.dx = self.ic[1]

        self.jstep = 0

        return [self.x, self.dx]

    def step(self, u):

        # --! define step timing
        t_start = self.jstep * self.dt_control
        t = t_start + np.arange(0, self.dt_control, self.dt_sim)
        t_span = (t[0], t[-1])

        # --! solve this initial value problem
        solution = solve_ivp(
            duffing_update,
            t_span,
            [self.x, self.dx],
            t_eval=t,
            args=(self, u))

        self.jstep += 1

        # --! save the last integrated values as the current state
        self.x = solution.y[0, -1]
        self.dx = solution.y[1, -1]

        return [self.x, self.dx]

    def simulate(self):
        """Simulates this duffing oscillator from time equals 0 seconds till time specified in ``t_final``."""

        x_buf = []
        dx_buf = []
        u_buf = []

        # --! first, we reset this simulator to start from the initial condition
        state = self.reset()

        # --! perform a number of simulation steps, where the number is defined by control steps
        for step in range(self.nstep):

            # --! derive control input
            if self.policy is not None:
                u = np.squeeze(self.policy(np.array(state).reshape((-1, 1))))
            else:
                u = 0.0

            # --! save current state and action in replay buffers
            x_buf.append(state[0])
            dx_buf.append(state[1])
            u_buf.append(u)

            # --! get the next state
            state = self.step(u)

        # --! reshape lists as column vectors, concatenate them, unsqueeze at axis 0 and return
        sim_o = np.concatenate([np.array(buf).reshape(-1, 1) for buf in [x_buf, dx_buf, u_buf]], axis=-1)
        return np.expand_dims(sim_o, axis=0)


def make_duffing(name):

    if name == 'ood':
        d = duffing(20.0, 20.0)
        d.ic = [2.5, 5.0]
        return d
    elif name == 'id':
        d = duffing(1.0, 0.1)
        d.ic = [0.2, 0.2]
        return d
    else:
        return None


def make_policy(duffing, q=[1.0, 0.1], r=[1.0]):
    """Makes a baseline LQR policy for a Duffing oscillator."""

    alpha = duffing.alpha
    delta = duffing.delta

    a = np.array([
        [ 0,      1    ],
        [-alpha, -delta],
    ])

    b = np.array([
        [ 0    ],
        [ 1    ],
    ])

    # --! discretize a continuous-time state-space system
    sys = signal.StateSpace(a, b, np.eye(2), np.zeros((2, 1)))
    sys = sys.to_discrete(1e-2)

    # --! take discrete matrices A and B
    a = sys.A
    b = sys.B

    p = sp.linalg.solve_discrete_are(a, b, np.diag(q), np.diag(r))

    # --! synthesize LQR gain matrix K = (B.T * P * B + R)^-1 * (B.T * P * A)
    bp = b.T.dot(p)
    lhs = bp.dot(b)
    lhs += r
    rhs = bp.dot(a)
    k = np.linalg.solve(lhs, rhs)

    # --! wrap the gain matrix in a callable policy strategy
    return util_dyna.lqr(k, noise=0.0)    


class dataset(util_data.dataset):
    """Represents synthetic Duffing data, both nominal and excursion."""

    state_ndim = 2
    control_ndim = 1
    mask_ndim = 1

    def __init__(self,
                 file_dir, file_name, file_ext,
                 data_nsample,
                 data_split_size,
                 batch_size, window_nsample):
        super().__init__(file_dir, file_name, file_ext,
                         data_nsample, data_split_size,
                         batch_size, window_nsample)

    def make_path(self, data_type='nom'):
        file_name = self.file_name + '_' + data_type + self.file_ext
        return os.path.join(self.file_dir, file_name)

    def extract_target(self, window):
        """Extracts the first two feature dimensions: position and velocity."""
        return window[:, :, :self.state_ndim]

    def init_normalization(self):

        # --! read nominal data
        timeseries_nom = self.read_timeseries(self.make_path(data_type='nom'))
        state_and_control, _ = torch.split(timeseries_nom, [self.state_ndim + self.control_ndim, self.mask_ndim], dim=-1)

        # --! take nominal statistics
        self.mean_nom = state_and_control.mean()
        self.std_nom = state_and_control.std()
        self.std_nom = torch.maximum(self.std_nom, self.min_std)

        # --! read excursion data
        timeseries_exc = self.read_timeseries(self.make_path(data_type='exc'))
        state_and_control, _ = torch.split(timeseries_exc, [self.state_ndim + self.control_ndim, self.mask_ndim], dim=-1)

        # --! take excursion statistics
        self.mean_exc = state_and_control.mean()
        self.std_exc = state_and_control.std()
        self.std_exc = torch.maximum(self.std_exc, self.min_std)

    def normalize(self, window, data_type='nom'):

        # --! this method is not supposed to be called for mixed data
        assert data_type in ['nom', 'exc']

        if data_type=='nom':
            mean = self.mean_nom
            std = self.std_nom
        else:
            mean = self.mean_exc
            std = self.std_exc

        if window.shape[-1]==self.state_ndim:
            window = (window - mean) / std
        else:
            state_and_control, mask = torch.split(window, [self.state_ndim + self.control_ndim, self.mask_ndim], dim=-1)

            state_and_control = (state_and_control - mean) / std
            state, control = torch.split(state_and_control, [self.state_ndim, self.control_ndim], dim=-1)
            control = control * mask

            window = torch.cat([state, control, mask], dim=-1)

        return window

    def denormalize(self, window, data_type='nom'):

        # --! this method is not supposed to be called for mixed data
        assert data_type in ['nom', 'exc']

        if data_type=='nom':
            mean = self.mean_nom
            std = self.std_nom
        else:
            mean = self.mean_exc
            std = self.std_exc

        if window.shape[-1]==self.state_ndim:
            window = window * std + mean
        else:
            state_and_control, mask = torch.split(window, [self.state_ndim + self.control_ndim, self.mask_ndim], dim=-1)
            state_and_control = state_and_control * std + mean

            window = torch.cat([state_and_control, mask], dim=-1)

        return window

