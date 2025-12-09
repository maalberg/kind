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


class duffing(util_dyna.simulator):
    """Simulates a Duffing oscillator."""

    def __init__(self, beta, gamma, alpha=-1.0, delta=0.2, omega=1.2):

        self.alpha = alpha
        self.beta  = beta
        self.gamma = gamma
        self.delta = delta
        self.omega = omega

        self.policy = None

    def simulate(self, ic, dt_control=1e-2, dt_sim=1e-4, t_final=100, skip_nsample=0):

        x_buf = []
        dx_buf = []
        u_buf = []

        x_buf.append(ic[0])
        dx_buf.append(ic[1])
        u_buf.append(0.0) # << the first action is initialized with zero

        for step in range(int(t_final / dt_control)):

            # --! define the timing of local simulation
            t_local_start = step * dt_control
            t_local = t_local_start + np.arange(0, dt_control, dt_sim)
            t_span = (t_local[0], t_local[-1])

            # --! simulate locally
            solution = solve_ivp(duffing_update, t_span, [x_buf[-1], dx_buf[-1]], t_eval=t_local, args=(self, u_buf[-1]))

            # --! save the last integrated value from the local simulation
            x_buf.append(solution.y[0, -1])
            dx_buf.append(solution.y[1, -1])

            # --! based on integrated state, compute corresponding control input u
            state = [x_buf[-1], dx_buf[-1]]
            u = np.squeeze(self.policy(np.array(state).reshape((-1, 1))))

            u_buf.append(u)

        # --! reshape lists as column vectors, concatenate them, unsqueeze at axis 0 and return
        sim_o = np.concatenate([np.array(buf[skip_nsample:]).reshape(-1, 1) for buf in [x_buf, dx_buf, u_buf]], axis=-1)
        return np.expand_dims(sim_o, axis=0)


def make_duffing(name):

    if name == 'ood':
        return duffing(20.0, 20.0)
    elif name == 'id':
        return duffing(1.0, 0.1)
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
    return util_dyna.lqr(k, noise=0.01)    


class dataset(util_data.dataset):
    """Represents synthetic Duffing data, both nominal and excursion."""

    # --! minimum standard deviation to avoid division by zero-like deviation values
    min_std = torch.tensor(1e-3, dtype=torch.float32)

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

        self.mean = 0.
        self.std = self.min_std

    def make_path(self, data_type='nom'):
        file_name = self.file_name + '_' + data_type + self.file_ext
        return os.path.join(self.file_dir, file_name)

    def extract_target(self, window):
        """Extracts the first two feature dimensions: position and velocity."""
        return window[:, :, :self.state_ndim]

    def adapt_mask(self, window):

        # --! mask stays as it is
        return window

    def init_normalization(self, timeseries):

        state_and_control, _ = torch.split(timeseries, [self.state_ndim + self.control_ndim, self.mask_ndim], dim=-1)

        # --! take global statistics
        self.mean = state_and_control.mean()
        self.std = state_and_control.std()
        self.std = torch.maximum(self.std, self.min_std)

    def normalize(self, window):

        if window.shape[-1]==self.state_ndim:
            window = torch.clip((window - self.mean) / self.std, min=-3.0, max=3.0)
        else:
            state_and_control, mask = torch.split(window, [self.state_ndim + self.control_ndim, self.mask_ndim], dim=-1)

            state_and_control = torch.clip((state_and_control - self.mean) / self.std, min=-3.0, max=3.0)
            state, control = torch.split(state_and_control, [self.state_ndim, self.control_ndim], dim=-1)
            control = control * mask

            window = torch.cat([state, control, mask], dim=-1)

        return window

    def denormalize(self, window):

        if window.shape[-1]==self.state_ndim:
            window = window * self.std + self.mean
        else:
            state_and_control, mask = torch.split(window, [self.state_ndim + self.control_ndim, self.mask_ndim], dim=-1)

            state_and_control = state_and_control * self.std + self.mean

            window = torch.cat([state_and_control, mask], dim=-1)

        return window

    def noise(self, window):

        # --! measured data already has noise
        return window


class dataset_factory(util_data.dataset_factory):

    def create_synthetic(self, args):
        return dataset(
            args.file_dir, args.file_name, args.file_ext,
            args.data_nsample,
            (args.data_train_size, args.data_test_size),
            args.batch_size, (args.lookback_nsample, args.forecast_nsample))

