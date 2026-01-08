# --! example: Duffing oscillator --!

import numpy as np
import scipy as sp

import os

import torch

import random
from scipy import signal
from scipy.integrate import solve_ivp
from collections import deque
from itertools import chain

import kind
import util_data
import util_dyna
import util_nn
import reinforcement_learning as rl


class model:
    """Wraps a KIND model in a data-normalizing adapter."""

    def __init__(self, model, normalizer):

        # --! freeze model
        model.eval()
        util_nn.freeze_module(model)

        self.model = model
        self.normalizer = normalizer

    def __call__(self, lookback):
        return self.forward(lookback)

    def forward(self, lookback):

        # --! normalize input data
        lookback, mask = self.normalizer.normalize(lookback)

        # --! pass normalized data to the model
        model_output = self.model(lookback)

        # --! denormalize model output
        #
        # --! extract predictions that need to be denormalized
        prediction = model_output[0]
        prediction_nom = model_output[1]
        prediction_exc = model_output[3]

        # --! denormalize extracted predictions
        prediction = self.normalizer.denormalize(prediction, mask)
        prediction_nom = self.normalizer.denormalize(prediction_nom, mask)
        prediction_exc = self.normalizer.denormalize(prediction_exc, mask)

        # --! put unscaled timeseries back to the result tuple and return the tuple
        model_output = list(model_output)
        model_output[0] = prediction
        model_output[1] = prediction_nom
        model_output[3] = prediction_exc

        model_output = tuple(model_output)

        # --! return model output
        return model_output


def duffing_update(t, state, sim, u):
    """Generates a Duffing ``state`` update."""

    x1, x2 = state
    dx1 = x2
    dx2 = -sim.delta * x2 - sim.alpha * x1 - sim.beta * x1**3 + u + sim.gamma * np.cos(sim.omega * t)
    return [dx1, dx2]


class duffing(util_dyna.environment):
    """Simulates a Duffing oscillator. Implements Dyna environment interface."""

    def __init__(self, beta, gamma, reward, alpha=-1.0, delta=0.2, omega=1.2, dt_sim=1e-4, dt_control=1e-2, t_final=100.0):

        self.alpha = alpha
        self.beta  = beta
        self.gamma = gamma
        self.delta = delta
        self.omega = omega

        self.ic = [0.0, 0.0]
        self.x = self.ic[0]
        self.dx = self.ic[1]

        self.policy = None
        self.reward = reward

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

        return np.array([[self.x, self.dx]])

    def step(self, u):

        # --! based on current state and action, calculate reward 
        obs = np.array([[self.x, self.dx]])
        action = np.array([[u]])
        reward = self.reward(np.concatenate([obs, action], axis=-1))

        # --! for integration below, define step timing
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

        # --! save the last integrated value as the next observation
        self.x = solution.y[0, -1]
        self.dx = solution.y[1, -1]

        next_obs = np.array([[self.x, self.dx]])

        self.jstep += 1
        done = self.jstep == self.nstep

        return next_obs, reward, done

    def step_batch(self, state, action):
        """Steps one time step for every ``state`` under corresponding ``action`` and returns the next states."""

        # --! for integration below, define step timing
        t_start = 0 * self.dt_control
        t = t_start + np.arange(0, self.dt_control, self.dt_sim)
        t_span = (t[0], t[-1])

        next_state_buf = []

        for s, a in zip(state, action):

            # --! solve current initial value problem
            solution = solve_ivp(
                duffing_update,
                t_span,
                [s[0], s[1]],
                t_eval=t,
                args=(self, a))

            # --! save the last integrated value as the next state
            next_x1 = solution.y[0, -1]
            next_x2 = solution.y[1, -1]

            next_state = np.array([[next_x1, next_x2]])
            next_state_buf.append(next_state)

        return np.stack(next_state_buf, axis=0)

    def simulate(self, skip_nsample=0):
        """Simulates this duffing oscillator from time equals 0 seconds till time specified in ``t_final``."""

        x_buf = []
        dx_buf = []
        u_buf = []

        # --! first, we reset this simulator to start from the initial condition
        obs = self.reset()
        done = False

        # --! step though this environment while not done
        while not done:

            # --! derive control input
            if self.policy is not None:
                action = np.squeeze(self.policy(obs))
            else:
                action = 0.0

            # --! step to get the next observation
            next_obs, reward, done = self.step(action)

            # --! save current observation and action in buffers
            x_buf.append(obs[0, 0])
            dx_buf.append(obs[0, 1])
            u_buf.append(action)

            # --! repeat
            obs = next_obs

        # --! reshape lists as column vectors, concatenate them, unsqueeze at axis 0 and return
        sim_o = np.concatenate([np.array(buf[skip_nsample:]).reshape(-1, 1) for buf in [x_buf, dx_buf, u_buf]], axis=-1)
        return np.expand_dims(sim_o, axis=0)


def make_duffing(name, reward):

    if name == 'ood':
        d = duffing(20.0, 20.0, reward)
        d.ic = [2.5, 5.0]
        return d
    elif name == 'id':
        d = duffing(1.0, 0.1, reward)
        d.ic = [0.2, 0.2]
        return d
    else:
        return None


class lqr(util_dyna.policy):

    def __init__(self, gain, noise=0.0):
        self.gain = np.atleast_2d(gain)
        self.noise = noise

    def act(self, state):
        u = -np.matmul(state, np.transpose(self.gain))
        return u + self.noise * np.random.standard_normal(size=u.shape)


def make_policy(duffing, q, r, noise=0.0):
    """Makes a baseline LQR policy for a Duffing oscillator.
    Parameters ``q`` and ``r`` are diagonal numpy arrays for state and action costs, respectively.
    Parameter ``noise`` defines the standard deviation of an additive Gaussian noise."""

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

    p = sp.linalg.solve_discrete_are(a, b, q, r)

    # --! synthesize LQR gain matrix K = (B.T * P * B + R)^-1 * (B.T * P * A)
    bp = b.T.dot(p)
    lhs = bp.dot(b)
    lhs += r
    rhs = bp.dot(a)
    k = np.linalg.solve(lhs, rhs)

    # --! wrap the gain matrix in a callable policy strategy
    return lqr(k, noise=noise)    


class normalizer(util_data.normalizer):

    def __init__(self, timeseries_nom, timeseries_exc, state_ndim, control_ndim, mask_ndim):

        # --! save data dimensions
        self.state_ndim = state_ndim
        self.control_ndim = control_ndim
        self.mask_ndim = mask_ndim

        # --! take nominal statistics
        state_and_control, _ = torch.split(timeseries_nom, [self.state_ndim + self.control_ndim, self.mask_ndim], dim=-1)
        self.mean_nom = state_and_control.mean()
        self.std_nom = state_and_control.std()
        self.std_nom = torch.maximum(self.std_nom, self.std_min)

        # --! take excursion statistics
        state_and_control, _ = torch.split(timeseries_exc, [self.state_ndim + self.control_ndim, self.mask_ndim], dim=-1)
        self.mean_exc = state_and_control.mean()
        self.std_exc = state_and_control.std()
        self.std_exc = torch.maximum(self.std_exc, self.std_min)

        # --! compute a separator between the two regimes based on state norm
        #
        # --! note that the ceil operation is supposed to push the resulting separator slightly above the
        # --! actual maximum state norms, so that later a simple 'less-then'
        # --! operator could be used
        self.regime_sep = util_data.ceil(self._compute_state_norm_max(self._extract_state(timeseries_nom)).max(), decimals=2)

    def mask(self, timeseries):
        """Determines a data mask that differentiates between nominal and excursion data."""
        mask = self._compute_state_norm_max(self._extract_state(timeseries)) < self.regime_sep
        return torch.squeeze(mask)

    def normalize(self, timeseries):

        mask = self.mask(timeseries)

        state_and_control, control_mask = torch.split(timeseries, [self.state_ndim + self.control_ndim, self.mask_ndim], dim=-1)
        state_and_control_norm = torch.empty_like(state_and_control)

        state_and_control_norm[mask] = (state_and_control[mask] - self.mean_nom) / self.std_nom
        state_and_control_norm[~mask] = (state_and_control[~mask] - self.mean_exc) / self.std_exc

        state, control = torch.split(state_and_control_norm, [self.state_ndim, self.control_ndim], dim=-1)
        control = control * control_mask

        return torch.cat([state, control, control_mask], dim=-1), mask

    def normalize_state(self, state):
        assert state.shape[-1]==self.state_ndim

        mask = self.mask(state)

        norm_state = torch.empty_like(state)
        norm_state[mask] = (state[mask] - self.mean_nom) / self.std_nom
        norm_state[~mask] = (state[~mask] - self.mean_exc) / self.std_exc

        return norm_state, mask

    def denormalize(self, timeseries, mask):

        assert timeseries.shape[-1]==self.state_ndim or timeseries.shape[-1]==self.control_ndim

        denorm_timeseries = torch.empty_like(timeseries)

        denorm_timeseries[mask] = timeseries[mask] * self.std_nom + self.mean_nom
        denorm_timeseries[~mask] = timeseries[~mask] * self.std_exc + self.mean_exc

        return denorm_timeseries

    def _extract_state(self, timeseries):
        """Extracts state dimensions from the given ``timeseries``."""

        if timeseries.shape[-1]==self.state_ndim:
            return timeseries

        state, _ = torch.split(timeseries, [self.state_ndim, self.control_ndim + self.mask_ndim], dim=-1)
        return state

    def _compute_state_norm_max(self, state):
        """Computes the maximum ``state`` norm along the time steps dimension. All dimensions are preserved."""
        return torch.max(torch.linalg.norm(state, dim=-1, keepdim=True), dim=1, keepdim=True)[0]


class dataset(util_data.dataset):
    """Represents synthetic Duffing data, both nominal and excursion."""

    state_ndim = 2
    control_ndim = 1
    mask_ndim = 1

    def __init__(self,
                 file_dir, file_name, file_ext,
                 data_nsample,
                 data_split_size,
                 batch_size, window_nsample, load_normalized=True):
        super().__init__(file_dir, file_name, file_ext,
                         data_nsample, data_split_size,
                         batch_size, window_nsample, load_normalized)

    def make_path(self, data_type='nom'):
        file_name = self.file_name + '_' + data_type + self.file_ext
        return os.path.join(self.file_dir, file_name)

    def extract_target(self, window):
        """Extracts the first two feature dimensions: position and velocity."""
        return window[:, :, :self.state_ndim]

    def init_normalization(self):

        # --! read data
        timeseries_nom = self.read_timeseries(self.make_path(data_type='nom'))
        timeseries_exc = self.read_timeseries(self.make_path(data_type='exc'))

        return normalizer(timeseries_nom, timeseries_exc, self.state_ndim, self.control_ndim, self.mask_ndim)


class replay_buffer(rl.replay_buffer):
    def __init__(self, buffer=None):
        self.buffer = buffer if buffer is not None else []

    def __add__(self, other):
        a = self.buffer
        b = other.buffer

        # --! interleave both buffers, such that the new buffer has elements: a[0], b[0], a[1], b[1], etc.
        c = list(chain.from_iterable(zip(a, b)))

        return replay_buffer(buffer=c)

    def add(self, lookback, reward, next_lookback, done):

        # --! convert a bool flag to a float which is either 0.0 or 1.0
        done = done.float()

        # --! all entities must be shaped as 3D data
        done = torch.atleast_3d(done)
        reward = torch.atleast_3d(reward)

        # --! pack all elements as a tuple and put the tuple into the buffer
        self.buffer.append((
            lookback.detach(),
            reward,
            next_lookback.detach(),
            done
        ))

    def random_batch(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        return map(torch.cat, zip(*batch))

    def empty(self):
        return len(self.buffer)==0

    def encode_lookback(self, sa_window):
        s, a = map(torch.cat, zip(*sa_window))
        return torch.unsqueeze(torch.cat([s, a, torch.ones_like(a)], dim=-1), 0)

    def encode_sa(self, s, a):
        return torch.cat([s, a, torch.ones_like(a)], dim=-1)

    def extract_current_state(self, lookback):
        return lookback[:, [-1], :2]

    def extract_current_action(self, lookback):
        return lookback[:, [-1], 2:3]

    def update_current_action(self, lookback, a):
        lookback[:, -2:, [2]] = a

    def update_lookback(self, lookback, s):

        # --! make a dummy (zero) action
        a = torch.zeros(s.shape[0], s.shape[1], 1)

        sa = self.encode_sa(s, a)
        return torch.cat([lookback[:, 1:], sa], dim=1)

    def get_coarse_zeta(self, lookback):

        # --! compute norms of states
        state, _ = torch.split(lookback, [2, 1 + 1], dim=-1) # <- get the number of data dimensions !!!
        return torch.mean(torch.linalg.norm(state, dim=-1, keepdim=True), dim=1, keepdim=True)

    def coarse_zeta_threshold(self):
        return 0.05

