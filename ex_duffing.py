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

import util_data
import util_nn
import reinforcement_learning as rl


def duffing_energy_torch(state, alpha=-1.0, beta=20.0):
    q = state[..., [0]]
    qdot = state[..., [1]]
    return 0.5 * qdot**2 + 0.5 * alpha * q**2 + 0.25 * beta * q**4


class DuffingRewardTorch:
    def __init__(self, Q, R, setpoint,
                 alpha=-1.0, beta=20.0,
                 lambda_E=0.1,
                 device=None,
                 dtype=torch.float32):

        self.Q = torch.atleast_2d(torch.as_tensor(Q, dtype=dtype, device=device))
        self.R = torch.atleast_2d(torch.as_tensor(R, dtype=dtype, device=device))
        self.setpoint = torch.atleast_2d(torch.as_tensor(setpoint, dtype=dtype, device=device))

        self.alpha = alpha
        self.beta = beta
        self.lambda_E = lambda_E

    def __call__(self, state, action):

        state = torch.atleast_2d(state)
        action = torch.atleast_2d(action)

        x_err = state - self.setpoint

        state_cost = x_err @ self.Q @ x_err.transpose(-1, -2)
        action_cost = action @ self.R @ action.transpose(-1, -2)

        # Duffing energy
        energy_cost = duffing_energy_torch(
            state,
            alpha=self.alpha,
            beta=self.beta
        )

        reward = -(state_cost + action_cost + self.lambda_E * energy_cost)
        return reward


def duffing_energy(state, alpha=-1.0, beta=20.0):
    q = state[..., 0]
    qdot = state[..., 1]
    return 0.5 * qdot**2 + 0.5 * alpha * q**2 + 0.25 * beta * q**4


class duffing_reward:
    def __init__(self, Q, R, setpoint,
                 alpha=-1.0, beta=20.0,
                 lambda_E=0.1):
        self.Q = np.atleast_2d(Q)
        self.R = np.atleast_2d(R)
        self.setpoint = np.atleast_2d(setpoint)
        self.alpha = alpha
        self.beta = beta
        self.lambda_E = lambda_E

    def __call__(self, state, action):
        x_err = state - self.setpoint

        state_cost = x_err @ self.Q @ x_err.T
        action_cost = action @ self.R @ action.T
        energy_cost = duffing_energy(state,
                                     alpha=self.alpha,
                                     beta=self.beta)

        return -(state_cost + action_cost + self.lambda_E * energy_cost)


def duffing_update(t, state, sim, u):
    """Generates a Duffing ``state`` update."""

    x1, x2 = state
    dx1 = x2
    dx2 = -sim.delta * x2 - sim.alpha * x1 - sim.beta * x1**3 + u + sim.gamma * np.cos(sim.omega * t)
    return [dx1, dx2]


class duffing(rl.environment):
    """Simulates a Duffing oscillator. Implements Dyna environment interface."""

    def __init__(self, beta, gamma, reward, alpha=-1.0, delta=0.2, omega=1.2, dt_sim=1e-4, dt_control=2e-2, t_final=100.0):

        self.alpha = alpha
        self.beta  = beta
        self.gamma = gamma
        self.delta = delta
        self.omega = omega

        self.ic = [0.0, 0.0]
        self.x = self.ic[0]
        self.dx = self.ic[1]

        self.base_policy = None
        self.residual_policy = None
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

    def step(self, action):

        # --! based on current state and action, calculate reward 
        state = np.array([[self.x, self.dx]])
        reward = self.reward(state, np.array([[action]]))

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
            args=(self, action))

        # --! save the last integrated value as the next observation
        self.x = solution.y[0, -1]
        self.dx = solution.y[1, -1]

        next_state = np.array([[self.x, self.dx]])

        self.jstep += 1
        done = self.jstep == self.nstep

        return next_state, reward, done

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
            if self.base_policy is not None:
                action = np.squeeze(self.base_policy(obs))
                if self.residual_policy is not None:
                    tata
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


class duffing_adapter(rl.environment):
    """Adapts a NumPy-based Duffing environment to PyTorch."""

    def __init__(self, env):
        self.env = env

    def reset(self):
        obs = self.env.reset()
        return torch.from_numpy(obs).to(dtype=torch.float32)

    def step(self, action):
        action = action.detach().cpu().numpy()

        next_state, reward, done = self.env.step(np.squeeze(action))

        next_state = torch.from_numpy(next_state).to(dtype=torch.float32)
        reward = torch.tensor(reward).to(dtype=torch.float32)
        done = torch.tensor(done)

        return next_state, reward, done


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


class lqr:

    def __init__(self, gain, setpoint=[0.0, 0.0], noise=0.0):
        self.gain = np.atleast_2d(gain)
        self.setpoint = np.atleast_2d(setpoint)
        self.noise = noise

    def act(self, state):
        state = state - self.setpoint
        u = -np.matmul(state, np.transpose(self.gain))
        return u + self.noise * np.random.standard_normal(size=u.shape)


def make_policy(duffing, q, r, setpoint=[0.0, 0.0], noise=0.0):
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

    # --! return gain matrix
    return np.atleast_2d(k)


class normalizer(util_data.normalizer):

    def __init__(self, timeseries_nom, timeseries_exc, setpoint, state_ndim, control_ndim, mask_ndim):

        # --! save data dimensions
        self.state_ndim = state_ndim
        self.control_ndim = control_ndim
        self.mask_ndim = mask_ndim

        # --! save setpoint before calling subtracting method
        self.setpoint = setpoint

        # --! compute a separator between the two regimes based on state norm
        #
        # --! note that the ceil operation is supposed to push the resulting separator slightly above the
        # --! actual maximum state norms, so that later a simple 'less-then'
        # --! operator could be used
        state_norm_max = self._compute_state_norm_max(self._extract_state(timeseries_nom) - self.setpoint).max()
        self.regime_sep = util_data.ceil(state_norm_max, decimals=2)

        # --! to take stats more efficiently below - subtract setpoint from data
        timeseries_nom = self._subtract_setpoint(timeseries_nom)
        timeseries_exc = self._subtract_setpoint(timeseries_exc)

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

    def mask(self, timeseries):
        """Determines a data mask that differentiates between nominal and excursion data."""

        # --! note that there is temporary setpoint subtraction from given time series
        mask = self._compute_state_norm_max(self._extract_state(timeseries) - self.setpoint) < self.regime_sep
        return torch.squeeze(mask)

    def normalize(self, timeseries):
        assert timeseries.shape[-1]==(self.state_ndim + self.control_ndim + self.mask_ndim)

        mask = self.mask(timeseries)

        timeseries = self._subtract_setpoint(timeseries)
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

        state = state - self.setpoint
        norm_state = torch.empty_like(state)
        norm_state[mask] = (state[mask] - self.mean_nom) / self.std_nom
        norm_state[~mask] = (state[~mask] - self.mean_exc) / self.std_exc

        return norm_state, mask

    def denormalize(self, timeseries, mask):

        assert timeseries.shape[-1]==self.state_ndim

        denorm_timeseries = torch.empty_like(timeseries)

        denorm_timeseries[mask] = timeseries[mask] * self.std_nom + self.mean_nom
        denorm_timeseries[~mask] = timeseries[~mask] * self.std_exc + self.mean_exc

        denorm_timeseries = denorm_timeseries + self.setpoint

        return denorm_timeseries

    def denormalize_action(self, action, mask):

        assert action.shape[-1]==self.control_ndim

        denorm_action = torch.empty_like(action)

        denorm_action[mask] = action[mask] * self.std_nom + self.mean_nom
        denorm_action[~mask] = action[~mask] * self.std_exc + self.mean_exc

        return denorm_action

    def _subtract_setpoint(self, timeseries):
        state, other = torch.split(timeseries, [self.state_ndim, self.control_ndim + self.mask_ndim], dim=-1)
        state = state - self.setpoint

        return torch.cat([state, other], dim=-1)

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
                 batch_size, window_nsample, setpoint, load_normalized=True):
        super().__init__(file_dir, file_name, file_ext,
                         data_nsample, data_split_size,
                         batch_size, window_nsample, setpoint, load_normalized)

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

        return normalizer(timeseries_nom, timeseries_exc, self.setpoint, self.state_ndim, self.control_ndim, self.mask_ndim)


class replay_factory(rl.replay_factory):

    state_ndim = 2
    action_ndim = 1
    mask_ndim = 1

    def __init__(self):
        pass

    def create(self, env, policy, zeta, state_nsample, skip_nsample):

        # --! create an empty replay buffer
        buf = rl.replay()

        # --! state is represented by a window of recent observations and actions
        #
        # --! a deque allows to push new data in,
        # --! which automatically pops out old data once deque's capacity is filled
        sa_window = deque(maxlen=state_nsample)

        # --! reset environment to begin replay from the first observation
        s = env.reset()

        # --! skip specified number of samples
        for _ in range(skip_nsample):
            a = policy.base(s)
            sa_window.append((s, a))
            s, reward, done = env.step(a)

        # --! fill replay with observations, rewards, etc.
        while not done:

            # --! encode state at time t from a window of recent observations
            state = self.encode_state(sa_window)

            # --! based on observation at time t + 1, compute action at time t + 1
            a = policy.base(s)

            # --! provided residual policy is available, add residual action to the base one
            if policy.residual is not None:
                a = a + torch.squeeze(policy.residual(s, zeta=zeta), 0)

            # --! update window with observation and action at time t + 1, and encode next state
            sa_window.append((s, a))
            next_state = self.encode_state(sa_window)

            # --! replay buffer receives:
            #
            # --! state at time t
            # --! reward at time t
            # --! state at time t + 1
            # --! done flag at time t
            buf.add(
                state,
                reward,
                next_state,
                done
            )

            # --! make environment step to get next observations, rewards, etc.
            s, reward, done = env.step(a)

        return buf

    def encode_state(self, sa_window):

        # --! unpack given deque into sa tuples, and then zip all s's together and all a's together
        s, a = zip(*sa_window)

        # --! concatenate zipped s tuples into torch tensor, do the same for a
        s = torch.cat(s, dim=0)
        a = torch.cat(a, dim=0)

        # --! concatenate s, a and mask tensors and return result as a 3D tensor
        return torch.unsqueeze(torch.cat([s, a, torch.ones_like(a)], dim=-1), 0)

    def extract_current_s(self, state):
        s, other = torch.split(state, [self.state_ndim, self.action_ndim + self.mask_ndim], dim=-1)
        return s[:, [-1]]

    def extract_current_a(self, state):
        s, a, mask = torch.split(state, [self.state_ndim, self.action_ndim, self.mask_ndim], dim=-1)
        return a[:, [-1]]

    def update_current_a(self, state, a):
        state[:, -2:, [2]] = a

    def update_state(self, state, s):

        # --! make a dummy (zero) action
        a = torch.zeros(s.shape[0], s.shape[1], 1)

        # --! concatenate state-action pair and mask
        sa = torch.cat([s, a, torch.ones_like(a)], dim=-1)

        # --! shift in new state-action pair from the right
        return torch.cat([state[:, 1:], sa], dim=1)

