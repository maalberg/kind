# --! example: Duffing oscillator --!

import numpy as np
import scipy as sp

import os

import torch

import random
from scipy import signal
from scipy.integrate import solve_ivp
from collections import deque

import util_data
import util_nn
import reinforcement_learning


class reward:
    def __init__(self, q, r, setpoint):
        self.q = np.atleast_2d(q)
        self.r = np.atleast_2d(r)
        self.setpoint = np.atleast_2d(setpoint)

    def __call__(self, state, action):
        x_err = state - self.setpoint

        state_cost = x_err @ self.q @ x_err.T
        action_cost = action @ self.r @ action.T

        return -(state_cost + action_cost)


class reward_adapter:
    def __init__(self, q, r, setpoint, device=None, dtype=torch.float32):

        self.q = torch.atleast_2d(torch.as_tensor(q, dtype=dtype, device=device))
        self.r = torch.atleast_2d(torch.as_tensor(r, dtype=dtype, device=device))
        self.setpoint = torch.atleast_2d(torch.as_tensor(setpoint, dtype=dtype, device=device))

    def __call__(self, state, action):

        state = torch.atleast_2d(state)
        action = torch.atleast_2d(action)

        x_err = state - self.setpoint

        state_cost = x_err @ self.q @ x_err.transpose(-1, -2)
        action_cost = action @ self.r @ action.transpose(-1, -2)

        return -(state_cost + action_cost)


def duffing_update(t, state, sim, u):

    x1, x2 = state
    dx1 = x2
    dx2 = -sim.delta * x2 - sim.alpha * x1 - sim.beta * x1**3 + u + sim.gamma * np.cos(sim.omega * t)
    return [dx1, dx2]


class environment(reinforcement_learning.environment):
    """Implements a Duffing oscillator using numpy."""

    def __init__(
        self,
        reward_fn,
        beta=20.0, gamma=10.0, alpha=-1.0, delta=0.5, omega=1.2,
        dt_sim=1e-4, dt_control=2e-2, t_end=100.0):

        self.alpha = alpha
        self.beta  = beta
        self.gamma = gamma
        self.delta = delta
        self.omega = omega

        self.x = None
        self.dx = None

        self.r_fn = reward_fn

        # --! define simulation timing
        self.dt_sim = dt_sim
        self.dt_control = dt_control

        self.step_cnt = 0
        self.nstep = int(t_end / self.dt_control)

    def reset(self, ic=[0.0, 0.0]):

        # --! reset the state to the initial condition
        self.x = ic[0]
        self.dx = ic[1]

        # --! reset step counter
        self.step_cnt = 0

        return np.array([[self.x, self.dx]])

    def step(self, action):

        # --! based on current state and action, calculate reward 
        state = np.array([[self.x, self.dx]])
        reward = self.r_fn(state, np.array([[action]]))

        # --! for integration below, define step timing
        t_start = self.step_cnt * self.dt_control
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

        self.step_cnt += 1
        done = self.step_cnt == self.nstep

        return next_state, reward, done

    def replay(self, ic, policy, obs_nsample=1, skip_nsample=0):
        raise NotImplementedError()

    @property
    def reward_fn(self):
        return self.r_fn


class environment_adapter(reinforcement_learning.environment):

    def __init__(self, env):
        self.env = env

        reward_fn = env.reward_fn
        self.r_fn = reward_adapter(reward_fn.q, reward_fn.r, reward_fn.setpoint)

    def reset(self, ic=torch.zeros(1,1,2)):
        ic = ic.detach().cpu().numpy()
        obs = self.env.reset(np.squeeze(ic))
        return torch.from_numpy(obs).to(dtype=torch.float32)

    def step(self, action):
        action = action.detach().cpu().numpy()

        next_state, reward, done = self.env.step(np.squeeze(action))

        next_state = torch.from_numpy(next_state).to(dtype=torch.float32)
        reward = torch.tensor(reward).to(dtype=torch.float32)
        done = torch.tensor(done)

        return next_state, reward, done

    def replay(self, ic, policy, obs_nsample=1, skip_nsample=0):

        # --! reset environment to begin replay from initial condition
        s = self.reset(ic)

        # --! done flag for sanity checks
        done = False

        # --! skip specified number of samples
        for k in range(skip_nsample):

            # --! compute action
            #
            # --! provided residual policy is available, add residual action to the base one
            a = policy.base(s)
            if policy.residual is not None:
                a = a + torch.squeeze(policy.residual(s), 0)

            s, reward, done = self.step(a)

            if done: break

        if done: return None

        # --! make a window for recent observations and actions
        #
        # --! a deque allows to push new data in,
        # --! which automatically pops out old data once deque's capacity is filled
        sa_window = deque(maxlen=obs_nsample)

        # --! fill initial window
        for _ in range(obs_nsample):

            # --! compute action
            #
            # --! provided residual policy is available, add residual action to the base one
            a = policy.base(s)
            if policy.residual is not None:
                a = a + torch.squeeze(policy.residual(s), 0)

            sa_window.append((s, a))
            s, reward, done = self.step(a)

            if done: break

        # --! make an empty replay buffer
        buf = replay()

        # --! fill replay with observation windows, rewards, etc.
        while not done:

            # --! encode state at time t from a window of recent observations
            state = buf.util.encode_obs(sa_window)

            # --! based on observation at time t + 1, compute action at time t + 1
            #
            # --! provided residual policy is available, add residual action to the base one
            a = policy.base(s)
            if policy.residual is not None:
                a = a + torch.squeeze(policy.residual(s), 0)

            # --! update window with observation and action at time t + 1, and encode next state
            sa_window.append((s, a))
            next_state = buf.util.encode_obs(sa_window)

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
            s, reward, done = self.step(a)

        return buf

    @property
    def reward_fn(self):
        return self.r_fn


class replay(reinforcement_learning.replay):
    def __init__(self, buffer=None):
        super().__init__(buffer)
        self._util = replay_util()

    @property
    def util(self):
        return self._util


class replay_util(reinforcement_learning.replay_util):

    obs_ndim = 2
    action_ndim = 1
    mask_ndim = 1

    def encode_obs(self, sa):

        # --! unpack given deque into sa tuples, and then zip all s's together and all a's together
        s, a = zip(*sa)

        # --! concatenate zipped s tuples into torch tensor, do the same for a
        s = torch.cat(s, dim=0)
        a = torch.cat(a, dim=0)

        # --! concatenate s, a and mask tensors and return result as a 3D tensor
        return torch.unsqueeze(torch.cat([s, a, torch.ones_like(a)], dim=-1), 0)

    def replay_obs(self, env, env_ic, policy, obs_nsample):
        """Replays and encodes observation of length ``obs_nsamples`` starting from initial condition ``env_ic``.
        It is expected that ``env_ic`` is a batch of initial conditions."""

        # --! process a batch of initial conditions
        return torch.cat([self._replay_obs_nonbatch(env, torch.unsqueeze(ic, 0), policy, obs_nsample) for ic in env_ic], dim=0)

    def _replay_obs_nonbatch(self, env, env_ic, policy, obs_nsample):

        # --! make a window for observations s and actions a
        #
        # --! a deque allows to push new data in,
        # --! which automatically pops out old data once deque's capacity is filled
        sa_window = deque(maxlen=obs_nsample)

        # --! reset environment to begin replay from initial condition
        s = env.reset(env_ic)

        done = False
        for _ in range(obs_nsample):

            # --! compute action a from observation s
            #
            # --! provided residual policy is available, add residual action to the base one
            a = policy.base(s)
            if policy.residual is not None:
                a = a + torch.squeeze(policy.residual(s), 0)

            sa_window.append((s, a))
            s, reward, done = env.step(a)

            if done: break
        if done: return None

        return self.encode_obs(sa_window)

    def get_s0(self, encoded_obs):
        """Gets initial, i.e. first, observation s from encoded observation ``encoded_obs``."""
        s, other = torch.split(encoded_obs, [self.obs_ndim, self.action_ndim + self.mask_ndim], dim=-1)
        return s[:, :1]

    def get_s(self, encoded_obs):
        """Gets 'current', i.e. last, observation s from encoded observation ``encoded_obs``."""
        s, other = torch.split(encoded_obs, [self.obs_ndim, self.action_ndim + self.mask_ndim], dim=-1)
        return s[:, [-1]]

    def get_a(self, encoded_obs):
        """Gets 'current', i.e. last, action a from encoded observation ``encoded_obs``."""
        s, a, mask = torch.split(encoded_obs, [self.obs_ndim, self.action_ndim, self.mask_ndim], dim=-1)
        return a[:, [-1]]

    def update_s(self, encoded_obs, s):

        # --! make a dummy (zero) action
        a = torch.zeros(s.shape[0], s.shape[1], 1)

        # --! concatenate state-action pair and mask
        sa = torch.cat([s, a, torch.ones_like(a)], dim=-1)

        # --! shift in new state-action pair from the right
        return torch.cat([encoded_obs[:, 1:], sa], dim=1)

    def update_a(self, encoded_obs, a):
        encoded_obs[:, -2:, [2]] = a


class base_policy:

    def __init__(self, gain, setpoint):
        self.gain = torch.from_numpy(gain).to(torch.float32)
        self.setpoint = torch.atleast_2d(torch.tensor(setpoint))

    def __call__(self, obs):
        obs = obs - self.setpoint
        return -torch.matmul(obs, torch.transpose(self.gain, 0, 1))


def make_base_policy(duffing_alpha, duffing_delta, q, r, dt=1e-2, setpoint=[0.0, 0.0]):

    a = np.array([
        [ 0,      1    ],
        [-duffing_alpha, -duffing_delta],
    ])

    b = np.array([
        [ 0    ],
        [ 1    ],
    ])

    # --! discretize a continuous-time state-space system
    sys = signal.StateSpace(a, b, np.eye(2), np.zeros((2, 1)))
    sys = sys.to_discrete(dt)

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

    # --! return base policy, and the solution to Riccati equation
    return base_policy(k, setpoint), p


def normalize_standard(timeseries, mean, std):
    return (timeseries - mean) / std


def denormalize_standard(timeseries, mean, std):
    return timeseries * std + mean


class normalizer(util_data.normalizer):

    def __init__(self, timeseries, setpoint, state_ndim, action_ndim, mask_ndim):

        # --! save data dimensions
        self.state_ndim = state_ndim
        self.action_ndim = action_ndim
        self.mask_ndim = mask_ndim

        self.setpoint = setpoint

        state, action, _ = torch.split(timeseries, [self.state_ndim, self.action_ndim, self.mask_ndim], dim=-1)
        state = state - self.setpoint
        
        # --! take statistics
        self.s_mean = [s.mean() for s in torch.split(state, 1, dim=-1)]
        self.s_std = [torch.maximum(s.std(), self.std_min) for s in torch.split(state, 1, dim=-1)]
        self.a_mean = [a.mean() for a in torch.split(action, 1, dim=-1)]
        self.a_std = [torch.maximum(a.std(), self.std_min) for a in torch.split(action, 1, dim=-1)]

    def normalize(self, timeseries):

        nfeature = timeseries.shape[-1]
        assert nfeature==(self.state_ndim + self.action_ndim + self.mask_ndim) or nfeature==self.state_ndim

        if nfeature==self.state_ndim:
            timeseries = self._normalize_state(timeseries)
        else:
            timeseries = self._normalize_timeseries(timeseries)

        return timeseries

    def _normalize_state(self, state):
        assert state.shape[-1]==self.state_ndim

        state = state - self.setpoint

        return torch.cat([
            normalize_standard(s, mean, std) for s, mean, std in zip(torch.split(state, 1, dim=-1), self.s_mean, self.s_std)], dim=-1)

    def _normalize_timeseries(self, timeseries):
        assert timeseries.shape[-1]==(self.state_ndim + self.action_ndim + self.mask_ndim)

        state, action, action_mask = torch.split(timeseries, [self.state_ndim, self.action_ndim, self.mask_ndim], dim=-1)

        state = self._normalize_state(state)
        action = torch.cat([
            normalize_standard(a, mean, std) for a, mean, std in zip(torch.split(action, 1, dim=-1), self.a_mean, self.a_std)], dim=-1)

        return torch.cat([state, action, action_mask], dim=-1)

    def denormalize(self, timeseries):

        nfeature = timeseries.shape[-1]
        assert nfeature==self.state_ndim or nfeature==self.action_ndim

        if nfeature==self.state_ndim:
            state = torch.cat([
                denormalize_standard(
                    s, mean, std) for s, mean, std in zip(torch.split(timeseries, 1, dim=-1), self.s_mean, self.s_std)], dim=-1)
            timeseries = state + self.setpoint
        else:
            timeseries = torch.cat([
                denormalize_standard(
                    s, mean, std) for s, mean, std in zip(torch.split(timeseries, 1, dim=-1), self.a_mean, self.a_std)], dim=-1)

        return timeseries


class dataset(util_data.dataset):

    state_ndim = 2
    action_ndim = 1
    mask_ndim = 1

    def __init__(self,
                 file_dir, file_name, file_index, file_ext,
                 data_nsample_nom, data_nsample_exc,
                 data_split_size,
                 batch_size, window_nsample, setpoint, load_normalized=True):
        super().__init__(file_dir, file_name, file_index, file_ext,
                         data_nsample_nom, data_nsample_exc, data_split_size,
                         batch_size, window_nsample, setpoint, load_normalized)

    def make_path(self, data_type='nom'):
        filename = f'{self.file_name}_{data_type}_{self.file_index}{self.file_ext}'
        return os.path.join(self.file_dir, filename)

    def extract_target(self, window):
        """Extracts the first two feature dimensions: position and velocity."""
        return window[:, :, :self.state_ndim]

    def init_normalization(self):

        # --! read data
        timeseries = self.read_timeseries(self.make_path(data_type='all'), 619)

        return normalizer(timeseries, self.setpoint, self.state_ndim, self.action_ndim, self.mask_ndim)


class dataset_factory(reinforcement_learning.dataset_factory):

    def __init__(self, setpoint):
        self.setpoint = setpoint
        self.normalizer = None

    def create_dataset(self, args, load_normalized=True):

        ds = dataset(
            args.file_dir, args.file_name, args.file_index, args.file_ext,
            args.data_nsample,
            (args.data_train_size, args.data_test_size),
            args.batch_size, (args.lookback_nsample, args.forecast_nsample), self.setpoint, load_normalized=load_normalized)

        self.normalizer = ds.normalizer
        return ds

    def create_normalizer(self, args, load_normalized=False):
        if self.normalizer is not None:
            print('using available normalizer')
            return self.normalizer

        ds = dataset(
            args.file_dir, args.file_name, args.file_index, args.file_ext,
            args.data_nsample,
            (args.data_train_size, args.data_test_size),
            args.batch_size, (args.lookback_nsample, args.forecast_nsample), self.setpoint, load_normalized=load_normalized)

        print('creating new normalizer')
        return ds.normalizer

