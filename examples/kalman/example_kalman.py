# --! example: function and class definitions for detuning

import os

import numpy as np
import scipy as sp

import torch

from filterpy.kalman import KalmanFilter as KF
from scipy.linalg import block_diag
import matplotlib.pyplot as plt

import reinforcement_learning
import util_data


def make_rf_a(f, q):
    w     = 2 * np.pi * f
    hbw   = w/2/q  # < half-bandwidth of an rf cavity in rad/s

    return np.array([
        [-hbw,     0.],
        [ 0.,    -hbw],
    ])


def make_rf_b(f, q):
    w     = 2 * np.pi * f
    hbw   = w/2/q  # < half-bandwidth of an rf cavity in rad/s

    return np.array([
        [hbw,    0.],
        [0.,    hbw],
    ])


def make_mm_a(f, q):
    """Constructs the matrix A of a single mechanical mode (mm)."""
    w = 2 * np.pi * f
    return np.array([
        [ 0,             1  ],
        [-np.square(w), -w/q],
    ])


def make_mm_b(f, k):
    """Constructs the matrix B of a single mechanical mode (mm)."""
    w = 2 * np.pi * f
    return np.array([
        [0               ],
        [k * np.square(w)],
    ])


def make_mm_c():
    return np.array([[1, 0]])


def make_mm_a_array(f, q):
    return block_diag(*[make_mm_a(f, q) for f, q in zip(f, q)])


def make_mm_b_array(f, k):
    return np.concatenate([make_mm_b(f, k) for f, k in zip(f, k)], axis=0)


def make_mm_c_array(nf=1):
    return np.tile(make_mm_c(), nf)


def filter_detuning(param):
    """ Creates a linear Kalman filter to filter a cavity detuning signal. """

    f = param['f']
    q = param['q']
    k = param['k']
    dt = param['dt']
    kalman_q = param['kalman_q']
    kalman_r = param['kalman_r']

    # --! create observed process matrices
    a = make_mm_a_array(f, q)
    b = make_mm_b_array(f, k)
    c = make_mm_c_array(len(f))

    # --! there two states (position and velocity) per one mechanical mode frequency
    nstate = 2 * len(f)
    nmeas = 1

    # --! create a linear Kalman filter
    kf = KF(dim_x=nstate, dim_z=nmeas)
    kf.F = sp.linalg.expm(a * dt)
    kf.B = b * dt
    kf.H = c

    # initial state and covariances
    kf.x = np.zeros((nstate, 1))
    kf.P *= 10. * np.eye(nstate)   # uncertainty about the initial condition

    kf.R *= kalman_r  # measurement noise
    kf.Q *= kalman_q  # process uncertainty

    return kf


def normalize_standard(timeseries, mean, std):
    return (timeseries - mean) / std


def denormalize_standard(timeseries, mean, std):
    return timeseries * std + mean


class normalizer(util_data.normalizer):

    def __init__(self, timeseries, setpoint, obs_ndim, act_ndim, mask_ndim):

        # --! save data dimensions
        self.obs_ndim = obs_ndim
        self.act_ndim = act_ndim
        self.mask_ndim = mask_ndim

        self.setpoint = setpoint

        # --! there is no action, so we extract only observation
        obs, _ = torch.split(timeseries, [self.obs_ndim, self.act_ndim + self.mask_ndim], dim=-1)
        obs = obs - self.setpoint

        # --! there is no action, so take the mean and variance of observation only
        self.s_mean = [s.mean() for s in torch.split(obs, 1, dim=-1)]
        self.s_std = [torch.maximum(s.std(), self.std_min) for s in torch.split(obs, 1, dim=-1)]

    def normalize(self, timeseries):

        nfeature = timeseries.shape[-1]
        assert nfeature==(self.obs_ndim + self.act_ndim + self.mask_ndim) or nfeature==self.obs_ndim

        if nfeature==self.obs_ndim:
            timeseries = self._normalize_state(timeseries)
        else:
            timeseries = self._normalize_timeseries(timeseries)

        return timeseries

    def _normalize_state(self, obs):
        assert obs.shape[-1]==self.obs_ndim

        obs = obs - self.setpoint

        return torch.cat([
            normalize_standard(s, mean, std) for s, mean, std in zip(torch.split(obs, 1, dim=-1), self.s_mean, self.s_std)], dim=-1)

    def _normalize_timeseries(self, timeseries):
        assert timeseries.shape[-1]==(self.obs_ndim + self.act_ndim + self.mask_ndim)

        obs, other = torch.split(timeseries, [self.obs_ndim, self.act_ndim + self.mask_ndim], dim=-1)
        obs = self._normalize_state(obs)

        return torch.cat([obs, other], dim=-1)

    def denormalize(self, timeseries):

        nfeature = timeseries.shape[-1]
        assert nfeature==self.obs_ndim

        obs = torch.cat([
            denormalize_standard(
                s, mean, std) for s, mean, std in zip(torch.split(timeseries, 1, dim=-1), self.s_mean, self.s_std)], dim=-1)
        timeseries = obs + self.setpoint

        return timeseries


class dataset(util_data.dataset):

    def __init__(self, args, setpoint, load_normalized=True, extract_windows=True):
        super().__init__(args, setpoint, load_normalized, extract_windows)

    def make_path(self, data_type='nom'):
        filename = f'{data_type}{self.args.file_ext}'
        return os.path.join(self.args.file_dir, filename)

    def extract_target(self, window):
        return window[:, :, :self.args.obs_ndim]

    def init_normalization(self):

        # --! read data
        timeseries = self.read_timeseries(self.make_path(data_type='baseline'), self.args.data_nsample_baseline)

        # --! create normalizer
        return normalizer(timeseries, self.setpoint, self.args.obs_ndim, self.args.act_ndim, self.args.mask_ndim)


class replay(reinforcement_learning.replay):
    def __init__(self, s_ndim=2, a_ndim=0, buffer=None):
        super().__init__(buffer)
        self._util = replay_util(s_ndim, a_ndim)

    @property
    def util(self):
        return self._util


class replay_util(reinforcement_learning.replay_util):

    def __init__(self, s_ndim, a_ndim):
        self.s_ndim = s_ndim
        self.a_ndim = a_ndim

    def encode_obs(self, sa):
        raise NotImplementedError()
        return

    def replay_obs(self, env, env_ic, policy, obs_nsample):
        raise NotImplementedError()
        return

    def _replay_obs_nonbatch(self, env, env_ic, policy, obs_nsample):
        raise NotImplementedError()
        return

    def get_s0(self, encoded_obs):
        raise NotImplementedError()
        return

    def get_s(self, encoded_obs):
        """Gets 'current', i.e. last, observation s from encoded observation ``encoded_obs``."""
        s = encoded_obs
        return s[:, [-1]]

    def get_a(self, encoded_obs):
        raise NotImplementedError()
        return

    def shift_obs(self, encoded_obs, s):

        # --! shift in new state-action pair from the right
        return torch.cat([encoded_obs[:, 1:], s], dim=1)

    def update_a(self, encoded_obs, a):
        raise NotImplementedError()
        return


def rollout_kf(ft, c, data, u):
    """Rollout a Kalman filter ``ft`` with ``c`` as system output matrix on given ``data`` and input ``u``.
    The ``data`` is expected to be shaped as [T, C], where T and C is the number
    of time steps and data dimensions, respectively."""
    detuning_est = []

    for j in range(data.shape[0]):
        # --! measurement
        z = data[j, 0]

        # --! kalman filter step
        ft.predict(u=u)
        ft.update(np.array([z]))

        # --! save estimated detuning
        ft_est = np.squeeze(c @ ft.x)
        detuning_est.append(ft_est)

    return np.array(detuning_est).reshape(-1, 1)


def plot_kf(data_true, data_est, title):
    plt.figure(figsize=(9, 3.5))

    plt.plot(data_true[:, 0], label='True detuning', color='tab:green', alpha=0.75, linestyle='solid')
    plt.plot(data_est[:, 0], label='KF estimated detuning', color='tab:blue', alpha=1., linestyle='dashed')
    plt.ylabel('Detuning [V]')
    plt.legend()
    plt.title(title)
    plt.tight_layout()

    plt.show()
