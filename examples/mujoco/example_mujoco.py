import os
import torch

import util_data
import reinforcement_learning

class dataset(util_data.dataset):

    obs_ndim = 11
    action_ndim = 3
    mask_ndim = 0

    def __init__(self, args, setpoint, load_normalized=True, extract_windows=True):
        super().__init__(args, setpoint, load_normalized, extract_windows)

    def make_path(self, data_type='nom'):
        filename = f'{self.args.file_name}_{data_type}_{self.args.file_index}{self.args.file_ext}'
        return os.path.join(self.args.file_dir, filename)

    def extract_target(self, window):
        return window[:, :, :self.obs_ndim]

    def init_normalization(self):

        # --! read data
        timeseries = self.read_timeseries(self.make_path(data_type='all'), self.args.data_nsample_all)

        # --! create normalizer
        return normalizer(timeseries, self.setpoint, self.obs_ndim, self.action_ndim, self.mask_ndim)


class normalizer(util_data.normalizer):

    def __init__(self, timeseries, setpoint, obs_ndim, action_ndim, mask_ndim):

        # --! save data dimensions
        self.obs_ndim = obs_ndim
        self.action_ndim = action_ndim
        self.mask_ndim = mask_ndim

        self.setpoint = setpoint

        obs, action, _ = torch.split(timeseries, [self.obs_ndim, self.action_ndim, self.mask_ndim], dim=-1)
        obs = obs - self.setpoint

        # --! take statistics
        self.s_mean = [s.mean() for s in torch.split(obs, 1, dim=-1)]
        self.s_std = [torch.maximum(s.std(), self.std_min) for s in torch.split(obs, 1, dim=-1)]
        self.a_mean = [a.mean() for a in torch.split(action, 1, dim=-1)]
        self.a_std = [torch.maximum(a.std(), self.std_min) for a in torch.split(action, 1, dim=-1)]

    def normalize(self, timeseries):

        nfeature = timeseries.shape[-1]
        assert nfeature==(self.obs_ndim + self.action_ndim + self.mask_ndim) or nfeature==self.state_ndim

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
        assert timeseries.shape[-1]==(self.obs_ndim + self.action_ndim + self.mask_ndim)

        obs, action, action_mask = torch.split(timeseries, [self.obs_ndim, self.action_ndim, self.mask_ndim], dim=-1)

        obs = self._normalize_state(obs)
        action = torch.cat([
            normalize_standard(a, mean, std) for a, mean, std in zip(torch.split(action, 1, dim=-1), self.a_mean, self.a_std)], dim=-1)

        return torch.cat([obs, action, action_mask], dim=-1)

    def denormalize(self, timeseries):

        nfeature = timeseries.shape[-1]
        assert nfeature==self.obs_ndim or nfeature==self.action_ndim

        if nfeature==self.obs_ndim:
            obs = torch.cat([
                denormalize_standard(
                    s, mean, std) for s, mean, std in zip(torch.split(timeseries, 1, dim=-1), self.s_mean, self.s_std)], dim=-1)
            timeseries = obs + self.setpoint
        else:
            timeseries = torch.cat([
                denormalize_standard(
                    s, mean, std) for s, mean, std in zip(torch.split(timeseries, 1, dim=-1), self.a_mean, self.a_std)], dim=-1)

        return timeseries


def normalize_standard(timeseries, mean, std):
    return (timeseries - mean) / std


def denormalize_standard(timeseries, mean, std):
    return timeseries * std + mean


class replay(reinforcement_learning.replay):
    def __init__(self, s_ndim=11, a_ndim=3, buffer=None):
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

        # --! unpack given deque into sa tuples, and then zip all s's together and all a's together
        s, a = zip(*sa)

        # --! concatenate zipped s tuples into torch tensor, do the same for a
        s = torch.cat(s, dim=0)
        a = torch.cat(a, dim=0)

        # --! concatenate s, a and return result as a 3D tensor
        return torch.unsqueeze(torch.cat([s, a], dim=-1), 0)

    def replay_obs(self, env, env_ic, policy, obs_nsample):
        pass

    def get_s0(self, encoded_obs):
        """Gets initial, i.e. first, observation s from encoded observation ``encoded_obs``."""
        s, other = torch.split(encoded_obs, [self.s_ndim, self.a_ndim], dim=-1)
        return s[:, :1]

    def get_s(self, encoded_obs):
        """Gets 'current', i.e. last, observation s from encoded observation ``encoded_obs``."""
        s, other = torch.split(encoded_obs, [self.s_ndim, self.a_ndim], dim=-1)
        return s[:, [-1]]

    def get_a(self, encoded_obs):
        """Gets 'current', i.e. last, action a from encoded observation ``encoded_obs``."""
        s, a = torch.split(encoded_obs, [self.s_ndim, self.a_ndim], dim=-1)
        return a[:, [-1]]

    def shift_obs(self, encoded_obs, s):

        # --! make a dummy (zero) action
        a = torch.zeros(s.shape[0], s.shape[1], self.a_ndim)

        # --! concatenate state-action pair
        sa = torch.cat([s, a], dim=-1)

        # --! shift in new state-action pair from the right
        return torch.cat([encoded_obs[:, 1:], sa], dim=1)

    def update_a(self, encoded_obs, a):
        encoded_obs[:, -2:, -(self.a_ndim + 1):] = a
