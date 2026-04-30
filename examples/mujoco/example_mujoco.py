import os
import numpy as np
import torch
import torch.nn.functional as F
import random

from matplotlib import pyplot as plt

import util_data
import reinforcement_learning

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


class normalizer(util_data.normalizer):

    def __init__(self, timeseries, setpoint, obs_ndim, act_ndim, mask_ndim):

        # --! save data dimensions
        self.obs_ndim = obs_ndim
        self.act_ndim = act_ndim
        self.mask_ndim = mask_ndim

        self.setpoint = setpoint

        obs, _ = torch.split(timeseries, [self.obs_ndim, self.act_ndim + self.mask_ndim], dim=-1)
        obs = obs - self.setpoint

        # --! take statistics
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


class baseline_dataset(torch.utils.data.Dataset):
    def __init__(self, states, actions, next_states):
        self.states = states
        self.actions = actions
        self.targets = next_states - states  # delta

    def __len__(self):
        return len(self.states)

    def __getitem__(self, idx):
        return self.states[idx], self.actions[idx], self.targets[idx]


class global_dynamics(torch.nn.Module):
    def __init__(self, obs_ndim, act_ndim, hidden_ndim=512):
        super().__init__()

        self.net = torch.nn.Sequential(
            torch.nn.Linear(obs_ndim + act_ndim, hidden_ndim),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_ndim, hidden_ndim),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_ndim, obs_ndim)  # predicts delta observation
        )

    def forward(self, s, a):
        """Returns predicted observation delta."""
        x = torch.cat([s, a], dim=-1)
        return self.net(x)


def train_global(model, dataloader, nepoch=1_000):
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = torch.nn.MSELoss()

    for epoch in range(nepoch):
        total_loss = 0.0

        for s, a, s_delta in dataloader:
            pred = model(s, a)
            loss = loss_fn(pred, s_delta)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        if epoch % 20 == 0:
            print(f"epoch {epoch}, loss: {total_loss / len(dataloader):.6f}")


def rollout_global(model, s0, obs, act, reset_nsample=1_000):
    s = s0
    traj = [s0]

    for j, a in enumerate(act):
        if j % reset_nsample == 0:
            s = obs[j]
        delta = model(s.unsqueeze(0), a.unsqueeze(0)).squeeze(0)
        s = s + delta
        traj.append(s)

    return torch.stack(traj)


class stochastic_dynamics(torch.nn.Module):
    def __init__(self, obs_ndim, act_ndim, hidden_ndim=512):
        super().__init__()

        self.net = torch.nn.Sequential(
            torch.nn.Linear(obs_ndim + act_ndim, hidden_ndim),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_ndim, hidden_ndim),
            torch.nn.ReLU(),
        )

        self.mean = torch.nn.Linear(hidden_ndim, obs_ndim)
        self.logvar = torch.nn.Linear(hidden_ndim, obs_ndim)

        # --! PETS trick: learned bounds
        self.max_logvar = torch.nn.Parameter(torch.ones(obs_ndim) * 0.5)
        self.min_logvar = torch.nn.Parameter(torch.ones(obs_ndim) * -10)

    def forward(self, s, a):
        x = torch.cat([s, a], dim=-1)
        h = self.net(x)

        mean = self.mean(h)
        logvar = self.logvar(h)

        # --! variance clipping
        logvar = self.max_logvar - F.softplus(self.max_logvar - logvar)
        logvar = self.min_logvar + F.softplus(logvar - self.min_logvar)

        return mean, logvar


class model_ensemble:
    def __init__(self, nmodel, obs_ndim, act_ndim):
        self.models = [
            stochastic_dynamics(obs_ndim, act_ndim)
            for _ in range(nmodel)
        ]

    def parameters(self):
        params = []
        for m in self.models:
            params += list(m.parameters())
        return params

    def eval(self):
        for m in self.models:
            m.eval()


def compute_stochastic_loss(mean, logvar, target):
    """Computes Gaussian negative log likelihood."""
    inv_var = torch.exp(-logvar)
    mse = (mean - target) ** 2
    return torch.mean(mse * inv_var + logvar)


def train_ensemble(ensemble, dataloaders, nepoch=50):

    optimizers = [
        torch.optim.Adam(m.parameters(), lr=1e-3, weight_decay=1e-8)
        for m in ensemble.models
    ]

    for epoch in range(nepoch):
        model_losses = [np.zeros(1) for _ in range(len(ensemble.models))]

        for model, dataloader, opt, model_loss in zip(ensemble.models, dataloaders, optimizers, model_losses):
            for s, a, target in dataloader:
                opt.zero_grad()

                mean, logvar = model(s, a)
                loss = compute_stochastic_loss(mean, logvar, target)

                loss.backward()
                opt.step()

                model_loss[0] += loss.item()

        for model_loss in model_losses:
            model_loss[0] /= len(ensemble.models)
        if epoch % 10 == 0:
            print(epoch, model_losses)


def step_ensemble_deterministic(ensemble, s, a):
    preds = []

    for m in ensemble.models:
        mean, _ = m(s, a)
        preds.append(mean)

    mean = torch.stack(preds).mean(0)
    return s + mean


def step_ensemble_stochastic(ensemble, s, a):

    m = random.choice(ensemble.models)

    mean, logvar = m(s, a)

    std = torch.exp(0.5 * logvar)
    eps = torch.randn_like(std)

    delta = mean + eps * std

    return s + delta


def rollout_ensemble(ensemble, s0, obs, act, deterministic=True, reanchor_nsample=1_000):
    states = [s0]
    s = s0

    for j, a in enumerate(act):
        if j % reanchor_nsample == 0:
            s = obs[j]
        if deterministic:
            s = step_ensemble_deterministic(ensemble, s, a)
        else:
            s = step_ensemble_stochastic(ensemble, s, a)

        states.append(s)

    return torch.stack(states)


class expert_dynamics(torch.nn.Module):
    def __init__(self, i_ndim, o_ndim, hidden_ndim=512):
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Linear(i_ndim, hidden_ndim),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_ndim, hidden_ndim),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_ndim, o_ndim)
        )

    def forward(self, x):
        return self.net(x)


class model_moe(torch.nn.Module):
    def __init__(self, obs_ndim, act_ndim, nexpert=3, smoothing=0.9, gumbel=1.0):
        super().__init__()

        self.i_ndim = obs_ndim + act_ndim
        self.o_ndim = obs_ndim
        self.nexpert = nexpert
        self.smoothing = smoothing
        self.gumbel = gumbel

        # --! experts
        self.experts = torch.nn.ModuleList([
            expert_dynamics(self.i_ndim, self.o_ndim)
            for _ in range(nexpert)
        ])

        # --! gating network
        self.gate = torch.nn.Sequential(
            torch.nn.Linear(self.i_ndim, 128),
            torch.nn.ReLU(),
            torch.nn.Linear(128, nexpert)
        )

        # --! buffer for temporal smoothing (NOT part of gradient graph)
        self.register_buffer("prev_logits", None)

    def reset(self):
        """Call at start of each rollout/sequence"""
        self.prev_logits = None

    def forward(self, s, a):
        x = torch.cat([s, a], dim=-1)

        logits = self.gate(x)

        # --! temporal smoothing (safe)
        if self.prev_logits is not None and self.smoothing is not None:
            logits = self.smoothing * self.prev_logits + (1 - self.smoothing) * logits

        # --! store detached version (IMPORTANT)
        with torch.no_grad():
            self.prev_logits = logits.detach()

        # --! gating
        if self.gumbel is not None:
            weights = F.gumbel_softmax(logits, tau=self.gumbel, hard=True)
        else:
            weights = F.softmax(logits, dim=-1)

        # --! expert outputs
        outputs = torch.stack([expert(x) for expert in self.experts], dim=-1)

        # --! weighted combination
        weights_expanded = weights.unsqueeze(-2)
        out = (outputs * weights_expanded).sum(dim=-1)

        return out, weights


def train_moe(model, dataloader, nepoch=100, ent_coef=0.01):

    opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-8)
    losses = []
    for epoch in range(nepoch):
        total_loss = 0.0
        for s, a, target in dataloader:
            opt.zero_grad()

            pred, weights = model(s, a)

            mse = F.mse_loss(pred, target)
            entropy = -(weights * torch.log(weights + 1e-8)).sum(dim=-1)
            entropy = entropy.mean()
            loss = mse - ent_coef * entropy

            loss.backward()
            opt.step()

            total_loss += loss.item()

        if epoch % 20 == 0:
            print(f"epoch {epoch}, loss: {total_loss / len(dataloader):.6f}")


def rollout_moe(model, s0, obs, act, reanchor_nsample=1_000):
    s = s0
    traj = [s0]

    for j, a in enumerate(act):
        if j % reanchor_nsample == 0:
            s = obs[j]
        delta, _ = model(s.unsqueeze(0), a.unsqueeze(0))
        delta = delta.squeeze(0)
        s = s + delta
        traj.append(s)

    return torch.stack(traj)


def rollout_kind(model, traj_true, horizon=1, reset_nsample=20, offset=0):

    args = model.args

    # --! we use this dummy replay just to get access to replay utilities
    dummy_replay = replay(s_ndim=args.obs_ndim, a_ndim=args.act_ndim)

    next_ss = []
    alphas = []
    means_nom = []
    means_exc = []
    zetas_nom = []
    zetas_exc = []

    back = traj_true[:, offset:offset + args.back_nsample].clone()
    true = traj_true[:, offset + args.back_nsample:offset + args.back_nsample + horizon]

    with torch.no_grad():

        for k in range(horizon):
            if k % reset_nsample == 0:
                t2 = offset + args.back_nsample + k
                t1 = t2 - args.back_nsample
                back = traj_true[:, t1:t2].clone()

            s = dummy_replay.util.get_s(back)
            t = offset + k + args.back_nsample - 1
            a = traj_true[:, [t], -(dummy_replay.util.a_ndim + 1):]

            dummy_replay.util.update_a(back, a)
            model_o = model(back)

            fore = model_o.blend[:, args.back_nsample:]
            next_s = fore[:, :1]
            back = dummy_replay.util.shift_obs(back, next_s)

            alpha = model_o.alpha[:, args.back_nsample:]
            alpha = alpha[:, :1]

            zeta_nom = model_o.zeta_nom[:, args.back_nsample:]
            zeta_nom = zeta_nom[:, :1]

            zeta_exc = model_o.zeta_exc[:, args.back_nsample:]
            zeta_exc = zeta_exc[:, :1]

            mean_nom = model_o.mean_nom[:, args.back_nsample:]
            mean_nom = mean_nom[:, :1]

            mean_exc = model_o.mean_exc[:, args.back_nsample:]
            mean_exc = mean_exc[:, :1]

            next_ss.append(next_s)
            alphas.append(alpha)
            zetas_nom.append(zeta_nom)
            zetas_exc.append(zeta_exc)
            means_nom.append(mean_nom)
            means_exc.append(mean_exc)

        next_ss = torch.cat(next_ss, dim=1)
        alphas = torch.cat(alphas, dim=1)
        zetas_nom = torch.cat(zetas_nom, dim=1)
        zetas_exc = torch.cat(zetas_exc, dim=1)
        means_nom = torch.cat(means_nom, dim=1)
        means_exc = torch.cat(means_exc, dim=1)

    return true, next_ss, alphas, zetas_nom, zetas_exc, means_nom, means_exc


def disp_rollout(obs_true, obs_rollout, obs_mean, obs_std, this_traj=0, disp_end=300):

    with torch.no_grad():
        plot_rollout_traj = torch.unsqueeze(obs_rollout, 0)
        plot_rollout_traj = torch.cat([
            denormalize_standard(
                s, mean, std) for s, mean, std in zip(torch.split(plot_rollout_traj, 1, dim=-1), obs_mean, obs_std)], dim=-1)
        plot_obs = torch.cat([
            denormalize_standard(
                s, mean, std) for s, mean, std in zip(torch.split(obs_true, 1, dim=-1), obs_mean, obs_std)], dim=-1)

        plt.figure(figsize=(6,6))

        plt.subplot(2,1,1)
        plt.plot(plot_obs[this_traj, :disp_end, 0], label='z')
        plt.plot(plot_rollout_traj[0, :disp_end, 0])
        plt.ylim((-1,2))
        plt.legend()

        plt.subplot(2,1,2)
        plt.plot(plot_obs[this_traj, :disp_end, 9], label='dz')
        plt.plot(plot_rollout_traj[0, :disp_end, 9])
        plt.legend()

        plt.show()

    return plot_obs, plot_rollout_traj

