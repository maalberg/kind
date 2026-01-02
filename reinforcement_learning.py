
from abc import abstractmethod
from abc import ABC as interface

from collections import deque
from collections import namedtuple

import torch

import util_nn

from matplotlib import pyplot as plt


class policy_iteration:
    def __init__(self, model, base_policy):

        self.model = model

        self.base_policy = base_policy
        self.res_policy = residual_policy()

        value_fn_ni = model.args.target_ndim
        value_fn_no = 1
        self.value_fn_nom = util_nn.fcnn(feat=[value_fn_ni, 64, 64, value_fn_no], actfun_hid='relu')
        self.value_fn_exc = util_nn.fcnn(feat=[value_fn_ni, 64, 64, value_fn_no], actfun_hid='relu')

        self.state_evaluate = iteration_evaluate(self)
        self.state_improve = iteration_improve(self)
        self.state = self.state_evaluate

    def iterate(self, replay_nom, replay_exc, dataset):
        self._get_state().iterate(replay_nom, replay_exc, dataset)

    def _get_state_evaluate(self):
        return self.state_evaluate

    def _get_state_improve(self):
        return self.state_improve

    def _get_state(self):
        return self.state

    def _set_state(self, state):
        self.state = state


class iteration_state(interface):

    @abstractmethod
    def iterate(self, replay_nom, replay_exc, dataset):
        return


class iteration_evaluate(iteration_state):

    def __init__(self, iteration):
        self.iter = iteration

    def iterate(self, replay_nom, replay_exc, dataset):

        loss_nom = self._evaluate_policy(self.iter.value_fn_nom, replay_nom, dataset)
        loss_exc = self._evaluate_policy(self.iter.value_fn_exc, replay_exc, dataset)

        self.iter.value_fn_nom.eval()
        self.iter.value_fn_exc.eval()

        lookback, reward, next_state, done = map(torch.cat, zip(*replay_nom.buffer))
        lookback = dataset.normalize(lookback, data_type='nom')
        obs = replay_nom.extract_current_state(lookback)
        obs_norm_nom = torch.squeeze(torch.linalg.norm(obs, dim=-1, ord=2))
        value_nom = torch.squeeze(self.iter.value_fn_nom(obs))

        lookback, reward, next_state, done = map(torch.cat, zip(*replay_exc.buffer))
        lookback = dataset.normalize(lookback, data_type='exc')
        obs = replay_exc.extract_current_state(lookback)
        obs_norm_exc = torch.squeeze(torch.linalg.norm(obs, dim=-1, ord=2))
        value_exc = torch.squeeze(self.iter.value_fn_exc(obs))

        with torch.no_grad():

            plt.figure(figsize=(6,10))

            plt.subplot(4,1,1)
            plt.plot(loss_nom)
            plt.xlabel('epochs')
            plt.ylabel('nominal loss')

            plt.subplot(4,1,2)
            plt.scatter(obs_norm_nom, value_nom)
            plt.scatter(obs_norm_nom[0], value_nom[0])
            plt.xlabel('state norm')
            plt.ylabel('nominal value')

            plt.subplot(4,1,3)
            plt.plot(loss_exc)
            plt.xlabel('epochs')
            plt.ylabel('excursion loss')

            plt.subplot(4,1,4)
            plt.scatter(obs_norm_exc, value_exc)
            plt.scatter(obs_norm_exc[0], value_exc[0])
            plt.xlabel('state norm')
            plt.ylabel('excursion value')

            plt.tight_layout()
            plt.show()

        self.iter._set_state(self.iter._get_state_improve())

    def _evaluate_policy(self, value_fn, replay, dataset):

        nepoch = 100
        batch_size = 128
        gamma = 0.96

        learning_rate = 1e-3
        weight_decay = 1e-5
        value_optim = torch.optim.Adam(value_fn.parameters(), lr=learning_rate, weight_decay=weight_decay)

        losses = []

        value_fn.train()

        for epoch in range(nepoch):
            value_optim.zero_grad()

            # --! sample a random batch
            lookback, reward, next_lookback, done = replay.random_batch(batch_size)

            # --! determine a data mask that differentiates between nominal and excursion batch elements
            with torch.no_grad():
                mask_nom = replay.get_coarse_zeta(lookback) < replay.coarse_zeta_threshold()
                mask_nom = torch.squeeze(mask_nom)

            # --! prepare data: normalize current states
            with torch.no_grad():
                obs = replay.extract_current_state(lookback)
                next_obs = replay.extract_current_state(next_lookback)

                obs = dataset.normalize_masked(obs, mask_nom)
                next_obs = dataset.normalize_masked(next_obs, mask_nom)

            # --! target must be treated as a constant, so restrict any gradient flow during target calculation
            with torch.no_grad():
                target = reward + gamma * (1.0 - done) * value_fn(next_obs)

            # --! compute the value of the current state (observation)
            value = value_fn(obs)

            criterion = torch.nn.MSELoss()
            loss = criterion(value, target)

            losses.append(loss.item())
            loss.backward()
            value_optim.step()

        return losses


class iteration_improve(iteration_state):

    def __init__(self, iteration):
        self.iter = iteration

    def iterate(self, replay_nom, replay_exc, dataset):
        self._improve_policy(dataset, replay_nom)
        self.iter._set_state(self.iter._get_state_evaluate())

    def _improve_policy(self, dataset, replay):

        value_fn_nom = self.iter.value_fn_nom
        value_fn_exc = self.iter.value_fn_exc

        res_policy = self.iter.res_policy
        base_policy = self.iter.base_policy

        model = self.iter.model

        gamma = 0.96
        batch_size = 16
        learning_rate = 1e-3
        weight_decay = 1e-5
        policy_optim = torch.optim.Adam(res_policy.parameters(), lr=learning_rate, weight_decay=weight_decay)

        losses = []

        # --! freeze KIND
        model.eval()
        for p in model.parameters():
            p.requires_grad = False

        # --! freeze value functions
        value_fn_nom.eval()
        value_fn_exc.eval()

        # --! train policy
        res_policy.train()

        # --! Assume:
        #
        # --! value_fn frozen
        # --! model frozen
        # --! policy_res trainable
        # --! policy_lqr fixed

        # --! now, model rollout horizon
        horizon = 30
        nepoch = 10

        for epoch in range(nepoch):
            policy_optim.zero_grad()

            # --! sample a random batch that may contain mixed - nominal and excursion - data
            lookback, reward, next_lookback, done = replay.random_batch(batch_size)

            # --! determine a data mask that differentiates between nominal and excursion batch elements
            with torch.no_grad():
                mask_nom = replay.get_coarse_zeta(lookback) < replay.coarse_zeta_threshold()
                mask_nom = torch.squeeze(mask_nom)

            # --! knowing the data mask, normalize lookbacks
            with torch.no_grad():
                lookback_norm = dataset.normalize_masked(lookback, mask_nom)

            # --! compute the value of the current observation (state)
            with torch.no_grad():

                # --! extract a normalized current observation (state)
                state = replay.extract_current_state(lookback_norm)

                value = torch.empty_like(state)

                # --! compute the current state value
                value[mask_nom] = value_fn_nom(state[mask_nom])
                value[~mask_nom] = value_fn_exc(state[~mask_nom])

            # --! compute current uncertainty: zeta(x, u)
            with torch.no_grad():

                # --! compute the uncertainty of a nominal KIND operator
                zeta = model(dataset, lookback_norm)[2]

                # --! uncertainty zeta is represented by the mean of batch elements
                zeta = torch.mean(zeta, dim=tuple(range(1, zeta.dim())), keepdim=True)

            print(zeta.mean())
            print(tata.shape)
            # --! make a working copy of the current lookback to perform model rollouts
            rollout_lookback = lookback.clone()

            rollout_return = 0.0

            # --! perform model rollouts upto a spefified horizon
            for k in range(horizon):
                rollout_obs = replay.extract_current_state(rollout_lookback)

                # --! compute action
                delta_u = res_policy(rollout_obs, zeta)
                rollout_u = base_policy(rollout_obs) + delta_u

                rollout_reward = reward_fn_torch(torch.cat([rollout_obs, rollout_u], dim=-1))
                rollout_return += gamma**k * rollout_reward

                replay.update_current_action(rollout_lookback, rollout_u)

                # --! KIND predicts next state
                #
                # --! we need to manually normalize data here
                rollout_lookback = dataset.normalize(rollout_lookback, data_type='nom') # <- data_type !!!
                rollout = model(dataset, rollout_lookback) # < gradients flow here

                #if k==horizon - 1:
                    #with torch.no_grad():
                        #plt.figure(figsize=(6,12))

                        #plt.subplot(7,1,1)
                        #plt.plot(rollout_lookback[0, :, :2])
                        #plt.plot(rollout[0][0, :, :], linestyle='dashed')

                        #plt.subplot(7,1,2)
                        #plt.plot(rollout_lookback[0, :, :2])
                        #plt.plot(rollout[1][0, :, :], linestyle='dashed')

                        #plt.subplot(7,1,3)
                        #plt.plot(rollout[2][0, :, :])

                        #plt.subplot(7,1,4)
                        #plt.plot(rollout_lookback[0, :, :2])
                        #plt.plot(rollout[3][0, :, :], linestyle='dashed')

                        #plt.subplot(7,1,5)
                        #plt.plot(rollout[4][0, :, :])

                        #plt.subplot(7,1,6)
                        #plt.plot(rollout[9][0, :, :])

                        #plt.subplot(7,1,7)
                        #plt.plot(rollout_lookback[0, :, 2])
                        #plt.xlabel('samples')

                        #plt.tight_layout()
                        #plt.show()

                # --! having a prediction, take its forecast part
                #
                # --! we need to manually denormalize data here
                rollout_lookback = dataset.denormalize(rollout_lookback, data_type='nom')
                rollout_pre = dataset.denormalize(rollout[0], data_type='nom')
                forecast = rollout_pre[:, model.args.lookback_nsample:]

                # --! take the first observation from the forecast
                rollout_next_obs = forecast[:, :1, :]

                # --! shift/update lookback using next observation
                rollout_lookback = replay.update_lookback(rollout_lookback, rollout_next_obs)

            with torch.no_grad():
                terminal_value = value_fn(rollout_next_obs)
                current_value = value_fn(rollout_obs0)

            advantage = rollout_return + gamma**horizon * terminal_value - current_value
            policy_loss = -advantage.mean()
            losses.append(policy_loss.item())
            policy_loss.backward()
            policy_optim.step()


class residual_policy(torch.nn.Module):

    def __init__(self):
        super().__init__()

        policy_ni = 2
        policy_no = 1
        self.net = util_nn.fcnn(feat=[policy_ni, 64, 64, policy_no], actfun_hid='relu', actfun_o='linear')

    def action(self, observation, uncertainty):
        """Returns an action for given ``observation``, while taking into account ``uncertainty``."""

        a = self.net(observation)
        return self._authorize_control(uncertainty) * torch.tanh(a)

    def __call__(self, obs, zeta):
        return self.action(obs, zeta)

    def _authorize_control(self, zeta,
                           u_min=0.01,
                           u_max_exc=0.05,
                           zeta_star=0.02,
                           zeta_exc=0.2):
        """
        zeta:        Tensor of shape (B, 1) or (B,)
        u_min:       float or Tensor
        u_max_exc:   float or Tensor
        zeta_star:   float
        zeta_exc:    float
        """

        # --! normalize zeta into [0, 1]
        alpha = (zeta - zeta_star) / (zeta_exc - zeta_star)

        # --! clamp to [0, 1]
        alpha = torch.clamp(alpha, 0.0, 1.0)

        # --! linear ramp
        u_max = u_min + alpha * (u_max_exc - u_min)

        return u_max


class environment(interface):

    @abstractmethod
    def reset(self):
        """Resets this environment to start a new episode and returns the first observation from that new episode."""
        return

    @abstractmethod
    def step(self, action):
        """Applies ``action`` and returns the next observation, reward and termination flags."""
        return

    @abstractmethod
    def step_batch(self, observation, action):
        """Applies every ``action`` in a given batch to corresponding ``observation`` and returns a batch of next observations."""
        return


class replay_buffer(interface):

    @abstractmethod
    def add(self, lookback, reward, next_lookback, done):
        return

    @abstractmethod
    def random_batch(self, batch_size):
        return

    @abstractmethod
    def empty(self):
        return

    @abstractmethod
    def encode_lookback(self, sa_window):
        """Encodes a KIND lookback from a window of state-action pairs ``sa_window``."""
        return

    @abstractmethod
    def encode_sa(self, s, a):
        """Encodes a state-action pair from state ``s`` and action ``a`` to be shifted into a KIND lookback."""
        return

    @abstractmethod
    def extract_current_state(self, lookback):
        """Extracts the current state from a given ``lookback``."""
        return

    @abstractmethod
    def update_current_action(self, lookback, a):
        return

    @abstractmethod
    def update_lookback(self, lookback, s):
        return

    @abstractmethod
    def get_coarse_zeta(self, lookback):
        return

    @abstractmethod
    def coarse_zeta_threshold(self):
        return

