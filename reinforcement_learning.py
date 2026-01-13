
from abc import abstractmethod
from abc import ABC as interface

from collections import deque
from collections import namedtuple

import torch

import util_nn

from matplotlib import pyplot as plt


class policy_iteration:
    """Implements a model-based policy iteration."""

    def __init__(self, model, base_policy, normalizer, reward_fn_nom, reward_fn_exc):

        self.model = model

        self.base_policy = base_policy
        self.res_policy = residual_policy(normalizer)

        # --! we use two separate value functions: one for nominal and another for excursion regimes
        self.value_fn_nom = value_fn(normalizer)
        self.value_fn_exc = value_fn(normalizer)

        self.reward_fn_nom = reward_fn_nom
        self.reward_fn_exc = reward_fn_exc

        # --! state machine
        self.state_evaluate = iteration_evaluate(self)
        self.state_improve = iteration_improve(self)
        self.state = self.state_evaluate

    def iterate(self, replay_nom, replay_exc, dataset):
        return self._get_state().iterate(replay_nom, replay_exc, dataset)

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

        self.iter._set_state(self.iter._get_state_improve())

        return loss_nom, loss_exc

    def _evaluate_policy(self, value_fn, replay, dataset):

        nepoch = 350
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

            # --! target must be treated as a constant, so restrict any gradient flow during target calculation
            with torch.no_grad():
                next_state = replay.extract_current_state(next_lookback)
                target = reward + gamma * (1.0 - done) * value_fn(next_state)

            # --! compute the value of the current state
            state = replay.extract_current_state(lookback)
            value = value_fn(state)

            # --! compute loss
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
        loss = self._improve_policy(dataset, replay_nom + replay_exc, self.iter.reward_fn_nom, self.iter.reward_fn_exc)
        self.iter._set_state(self.iter._get_state_evaluate())

        return loss

    def _improve_policy(self, dataset, replay, reward_fn_nom, reward_fn_exc):

        value_fn_nom = self.iter.value_fn_nom
        value_fn_exc = self.iter.value_fn_exc

        res_policy = self.iter.res_policy
        base_policy = self.iter.base_policy

        model = self.iter.model

        gamma = 0.96
        batch_size = 128
        learning_rate = 1e-3
        weight_decay = 1e-5
        policy_optim = torch.optim.Adam(res_policy.parameters(), lr=learning_rate, weight_decay=weight_decay)

        losses = []

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
        horizon = 200
        nepoch = 60

        for epoch in range(nepoch):
            policy_optim.zero_grad()

            # --! sample a random batch that may contain mixed - nominal and excursion - data
            lookback, reward, next_lookback, done = replay.random_batch(batch_size)

            # --! determine a data mask that differentiates between nominal and excursion batch elements
            mask = dataset.normalizer.mask(lookback)

            # --! restrict any gradient flow while computing the value of the current state
            with torch.no_grad():

                # --! extract the current state
                state = replay.extract_current_state(lookback)

                # --! we cannot use empty_like here, because state and value have different dimensions,
                # --! therefore, ensure proper size of value manually
                current_value = torch.empty(state.shape[0], 1, 1)

                # --! compute the current state value
                current_value[mask] = value_fn_nom(state[mask])
                current_value[~mask] = value_fn_exc(state[~mask])

            # --! compute current uncertainty: zeta(x, u)
            with torch.no_grad():

                # --! compute the uncertainty of a nominal KIND operator
                zeta = model(lookback)[2]

                # --! uncertainty zeta is represented by the mean of batch elements
                zeta = torch.mean(zeta, dim=tuple(range(1, zeta.dim())), keepdim=True)

            # --! make a working copy of the current lookback to perform model rollouts
            rollout = lookback.clone()

            rollout_return = 0.0

            # --! perform model rollouts up to a spefified horizon - 1
            for k in range(horizon):
                state = replay.extract_current_state(rollout)

                # --! compute action
                delta_u = res_policy.forward_train(state, zeta, epoch)
                u = base_policy(state) + delta_u

                # --! update action in replay buffer with newly computed action
                replay.update_current_action(rollout, u)

                # --! allocate reward
                #
                # --! note that we cannot use empty_like here, because reward and state-action pair
                # --! have different dimensions, so we need to ensure dimensions manually
                rollout_reward = torch.empty(state.shape[0], 1, 1)

                # --! calculate reward for nominal or excursion timeseries present in current batch
                rollout_reward[mask] = reward_fn_nom(state[mask], u[mask])
                rollout_reward[~mask] = reward_fn_exc(state[~mask], u[~mask])

                rollout_return += gamma**k * rollout_reward

                # --! KIND predicts next state
                model_output = model(rollout) # < gradients flow here

                #if k==horizon - 1:
                    #with torch.no_grad():
                        #plt.figure(figsize=(6,14))

                        #plt.subplot(7,1,1)
                        #plt.plot(rollout[0, :, :2])
                        #plt.plot(model_output[0][0, :, :], linestyle='dashed')

                        #plt.subplot(7,1,2)
                        #plt.plot(rollout[0, :, :2])
                        #plt.plot(model_output[1][0, :, :], linestyle='dashed')

                        #plt.subplot(7,1,3)
                        #plt.plot(model_output[2][0, :, :])

                        #plt.subplot(7,1,4)
                        #plt.plot(rollout[0, :, :2])
                        #plt.plot(model_output[3][0, :, :], linestyle='dashed')

                        #plt.subplot(7,1,5)
                        #plt.plot(model_output[4][0, :, :])

                        #plt.subplot(7,1,6)
                        #plt.plot(model_output[9][0, :, :])

                        #plt.subplot(7,1,7)
                        #plt.plot(rollout[0, :, 2])
                        #plt.xlabel('samples')

                        #plt.tight_layout()
                        #plt.show()
                        #print(tata.shape)

                # --! having a prediction, take its forecast part
                forecast = model_output[0][:, 384:]

                # --! take the first observation from the forecast
                next_state = forecast[:, :1, :]

                # --! shift/update lookback using next observation
                rollout = replay.update_lookback(rollout, next_state)

            # --! compute the terminal value
            with torch.no_grad():

                # --! we cannot use empty_like here, because state and value have different dimensions,
                # --! therefore, ensure proper size of value manually
                terminal_value = torch.empty(next_state.shape[0], 1, 1)
                terminal_value[mask] = value_fn_nom(next_state[mask])
                terminal_value[~mask] = value_fn_exc(next_state[~mask])

            advantage = rollout_return + gamma**horizon * terminal_value - current_value
            policy_loss = -advantage.mean()
            losses.append(policy_loss.item())
            policy_loss.backward()
            policy_optim.step()

        return losses


class value_fn:
    """Wraps a learned value function to add the capability of internal state normalization."""

    def __init__(self, state_normalizer):
        self.state_normalizer = state_normalizer

        value_fn_ni = 2
        value_fn_no = 1
        self.value_fn = util_nn.fcnn(feat=[value_fn_ni, 64, 64, value_fn_no], actfun_hid='relu')

    def __call__(self, state):
        return self.forward(state)

    def forward(self, state):
        """Normalizes the given ``state`` and computes the corresponding value."""

        # --! normalize the state, but do not save the normalization mask (see below)
        state, _ = self.state_normalizer.normalize_state(state)

        # --! get a value from the normalized state
        value = self.value_fn(state)

        # --! it makes no physical sense to denormalize the state value, so simply return the value
        return value

    def train(self, mode=True):
        return self.value_fn.train(mode)

    def eval(self):
        return self.value_fn.eval()

    def parameters(self):
        return self.value_fn.parameters()


class residual_policy:
    """Wraps a learned residual policy to add the capability of internal state, action (de)normalization."""

    def __init__(self, state_normalizer):

        self.state_normalizer = state_normalizer

        policy_ni = 2
        policy_no = 1
        self.policy = util_nn.fcnn(feat=[policy_ni, 64, 64, policy_no], actfun_hid='relu', actfun_o='linear')

    def __call__(self, state, zeta):
        return self.forward(state, zeta)

    def forward(self, state, zeta):
        """Returns an authorized action for the given ``state``, while taking into account uncertainty ``zeta``."""

        state, mask = self.state_normalizer.normalize_state(state)

        action = self._ramp_action(zeta) * torch.tanh(self.policy(state))

        return self.state_normalizer.denormalize_action(action, mask)

    def forward_train(self, state, zeta, epoch):
        state, mask = self.state_normalizer.normalize_state(state)

        u_max_exc = self._schedule(epoch)

        action = self._ramp_action(zeta, u_max_exc=u_max_exc) * torch.tanh(self.policy(state))

        return self.state_normalizer.denormalize_action(action, mask)

    def train(self, mode=True):
        return self.policy.train(mode)

    def eval(self):
        return self.policy.eval()

    def parameters(self):
        return self.policy.parameters()

    def _schedule(self, epoch, u_max_init=0.1, u_max_final=0.5):
        return min(u_max_final, u_max_init + 0.05 * epoch)

    def _ramp_action(self, zeta, u_min=0.005, u_max_exc=0.5, zeta_star=0.002, zeta_exc=0.44):

        # --! normalize zeta into [0, 1]
        beta = (zeta - zeta_star) / (zeta_exc - zeta_star)

        # --! clamp to [0, 1]
        beta = torch.clamp(beta, 0.0, 1.0)

        # --! linear ramp
        u_max = u_min + beta * (u_max_exc - u_min)

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

