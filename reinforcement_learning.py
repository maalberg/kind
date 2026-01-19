
from abc import abstractmethod
from abc import ABC as interface

from collections import deque
from collections import namedtuple
from itertools import chain

import random
import torch

import util_nn
import util_data


policies = namedtuple('policies', 'base residual')
reward_functions = namedtuple('reward_functions', 'nominal excursion')
value_functions = namedtuple('value_functions', 'nominal excursion')


def advantage(model, policy, reward_fn, value_fn, normalizer, factory, replay_data, horizon, gamma, epoch=0):
    """Computes advantage for reinforcement learning policy."""

    lookback, reward, next_lookback, done = replay_data

    # --! determine a data mask that differentiates between nominal and excursion batch elements
    mask = normalizer.mask(lookback)

    # --! restrict any gradient flow while computing the value of the current state
    with torch.no_grad():

        # --! extract the current state
        state = factory.extract_current_s(lookback)

        # --! we cannot use empty_like here, because state and value have different dimensions,
        # --! therefore, ensure proper size of value manually
        current_value = torch.empty(state.shape[0], 1, 1)

        # --! compute the current state value
        current_value[mask] = value_fn.nominal(state[mask])
        current_value[~mask] = value_fn.excursion(state[~mask])

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
        state = factory.extract_current_s(rollout)

        # --! compute action
        delta_u = policy.residual.forward(state, zeta=zeta, epoch=epoch)
        u = policy.base(state) + delta_u

        # --! update action in replay buffer with newly computed action
        factory.update_current_a(rollout, u)

        # --! allocate reward
        #
        # --! note that we cannot use empty_like here, because reward and state-action pair
        # --! have different dimensions, so we need to ensure dimensions manually
        rollout_reward = torch.empty(state.shape[0], 1, 1)

        # --! calculate reward for nominal or excursion timeseries present in current batch
        rollout_reward[mask] = reward_fn.nominal(state[mask], u[mask])
        rollout_reward[~mask] = reward_fn.excursion(state[~mask], u[~mask])

        rollout_return += gamma**k * rollout_reward

        # --! KIND predicts next state
        model_output = model(rollout) # < gradients flow here

        # --! having a prediction, take its forecast part
        forecast = model_output[0][:, 384:]

        # --! take the first observation from the forecast
        next_state = forecast[:, :1, :]

        # --! shift/update lookback using next observation
        rollout = factory.update_state(rollout, next_state)

    # --! compute the terminal value
    with torch.no_grad():

        # --! we cannot use empty_like here, because state and value have different dimensions,
        # --! therefore, ensure proper size of value manually
        terminal_value = torch.empty(next_state.shape[0], 1, 1)
        terminal_value[mask] = value_fn.nominal(next_state[mask])
        terminal_value[~mask] = value_fn.excursion(next_state[~mask])

    return rollout_return + gamma**horizon * terminal_value - current_value


class policy_iteration:
    """Implements a model-based policy iteration."""

    def __init__(self, model, base_policy, normalizer, reward_fn_nom, reward_fn_exc):

        self.model = model

        self.base_policy = base_policy
        self.res_policy = policy(normalizer)

        # --! we use two separate value functions: one for nominal and another for excursion regimes
        self.value_fn_nom = value_fn(normalizer)
        self.value_fn_exc = value_fn(normalizer)

        self.reward_fn_nom = reward_fn_nom
        self.reward_fn_exc = reward_fn_exc

        # --! state machine
        self.state_evaluate = iteration_evaluate(self)
        self.state_improve = iteration_improve(self)
        self.state = self.state_evaluate

    def iterate(self, factory, replay_nom, replay_exc, dataset):
        return self._get_state().iterate(factory, replay_nom, replay_exc, dataset)

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
    def iterate(self, factory, replay_nom, replay_exc, dataset):
        return


class iteration_evaluate(iteration_state):

    def __init__(self, iteration):
        self.iter = iteration

    def iterate(self, factory, replay_nom, replay_exc, dataset):

        loss_nom = self._evaluate_policy(self.iter.value_fn_nom, factory, replay_nom, dataset)
        loss_exc = self._evaluate_policy(self.iter.value_fn_exc, factory, replay_exc, dataset)

        self.iter._set_state(self.iter._get_state_improve())

        return loss_nom, loss_exc

    def _evaluate_policy(self, value_fn, factory, replay, dataset):

        nepoch = 350
        batch_size = 128
        gamma = 0.94

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
                next_state = factory.extract_current_s(next_lookback)
                target = reward + gamma * (1.0 - done) * value_fn(next_state)

            # --! compute the value of the current state
            state = factory.extract_current_s(lookback)
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

    def iterate(self, factory, replay_nom, replay_exc, dataset):
        loss = self._improve_policy(dataset, factory, replay_nom + replay_exc, self.iter.reward_fn_nom, self.iter.reward_fn_exc)
        self.iter._set_state(self.iter._get_state_evaluate())

        return loss

    def _improve_policy(self, dataset, factory, replay, reward_fn_nom, reward_fn_exc):

        value_fn_nom = self.iter.value_fn_nom
        value_fn_exc = self.iter.value_fn_exc

        res_policy = self.iter.res_policy
        base_policy = self.iter.base_policy

        model = self.iter.model

        gamma = 0.94
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

        # --! now, model rollout horizon
        horizon = 1
        nepoch = 100

        for epoch in range(nepoch):
            policy_optim.zero_grad()

            # --! sample a random batch that may contain mixed - nominal and excursion - data
            replay_data = replay.random_batch(batch_size)

            a = advantage(
                model,
                policies(base_policy, res_policy),
                reward_functions(reward_fn_nom, reward_fn_exc),
                value_functions(value_fn_nom, value_fn_exc),
                dataset.normalizer,
                factory, replay_data,
                horizon, gamma, epoch
            )

            policy_loss = -a.mean()
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


class policy:

    def __init__(self, normalizer):

        self.normalizer = normalizer

        policy_ni = 2
        policy_no = 1
        self.net = util_nn.fcnn(feat=[policy_ni, 64, 64, policy_no], actfun_hid='relu', actfun_o='linear')

        self.state_train = policy_train(self)
        self.state_eval = policy_eval(self)
        self.state = self.state_train

    def __call__(self, state, **kwargs):
        return self.forward(state, **kwargs)

    def forward(self, state, **kwargs):
        return self.state.forward(state, **kwargs)

    def train(self, mode=True):
        if mode is True:
            self.state = self.state_train
        else:
            self.state = self.state_eval

        return self.net.train(mode)

    def eval(self):
        return self.train(mode=False)

    def parameters(self):
        return self.net.parameters()

    def load_state_dict(self, state_dict, strict=True, assign=False):
        return self.net.load_state_dict(state_dict, strict, assign)

    def state_dict(self, prefix='', keep_vars=False):
        return self.net.state_dict(prefix=prefix, keep_vars=keep_vars)


class policy_state(interface):

    @abstractmethod
    def forward(self, state, **kwargs):
        return

    def _ramp_action(self, zeta, u_min=0.005, u_max_exc=0.5, zeta_star=0.002, zeta_exc=0.44):

        # --! normalize zeta into [0, 1]
        beta = (zeta - zeta_star) / (zeta_exc - zeta_star)

        # --! clamp to [0, 1]
        beta = torch.clamp(beta, 0.0, 1.0)

        # --! linear ramp
        u_max = u_min + beta * (u_max_exc - u_min)

        return u_max


class policy_train(policy_state):

    def __init__(self, statemachine):
        self.statemachine = statemachine

    def forward(self, state, **kwargs):

        zeta = kwargs.get('zeta', 0.0)
        epoch = kwargs.get('epoch', 0)

        state, mask = self.statemachine.normalizer.normalize_state(state)

        u_max_exc = self._schedule_max_action(epoch)
        action = self._ramp_action(zeta, u_max_exc=u_max_exc) * torch.tanh(self.statemachine.net(state))

        return self.statemachine.normalizer.denormalize_action(action, mask)

    def _schedule_max_action(self, epoch, u_max_init=0.1, u_max_final=0.5):
        return min(u_max_final, u_max_init + 0.05 * epoch)


class policy_eval(policy_state):

    def __init__(self, statemachine):
        self.statemachine = statemachine

    def forward(self, state, **kwargs):

        zeta = kwargs.get('zeta', 0.0)

        state, mask = self.statemachine.normalizer.normalize_state(state)
        action = self._ramp_action(zeta) * torch.tanh(self.statemachine.net(state))

        return self.statemachine.normalizer.denormalize_action(action, mask)


class environment(interface):

    @abstractmethod
    def reset(self):
        """Resets this environment to start a new episode and returns the first observation from that new episode."""
        return

    @abstractmethod
    def step(self, action):
        """Applies ``action`` and returns the next observation, reward and termination flags."""
        return


class replay_factory(interface):

    @abstractmethod
    def create(self, env, policy, zeta, state_nsample, skip_nsample):
        pass

    @abstractmethod
    def encode_state(self, sa_window):
        pass

    @abstractmethod
    def extract_current_s(self, state):
        pass

    @abstractmethod
    def extract_current_a(self, state):
        pass

    @abstractmethod
    def update_current_a(self, state, a):
        pass

    @abstractmethod
    def update_state(self, state, s):
        pass


class replay:

    def __init__(self, buffer=None):
        self.buffer = buffer if buffer is not None else []

    def __add__(self, other):
        a = self.buffer
        b = other.buffer

        # --! interleave both buffers, such that the new buffer has elements: a[0], b[0], a[1], b[1], etc.
        ab = list(chain.from_iterable(zip(a, b)))

        return replay(buffer=ab)

    def add(self, state, reward, next_state, done):

        # --! convert a bool flag to a float which is either 0.0 or 1.0
        done = done.float()

        # --! all entities must be shaped as 3D data
        done = torch.atleast_3d(done)
        reward = torch.atleast_3d(reward)

        # --! pack all elements as a tuple and put the tuple into the buffer
        self.buffer.append((
            state.detach(),
            reward,
            next_state.detach(),
            done
        ))

    def random_batch(self, batch_size):

        batch = random.sample(self.buffer, batch_size)
        return [torch.cat(item) for item in zip(*batch)]

    def empty(self):
        return len(self.buffer)==0

    def to_file(self, filepath):

        # --! extract the last data element (s, a, mask) from every state (lookback)
        state, reward, next_state, done = map(torch.cat, zip(*self.buffer))
        data = state[:, [-1]]

        # --! for purity, reshape this 3D data, such that there is one n-step trajectory with m features, i.e. [1, n, m]
        data = torch.transpose(data, 0, 1)
        print(f'saving data with a shape {data.shape} to a file')

        util_data.write_datafile(filepath, data.numpy())

