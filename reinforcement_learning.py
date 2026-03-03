
from abc import abstractmethod
from abc import ABC as interface

from collections import deque
from collections import namedtuple
from itertools import chain

from matplotlib import pyplot as plt

import random
import torch

import os

import kind
import util_nn
import util_data


environments = namedtuple('environments', 'nominal excursion')
policies = namedtuple('policies', 'base residual')
reward_functions = namedtuple('reward_functions', 'nominal excursion')
value_functions = namedtuple('value_functions', 'nominal excursion')
zeta_thresholds = namedtuple('zeta_thresholds', 'nominal excursion')


def advantage(model, policy, reward_fn, value_fn, normalizer, factory, replay_data, env, horizon, gamma):
    """Computes advantage for reinforcement learning policy."""

    lookback, reward, next_lookback, done = replay_data

    # --! restrict any gradient flow while computing the value of the current state
    with torch.no_grad():

        # --! extract the current state
        state = factory.extract_current_s(lookback)

        # --! compute the current state value
        current_value = value_fn(state)

    # --! make a working copy of the current lookback to perform model rollouts
    rollout = lookback.clone()
    rollout_return = 0.0
    rollout_nsample_max = 20 # todo: place this as a parameter

    # --! perform model rollouts up to a spefified horizon - 1
    for k in range(horizon):

        if k and k % rollout_nsample_max == 0:
            with torch.no_grad():
                env_ic = factory.extract_first_s(rollout)
                rollout = factory.create_back(64, env, env_ic, policies(policy.base, None))

        state = factory.extract_current_s(rollout)

        # --! compute action
        delta_u = policy.residual(state)
        u = policy.base(state) + delta_u

        # --! update action in replay buffer with newly computed action
        factory.update_current_a(rollout, u)

        # --! calculate reward for nominal or excursion timeseries present in current batch
        rollout_reward = reward_fn(state, u)

        rollout_return += gamma**k * rollout_reward

        # --! KIND predicts next state
        model_o = model(rollout) # < gradients flow here

        # --! having a prediction, take its forecast part
        forecast = model_o.blend[:, 64:] # todo: put this constant as a parameter

        # --! take the first observation from the forecast
        next_state = forecast[:, :1, :]

        # --! shift/update lookback using next observation
        rollout = factory.update_state(rollout, next_state)

    # --! compute the terminal value
    with torch.no_grad():
        terminal_value = value_fn(next_state)

    return rollout_return + gamma**horizon * terminal_value - current_value


class policy_iteration:
    """Implements a model-based policy iteration."""

    def __init__(self, base_policy, reward_fn, normalizer):

        self.base_policy = base_policy
        self.residual_policy = policy(normalizer)

        self.value_fn = value_fn(normalizer)
        self.reward_fn = reward_fn
        self.normalizer = normalizer

    def evaluate_policy(self, replay_factory, replay):
        return self._train_value(self.value_fn, replay_factory, replay)

    def improve_policy(self, model, replay_factory, replay, env):

        value_fn = self.value_fn

        residual_policy = self.residual_policy
        base_policy = self.base_policy

        gamma = 0.995
        batch_size = 64
        learning_rate = 1e-3
        weight_decay = 1e-6
        policy_optim = torch.optim.Adam(residual_policy.parameters(), lr=learning_rate, weight_decay=weight_decay)

        losses = []

        # --! freeze value function
        value_fn.eval()

        # --! train policy
        residual_policy.train()

        # --! now, model rollout horizon
        horizon = 1
        nepoch = 100

        for epoch in range(nepoch):
            policy_optim.zero_grad()

            # --! sample a random batch that may contain mixed - nominal and excursion - data
            replay_data = replay.random_batch(batch_size)

            a = advantage(
                model,
                policies(base_policy, residual_policy),
                self.reward_fn,
                value_fn,
                self.normalizer,
                replay_factory, replay_data, env,
                horizon, gamma
            )

            policy_loss = -a.mean()
            losses.append(policy_loss.item())
            policy_loss.backward()
            policy_optim.step()

        residual_policy.eval()
        return losses

    def _train_value(self, value_fn, replay_factory, replay):

        nepoch = 500
        batch_size = 64
        gamma = 0.995

        learning_rate = 1e-3
        weight_decay = 1e-6
        value_optim = torch.optim.Adam(value_fn.parameters(), lr=learning_rate, weight_decay=weight_decay)

        losses = []

        value_fn.train()

        for epoch in range(nepoch):
            value_optim.zero_grad()

            # --! sample a random batch
            lookback, reward, next_lookback, done = replay.random_batch(batch_size)

            # --! target must be treated as a constant, so restrict any gradient flow during target calculation
            with torch.no_grad():
                next_state = replay_factory.extract_current_s(next_lookback)
                target = reward + gamma * (1.0 - done) * value_fn(next_state)

            # --! compute the value of the current state
            state = replay_factory.extract_current_s(lookback)
            value = value_fn(state)

            # --! compute loss
            criterion = torch.nn.MSELoss()
            loss = criterion(value, target)

            losses.append(loss.item())
            loss.backward()
            value_optim.step()

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
        state = self.state_normalizer.normalize(state)

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

    def __init__(self, normalizer, gain=0.5):

        self.gain = gain
        self.normalizer = normalizer

        policy_ni = 2
        policy_no = 1
        self.net = util_nn.fcnn(feat=[policy_ni, 64, 64, policy_no], actfun_hid='relu', actfun_o='linear')

    def __call__(self, state):
        return self.forward(state)

    def forward(self, state):

        state = self.normalizer.normalize(state)
        action = self.gain * torch.tanh(self.net(state))

        return self.normalizer.denormalize(action)

    def train(self, mode=True):
        return self.net.train(mode)

    def eval(self):
        return self.train(mode=False)

    def parameters(self):
        return self.net.parameters()

    def load_state_dict(self, state_dict, strict=True, assign=False):
        return self.net.load_state_dict(state_dict, strict, assign)

    def state_dict(self, prefix='', keep_vars=False):
        return self.net.state_dict(prefix=prefix, keep_vars=keep_vars)


class environment(interface):

    @abstractmethod
    def reset(self, ic):
        """Resets this environment to start a new episode from initial condition ``ic``.
        Returns the first observation from that new episode."""
        pass

    @abstractmethod
    def step(self, action):
        """Applies ``action`` and returns the next observation, reward and termination flags."""
        pass

    @property
    @abstractmethod
    def reward_fn(self):
        """Returns reward function used by this environment."""
        pass


class replay_factory(interface):

    @abstractmethod
    def create(self, env, env_ic, policy, zeta, zeta_star, state_nsample, skip_nsample):
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


class dataset_factory(interface):

    @abstractmethod
    def create_dataset(self, args, load_normalized=True):
        pass

    @abstractmethod
    def create_normalizer(self, args, load_normalized=False):
        pass


class agent:
    """Implements a Dyna-style reinforcement learning agent."""

    def __init__(self, env, base_policy, dataset_factory, replay_factory, args):

        self.env = env
        self.base_policy = base_policy
        self.dataset_factory = dataset_factory
        self.replay_factory = replay_factory
        self.args = args

        # --! construct policy iteration
        self.piter = policy_iteration(
            base_policy,
            kind.regimes(env.nominal.reward_fn, env.excursion.reward_fn),
            dataset_factory.create_normalizer(args))

    def train(self, niter=1):

        # --! initially there is no residual policy,
        # --! so zeta star constants (maximum zetas of a nominal model on nominal and excursion data) do not matter and are set to 0
        policy = None
        zeta_star = kind.regimes(0.0, 0.0)

        replay_nsample = self.args.lookback_nsample
        replay_skip_nsample = replay_nsample*3

        for i in range(niter):

            # --! first, acquire replay data buffers from environment
            replay = self._acquire_replay(policy, zeta_star, replay_nsample, replay_skip_nsample)

            # --! second, train a KIND model on the acquired data
            model, zeta_star = self._train_model(replay, i + 1)

            # --! third, run policy iteration
            policy = self._iterate_policy(model, zeta_star, replay)

            # --! before advancing to the next iteration, save current progress
            self._save_progress(replay, replay_nsample, replay_skip_nsample, model, zeta_star, policy, i + 1)

    def _acquire_replay(self, residual_policy, zeta_star, state_nsample, skip_nsample):

        replay_nom = self.replay_factory.create(
            self.env.nominal,
            policies(self.base_policy, residual_policy),
            zeta_star.nominal, zeta_star,
            state_nsample, skip_nsample)
        replay_exc = self.replay_factory.create(
            self.env.excursion,
            policies(self.base_policy, residual_policy),
            zeta_star.excursion, zeta_star,
            state_nsample, skip_nsample)

        return kind.regimes(replay_nom, replay_exc)

    def _train_model(self, replay, i):

        # --! first, save replay into dataset files to comply with KIND model interface
        filename = f'{self.args.file_name}_nom_{i}'
        replay.nominal.to_file(os.path.join(self.args.file_dir, filename))
        filename = f'{self.args.file_name}_exc_{i}'
        replay.excursion.to_file(os.path.join(self.args.file_dir, filename))

        # --! update file index in KIND arguments
        self.args.file_index = i
 
        # --! create model and dataset
        model = kind.model(self.args)
        training = kind.training(model)
        dataset = self.dataset_factory.create_dataset(self.args)

        # --! train model
        model.train()
        keep_training = True
        while keep_training:
            training.fit(dataset)
            keep_training = training.fit_next()

        # --! having a model, we can now estimate zeta star constants
        zeta_star = self._estimate_zeta_star(model, dataset)

        # --! wrap model with a model adapter which normalizes/denormalizes input data on the fly
        model = kind.model_adapter(model, dataset.normalizer)

        return model, zeta_star

    def _iterate_policy(self, model, zeta_star, replay):
        loss_nom, loss_exc = self.piter.evaluate_policy(self.replay_factory, replay)
        loss_policy = self.piter.improve_policy(model, zeta_star, self.replay_factory, replay)

        self.piter.value_fn_nom.eval()
        self.piter.value_fn_exc.eval()

        state, reward, next_state, done = map(torch.cat, zip(*replay.nominal.buffer))
        obs_nom = self.replay_factory.extract_current_s(state)
        obs_norm_nom = torch.squeeze(torch.linalg.norm(obs_nom, dim=-1, ord=2))
        value_nom = torch.squeeze(self.piter.value_fn_nom(obs_nom))

        state, reward, next_state, done = map(torch.cat, zip(*replay.excursion.buffer))
        obs_exc = self.replay_factory.extract_current_s(state)
        obs_norm_exc = torch.squeeze(torch.linalg.norm(obs_exc, dim=-1, ord=2))
        value_exc = torch.squeeze(self.piter.value_fn_exc(obs_exc))

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

        with torch.no_grad():

            plt.figure(figsize=(6,3))

            plt.plot(loss_policy)
            plt.xlabel('epochs')
            plt.ylabel('policy loss')

            plt.show()

        return self.piter.residual_policy

    def _estimate_zeta_star(self, model, dataset):
        """Estimates zeta star constants as a mean over ``dataset``.
        Constants zeta star are a mean zeta response of a nominal model to nominal and excursion data."""

        # --! check that this model works with normalized inputs only
        assert isinstance(model, kind.model) and dataset.load_normalized==True

        model.eval()

        # --! estimate zeta star of a nominal model on nominal data
        train_loader, _, _ = dataset.load(data_type='nom') # training loader provides more data for estimation
        zeta = []
        with torch.no_grad():
            for back, fore in train_loader:

                model_o = model(back)
                zeta_nom = model_o.zeta_nom

                zeta.append(torch.mean(zeta_nom).item())

        zeta = torch.tensor(zeta, dtype=torch.float32)
        zeta_nom_nom = torch.mean(zeta)

        # --! estimate zeta star of a nominal model on excursion data
        train_loader, _, _ = dataset.load(data_type='exc')
        zeta = []
        with torch.no_grad():
            for back, fore in train_loader:

                model_o = model(back)
                zeta_nom = model_o.zeta_nom

                zeta.append(torch.mean(zeta_nom).item())

        zeta = torch.tensor(zeta, dtype=torch.float32)
        zeta_nom_exc = torch.mean(zeta)

        return kind.regimes(zeta_nom_nom, zeta_nom_exc)

    def _save_progress(self, replay, replay_nsample, replay_skip_nsample, model, zeta_star, residual_policy, i):
        print(zeta_star)

        self._save_model_progress(model, i)
        self._save_policy_progress(replay, replay_nsample, replay_skip_nsample, model, zeta_star, residual_policy, i)

    def _save_model_progress(self, model, i):

        # --! check that this model is already wrapped in an adapter that works with unnormalized data
        #
        # --! in addition, the fact that this model is adapter-wrapped means that it is set to
        # --! evaluation mode with frozen weights
        assert isinstance(model, kind.model_adapter)

        # --! as given model is supposed to handle unnormalized inputs, create a dataset accordingly
        dataset = self.dataset_factory.create_dataset(self.args, load_normalized=False)

        dirpath = '../../results/dreamer'
        name = f'{dirpath}/{i}_model_outputs.pdf'

        _, _, data_loader = dataset.load(data_type='mixed')

        with torch.no_grad():
            for back, fore in data_loader:
                true = torch.cat([back, fore], dim=1)

                model_o = model(back[:2])

                plt.figure(figsize=(11,20))

                # --! define helping index constants
                jnom = 0 # nominal data
                jexc = 1 # excursion data
                js1 = 0 # first state
                js2 = 1 # second state

                plt.subplot(8,2,1)
                plt.plot(true[jnom, :, js1])
                plt.plot(model_o.blend[jnom, :, js1], linestyle='dashed')

                plt.subplot(8,2,2)
                plt.plot(true[jexc, :, js1])
                plt.plot(model_o.blend[jexc, :, js1], linestyle='dashed')

                plt.subplot(8,2,3)
                plt.plot(true[jnom, :, js2])
                plt.plot(model_o.blend[jnom, :, js2], linestyle='dashed')

                plt.subplot(8,2,4)
                plt.plot(true[jexc, :, js2])
                plt.plot(model_o.blend[jexc, :, js2], linestyle='dashed')

                plt.subplot(8,2,5)
                plt.plot(true[jnom, :, js1])
                plt.plot(model_o.mean_nom[jnom, :, js1], linestyle='dashed')

                plt.subplot(8,2,6)
                plt.plot(true[jexc, :, js1])
                plt.plot(model_o.mean_nom[jexc, :, js1], linestyle='dashed')

                plt.subplot(8,2,7)
                plt.plot(true[jnom, :, js2])
                plt.plot(model_o.mean_nom[jnom, :, js2], linestyle='dashed')

                plt.subplot(8,2,8)
                plt.plot(true[jexc, :, js2])
                plt.plot(model_o.mean_nom[jexc, :, js2], linestyle='dashed')

                plt.subplot(8,2,9)
                plt.plot(true[jnom, :, js1])
                plt.plot(model_o.mean_exc[jnom, :, js1], linestyle='dashed')

                plt.subplot(8,2,10)
                plt.plot(true[jexc, :, js1])
                plt.plot(model_o.mean_exc[jexc, :, js1], linestyle='dashed')

                plt.subplot(8,2,11)
                plt.plot(true[jnom, :, js2])
                plt.plot(model_o.mean_exc[jnom, :, js2], linestyle='dashed')

                plt.subplot(8,2,12)
                plt.plot(true[jexc, :, js2])
                plt.plot(model_o.mean_exc[jexc, :, js2], linestyle='dashed')

                plt.subplot(8,2,13)
                plt.plot(model_o.zeta_nom[jnom, :, :2], label='zeta')
                plt.legend()

                plt.subplot(8,2,14)
                plt.plot(model_o.zeta_exc[jexc, :, :2], label='zeta')
                plt.legend()

                plt.subplot(8,2,15)
                plt.plot(model_o.alpha[jnom, :, :2], label='alpha')
                plt.ylim((0.0, 1.05))
                plt.legend()

                plt.subplot(8,2,16)
                plt.plot(model_o.alpha[jexc, :, :2], label='alpha')
                plt.ylim((0.0, 1.05))
                plt.legend()

                plt.savefig(name, bbox_inches='tight')
                plt.close() # prevent saved figure from showing on screen
                break

    def _save_policy_progress(self, replay, replay_nsample, replay_skip_nsample, model, zeta_star, residual_policy, i):

        new_replay = self._acquire_replay(residual_policy, zeta_star, replay_nsample, replay_skip_nsample)

        # --! gather initial and new nominal data
        state, reward, next_state, done = map(torch.cat, zip(*replay.nominal.buffer))
        s_nom = self.replay_factory.extract_current_s(state)
        a_nom = self.replay_factory.extract_current_a(state)
        state, reward, next_state, done = map(torch.cat, zip(*new_replay.nominal.buffer))
        new_s_nom = self.replay_factory.extract_current_s(state)
        new_a_nom = self.replay_factory.extract_current_a(state)

        # --! gather initial and new excursion data
        state, reward, next_state, done = map(torch.cat, zip(*replay.excursion.buffer))
        s_exc = self.replay_factory.extract_current_s(state)
        a_exc = self.replay_factory.extract_current_a(state)
        state, reward, next_state, done = map(torch.cat, zip(*new_replay.excursion.buffer))
        new_s_exc = self.replay_factory.extract_current_s(state)
        new_a_exc = self.replay_factory.extract_current_a(state)

        dirpath = '../../results/dreamer'
        name = f'{dirpath}/{i}_policy_outputs.pdf'

        with torch.no_grad():
            plt.figure(figsize=(11,7.5))

            plt.subplot(3,2,1)
            plt.plot(s_nom[3000:, 0, 0])
            plt.plot(new_s_nom[3000:, 0, 0])

            plt.subplot(3,2,2)
            plt.plot(s_exc[3000:, 0, 0])
            plt.plot(new_s_exc[3000:, 0, 0])

            plt.subplot(3,2,3)
            plt.plot(s_nom[3000:, 0, 1])
            plt.plot(new_s_nom[3000:, 0, 1])

            plt.subplot(3,2,4)
            plt.plot(s_exc[3000:, 0, 1])
            plt.plot(new_s_exc[3000:, 0, 1])

            plt.subplot(3,2,5)
            plt.plot(a_nom[3000:, 0, 0])
            plt.plot(new_a_nom[3000:, 0, 0])

            plt.subplot(3,2,6)
            plt.plot(a_exc[3000:, 0, 0])
            plt.plot(new_a_exc[3000:, 0, 0])

            plt.savefig(name, bbox_inches='tight')
            plt.close() # prevent saved figure from showing on screen

