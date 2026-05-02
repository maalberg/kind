
from abc import abstractmethod
from abc import ABC as interface

from collections import deque
from collections import namedtuple
from itertools import chain

from matplotlib import pyplot as plt

import argparse
import random
import torch

import os

import kind
import util_nn
import util_data


environments = namedtuple('environments', 'nominal excursion')
policies = namedtuple('policies', 'base residual')


def create_args_parser():
    """Creates a command-line parser for policy iteration arguments."""

    parser = argparse.ArgumentParser(description='Policy iteration')

    parser.add_argument('--gamma', type=float, required=True, help='discount factor')

    parser.add_argument('--learning_rate_value', type=float, required=False, default=0.001, help='learning rate for value evaluation')
    parser.add_argument('--nepoch_value', type=int, required=False, default=100, help='number of training epochs during policy evaluation')
    parser.add_argument('--batch_size_value', type=int, required=False, default=64, help='batch size during policy evaluation')
    parser.add_argument('--weight_decay_value', type=float, required=False, default=1e-6, help='weight decay during policy evaluation')

    parser.add_argument('--learning_rate_policy', type=float, required=False, default=0.001, help='learning rate for policy improvement')
    parser.add_argument('--nepoch_policy', type=int, required=False, default=100, help='number of training epochs during policy improvement')
    parser.add_argument('--batch_size_policy', type=int, required=False, default=64, help='batch size during policy improvement')
    parser.add_argument('--weight_decay_policy', type=float, required=False, default=1e-6, help='weight decay during policy improvement')
    parser.add_argument('--rollout_nsample', type=int, required=True, help='model rollout horizon during policy improvement')
    parser.add_argument('--back_reset_nsample', type=int, required=True,
                        help='number of samples after which a lookback window is re-anchored to real data during model rollouts')

    return parser


class policy_iteration:
    """Implements policy iteration."""

    def __init__(self, args):

        if args.rollout_nsample < 2:
            raise Exception('cannot train a policy when rollout horizon is less than 2 samples!')
        self.args = args

    def evaluate_policy(self, value_fn, replay):

        value_fn.train()
        value_optim = torch.optim.Adam(value_fn.parameters(), lr=self.args.learning_rate_value, weight_decay=self.args.weight_decay_value)

        losses = []

        for epoch in range(self.args.nepoch_value):
            value_optim.zero_grad()

            # --! sample a random batch
            obs, reward, next_obs, done = replay.random_batch(self.args.batch_size_value)

            # --! target must be treated as a constant, so restrict any gradient flow during target calculation
            with torch.no_grad():
                next_s = replay.util.get_s(next_obs)
                target = reward + self.args.gamma * (1.0 - done) * value_fn(next_s)

            # --! compute value of current s
            s = replay.util.get_s(obs)
            value = value_fn(s)

            # --! compute loss
            criterion = torch.nn.MSELoss()
            loss = criterion(value, target)

            losses.append(loss.item())
            loss.backward()
            value_optim.step()

        return losses

    def improve_policy(self, model, policy, value_fn, env, replay):

        value_fn.eval()
        policy.residual.train()

        policy_optim = torch.optim.Adam(
            policy.residual.parameters(), lr=self.args.learning_rate_policy, weight_decay=self.args.weight_decay_policy)

        losses = []

        for epoch in range(self.args.nepoch_policy):
            policy_optim.zero_grad()

            a = self.compute_advantage(
                model,
                policy,
                value_fn,
                env, replay, training=True
            )

            policy_loss = -a.mean()
            losses.append(policy_loss.item())
            policy_loss.backward()
            policy_optim.step()

        policy.residual.eval()
        return losses

    def compute_advantage(self, model, policy, value_fn, env, replay, training=False):

        if training:
            # --! sample a random batch of replay data
            obs, reward, next_obs, done = replay.random_batch(self.args.batch_size_policy)
        else:
            obs, reward, next_obs, done = replay.to_data()

        # --! restrict any gradient flow while computing the value of observation s
        with torch.no_grad():
            current_value = value_fn(replay.util.get_s(obs))

        # --! initialize next observation s, so that in case H=1 and we do not compute
        # --! rollout return we have a proper next s to compute terminal value
        next_s = replay.util.get_s(next_obs)

        # --! make a working copy of current observation to perform model rollouts
        rollout = obs.clone()

        rollout_return = 0.0
        gamma = self.args.gamma
        horizon = self.args.rollout_nsample
        obs_nsample = model.args.back_nsample

        # --! perform model rollouts up to a spefified horizon - 1
        for k in range(horizon - 1):

            if k and k % self.args.back_reset_nsample == 0:
                with torch.no_grad():
                    env_ic = replay.util.get_s0(rollout)
                    rollout = replay.util.replay_obs(
                        env, env_ic, policies(policy.base, None), obs_nsample) # TODO: residual policy is not always None!

            # --! get current observation as s_{t+k}
            s = replay.util.get_s(rollout)

            # --! compute action
            delta_a = policy.residual(s)
            a = policy.base(s) + delta_a

            # --! update action inside rollout window with newly computed action
            replay.util.update_a(rollout, a)

            # --! calculate reward for nominal or excursion timeseries present in current batch
            rollout_reward = env.reward_fn(s, a)

            rollout_return += gamma**k * rollout_reward

            # --! model predicts next state
            #
            # --! gradients flow through model, but the model is supposed to be fixed
            model_o = model(rollout)

            # --! having a prediction, take its forecast part
            fore = model_o.blend[:, model.args.back_nsample:]

            # --! take first observation from forecast as s_{t+k+1}
            next_s = fore[:, :1, :]

            # --! shift/update rollout window using predicted next observation
            rollout = replay.util.shift_obs(rollout, next_s)

        # --! compute terminal value
        #
        # --! next s is either the one initialized at the start of the algorithm or the one
        # --! updated during rollout return computation
        with torch.no_grad():
            terminal_value = value_fn(next_s)

        return rollout_return + gamma**horizon * terminal_value - current_value


class value_fn:
    """Wraps a learned value function to add the capability of internal state normalization."""

    def __init__(self, normalizer):
        self.normalizer = normalizer

        value_fn_ni = 2
        value_fn_no = 1
        self.value_fn = util_nn.fcnn(feat=[value_fn_ni, 64, 64, value_fn_no], actfun_hid='relu')

    def __call__(self, state):
        return self.forward(state)

    def forward(self, state):
        """Normalizes the given ``state`` and computes the corresponding value."""

        # --! normalize the state, but do not save the normalization mask (see below)
        state = self.normalizer.normalize(state)

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

    @abstractmethod
    def replay(self, ic, policy, obs_nsample=1, skip_nsample=0):
        """Replays this environment under given ``policy`` starting from initial condition ``ic``."""
        pass

    @property
    @abstractmethod
    def reward_fn(self):
        """Returns reward function used by this environment."""
        pass


class replay_util(interface):

    @abstractmethod
    def encode_obs(self, sa):
        pass

    @abstractmethod
    def replay_obs(self, env, env_ic, policy, obs_nsample):
        """Replays and encodes observation of length ``obs_nsamples`` starting from initial condition ``env_ic``."""
        pass

    @abstractmethod
    def get_s(self, encoded_obs):
        """Gets 'current', i.e. last, observation s from encoded observation ``encoded_obs``."""
        pass

    @abstractmethod
    def get_s0(self, encoded_obs):
        """Gets initial, i.e. first, observation s from encoded observation ``encoded_obs``."""
        pass

    @abstractmethod
    def get_a(self, encoded_obs):
        """Gets 'current', i.e. last, action a from encoded observation ``encoded_obs``."""
        pass

    @abstractmethod
    def shift_obs(self, encoded_obs, s):
        pass

    @abstractmethod
    def update_a(self, encoded_obs, a):
        pass


class replay(interface):

    def __init__(self, buffer=None):
        self.buffer = buffer if buffer is not None else []

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

    def to_data(self):
        return map(torch.cat, zip(*self.buffer))

    def to_file(self, filepath):

        # --! extract the last data element (s, a, mask) from every state (lookback)
        state, reward, next_state, done = self.to_data()
        data = state[:, [-1]]

        # --! for purity, reshape this 3D data, such that there is one n-step trajectory with m features, i.e. [1, n, m]
        data = torch.transpose(data, 0, 1)
        print(f'saving data with a shape {data.shape} to a file')

        util_data.write_datafile(filepath, data.numpy())

    @property
    @abstractmethod
    def util(self):
        pass


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

