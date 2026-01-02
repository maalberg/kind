from abc import abstractmethod
from abc import ABC as interface

from collections import deque
from collections import namedtuple

import torch
import numpy as np
import random


class replay_buffer:
    def __init__(self, capacity=None):
        self.buffer = deque(maxlen=capacity)

    def add(self, lookback, reward, next_lookback, done):

        # --! convert a bool flag to a float which is either 0.0 or 1.0
        done = done.float()

        # --! all entities must be shaped as 3D data
        done = torch.atleast_3d(done)
        reward = torch.atleast_3d(reward)

        self.buffer.append((
            lookback.detach(),
            reward,
            next_lookback.detach(),
            done
        ))

    def random_batch(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        return map(torch.cat, zip(*batch))

    def empty(self):
        return len(self.buffer)==0


class policy(interface):

    @abstractmethod
    def act(self, obs):
        """Returns action for given ``obs``."""
        return

    def __call__(self, obs):
        return self.act(obs)


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


class torch_environment(environment):

    def __init__(self, env):
        self.env = env

    def reset(self):
        obs = self.env.reset()
        return torch.from_numpy(obs).to(dtype=torch.float32)

    def step(self, action):
        action = action.detach().cpu().numpy()

        next_obs, reward, done = self.env.step(np.squeeze(action))

        next_obs = torch.from_numpy(next_obs).to(dtype=torch.float32)
        reward = torch.tensor(reward).to(dtype=torch.float32)
        done = torch.tensor(done)

        return next_obs, reward, done

    def step_batch(self, observation, action):
        observation = observation.detach().cpu().numpy()
        action = action.detach().cpu().numpy()

        next_obs = self.env.step_batch(np.squeeze(observation), np.squeeze(action))
        return torch.from_numpy(next_obs).to(dtype=torch.float32)

