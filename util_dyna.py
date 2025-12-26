from abc import abstractmethod
from abc import ABC as interface

import itertools
from collections import deque
from collections import namedtuple

import torch
import numpy as np
import random

state_preview = namedtuple('state_preview', 'state preview')


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


class replay_buffer:

    def __init__(self, capacity):
        """Creates a replay buffer with given ``capacity``."""
        self.buffer = deque(maxlen=capacity)

    def add_sample(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self):
        return self.buffer[0]

    def sample_batch(self, size):
        """Samples a random batch from this replay buffer sized according to given ``size``."""
        batch = random.sample(self.buffer, size)

        states, actions, rewards, next_states, dones = zip(*batch)

        return (
            torch.cat(states, dim=0),
            torch.cat(actions, dim=0),
            torch.cat(rewards, dim=0),
            torch.cat(next_states, dim=0),
            torch.cat(dones, dim=0),
        )

    def obs(self, n=None):
        states, actions, rewards, next_states, dones = zip(*self.buffer)
        states = torch.cat(states, dim=0)
        return states if n is None else states[:n]

    def obs_actions(self, n=None):
        states, actions, rewards, next_states, dones = zip(*self.buffer)
        states = torch.cat(states, dim=0)
        actions = torch.cat(actions, dim=0)
        obs_actions = torch.cat([states, actions], dim=-1)
        return obs_actions if n is None else obs_actions[:n]

    def __len__(self):
        return len(self.buffer)


class kind_environment(environment):
    def __init__(self, environment_true, model, dataset, policy_base, reward):

        self.env_true = environment_true
        self.model = model
        self.dataset = dataset
        self.policy_base = policy_base
        self.reward = reward

        self.lookback_state = deque(maxlen=model.args.lookback_nsample)
        self.lookback_action = deque(maxlen=model.args.lookback_nsample)

    def reset(self):
        """Resets this environment to start a new episode and returns the first observation from the new episode."""

        self.lookback_state.clear()
        self.lookback_action.clear()

        # --! reset a true environment and obtain its initial condition
        state = self.env_true.reset()

        # --! fill lookback windows with real states and actions
        for step in range(self.model.args.lookback_nsample):

            u = self.policy_base(state)

            self.lookback_state.append(state)
            self.lookback_action.append(u)

            # --! next state
            state = self.env_true.step(np.squeeze(u))

        # --! save the last next state and return it - this becomes the current observation for an RL agent
        #
        # --! note that the state lookback window is ahead of action window by one
        self.lookback_state.append(state)
        return state
 
    def step(self, action):
        """Applies ``action`` and returns the next observation, reward and termination flags."""

        # --! take the current observation that RL reacted to by issuing given action
        state = self.lookback_state[-1]

        # --! action from RL is treated as residual control, which together with a baseline policy forms the final action u
        #
        # --! the final action is saved in its lookback window, thus making the state and action windows equal in size
        u = self.policy_base(state) + action
        self.lookback_action.append(u)

        # --! as we have a state and action pair we can immediately query current reward
        reward = self.reward(np.concatenate([state, u], axis=-1))

        # --! moreover, as we have balanced lookback windows we can call KIND model to get the next state forecast
        forecast = self.model(self.dataset, self._get_lookback_tensor())[0]

        # --! act on the true environment with a new RL-based action to get the real next state
        #
        # --! note that if the forecast above is from +1 to +H, then the real next state is +1
        state = self.env_true.step(np.squeeze(u))

        # --! finally, test a done condition
        done = self.env_true.jstep >= self.env_true.nstep - self.model.args.forecast_nsample

        # --! return the next state with preview, reward and a done flag
        return state, forecast, reward, done

    def _get_lookback_tensor(self):
        """Returns the lookback windows converted from deques to a single tensor."""

        a = np.array(self.lookback_state).reshape(-1, 2)
        lookback_state = torch.tensor(a, dtype=torch.float32)

        a = np.array(self.lookback_action).reshape(-1, 1)
        lookback_action = torch.tensor(a, dtype=torch.float32)

        mask_action = torch.ones_like(lookback_action)

        return torch.unsqueeze(torch.cat([lookback_state, lookback_action, mask_action], dim=-1), 0)

